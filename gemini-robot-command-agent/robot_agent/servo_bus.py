from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Iterable

import serial


class ServoBusError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServoCommand:
    servo_id: int
    position: int
    duration_ms: int


class LewanSoulLX16ABus:
    """Minimal bus-servo writer for LX-16A compatible serial boards.

    Packet format:
    0x55 0x55 <id> <length> <cmd> <params...> <checksum>
    """

    MOVE_TIME_WRITE = 1

    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 0.1, dry_run: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self._serial: serial.Serial | None = None

    @property
    def is_open(self) -> bool:
        return self.dry_run or (self._serial is not None and self._serial.is_open)

    def open(self) -> None:
        if self.dry_run:
            return
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s, write_timeout=1.0)
        self._serial.dtr = False
        self._serial.rts = False
        time.sleep(0.2)

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def move(self, command: ServoCommand) -> str:
        packet = self._move_packet(command.servo_id, command.position, command.duration_ms)
        if self.dry_run:
            return f"DRY-RUN servo {command.servo_id} -> {command.position} in {command.duration_ms}ms"
        self.open()
        if self._serial is None:
            raise ServoBusError("Serial bus did not open")
        self._serial.write(packet)
        self._serial.flush()
        return f"servo {command.servo_id} -> {command.position} in {command.duration_ms}ms"

    def move_many(self, commands: Iterable[ServoCommand]) -> list[str]:
        return [self.move(command) for command in commands]

    def _move_packet(self, servo_id: int, position: int, duration_ms: int) -> bytes:
        if not 0 <= servo_id <= 253:
            raise ServoBusError(f"Invalid servo id {servo_id}")
        position = max(0, min(1000, int(position)))
        duration_ms = max(0, min(30000, int(duration_ms)))
        params = [
            position & 0xFF,
            (position >> 8) & 0xFF,
            duration_ms & 0xFF,
            (duration_ms >> 8) & 0xFF,
        ]
        length = 3 + len(params)
        body = [servo_id, length, self.MOVE_TIME_WRITE, *params]
        checksum = (~sum(body)) & 0xFF
        return bytes([0x55, 0x55, *body, checksum])


class FeetechSTSBus:
    """Writer/scanner for Waveshare/Feetech ST/SC serial bus servos.

    This matches the SCServo library used by the Waveshare ESP32 serial bus
    servo board when serial-forwarding/direct-USB mode is active.
    """

    INST_PING = 1
    INST_READ = 2
    INST_WRITE = 3
    SMS_STS_ACC = 41
    GOAL_POSITION_L = 42
    PRESENT_POSITION_L = 56

    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 0.1, dry_run: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self._serial: serial.Serial | None = None

    @property
    def is_open(self) -> bool:
        return self.dry_run or (self._serial is not None and self._serial.is_open)

    def open(self) -> None:
        if self.dry_run:
            return
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s, write_timeout=1.0)
        self._serial.dtr = False
        self._serial.rts = False
        time.sleep(0.2)

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def move(self, command: ServoCommand) -> str:
        try:
            before = None if self.dry_run else self.read_position(command.servo_id)
            packet = self._write_position_packet(command.servo_id, command.position, command.duration_ms, speed=700)
            if self.dry_run:
                return f"DRY-RUN ST/SC servo {command.servo_id} -> {command.position} in {command.duration_ms}ms"
            self.open()
            if self._serial is None:
                raise ServoBusError("Serial bus did not open")
            self._serial.reset_input_buffer()
            self._serial.write(packet)
            self._serial.flush()
            ack = self._read_status(command.servo_id)
            time.sleep(max(0.08, command.duration_ms / 1000.0))
            after = self.read_position(command.servo_id)
            ack_text = "ack" if ack else "no-ack"
            return (
                f"ST/SC servo {command.servo_id}: {before} -> target {command.position} -> readback {after} "
                f"in {command.duration_ms}ms ({ack_text})"
            )
        except Exception:
            self.close()
            raise

    def move_many(self, commands: Iterable[ServoCommand]) -> list[str]:
        return [self.move(command) for command in commands]

    def scan(self, start_id: int = 1, end_id: int = 30) -> list[int]:
        if self.dry_run:
            return []
        try:
            self.open()
            serial_port = self._serial
            if serial_port is None:
                raise ServoBusError("Serial bus did not open")
            found: list[int] = []
            old_timeout = serial_port.timeout
            serial_port.timeout = min(self.timeout_s, 0.05)
            for servo_id in range(start_id, end_id + 1):
                serial_port.reset_input_buffer()
                serial_port.write(self._packet(servo_id, self.INST_PING, []))
                serial_port.flush()
                time.sleep(0.015)
                data = serial_port.read(8)
                if len(data) >= 6 and data[0] == 0xFF and data[1] == 0xFF and data[2] == servo_id:
                    found.append(servo_id)
            serial_port.timeout = old_timeout
            return found
        except Exception:
            self.close()
            raise
        finally:
            if self._serial is not None:
                self._serial.timeout = self.timeout_s

    def read_position(self, servo_id: int) -> int | None:
        if self.dry_run:
            return None
        try:
            self.open()
            if self._serial is None:
                raise ServoBusError("Serial bus did not open")
            self._serial.reset_input_buffer()
            self._serial.write(self._packet(servo_id, self.INST_READ, [self.PRESENT_POSITION_L, 2]))
            self._serial.flush()
            response = self._read_packet(servo_id, min_len=8)
            if response is None:
                return None
            payload = response["params"]
            if len(payload) < 2:
                return None
            return int(payload[0]) | (int(payload[1]) << 8)
        except Exception:
            self.close()
            raise

    def _write_position_packet(self, servo_id: int, position: int, duration_ms: int, speed: int) -> bytes:
        if not 0 <= servo_id <= 253:
            raise ServoBusError(f"Invalid servo id {servo_id}")
        position = max(0, min(4095, int(position)))
        duration_ms = max(0, min(30000, int(duration_ms)))
        speed = max(0, min(4095, int(speed)))
        params = [
            self.SMS_STS_ACC,
            10,
            position & 0xFF,
            (position >> 8) & 0xFF,
            0,
            0,
            speed & 0xFF,
            (speed >> 8) & 0xFF,
        ]
        return self._packet(servo_id, self.INST_WRITE, params)

    def _packet(self, servo_id: int, instruction: int, params: list[int]) -> bytes:
        length = len(params) + 2
        body = [servo_id, length, instruction, *params]
        checksum = (~sum(body)) & 0xFF
        return bytes([0xFF, 0xFF, *body, checksum])

    def _read_status(self, servo_id: int) -> dict | None:
        return self._read_packet(servo_id, min_len=6)

    def _read_packet(self, servo_id: int, min_len: int) -> dict | None:
        if self._serial is None:
            return None
        deadline = time.monotonic() + max(0.08, self.timeout_s)
        data = bytearray()
        while time.monotonic() < deadline:
            chunk = self._serial.read(self._serial.in_waiting or 1)
            if chunk:
                data.extend(chunk)
                start = data.find(b"\xff\xff")
                if start > 0:
                    del data[:start]
                if len(data) >= min_len and len(data) >= 4:
                    packet_len = data[3] + 4
                    if len(data) >= packet_len:
                        raw = bytes(data[:packet_len])
                        if raw[2] != servo_id:
                            return None
                        checksum = (~sum(raw[2:-1])) & 0xFF
                        if checksum != raw[-1]:
                            return None
                        length = raw[3]
                        return {"id": raw[2], "error": raw[4], "params": raw[5 : 5 + length - 2], "raw": raw}
        return None


class RoArmJsonBus:
    """Waveshare RoArm-style newline-delimited JSON serial protocol."""

    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 0.1, dry_run: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self._serial: serial.Serial | None = None

    @property
    def is_open(self) -> bool:
        return self.dry_run or (self._serial is not None and self._serial.is_open)

    def open(self) -> None:
        if self.dry_run:
            return
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s, write_timeout=1.0)
        self._serial.dtr = False
        self._serial.rts = False
        time.sleep(0.2)

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def move(self, command: ServoCommand) -> str:
        payload = {
            "T": 121,
            "joint": command.servo_id,
            "angle": command.position,
            "spd": 1000,
        }
        return self.send_json(payload)

    def move_many(self, commands: Iterable[ServoCommand]) -> list[str]:
        messages: list[str] = []
        for command in commands:
            messages.append(self.move(command))
            time.sleep(0.08)
        return messages

    def send_json(self, payload: dict) -> str:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        if self.dry_run:
            return f"DRY-RUN JSON {line.strip()}"
        self.open()
        if self._serial is None:
            raise ServoBusError("Serial bus did not open")
        self._serial.write(line.encode("utf-8"))
        self._serial.flush()
        time.sleep(0.05)
        response = self._serial.read(self._serial.in_waiting or 1).decode(errors="replace").strip()
        suffix = f" response={response}" if response else ""
        return f"JSON {line.strip()}{suffix}"

    def scan(self, start_id: int = 1, end_id: int = 30) -> list[int]:
        # RoArm JSON firmware does not expose servo-bus ping directly.
        self.send_json({"T": 605, "cmd": 1})
        return []
