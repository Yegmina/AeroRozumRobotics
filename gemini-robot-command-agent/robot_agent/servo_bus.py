from __future__ import annotations

from dataclasses import dataclass
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
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
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
