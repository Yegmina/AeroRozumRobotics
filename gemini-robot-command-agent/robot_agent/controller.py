from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from .config import AppConfig
from .servo_bus import FeetechSTSBus, LewanSoulLX16ABus, RoArmJsonBus, ServoCommand


@dataclass
class RobotState:
    positions: dict[str, int] = field(default_factory=dict)
    motion_enabled: bool = False
    last_actions: list[str] = field(default_factory=list)


class RobotController:
    def __init__(self, config: AppConfig, enable_motion: bool = False):
        self.config = config
        self.state = RobotState(
            positions={name: joint.home for name, joint in config.joints.items()},
            motion_enabled=enable_motion,
        )
        dry_run = not enable_motion
        if config.serial.protocol == "lewansoul_lx16a":
            bus_type = LewanSoulLX16ABus
        elif config.serial.protocol == "feetech_sts":
            bus_type = FeetechSTSBus
        elif config.serial.protocol == "roarm_json":
            bus_type = RoArmJsonBus
        else:
            raise ValueError(f"Unsupported servo protocol: {config.serial.protocol}")
        self.bus = bus_type(
            port=config.serial.port,
            baudrate=config.serial.baudrate,
            timeout_s=config.serial.timeout_s,
            dry_run=dry_run,
        )
        self._lock = threading.Lock()

    @property
    def dry_run(self) -> bool:
        return self.bus.dry_run

    def close(self) -> None:
        self.bus.close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "positions": dict(self.state.positions),
                "servo_ids": {name: joint.servo_id for name, joint in self.config.joints.items()},
                "motion_enabled": self.state.motion_enabled,
                "dry_run": self.dry_run,
                "protocol": self.config.serial.protocol,
                "serial_port": self.config.serial.port,
                "last_actions": list(self.state.last_actions[-10:]),
            }

    def scan_servos(self, start_id: int = 1, end_id: int = 30) -> list[int]:
        scan = getattr(self.bus, "scan", None)
        if scan is None:
            raise ValueError(f"Scan is not implemented for {self.config.serial.protocol}")
        found = scan(start_id, end_id)
        with self._lock:
            self.state.last_actions.append(f"scan {start_id}-{end_id}: {found}")
        return found

    def execute(self, action: dict[str, Any]) -> str:
        kind = str(action.get("type", "")).lower()
        if kind == "json":
            send_json = getattr(self.bus, "send_json", None)
            if send_json is None:
                raise ValueError(f"JSON commands are not supported by {self.config.serial.protocol}")
            result = send_json(dict(action.get("payload", {})))
            with self._lock:
                self.state.last_actions.append(result)
            return result
        if kind == "preset":
            return self.move_preset(str(action.get("name", "")))
        if kind == "move_joint":
            return self.move_joint(
                str(action.get("joint", "")),
                int(action.get("position")),
                int(action.get("duration_ms", self.config.motion.default_duration_ms)),
            )
        if kind == "nudge_joint":
            return self.nudge_joint(
                str(action.get("joint", "")),
                int(action.get("delta")),
                int(action.get("duration_ms", self.config.motion.default_duration_ms)),
            )
        if kind == "set_gripper":
            return self.set_gripper(str(action.get("state", "open")))
        if kind in {"observe", "wait"}:
            return f"{kind}: no servo motion"
        if kind in {"finish", "stop", "ask_human"}:
            return f"{kind}: {action.get('reason', '')}".strip()
        raise ValueError(f"Unsupported action type: {kind}")

    def move_preset(self, name: str) -> str:
        if self.config.serial.protocol == "roarm_json" and name == "home":
            send_json = getattr(self.bus, "send_json", None)
            result = send_json({"T": 100})
            with self._lock:
                for joint_name, joint in self.config.joints.items():
                    self.state.positions[joint_name] = joint.home
                self.state.last_actions.append(result)
            return result
        if name not in self.config.presets:
            raise ValueError(f"Unknown preset: {name}")
        commands: list[ServoCommand] = []
        with self._lock:
            for joint_name, raw_position in self.config.presets[name].items():
                joint = self._joint(joint_name)
                position = joint.clamp(raw_position)
                self.state.positions[joint_name] = position
                commands.append(ServoCommand(joint.servo_id, position, self.config.motion.default_duration_ms))
            messages = self.bus.move_many(commands)
            result = f"preset {name}: " + "; ".join(messages)
            self.state.last_actions.append(result)
            return result

    def move_joint(self, joint_name: str, position: int, duration_ms: int) -> str:
        joint = self._joint(joint_name)
        target = joint.clamp(position)
        duration_ms = self._duration(duration_ms)
        with self._lock:
            self.state.positions[joint_name] = target
            result = self.bus.move(ServoCommand(joint.servo_id, target, duration_ms))
            self.state.last_actions.append(result)
            return result

    def nudge_joint(self, joint_name: str, delta: int, duration_ms: int) -> str:
        joint = self._joint(joint_name)
        delta = max(-self.config.motion.max_step_units, min(self.config.motion.max_step_units, delta))
        with self._lock:
            current = self.state.positions.get(joint_name, joint.home)
        return self.move_joint(joint_name, current + delta, duration_ms)

    def set_gripper(self, state: str) -> str:
        gripper = self._joint("gripper")
        normalized = state.lower()
        if normalized in {"close", "closed", "grip"}:
            if gripper.closed is None:
                raise ValueError("Gripper closed position is not configured")
            return self.move_joint("gripper", gripper.closed, self.config.motion.default_duration_ms)
        if gripper.open is None:
            raise ValueError("Gripper open position is not configured")
        return self.move_joint("gripper", gripper.open, self.config.motion.default_duration_ms)

    def _joint(self, name: str):
        if name not in self.config.joints:
            raise ValueError(f"Unknown joint: {name}")
        return self.config.joints[name]

    def _duration(self, duration_ms: int) -> int:
        return max(100, min(5000, int(duration_ms)))
