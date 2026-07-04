from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from .config import AppConfig
from .servo_bus import LewanSoulLX16ABus, ServoCommand


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
            motion_enabled=enable_motion and not config.motion.dry_run,
        )
        dry_run = (not enable_motion) or config.motion.dry_run
        if config.serial.protocol != "lewansoul_lx16a":
            raise ValueError(f"Unsupported servo protocol: {config.serial.protocol}")
        self.bus = LewanSoulLX16ABus(
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
                "motion_enabled": self.state.motion_enabled,
                "dry_run": self.dry_run,
                "serial_port": self.config.serial.port,
                "last_actions": list(self.state.last_actions[-10:]),
            }

    def execute(self, action: dict[str, Any]) -> str:
        kind = str(action.get("type", "")).lower()
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
