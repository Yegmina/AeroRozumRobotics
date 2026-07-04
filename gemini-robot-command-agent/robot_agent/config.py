from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JointConfig:
    name: str
    servo_id: int
    minimum: int
    maximum: int
    home: int
    open: int | None = None
    closed: int | None = None

    @classmethod
    def from_json(cls, name: str, data: dict[str, Any]) -> "JointConfig":
        return cls(
            name=name,
            servo_id=int(data["servo_id"]),
            minimum=int(data["min"]),
            maximum=int(data["max"]),
            home=int(data["home"]),
            open=None if "open" not in data else int(data["open"]),
            closed=None if "closed" not in data else int(data["closed"]),
        )

    def clamp(self, value: int) -> int:
        return max(self.minimum, min(self.maximum, int(value)))


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baudrate: int
    timeout_s: float
    protocol: str


@dataclass(frozen=True)
class MotionConfig:
    default_duration_ms: int
    max_step_units: int
    dry_run: bool


@dataclass(frozen=True)
class AppConfig:
    serial: SerialConfig
    motion: MotionConfig
    joints: dict[str, JointConfig]
    presets: dict[str, dict[str, int]]


def load_config(path: Path) -> AppConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppConfig(
        serial=SerialConfig(
            port=str(data["serial"]["port"]),
            baudrate=int(data["serial"].get("baudrate", 115200)),
            timeout_s=float(data["serial"].get("timeout_s", 0.1)),
            protocol=str(data["serial"].get("protocol", "lewansoul_lx16a")),
        ),
        motion=MotionConfig(
            default_duration_ms=int(data["motion"].get("default_duration_ms", 800)),
            max_step_units=int(data["motion"].get("max_step_units", 120)),
            dry_run=bool(data["motion"].get("dry_run", True)),
        ),
        joints={name: JointConfig.from_json(name, item) for name, item in data["joints"].items()},
        presets={
            str(name): {str(joint): int(value) for joint, value in positions.items()}
            for name, positions in data.get("presets", {}).items()
        },
    )
