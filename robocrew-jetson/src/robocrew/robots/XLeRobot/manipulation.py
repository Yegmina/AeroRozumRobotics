"""Deterministic RGB-D perception and calibrated tabletop grasp execution."""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from lerobot.model.SO101Robot import SO101Kinematics
from robocrew.robots.XLeRobot.rgbd_client import RGBDFrame
from robocrew.robots.XLeRobot.servo_controls import ARM_SERVO_MAPS


CALIBRATION_PATH = Path("~/.cache/robocrew/calibrations/manipulation.json").expanduser()
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
COLOR_BGR = {
    "black": np.array([30, 30, 30], dtype=np.float32),
    "white": np.array([225, 225, 225], dtype=np.float32),
    "red": np.array([45, 45, 190], dtype=np.float32),
    "green": np.array([55, 145, 55], dtype=np.float32),
    "blue": np.array([180, 95, 45], dtype=np.float32),
    "yellow": np.array([40, 205, 220], dtype=np.float32),
}


def default_manipulation_calibration(servo_controller=None) -> dict:
    def arm(side: str, base_xyz: list[float]) -> dict:
        calibrations = getattr(servo_controller, f"_{side}_arm_calibration", {})
        servo_map = getattr(__import__(
            "robocrew.robots.XLeRobot.servo_controls", fromlist=["ARM_SERVO_MAPS"]
        ), "ARM_SERVO_MAPS")[side]
        zero_raw = {}
        for joint, servo_id in servo_map.items():
            calibration = calibrations.get(servo_id)
            zero_raw[joint] = int((calibration.range_min + calibration.range_max) / 2) if calibration else 2048
        gripper_cal = calibrations.get(servo_map["gripper"])
        return {
            "base_xyz_m": base_xyz,
            "zero_raw": zero_raw,
            "raw_sign_per_positive_degree": {
                "shoulder_pan": 1,
                "shoulder_lift": -1,
                "elbow_flex": 1,
                "wrist_flex": 1,
                "wrist_roll": 1,
            },
            "gripper_open_raw": int(gripper_cal.range_max) if gripper_cal else 2800,
            "gripper_closed_raw": int(gripper_cal.range_min) if gripper_cal else 1750,
            "gripper_load_threshold": 250,
            "directions_verified": False,
        }

    # Optical camera coordinates (x right, y down, z forward) to robot base
    # coordinates (x forward, y left, z up), initialized from the XLeRobot URDF.
    return {
        "version": 1,
        "ready": False,
        "board_validated": False,
        "head_to_base": [
            [0.0, 0.469472, 0.882948, -0.092],
            [-1.0, 0.0, 0.0, 0.020],
            [0.0, -0.882948, 0.469472, 1.19065],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "arms": {
            "right": arm("right", [-0.135, -0.133, 0.760]),
            "left": arm("left", [-0.135, 0.133, 0.760]),
        },
        "pregrasp_offset_m": 0.075,
        "lift_height_m": 0.05,
        "max_retries": 2,
    }


class ManipulationCalibration:
    def __init__(self, data: dict):
        self.data = data

    @classmethod
    def load(cls, servo_controller=None, path: Path = CALIBRATION_PATH) -> "ManipulationCalibration":
        if not path.exists():
            return cls(default_manipulation_calibration(servo_controller))
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.data.get("version") != 1:
            errors.append("Unsupported calibration version")
        matrix = np.asarray(self.data.get("head_to_base"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append("Head-to-base transform is invalid")
        for side in ("left", "right"):
            arm = self.data.get("arms", {}).get(side, {})
            if not arm.get("directions_verified"):
                errors.append(f"{side} arm directions and gripper endpoints are not verified")
            if not all(joint in arm.get("zero_raw", {}) for joint in ARM_JOINTS):
                errors.append(f"{side} arm zero positions are incomplete")
        if not self.data.get("ready"):
            errors.append("Manipulation calibration has not been approved")
        if not self.data.get("board_validated"):
            errors.append("Calibration board has not been detected by the head camera")
        return errors

    @property
    def ready(self) -> bool:
        return not self.validation_errors()


@dataclass
class ObjectCandidate:
    candidate_id: int
    bbox: tuple[int, int, int, int]
    center_camera_m: np.ndarray
    height_m: float
    width_m: float
    mean_bgr: np.ndarray
    pixel_area: int

    def as_dict(self) -> dict:
        return {
            "id": self.candidate_id,
            "bbox": list(self.bbox),
            "center_camera_m": self.center_camera_m.round(4).tolist(),
            "height_m": round(self.height_m, 3),
            "width_m": round(self.width_m, 3),
            "pixel_area": self.pixel_area,
        }


@dataclass
class PerceptionResult:
    candidates: list[ObjectCandidate]
    annotated_jpeg: bytes
    plane: np.ndarray


class TabletopPerception:
    def __init__(self, random_seed: int = 7):
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def deproject(frame: RGBDFrame) -> np.ndarray:
        depth_m = frame.depth_mm.astype(np.float32) / 1000.0
        height, width = depth_m.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        intr = frame.intrinsics
        x = (u - float(intr["cx"])) * depth_m / float(intr["fx"])
        y = (v - float(intr["cy"])) * depth_m / float(intr["fy"])
        return np.dstack((x, y, depth_m))

    def _fit_table_plane(self, points: np.ndarray, valid: np.ndarray) -> np.ndarray:
        height = points.shape[0]
        sample_mask = valid.copy()
        sample_mask[: int(height * 0.28)] = False
        samples = points[sample_mask]
        if samples.shape[0] < 500:
            raise RuntimeError("Not enough valid depth points to find a tabletop")
        if samples.shape[0] > 18000:
            samples = samples[self.rng.choice(samples.shape[0], 18000, replace=False)]

        best_plane = None
        best_count = 0
        for _ in range(90):
            selected = samples[self.rng.choice(samples.shape[0], 3, replace=False)]
            normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm
            offset = -float(np.dot(normal, selected[0]))
            distances = np.abs(samples @ normal + offset)
            count = int(np.count_nonzero(distances < 0.009))
            if count > best_count:
                best_count = count
                best_plane = np.r_[normal, offset]
        if best_plane is None or best_count < 350:
            raise RuntimeError("No stable tabletop plane found")
        # Point the normal toward the camera so objects above the table are positive.
        if best_plane[3] < 0:
            best_plane *= -1
        return best_plane

    def detect(self, frame: RGBDFrame) -> PerceptionResult:
        points = self.deproject(frame)
        depth = points[:, :, 2]
        valid = (depth > 0.18) & (depth < 1.8)
        plane = self._fit_table_plane(points, valid)
        signed = points @ plane[:3] + plane[3]
        object_mask = valid & (signed > 0.025) & (signed < 0.26)
        object_mask = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask, connectivity=8)

        candidates: list[ObjectCandidate] = []
        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if area < 180 or area > 30000 or width < 8 or height < 12:
                continue
            image_height, image_width = object_mask.shape
            if x <= 4 or y <= 4 or x + width >= image_width - 4 or y + height >= image_height - 4:
                continue
            if height < width * 1.05:
                continue
            component = labels == label
            component_points = points[component]
            component_heights = signed[component]
            center = np.median(component_points, axis=0)
            span = np.percentile(component_points, 95, axis=0) - np.percentile(component_points, 5, axis=0)
            # Depth above the table carries object height; horizontal camera X
            # carries the side-grasp width. Camera Y includes that same height
            # and must not be mistaken for object diameter.
            physical_width = float(span[0])
            physical_height = float(np.percentile(component_heights, 95))
            if not (0.025 <= physical_width <= 0.15 and 0.05 <= physical_height <= 0.24):
                continue
            mean_bgr = frame.color_bgr[component].mean(axis=0)
            candidates.append(ObjectCandidate(
                len(candidates) + 1,
                (int(x), int(y), int(x + width), int(y + height)),
                center,
                physical_height,
                physical_width,
                mean_bgr,
                int(area),
            ))

        annotated = frame.color_bgr.copy()
        for candidate in candidates:
            x0, y0, x1, y1 = candidate.bbox
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 220, 255), 2)
            cv2.putText(
                annotated,
                f"#{candidate.candidate_id} {candidate.width_m*100:.0f}x{candidate.height_m*100:.0f}cm",
                (x0, max(20, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2,
            )
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            raise RuntimeError("Candidate image encoding failed")
        return PerceptionResult(candidates, encoded.tobytes(), plane)

    @staticmethod
    def select_candidate(
        result: PerceptionResult,
        description: str,
        candidate_id: int | None = None,
    ) -> ObjectCandidate:
        if candidate_id is not None:
            for candidate in result.candidates:
                if candidate.candidate_id == candidate_id:
                    return candidate
            raise RuntimeError(f"Candidate #{candidate_id} is not present in the current frame")
        if not result.candidates:
            raise RuntimeError("No graspable upright tabletop objects were detected")

        description_lower = description.lower()
        requested_color = next((name for name in COLOR_BGR if name in description_lower), None)
        scored: list[tuple[float, ObjectCandidate]] = []
        for candidate in result.candidates:
            can_shape = abs(candidate.width_m - 0.065) + 0.5 * abs(candidate.height_m - 0.12)
            color_score = 0.0
            if requested_color:
                color_score = float(np.linalg.norm(candidate.mean_bgr - COLOR_BGR[requested_color])) / 255.0
            scored.append((can_shape + color_score, candidate))
        scored.sort(key=lambda item: item[0])
        if len(scored) > 1 and scored[1][0] - scored[0][0] < 0.12:
            raise RuntimeError(
                "AMBIGUOUS: multiple objects match; ask the user to choose candidate "
                + ", ".join(f"#{item[1].candidate_id}" for item in scored[:3])
            )
        return scored[0][1]


def image_content(label: str, jpeg_bytes: bytes) -> list[dict]:
    return [
        {"type": "text", "text": label},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('ascii')}"},
        },
    ]


class GraspController:
    """Closed-loop tabletop grasp state machine for upright rigid objects."""

    def __init__(self, servo_controller, main_camera, camera_rig, calibration=None):
        self.servo_controller = servo_controller
        self.main_camera = main_camera
        self.camera_rig = camera_rig
        self.calibration = calibration or ManipulationCalibration.load(servo_controller)
        self.perception = TabletopPerception()
        self.kinematics = SO101Kinematics()
        self.cancel_event = threading.Event()
        self.status = {
            "stage": "idle",
            "message": "Ready",
            "target": None,
            "arm": None,
            "action": None,
            "candidate_image": None,
        }
        self._lock = threading.RLock()

    def cancel(self) -> None:
        self.cancel_event.set()
        try:
            self.servo_controller._wheels_stop()
        except Exception:
            pass
        self._set_status("cancelled", "Emergency stop requested")

    def _set_status(self, stage: str, message: str, **values) -> None:
        self.status.update({"stage": stage, "message": message, **values})

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise RuntimeError("Manipulation cancelled by operator")

    def _capture_candidates(self) -> tuple[RGBDFrame, PerceptionResult]:
        self.servo_controller.point_head_relative(yaw_degrees=0, pitch_degrees=28)
        frame = self.main_camera.capture_rgbd_frame()
        result = self.perception.detect(frame)
        self.status["candidate_image"] = result.annotated_jpeg
        return frame, result

    def _camera_to_base(self, camera_point: np.ndarray) -> np.ndarray:
        transform = np.asarray(self.calibration.data["head_to_base"], dtype=np.float64)
        homogeneous = np.r_[camera_point.astype(np.float64), 1.0]
        return (transform @ homogeneous)[:3]

    def _select(
        self,
        description: str,
        candidate_id: int | None,
    ) -> tuple[RGBDFrame, PerceptionResult, ObjectCandidate, np.ndarray]:
        frame, result = self._capture_candidates()
        candidate = self.perception.select_candidate(result, description, candidate_id)
        target_base = self._camera_to_base(candidate.center_camera_m)
        self._set_status(
            "target_selected",
            f"Candidate #{candidate.candidate_id} at {target_base.round(3).tolist()} m in base frame",
            target=candidate.as_dict(),
        )
        return frame, result, candidate, target_base

    @staticmethod
    def _drive_path_clear(frame: RGBDFrame) -> bool:
        height, width = frame.depth_mm.shape
        region = frame.depth_mm[int(height * 0.58):, int(width * 0.32):int(width * 0.68)]
        valid = region[(region > 80) & (region < 900)]
        if valid.size < region.size * 0.03:
            return False
        return float(np.percentile(valid, 10)) > 280.0

    def _approach(
        self,
        description: str,
        candidate_id: int | None,
        target_base: np.ndarray,
    ) -> tuple[ObjectCandidate, np.ndarray]:
        desired_x = 0.12
        for _ in range(5):
            self._check_cancelled()
            remaining = float(target_base[0] - desired_x)
            if remaining <= 0.045:
                break
            frame = self.main_camera.capture_rgbd_frame()
            if not self._drive_path_clear(frame):
                raise RuntimeError("Low obstacle detected in the drive path; approach aborted")
            step = min(0.20, remaining)
            self._set_status("approach", f"Driving forward {step:.2f} m", distance_m=step)
            self.servo_controller.reset_head_position()
            self.servo_controller.go_forward(step)
            _, _, candidate, target_base = self._select(description, candidate_id)
        if target_base[0] - desired_x > 0.07:
            raise RuntimeError("Target remains outside arm reach after bounded approach")
        return candidate, target_base

    def _choose_arm(self, requested: str, target_base: np.ndarray) -> Literal["left", "right"]:
        if requested in ("left", "right"):
            return requested  # type: ignore[return-value]
        return "left" if target_base[1] >= 0 else "right"

    def _joint_raw_targets(
        self,
        arm: Literal["left", "right"],
        target_base: np.ndarray,
        vertical_offset: float = 0.0,
    ) -> dict[int, int]:
        arm_config = self.calibration.data["arms"][arm]
        arm_origin = np.asarray(arm_config["base_xyz_m"], dtype=np.float64)
        relative = target_base - arm_origin
        relative[2] += vertical_offset
        radial = float(np.hypot(relative[0], relative[1]))
        vertical = float(relative[2])
        if radial > 0.255 or radial < 0.025 or not (-0.08 <= vertical <= 0.30):
            raise RuntimeError(
                f"Target is outside {arm} arm workspace: radial={radial:.3f} m, vertical={vertical:.3f} m"
            )
        pan_deg = math.degrees(math.atan2(-relative[1], relative[0]))
        shoulder_deg, elbow_deg = self.kinematics.inverse_kinematics(radial, vertical)
        wrist_deg = -shoulder_deg - elbow_deg
        semantic_degrees = {
            "shoulder_pan": pan_deg,
            "shoulder_lift": shoulder_deg,
            "elbow_flex": elbow_deg,
            "wrist_flex": wrist_deg,
            "wrist_roll": 0.0,
        }
        bus_calibration = getattr(self.servo_controller, f"_{arm}_arm_calibration")
        targets: dict[int, int] = {}
        for joint, degrees in semantic_degrees.items():
            servo_id = ARM_SERVO_MAPS[arm][joint]
            zero = int(arm_config["zero_raw"][joint])
            sign = int(arm_config["raw_sign_per_positive_degree"][joint])
            raw = int(round(zero + sign * degrees * 4095.0 / 360.0))
            limits = bus_calibration[servo_id]
            targets[servo_id] = max(limits.range_min, min(limits.range_max, raw))
        return targets

    def _arm_bus(self, arm: str):
        return self.servo_controller.head_bus if arm == "left" else self.servo_controller.wheel_bus

    def _move_arm_trajectory(
        self,
        arm: Literal["left", "right"],
        targets: dict[int, int],
        duration: float = 2.0,
    ) -> None:
        bus = self._arm_bus(arm)
        starts = {
            servo_id: int(bus.read("Present_Position", servo_id, normalize=False))
            for servo_id in targets
        }
        steps = max(12, int(duration * 20))
        for index in range(1, steps + 1):
            self._check_cancelled()
            alpha = 0.5 - 0.5 * math.cos(math.pi * index / steps)
            values = {
                servo_id: int(round(starts[servo_id] + alpha * (target - starts[servo_id])))
                for servo_id, target in targets.items()
            }
            bus.sync_write("Goal_Position", values, normalize=False)
            time.sleep(duration / steps)

    def _gripper_move(self, arm: Literal["left", "right"], close: bool) -> tuple[int, int]:
        bus = self._arm_bus(arm)
        config = self.calibration.data["arms"][arm]
        servo_id = ARM_SERVO_MAPS[arm]["gripper"]
        start = int(bus.read("Present_Position", servo_id, normalize=False))
        target = int(config["gripper_closed_raw" if close else "gripper_open_raw"])
        steps = 22
        final_load = 0
        for index in range(1, steps + 1):
            self._check_cancelled()
            value = int(round(start + (target - start) * index / steps))
            bus.sync_write("Goal_Position", {servo_id: value}, normalize=False)
            time.sleep(0.06)
            try:
                final_load = abs(int(bus.read("Present_Load", servo_id, normalize=False)))
            except Exception:
                final_load = 0
            if close and final_load >= int(config["gripper_load_threshold"]):
                break
        position = int(bus.read("Present_Position", servo_id, normalize=False))
        return position, final_load

    def _capture_wrist(self, arm: str) -> bytes:
        return self.camera_rig.capture(arm).jpeg_bytes

    @staticmethod
    def _target_visible_in_wrist(jpeg_bytes: bytes, description: str) -> bool:
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False
        requested_color = next((name for name in COLOR_BGR if name in description.lower()), None)
        if requested_color is None:
            return False
        if requested_color == "black":
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 65
        else:
            distance = np.linalg.norm(image.astype(np.float32) - COLOR_BGR[requested_color], axis=2)
            mask = distance < 85
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = mask.shape
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            center_x, center_y = centers[label]
            if (
                120 <= area <= width * height * 0.30
                and 0.15 * width <= center_x <= 0.85 * width
                and 0.12 * height <= center_y <= 0.90 * height
                and x > 2 and y > 2
                and x + component_width < width - 2
                and y + component_height < height - 2
            ):
                return True
        return False

    def _verify_grasp(self, arm: str, position: int, load: int) -> bool:
        config = self.calibration.data["arms"][arm]
        closed = int(config["gripper_closed_raw"])
        open_ = int(config["gripper_open_raw"])
        travel = abs(open_ - closed)
        not_fully_closed = abs(position - closed) > max(20, int(travel * 0.08))
        loaded = load >= max(20, int(config["gripper_load_threshold"] * 0.35))
        return not_fully_closed and loaded

    def execute(
        self,
        target_description: str,
        arm: Literal["auto", "left", "right"] = "auto",
        action: Literal["touch", "grasp", "grasp_and_lift"] = "grasp_and_lift",
        candidate_id: int | None = None,
    ) -> tuple[str, list[dict]]:
        with self._lock:
            errors = self.calibration.validation_errors()
            if errors:
                return "BLOCKED: " + "; ".join(errors), []
            self.cancel_event.clear()
            self._set_status("search", f"Searching for {target_description}", action=action)
            content: list[dict] = []
            selected_arm: Literal["left", "right"] | None = None
            try:
                _, perception, candidate, target_base = self._select(target_description, candidate_id)
                content.extend(image_content("Detected grasp candidates", perception.annotated_jpeg))
                candidate, target_base = self._approach(target_description, None, target_base)
                selected_arm = self._choose_arm(arm, target_base)
                self.status["arm"] = selected_arm
                arm_origin = np.asarray(
                    self.calibration.data["arms"][selected_arm]["base_xyz_m"], dtype=np.float64
                )
                direction = target_base - arm_origin
                direction[2] = 0.0
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    raise RuntimeError("Cannot determine pre-grasp direction")
                pregrasp = target_base - direction / norm * float(self.calibration.data["pregrasp_offset_m"])

                self._set_status("pregrasp", f"Moving {selected_arm} arm to pre-grasp")
                self._gripper_move(selected_arm, close=False)
                self._move_arm_trajectory(selected_arm, self._joint_raw_targets(selected_arm, pregrasp))
                wrist_before = self._capture_wrist(selected_arm)
                content.extend(image_content(f"{selected_arm.title()} wrist pre-grasp view", wrist_before))
                if not self._target_visible_in_wrist(wrist_before, target_description):
                    raise RuntimeError(
                        "Requested target is not visible in the wrist camera at pre-grasp; refusing blind reach"
                    )

                self._set_status("final_reach", "Executing bounded final reach")
                self._move_arm_trajectory(
                    selected_arm,
                    self._joint_raw_targets(selected_arm, target_base),
                    duration=1.4,
                )
                if action == "touch":
                    wrist_after = self._capture_wrist(selected_arm)
                    content.extend(image_content("Post-contact wrist verification", wrist_after))
                    if not self._target_visible_in_wrist(wrist_after, target_description):
                        raise RuntimeError("Target was lost from the wrist camera during touch")
                    self._set_status("complete", "Touch completed and imaged")
                    return f"Touched {target_description} with the {selected_arm} arm.", content

                self._set_status("close", "Closing gripper with load monitoring")
                position, load = self._gripper_move(selected_arm, close=True)
                if not self._verify_grasp(selected_arm, position, load):
                    raise RuntimeError("Gripper closed without evidence that an object was retained")

                if action == "grasp_and_lift":
                    self._set_status("lift", "Lifting object 5 cm for verification")
                    lift = float(self.calibration.data.get("lift_height_m", 0.05))
                    self._move_arm_trajectory(
                        selected_arm,
                        self._joint_raw_targets(selected_arm, target_base, vertical_offset=lift),
                        duration=1.5,
                    )
                    time.sleep(3.0)

                wrist_after = self._capture_wrist(selected_arm)
                content.extend(image_content("Post-grasp wrist verification", wrist_after))
                if not self._target_visible_in_wrist(wrist_after, target_description):
                    raise RuntimeError("Post-grasp wrist image does not contain the requested target")
                self._set_status(
                    "complete",
                    f"Grasp verified at position {position}, load {load}",
                    gripper_position=position,
                    gripper_load=load,
                )
                verb = "grasped and lifted" if action == "grasp_and_lift" else "grasped"
                return f"{verb.title()} {target_description} with the {selected_arm} arm.", content
            except Exception as error:
                self._set_status("failed", str(error))
                if selected_arm is not None:
                    try:
                        self._gripper_move(selected_arm, close=False)
                    except Exception:
                        pass
                try:
                    self.servo_controller._wheels_stop()
                except Exception:
                    pass
                if not content and self.status.get("candidate_image"):
                    content.extend(image_content("Detected grasp candidates", self.status["candidate_image"]))
                return f"FAILED: {error}", content
            finally:
                try:
                    self.servo_controller.reset_head_position()
                except Exception:
                    pass
