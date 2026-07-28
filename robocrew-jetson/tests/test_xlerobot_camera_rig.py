import base64
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from robocrew.robots.XLeRobot.camera_rig import (
    AuxiliaryCameraRig,
    CameraObservation,
    CameraUnavailable,
)
from robocrew.robots.XLeRobot.tools import (
    RobotSafetyState,
    create_inspect_arm_workspace,
    create_move_arm_joint,
    create_move_forward,
    create_verified_finish_task,
)
from robocrew.robots.XLeRobot.tools import create_inspect_cameras


class TestAuxiliaryCameraRig(unittest.TestCase):
    def make_rig(self):
        return AuxiliaryCameraRig(MagicMock(), "/left", "/right", "/depth")

    def test_colorize_depth_returns_jpeg_and_distance_metadata(self):
        frame = np.full((100, 120), 1500, dtype=np.uint16)

        jpeg, details = AuxiliaryCameraRig.colorize_depth(frame)

        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertIn("1.50 m", details)
        self.assertIn("Valid pixels: 100.0%", details)

    def test_colorize_depth_rejects_frame_without_valid_pixels(self):
        frame = np.zeros((20, 20), dtype=np.uint16)

        with self.assertRaises(CameraUnavailable):
            AuxiliaryCameraRig.colorize_depth(frame)

    def test_depth_capture_reopens_center_rgb_after_success(self):
        rig = self.make_rig()
        frame = np.full((40, 40), 1200, dtype=np.uint16)
        with patch.object(rig, "_read_depth_frame", return_value=frame):
            observation = rig.capture("depth")

        rig.main_camera.release.assert_called_once_with()
        rig.main_camera.reopen.assert_called_once_with()
        self.assertEqual(observation.view, "depth")

    def test_depth_capture_reopens_center_rgb_after_failure(self):
        rig = self.make_rig()
        with patch.object(rig, "_read_depth_frame", side_effect=CameraUnavailable("offline")):
            with self.assertRaises(CameraUnavailable):
                rig.capture("depth")

        rig.main_camera.release.assert_called_once_with()
        rig.main_camera.reopen.assert_called_once_with()

    def test_unknown_view_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_rig().capture("rear")


class TestInspectCamerasTool(unittest.TestCase):
    def test_all_expands_to_three_labeled_images(self):
        rig = MagicMock()

        def capture(view):
            return CameraObservation(view, f"{view.title()} view", f"{view}-jpeg".encode(), "details")

        rig.capture.side_effect = capture
        result, content = create_inspect_cameras(rig).invoke({"views": ["all"]})

        self.assertIn("left, right, depth", result)
        self.assertEqual(rig.capture.call_count, 3)
        images = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(images), 3)
        encoded = images[0]["image_url"]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), b"left-jpeg")

    def test_partial_failure_keeps_successful_camera_output(self):
        rig = MagicMock()

        def capture(view):
            if view == "right":
                raise CameraUnavailable("busy")
            return CameraObservation(view, "Left view", b"jpeg")

        rig.capture.side_effect = capture
        result, content = create_inspect_cameras(rig).invoke({"views": ["left", "right"]})

        self.assertIn("Captured auxiliary camera views: left", result)
        self.assertIn("Unavailable: right", result)
        self.assertTrue(any(item["type"] == "image_url" for item in content))
        self.assertTrue(any("Right camera unavailable" in item.get("text", "") for item in content))


class TestRobotSafetyGates(unittest.TestCase):
    def test_translation_requires_fresh_drive_path_scan(self):
        state = RobotSafetyState()
        controller = MagicMock()
        move = create_move_forward(controller, safety_state=state)

        self.assertTrue(move.invoke({"distance_meters": 0.2}).startswith("BLOCKED:"))
        controller.go_forward.assert_not_called()

        state.mark_drive_path_checked()
        move.invoke({"distance_meters": 0.2})
        controller.go_forward.assert_called_once_with(0.2)

    def test_arm_motion_requires_matching_arm_camera_and_invalidates_it(self):
        state = RobotSafetyState()
        controller = MagicMock()
        controller.move_arm_joint_relative_raw.return_value = (1500, 2000)
        move = create_move_arm_joint(controller, state)

        self.assertTrue(move.invoke({"arm": "right", "joint": "elbow_flex", "relative_steps": 600}).startswith("BLOCKED:"))
        state.mark_arm_camera_checked("right")
        move.invoke({"arm": "right", "joint": "elbow_flex", "relative_steps": 600})

        controller.move_arm_joint_relative_raw.assert_called_once_with("right", "elbow_flex", 600)
        self.assertEqual(state.unverified_arms(), ["right"])

    def test_right_shoulder_lift_positive_step_uses_decreasing_raw_direction(self):
        state = RobotSafetyState()
        state.mark_arm_camera_checked("right")
        controller = MagicMock()
        controller.move_arm_joint_relative_raw.return_value = (2500, 1900)

        create_move_arm_joint(controller, state).invoke(
            {"arm": "right", "joint": "shoulder_lift", "relative_steps": 600}
        )

        controller.move_arm_joint_relative_raw.assert_called_once_with("right", "shoulder_lift", -600)

    def test_left_shoulder_lift_positive_step_uses_decreasing_raw_direction(self):
        state = RobotSafetyState()
        state.mark_arm_camera_checked("left")
        controller = MagicMock()
        controller.move_arm_joint_relative_raw.return_value = (2100, 1500)

        create_move_arm_joint(controller, state).invoke(
            {"arm": "left", "joint": "shoulder_lift", "relative_steps": 600}
        )

        controller.move_arm_joint_relative_raw.assert_called_once_with("left", "shoulder_lift", -600)

    def test_arm_camera_inspection_clears_finish_block(self):
        state = RobotSafetyState()
        state.mark_arm_moved("left")
        finish = create_verified_finish_task(state)
        self.assertTrue(finish.invoke({"report": "done"}).startswith("BLOCKED:"))

        rig = MagicMock()
        rig.capture.side_effect = [
            CameraObservation("left", "Left", b"left"),
            CameraObservation("depth", "Depth", b"depth", "0.5 m"),
        ]
        create_inspect_arm_workspace(rig, state).invoke({"arm": "left"})

        self.assertEqual(finish.invoke({"report": "done"}), "done")
        self.assertEqual([call.args[0] for call in rig.capture.call_args_list], ["left", "depth"])


if __name__ == "__main__":
    unittest.main()
