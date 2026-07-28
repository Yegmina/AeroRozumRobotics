import socket
import threading
import unittest
from unittest.mock import MagicMock

import numpy as np
import cv2

from robocrew.robots.XLeRobot.manipulation import (
    GraspController,
    ManipulationCalibration,
    TabletopPerception,
    default_manipulation_calibration,
)
from robocrew.robots.XLeRobot.rgbd_client import RGBDFrame
from robocrew.robots.XLeRobot.rgbd_protocol import receive_packet, send_packet


class TestRGBDProtocol(unittest.TestCase):
    def test_packet_round_trip(self):
        left, right = socket.socketpair()
        try:
            sender = threading.Thread(target=send_packet, args=(left, {"ok": True}, b"depth"))
            sender.start()
            metadata, payload = receive_packet(right)
            sender.join()
        finally:
            left.close()
            right.close()
        self.assertEqual(metadata, {"ok": True})
        self.assertEqual(payload, b"depth")


class TestTabletopPerception(unittest.TestCase):
    def make_frame(self):
        height, width = 240, 320
        depth = np.full((height, width), 1000, dtype=np.uint16)
        depth[85:165, 135:185] = 880
        color = np.full((height, width, 3), 180, dtype=np.uint8)
        color[85:165, 135:185] = 15
        return RGBDFrame(
            color,
            depth,
            {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "width": width, "height": height},
            1,
        )

    def test_detects_black_object_above_plane(self):
        result = TabletopPerception().detect(self.make_frame())
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertGreater(candidate.height_m, 0.09)
        self.assertLess(float(candidate.mean_bgr.mean()), 30)

    def test_color_description_selects_candidate(self):
        perception = TabletopPerception()
        result = perception.detect(self.make_frame())
        selected = perception.select_candidate(result, "black Vichy can")
        self.assertEqual(selected.candidate_id, 1)


class TestManipulationCalibration(unittest.TestCase):
    def test_default_calibration_blocks_motion(self):
        calibration = ManipulationCalibration(default_manipulation_calibration())
        self.assertFalse(calibration.ready)
        self.assertTrue(any("not been approved" in error for error in calibration.validation_errors()))

    def test_grasp_controller_returns_block_without_hardware_motion(self):
        controller = MagicMock()
        calibration = ManipulationCalibration(default_manipulation_calibration())
        grasp = GraspController(controller, MagicMock(), MagicMock(), calibration=calibration)
        result, content = grasp.execute("black can")
        self.assertTrue(result.startswith("BLOCKED:"))
        self.assertEqual(content, [])
        controller.go_forward.assert_not_called()

    def test_wrist_verification_requires_centered_requested_color(self):
        image = np.full((240, 320, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (130, 70), (190, 190), (10, 10, 10), -1)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        self.assertTrue(GraspController._target_visible_in_wrist(encoded.tobytes(), "black can"))
        self.assertFalse(GraspController._target_visible_in_wrist(encoded.tobytes(), "red can"))

    def test_drive_path_with_no_depth_is_not_clear(self):
        frame = RGBDFrame(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100), dtype=np.uint16),
            {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
            1,
        )
        self.assertFalse(GraspController._drive_path_clear(frame))


if __name__ == "__main__":
    unittest.main()
