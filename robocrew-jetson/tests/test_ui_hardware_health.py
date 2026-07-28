import unittest
from unittest.mock import patch

from robocrew.ui.utils import _device_available


class TestHardwareAvailability(unittest.TestCase):
    @patch("robocrew.ui.utils.os.path.exists", return_value=False)
    @patch("robocrew.ui.utils.RGBDServiceClient")
    def test_rgbd_service_keeps_center_camera_available_without_v4l_node(self, client, _exists):
        client.return_value.health.return_value = True
        self.assertTrue(_device_available("camera_center", "/dev/video11"))

    @patch("robocrew.ui.utils.os.path.exists", return_value=False)
    @patch("robocrew.ui.utils.RGBDServiceClient")
    def test_center_camera_offline_when_service_and_device_are_missing(self, client, _exists):
        client.return_value.health.return_value = False
        self.assertFalse(_device_available("camera_center", "/dev/video11"))


if __name__ == "__main__":
    unittest.main()
