# RoboCrew Jetson Deployment

This directory contains the customized RoboCrew runtime for the XLeRobot
deployment. It keeps the upstream RoboCrew functionality and adds a shared
Gemini configuration plus Gemini TTS.

## Environment

Create a `.env` file in this directory or its parent repository directory.
Do not commit the key.

```dotenv
GEMINI_API_KEY=replace-with-your-key
GEMINI_ROBOTICS_MODEL=gemini-robotics-er-1.6-preview
GEMINI_REASONING_MODEL=gemini-3.1-pro-preview
GEMINI_SPEAKING_MODEL=gemini-3.5-flash
GEMINI_TTS_MODEL=gemini-2.5-flash-preview-tts
GEMINI_TTS_VOICE=Zephyr
```

`GOOGLE_API_KEY` is also accepted. The runtime maps either key name to the
provider environment needed by the Gemini integrations.

## Install on Jetson

Use a Python version supported by the pinned RoboCrew dependencies, then run:

```bash
cd robocrew-jetson
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Hardware Mapping

Create stable udev aliases before enabling real motion:

```text
/dev/camera_center
/dev/camera_left
/dev/camera_right
/dev/arm_left
/dev/arm_right
/dev/lidar        # optional
```

The depth camera can be used initially through its RGB stream. Arm cameras
are configured per VLA manipulation tool. LiDAR is optional and is enabled
only when `/dev/lidar` exists.

## Verified XLERobot Jetson Wiring

For the connected robot, set these before starting the UI (or add them to
`.env`):

```dotenv
ROBOCREW_LEFT_ARM_WHEEL_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61035726-if00
ROBOCREW_RIGHT_ARM_HEAD_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00
ROBOCREW_LEFT_CAMERA_PORT=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0
ROBOCREW_RIGHT_CAMERA_PORT=/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0
ROBOCREW_ORBBEC_DEPTH_PORT=/dev/video4
ROBOCREW_CENTER_CAMERA_PORT=/dev/v4l/by-path/platform-3610000.usb-usb-0:2:1.4-video-index0
ROBOCREW_RGBD_SOCKET=/tmp/robocrew-orbbec.sock
ROBOCREW_ORBBEC_PYTHON=/home/jetsonl4/aerorozumdatacollectiondepth/.venv/bin/python
```

The serial aliases are stable across `/dev/ttyACM*` renumbering.
`5B61035726` contains the physical right arm's servos `1-6` and wheel servos
`7-9`. `5B3D040991` contains the physical left arm's servos `1-6` and
head/depth-camera servos `7-8`. Orbbec video nodes can renumber after USB reconnects: identify
the RGB YUYV and Z16 depth streams and update the Orbbec environment paths
before starting the UI. Calibrate both arms before enabling VLA manipulation.

## First Run

Start with camera-only observation and manual hardware checks. Do not give the
agent movement or arm tools until servo IDs, calibration files, wheel control,
and emergency stop behavior are verified.

```bash
robocrew-gui
```

Use a phone or laptop browser to type tasks. A Jetson microphone is not
required. Voice input is optional; Gemini TTS is only used when the `say` tool
is enabled by an agent configuration.
