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
