# AeroRozum Robotics Camera Feed

This project accesses the camera feed from the left robo arm through its unique
Windows DirectShow device path.

We avoid OpenCV camera indexes because index `0` opened the laptop webcam on
this machine. Both arm cameras report the same friendly name, `USB2.0_CAM1`, so
the reusable wrapper opens the left USB camera by its DirectShow device path
through FFmpeg and returns normal OpenCV BGR `numpy` frames for future AI agent
development.

## Live Feed

```powershell
py -3 camera_livefeed.py
```

## Bottle And Can-Like Detection

```powershell
py -3 detect_cans_bottles.py
```

This uses Faster R-CNN ResNet50 FPN V2 from torchvision instead of YOLO. It is
larger than the previous nano model and uses COCO `bottle`, `cup`, and
`wine glass` classes by default. Aluminum cans may appear as cup/bottle-like
objects unless a can-specific custom model is added.

## Use In Agent Code

```python
from robotics_camera import CameraConfig, DirectShowCamera

config = CameraConfig(width=1280, height=720, fps=30)

with DirectShowCamera(config).start_background() as camera:
    frame = camera.latest_frame()
```

`frame` is a writable OpenCV BGR image. Use `camera.read()` instead when blocking
one-frame-at-a-time access is preferred.
