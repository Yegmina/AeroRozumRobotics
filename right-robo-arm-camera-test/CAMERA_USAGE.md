# Right USB Camera Access

The right robo arm camera is exposed through a unique DirectShow device path.
Both arm cameras use the same friendly name:

```text
USB2.0_CAM1
```

Use the reusable wrapper instead of OpenCV camera indexes, because index `0` can
open the laptop webcam, and the friendly name is duplicated when both arm
cameras are connected.

## Live Viewer

```powershell
py -3 camera_livefeed.py
```

Optional settings:

```powershell
py -3 camera_livefeed.py --width 1280 --height 720 --fps 30
```

Press `q` or `Esc` in the feed window to quit.

## Read Frames In Robotics Code

Blocking one-frame-at-a-time access:

```python
from robotics_camera import CameraConfig, DirectShowCamera

config = CameraConfig(width=1280, height=720, fps=30)

with DirectShowCamera(config) as camera:
    frame = camera.read()
    # frame is a writable OpenCV BGR numpy array
```

Background latest-frame access for control loops:

```python
import time

from robotics_camera import CameraConfig, DirectShowCamera

config = CameraConfig(width=1280, height=720, fps=30)

with DirectShowCamera(config).start_background() as camera:
    while True:
        frame = camera.latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Perception/control code goes here.
```
