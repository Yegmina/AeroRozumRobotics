# Gemini Robot Command Agent

Local app for giving text goals to the XLERobot arm while using the left arm
camera as visual feedback.

The app uses:

- Gemini Robotics-ER 1.6 for camera-frame spatial reasoning.
- A separate Gemini reasoning model for task planning and progress checks.
- A serial servo adapter for the USB Serial Bus Servo Driver Board on `COM11`.
- Dry-run motion by default until the arm servo IDs and joint limits are
  calibrated in `servo_config.json`.

## Run

```powershell
cd C:\business\AeroRozum\AeroRozumRobotics\gemini-robot-command-agent
py -3 app.py
```

Open `http://127.0.0.1:7860`.

Real servo writes are disabled unless you launch:

```powershell
py -3 app.py --enable-motion
```

The `--enable-motion` flag is the final hardware-enable switch. The joint
limits in `servo_config.json` are still applied before any serial packet is
written.

The default serial protocol is `feetech_sts`, which controls the transparent
ST/SC serial bus at `1000000` baud on `COM11`.

If the arm does not move while `dry_run=false`, check these first:

- The arm controller is powered and switched on.
- The USB cable is plugged into the controller port used for serial control.
- `COM11` is still the CH343/robot controller port.
- Press `Scan Servos`; expected responding IDs on this setup are `2`, `3`,
  `4`, `5`, and sometimes `6`.

## Environment

The app loads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment or from
the repo root `.env`.

Optional overrides:

```powershell
$env:GEMINI_ROBOTICS_MODEL="gemini-robotics-er-1.6-preview"
$env:GEMINI_REASONING_MODEL="gemini-3.1-pro-preview"
$env:LEFT_ARM_CAMERA_DEVICE="USB2.0_CAM1"
```

Use `LEFT_ARM_CAMERA_DEVICE` when Windows changes the USB instance path after
replugging. If both arm cameras are connected at the same time, set it to the
exact DirectShow alternative name for the left camera instead of the friendly
`USB2.0_CAM1` name.

## Calibration

Before enabling motion, verify every joint in `servo_config.json`:

- `servo_id` must match the physical bus servo ID.
- `min` and `max` must be safe for the mounted arm.
- `home`, `open`, and `closed` should be tested in small steps.

Use dry-run first. Then enable motion only with the arm clear of people,
cables, and fragile objects.

## Hardware Evidence Test

With the server running in real-motion mode:

```powershell
py -3 run_hardware_evidence_test.py
```

This runs a servo scan and multiple live-readback joint moves, captures
before/after camera frames, writes `hardware_evidence.json`, and generates:

```text
..\output\pdf\robot_hardware_evidence\robot_hardware_evidence_report.pdf
```

## Runtime Connection Diagnostics

If the web backend is running and holding `COM11`:

```powershell
py -3 runtime_hardware_diagnostics.py --via-http --count 0
```

If the backend is stopped and you want direct serial diagnostics:

```powershell
py -3 runtime_hardware_diagnostics.py --count 0
```

It prints available COM ports, scanned servo IDs, live readback positions, and
position deltas. Add `--log-file logs\hardware_diagnostics.log` to save output.
