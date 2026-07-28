from __future__ import annotations

import os
import subprocess
import time

import cv2
import numpy as np
import serial
import streamlit as st


LEFT_BUS = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B61035726-if00"
RIGHT_BUS = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D040991-if00"

CAMERAS = (
    ("Left camera", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"),
    ("Right camera", "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0"),
    (
        "Center camera (Orbbec RGB)",
        os.environ.get(
            "ROBOCREW_CENTER_CAMERA_PORT",
            "/dev/v4l/by-path/platform-3610000.usb-usb-0:2:1.4-video-index0",
        ),
    ),
    ("Depth camera (Orbbec depth)", os.environ.get("ROBOCREW_ORBBEC_DEPTH_PORT", "/dev/video4")),
)

CAMERA_VIEWS = {
    "Left camera": "left",
    "Right camera": "right",
    "Depth camera (Orbbec depth)": "depth",
}

SERVOS = (
    ("Right arm shoulder pan", LEFT_BUS, 1, "position"),
    ("Right arm shoulder lift", LEFT_BUS, 2, "position"),
    ("Right arm elbow", LEFT_BUS, 3, "position"),
    ("Right arm wrist flex", LEFT_BUS, 4, "position"),
    ("Right arm wrist roll", LEFT_BUS, 5, "position"),
    ("Right arm gripper", LEFT_BUS, 6, "position"),
    ("Base wheel A", LEFT_BUS, 7, "wheel"),
    ("Base wheel B", LEFT_BUS, 8, "wheel"),
    ("Base wheel C", LEFT_BUS, 9, "wheel"),
    ("Left arm shoulder pan", RIGHT_BUS, 1, "position"),
    ("Left arm shoulder lift", RIGHT_BUS, 2, "position"),
    ("Left arm elbow", RIGHT_BUS, 3, "position"),
    ("Left arm wrist flex", RIGHT_BUS, 4, "position"),
    ("Left arm wrist roll", RIGHT_BUS, 5, "position"),
    ("Left arm gripper", RIGHT_BUS, 6, "position"),
    ("Depth camera pan", RIGHT_BUS, 7, "position"),
    ("Depth camera tilt", RIGHT_BUS, 8, "position"),
)


def _packet(servo_id: int, instruction: int, params: list[int]) -> bytes:
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def _read_packet(port: serial.Serial, servo_id: int, timeout_s: float = 0.25) -> bytes | None:
    deadline = time.monotonic() + timeout_s
    data = bytearray()
    while time.monotonic() < deadline:
        chunk = port.read(port.in_waiting or 1)
        if chunk:
            data.extend(chunk)
        start = data.find(b"\xff\xff")
        if start < 0:
            continue
        if start:
            del data[:start]
        if len(data) < 4:
            continue
        size = data[3] + 4
        if len(data) < size:
            continue
        raw = bytes(data[:size])
        del data[:size]
        if raw[2] == servo_id and ((~sum(raw[2:-1])) & 0xFF) == raw[-1]:
            return raw
    return None


def _read_register(port: serial.Serial, servo_id: int, address: int, length: int) -> list[int] | None:
    port.reset_input_buffer()
    port.write(_packet(servo_id, 2, [address, length]))
    port.flush()
    raw = _read_packet(port, servo_id)
    return list(raw[5:-1]) if raw else None


def _write_register(port: serial.Serial, servo_id: int, address: int, values: list[int]) -> None:
    port.reset_input_buffer()
    port.write(_packet(servo_id, 3, [address, *values]))
    port.flush()
    _read_packet(port, servo_id, 0.15)


def _u16(value: list[int] | None) -> int | None:
    return value[0] | (value[1] << 8) if value and len(value) >= 2 else None


def _open_bus(path: str) -> serial.Serial:
    return serial.Serial(path, baudrate=1_000_000, timeout=0.03, write_timeout=1.0)


def _test_position(path: str, servo_id: int) -> str:
    with _open_bus(path) as port:
        initial = _u16(_read_register(port, servo_id, 56, 2))
        if initial is None:
            return "No position response."
        target = initial + 180 if initial <= 3915 else initial - 180
        mode = _read_register(port, servo_id, 33, 1)
        torque = _read_register(port, servo_id, 40, 1)
        original_mode = mode[0] if mode else 0
        original_torque = torque[0] if torque else 0
        try:
            _write_register(port, servo_id, 40, [0])
            _write_register(port, servo_id, 33, [0])
            _write_register(port, servo_id, 42, [initial & 0xFF, initial >> 8])
            _write_register(port, servo_id, 40, [1])
            _write_register(port, servo_id, 41, [6, target & 0xFF, target >> 8, 0xDC, 0x05, 130, 0])
            time.sleep(1.8)
            reached = _u16(_read_register(port, servo_id, 56, 2))
            _write_register(port, servo_id, 41, [6, initial & 0xFF, initial >> 8, 0xDC, 0x05, 130, 0])
            time.sleep(1.8)
            returned = _u16(_read_register(port, servo_id, 56, 2))
            return f"{initial} -> {target} -> {reached} -> {returned}"
        finally:
            _write_register(port, servo_id, 40, [0])
            _write_register(port, servo_id, 33, [original_mode])
            if original_mode == 0:
                _write_register(port, servo_id, 42, [initial & 0xFF, initial >> 8])
            _write_register(port, servo_id, 40, [original_torque])


def _test_wheel(path: str, servo_id: int) -> str:
    with _open_bus(path) as port:
        mode = _read_register(port, servo_id, 33, 1)
        torque = _read_register(port, servo_id, 40, 1)
        original_mode = mode[0] if mode else 0
        original_torque = torque[0] if torque else 0
        try:
            _write_register(port, servo_id, 40, [0])
            _write_register(port, servo_id, 33, [1])
            _write_register(port, servo_id, 40, [1])
            _write_register(port, servo_id, 46, [500, 0])
            time.sleep(3.0)
            running = _u16(_read_register(port, servo_id, 58, 2))
            _write_register(port, servo_id, 46, [0, 0])
            time.sleep(0.5)
            stopped = _u16(_read_register(port, servo_id, 58, 2))
            return f"Velocity {running}; stopped {stopped}."
        finally:
            _write_register(port, servo_id, 46, [0, 0])
            _write_register(port, servo_id, 40, [0])
            _write_register(port, servo_id, 33, [original_mode])
            _write_register(port, servo_id, 40, [original_torque])


def _capture(path: str):
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    ok, frame = capture.read() if capture.isOpened() else (False, None)
    capture.release()
    if not ok or frame is None:
        input_format = "gray16le" if path == "/dev/video4" else "yuyv422"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
                    "-input_format", input_format, "-video_size", "640x480", "-i", path,
                    "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
                ],
                capture_output=True,
                timeout=8,
                check=True,
            )
            frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
    if frame is None:
        return None
    if len(frame.shape) == 2:
        depth = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        return cv2.cvtColor(cv2.applyColorMap(depth, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _capture_agent_rgb():
    agent = st.session_state.get("agent")
    if agent is None:
        return None
    image = agent.main_camera.capture_image()
    frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None


def _capture_depth_with_rgb_paused(path: str):
    agent = st.session_state.get("agent")
    camera = getattr(agent, "main_camera", None)
    if camera is not None:
        camera.release()
        time.sleep(0.3)
    try:
        return _capture(path)
    finally:
        if camera is not None:
            camera.reopen()


def render_hardware_tab():
    st.subheader("Hardware Validation")
    st.caption("Camera previews are read-only. Servo tests run one device at a time and restore its prior state.")

    st.markdown("#### Cameras")
    columns = st.columns(4)
    for column, (name, path) in zip(columns, CAMERAS):
        with column:
            st.markdown(f"**{name}**")
            st.caption(path)
            if st.button("Refresh", key=f"camera_{name}", use_container_width=True):
                camera_rig = st.session_state.get("camera_rig")
                view = CAMERA_VIEWS.get(name)
                if camera_rig is not None and view is not None:
                    try:
                        frame = camera_rig.capture(view).jpeg_bytes
                    except Exception:
                        frame = None
                elif name.startswith("Center camera"):
                    frame = _capture_agent_rgb()
                elif name.startswith("Depth camera"):
                    frame = _capture_depth_with_rgb_paused(path)
                else:
                    frame = _capture(path)
                if frame is None:
                    st.error("No frame available")
                else:
                    st.session_state[f"camera_frame_{name}"] = frame
            frame = st.session_state.get(f"camera_frame_{name}")
            if frame is not None:
                st.image(frame, channels="RGB", use_container_width=True)

    st.markdown("#### Servo Tests")
    confirmed = st.checkbox("Robot is clear and it is safe to move one servo at a time", key="hardware_motion_confirm")
    for name, path, servo_id, kind in SERVOS:
        left, middle, right = st.columns([3, 2, 2])
        left.write(name)
        middle.caption(f"ID {servo_id} | {'wheel velocity' if kind == 'wheel' else 'position test'}")
        if right.button("Test", key=f"servo_{path}_{servo_id}", disabled=not confirmed, use_container_width=True):
            try:
                with st.spinner(f"Testing {name}..."):
                    result = _test_wheel(path, servo_id) if kind == "wheel" else _test_position(path, servo_id)
                st.success(result)
            except Exception as error:
                st.error(f"Test failed: {error}")
