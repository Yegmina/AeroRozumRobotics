import os
import re
import socket
import subprocess
from pathlib import Path

import streamlit as st
from robocrew.robots.XLeRobot.rgbd_client import RGBDServiceClient
from dotenv import load_dotenv

RULES_FILE = "/etc/udev/rules.d/99-robocrew.rules"

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

HARDWARE_PATHS = {
    "camera_center": os.environ.get(
        "ROBOCREW_CENTER_CAMERA_PORT",
        "/dev/v4l/by-path/platform-3610000.usb-usb-0:2:1.4-video-index0",
    ),
    "arm_right_wheels": os.environ.get("ROBOCREW_LEFT_ARM_WHEEL_PORT", "/dev/ttyACM0"),
    "arm_left_head": os.environ.get("ROBOCREW_RIGHT_ARM_HEAD_PORT", "/dev/ttyACM1"),
}


def _device_available(alias: str, path: str) -> bool:
    if alias == "camera_center":
        # The Orbbec SDK owns the USB interfaces directly. Its V4L nodes are
        # removed while streaming, so the service socket is the source of truth.
        if RGBDServiceClient(timeout=0.3).health():
            return True
    return os.path.exists(path)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def save_udev_rules(new_content):
    try:
        if os.access(os.path.dirname(RULES_FILE), os.W_OK):
            with open(RULES_FILE, "w") as f:
                f.write(new_content)
            subprocess.run(["udevadm", "control", "--reload-rules"], check=False)
            subprocess.run(["udevadm", "trigger"], check=False)
            return True, ""
        else:
            import tempfile
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(new_content)
                tmp_name = tmp.name
            res = subprocess.run(["sudo", "-n", "cp", tmp_name, RULES_FILE], capture_output=True, text=True)
            if res.returncode == 0:
                subprocess.run(["sudo", "-n", "udevadm", "control", "--reload-rules"])
                subprocess.run(["sudo", "-n", "udevadm", "trigger"])
                return True, ""
            return False, res.stderr
    except Exception as e:
        return False, str(e)

def get_hardware_status():
    aliases_in_rules = set()
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r") as f:
            content = f.read()
        aliases_in_rules.update(re.findall(r'SYMLINK\+="(.*?)"', content))
    
    status = {}
    err_msg = st.session_state.get("init_error", "")
    is_recording = st.session_state.recording_process is not None
    
    for alias, path in HARDWARE_PATHS.items():
        is_required = True
        if not _device_available(alias, path):
            status[alias] = {"state": "disconnected", "label": "Disconnected", "required": is_required}
        elif is_recording:
            status[alias] = {"state": "warning", "label": "Busy (Recording)", "required": is_required}
        elif err_msg and path in err_msg:
            status[alias] = {"state": "error", "label": "Power/Comm Error", "required": is_required}
        elif not st.session_state.agent:
            status[alias] = {"state": "warning", "label": "Standby", "required": is_required}
        else:
            status[alias] = {"state": "success", "label": "Ready", "required": is_required}

    for alias in sorted(aliases_in_rules):
        if alias in status:
            continue
        path = f"/dev/{alias}"
        status[alias] = {
            "state": "success" if os.path.exists(path) else "disconnected",
            "label": "Ready" if os.path.exists(path) else "Disconnected",
            "required": False,
        }
    return status
