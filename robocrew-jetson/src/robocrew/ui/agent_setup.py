from robocrew.robots.XLeRobot.tools import create_vla_single_arm_manipulation
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from robocrew.core.gemini_config import get_gemini_robotics_langchain_model
from robocrew.robots.XLeRobot.camera_rig import AuxiliaryCameraRig
from robocrew.robots.XLeRobot.manipulation import GraspController, ManipulationCalibration
from robocrew.robots.XLeRobot.rgbd_client import AlignedRGBDCamera, RGBDServiceManager
from robocrew.robots.XLeRobot.servo_controls import ServoControler, _check_calibration_file
from robocrew.robots.XLeRobot.xlerobot_LLM_agent import XLeRobotAgent
from robocrew.robots.XLeRobot.tools import (
    RobotSafetyState, \
    create_go_to_precision_mode, \
    create_go_to_normal_mode, \
    create_grasp_object, \
    create_inspect_cameras, \
    create_inspect_arm_workspace, \
    create_move_backward, \
    create_move_forward, \
    create_strafe_right, \
    create_strafe_left, \
    create_look_around, \
    create_move_depth_camera, \
    create_report_observation_and_plan, \
    create_scan_drive_path, \
    create_scan_close_workspace, \
    create_stop_wheels, \
    create_turn_right, \
    create_turn_left, \
    create_verified_finish_task
)

CALIBRATION_TTYD_PORT = 8283
VLA_FILE = os.path.join(os.path.expanduser("~"), ".cache", "robocrew", "tools", "vla_tools.json")

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

# Historical environment names are retained for compatibility. Physically,
# the right arm shares the wheel bus and the left arm shares the head bus.
LEFT_ARM_WHEEL_PORT = os.environ.get("ROBOCREW_LEFT_ARM_WHEEL_PORT", "/dev/ttyACM0")
RIGHT_ARM_HEAD_PORT = os.environ.get("ROBOCREW_RIGHT_ARM_HEAD_PORT", "/dev/ttyACM1")
CENTER_CAMERA_PORT = os.environ.get(
    "ROBOCREW_CENTER_CAMERA_PORT",
    "/dev/v4l/by-path/platform-3610000.usb-usb-0:2:1.4-video-index0",
)
LEFT_CAMERA_PORT = os.environ.get(
    "ROBOCREW_LEFT_CAMERA_PORT",
    "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0",
)
RIGHT_CAMERA_PORT = os.environ.get(
    "ROBOCREW_RIGHT_CAMERA_PORT",
    "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0",
)
DEPTH_CAMERA_PORT = os.environ.get("ROBOCREW_ORBBEC_DEPTH_PORT", "/dev/video4")

def _is_process_running(process) -> bool:
    return process is not None and process.poll() is None


def _start_calibration_terminal(missing_files: list[str]) -> None:
    if not missing_files:
        return

    file_to_port = {
        # ServoControler currently calls these historical calibration files
        # from its wheel and head bus constructors respectively.
        "left_arm.json": RIGHT_ARM_HEAD_PORT,
        "right_arm.json": LEFT_ARM_WHEEL_PORT,
    }
    pyexe = shlex.quote(sys.executable)
    steps: list[str] = []
    for file_name in missing_files:
        calibration_id = os.path.splitext(file_name)[0]
        arm_port = file_to_port[file_name]
        code = (
            "from robocrew.robots.XLeRobot.servo_controls import _run_lerobot_calibrate, _check_calibration_file;"
            f"print('Starting calibration: {file_name} on {arm_port}');"
            f"_run_lerobot_calibrate('{arm_port}', '{calibration_id}', _check_calibration_file('{file_name}'));"
            f"print('Calibration finished: {file_name}')"
        )
        steps.append(f"{pyexe} -c {shlex.quote(code)}")

    bash_cmd = " ; ".join(steps) + " ; sleep 2 ; kill -9 $PPID"
    ttyd_cmd = ["ttyd", "-W", "-p", str(CALIBRATION_TTYD_PORT), "bash", "-c", bash_cmd]
    st.session_state.calibration_process = subprocess.Popen(ttyd_cmd, env=os.environ.copy())


def _get_missing_calibration_files() -> list[str]:
    return [
        name
        for name in ("left_arm.json", "right_arm.json")
        if not _check_calibration_file(name).exists()
    ]

def get_hardware():
    rgbd_manager = RGBDServiceManager(
        socket_path=os.environ.get("ROBOCREW_RGBD_SOCKET", "/tmp/robocrew-orbbec.sock"),
        python_executable=os.environ.get(
            "ROBOCREW_ORBBEC_PYTHON",
            "/home/jetsonl4/aerorozumdatacollectiondepth/.venv/bin/python",
        ),
    )
    main_camera = AlignedRGBDCamera(rgbd_manager.ensure_running())
    try:
        servo_controller = ServoControler(
            right_arm_wheel_usb=LEFT_ARM_WHEEL_PORT,
            left_arm_head_usb=RIGHT_ARM_HEAD_PORT,
        )
    except Exception:
        main_camera.release()
        raise
    camera_rig = AuxiliaryCameraRig(
        main_camera,
        left_port=LEFT_CAMERA_PORT,
        right_port=RIGHT_CAMERA_PORT,
        depth_port=DEPTH_CAMERA_PORT,
    )
    return main_camera, servo_controller, camera_rig, rgbd_manager

def init_agent():
    if st.session_state.recording_process:
        st.session_state.init_error = "Hardware busy: Recording in progress."
        return

    missing_files = _get_missing_calibration_files()
    main_camera = None
    servo_controller = None
    st.session_state.init_error = ""
        
    with st.spinner("Initializing Robot Agent..."):
        try:
            main_camera, servo_controller, camera_rig, rgbd_manager = get_hardware()
            calibration = ManipulationCalibration.load(servo_controller)
            grasp_controller = GraspController(
                servo_controller,
                main_camera,
                camera_rig,
                calibration=calibration,
            )
            
            vla_tools = []
            if not missing_files and os.path.exists(VLA_FILE):
                with open(VLA_FILE, "r") as f:
                    for t in json.load(f):

                        if not t.get("active", True): 
                            continue
                            
                        cam_cfg = {"main": {"index_or_path": "/dev/camera_center"}, "right_arm": {"index_or_path": "/dev/camera_right"}}
                        
                        vla_tools.append(create_vla_single_arm_manipulation(
                            tool_name=t["tool_name"], tool_description=t["tool_description"],
                            task_prompt=t["task_prompt"], server_address=t["server_address"],
                            policy_name=t["policy_name"], policy_type=t["policy_type"],
                            arm_port=t["arm_port"], servo_controler=servo_controller,
                            camera_config=cam_cfg, main_camera_object=main_camera,
                            policy_device=t["policy_device"], execution_time=t["execution_time"],
                            load_on_startup=False
                        ))

            safety_state = RobotSafetyState()
            tools = [
                create_move_forward(servo_controller, safety_state=safety_state),
                create_move_backward(servo_controller, safety_state=safety_state),
                create_turn_left(servo_controller),
                create_turn_right(servo_controller),
                create_strafe_left(servo_controller, safety_state=safety_state),
                create_strafe_right(servo_controller, safety_state=safety_state),
                create_go_to_precision_mode(servo_controller),
                create_go_to_normal_mode(servo_controller),
                create_grasp_object(grasp_controller),
                create_inspect_cameras(camera_rig),
                create_scan_drive_path(servo_controller, main_camera, camera_rig, safety_state),
                create_inspect_arm_workspace(camera_rig, safety_state),
                create_scan_close_workspace(servo_controller, main_camera, camera_rig, safety_state),
                create_look_around(servo_controller, main_camera),
                create_move_depth_camera(servo_controller),
                create_report_observation_and_plan(),
                create_stop_wheels(servo_controller),
                create_verified_finish_task(safety_state),
            ] + vla_tools

            st.session_state.camera_rig = camera_rig
            st.session_state.rgbd_manager = rgbd_manager
            st.session_state.grasp_controller = grasp_controller
            st.session_state.agent = XLeRobotAgent(
                model=get_gemini_robotics_langchain_model(),
                tools=tools,
                main_camera=main_camera,
                servo_controler=servo_controller,
                lidar_usb_port="/dev/lidar" if os.path.exists("/dev/lidar") else None,
                history_len=8
            )
            st.session_state.init_error = (
                "Arm calibration files are missing. VLA manipulation is disabled until calibration is completed."
                if missing_files
                else ""
            )
        except Exception as e:
            if main_camera is not None:
                main_camera.release()
            if servo_controller is not None:
                try:
                    servo_controller.disconnect()
                except Exception:
                    pass
            st.session_state.camera_rig = None
            st.session_state.grasp_controller = None
            st.session_state.rgbd_manager = None
            st.session_state.agent = None
            st.session_state.init_error = str(e)
            st.error(f"Init failed: {e}")
