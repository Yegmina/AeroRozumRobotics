from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
import time

from robocrew.robots.XLeRobot.manipulation import ManipulationCalibration
from robocrew.robots.XLeRobot.servo_controls import ARM_SERVO_MAPS


JOINT_DIRECTIONS = (
    ("shoulder_pan", "right"),
    ("shoulder_lift", "up"),
    ("elbow_flex", "bend inward"),
    ("wrist_flex", "tilt the gripper up"),
    ("wrist_roll", "rotate clockwise when viewed from the gripper camera"),
)
GUIDED_STEPS = tuple(
    {"arm": arm, "kind": "joint", "joint": joint, "expected": expected}
    for arm in ("right", "left")
    for joint, expected in JOINT_DIRECTIONS
) + tuple(
    {"arm": arm, "kind": kind, "joint": "gripper", "expected": expected}
    for arm in ("right", "left")
    for kind, expected in (("gripper_open", "open"), ("gripper_close", "close"))
)


def _charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard((5, 7), 0.030, 0.022, dictionary)


def _board_png() -> bytes:
    board = _charuco_board().generateImage((600, 840), marginSize=24)
    ok, encoded = cv2.imencode(".png", board)
    return encoded.tobytes() if ok else b""


def _step_diagram(step: dict) -> bytes:
    canvas = np.full((400, 760, 3), 248, dtype=np.uint8)
    dark = (45, 45, 45)
    muted = (150, 150, 150)
    active = (40, 55, 220)
    expected = (45, 170, 55)
    cv2.putText(
        canvas,
        f"{step['arm'].upper()} ARM - {step['joint'].replace('_', ' ').upper()}",
        (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.82, dark, 2,
    )
    cv2.putText(
        canvas,
        f"Expected: {step['expected'].upper()}",
        (28, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, expected, 2,
    )

    if step["joint"] == "shoulder_pan":
        cv2.putText(canvas, "TOP VIEW", (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, muted, 1)
        shoulder = (320, 245)
        cv2.rectangle(canvas, (245, 205), (395, 285), muted, 3)
        cv2.line(canvas, shoulder, (560, 245), dark, 18)
        cv2.circle(canvas, shoulder, 24, active, -1)
        cv2.arrowedLine(canvas, (430, 330), (610, 330), expected, 10, tipLength=0.18)
        cv2.putText(canvas, "RIGHT", (615, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, expected, 2)
    elif step["joint"] in ("gripper",):
        cv2.putText(canvas, "GRIPPER VIEW", (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, muted, 1)
        cv2.rectangle(canvas, (285, 180), (475, 230), muted, 3)
        cv2.line(canvas, (330, 230), (295, 330), active, 15)
        cv2.line(canvas, (430, 230), (465, 330), active, 15)
        if step["kind"] == "gripper_open":
            cv2.arrowedLine(canvas, (320, 285), (245, 285), expected, 9, tipLength=0.2)
            cv2.arrowedLine(canvas, (440, 285), (515, 285), expected, 9, tipLength=0.2)
        else:
            cv2.arrowedLine(canvas, (245, 285), (320, 285), expected, 9, tipLength=0.2)
            cv2.arrowedLine(canvas, (515, 285), (440, 285), expected, 9, tipLength=0.2)
    else:
        cv2.putText(canvas, "SIDE VIEW", (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, muted, 1)
        shoulder, elbow, wrist, tip = (190, 285), (355, 190), (520, 245), (665, 225)
        cv2.rectangle(canvas, (85, 270), (175, 340), muted, 3)
        cv2.line(canvas, shoulder, elbow, dark, 18)
        cv2.line(canvas, elbow, wrist, dark, 18)
        cv2.line(canvas, wrist, tip, dark, 14)
        cv2.circle(canvas, shoulder, 19, muted, -1)
        cv2.circle(canvas, elbow, 19, muted, -1)
        cv2.circle(canvas, wrist, 19, muted, -1)

        if step["joint"] == "shoulder_lift":
            cv2.circle(canvas, shoulder, 24, active, -1)
            cv2.arrowedLine(canvas, (250, 270), (250, 150), expected, 10, tipLength=0.2)
            cv2.putText(canvas, "UP", (265, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, expected, 2)
        elif step["joint"] == "elbow_flex":
            cv2.circle(canvas, elbow, 24, active, -1)
            cv2.ellipse(canvas, elbow, (85, 70), 0, 205, 345, expected, 9)
            cv2.arrowedLine(canvas, (425, 225), (395, 250), expected, 9, tipLength=0.4)
        elif step["joint"] == "wrist_flex":
            cv2.circle(canvas, wrist, 24, active, -1)
            cv2.arrowedLine(canvas, (610, 255), (625, 145), expected, 10, tipLength=0.2)
            cv2.putText(canvas, "GRIPPER UP", (540, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, expected, 2)
        elif step["joint"] == "wrist_roll":
            cv2.circle(canvas, wrist, 24, active, -1)
            cv2.ellipse(canvas, (610, 225), (58, 58), 0, 25, 315, expected, 9)
            cv2.arrowedLine(canvas, (655, 185), (670, 215), expected, 9, tipLength=0.35)
            cv2.putText(canvas, "CLOCKWISE", (555, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, expected, 2)

    cv2.putText(canvas, "RED = JOINT   GREEN = EXPECTED MOTION", (28, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dark, 2)
    ok, encoded = cv2.imencode(".png", canvas)
    return encoded.tobytes() if ok else b""


def _render_status(controller) -> None:
    status = controller.status
    columns = st.columns(4)
    columns[0].metric("Stage", status.get("stage", "idle"))
    columns[1].metric("Arm", status.get("arm") or "-")
    columns[2].metric("Action", status.get("action") or "-")
    target = status.get("target") or {}
    columns[3].metric("Candidate", f"#{target.get('id')}" if target else "-")
    st.caption(status.get("message", ""))
    if status.get("candidate_image"):
        st.image(status["candidate_image"], width="stretch")


def _arm_bus(controller, arm: str):
    return controller.servo_controller.head_bus if arm == "left" else controller.servo_controller.wheel_bus


def _restore_wizard_step(controller, state: dict) -> None:
    if state.get("start_raw") is None:
        return
    step = GUIDED_STEPS[state["index"]]
    servo_id = ARM_SERVO_MAPS[step["arm"]][step["joint"]]
    _arm_bus(controller, step["arm"]).sync_write(
        "Goal_Position", {servo_id: int(state["start_raw"])}, normalize=False
    )
    time.sleep(0.55)


def _move_wizard_step(controller, calibration, state: dict) -> None:
    step = GUIDED_STEPS[state["index"]]
    arm = step["arm"]
    joint = step["joint"]
    bus = _arm_bus(controller, arm)
    servo_id = ARM_SERVO_MAPS[arm][joint]
    state["start_raw"] = int(bus.read("Present_Position", servo_id, normalize=False))
    arm_config = calibration.data["arms"][arm]
    if step["kind"] == "joint":
        sign = int(arm_config["raw_sign_per_positive_degree"][joint])
        controller.servo_controller.move_arm_joint_relative_raw(arm, joint, sign * 240)
    else:
        key = "gripper_open_raw" if step["kind"] == "gripper_open" else "gripper_closed_raw"
        bus.sync_write("Goal_Position", {servo_id: int(arm_config[key])}, normalize=False)
        time.sleep(1.0)


def _finish_guided_calibration(controller, calibration) -> None:
    for arm in ("right", "left"):
        calibration.data["arms"][arm]["directions_verified"] = True
    frame = controller.main_camera.capture_rgbd_frame()
    valid = (frame.depth_mm > 0) & (frame.depth_mm < 65535)
    valid_percent = float(np.count_nonzero(valid)) * 100.0 / valid.size
    calibration.data["board_validated"] = valid_percent >= 70.0
    calibration.data["ready"] = calibration.data["board_validated"]
    calibration.save()
    controller.calibration = ManipulationCalibration.load(controller.servo_controller)


def _render_guided_calibration(controller, calibration) -> None:
    state = st.session_state.get("grasp_calibration_wizard")
    if state is None:
        st.markdown("#### Guided calibration")
        if st.button("Start guided calibration", type="primary", use_container_width=True):
            calibration.data["ready"] = False
            calibration.save()
            state = {"index": 0, "start_raw": None, "corrected": False}
            st.session_state.grasp_calibration_wizard = state
            _move_wizard_step(controller, calibration, state)
            st.rerun()
        return

    index = int(state["index"])
    step = GUIDED_STEPS[index]
    st.progress((index + 1) / len(GUIDED_STEPS), text=f"Step {index + 1} of {len(GUIDED_STEPS)}")
    st.image(_step_diagram(step), width="stretch")
    st.subheader(
        f"Did the {step['arm'].upper()} {step['joint'].replace('_', ' ').upper()} move {step['expected'].upper()}?"
    )
    if state.get("corrected"):
        st.caption("Direction was reversed automatically. Confirm this corrected movement.")

    yes, no, repeat, stop = st.columns(4)
    if yes.button("Yes", type="primary", use_container_width=True):
        _restore_wizard_step(controller, state)
        state["index"] += 1
        state["corrected"] = False
        state["start_raw"] = None
        if state["index"] >= len(GUIDED_STEPS):
            _finish_guided_calibration(controller, calibration)
            st.session_state.grasp_calibration_wizard = None
        else:
            _move_wizard_step(controller, calibration, state)
        st.rerun()
    if no.button("No, reverse it", use_container_width=True):
        _restore_wizard_step(controller, state)
        arm_config = calibration.data["arms"][step["arm"]]
        if step["kind"] == "joint":
            joint = step["joint"]
            arm_config["raw_sign_per_positive_degree"][joint] *= -1
        else:
            arm_config["gripper_open_raw"], arm_config["gripper_closed_raw"] = (
                arm_config["gripper_closed_raw"], arm_config["gripper_open_raw"]
            )
        calibration.save()
        state["corrected"] = True
        _move_wizard_step(controller, calibration, state)
        st.rerun()
    if repeat.button("Repeat", use_container_width=True):
        _restore_wizard_step(controller, state)
        _move_wizard_step(controller, calibration, state)
        st.rerun()
    if stop.button("Stop", use_container_width=True):
        _restore_wizard_step(controller, state)
        st.session_state.grasp_calibration_wizard = None
        st.rerun()


def render_manipulation_tab():
    controller = st.session_state.get("grasp_controller")
    if controller is None:
        return st.info("Initialize hardware to calibrate and monitor grasping.")

    st.subheader("Manipulation")
    _render_status(controller)
    calibration = controller.calibration
    errors = calibration.validation_errors()
    if errors:
        st.warning("Autonomous grasping blocked: " + "; ".join(errors))
    else:
        st.success("Manipulation calibration ready")

    _render_guided_calibration(controller, calibration)

    with st.expander("Advanced calibration"):
        with st.form("manipulation_calibration"):
            transform = calibration.data["head_to_base"]
            xyz = st.columns(3)
            transform[0][3] = xyz[0].number_input(
                "Camera X (m)", value=float(transform[0][3]), step=0.001, format="%.3f"
            )
            transform[1][3] = xyz[1].number_input(
                "Camera Y (m)", value=float(transform[1][3]), step=0.001, format="%.3f"
            )
            transform[2][3] = xyz[2].number_input(
                "Camera Z (m)", value=float(transform[2][3]), step=0.001, format="%.3f"
            )

            for side in ("right", "left"):
                st.markdown(f"**{side.title()} arm**")
                arm = calibration.data["arms"][side]
                cols = st.columns(4)
                arm["gripper_open_raw"] = cols[0].number_input(
                    "Open raw", value=int(arm["gripper_open_raw"]), key=f"{side}_open"
                )
                arm["gripper_closed_raw"] = cols[1].number_input(
                    "Closed raw", value=int(arm["gripper_closed_raw"]), key=f"{side}_closed"
                )
                arm["gripper_load_threshold"] = cols[2].number_input(
                    "Load limit", min_value=20, max_value=1000,
                    value=int(arm["gripper_load_threshold"]), key=f"{side}_load"
                )
                arm["directions_verified"] = cols[3].checkbox(
                    "Directions verified", value=bool(arm["directions_verified"]), key=f"{side}_verified"
                )
                sign_cols = st.columns(5)
                for column, joint in zip(
                    sign_cols,
                    ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
                ):
                    current = int(arm["raw_sign_per_positive_degree"][joint])
                    arm["raw_sign_per_positive_degree"][joint] = column.selectbox(
                        joint.replace("_", " "), (-1, 1), index=0 if current == -1 else 1,
                        key=f"{side}_{joint}_sign",
                    )

            approve = st.checkbox(
                "Approve calibration for autonomous grasping", value=bool(calibration.data["ready"])
            )
            submitted = st.form_submit_button("Save manipulation calibration", use_container_width=True)
            if submitted:
                calibration.data["ready"] = approve
                calibration.save()
                controller.calibration = ManipulationCalibration.load(controller.servo_controller)
                st.rerun()
