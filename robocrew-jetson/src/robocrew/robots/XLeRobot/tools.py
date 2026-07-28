import base64

from pathlib import Path

import math
import cv2
import numpy as np
from typing import Literal
from langchain_core.tools import tool  # type: ignore[import]
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from robocrew.robots.XLeRobot.groot_client import PolicyClient

from robocrew.core.utils import stop_listening_during_tool_execution
from robocrew.robots.XLeRobot.servo_controls import DEFAULT_ARM_CALIBRATION_DIR
import time
import threading


class RobotSafetyState:
    """Tracks camera checks that must precede physical movement."""

    def __init__(self):
        self.drive_path_checked = False
        self.arm_camera_checked = {"left": False, "right": False}
        self.arm_motion_unverified = {"left": False, "right": False}

    def mark_drive_path_checked(self) -> None:
        self.drive_path_checked = True

    def consume_drive_path_check(self) -> bool:
        checked = self.drive_path_checked
        self.drive_path_checked = False
        return checked

    def mark_arm_camera_checked(self, arm: str) -> None:
        self.arm_camera_checked[arm] = True
        self.arm_motion_unverified[arm] = False

    def consume_arm_camera_check(self, arm: str) -> bool:
        checked = self.arm_camera_checked[arm]
        self.arm_camera_checked[arm] = False
        return checked

    def mark_arm_moved(self, arm: str) -> None:
        self.arm_motion_unverified[arm] = True

    def unverified_arms(self) -> list[str]:
        return [arm for arm, unverified in self.arm_motion_unverified.items() if unverified]


def create_move_forward(servo_controller, sound_receiver=None, safety_state=None):
    @tool
    @stop_listening_during_tool_execution(sound_receiver)
    def move_forward(distance_meters: float) -> str:
        """Drives the robot forward (or backward) for a specific distance."""

        if safety_state and not safety_state.consume_drive_path_check():
            return "BLOCKED: run scan_drive_path before translating the robot."
        distance = float(distance_meters)
        servo_controller.reset_head_position()
        if distance >= 0:
            servo_controller.go_forward(distance)
        else:
            servo_controller.go_backward(-distance)
        return f"Moved {'forward' if distance >= 0 else 'backward'} {abs(distance):.2f} meters."

    return move_forward

def create_move_backward(servo_controller, sound_receiver=None, safety_state=None):
    @tool
    @stop_listening_during_tool_execution(sound_receiver)
    def move_backward(distance_meters: float) -> str:
        """Drives the robot forward (or backward) for a specific distance."""

        if safety_state and not safety_state.consume_drive_path_check():
            return "BLOCKED: run scan_drive_path before translating the robot."
        distance = float(distance_meters)
        servo_controller.reset_head_position()
        servo_controller.go_backward(distance)
        return f"Moved backward {distance} meters."

    return move_backward

def create_turn_right(servo_controller, sound_receiver=None):
    @tool
    @stop_listening_during_tool_execution(sound_receiver)
    def turn_right(angle_degrees: float) -> str:
        """Turns the robot right by angle in degrees. Use only when robot body not touches any obstacle."""
        angle = float(angle_degrees)
        servo_controller.turn_right(angle)
        time.sleep(0.4)  # wait a bit after turn for stabilization
        return f"Turned right by {angle} degrees."

    return turn_right

def create_turn_left(servo_controller, sound_receiver=None):
    @tool
    @stop_listening_during_tool_execution(sound_receiver)
    def turn_left(angle_degrees: float) -> str:
        """Turns the robot left by angle in degrees. Use only when robot body not touches any obstacle."""
        angle = float(angle_degrees)
        servo_controller.turn_left(angle)
        time.sleep(0.4)  # wait a bit after turn for stabilization
        return f"Turned left by {angle} degrees."

    return turn_left


def create_strafe_left(servo_controller, sound_receiver=None, safety_state=None):
    @tool
    @stop_listening_during_tool_execution(sound_receiver)
    def strafe_left(distance_meters: float) -> str:
        """Moves the robot sideways left by a specific distance in meters."""
        if safety_state and not safety_state.consume_drive_path_check():
            return "BLOCKED: run scan_drive_path before translating the robot."
        distance = float(distance_meters)
        servo_controller.reset_head_position()
        servo_controller.strafe_left(distance)
        return f"Strafed left by {distance} meters."

    return strafe_left

def create_strafe_right(servo_controller, sound_receiver=None, safety_state=None):
    @tool
    @stop_listening_during_tool_execution(sound_receiver)
    def strafe_right(distance_meters: float) -> str:
        """Moves the robot sideways right by a specific distance in meters."""
        if safety_state and not safety_state.consume_drive_path_check():
            return "BLOCKED: run scan_drive_path before translating the robot."
        distance = float(distance_meters)
        servo_controller.reset_head_position()
        servo_controller.strafe_right(distance)
        return f"Strafed right by {distance} meters."

    return strafe_right

def create_go_to_precision_mode(servo_controller):
    @tool
    def go_to_precision_mode() -> str:
        """Sets the robot to precision movement mode. Use it when close to obstacles or target."""
        servo_controller.turn_head_to_vla_position(50)
        return "Robot set to precision movement mode."

    return go_to_precision_mode

def create_go_to_normal_mode(servo_controller):
    @tool
    def go_to_normal_mode() -> str:
        """Sets the robot to normal movement mode for long distance rides."""
        servo_controller.reset_head_position()
        return "Robot set to normal movement mode."

    return go_to_normal_mode


def create_wave_right_hand(servo_controller):
    @tool
    def wave_right_hand() -> str:
        """Make a small right-hand wrist wave when the area around the arm is clear."""
        if not hasattr(servo_controller, "wheel_bus"):
            raise RuntimeError("Right arm bus is not available.")

        # The physical right arm shares the wheel bus.
        # Servo 5 is wrist roll. Use its current raw position as the reference
        # because no calibrated arm pose is available yet.
        bus = servo_controller.wheel_bus
        servo_id = 5
        initial = int(bus.read("Present_Position", servo_id, normalize=False))
        delta = min(160, initial - 80, 4015 - initial)
        if delta < 40:
            raise RuntimeError("Right wrist is too close to a position limit to wave safely.")

        try:
            for target in (initial + delta, initial - delta, initial + delta, initial):
                bus.write("Goal_Position", servo_id, target, normalize=False)
                time.sleep(0.55)
        finally:
            bus.write("Goal_Position", servo_id, initial, normalize=False)
        return "Performed a small right-hand wrist wave and returned to the starting position."

    return wave_right_hand


def create_move_arm_joint(servo_controller, safety_state=None):
    @tool
    def move_arm_joint(
        arm: Literal["left", "right"],
        joint: Literal["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        relative_steps: int,
    ) -> str:
        """Move one named arm joint relative to its current position.

        Use only when the arm is clear. `relative_steps` is limited to -600..600
        and the target is clamped to that joint's calibrated physical range.
        For both arms, positive shoulder_pan moves right and positive
        shoulder_lift moves up; negative values move left and down respectively.
        """
        if safety_state and not safety_state.consume_arm_camera_check(arm):
            return f"BLOCKED: run inspect_arm_workspace with arm='{arm}' before moving that arm."
        step = max(-600, min(600, int(relative_steps)))
        if step == 0:
            return "No arm motion requested."

        # Both physical shoulder-lift servos raise the hand when their raw
        # positions decrease.
        physical_step = -step if joint == "shoulder_lift" else step
        initial, target = servo_controller.move_arm_joint_relative_raw(arm, joint, physical_step)
        if safety_state:
            safety_state.mark_arm_moved(arm)
        return f"Moved {arm} {joint} from {initial} to {target} (semantic step {step})."

    return move_arm_joint


def create_move_depth_camera(servo_controller):
    @tool
    def move_depth_camera(axis: Literal["pan", "tilt"], degrees: float) -> str:
        """Move camera pan or tilt relative to the calibrated forward pose."""
        if axis == "pan":
            servo_controller.point_head_relative(yaw_degrees=float(degrees))
        else:
            servo_controller.point_head_relative(pitch_degrees=float(degrees))
        return f"Moved depth camera {axis} to {degrees} degrees from calibrated forward."

    return move_depth_camera


def create_stop_wheels(servo_controller):
    @tool
    def stop_wheels() -> str:
        """Immediately stop all three drive wheels."""
        servo_controller._wheels_stop()
        return "All drive wheels stopped."

    return stop_wheels


def create_report_observation_and_plan():
    @tool
    def report_observation_and_plan(observation: str, next_action: str) -> str:
        """Publish a short, user-visible observation and next action.

        Use this before a movement or arm action. State only what is directly
        visible from the cameras/sensors and the immediate next action. Do not
        claim that an object or person was found unless it is visible.
        """
        return (
            f"Observation: {observation}\nNext action: {next_action}\n"
            "Status published. Execute that action now; do not publish another "
            "status update until an action or camera inspection has run."
        )

    return report_observation_and_plan


def create_inspect_cameras(camera_rig):
    @tool
    def inspect_cameras(
        views: list[Literal["left", "right", "depth", "all"]],
    ) -> tuple[str, list[dict]]:
        """Capture extra robot camera views for visual reasoning.

        Request `left` or `right` to inspect an arm/workspace, `depth` for
        obstacle distance, or `all` for all three auxiliary views. Center RGB
        is already supplied automatically and does not need to be requested.
        """
        requested: list[str] = []
        for view in views:
            expanded = ("left", "right", "depth") if view == "all" else (view,)
            for item in expanded:
                if item not in requested:
                    requested.append(item)

        if not requested:
            return "No auxiliary camera views were requested.", [
                {"type": "text", "text": "No auxiliary camera views were requested."}
            ]

        content: list[dict] = []
        captured: list[str] = []
        unavailable: list[str] = []
        for view in requested:
            try:
                observation = camera_rig.capture(view)
                label = observation.label
                if observation.details:
                    label = f"{label}\n{observation.details}"
                content.extend([
                    {"type": "text", "text": label},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(observation.jpeg_bytes).decode('ascii')}"
                        },
                    },
                ])
                captured.append(view)
            except Exception as error:
                content.append({"type": "text", "text": f"{view.title()} camera unavailable: {error}"})
                unavailable.append(view)

        summary = f"Captured auxiliary camera views: {', '.join(captured) or 'none'}."
        if unavailable:
            summary += f" Unavailable: {', '.join(unavailable)}."
        return summary, content

    return inspect_cameras


def create_scan_drive_path(servo_controller, main_camera, camera_rig, safety_state):
    @tool
    def scan_drive_path() -> tuple[str, list[dict]]:
        """Look down and inspect RGB plus depth immediately in front of the base.

        Run this before each forward, backward, or sideways translation so low
        obstacles near the wheels are checked. The camera returns to calibrated
        forward afterward.
        """
        try:
            servo_controller.point_head_relative(yaw_degrees=0, pitch_degrees=38)
            rgb_bytes = main_camera.capture_image(camera_fov=90, center_angle=0, navigation_mode="precision")
            depth = camera_rig.capture("depth")
        finally:
            servo_controller.reset_head_position()

        safety_state.mark_drive_path_checked()
        content = [
            {"type": "text", "text": "Downward drive-path RGB view"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(rgb_bytes).decode('ascii')}"},
            },
            {"type": "text", "text": f"Downward drive-path depth\n{depth.details}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(depth.jpeg_bytes).decode('ascii')}"},
            },
        ]
        return "Checked the low drive path with downward RGB and depth.", content

    return scan_drive_path


def create_inspect_arm_workspace(camera_rig, safety_state):
    @tool
    def inspect_arm_workspace(arm: Literal["left", "right"]) -> tuple[str, list[dict]]:
        """Inspect one arm using that arm's own camera plus center depth.

        Run immediately before every movement of that arm and immediately after
        its final movement to visually validate the target and result.
        """
        arm_view = camera_rig.capture(arm)
        depth = camera_rig.capture("depth")
        safety_state.mark_arm_camera_checked(arm)
        content = [
            {"type": "text", "text": f"{arm.title()} arm camera workspace"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(arm_view.jpeg_bytes).decode('ascii')}"},
            },
            {"type": "text", "text": f"Center depth for {arm} arm validation\n{depth.details}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(depth.jpeg_bytes).decode('ascii')}"},
            },
        ]
        return f"Inspected the {arm} arm workspace with its own camera and depth.", content

    return inspect_arm_workspace


def create_verified_finish_task(safety_state):
    @tool("finish_task")
    def finish_task(report: str = "Task finished") -> str:
        """Finish only after camera validation of any arm movement."""
        unverified = safety_state.unverified_arms()
        if unverified:
            return (
                "BLOCKED: inspect_arm_workspace must validate the latest movement "
                f"for arm(s): {', '.join(unverified)} before finishing."
            )
        return report

    return finish_task


def create_grasp_object(grasp_controller):
    @tool
    def grasp_object(
        target_description: str,
        arm: Literal["auto", "left", "right"] = "auto",
        action: Literal["touch", "grasp", "grasp_and_lift"] = "grasp_and_lift",
        candidate_id: int | None = None,
    ) -> tuple[str, list[dict]]:
        """Find and manipulate an upright can-like tabletop object.

        This is the only conversation tool for arm manipulation. It performs
        aligned RGB-D target measurement, bounded base approach, Cartesian IK,
        wrist-camera validation, load-monitored gripper closure, and optional
        lift verification. Use candidate_id only after the operator resolves an
        ambiguous numbered-candidate image.
        """
        return grasp_controller.execute(target_description, arm, action, candidate_id)

    return grasp_object


def create_scan_close_workspace(servo_controller, main_camera, camera_rig, safety_state=None):
    @tool
    def scan_close_workspace() -> tuple[str, list[dict]]:
        """Sweep the close workspace with paired center RGB and depth views.

        Use this to locate a tabletop object before reaching, and again after
        reaching to verify the gripper's position. It scans down-center,
        down-left, and down-right relative to the calibrated forward pose.
        """
        poses = (
            (0.0, 28.0, "Down-center workspace"),
            (-35.0, 28.0, "Down-left workspace"),
            (35.0, 28.0, "Down-right workspace"),
        )
        content: list[dict] = []
        completed: list[str] = []
        try:
            for yaw, pitch, label in poses:
                servo_controller.point_head_relative(yaw_degrees=yaw, pitch_degrees=pitch)
                rgb_bytes = main_camera.capture_image(camera_fov=90, center_angle=yaw, navigation_mode="precision")
                depth = camera_rig.capture("depth")
                content.extend([
                    {"type": "text", "text": f"{label} RGB (pan {yaw:+.0f}°, tilt {pitch:+.0f}°)"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(rgb_bytes).decode('ascii')}"},
                    },
                    {"type": "text", "text": f"{label} depth\n{depth.details}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(depth.jpeg_bytes).decode('ascii')}"},
                    },
                ])
                completed.append(label)
        finally:
            servo_controller.reset_head_position()

        if safety_state:
            safety_state.mark_drive_path_checked()
        summary = "Scanned close workspace with RGB and depth: " + ", ".join(completed) + "."
        return summary, content

    return scan_close_workspace


def create_look_around(servo_controller, main_camera):
    @tool
    def look_around() -> list:
        """Look around yourself to find a thing you looking for or to understand an envinronment."""
        movement_delay = 0.9  # seconds
        print("Looking around...")
        servo_controller.point_head_relative(yaw_degrees=-120, pitch_degrees=0)
        time.sleep(movement_delay)
        image_1 = main_camera.capture_image(center_angle=-120)
        image_1_64 = base64.b64encode(image_1).decode('utf-8')
        servo_controller.point_head_relative(yaw_degrees=-40, pitch_degrees=0)
        time.sleep(movement_delay)
        image_2 = main_camera.capture_image(center_angle=-40)
        image_2_64 = base64.b64encode(image_2).decode('utf-8')  
        servo_controller.point_head_relative(yaw_degrees=40, pitch_degrees=0)
        time.sleep(movement_delay)
        image_3 = main_camera.capture_image(center_angle=40)
        image_3_64 = base64.b64encode(image_3).decode('utf-8')
        servo_controller.point_head_relative(yaw_degrees=120, pitch_degrees=0)
        time.sleep(movement_delay)
        image_4 = main_camera.capture_image(center_angle=120)
        image_4_64 = base64.b64encode(image_4).decode('utf-8')
        servo_controller.reset_head_position()
        time.sleep(movement_delay)

        return "Looked around", [
            {"type": "text", "text": "Left"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_1_64}",}},
            {"type": "text", "text": "Left-Center"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_2_64}"}},
            {"type": "text", "text": "Right-Center"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_3_64}"}},
            {"type": "text", "text": "Right"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_4_64}"}},         
        ]
    return look_around


def create_vla_single_arm_manipulation(
        tool_name: str,
        tool_description: str,
        task_prompt: str,
        server_address: str,
        policy_name: str, 
        policy_type: str, 
        arm_port: str,
        servo_controler, 
        camera_config: dict[str, dict], 
        main_camera_object,
        execution_time: int = 30,
        policy_device: str = "cuda",
        fps: int = 30,
        actions_per_chunk: int = 50,
        load_on_startup: bool = True,
    ):
    """Creates a tool that makes the robot pick up a cup using its arm.
    Args:
        tool_name (str): The name of the tool AI agent will see.
        tool_description (str): The description of the tool AI agent will see.
        task_prompt (str): The task prompt to give to the VLA policy.
        server_address (str): The address of the server to connect to.
        policy_name (str): The name or path of the pretrained policy.
        policy_type (str): The type of policy to use.
        arm_port (str): The USB port of the robot's arm.
        camera_config (dict, optional): Lerobot-type camera configuration. (E.g., "{ main: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, left_arm: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}")
        execution_time (int, optional): Time in seconds to run the manipulation.
        policy_device (str, optional): The device to run the policy on. Defaults to "cuda".
        fps (int, optional): The fps to run the policy at.
        actions_per_chunk (int, optional): Number of actions VLA calculates at once.
        load_on_startup (bool, optional): Whether to load the VLA policy on startup. If False, the policy will be loaded every time the tool used, which may cause a delay. If True for many tools, you may overload server's GPU.
    """
    # The async policy stack imports optional model backends. Keep it out of
    # normal dashboard startup when no VLA tool is configured.
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

    right_port = getattr(servo_controler, "right_arm_wheel_usb", None)
    left_port = getattr(servo_controler, "left_arm_head_usb", None)
    arm_side = (
        "right" if arm_port == right_port else
        "left" if arm_port == left_port else
        "right" if "right" in str(arm_port).lower() else
        "left" if "left" in str(arm_port).lower() else
        None
    )


    configured_cameras = {}
    for cam_name, cam_settings in camera_config.items():
        # Unpack the dictionary settings directly into the Config class
        configured_cameras[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_settings["index_or_path"],
            width=cam_settings.get("width", 640),
            height=cam_settings.get("height", 480),
            fps=cam_settings.get("fps", 30)
        )

    robot_config = SOFollowerConfig(
        port=arm_port,
        cameras=configured_cameras,
    )

    robot_config.type = "so101_follower"

    robot_config.id="robot_arm"

    # LeRobot expects a Path-like object here (it calls mkdir on this value).
    robot_config.calibration_dir = Path(DEFAULT_ARM_CALIBRATION_DIR).expanduser()
    

    cfg = RobotClientConfig(
        robot=robot_config,
        task=task_prompt,
        server_address=server_address,
        policy_type=policy_type,
        pretrained_name_or_path=policy_name,
        policy_device=policy_device,
        actions_per_chunk=actions_per_chunk,
        chunk_size_threshold=0.5,
        fps=fps
    )


    preloaded_client = None

    if load_on_startup:
        print(f" Loading Policy for {tool_name}...")
        # release main camera from agent
        main_camera_object.release()
        time.sleep(1) 

        preloaded_client = RobotClient(cfg)
        preloaded_client.robot.disconnect()

        # Warm up once at startup so server loads policy weights before first real execution.
        warmup_client = RobotClient(cfg)
        warmup_client.robot.disconnect()

        #assign main camera back to agent
        time.sleep(0.5)
        main_camera_object.reopen()
    
    @tool
    def tool_name_to_override() -> str:
        """Tool description to override."""
        print("Manipulation tool activated")

        servo_controler.set_saved_position("cobra", arm_side=arm_side)

        servo_controler.turn_head_to_vla_position()
        # release main camera from agent, so arm policy can use it
        main_camera_object.release()
        time.sleep(1)  # give some time to release camera

        client = None
        try:

            if not load_on_startup:
                client = RobotClient(cfg)
            else:
                client = preloaded_client
                client.robot.connect()

            # Use a fresh RobotClient per invocation so worker threads can be stopped cleanly.
            client = RobotClient(cfg)

            if not client.start():
                return "Failed to connect to robot server."

            threading.Thread(target=client.receive_actions, daemon=True).start()
            threading.Timer(execution_time, _shutdown_robot_client, args=(client,)).start()
            try:
                client.control_loop(task=task_prompt)
            except Exception:
                pass
        
        finally:

            #if client and client.robot.is_connected:
            if not load_on_startup and client:
                client.stop()
            # Re-open main camera for agent use. 
            time.sleep(1)
            main_camera_object.reopen()
            # set head back to precize mode
            servo_controler.turn_head_to_vla_position(50)

            if client:
                try:
                    client.stop()
                except Exception:
                    pass
            # Re-open main camera for agent use. 
            time.sleep(1)
            main_camera_object.reopen()
            time.sleep(0.3)
            # set head back to precize mode
            servo_controler.turn_head_to_vla_position(50)
            servo_controler.set_saved_position("default", arm_side="both")  # optionally set a default position for both arms after manipulation

        
        return "Arm manipulation done"
    
    tool_name_to_override.name = tool_name
    tool_name_to_override.description = tool_description

    return tool_name_to_override


def _shutdown_robot_client(client: "RobotClient") -> None:
    """Gracefully stop the control loop before disconnecting the robot.

    Signals the running control loop to exit on its next iteration before
    hardware disconnection, preventing race conditions.
    """
    client.stop()



def _groot_recursive_add_extra_dim(obs: dict) -> dict:
    """Add one (batch or time) dimension to every leaf in the obs dict recursively."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = _groot_recursive_add_extra_dim(val)
        else:
            obs[key] = [val]  # scalar / string -> list
    return obs


def _groot_build_observation(frame1_rgb, frame2_rgb, state_rad, task_prompt: str) -> dict:
    """Convert raw sensor data into the nested dict GR00T policy server expects.

    Camera keys must match those in modality.json (camera1, camera2).
    State is split into single_arm (5 joints) and gripper (1 joint).
    All arrays get (B=1, T=1) dims via two recursive calls.
    """
    obs = {
        "video": {
            "camera1": frame1_rgb,                       # (H, W, 3)  uint8
            "camera2": frame2_rgb,                       # (H, W, 3)  uint8
        },
        "state": {
            "single_arm": state_rad[:5].astype(np.float32),  # (5,)
            "gripper":    state_rad[5:6].astype(np.float32), # (1,)
        },
        "language": {
            "annotation.human.task_description": task_prompt,
        },
    }
    obs = _groot_recursive_add_extra_dim(obs)  # -> (1, ...)
    obs = _groot_recursive_add_extra_dim(obs)  # -> (1, 1, ...)
    return obs


def _groot_decode_action_chunk(chunk: dict, t: int, motor_ids: list) -> dict:
    """Extract timestep t from action chunk dict and map to {motor_id: degrees}.

    chunk["single_arm"]: (B, T, 5)  radians
    chunk["gripper"]:    (B, T, 1)  radians
    Returns: {motor_id: float_degrees}
    """
    single_arm = chunk["single_arm"][0][t]  # (5,)
    gripper    = chunk["gripper"][0][t]      # (1,)
    full_rad   = np.concatenate([single_arm, gripper], axis=0)  # (6,)
    return {
        mid: math.degrees(float(full_rad[i]))
        for i, mid in enumerate(motor_ids)
    }

def create_groot_single_arm_manipulation(
        tool_name: str,
        tool_description: str,
        task_prompt: str,
        server_host: str,
        server_port: int,
        arm_port: str,
        motor_ids: list,
        camera1_index_or_path,
        camera2_index_or_path,
        camera_width: int,
        camera_height: int,
        main_camera_object,
        servo_controller,
        execution_time: int = 30,
        fps: int = 30,
        timeout_ms: int = 15000,
        calibration_path: str = "/home/pi/.cache/robocrew/calibrations/right_arm.json",
    ):
    """Creates a LangChain tool that runs a GR00T policy for single-arm manipulation.

    Args:
        tool_name (str): The name of the tool the AI agent will see.
        tool_description (str): The description of the tool the AI agent will see.
        task_prompt (str): Natural-language task instruction sent to the GR00T policy.
        server_host (str): Hostname of the running GR00T policy server.
        server_port (int): Port of the GR00T policy server (default 5555).
        arm_port (str): USB device path for the arm's FeetechMotorsBus (e.g. "/dev/arm_right").
        motor_ids (list): Ordered list of motor IDs on the arm (e.g. [1,2,3,4,5,6]).
        camera1_index_or_path: OpenCV index or device path for the primary arm camera.
        camera2_index_or_path: OpenCV index or device path for the secondary/overview camera.
        camera_width (int): Camera capture width in pixels.
        camera_height (int): Camera capture height in pixels.
        main_camera_object: The agent's main camera — released before and restored after execution.
        servo_controller: Robot servo controller used to position the head for manipulation.
        execution_time (int): How long in seconds to run the policy.
        fps (int): Control loop frequency.
        timeout_ms (int): PolicyClient request timeout in milliseconds.
        calibration_path (str): Path to a lerobot calibration JSON file. Required for
            normalized (degree-mode) motor reads. Typically found at
            ~/.cache/huggingface/lerobot/calibration/robots/<robot>/<id>.json
    """

    @tool
    def tool_name_to_override() -> str:
        """Tool description to override."""
        print(f"GR00T manipulation tool activated: {tool_name}")

        servo_controller.turn_head_to_vla_position()
        main_camera_object.release()
        time.sleep(1)

        cap1 = cv2.VideoCapture(camera1_index_or_path)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        cap2 = cv2.VideoCapture(camera2_index_or_path)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        # Load calibration from lerobot JSON if provided
        calibration = None
        if calibration_path:
            import json
            from lerobot.motors import MotorCalibration
            with open(calibration_path) as f:
                raw = json.load(f)
            calibration = {
                entry["id"]: MotorCalibration(
                    id=entry["id"],
                    drive_mode=entry["drive_mode"],
                    homing_offset=entry["homing_offset"],
                    range_min=entry["range_min"],
                    range_max=entry["range_max"],
                )
                for entry in raw.values()
            }

        arm_bus = FeetechMotorsBus(
            port=arm_port,
            motors={mid: Motor(mid, "sts3215", MotorNormMode.DEGREES) for mid in motor_ids},
            calibration=calibration,
        )
        arm_bus.connect()

        policy = PolicyClient(host=server_host, port=server_port, timeout_ms=timeout_ms)
        if not policy.ping():
            arm_bus.disconnect()
            cap1.release()
            cap2.release()
            time.sleep(1)
            main_camera_object.reopen()
            servo_controller.turn_head_to_vla_position(50)
            return "Failed to connect to GR00T policy server."

        policy.reset()

        dt = 1.0 / fps
        start_time = time.time()

        try:
            while time.time() - start_time < execution_time:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break

                frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                positions_deg = [arm_bus.read("Present_Position", mid) for mid in motor_ids]
                state_rad = np.array(
                    [math.radians(deg) for deg in positions_deg], dtype=np.float32
                )  # (6,)

                obs = _groot_build_observation(frame1_rgb, frame2_rgb, state_rad, task_prompt)

                action_chunk, _ = policy.get_action(obs)
                # action_chunk = {"single_arm": (1, T, 5), "gripper": (1, T, 1)}

                horizon = action_chunk["single_arm"].shape[1]
                for t in range(horizon):
                    if time.time() - start_time >= execution_time:
                        break
                    action_deg = _groot_decode_action_chunk(action_chunk, t, motor_ids)
                    arm_bus.sync_write("Goal_Position", action_deg)
                    time.sleep(dt)

        finally:
            arm_bus.disconnect()
            cap1.release()
            cap2.release()
            time.sleep(1)
            main_camera_object.reopen()
            servo_controller.turn_head_to_vla_position(50)

        return "GR00T arm manipulation done."

    tool_name_to_override.name = tool_name
    tool_name_to_override.description = tool_description

    return tool_name_to_override


def _shutdown_robot_client(client: "RobotClient") -> None:
    """Gracefully stop the control loop before disconnecting the robot.

    Signals the running control loop to exit on its next iteration before
    hardware disconnection, preventing race conditions.
    """
    client.stop()
