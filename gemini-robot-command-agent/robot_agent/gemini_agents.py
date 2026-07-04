from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


ACTION_SCHEMA = """
Return strict JSON only:
{
  "thought": "short private-to-operator reasoning summary",
  "progress": "what changed since last observation",
  "done": false,
  "action": {
    "type": "observe|preset|move_joint|nudge_joint|set_gripper|wait|ask_human|finish|stop",
    "joint": "base|shoulder|elbow|wrist|gripper",
    "position": 500,
    "delta": 20,
    "duration_ms": 800,
    "name": "home|ready",
    "state": "open|closed",
    "reason": "why"
  }
}
Use only one action at a time. Prefer observe/ask_human if calibration or object
location is uncertain. Never command motion outside configured limits.
"""


class GeminiRobotAgents:
    def __init__(
        self,
        robotics_model: str | None = None,
        reasoning_model: str | None = None,
        thinking_budget: int = 512,
    ):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment or repo .env")
        self.client = genai.Client(api_key=api_key)
        self.robotics_model = robotics_model or os.environ.get("GEMINI_ROBOTICS_MODEL", "gemini-robotics-er-1.6-preview")
        self.reasoning_model = reasoning_model or os.environ.get("GEMINI_REASONING_MODEL", "gemini-3.1-pro-preview")
        self.thinking_budget = thinking_budget

    def plan(self, goal: str, robot_state: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""
You are the high-level reasoning agent for a small tabletop robot arm.
Goal: {goal}
Current robot state JSON: {json.dumps(robot_state, indent=2)}

Create a cautious subplan. The robot has one left-arm camera and calibrated
serial servo tools only. Do not invent direct Cartesian control. If calibration
is missing or the task needs object location, start with observe.

Return JSON only:
{{
  "goal": "...",
  "subtasks": ["...", "..."],
  "risk_check": "...",
  "first_action": {{"type": "observe", "reason": "..."}}
}}
"""
        text = self._text_call(self.reasoning_model, prompt, include_json_mode=True)
        return self._parse_json(text)

    def next_action(
        self,
        goal: str,
        plan: dict[str, Any],
        robot_state: dict[str, Any],
        frame_jpeg: bytes,
        last_result: str,
    ) -> dict[str, Any]:
        prompt = f"""
You are Gemini Robotics controlling a small tabletop robot arm through safe,
bounded local tools. You receive the current camera frame, a goal, the plan,
the servo state, and the result of the previous action.

Goal: {goal}
Plan JSON: {json.dumps(plan, indent=2)}
Robot state JSON: {json.dumps(robot_state, indent=2)}
Previous tool result: {last_result}

Use visual/spatial reasoning to choose the next safest single tool action.
If the target object is not visible or if movement would require uncalibrated
kinematics, return ask_human or observe. To approach an object, prefer small
nudge_joint actions and observe after each motion.

{ACTION_SCHEMA}
"""
        response = self.client.models.generate_content(
            model=self.robotics_model,
            contents=[
                types.Part.from_bytes(data=frame_jpeg, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.4,
                thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
                response_mime_type="application/json",
            ),
        )
        return self._parse_json(response.text or "{}")

    def _text_call(self, model: str, prompt: str, include_json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {
            "temperature": 0.2,
            "thinking_config": types.ThinkingConfig(thinking_budget=self.thinking_budget),
        }
        if include_json_mode:
            kwargs["response_mime_type"] = "application/json"
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**kwargs),
        )
        return response.text or ""

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("Gemini response was not a JSON object")
        return value
