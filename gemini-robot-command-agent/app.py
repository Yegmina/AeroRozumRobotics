from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from robot_agent.camera_source import LeftArmCamera, error_frame
from robot_agent.config import load_config
from robot_agent.controller import RobotController
from robot_agent.gemini_agents import GeminiRobotAgents


APP_DIR = Path(__file__).resolve().parent

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Robot Command Agent</title>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #101418; color: #edf2f7; }
    main { display: grid; grid-template-columns: 1.3fr 1fr; min-height: 100vh; }
    section { padding: 18px; }
    img { width: 100%; max-height: 92vh; object-fit: contain; background: #05070a; }
    textarea { width: 100%; min-height: 86px; box-sizing: border-box; background: #17202a; color: #fff; border: 1px solid #334155; padding: 10px; }
    button { background: #1f8a70; color: #fff; border: 0; padding: 10px 14px; margin: 8px 8px 8px 0; cursor: pointer; }
    button.stop { background: #b42318; }
    button.manual { background: #315a9f; }
    pre { white-space: pre-wrap; background: #17202a; border: 1px solid #334155; padding: 12px; max-height: 58vh; overflow: auto; }
    .status { color: #9ae6b4; font-size: 14px; }
  </style>
</head>
<body>
<main>
  <section><img src="/video_feed" /></section>
  <section>
    <h2>Gemini Robot Command Agent</h2>
    <p class="status" id="status">loading...</p>
    <textarea id="goal" placeholder="take the cup in the robot hand"></textarea>
    <br>
    <button onclick="startGoal()">Start Goal</button>
    <button onclick="singleStep()">Run One Step</button>
    <button class="stop" onclick="stopGoal()">Stop</button>
    <h3>Manual Motion Test</h3>
    <button class="manual" onclick="scanServos()">Scan Servos</button>
    <button class="manual" onclick="manual({type:'preset', name:'ready'})">Ready</button>
    <button class="manual" onclick="manual({type:'preset', name:'home'})">Home</button>
    <br>
    <button class="manual" onclick="manual({type:'json', payload:{T:100}})">RoArm Home</button>
    <button class="manual" onclick="manual({type:'json', payload:{T:210,cmd:1}})">Torque On</button>
    <button class="manual" onclick="manual({type:'json', payload:{T:210,cmd:0}})">Torque Off</button>
    <br>
    <button class="manual" onclick="manual({type:'json', payload:{T:121,joint:1,angle:10,spd:1000}})">JSON Base 10</button>
    <button class="manual" onclick="manual({type:'json', payload:{T:121,joint:1,angle:-10,spd:1000}})">JSON Base -10</button>
    <button class="manual" onclick="manual({type:'json', payload:{T:101,joint:1,rad:0.18,spd:0,acc:10}})">Legacy Base +</button>
    <button class="manual" onclick="manual({type:'json', payload:{T:101,joint:1,rad:-0.18,spd:0,acc:10}})">Legacy Base -</button>
    <br>
    <button class="manual" onclick="manual({type:'nudge_joint', joint:'base', delta:-8})">Base -</button>
    <button class="manual" onclick="manual({type:'nudge_joint', joint:'base', delta:8})">Base +</button>
    <button class="manual" onclick="manual({type:'nudge_joint', joint:'shoulder', delta:-8})">Shoulder -</button>
    <button class="manual" onclick="manual({type:'nudge_joint', joint:'shoulder', delta:8})">Shoulder +</button>
    <br>
    <button class="manual" onclick="manual({type:'nudge_joint', joint:'elbow', delta:-8})">Elbow -</button>
    <button class="manual" onclick="manual({type:'nudge_joint', joint:'elbow', delta:8})">Elbow +</button>
    <button class="manual" onclick="manual({type:'set_gripper', state:'open'})">Open</button>
    <button class="manual" onclick="manual({type:'set_gripper', state:'closed'})">Close</button>
    <pre id="log"></pre>
  </section>
</main>
<script>
async function api(path, body) {
  const res = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
  const data = await res.json();
  if (!res.ok) data.http_error = res.status;
  return data;
}
async function refresh() {
  const res = await fetch('/state');
  const data = await res.json();
  document.getElementById('status').textContent = `running=${data.running} dry_run=${data.robot.dry_run} port=${data.robot.serial_port}`;
  document.getElementById('log').textContent = JSON.stringify(data, null, 2);
}
async function startGoal() { const data = await api('/goal', {goal: document.getElementById('goal').value}); render(data); }
async function singleStep() { const data = await api('/step', {goal: document.getElementById('goal').value}); render(data); }
async function stopGoal() { await api('/stop', {}); refresh(); }
async function manual(action) { const data = await api('/manual', {action}); render(data); }
async function scanServos() { const data = await api('/scan', {start_id: 1, end_id: 30}); render(data); }
function render(data) {
  document.getElementById('status').textContent = `running=${data.running} dry_run=${data.robot.dry_run} port=${data.robot.serial_port} protocol=${data.robot.protocol}`;
  document.getElementById('log').textContent = JSON.stringify(data, null, 2);
}
setInterval(refresh, 1500); refresh();
</script>
</body>
</html>
"""


class GoalRunner:
    def __init__(self, controller: RobotController, agents: GeminiRobotAgents, camera: LeftArmCamera):
        self.controller = controller
        self.agents = agents
        self.camera = camera
        self.goal = ""
        self.plan: dict[str, Any] = {}
        self.last_result = "not started"
        self.camera_error = ""
        self.history: list[dict[str, Any]] = []
        self.running = False
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start_goal(self, goal: str) -> dict[str, Any]:
        if not goal.strip():
            raise ValueError("Goal is empty")
        with self._lock:
            self.goal = goal.strip()
            self.history.clear()
            self.last_result = "goal accepted"
            self.plan = self.agents.plan(self.goal, self.controller.snapshot())
            self.stop_event.clear()
            self.running = True
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="GoalRunner", daemon=True)
                self._thread.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        with self._lock:
            self.running = False
            self.last_result = "operator stopped"
            return self.snapshot()

    def step_once(self) -> dict[str, Any]:
        self._step()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "plan": self.plan,
            "last_result": self.last_result,
            "camera_error": self.camera_error,
            "history": self.history[-20:],
            "running": self.running,
            "robot": self.controller.snapshot(),
        }

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                with self._lock:
                    should_run = self.running
                if should_run:
                    self._step()
                time.sleep(2.0)
            except Exception as exc:
                with self._lock:
                    self.last_result = f"error: {exc}"
                    self.running = False

    def _step(self) -> None:
        try:
            frame = self.camera.latest()
            jpeg = self.camera.jpeg(frame)
            camera_error = ""
        except Exception as exc:
            with self._lock:
                self.camera_error = str(exc)
                self.last_result = f"camera error: {exc}"
                self.running = False
            return
        with self._lock:
            goal = self.goal
            plan = dict(self.plan)
            last_result = self.last_result
        if not goal:
            raise ValueError("No active goal")
        decision = self.agents.next_action(goal, plan, self.controller.snapshot(), jpeg, last_result)
        action = decision.get("action", {"type": "observe"})
        result = self.controller.execute(action)
        record = {"decision": decision, "tool_result": result, "time": time.strftime("%H:%M:%S")}
        with self._lock:
            self.camera_error = camera_error
            self.last_result = result
            self.history.append(record)
            if bool(decision.get("done")) or action.get("type") in {"finish", "stop", "ask_human"}:
                self.running = False


def create_app(enable_motion: bool, config_path: Path) -> Flask:
    config = load_config(config_path)
    controller = RobotController(config, enable_motion=enable_motion)
    agents = GeminiRobotAgents()
    camera = LeftArmCamera()
    camera.__enter__()
    runner = GoalRunner(controller, agents, camera)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/state")
    def state():
        return jsonify(runner.snapshot())

    @app.post("/goal")
    def goal():
        try:
            data = request.get_json(force=True)
            return jsonify(runner.start_goal(str(data.get("goal", ""))))
        except Exception as exc:
            return jsonify({"error": str(exc), **runner.snapshot()}), 400

    @app.post("/step")
    def step():
        try:
            data = request.get_json(silent=True) or {}
            if not runner.goal and str(data.get("goal", "")).strip():
                runner.start_goal(str(data.get("goal", "")))
            return jsonify(runner.step_once())
        except Exception as exc:
            return jsonify({"error": str(exc), **runner.snapshot()}), 400

    @app.post("/manual")
    def manual():
        try:
            data = request.get_json(force=True)
            result = controller.execute(data.get("action", {}))
            with runner._lock:
                runner.last_result = result
            return jsonify(runner.snapshot())
        except Exception as exc:
            return jsonify({"error": str(exc), **runner.snapshot()}), 400

    @app.post("/scan")
    def scan():
        try:
            data = request.get_json(silent=True) or {}
            found = controller.scan_servos(int(data.get("start_id", 1)), int(data.get("end_id", 30)))
            with runner._lock:
                runner.last_result = f"scan found ids: {found}"
            snapshot = runner.snapshot()
            snapshot["scan"] = found
            return jsonify(snapshot)
        except Exception as exc:
            return jsonify({"error": str(exc), **runner.snapshot()}), 400

    @app.post("/stop")
    def stop():
        return jsonify(runner.stop())

    @app.get("/video_feed")
    def video_feed():
        def stream():
            while True:
                try:
                    frame = camera.latest()
                    label = f"left arm camera: {camera.device_name}"
                    cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
                    with runner._lock:
                        runner.camera_error = ""
                except Exception as exc:
                    frame = error_frame(str(exc))
                    with runner._lock:
                        runner.camera_error = str(exc)
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                time.sleep(0.04)

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.teardown_appcontext
    def cleanup(_exc):
        controller.close()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini goal-command app for the left camera and robot arm.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", type=Path, default=APP_DIR / "servo_config.json")
    parser.add_argument("--enable-motion", action="store_true", help="Actually write serial servo commands.")
    args = parser.parse_args()

    app = create_app(enable_motion=args.enable_motion, config_path=args.config)
    print(f"Open http://{args.host}:{args.port}")
    print("Motion is ENABLED" if args.enable_motion else "Motion is DRY-RUN only")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
