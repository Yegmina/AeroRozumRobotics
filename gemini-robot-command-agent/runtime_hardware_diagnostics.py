from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import serial.tools.list_ports

from robot_agent.config import load_config
from robot_agent.controller import RobotController


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = APP_DIR / "servo_config.json"


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def print_line(message: str, log_file: Path | None = None) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def list_ports() -> list[dict]:
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append(
            {
                "device": port.device,
                "description": port.description,
                "hwid": port.hwid,
                "manufacturer": port.manufacturer,
                "serial_number": port.serial_number,
            }
        )
    return ports


def http_json(base_url: str, path: str, payload: dict | None = None, timeout_s: float = 10.0) -> dict:
    if payload is None:
        with urllib.request.urlopen(base_url + path, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def run_http(args: argparse.Namespace) -> None:
    base_url = args.base_url.rstrip("/")
    print_line(f"HTTP diagnostics via {base_url}", args.log_file)
    previous: dict[str, int | None] = {}
    for index in loop_range(args.count):
        try:
            state = http_json(base_url, "/state", timeout_s=args.timeout)
        except urllib.error.URLError as exc:
            print_line(f"backend unavailable: {exc}. Start app.py or run without --via-http.", args.log_file)
            return
        robot = state["robot"]
        print_line(
            "state "
            f"dry_run={robot['dry_run']} motion_enabled={robot['motion_enabled']} "
            f"protocol={robot['protocol']} port={robot['serial_port']}",
            args.log_file,
        )
        if index % args.scan_every == 0:
            try:
                scan = http_json(
                    base_url,
                    "/scan",
                    {"start_id": args.start_id, "end_id": args.end_id},
                    timeout_s=args.timeout,
                )
            except urllib.error.URLError as exc:
                print_line(f"scan failed through backend: {exc}", args.log_file)
                return
            print_line(f"scan ids={scan.get('scan')}", args.log_file)
            robot = scan["robot"]
        readbacks = robot.get("readback_positions", {})
        print_readbacks(readbacks, previous, args.log_file)
        previous = dict(readbacks)
        time.sleep(args.interval)


def run_direct(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print_line("serial ports:", args.log_file)
    for port in list_ports():
        print_line(f"  {json.dumps(port, separators=(',', ':'))}", args.log_file)
    print_line(
        f"direct diagnostics port={config.serial.port} baud={config.serial.baudrate} "
        f"protocol={config.serial.protocol}",
        args.log_file,
    )
    controller = RobotController(config, enable_motion=True)
    previous: dict[str, int | None] = {}
    try:
        for index in loop_range(args.count):
            if index % args.scan_every == 0:
                found = controller.scan_servos(args.start_id, args.end_id)
                print_line(f"scan ids={found}", args.log_file)
            readbacks = controller.readback_positions()
            print_readbacks(readbacks, previous, args.log_file)
            previous = dict(readbacks)
            time.sleep(args.interval)
    finally:
        controller.close()


def print_readbacks(readbacks: dict[str, int | None], previous: dict[str, int | None], log_file: Path | None) -> None:
    for name, value in readbacks.items():
        old = previous.get(name)
        delta = None if old is None or value is None else value - old
        print_line(f"readback {name}={value} delta={delta}", log_file)


def loop_range(count: int):
    index = 0
    while count <= 0 or index < count:
        yield index
        index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime hardware connection diagnostics for the robot servo bus.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=12)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=5, help="Loop count. Use 0 for infinite.")
    parser.add_argument("--scan-every", type=int, default=1, help="Run ID scan every N loops.")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--via-http", action="store_true", help="Use the running Flask backend instead of opening COM directly.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_every < 1:
        raise ValueError("--scan-every must be >= 1")
    if args.via_http:
        run_http(args)
    else:
        run_direct(args)


if __name__ == "__main__":
    main()
