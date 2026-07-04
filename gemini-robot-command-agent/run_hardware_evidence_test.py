from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import urllib.request
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "robot_hardware_evidence"


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    action: dict
    settle_s: float = 2.0


SCENARIOS = [
    Scenario(
        name="01_scan",
        description="Scan the transparent ST/SC serial bus for responding servo IDs.",
        action={"type": "scan", "start_id": 1, "end_id": 12},
        settle_s=1.0,
    ),
    Scenario(
        name="02_shoulder_positive",
        description="Move shoulder joint by +60 ST/SMS position units from live readback.",
        action={"type": "nudge_joint", "joint": "shoulder", "delta": 60},
    ),
    Scenario(
        name="03_shoulder_negative",
        description="Move shoulder joint by -60 ST/SMS position units from live readback.",
        action={"type": "nudge_joint", "joint": "shoulder", "delta": -60},
    ),
    Scenario(
        name="04_shoulder_negative",
        description="Move shoulder joint by -60 ST/SMS position units from live readback.",
        action={"type": "nudge_joint", "joint": "shoulder", "delta": -60},
    ),
    Scenario(
        name="05_elbow_positive",
        description="Move elbow joint by +60 ST/SMS position units from live readback.",
        action={"type": "nudge_joint", "joint": "elbow", "delta": 60},
    ),
    Scenario(
        name="06_wrist_negative",
        description="Move wrist joint by -60 ST/SMS position units from live readback.",
        action={"type": "nudge_joint", "joint": "wrist", "delta": -60},
    ),
    Scenario(
        name="07_wrist_positive",
        description="Move wrist joint by +60 ST/SMS position units from live readback.",
        action={"type": "nudge_joint", "joint": "wrist", "delta": 60},
    ),
]


def api_json(base_url: str, path: str, payload: dict | None = None, timeout_s: float = 20.0) -> dict:
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


def capture_mjpeg_frame(base_url: str, output_path: Path, timeout_s: float = 10.0) -> np.ndarray:
    with urllib.request.urlopen(base_url + "/video_feed", timeout=timeout_s) as response:
        buffer = bytearray()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            chunk = response.read(4096)
            if not chunk:
                continue
            buffer.extend(chunk)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                jpg = bytes(buffer[start : end + 2])
                image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("Could not decode MJPEG frame")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), image)
                return image
    raise TimeoutError("Timed out waiting for MJPEG camera frame")


def movement_metrics(before: np.ndarray, after: np.ndarray, diff_path: Path) -> dict:
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_before, gray_after)
    mean_abs = float(np.mean(diff))
    changed_percent = float(np.mean(diff > 25) * 100.0)
    heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(diff_path), heat)
    return {
        "mean_abs_diff": round(mean_abs, 2),
        "changed_percent": round(changed_percent, 2),
        "diff_image": str(diff_path),
    }


def run_scenarios(base_url: str, output_dir: Path) -> dict:
    image_dir = output_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    results = []
    initial_state = api_json(base_url, "/state")
    for scenario in SCENARIOS:
        before_path = image_dir / f"{scenario.name}_before.jpg"
        after_path = image_dir / f"{scenario.name}_after.jpg"
        diff_path = image_dir / f"{scenario.name}_diff.jpg"
        before = capture_mjpeg_frame(base_url, before_path)
        before_state = api_json(base_url, "/state")
        if scenario.action.get("type") == "scan":
            response = api_json(
                base_url,
                "/scan",
                {"start_id": scenario.action["start_id"], "end_id": scenario.action["end_id"]},
            )
        else:
            response = api_json(base_url, "/manual", {"action": scenario.action})
        time.sleep(scenario.settle_s)
        after_state = api_json(base_url, "/state")
        after = capture_mjpeg_frame(base_url, after_path)
        metrics = movement_metrics(before, after, diff_path)
        servo_before = before_state["robot"].get("readback_positions", {})
        servo_after = after_state["robot"].get("readback_positions", {})
        servo_delta = {
            key: None if servo_before.get(key) is None or servo_after.get(key) is None else servo_after[key] - servo_before[key]
            for key in sorted(set(servo_before) | set(servo_after))
        }
        results.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "action": scenario.action,
                "response": response,
                "before": str(before_path),
                "after": str(after_path),
                "metrics": metrics,
                "servo_before": servo_before,
                "servo_after": servo_after,
                "servo_delta": servo_delta,
            }
        )
    final_state = api_json(base_url, "/state")
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_url": base_url,
        "initial_state": initial_state,
        "final_state": final_state,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hardware_evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fit_image(path: str, width_mm: float) -> PdfImage:
    with Image.open(path) as image:
        width_px, height_px = image.size
    width = width_mm * mm
    height = width * (height_px / width_px)
    return PdfImage(path, width=width, height=height)


def build_pdf(summary: dict, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    small = styles["Normal"].clone("SmallTable")
    small.fontSize = 6
    small.leading = 7

    def para(value: object) -> Paragraph:
        return Paragraph(escape(str(value)).replace("\n", "<br/>"), small)

    def servo_lines(value: dict) -> str:
        return "\n".join(f"{key}: {item}" for key, item in value.items())

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
    )
    story = []
    story.append(Paragraph("Robot Hardware Evidence Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {summary['timestamp']}", styles["Normal"]))
    final_robot = summary["final_state"]["robot"]
    story.append(
        Paragraph(
            "Server state: "
            f"dry_run={final_robot['dry_run']}, motion_enabled={final_robot['motion_enabled']}, "
            f"protocol={final_robot['protocol']}, port={final_robot['serial_port']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    rows = [["Scenario", "Command result", "Servo deltas", "Mean diff", "Changed pixels"]]
    for item in summary["results"]:
        result = item["response"].get("last_result", "").replace(" -> ", "\n-> ")
        rows.append(
            [
                para(item["name"]),
                para(result),
                para(servo_lines(item.get("servo_delta", {}))),
                para(item["metrics"]["mean_abs_diff"]),
                para(f"{item['metrics']['changed_percent']}%"),
            ]
        )
    table = Table(rows, colWidths=[31 * mm, 78 * mm, 38 * mm, 19 * mm, 20 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f7")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(table)
    story.append(PageBreak())

    for item in summary["results"]:
        story.append(Paragraph(item["name"], styles["Heading2"]))
        story.append(Paragraph(item["description"], styles["Normal"]))
        story.append(Paragraph(f"Action JSON: {json.dumps(item['action'], separators=(',', ':'))}", styles["Code"]))
        story.append(Paragraph(f"App result: {item['response'].get('last_result', '')}", styles["Normal"]))
        story.append(Paragraph(f"Servo before: {json.dumps(item.get('servo_before', {}), separators=(',', ':'))}", styles["Code"]))
        story.append(Paragraph(f"Servo after: {json.dumps(item.get('servo_after', {}), separators=(',', ':'))}", styles["Code"]))
        story.append(Paragraph(f"Servo delta: {json.dumps(item.get('servo_delta', {}), separators=(',', ':'))}", styles["Code"]))
        story.append(
            Paragraph(
                f"Movement score: mean_abs_diff={item['metrics']['mean_abs_diff']}, "
                f"changed_pixels={item['metrics']['changed_percent']}%",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 3 * mm))
        image_table = Table(
            [
                [Paragraph("Before", styles["Heading4"]), Paragraph("After", styles["Heading4"])],
                [fit_image(item["before"], 86), fit_image(item["after"], 86)],
                [Paragraph("Difference heatmap", styles["Heading4"]), ""],
                [fit_image(item["metrics"]["diff_image"], 86), ""],
            ],
            colWidths=[88 * mm, 88 * mm],
        )
        image_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(image_table)
        story.append(PageBreak())
    doc.build(story)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robot hardware evidence scenarios and generate a PDF report.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_scenarios(args.base_url.rstrip("/"), args.output_dir)
    pdf_path = args.output_dir / "robot_hardware_evidence_report.pdf"
    build_pdf(summary, pdf_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
