"""
api.py — Flask REST backend for FloorPlan AI web interface.

Endpoints:
  POST /api/process          — Upload floor plan, kick off pipeline job
  GET  /api/status/<job_id>  — Poll job status + completed stages
  GET  /api/output/<job_id>/<filename> — Serve output images
"""

import os
import sys
import uuid
import threading
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import werkzeug

# ── ensure project root is on PYTHONPATH ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml

app = Flask(__name__)
CORS(app)  # allow React dev server on :3000

UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "assets"
CONFIG_PATH = ROOT / "config.yaml"
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── In-memory job store (use Redis for production) ────────────────────────────
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Pipeline runner (runs in background thread) ───────────────────────────────
def run_pipeline(job_id: str, input_path: str, render: bool):
    try:
        import cv2
        import numpy as np

        from src.floorplan.ocr import extract_labels
        from src.floorplan.walls import extract_walls
        from src.floorplan.segment import segment_rooms
        from src.floorplan.classify import colorize_rooms
        from src.floorplan.furniture import furnish_all, place_room_furniture
        from src.floorplan.draw_symbols import draw_architectural_plan

        cfg = load_config()
        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        STAGES = [
            "ocr", "walls", "segment", "classify", "furniture", "symbols", "render"
        ]

        def stage(idx: int):
            update_job(job_id, stage_index=idx, current_stage=STAGES[idx])

        def complete_stage(name: str):
            with jobs_lock:
                jobs[job_id]["completed_stages"].append(name)

        # 1 — Load image
        stage(0)
        color = cv2.imread(input_path)
        if color is None:
            raise ValueError(f"Cannot read image: {input_path}")
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # 2 — OCR
        ocr_labels = extract_labels(color, min_length=cfg["ocr"]["min_text_length"])
        complete_stage("ocr")

        # 3 — Wall extraction
        stage(1)
        w = cfg["walls"]
        wall_mask = extract_walls(
            gray,
            adaptive_block_size=w["adaptive_block_size"],
            adaptive_C=w["adaptive_C"],
            morph_kernel=tuple(w["morph_kernel"]),
            min_component_area=w["min_component_area"],
            min_wall_area=w["min_wall_area"],
            min_wall_thickness=w["min_wall_thickness"],
        )
        cv2.imwrite(str(out_dir / "walls_no_text.png"), wall_mask)
        complete_stage("walls")

        # 4 — Segmentation
        stage(2)
        rooms = segment_rooms(wall_mask, ocr_labels)
        complete_stage("segment")

        # 5 — Classify + colorize
        stage(3)
        colored = colorize_rooms(color, rooms)
        cv2.imwrite(str(out_dir / "colored_plan.png"), colored)
        complete_stage("classify")

        # 6 — Furniture
        stage(4)
        furniture_by_room = []
        for room in rooms:
            items = place_room_furniture(room, wall_mask, None, None)
            furniture_by_room.append({"room": room, "items": items})

        furnished = furnish_all(
            rooms, wall_mask,
            door_mask=None, clearance_map=None,
            cfg=cfg["furniture"],
            original_image=colored,
        )
        cv2.imwrite(str(out_dir / "furnished_plan.png"), furnished)
        complete_stage("furniture")

        # 7 — Symbol drawing
        stage(5)
        line_drawing = draw_architectural_plan(
            wall_mask=wall_mask,
            rooms=rooms,
            furniture_by_room=furniture_by_room,
            target_size=cfg["render"]["image_size"],
        )
        cv2.imwrite(str(out_dir / "line_drawing.png"), line_drawing)
        complete_stage("symbols")

        outputs = {
            "colored_plan": True,
            "furnished_plan": True,
            "line_drawing": True,
        }

        # 8 — Optional 3D render
        if render:
            stage(6)
            from src.floorplan.render import load_pipeline, render_isometric
            import cv2 as cv

            rc = cfg["render"]
            pipe, device = load_pipeline()
            blurred = cv.GaussianBlur(line_drawing, (5, 5), 0)
            edges = cv.Canny(blurred, 100, 200)
            line_rgb = cv.cvtColor(edges, cv.COLOR_GRAY2RGB)
            line_rgb = cv.resize(line_rgb, (512, 512))

            result = render_isometric(
                pipe=pipe, device=device,
                line_drawing_rgb=line_rgb,
                image_size=rc["image_size"],
                num_inference_steps=rc["num_inference_steps"],
                guidance_scale=rc["guidance_scale"],
                controlnet_conditioning_scale=rc["controlnet_conditioning_scale"],
                seed=rc["seed"],
            )
            result.save(str(out_dir / "isometric_render.png"))
            outputs["isometric_render"] = True
            complete_stage("render")
        else:
            complete_stage("render")

        update_job(job_id, status="done", outputs=outputs, stage_index=-1)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[JOB {job_id}] ERROR:\n{tb}")
        update_job(job_id, status="error", message=str(e))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename or not allowed(f.filename):
        return jsonify({"error": "Invalid file type. Upload JPG or PNG."}), 400

    render = request.form.get("render", "false").lower() == "true"

    job_id = str(uuid.uuid4())
    safe_name = werkzeug.utils.secure_filename(f.filename)
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    f.save(str(upload_path))

    with jobs_lock:
        jobs[job_id] = {
            "status": "processing",
            "stage_index": 0,
            "current_stage": "ocr",
            "completed_stages": [],
            "outputs": {},
            "message": "",
        }

    t = threading.Thread(target=run_pipeline, args=(job_id, str(upload_path), render), daemon=True)
    t.start()

    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "stage_index": job["stage_index"],
        "current_stage": job["current_stage"],
        "completed_stages": job["completed_stages"],
        "outputs": job["outputs"],
        "message": job.get("message", ""),
    })


@app.route("/api/output/<job_id>/<filename>", methods=["GET"])
def serve_output(job_id: str, filename: str):
    # Security: only allow known output filenames
    allowed_files = {
        "colored_plan.png", "furnished_plan.png",
        "line_drawing.png", "isometric_render.png", "walls_no_text.png"
    }
    if filename not in allowed_files:
        return jsonify({"error": "Not found"}), 404

    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        return jsonify({"error": "File not ready"}), 404

    return send_file(str(path), mimetype="image/png")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
