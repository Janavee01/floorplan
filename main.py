"""
main.py — CLI entry point for the floorplan AI pipeline.

Usage
-----
    python main.py --input floorplan.jpg
    python main.py --input floorplan.jpg --render
"""
import argparse, sys
from pathlib import Path
import cv2, yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def run(input_path, config, render):
    from src.floorplan.ocr       import extract_labels
    from src.floorplan.walls     import extract_walls
    from src.floorplan.doors     import detect_and_remove_doors, build_clearance_map
    from src.floorplan.segment   import segment_rooms, extract_room_features
    from src.floorplan.classify  import classify_rooms
    from src.floorplan.furniture import furnish_all

    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    debug = config["output"]["save_debug"]

    # 1. Load
    color = cv2.imread(input_path)
    if color is None:
        print(f"[ERROR] Cannot read: {input_path}"); sys.exit(1)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    print(f"[1/6] Loaded {input_path} ({color.shape[1]}x{color.shape[0]})")

    # 2. OCR — read text labels BEFORE anything else
    ocr_labels = extract_labels(color, min_length=config["ocr"]["min_text_length"])
    print(f"[2/6] OCR — {len(ocr_labels)} text regions: "
          f"{[l['text'] for l in ocr_labels]}")

    # 3. Wall extraction
    w = config["walls"]
    wall_mask = extract_walls(gray,
        adaptive_block_size=w["adaptive_block_size"], adaptive_C=w["adaptive_C"],
        morph_kernel=tuple(w["morph_kernel"]), min_component_area=w["min_component_area"],
        min_wall_area=w["min_wall_area"], min_wall_thickness=w["min_wall_thickness"])
    if debug: cv2.imwrite(str(out_dir / "walls_no_text.png"), wall_mask)
    print("[3/6] Wall extraction done")

    # 4. Door detection + removal
    d = config.get("doors", {})
    clean_walls, door_mask = detect_and_remove_doors(
        wall_mask,
        min_radius   = d.get("min_radius",   15),
        max_radius   = d.get("max_radius",   60),
        hough_param2 = d.get("hough_param2", 18),
    )
    clearance_map = build_clearance_map(door_mask, d.get("clearance_px", 80))
    n_doors = int(cv2.countNonZero(door_mask) > 0)
    print(f"[4/6] Door detection — {'doors found, gaps sealed' if n_doors else 'no doors detected'}")
    if debug: cv2.imwrite(str(out_dir / "debug_doors.png"), door_mask)

    # 5. Room segmentation — corridor_thresh / min_room_area = None means auto
    s = config["segmentation"]
    corridor_thresh = s["corridor_thresh"]   # None or explicit int from config
    min_room_area   = s["min_room_area"]     # None or explicit int from config

    rooms, room_core, dist = segment_rooms(
        clean_walls,
        corridor_thresh = corridor_thresh,
        min_room_area   = min_room_area,
        morph_kernel    = tuple(s["morph_kernel"]),
    )
    rooms_data = extract_room_features(rooms)
    if debug: cv2.imwrite(str(out_dir / "debug_room_core.png"), room_core)
    print(f"[5/6] Segmentation — {len(rooms_data)} rooms")

    # 6. Classify
    rooms_data = classify_rooms(rooms_data, ocr_labels, color)
    for r in rooms_data:
        src = r.get("label_source", "?")
        print(f"      Room {r['id']:>2}  area={r['area']:>7}  -> {r['type']:<14}  [{src}]")

    # 7. Furniture placement
    furnished = furnish_all(rooms_data, clean_walls, door_mask, clearance_map, config["furniture"])
    out_path  = out_dir / "furnished_plan.png"
    cv2.imwrite(str(out_path), furnished)
    print(f"[6/6] Furniture placed -> {out_path}")

    # 8. Optional ControlNet render
    if render:
        from src.floorplan.render import load_pipeline, render_isometric
        r = config["render"]
        print("[7/7] Loading ControlNet ...")
        pipe, device = load_pipeline()
        result = render_isometric(furnished, pipe, device,
            image_size=r["image_size"], num_inference_steps=r["num_inference_steps"],
            guidance_scale=r["guidance_scale"],
            controlnet_conditioning_scale=r["controlnet_conditioning_scale"],
            seed=r["seed"])
        rpath = out_dir / "isometric_render.png"
        result.save(str(rpath))
        print(f"[7/7] Render saved -> {rpath}")
    else:
        print("[7/7] Skipped render (use --render to enable)")

    print(f"\nDone. Outputs in: {out_dir.resolve()}")

def main():
    p = argparse.ArgumentParser(description="Floorplan AI")
    p.add_argument("--input",  required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--render", action="store_true", default=False)
    args = p.parse_args()
    run(args.input, load_config(args.config), args.render)

if __name__ == "__main__":
    main()