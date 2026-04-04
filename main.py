"""
main.py — CLI entry point for the floorplan AI pipeline.
"""
import argparse, sys
import numpy as np
from pathlib import Path
import cv2, yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def run(input_path, config, render):
    from src.floorplan.ocr       import extract_labels
    from src.floorplan.walls     import extract_walls
    from src.floorplan.segment   import segment_rooms
    from src.floorplan.classify  import colorize_rooms
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

    # 2. OCR
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

    # 4. Segment
    rooms = segment_rooms(wall_mask, ocr_labels)
    print(f"[4/6] Segmentation — {len(rooms)} rooms")
    for r in rooms:
        print(f"      {r['type']:<14} area={r['area']:>7}  label='{r['label_text']}'")

    # 5. Colorize
    colored = colorize_rooms(color, rooms)
    cv2.imwrite(str(out_dir / "colored_plan.png"), colored)
    print(f"[5/6] Colored plan saved -> {out_dir / 'colored_plan.png'}")

    # 6. Furniture placement — draw on colored plan, no door_mask needed
    furnished = furnish_all(
        rooms, wall_mask,
        door_mask=None, clearance_map=None,
        cfg=config["furniture"],
        original_image=colored,
    )
    furnished_path = out_dir / "furnished_plan.png"
    cv2.imwrite(str(furnished_path), furnished)
    print(f"[6/6] Furniture placed -> {furnished_path}")

    # 7. Optional ControlNet render
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