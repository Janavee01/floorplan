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
    from src.floorplan.ocr          import extract_labels
    from src.floorplan.walls        import extract_walls
    from src.floorplan.segment      import segment_rooms
    from src.floorplan.classify     import colorize_rooms
    from src.floorplan.furniture    import furnish_all, place_room_furniture
    from src.floorplan.draw_symbols import draw_architectural_plan, plan_to_rgb_control

    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    debug = config["output"]["save_debug"]

    # 1. Load
    color = cv2.imread(input_path)
    if color is None:
        print(f"[ERROR] Cannot read: {input_path}"); sys.exit(1)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    print(f"[1/7] Loaded {input_path} ({color.shape[1]}x{color.shape[0]})")

    # 2. OCR
    ocr_labels = extract_labels(color, min_length=config["ocr"]["min_text_length"])
    print(f"[2/7] OCR — {len(ocr_labels)} text regions: "
          f"{[l['text'] for l in ocr_labels]}")

    # 3. Wall extraction
    w = config["walls"]
    wall_mask = extract_walls(gray,
        adaptive_block_size=w["adaptive_block_size"], adaptive_C=w["adaptive_C"],
        morph_kernel=tuple(w["morph_kernel"]), min_component_area=w["min_component_area"],
        min_wall_area=w["min_wall_area"], min_wall_thickness=w["min_wall_thickness"])
    if debug: cv2.imwrite(str(out_dir / "walls_no_text.png"), wall_mask)
    print("[3/7] Wall extraction done")

    # 4. Segment
    rooms = segment_rooms(wall_mask, ocr_labels)
    print(f"[4/7] Segmentation — {len(rooms)} rooms")
    for r in rooms:
        print(f"      {r['type']:<14} area={r['area']:>7}  label='{r['label_text']}'")

    # 5. Colorize
    colored = colorize_rooms(color, rooms)
    cv2.imwrite(str(out_dir / "colored_plan.png"), colored)
    print(f"[5/7] Colored plan saved")

    # 6. Furniture placement
    # Place furniture and collect positions for symbol drawing
    furniture_by_room = []
    for room in rooms:
        items = place_room_furniture(room, wall_mask, None, None)
        furniture_by_room.append({"room": room, "items": items})

    # Draw colored furnished plan (for display)
    furnished = furnish_all(
        rooms, wall_mask,
        door_mask=None, clearance_map=None,
        cfg=config["furniture"],
        original_image=colored,
    )
    cv2.imwrite(str(out_dir / "furnished_plan.png"), furnished)
    print(f"[6/7] Furniture placed")

    print("[7/7] Building architectural line drawing ...")

    line_drawing = draw_architectural_plan(
        wall_mask=wall_mask,
        rooms=rooms,
        furniture_by_room=furniture_by_room,
        target_size=config["render"]["image_size"],
    )

    cv2.imwrite(str(out_dir / "line_drawing.png"), line_drawing)
    print(f"      Line drawing saved -> {out_dir / 'line_drawing.png'}")

    # 7. Optional ControlNet render
    if render:
        from src.floorplan.render import load_pipeline, render_isometric
        rc = config["render"]
        print("[7/7] Building architectural line drawing ...")

        # Draw walls + furniture symbols as clean line art

        line_rgb = plan_to_rgb_control(line_drawing)

        print("[7/7] Loading ControlNet ...")
        pipe, device = load_pipeline()
        result = render_isometric(
            pipe=pipe,
            device=device,
            line_drawing_rgb=line_rgb,
            image_size=rc["image_size"],
            num_inference_steps=rc["num_inference_steps"],
            guidance_scale=rc["guidance_scale"],
            controlnet_conditioning_scale=rc["controlnet_conditioning_scale"],
            seed=rc["seed"],
        )
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