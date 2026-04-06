import cv2
import yaml
import numpy as np
from src.floorplan.ocr import extract_labels
from src.floorplan.walls import extract_walls
from src.floorplan.segment import segment_rooms
from src.floorplan.furniture import place_room_furniture

# Load your existing config
config = yaml.safe_load(open("config.yaml"))
input_img = "fp.jpg" # Make sure this matches your filename

color = cv2.imread(input_img)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# Standard pipeline steps to get the wall mask and rooms
w = config["walls"]
wall_mask = extract_walls(gray, 
    adaptive_block_size=w["adaptive_block_size"], 
    adaptive_C=w["adaptive_C"], 
    morph_kernel=tuple(w["morph_kernel"]), 
    min_component_area=w["min_component_area"], 
    min_wall_area=w["min_wall_area"], 
    min_wall_thickness=w["min_wall_thickness"])

ocr_labels = extract_labels(color, min_length=config["ocr"]["min_text_length"])
rooms = segment_rooms(wall_mask, ocr_labels)

# The loop you need to run
for room in rooms:
    items = place_room_furniture(room, wall_mask, None, None)
    print(f"\n--- {room['type'].upper()} ---")
    if not items:
        print("  No furniture placed.")
    for name, rect in items.items():
        if rect is None:
            print(f"  [MISSING] {name}: Returned None")
        else:
            print(f"  [PLACED] {name}: {rect}")