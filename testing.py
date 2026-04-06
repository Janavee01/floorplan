import cv2, yaml, numpy as np
from src.floorplan.ocr import extract_labels
from src.floorplan.walls import extract_walls
from src.floorplan.segment import segment_rooms
from src.floorplan.furniture import place_room_furniture

config = yaml.safe_load(open("config.yaml"))
color = cv2.imread("fp.jpg")
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
w = config["walls"]
wall_mask = extract_walls(gray, adaptive_block_size=w["adaptive_block_size"],
    adaptive_C=w["adaptive_C"], morph_kernel=tuple(w["morph_kernel"]),
    min_component_area=w["min_component_area"], min_wall_area=w["min_wall_area"],
    min_wall_thickness=w["min_wall_thickness"])
ocr_labels = extract_labels(color, min_length=config["ocr"]["min_text_length"])
rooms = segment_rooms(wall_mask, ocr_labels)

for room in rooms:
    items = place_room_furniture(room, wall_mask, None, None)
    print(f"\n{room['type']}:")
    for name, rect in items.items():
        print(f"  {name}: {rect}")