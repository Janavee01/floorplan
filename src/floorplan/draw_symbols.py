"""
draw_symbols.py — draw standard architectural furniture symbols as line art.
"""
from __future__ import annotations
import cv2
import numpy as np


def _draw_bed(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    headboard_h = max(6, h // 4)
    cv2.rectangle(img, (x, y), (x+w, y+headboard_h), 0, -1)
    cx = x + w // 2
    cv2.line(img, (cx, y+headboard_h), (cx, y+h), 0, 2)


def _draw_sofa(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    back_h = max(6, h // 3)
    cv2.rectangle(img, (x, y), (x+w, y+back_h), 0, -1)
    arm_w = max(4, w // 7)
    cv2.rectangle(img, (x, y+back_h), (x+arm_w, y+h), 0, -1)
    cv2.rectangle(img, (x+w-arm_w, y+back_h), (x+w, y+h), 0, -1)


def _draw_table(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    cv2.line(img, (x, y), (x+w, y+h), 0, 2)
    cv2.line(img, (x+w, y), (x, y+h), 0, 2)


def _draw_wc(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    tank_h = max(5, h // 3)
    cv2.rectangle(img, (x, y), (x+w, y+tank_h), 0, -1)
    bowl_cx = x + w // 2
    bowl_cy = y + tank_h + (h - tank_h) // 2
    bowl_rx = max(4, w // 2 - 3)
    bowl_ry = max(4, (h - tank_h) // 2 - 2)
    cv2.ellipse(img, (bowl_cx, bowl_cy), (bowl_rx, bowl_ry), 0, 0, 360, 0, 2)


def _draw_sink(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    cx = x + w // 2
    cy = y + h // 2
    r = max(4, min(w, h) // 3)
    cv2.circle(img, (cx, cy), r, 0, 2)
    cv2.circle(img, (cx, y + 4), 3, 0, -1)


def _draw_wardrobe(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    third = w // 3
    cv2.line(img, (x+third,   y), (x+third,   y+h), 0, 2)
    cv2.line(img, (x+2*third, y), (x+2*third, y+h), 0, 2)
    for door_x in [x + third//2, x + third + third//2, x + 2*third + third//2]:
        handle_y = y + h // 2
        cv2.line(img, (door_x-4, handle_y), (door_x+4, handle_y), 0, 2)


def _draw_counter(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    edge_h = max(4, h // 3)
    cv2.line(img, (x, y+edge_h), (x+w, y+edge_h), 0, 2)
    sink_r = max(4, min(w//5, h//3))
    cv2.circle(img, (x + w//4, y + h//2 + edge_h//2), sink_r, 0, 2)


def _draw_washing_machine(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, 4)
    cx = x + w // 2
    cy = y + h // 2
    r = max(5, min(w, h) // 2 - 4)
    cv2.circle(img, (cx, cy), r, 0, 2)
    cv2.circle(img, (cx, cy), max(2, r-4), 0, 2)


SYMBOL_DRAWERS = {
    "bed":             _draw_bed,
    "sofa":            _draw_sofa,
    "table":           _draw_table,
    "wc":              _draw_wc,
    "sink":            _draw_sink,
    "wardrobe":        _draw_wardrobe,
    "counter":         _draw_counter,
    "washing_machine": _draw_washing_machine,
}


def draw_architectural_plan(
    wall_mask: np.ndarray,
    rooms: list[dict],
    furniture_by_room: list[dict],
    target_size: int = 512,
) -> np.ndarray:
    h, w = wall_mask.shape

    # White background
    drawing = np.ones((h, w), dtype=np.uint8) * 255

    # Draw walls — thin them down so furniture is relatively bolder
    # Erode wall mask slightly so walls are 2-3px not 10px
    kernel = np.ones((2, 2), np.uint8)
    thinned_walls = cv2.erode(wall_mask, kernel, iterations=1)
    drawing[thinned_walls == 255] = 0
    # Draw furniture symbols with thickness=2
    for entry in furniture_by_room:
        items = entry.get("items", {})
        for name, rect in items.items():
            if rect is None:
                continue
            fx, fy, fw, fh = rect
            drawer = SYMBOL_DRAWERS.get(name)
            if drawer:
                drawer(drawing, fx, fy, fw, fh)

    # Resize to target size
    resized = cv2.resize(drawing, (target_size, target_size),
                         interpolation=cv2.INTER_NEAREST)

    return resized


def plan_to_rgb_control(line_drawing: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(line_drawing, cv2.COLOR_GRAY2RGB)