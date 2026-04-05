"""
draw_symbols.py — draw standard architectural furniture symbols as line art.

Each symbol is drawn as black lines on a white background using the exact
positions calculated by furniture.py. This gives ControlNet MLSD clean
architectural line drawings it was trained on — producing accurate 3D renders
with furniture in the correct positions.

Symbol conventions (standard architectural drawing):
  bed       — rectangle + headboard line at top + center vertical line
  sofa      — rectangle + back rest rectangle along one edge
  table     — rectangle + X diagonal inside
  wc        — rectangle + oval inside
  sink      — rectangle + circle inside
  wardrobe  — rectangle + 2 vertical divider lines
  counter   — rectangle + horizontal line inside
  washing_machine — rectangle + large circle inside
"""
from __future__ import annotations
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual symbol drawers
# ---------------------------------------------------------------------------

def _draw_bed(img, x, y, w, h, thickness=1):
    """Rectangle + headboard (top 20%) + center vertical line."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    # Headboard
    headboard_h = max(4, h // 5)
    cv2.rectangle(img, (x, y), (x+w, y+headboard_h), 0, -1)
    # Center vertical line (sheet fold)
    cx = x + w // 2
    cv2.line(img, (cx, y+headboard_h), (cx, y+h), 0, thickness)


def _draw_sofa(img, x, y, w, h, thickness=1):
    """Rectangle + back rest along top edge + two arm rests."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    # Back rest (top 25%)
    back_h = max(4, h // 4)
    cv2.rectangle(img, (x, y), (x+w, y+back_h), 0, -1)
    # Arm rests (left and right, bottom 75%)
    arm_w = max(3, w // 8)
    cv2.rectangle(img, (x, y+back_h), (x+arm_w, y+h), 0, -1)
    cv2.rectangle(img, (x+w-arm_w, y+back_h), (x+w, y+h), 0, -1)


def _draw_table(img, x, y, w, h, thickness=1):
    """Rectangle + X diagonal inside."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    cv2.line(img, (x, y), (x+w, y+h), 0, thickness)
    cv2.line(img, (x+w, y), (x, y+h), 0, thickness)


def _draw_wc(img, x, y, w, h, thickness=1):
    """Rectangle + oval inside (toilet bowl)."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    # Tank (top 30%)
    tank_h = max(4, h // 3)
    cv2.rectangle(img, (x, y), (x+w, y+tank_h), 0, -1)
    # Bowl (oval in bottom 70%)
    bowl_cx = x + w // 2
    bowl_cy = y + tank_h + (h - tank_h) // 2
    bowl_rx = max(4, w // 2 - 3)
    bowl_ry = max(4, (h - tank_h) // 2 - 3)
    cv2.ellipse(img, (bowl_cx, bowl_cy), (bowl_rx, bowl_ry), 0, 0, 360, 0, thickness)


def _draw_sink(img, x, y, w, h, thickness=1):
    """Rectangle + circle inside (basin)."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    cx = x + w // 2
    cy = y + h // 2
    r  = max(3, min(w, h) // 3)
    cv2.circle(img, (cx, cy), r, 0, thickness)
    # Tap dot
    cv2.circle(img, (cx, y + 4), 2, 0, -1)


def _draw_wardrobe(img, x, y, w, h, thickness=1):
    """Rectangle + 2 vertical dividers + door handles."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    # Two door dividers
    third = w // 3
    cv2.line(img, (x+third,   y), (x+third,   y+h), 0, thickness)
    cv2.line(img, (x+2*third, y), (x+2*third, y+h), 0, thickness)
    # Door handles (small horizontal lines near center of each door)
    for door_x in [x + third//2, x + third + third//2, x + 2*third + third//2]:
        handle_y = y + h // 2
        cv2.line(img, (door_x-3, handle_y), (door_x+3, handle_y), 0, thickness+1)


def _draw_counter(img, x, y, w, h, thickness=1):
    """Rectangle + horizontal line (counter edge) + sink circle."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    # Counter edge line
    edge_h = max(3, h // 4)
    cv2.line(img, (x, y+edge_h), (x+w, y+edge_h), 0, thickness)
    # Sink on counter
    sink_r = max(3, min(w//6, h//3))
    cv2.circle(img, (x + w//4, y + h//2 + edge_h//2), sink_r, 0, thickness)


def _draw_washing_machine(img, x, y, w, h, thickness=1):
    """Rectangle + large circle (drum) inside."""
    cv2.rectangle(img, (x, y), (x+w, y+h), 0, thickness)
    cx = x + w // 2
    cy = y + h // 2
    r  = max(4, min(w, h) // 2 - 4)
    cv2.circle(img, (cx, cy), r, 0, thickness)
    # Inner circle (door seal)
    cv2.circle(img, (cx, cy), max(2, r-3), 0, thickness)


# Map furniture name → drawing function
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_architectural_plan(
    wall_mask: np.ndarray,
    rooms: list[dict],
    furniture_by_room: list[dict],
    target_size: int = 512,
) -> np.ndarray:
    """
    Draw a clean architectural line drawing:
      - White background
      - Black wall lines (from wall_mask)
      - Furniture symbols at exact placement positions

    Parameters
    ----------
    wall_mask          : binary wall image (255=wall)
    rooms              : room list from segment_rooms()
    furniture_by_room  : list of {room_index: int, items: {name: (x,y,w,h)}}
    target_size        : output image size for ControlNet (512 or 768)

    Returns
    -------
    line_drawing : grayscale image, white bg + black lines, ready for ControlNet
    """
    h, w = wall_mask.shape

    # White background
    drawing = np.ones((h, w), dtype=np.uint8) * 255

    # Draw walls as black lines
    drawing[wall_mask == 255] = 0

    # Draw furniture symbols
    for entry in furniture_by_room:
        items = entry.get("items", {})
        for name, rect in items.items():
            if rect is None:
                continue
            fx, fy, fw, fh = rect
            drawer = SYMBOL_DRAWERS.get(name)
            if drawer:
                drawer(drawing, fx, fy, fw, fh, thickness=1)

    # Resize to target size for ControlNet
    resized = cv2.resize(drawing, (target_size, target_size),
                         interpolation=cv2.INTER_NEAREST)

    return resized


def plan_to_rgb_control(line_drawing: np.ndarray) -> np.ndarray:
    """Convert grayscale line drawing to RGB for ControlNet input."""
    return cv2.cvtColor(line_drawing, cv2.COLOR_GRAY2RGB)