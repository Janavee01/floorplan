"""
furniture.py — place furniture inside detected rooms.

Placement method
----------------
- Furniture sizes scale relative to each room's bounding box
- Items are placed one at a time; after each placement the occupied
  pixels are REMOVED from the available room mask so the next item
  cannot overlap the previous one
- Prefers positions that touch the longest available wall segment
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Furniture size ratios — fraction of room bounding box (w, h)
# ---------------------------------------------------------------------------

FURNITURE_RATIOS = {
    "bed":             (0.50, 0.32),
    "wardrobe":        (0.18, 0.40),
    "sofa":            (0.48, 0.16),
    "table":           (0.28, 0.22),
    "wc":              (0.25, 0.30),    
    "sink":            (0.22, 0.20),    
    "counter":         (0.60, 0.22),    
    "washing_machine": (0.22, 0.22),
}

ROOM_FURNITURE = {
    "bedroom":     ["bed", "wardrobe"],
    "living_room": ["sofa", "table"],
    "kitchen":     ["counter", "table"],
    "bathroom":    ["wc", "sink"],
    "dining_room": ["table"],
    "utility":     ["washing_machine"],
}

ROOM_COLORS = {
    "living_room": (0,   180,  60),
    "bedroom":     (60,  100, 255),
    "kitchen":     (255, 160,   0),
    "bathroom":    (0,   160, 255),
    "dining_room": (180,  60, 255),
    "utility":     (180, 180,   0),
}


# ---------------------------------------------------------------------------
# Core placement
# ---------------------------------------------------------------------------

def _furniture_size(item_name: str, room_bbox: tuple) -> tuple[int, int]:
    _, _, rw, rh = room_bbox
    wr, hr = FURNITURE_RATIOS.get(item_name, (0.3, 0.3))
    return max(20, int(rw * wr)), max(15, int(rh * hr))


def place_against_wall(
    available_mask: np.ndarray,
    wall_mask: np.ndarray,
    fw: int,
    fh: int,
    door_mask: np.ndarray | None = None,
    clearance_map: np.ndarray | None = None,
    step: int = 4,
) -> tuple | None:
    """
    Find the best (x, y, w, h) for a (fw x fh) rectangle inside
    available_mask — pixels already occupied by furniture have been
    removed from available_mask by the caller before this runs.

    Scoring:
      - +1000 if rectangle edge touches a wall
      - +1    if rectangle fits inside available_mask
      Best score wins.
    """
    h_img, w_img = available_mask.shape
    best, best_score = None, -1

    ys, xs = np.where(available_mask > 0)
    if len(xs) == 0:
        return None

    stride = max(step, int(np.sqrt(len(xs)) // 15))

    for x, y in zip(xs[::stride], ys[::stride]):
        x1, y1 = x - fw // 2, y - fh // 2
        x2, y2 = x1 + fw,     y1 + fh

        if x1 < 0 or y1 < 0 or x2 >= w_img or y2 >= h_img:
            continue

        # Must fit entirely inside available (room minus already placed)
        if np.any(available_mask[y1:y2, x1:x2] == 0):
            continue

        # Must not overlap door or clearance
        if door_mask is not None and np.any(door_mask[y1:y2, x1:x2] > 0):
            continue
        if clearance_map is not None and np.any(clearance_map[y1:y2, x1:x2] > 0):
            continue

        edge = wall_mask[y1:y2, x1:x2]
        touches_wall = (
            np.any(edge[0, :]) or np.any(edge[-1, :]) or
            np.any(edge[:, 0]) or np.any(edge[:, -1])
        )
        score = 1000 if touches_wall else 1

        if score > best_score:
            best_score = score
            best = (x1, y1, fw, fh)

    return best


def largest_inner_rect(mask: np.ndarray) -> tuple:
    """Largest axis-aligned rectangle fully inside mask (for tables)."""
    h, w = mask.shape
    hist  = [0] * w
    best, best_area = (0, 0, 1, 1), 0

    for y in range(h):
        for x in range(w):
            hist[x] = hist[x] + 1 if mask[y, x] else 0
        stack, xi = [], 0
        while xi <= w:
            cur = hist[xi] if xi < w else 0
            if not stack or cur >= hist[stack[-1]]:
                stack.append(xi); xi += 1
            else:
                top   = stack.pop()
                width = xi if not stack else xi - stack[-1] - 1
                area  = hist[top] * width
                if area > best_area:
                    best_area = area
                    best = (xi - width, y - hist[top] + 1, width, hist[top])
    return best


# ---------------------------------------------------------------------------
# Per-room placement — items placed sequentially, each shrinks available mask
# ---------------------------------------------------------------------------

def place_room_furniture(
    room: dict,
    wall_mask: np.ndarray,
    door_mask: np.ndarray | None,
    clearance_map: np.ndarray | None,
) -> dict[str, tuple | None]:
    rtype  = room.get("type", "bedroom")
    items  = ROOM_FURNITURE.get(rtype, ["bed"])
    bbox   = room["bbox"]

    # Start with the full room mask as available space
    available = room["mask"].copy()
    results   = {}

    for item_name in items:
        fw, fh = _furniture_size(item_name, bbox)

        if item_name == "table":
            # Table goes in the largest remaining open space
            rect = largest_inner_rect(available)
            if rect != (0, 0, 1, 1):
                x, y, rw, rh = rect
                _, _, bbw, bbh = bbox
                tw = max(20, int(bbw * 0.25))
                th = max(15, int(bbh * 0.20))
                rect = (x, y, tw, th)
            else:
                rect = None
        else:
            rect = place_against_wall(
                available, wall_mask, fw, fh,
                door_mask, clearance_map
            )

        results[item_name] = rect

        # ── KEY FIX: remove placed furniture pixels from available mask ──
        # Next item cannot overlap this one because those pixels are gone
        if rect is not None:
            x, y, w, h = rect
            pad = 4   # small clearance gap between furniture items
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(available.shape[1], x + w + pad)
            y2 = min(available.shape[0], y + h + pad)
            available[y1:y2, x1:x2] = 0

    return results


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_furniture(image: np.ndarray, items: dict, color: tuple) -> None:
    for name, rect in items.items():
        if rect is None:
            continue
        x, y, w, h = rect
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 1)
        font_scale = max(0.28, min(0.45, w / 130))
        cv2.putText(image, name, (x+3, y + h//2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1)


def furnish_all(
    rooms: list[dict],
    wall_mask: np.ndarray,
    door_mask: np.ndarray | None,
    clearance_map: np.ndarray | None,
    cfg: dict,
    original_image: np.ndarray | None = None,
) -> np.ndarray:
    out = original_image.copy() if original_image is not None \
          else cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)

    for room in rooms:
        rtype = room.get("type", "bedroom")
        color = ROOM_COLORS.get(rtype, (180, 180, 180))
        items = place_room_furniture(room, wall_mask, door_mask, clearance_map)
        draw_furniture(out, items, color)
        cx, cy = room["centroid"]
        cv2.putText(out, rtype.replace("_", " "),
                    (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1)

    return out