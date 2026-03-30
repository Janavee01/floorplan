"""
furniture.py — place furniture inside detected rooms,
               respecting walls, doors, and clearance zones.
"""

import cv2
import numpy as np


def place_against_wall(
    room_mask, wall_mask, door_mask, clearance_map,
    size=(80, 40), step=5,
):
    fw, fh = size
    h_img, w_img = room_mask.shape
    best, best_score = None, -1

    ys, xs = np.where(room_mask > 0)
    candidates = list(zip(xs.tolist(), ys.tolist()))
    stride = max(step, int(np.sqrt(len(candidates)) // 20))

    for x, y in candidates[::stride]:
        x1, y1 = x - fw // 2, y - fh // 2
        x2, y2 = x1 + fw, y1 + fh
        if x1 < 0 or y1 < 0 or x2 >= w_img or y2 >= h_img:
            continue
        if np.any(room_mask[y1:y2, x1:x2] == 0):
            continue
        if np.any(door_mask[y1:y2, x1:x2] > 0):
            continue
        if np.any(clearance_map[y1:y2, x1:x2] > 0):
            continue

        edge = wall_mask[y1:y2, x1:x2]
        touches_wall = (
            np.any(edge[0, :]) or np.any(edge[-1, :]) or
            np.any(edge[:, 0]) or np.any(edge[:, -1])
        )
        score = 1000 if touches_wall else 0
        if score > best_score:
            best_score = score
            best = (x1, y1, fw, fh)

    return best


def largest_inner_rect(mask):
    h, w = mask.shape
    hist = [0] * w
    best, best_area = (0, 0, 0, 0), 0
    for y in range(h):
        for x in range(w):
            hist[x] = hist[x] + 1 if mask[y, x] else 0
        stack, xi = [], 0
        while xi <= w:
            cur = hist[xi] if xi < w else 0
            if not stack or cur >= hist[stack[-1]]:
                stack.append(xi); xi += 1
            else:
                top = stack.pop()
                width = xi if not stack else xi - stack[-1] - 1
                area = hist[top] * width
                if area > best_area:
                    best_area = area
                    best = (xi - width, y - hist[top] + 1, width, hist[top])
    return best


def place_living(room_mask, wall_mask, door_mask, clearance_map, cfg):
    return {
        "sofa":  place_against_wall(room_mask, wall_mask, door_mask, clearance_map, size=tuple(cfg["sofa"])),
        "table": largest_inner_rect(room_mask),
    }

def place_bedroom(room_mask, wall_mask, door_mask, clearance_map, cfg):
    return {
        "bed":      place_against_wall(room_mask, wall_mask, door_mask, clearance_map, size=tuple(cfg["bed"])),
        "wardrobe": place_against_wall(room_mask, wall_mask, door_mask, clearance_map, size=tuple(cfg["wardrobe"])),
    }

def place_bathroom(room_mask, wall_mask, door_mask, clearance_map, cfg):
    return {
        "wc":   place_against_wall(room_mask, wall_mask, door_mask, clearance_map, size=tuple(cfg["wc"])),
        "sink": place_against_wall(room_mask, wall_mask, door_mask, clearance_map, size=tuple(cfg["sink"])),
    }

def place_utility(room_mask, wall_mask, door_mask, clearance_map, cfg):
    return {
        "washing_machine": place_against_wall(room_mask, wall_mask, door_mask, clearance_map, size=tuple(cfg["washing_machine"])),
    }

PLACERS = {
    "living_room": place_living,
    "bedroom":     place_bedroom,
    "kitchen":     place_bedroom,
    "bathroom":    place_bathroom,
    "utility":     place_utility,
}

ROOM_COLORS = {
    "living_room": (0,   200,  80),
    "bedroom":     (80,  120, 255),
    "kitchen":     (255, 180,   0),
    "bathroom":    (0,   180, 255),
    "utility":     (200, 200,   0),
}

def draw_furniture(image, items, color):
    for name, rect in items.items():
        if rect is None:
            continue
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, name, (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def furnish_all(rooms_data, wall_mask, door_mask, clearance_map, cfg):
    out = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)
    for room in rooms_data:
        rtype  = room.get("type", "bedroom")
        color  = ROOM_COLORS.get(rtype, (180, 180, 180))
        placer = PLACERS.get(rtype, place_bedroom)
        items  = placer(room["mask"], wall_mask, door_mask, clearance_map, cfg)
        draw_furniture(out, items, color)
        cx, cy = room["centroid"]
        cv2.putText(out, rtype, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
