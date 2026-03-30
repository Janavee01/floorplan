"""
tests/test_segment.py — unit tests for wall extraction and room segmentation.
Run with: pytest tests/ -v
"""
import numpy as np
import pytest
from src.floorplan.walls   import extract_walls
from src.floorplan.segment import segment_rooms, extract_room_features

def make_simple_floorplan(h=300, w=300, wall_thickness=10):
    img = np.ones((h, w), dtype=np.uint8) * 255
    img[:wall_thickness, :] = 0
    img[-wall_thickness:, :] = 0
    img[:, :wall_thickness] = 0
    img[:, -wall_thickness:] = 0
    mid = w // 2
    img[:, mid - wall_thickness // 2: mid + wall_thickness // 2] = 0
    return img

class TestExtractWalls:
    def test_returns_binary(self):
        result = extract_walls(make_simple_floorplan())
        assert set(result.flatten().tolist()).issubset({0, 255})

    def test_shape_preserved(self):
        result = extract_walls(make_simple_floorplan(400, 500))
        assert result.shape == (400, 500)

    def test_walls_detected(self):
        result = extract_walls(make_simple_floorplan())
        assert result.max() == 255

class TestSegmentRooms:
    def test_rooms_detected(self):
        walls = extract_walls(make_simple_floorplan())
        rooms, _, _ = segment_rooms(walls, corridor_thresh=5, min_room_area=500)
        assert len(rooms) >= 1

    def test_masks_are_binary(self):
        walls = extract_walls(make_simple_floorplan())
        rooms, _, _ = segment_rooms(walls, corridor_thresh=5, min_room_area=500)
        for mask in rooms:
            assert set(mask.flatten().tolist()).issubset({0, 255})

    def test_masks_correct_shape(self):
        walls = extract_walls(make_simple_floorplan(200, 200))
        rooms, _, _ = segment_rooms(walls, corridor_thresh=5, min_room_area=100)
        for mask in rooms:
            assert mask.shape == (200, 200)

class TestExtractRoomFeatures:
    def test_required_keys(self):
        walls = extract_walls(make_simple_floorplan())
        rooms, _, _ = segment_rooms(walls, corridor_thresh=5, min_room_area=500)
        data = extract_room_features(rooms)
        required = {"id", "area", "centroid", "bbox", "aspect_ratio", "compactness", "rectangularity"}
        for r in data:
            assert required.issubset(r.keys())

    def test_area_positive(self):
        walls = extract_walls(make_simple_floorplan())
        rooms, _, _ = segment_rooms(walls, corridor_thresh=5, min_room_area=500)
        for r in extract_room_features(rooms):
            assert r["area"] > 0
