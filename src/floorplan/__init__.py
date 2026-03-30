"""
floorplan_ai — automatic floorplan analysis and furniture placement.

Modules
-------
ocr         : extract room labels from the raw floorplan image
walls       : binarise walls and remove text noise
segment     : segment free-space into individual rooms
classify    : label each room (rule-based now, CLIP in Phase 2)
furniture   : place furniture respecting walls and clearance
render      : generate isometric 3-D render via ControlNet
"""

__version__ = "0.1.0"
