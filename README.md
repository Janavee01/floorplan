# Floorplan AI

Automatic floorplan analysis — detects walls, segments rooms, places furniture, and generates an isometric 3D render using Stable Diffusion + ControlNet.

Built as a placement project. All processing is fully local (no API keys needed). GPU-accelerated.

---

## Demo

| Input floorplan | Detected rooms | Furnished plan | Isometric render |
|:-:|:-:|:-:|:-:|
| ![input](assets/input.jpg) | ![rooms](assets/debug_rooms.png) | ![furnished](assets/furnished_plan.png) | ![render](assets/isometric_render.png) |

---

## Pipeline

```
floorplan.jpg
    │
    ├─ [OCR]         extract room labels (Tesseract)
    ├─ [Walls]       binarise + strip text noise (adaptive threshold + morphology)
    ├─ [Segment]     distance-transform → connected components → room masks
    ├─ [Classify]    assign room types  (rule-based now → CLIP in Phase 2)
    ├─ [Doors]       detect door arcs via Hough circles → clearance map
    ├─ [Furniture]   place items against walls, respecting doors + clearance
    └─ [Render]      ControlNet MLSD → Stable Diffusion isometric render
```

---

## Quickstart

```bash
# 1. Install system dependency
sudo apt-get install tesseract-ocr

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run (furniture placement only — fast)
python main.py --input floorplan.jpg

# 4. Run with full ControlNet render (needs GPU, ~2 min)
python main.py --input floorplan.jpg --render

# 5. Tests
pytest tests/ -v
```

Outputs land in `results/`:
- `walls_no_text.png` — cleaned wall mask
- `debug_room_core.png` — segmented room cores
- `furnished_plan.png` — furniture placed on floorplan
- `isometric_render.png` — final 3D render (if `--render`)

---

## Project structure

```
floorplan_ai/
├── src/floorplan/
│   ├── ocr.py          # Tesseract label extraction
│   ├── walls.py        # binarisation + noise removal
│   ├── segment.py      # room segmentation + feature extraction
│   ├── classify.py     # room type classifier  ← CLIP upgrade goes here
│   ├── doors.py        # Hough-circle door detection + clearance map
│   ├── furniture.py    # wall-constrained furniture placement
│   └── render.py       # ControlNet isometric render
├── tests/
│   └── test_segment.py
├── config.yaml         # all tunable parameters
├── main.py             # CLI entry point
└── requirements.txt
```

---

## Configuration

All thresholds live in `config.yaml` — no magic numbers buried in code.

```yaml
segmentation:
  corridor_thresh: 12    # px — narrows below this become corridor cuts
  min_room_area: 1500    # px² — ignore smaller blobs

furniture:
  sofa: [140, 60]        # width × height in pixels
  bed:  [140, 80]
```

---

## Roadmap

- [x] Phase 1 — modular package, CLI, config, tests
- [ ] Phase 2 — CLIP zero-shot room classifier, SAM segmentation, scale calibration
- [ ] Phase 3 — Streamlit web app with interactive room editor + PDF export

---

## Tech stack

`OpenCV` · `Tesseract OCR` · `PyTorch` · `Stable Diffusion` · `ControlNet MLSD` · `Diffusers` · `pytest`
