# Dataset Structure

This directory is excluded from Git due to file size. Below is the complete structure
and format specification so that collaborators and AI agents can write compatible code.

## Folder Structure

```
dataset/
└── {student_no}_{surname}/          # e.g., 25100812_Liang
    ├── G01_call/
    │   ├── clip01/
    │   │   ├── rgb/
    │   │   │   ├── frame_001.png    # 640×480, uint8, 3-channel BGR
    │   │   │   └── ...  (frame_001 to frame_015)
    │   │   ├── depth/
    │   │   │   ├── frame_001.png    # 640×480, uint8, 3-channel BGR (colorized visualization)
    │   │   │   └── ...  (frame_001 to frame_015)
    │   │   ├── depth_raw/
    │   │   │   ├── frame_001.npy   # shape (480, 640), dtype uint16, raw sensor counts
    │   │   │   └── ...  (frame_001 to frame_015)
    │   │   ├── annotation/
    │   │   │   ├── frame_XXX.png   # 640×480, uint8, binary mask: 0=background, 255=hand
    │   │   │   └── frame_YYY.png   # only 2 keyframe masks per clip (frame indices vary per clip)
    │   │   └── depth_metadata.json
    │   ├── clip02/ ...
    │   └── clip05/
    ├── G02_dislike/ ...
    ├── G03_like/
    ├── G04_ok/
    ├── G05_one/
    ├── G06_palm/
    ├── G07_peace/
    ├── G08_rock/
    ├── G09_stop/
    └── G10_three/
```

## Dataset Statistics

| Property            | Value                                  |
|---------------------|----------------------------------------|
| Gestures            | 10 (G01–G10)                           |
| Clips per gesture   | 5 (clip01–clip05)                      |
| Frames per clip     | 15 (captured at 3 fps × 5 sec)         |
| Keyframes annotated | 2 per clip (≥ 1 sec apart)             |
| Total images        | 750 RGB + 750 depth + 750 depth_raw    |
| Total annotations   | 100 binary masks                       |
| Image resolution    | 640 × 480 pixels                       |

## File Format Details

**RGB frame** (`rgb/frame_XXX.png`):
- OpenCV BGR format, 8-bit per channel
- Load: `cv2.imread(path)` → shape `(480, 640, 3)`, dtype `uint8`

**Depth visualization** (`depth/frame_XXX.png`):
- Colorized depth image for visual reference (3-channel BGR, not raw depth values)
- Load: `cv2.imread(path)` → shape `(480, 640, 3)`, dtype `uint8`

**Depth raw** (`depth_raw/frame_XXX.npy`):
- Raw depth values in **sensor integer units** (NOT meters)
- Load: `np.load(path)` → shape `(480, 640)`, dtype `uint16`
- Convert to meters: `depth_meters = depth_raw * depth_scale` (depth_scale ≈ 0.001)
- Convert to mm: `depth_mm = depth_raw * depth_scale * 1000`

**Annotation mask** (`annotation/frame_XXX.png`):
- Binary grayscale image: pixel value `255` = hand region, `0` = background
- Load: `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` → shape `(480, 640)`, dtype `uint8`
- Only 2 keyframes per clip have a corresponding annotation file; the frame indices
  vary per clip (e.g. `frame_004.png` and `frame_011.png` for G01_call/clip01)

**Depth metadata** (`depth_metadata.json`):
```json
{
    "depth_scale": 0.0010000000474974513,
    "unit": "meters per depth unit",
    "description": "Multiply raw depth values by depth_scale to get depth in meters"
}
```
Load example:
```python
import json
with open("depth_metadata.json") as f:
    meta = json.load(f)
depth_scale = meta["depth_scale"]   # ≈ 0.001
depth_meters = depth_raw * depth_scale
```

## Gesture Labels

| ID  | Name    | Description                      |
|-----|---------|----------------------------------|
| G01 | call    | Thumb and pinky extended         |
| G02 | dislike | Thumbs down                      |
| G03 | like    | Thumbs up                        |
| G04 | ok      | Index and thumb forming a circle |
| G05 | one     | Index finger pointing up         |
| G06 | palm    | All fingers extended, palm open  |
| G07 | peace   | Index and middle finger extended |
| G08 | rock    | Index and pinky extended         |
| G09 | stop    | All fingers extended upward      |
| G10 | three   | Three fingers extended           |

## Naming Convention

Frame filenames follow `frame_XXX.png` where `XXX` is zero-padded to 3 digits (e.g., `frame_001.png`).

Image files in `dataset/labelling/images/` are renamed by flattening the path:
```
{gesture}_clip{N}_rgb_frame_{XXX}.png
# e.g., G01_call_clip01_rgb_frame_004.png
```

## Typical Data Loading Pattern

```python
import cv2
import numpy as np
import json

DATASET_ROOT = "dataset/25100812_Liang"
GESTURES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three"
]
GESTURE_LABELS = {g: i for i, g in enumerate(GESTURES)}  # G01_call→0, ..., G10_three→9

# Load a single clip
gesture, clip = "G01_call", "clip01"
clip_dir = f"{DATASET_ROOT}/{gesture}/{clip}"

# RGB
rgb = cv2.imread(f"{clip_dir}/rgb/frame_001.png")          # (480, 640, 3) uint8 BGR

# Depth raw → meters
depth_raw = np.load(f"{clip_dir}/depth_raw/frame_001.npy") # (480, 640) uint16
with open(f"{clip_dir}/depth_metadata.json") as f:
    depth_scale = json.load(f)["depth_scale"]               # ≈ 0.001
depth_meters = depth_raw.astype(np.float32) * depth_scale  # (480, 640) float32 in meters

# Annotation mask (only exists for 2 keyframes per clip)
mask = cv2.imread(f"{clip_dir}/annotation/frame_004.png", cv2.IMREAD_GRAYSCALE)  # (480, 640) uint8
```

## Camera

RealSense D455, captured at 1280×720 and center-cropped to 640×480.
