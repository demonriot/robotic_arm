
from pathlib import Path
import pickle
import re

import cv2
import numpy as np


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

LEFT_DIR = ROOT / "data" / "calibration" / "left"
RIGHT_DIR = ROOT / "data" / "calibration" / "right"

PARAMS_PATH = (
    ROOT / "data" / "calibration" / "params" / "stereoRemap.p"
)

OUTPUT_DIR = ROOT / "outputs" / "rectification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CONFIGURATION
# ==========================================

# Set to a specific pair number, e.g. 110.
# Set to None to use the latest matching pair.
PAIR_NUMBER = None

LINE_SPACING = 20


# ==========================================
# FIND MATCHING IMAGE PAIRS
# ==========================================

def get_pair_number(path):
    match = re.fullmatch(
        r"pair_(\d+)\.png",
        path.name
    )

    if match:
        return int(match.group(1))

    return None


def find_stereo_pair():
    left_images = {}

    for path in LEFT_DIR.glob("pair_*.png"):
        number = get_pair_number(path)

        if number is not None:
            left_images[number] = path

    right_images = {}

    for path in RIGHT_DIR.glob("pair_*.png"):
        number = get_pair_number(path)

        if number is not None:
            right_images[number] = path

    common = sorted(
        set(left_images.keys())
        & set(right_images.keys())
    )

    if not common:
        raise RuntimeError(
            "No matching stereo pairs found."
        )

    if PAIR_NUMBER is None:
        number = common[-1]
    else:
        number = PAIR_NUMBER

    if number not in common:
        raise RuntimeError(
            f"Pair {number} not found in both folders."
        )

    return (
        number,
        left_images[number],
        right_images[number]
    )


# ==========================================
# LOAD RECTIFICATION MAPS
# ==========================================

def load_rectification_maps():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {PARAMS_PATH}"
        )

    with open(PARAMS_PATH, "rb") as f:
        stereo_remap = pickle.load(f)

    print("Loaded rectification parameters.")
    print("Available keys:", list(stereo_remap.keys()))

    # Keys used by our calibration script
    mapxL = stereo_remap["Left_Stereo_Map_x"]
    mapyL = stereo_remap["Left_Stereo_Map_y"]

    mapxR = stereo_remap["Right_Stereo_Map_x"]
    mapyR = stereo_remap["Right_Stereo_Map_y"]

    return mapxL, mapyL, mapxR, mapyR


# ==========================================
# DRAW HORIZONTAL REFERENCE LINES
# ==========================================

def draw_horizontal_lines(image, spacing=20):
    result = image.copy()

    height, width = result.shape[:2]

    for y in range(spacing, height, spacing):
        cv2.line(
            result,
            (0, y),
            (width - 1, y),
            (0, 255, 0),
            1
        )

    return result


def draw_stereo_lines(left, right, spacing=20):
    combined = np.hstack([left, right])

    height, width = combined.shape[:2]

    for y in range(spacing, height, spacing):
        cv2.line(
            combined,
            (0, y),
            (width - 1, y),
            (0, 255, 0),
            1
        )

    return combined


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n====================================")
    print("STEREO RECTIFICATION VALIDATION")
    print("====================================")

    # Load calibration maps
    mapxL, mapyL, mapxR, mapyR = (
        load_rectification_maps()
    )

    # Find matching image pair
    number, left_path, right_path = (
        find_stereo_pair()
    )

    print(f"\nUsing pair: {number}")
    print(f"Left:  {left_path}")
    print(f"Right: {right_path}")

    # Read images
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))

    if left is None:
        raise RuntimeError(
            "Failed to read left image."
        )

    if right is None:
        raise RuntimeError(
            "Failed to read right image."
        )

    if left.shape != right.shape:
        raise RuntimeError(
            "Left and right image sizes do not match."
        )

    image_size = left.shape[:2]

    print(f"\nImage resolution: {left.shape[1]}x{left.shape[0]}")

    # Verify map dimensions
    if (
        mapxL.shape[:2] != image_size
        or mapyL.shape[:2] != image_size
        or mapxR.shape[:2] != image_size
        or mapyR.shape[:2] != image_size
    ):
        raise RuntimeError(
            "Rectification map dimensions do not "
            "match the input images."
        )

    # ======================================
    # APPLY RECTIFICATION
    # ======================================

    rect_left = cv2.remap(
        left,
        mapxL,
        mapyL,
        interpolation=cv2.INTER_LINEAR
    )

    rect_right = cv2.remap(
        right,
        mapxR,
        mapyR,
        interpolation=cv2.INTER_LINEAR
    )

    # ======================================
    # BUILD COMPARISON IMAGES
    # ======================================

    before = draw_stereo_lines(
        left,
        right,
        LINE_SPACING
    )

    after = draw_stereo_lines(
        rect_left,
        rect_right,
        LINE_SPACING
    )

    # ======================================
    # SAVE RESULTS
    # ======================================

    before_path = (
        OUTPUT_DIR / f"before_{number:03d}.jpg"
    )

    after_path = (
        OUTPUT_DIR / f"after_{number:03d}.jpg"
    )

    rect_left_path = (
        OUTPUT_DIR / f"rect_left_{number:03d}.png"
    )

    rect_right_path = (
        OUTPUT_DIR / f"rect_right_{number:03d}.png"
    )

    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)

    cv2.imwrite(str(rect_left_path), rect_left)
    cv2.imwrite(str(rect_right_path), rect_right)

    print("\nSaved results:")
    print(before_path)
    print(after_path)
    print(rect_left_path)
    print(rect_right_path)

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "Before Rectification",
        before
    )

    cv2.imshow(
        "After Rectification",
        after
    )

    print("\nPress any key to close the windows.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
