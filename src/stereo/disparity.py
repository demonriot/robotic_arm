
from pathlib import Path
import pickle

import cv2
import numpy as np
import requests
import yaml


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "stereo.yaml"
PARAMS_DIR = ROOT / "data" / "calibration" / "params"
OUTPUT_DIR = ROOT / "outputs" / "disparity"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

LEFT_URL = config["camera"]["left_id"]
RIGHT_URL = config["camera"]["right_id"]

# StereoSGBM settings for 320x240 images
MIN_DISPARITY = 0
NUM_DISPARITIES = 64  # Must be divisible by 16
BLOCK_SIZE = 5


# ==========================================
# LOAD CALIBRATION PARAMETERS
# ==========================================

def load_calibration():
    with open(PARAMS_DIR / "stereoRemap.p", "rb") as f:
        remap = pickle.load(f)

    with open(PARAMS_DIR / "disp2depth.p", "rb") as f:
        depth_params = pickle.load(f)

    maps = (
        remap["Left_Stereo_Map_x"],
        remap["Left_Stereo_Map_y"],
        remap["Right_Stereo_Map_x"],
        remap["Right_Stereo_Map_y"],
    )

    Q = depth_params["Q"]

    return maps, Q


# ==========================================
# CAPTURE IMAGES FROM ESP32-CAMS
# ==========================================

def capture_image(url):
    # Accept /capture or /stream in stereo.yaml
    base_url = url.rsplit("/", 1)[0]
    capture_url = base_url + "/capture"

    response = requests.get(capture_url, timeout=10)
    response.raise_for_status()

    data = np.frombuffer(
        response.content,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            f"Could not decode image from {capture_url}"
        )

    return image


def capture_stereo_pair():
    print("Capturing left image...")
    left = capture_image(LEFT_URL)

    print("Capturing right image...")
    right = capture_image(RIGHT_URL)

    if left.shape != right.shape:
        raise RuntimeError(
            "Left and right image dimensions differ."
        )

    cv2.imwrite(
        str(OUTPUT_DIR / "raw_left.png"),
        left
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "raw_right.png"),
        right
    )

    return left, right


# ==========================================
# RECTIFICATION
# ==========================================

def rectify_pair(left, right, maps):
    mapxL, mapyL, mapxR, mapyR = maps

    expected = mapxL.shape[:2]

    if (
        left.shape[:2] != expected
        or right.shape[:2] != expected
    ):
        raise RuntimeError(
            f"Image resolution does not match calibration: "
            f"expected {expected}, "
            f"got {left.shape[:2]}"
        )

    rect_left = cv2.remap(
        left, mapxL, mapyL, cv2.INTER_LINEAR
    )

    rect_right = cv2.remap(
        right, mapxR, mapyR, cv2.INTER_LINEAR
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "rect_left.png"),
        rect_left
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "rect_right.png"),
        rect_right
    )

    return rect_left, rect_right


# ==========================================
# STEREO DISPARITY
# ==========================================

def compute_disparity(left, right):
    gray_left = cv2.cvtColor(
        left, cv2.COLOR_BGR2GRAY
    )

    gray_right = cv2.cvtColor(
        right, cv2.COLOR_BGR2GRAY
    )

    channels = 1

    stereo = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * channels * BLOCK_SIZE ** 2,
        P2=32 * channels * BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # OpenCV returns disparity scaled by 16
    disparity = stereo.compute(
        gray_left,
        gray_right
    ).astype(np.float32) / 16.0

    return disparity


# ==========================================
# VISUALIZATION
# ==========================================

def create_disparity_preview(disparity):
    valid = (
        np.isfinite(disparity)
        & (disparity > MIN_DISPARITY)
    )

    normalized = np.zeros(
        disparity.shape,
        dtype=np.uint8
    )

    if np.any(valid):
        values = disparity[valid]

        low = np.percentile(values, 2)
        high = np.percentile(values, 98)

        if high > low:
            normalized[valid] = np.clip(
                (values - low) * 255.0 / (high - low),
                0,
                255
            ).astype(np.uint8)

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    colored[~valid] = (0, 0, 0)

    return colored, valid


# ==========================================
# MAIN
# ==========================================

def main():
    print("\n===================================")
    print("ESP32-CAM STEREO DEPTH TEST")
    print("===================================")

    maps, Q = load_calibration()

    left, right = capture_stereo_pair()

    rect_left, rect_right = rectify_pair(
        left, right, maps
    )

    # ==========================================
    # RECTIFICATION ALIGNMENT CHECK
    # ==========================================

    comparison = np.hstack((
        rect_left.copy(),
        rect_right.copy()
    ))

    for y in range(0, comparison.shape[0], 20):
        cv2.line(
            comparison,
            (0, y),
            (comparison.shape[1] - 1, y),
            (0, 255, 0),
            1
        )

    cv2.imwrite(
        str(OUTPUT_DIR / "rectification_check.png"),
        comparison
    )

    disparity = compute_disparity(
        rect_left, rect_right
    )

    np.save(
        OUTPUT_DIR / "disparity.npy",
        disparity
    )

    preview, valid = create_disparity_preview(
        disparity
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "disparity_preview.png"),
        preview
    )

    # Q converts rectified image coordinates
    # and disparity into 3D coordinates.
    points_3d = cv2.reprojectImageTo3D(
        disparity,
        Q
    )

    print("\nValid disparity pixels:",
          int(np.count_nonzero(valid)))

    print("Total pixels:", disparity.size)

    print("\nClick a point on the LEFT image")
    print("to estimate its 3D coordinates.")
    print("Press any key to exit.")

    display = rect_left.copy()

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not valid[y, x]:
            print(
                f"({x}, {y}): No valid disparity"
            )
            return

        X, Y, Z = points_3d[y, x]

        if not np.all(np.isfinite([X, Y, Z])):
            print("Invalid 3D coordinates")
            return

        print(
            f"Pixel ({x}, {y}) | "
            f"Disparity: {disparity[y, x]:.2f} px | "
            f"X: {X:.2f} cm | "
            f"Y: {Y:.2f} cm | "
            f"Z: {Z:.2f} cm"
        )

        annotated = display.copy()

        cv2.circle(
            annotated,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

        cv2.imshow(
            "Rectified Left - Click for Depth",
            annotated
        )

    cv2.namedWindow(
        "Rectified Left - Click for Depth"
    )

    cv2.setMouseCallback(
        "Rectified Left - Click for Depth",
        on_mouse
    )

    cv2.imshow(
        "Rectified Left - Click for Depth",
        display
    )

    cv2.imshow(
        "Rectified Right",
        rect_right
    )

    cv2.imshow(
        "Disparity",
        preview
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
