
from pathlib import Path
import pickle
import re

import cv2
import numpy as np
import yaml


# ==========================================
# PATHS AND CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "stereo.yaml"

LEFT_DIR = ROOT / "data" / "calibration" / "left"
RIGHT_DIR = ROOT / "data" / "calibration" / "right"

PARAMS_DIR = ROOT / "data" / "calibration" / "params"
DEBUG_DIR = ROOT / "outputs" / "calibration" / "corners"

PARAMS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

START_PAIR = 254

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

checkerboard = config["calibration"]["checkerboard"]

NX = int(checkerboard["Nx"])
NY = int(checkerboard["Ny"])

SQUARE_SIZE_CM = float(
    checkerboard["square_size_cm"]
)

# OpenCV expects (columns, rows) of INTERNAL corners.
PATTERN_SIZE = (NX, NY)


# ==========================================
# MATCH IMAGE PAIRS
# ==========================================

def get_pair_number(path):
    match = re.fullmatch(r"pair_(\d+)\.png", path.name)
    return int(match.group(1)) if match else None


def find_image_pairs():
    left_images = {}

    for path in LEFT_DIR.glob("pair_*.png"):
        number = get_pair_number(path)
        if number is not None and number >= START_PAIR:
            left_images[number] = path

    right_images = {}

    for path in RIGHT_DIR.glob("pair_*.png"):
        number = get_pair_number(path)
        if number is not None and number >= START_PAIR:
            right_images[number] = path

    common = sorted(set(left_images) & set(right_images))

    return [
        (number, left_images[number], right_images[number])
        for number in common
    ]


# ==========================================
# CHECKERBOARD CORNER DETECTION
# ==========================================

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        flags
    )

    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        criteria
    )

    return corners


# ==========================================
# LOAD VALID STEREO PAIRS
# ==========================================

def collect_calibration_points():
    pairs = find_image_pairs()

    print(f"\nStarting from pair {START_PAIR}")
    print(f"Matching pairs found: {len(pairs)}")

    if not pairs:
        raise RuntimeError("No matching image pairs found.")

    # Checkerboard points in centimeters.
    objp = np.zeros((NX * NY, 3), np.float32)

    objp[:, :2] = (
        np.mgrid[0:NX, 0:NY]
        .T
        .reshape(-1, 2)
        * SQUARE_SIZE_CM
    )

    object_points = []
    image_points_left = []
    image_points_right = []

    image_size = None

    for number, left_path, right_path in pairs:
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))

        if left is None or right is None:
            print(f"Pair {number:03d}: unreadable image")
            continue

        if left.shape[:2] != right.shape[:2]:
            print(f"Pair {number:03d}: size mismatch")
            continue

        current_size = (left.shape[1], left.shape[0])

        if image_size is None:
            image_size = current_size

        if current_size != image_size:
            print(f"Pair {number:03d}: unexpected resolution")
            continue

        corners_left = detect_corners(left)
        corners_right = detect_corners(right)

        if corners_left is None or corners_right is None:
            print(f"Pair {number:03d}: corners not found in both")
            continue

        object_points.append(objp.copy())
        image_points_left.append(corners_left)
        image_points_right.append(corners_right)

        left_debug = left.copy()
        right_debug = right.copy()

        cv2.drawChessboardCorners(
            left_debug, PATTERN_SIZE, corners_left, True
        )
        cv2.drawChessboardCorners(
            right_debug, PATTERN_SIZE, corners_right, True
        )

        cv2.imwrite(
            str(DEBUG_DIR / f"left_{number:03d}.jpg"),
            left_debug
        )
        cv2.imwrite(
            str(DEBUG_DIR / f"right_{number:03d}.jpg"),
            right_debug
        )

        print(f"Pair {number:03d}: OK")

    print("\nValid stereo pairs:", len(object_points))

    if len(object_points) < 10:
        raise RuntimeError(
            "Too few valid stereo pairs. "
            "At least 10 are required by this script; "
            "15-30 varied poses are preferable."
        )

    return (
        object_points,
        image_points_left,
        image_points_right,
        image_size
    )


# ==========================================
# SAVE HELPERS
# ==========================================

def save_pickle(filename, data):
    path = PARAMS_DIR / filename

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print("Saved:", path)


# ==========================================
# MAIN CALIBRATION
# ==========================================

def main():
    (
        object_points,
        points_left,
        points_right,
        image_size
    ) = collect_calibration_points()

    print("\nImage size:", image_size)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6
    )

    # --------------------------------------
    # LEFT CAMERA INTRINSICS
    # --------------------------------------

    print("\nCalibrating LEFT camera...")

    rms_left, K1, D1, _, _ = cv2.calibrateCamera(
        object_points,
        points_left,
        image_size,
        None,
        None
    )

    # --------------------------------------
    # RIGHT CAMERA INTRINSICS
    # --------------------------------------

    print("Calibrating RIGHT camera...")

    rms_right, K2, D2, _, _ = cv2.calibrateCamera(
        object_points,
        points_right,
        image_size,
        None,
        None
    )

    # --------------------------------------
    # STEREO EXTRINSICS
    # --------------------------------------

    print("Running stereo calibration...")

    stereo_rms, K1, D1, K2, D2, R, T, E, F = (
        cv2.stereoCalibrate(
            object_points,
            points_left,
            points_right,
            K1,
            D1,
            K2,
            D2,
            image_size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
    )

    # --------------------------------------
    # RECTIFICATION
    # --------------------------------------

    print("Computing stereo rectification...")

    R1, R2, P1, P2, Q, roi1, roi2 = (
        cv2.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            image_size,
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.5
        )
    )

    left_map_x, left_map_y = (
        cv2.initUndistortRectifyMap(
            K1,
            D1,
            R1,
            P1,
            image_size,
            cv2.CV_32FC1
        )
    )

    right_map_x, right_map_y = (
        cv2.initUndistortRectifyMap(
            K2,
            D2,
            R2,
            P2,
            image_size,
            cv2.CV_32FC1
        )
    )

    # --------------------------------------
    # SAVE PARAMETERS
    # --------------------------------------

    save_pickle(
        "CamParasL.p",
        {
            "cameraMatrix": K1,
            "distCoeffs": D1,
            "imageSize": image_size,
            "rms": rms_left
        }
    )

    save_pickle(
        "CamParasR.p",
        {
            "cameraMatrix": K2,
            "distCoeffs": D2,
            "imageSize": image_size,
            "rms": rms_right
        }
    )

    save_pickle(
        "stereoRemap.p",
        {
            "Left_Stereo_Map_x": left_map_x,
            "Left_Stereo_Map_y": left_map_y,
            "Right_Stereo_Map_x": right_map_x,
            "Right_Stereo_Map_y": right_map_y
        }
    )

    save_pickle(
        "disp2depth.p",
        {
            "Q": Q,
            "R": R,
            "T": T,
            "P1": P1,
            "P2": P2,
            "E": E,
            "F": F
        }
    )

    # --------------------------------------
    # RESULTS
    # --------------------------------------

    baseline_cm = float(np.linalg.norm(T))

    print("\n====================================")
    print("STEREO CALIBRATION RESULTS")
    print("====================================")

    print(f"Left RMS:   {rms_left:.4f} px")
    print(f"Right RMS:  {rms_right:.4f} px")
    print(f"Stereo RMS: {stereo_rms:.4f} px")

    print(f"\nEstimated baseline: {baseline_cm:.2f} cm")

    print("\nLeft camera matrix:")
    print(K1)

    print("\nRight camera matrix:")
    print(K2)

    print("\nRotation R:")
    print(R)

    print("\nTranslation T (cm):")
    print(T.flatten())

    print("\nCalibration files saved to:")
    print(PARAMS_DIR)

    print("\nCalibration complete!")


if __name__ == "__main__":
    main()
