
import cv2
import yaml
import numpy as np
from pathlib import Path

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "monocular.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

nx = int(config["calibration"]["checkerboard"]["Nx"])
ny = int(config["calibration"]["checkerboard"]["Ny"])
square_cm = float(
    config["calibration"]["checkerboard"]["square_size_cm"]
)

image_dir = ROOT / config["paths"]["capture_dir"]

output_dir = ROOT / "outputs" / "monocular"
output_dir.mkdir(parents=True, exist_ok=True)

print("\n========== MONOCULAR HOMOGRAPHY CALIBRATION ==========")
print(f"Project root: {ROOT}")
print(f"Calibration images: {image_dir}")
print(f"Checkerboard internal corners: {nx} x {ny}")
print(f"Square size: {square_cm} cm")
print(f"Output directory: {output_dir}")

if nx < 2 or ny < 2 or square_cm <= 0:
    raise ValueError("Invalid checkerboard configuration")

# ============================================================
# PHYSICAL CHECKERBOARD COORDINATES
# ============================================================

# Each internal corner is assigned a coordinate in cm.
#
# Example for 3 cm squares:
#
# (0,0), (3,0), (6,0), ...
# (0,3), (3,3), (6,3), ...
#
# The origin is the first detected internal corner.

world_points = np.zeros((nx * ny, 2), dtype=np.float32)

world_points[:] = (
    np.mgrid[0:nx, 0:ny]
    .T.reshape(-1, 2) * square_cm
)

# ============================================================
# CHECKERBOARD DETECTION
# ============================================================

def detect_checkerboard(gray):
    pattern_size = (nx, ny)

    # Method 1: Robust SB detector
    flags = (
        cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern_size,
        flags=flags
    )

    if found:
        return True, corners, "SB exhaustive"

    # Method 2: Standard detector fallback
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    )

    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.001
        )

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        return True, corners, "Standard + subpixel"

    return False, None, "Failed"


# ============================================================
# LOAD CALIBRATION IMAGES
# ============================================================

image_files = sorted(
    p for p in image_dir.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
) if image_dir.exists() else []

if not image_files:
    raise RuntimeError(
        f"No calibration images found in {image_dir}"
    )

print(f"\nFound {len(image_files)} calibration images")

calibration_successful = False

# ============================================================
# PROCESS IMAGES
# ============================================================

for image_path in image_files:

    print(f"\nProcessing: {image_path.name}")

    frame = cv2.imread(str(image_path))

    if frame is None:
        print("ERROR: Could not read image")
        continue

    height, width = frame.shape[:2]

    print(f"Image resolution: {width} x {height}")

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    found, corners, method = detect_checkerboard(gray)

    print(f"Detection method: {method}")

    if not found:
        print("Checkerboard NOT detected")

        # Save the image for inspection
        cv2.imwrite(
            str(output_dir / f"failed_{image_path.stem}.jpg"),
            frame
        )

        continue

    print(f"Detected {len(corners)} corners")

    if len(corners) != nx * ny:
        print("ERROR: Incorrect number of corners")
        continue

    # ========================================================
    # COMPUTE HOMOGRAPHY
    # ========================================================

    image_points = corners.reshape(-1, 2).astype(
        np.float32
    )

    H, mask = cv2.findHomography(
        image_points,
        world_points,
        cv2.RANSAC,
        2.0
    )

    if H is None:
        print("ERROR: Homography computation failed")
        continue

    # ========================================================
    # REPROJECTION ERROR
    # ========================================================

    projected_points = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected_points - world_points,
        axis=1
    )

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))

    print(f"Mean reprojection error: {mean_error:.4f} cm")
    print(f"Maximum reprojection error: {max_error:.4f} cm")

    # ========================================================
    # SAVE DETECTION PREVIEW
    # ========================================================

    preview = frame.copy()

    cv2.drawChessboardCorners(
        preview,
        (nx, ny),
        corners,
        True
    )

    preview_path = (
        output_dir / f"corners_{image_path.stem}.jpg"
    )

    cv2.imwrite(str(preview_path), preview)

    print(f"Corner preview saved: {preview_path}")

    # ========================================================
    # SAVE HOMOGRAPHY
    # ========================================================

    output_path = output_dir / "homography.npz"

    np.savez(
        output_path,
        H=H,
        image_size=np.array([width, height]),
        square_size_cm=square_cm
    )

    print("\n========== CALIBRATION SUCCESSFUL ==========")
    print("Homography matrix:")
    print(H)

    print(f"\nSaved homography: {output_path}")

    calibration_successful = True

    # Use the first successfully detected image.
    break


# ============================================================
# FINAL STATUS
# ============================================================

if not calibration_successful:
    raise RuntimeError(
        "\nCalibration failed for all images.\n"
        "Check checkerboard dimensions, image quality, "
        "and the saved debug images."
    )

print("\nCalibration complete.")
