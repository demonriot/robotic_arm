
from pathlib import Path
import csv
import pickle
import re

import cv2
import numpy as np


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEFT_DIR = PROJECT_ROOT / "data/calibration/left"
RIGHT_DIR = PROJECT_ROOT / "data/calibration/right"
PARAMS_DIR = PROJECT_ROOT / "data/calibration/params"

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/calibration/corner_verification"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# CHECKERBOARD CONFIGURATION
# ==========================================

CHECKERBOARD = (9, 6)

# Set to 78 if your calibration still uses
# only pairs starting from pair_078.
START_PAIR = 254

# Limit diagnostic images if you have many pairs.
# CSV statistics will still cover all valid pairs.
MAX_VISUALIZATIONS = 30


# ==========================================
# HELPERS
# ==========================================

def get_pair_number(path):
    match = re.search(r"pair_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def load_pickle(filename):
    with open(PARAMS_DIR / filename, "rb") as f:
        return pickle.load(f)


def find_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        ),
    )

    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )

    return corners.reshape(-1, 2)


# ==========================================
# EPIPOLAR ERROR
# ==========================================

def symmetric_epipolar_errors(left, right, F):
    """
    Symmetric point-to-epipolar-line distances.

    Note:
    These are diagnostics against the already
    estimated F, not independent ground truth.
    """

    ones = np.ones((len(left), 1))

    pts_l = np.hstack([left, ones])
    pts_r = np.hstack([right, ones])

    # Lines in right image from left points
    lines_r = (F @ pts_l.T).T

    # Lines in left image from right points
    lines_l = (F.T @ pts_r.T).T

    numerator = np.abs(
        np.sum(pts_r * lines_r, axis=1)
    )

    dist_r = numerator / np.maximum(
        np.linalg.norm(lines_r[:, :2], axis=1),
        1e-12,
    )

    dist_l = numerator / np.maximum(
        np.linalg.norm(lines_l[:, :2], axis=1),
        1e-12,
    )

    return 0.5 * (dist_l + dist_r)


# ==========================================
# VISUALIZATION
# ==========================================

def draw_corner_matches(left_img, right_img,
                        corners_l, corners_r,
                        errors, pair_num):

    height = max(
        left_img.shape[0],
        right_img.shape[0],
    )

    width_l = left_img.shape[1]
    width_r = right_img.shape[1]

    canvas = np.zeros(
        (height, width_l + width_r, 3),
        dtype=np.uint8,
    )

    canvas[:left_img.shape[0], :width_l] = left_img
    canvas[:right_img.shape[0], width_l:] = right_img

    for i, (pt_l, pt_r) in enumerate(
        zip(corners_l, corners_r)
    ):
        x1, y1 = np.round(pt_l).astype(int)
        x2, y2 = np.round(pt_r).astype(int)

        x2 += width_l

        # Green: low error
        # Orange: moderate error
        # Red: high error
        if errors[i] < 1:
            color = (0, 255, 0)
        elif errors[i] < 3:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            color,
            1,
        )

        cv2.circle(
            canvas,
            (x1, y1),
            3,
            color,
            -1,
        )

        cv2.circle(
            canvas,
            (x2, y2),
            3,
            color,
            -1,
        )

        # Label a subset to reduce clutter
        if i in [0, 8, 9, 26, 44, 45, 53]:
            cv2.putText(
                canvas,
                str(i),
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                canvas,
                str(i),
                (x2 + 4, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    cv2.putText(
        canvas,
        f"Pair {pair_num} | Mean epipolar error: "
        f"{np.mean(errors):.2f} px",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return canvas


# ==========================================
# MAIN
# ==========================================

def main():
    params = load_pickle("disp2depth.p")

    F = np.asarray(params["F"], dtype=np.float64)

    left_files = {
        get_pair_number(p): p
        for p in LEFT_DIR.glob("pair_*.png")
    }

    right_files = {
        get_pair_number(p): p
        for p in RIGHT_DIR.glob("pair_*.png")
    }

    pair_numbers = sorted(
        (set(left_files) & set(right_files))
    )

    pair_numbers = [
        n for n in pair_numbers
        if n is not None and n >= START_PAIR
    ]

    print(f"Found {len(pair_numbers)} paired images.")

    results = []
    visualized = 0

    for pair_num in pair_numbers:
        left_img = cv2.imread(
            str(left_files[pair_num])
        )

        right_img = cv2.imread(
            str(right_files[pair_num])
        )

        if left_img is None or right_img is None:
            print(f"Pair {pair_num}: unreadable image")
            continue

        if left_img.shape[:2] != right_img.shape[:2]:
            print(
                f"Pair {pair_num}: image size mismatch"
            )
            continue

        corners_l = find_corners(left_img)
        corners_r = find_corners(right_img)

        if corners_l is None or corners_r is None:
            print(
                f"Pair {pair_num}: checkerboard not found"
            )
            continue

        errors = symmetric_epipolar_errors(
            corners_l,
            corners_r,
            F,
        )

        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))
        max_error = float(np.max(errors))

        results.append({
            "pair": pair_num,
            "mean_error_px": round(mean_error, 4),
            "median_error_px": round(median_error, 4),
            "max_error_px": round(max_error, 4),
        })

        print(
            f"Pair {pair_num:03d} | "
            f"mean={mean_error:.2f}px | "
            f"max={max_error:.2f}px"
        )

        if visualized < MAX_VISUALIZATIONS:
            canvas = draw_corner_matches(
                left_img,
                right_img,
                corners_l,
                corners_r,
                errors,
                pair_num,
            )

            cv2.imwrite(
                str(
                    OUTPUT_DIR
                    / f"matches_{pair_num:03d}.jpg"
                ),
                canvas,
            )

            visualized += 1

    if not results:
        print("No valid checkerboard pairs found.")
        return

    csv_path = OUTPUT_DIR / "pair_errors.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair",
                "mean_error_px",
                "median_error_px",
                "max_error_px",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    sorted_results = sorted(
        results,
        key=lambda r: r["mean_error_px"],
        reverse=True,
    )

    print("\n================================")
    print("WORST 10 PAIRS")
    print("================================")

    for row in sorted_results[:10]:
        print(
            f"Pair {row['pair']:03d}: "
            f"{row['mean_error_px']:.2f} px"
        )

    all_means = [
        r["mean_error_px"] for r in results
    ]

    print("\n================================")
    print("SUMMARY")
    print("================================")

    print(f"Valid pairs: {len(results)}")
    print(
        f"Mean pair error: "
        f"{np.mean(all_means):.3f} px"
    )
    print(
        f"Median pair error: "
        f"{np.median(all_means):.3f} px"
    )

    print(f"\nCSV saved: {csv_path}")
    print(f"Visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
