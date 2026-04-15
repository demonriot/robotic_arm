import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


CLICKED_POINT: Optional[Tuple[int, int]] = None


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global CLICKED_POINT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICKED_POINT = (x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate stereo depth Z at a clicked point on a rectified stereo pair."
    )

    parser.add_argument(
        "--left-image",
        type=str,
        required=True,
        help="Path to the left image.",
    )
    parser.add_argument(
        "--right-image",
        type=str,
        required=True,
        help="Path to the right image.",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing stereoRemap.p, CamParasL.p, CamParasR.p, disp2depth.p",
    )
    parser.add_argument(
        "--actual-distance-cm",
        type=float,
        required=True,
        help="Measured real-world distance in cm for this validation trial.",
    )
    parser.add_argument(
        "--trial-name",
        type=str,
        required=True,
        help="Trial label, e.g. 114cm_trial1",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        default=9,
        help="Odd ROI window size around clicked point. Default: 9",
    )
    parser.add_argument(
        "--disp-stat",
        type=lambda s: s.strip().lower(),
        default="median",
        choices=["q1", "median", "q3"],
        help="Statistic to use from ROI disparity values. Choices: q1, median, q3. Default: median",
    )
    parser.add_argument(
        "--min-disparity",
        type=int,
        default=None,
        help="StereoSGBM minDisparity. Default loops to yaml.",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=None,
        help="StereoSGBM numDisparities, must be a multiple of 16. Default loops to yaml.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="StereoSGBM blockSize, must be odd. Default loops to yaml.",
    )
    parser.add_argument(
        "--save-rectified",
        action="store_true",
        help="Save rectified images for debugging.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to stereo_cam_validation/output relative to this script.",
    )

    return parser.parse_args()


def ensure_odd(n: int, name: str) -> int:
    if n % 2 == 0:
        raise ValueError(f"{name} must be odd, got {n}.")
    return n


def ensure_num_disparities(n: int) -> int:
    if n <= 0 or n % 16 != 0:
        raise ValueError(f"num_disparities must be positive and a multiple of 16, got {n}.")
    return n


def load_pickle_any(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_first_key(data: Dict[str, Any], candidates: list[str], label: str) -> Any:
    for key in candidates:
        if key in data:
            return data[key]
    raise KeyError(f"Could not find {label}. Tried keys: {candidates}")


def load_q_matrix(data: Any, source_path: Path) -> np.ndarray:
    if isinstance(data, dict):
        for key in ["Q", "q", "reprojectionMatrix", "reprojection_matrix"]:
            if key in data:
                q = np.asarray(data[key], dtype=np.float64)
                if q.shape != (4, 4):
                    raise ValueError(f"Q matrix in {source_path} must be 4x4, got {q.shape}")
                return q
        raise KeyError(f"Could not find Q matrix inside dict from {source_path}")

    if isinstance(data, np.ndarray):
        q = np.asarray(data, dtype=np.float64)
        if q.shape != (4, 4):
            raise ValueError(f"Q matrix in {source_path} must be 4x4, got {q.shape}")
        return q

    raise ValueError(f"Unsupported Q data type in {source_path}: {type(data)}")


def load_calibration_bundle(
    calib_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stereo_remap_path = calib_dir / "stereoRemap.p"
    cam_paras_l_path = calib_dir / "CamParasL.p"
    cam_paras_r_path = calib_dir / "CamParasR.p"

    disp2depth_path = calib_dir / "disp2depth.p"
    dis2depth_path = calib_dir / "dis2depth.p"

    if disp2depth_path.exists():
        depth_path = disp2depth_path
    elif dis2depth_path.exists():
        depth_path = dis2depth_path
    else:
        raise FileNotFoundError(
            f"Could not find either {disp2depth_path.name} or {dis2depth_path.name} in {calib_dir}"
        )

    for p in [stereo_remap_path, cam_paras_l_path, cam_paras_r_path, depth_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required calibration file not found: {p}")

    stereo_remap = load_pickle_any(stereo_remap_path)
    cam_l = load_pickle_any(cam_paras_l_path)
    cam_r = load_pickle_any(cam_paras_r_path)
    depth_data = load_pickle_any(depth_path)

    if not isinstance(stereo_remap, dict):
        raise ValueError(
            f"Expected stereoRemap.p to contain a dictionary, got {type(stereo_remap)}"
        )

    if all(k in stereo_remap for k in ["mapxL", "mapyL", "mapxR", "mapyR"]):
        left_map_x = np.asarray(stereo_remap["mapxL"])
        left_map_y = np.asarray(stereo_remap["mapyL"])
        right_map_x = np.asarray(stereo_remap["mapxR"])
        right_map_y = np.asarray(stereo_remap["mapyR"])
    elif all(k in stereo_remap for k in ["leftMapX", "leftMapY", "rightMapX", "rightMapY"]):
        left_map_x = np.asarray(stereo_remap["leftMapX"])
        left_map_y = np.asarray(stereo_remap["leftMapY"])
        right_map_x = np.asarray(stereo_remap["rightMapX"])
        right_map_y = np.asarray(stereo_remap["rightMapY"])
    elif all(k in stereo_remap for k in ["stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y"]):
        left_map_x = np.asarray(stereo_remap["stereoMapL_x"])
        left_map_y = np.asarray(stereo_remap["stereoMapL_y"])
        right_map_x = np.asarray(stereo_remap["stereoMapR_x"])
        right_map_y = np.asarray(stereo_remap["stereoMapR_y"])
    elif "stereoMapL" in stereo_remap and "stereoMapR" in stereo_remap:
        stereo_map_l = stereo_remap["stereoMapL"]
        stereo_map_r = stereo_remap["stereoMapR"]

        if len(stereo_map_l) != 2 or len(stereo_map_r) != 2:
            raise ValueError("stereoMapL/stereoMapR must each contain (mapX, mapY).")

        left_map_x = np.asarray(stereo_map_l[0])
        left_map_y = np.asarray(stereo_map_l[1])
        right_map_x = np.asarray(stereo_map_r[0])
        right_map_y = np.asarray(stereo_map_r[1])
    else:
        raise KeyError(
            f"Could not extract remap maps from stereoRemap.p. Available keys: {list(stereo_remap.keys())}"
        )

    def extract_camera_matrix(data: Any, label: str) -> np.ndarray:
        if isinstance(data, dict):
            K = find_first_key(
                data,
                [
                    "cameraMatrix",
                    "cameraMat",
                    "camMat",
                    "K",
                    "K1",
                    "K2",
                    "mtx",
                    "mtxL",
                    "mtxR",
                    "camera_matrix",
                    "intrinsic",
                    "intrinsics",
                ],
                f"{label} camera matrix",
            )
        elif isinstance(data, np.ndarray):
            K = data
        elif isinstance(data, (tuple, list)):
            K = None
            for item in data:
                arr = np.asarray(item)
                if arr.shape == (3, 3):
                    K = arr
                    break
            if K is None:
                raise ValueError(
                    f"Could not find a 3x3 camera matrix inside {label} camera parameters."
                )
        else:
            raise ValueError(
                f"Unsupported type for {label} camera parameters: {type(data)}"
            )

        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"{label} camera matrix must be 3x3, got {K.shape}")
        return K

    K1 = extract_camera_matrix(cam_l, "left")
    K2 = extract_camera_matrix(cam_r, "right")
    Q = load_q_matrix(depth_data, depth_path)

    return left_map_x, left_map_y, right_map_x, right_map_y, K1, K2, Q


def rectify_images(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_map_x: np.ndarray,
    left_map_y: np.ndarray,
    right_map_x: np.ndarray,
    right_map_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    left_rect = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)
    return left_rect, right_rect


def create_matcher(min_disp: int, num_disp: int, block_size: int) -> cv2.StereoSGBM:
    p1 = 8 * 3 * block_size * block_size
    p2 = 32 * 3 * block_size * block_size

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def compute_disparity(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    matcher: cv2.StereoSGBM,
) -> np.ndarray:
    gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity


def normalize_disparity_for_display(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0
    disp_vis = np.zeros_like(disparity, dtype=np.uint8)

    if np.any(valid):
        valid_disp = disparity[valid]
        dmin = float(np.min(valid_disp))
        dmax = float(np.max(valid_disp))

        if dmax > dmin:
            scaled = ((disparity - dmin) * 255.0 / (dmax - dmin)).clip(0, 255)
            disp_vis = scaled.astype(np.uint8)

    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return disp_color


def get_roi_bounds(
    x: int,
    y: int,
    roi_size: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    half = roi_size // 2
    x0 = max(0, x - half)
    x1 = min(width, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(height, y + half + 1)
    return x0, x1, y0, y1


def compute_roi_disparity_stats(
    disparity: np.ndarray,
    x: int,
    y: int,
    roi_size: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    h, w = disparity.shape
    x0, x1, y0, y1 = get_roi_bounds(x, y, roi_size, w, h)
    roi = disparity[y0:y1, x0:x1]

    valid_mask = np.isfinite(roi) & (roi > 0.0)
    valid_values = roi[valid_mask]

    if valid_values.size == 0:
        raise ValueError("No valid disparity values found in ROI around clicked point.")

    stats = {
        "q1": float(np.percentile(valid_values, 25)),
        "median": float(np.median(valid_values)),
        "q3": float(np.percentile(valid_values, 75)),
    }

    return stats, roi


def reproject_point_with_q(
    x: int,
    y: int,
    disparity_value: float,
    Q: np.ndarray,
) -> Tuple[float, float, float]:
    pixel = np.array([x, y, disparity_value, 1.0], dtype=np.float64)
    world_h = Q @ pixel

    w = world_h[3]
    if abs(w) < 1e-12:
        raise ValueError("Invalid reprojection result: homogeneous scale W is too close to zero.")

    X = world_h[0] / w
    Y = world_h[1] / w
    Z = world_h[2] / w
    return float(X), float(Y), float(Z)


def convert_z_to_cm(z_value: float) -> float:
    """
    Heuristic conversion:
    - If Z looks like meters, convert to cm
    - If Z looks like mm, convert to cm
    - Else assume already cm
    """
    z_abs = abs(z_value)

    if z_abs < 10:
        return z_value * 100.0
    if z_abs > 1000:
        return z_value / 10.0
    return z_value


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial_name",
        "actual_distance_cm",
        "clicked_u",
        "clicked_v",
        "roi_size",
        "q1_disparity_px",
        "median_disparity_px",
        "q3_disparity_px",
        "used_disp_stat",
        "used_disparity_px",
        "estimated_z_raw",
        "estimated_z_cm",
        "error_cm",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def draw_click_visualization(
    image: np.ndarray,
    x: int,
    y: int,
    roi_size: int,
    estimated_z_cm: float,
    disparity_value: float,
    disp_stat_name: str,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    x0, x1, y0, y1 = get_roi_bounds(x, y, roi_size, w, h)

    cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
    cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

    text1 = f"{disp_stat_name}={disparity_value:.2f}px"
    text2 = f"Z={estimated_z_cm:.2f} cm"

    cv2.putText(
        vis,
        text1,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        text2,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return vis


def main() -> None:
    global CLICKED_POINT
    CLICKED_POINT = None

    args = parse_args()
    
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    import yaml
    config_path = base_dir / 'configs' / 'stereo.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Config fallbacks
    min_disp = args.min_disparity if args.min_disparity is not None else config['disparity']['min_disp']
    num_disp = args.num_disparities if args.num_disparities is not None else (config['disparity']['num_disp_multiplier'] * 16)
    b_size = args.block_size if args.block_size is not None else config['disparity']['block_size']

    ensure_odd(args.roi_size, "roi_size")
    ensure_odd(b_size, "block_size")
    ensure_num_disparities(num_disp)

    output_dir = Path(args.output_dir) if args.output_dir else script_dir.parent / "outputs" / "validation"

    disparity_dir = output_dir / "disparity_maps"
    clicked_dir = output_dir / "clicked_visualizations"
    rectified_dir = output_dir / "rectified_frames"
    log_csv_path = output_dir / "depth_logs" / "z_validation_results.csv"

    disparity_dir.mkdir(parents=True, exist_ok=True)
    clicked_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    left_path = Path(args.left_image)
    right_path = Path(args.right_image)
    calib_dir = Path(args.calib_dir) if args.calib_dir else base_dir / 'data' / 'calibration' / 'params'

    if not left_path.exists():
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right image not found: {right_path}")
    if not calib_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")

    print("[INFO] Loading calibration bundle...")
    left_map_x, left_map_y, right_map_x, right_map_y, K1, K2, Q = load_calibration_bundle(calib_dir)

    print("[INFO] Loading images...")
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None:
        raise ValueError(f"Failed to read left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Failed to read right image: {right_path}")

    print("[INFO] Rectifying images...")
    left_rect, right_rect = rectify_images(
        left_img,
        right_img,
        left_map_x,
        left_map_y,
        right_map_x,
        right_map_y,
    )

    if args.save_rectified:
        cv2.imwrite(str(rectified_dir / f"{args.trial_name}_left_rectified.png"), left_rect)
        cv2.imwrite(str(rectified_dir / f"{args.trial_name}_right_rectified.png"), right_rect)

    print("[INFO] Computing disparity...")
    matcher = create_matcher(min_disp, num_disp, b_size)
    disparity = compute_disparity(left_rect, right_rect, matcher)

    disp_vis = normalize_disparity_for_display(disparity)
    disparity_out_path = disparity_dir / f"{args.trial_name}_disparity.png"
    cv2.imwrite(str(disparity_out_path), disp_vis)

    print("[INFO] Click the marked point in the displayed rectified LEFT image.")
    display_img = left_rect.copy()
    window_name = "Rectified Left - Click Point"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display_img.copy()

        if CLICKED_POINT is not None:
            cv2.circle(temp, CLICKED_POINT, 4, (0, 0, 255), -1)

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF

        if CLICKED_POINT is not None:
            break

        if key == 27:
            cv2.destroyAllWindows()
            print("[INFO] Exiting without measurement.")
            return

    cv2.destroyAllWindows()

    x, y = CLICKED_POINT
    print(f"[INFO] Clicked point: (u={x}, v={y})")

    print("[INFO] Computing ROI disparity statistics...")
    disp_stats, _ = compute_roi_disparity_stats(disparity, x, y, args.roi_size)
    disp_value = disp_stats[args.disp_stat]

    print(
        f"[INFO] ROI disparity stats -> "
        f"Q1: {disp_stats['q1']:.4f}, "
        f"Median: {disp_stats['median']:.4f}, "
        f"Q3: {disp_stats['q3']:.4f}"
    )
    print(f"[INFO] Using disparity statistic: {args.disp_stat} = {disp_value:.4f}")

    print("[INFO] Reprojecting point with Q matrix...")
    X_raw, Y_raw, Z_raw = reproject_point_with_q(x, y, disp_value, Q)
    estimated_z_cm = convert_z_to_cm(Z_raw)
    error_cm = estimated_z_cm - args.actual_distance_cm
    print(f"X: {X_raw}")
    print(f"Y: {Y_raw}")
    print(f"Z: {Z_raw}")

    clicked_vis = draw_click_visualization(
        left_rect,
        x,
        y,
        args.roi_size,
        estimated_z_cm,
        disp_value,
        args.disp_stat,
    )
    clicked_out_path = clicked_dir / f"{args.trial_name}_clicked.png"
    cv2.imwrite(str(clicked_out_path), clicked_vis)

    row = {
        "trial_name": args.trial_name,
        "actual_distance_cm": round(args.actual_distance_cm, 4),
        "clicked_u": x,
        "clicked_v": y,
        "roi_size": args.roi_size,
        "q1_disparity_px": round(disp_stats["q1"], 6),
        "median_disparity_px": round(disp_stats["median"], 6),
        "q3_disparity_px": round(disp_stats["q3"], 6),
        "used_disp_stat": args.disp_stat,
        "used_disparity_px": round(disp_value, 6),
        "estimated_z_raw": round(Z_raw, 6),
        "estimated_z_cm": round(estimated_z_cm, 6),
        "error_cm": round(error_cm, 6),
    }
    append_csv_row(log_csv_path, row)

    print("\n===== RESULT =====")
    print(f"Trial name         : {args.trial_name}")
    print(f"Actual distance    : {args.actual_distance_cm:.2f} cm")
    print(f"Clicked point      : (u={x}, v={y})")
    print(f"ROI size           : {args.roi_size} x {args.roi_size}")
    print(f"Q1 disparity       : {disp_stats['q1']:.4f} px")
    print(f"Median disparity   : {disp_stats['median']:.4f} px")
    print(f"Q3 disparity       : {disp_stats['q3']:.4f} px")
    print(f"Used disparity     : {args.disp_stat} = {disp_value:.4f} px")
    print(f"Raw Z from Q       : {Z_raw:.4f}")
    print(f"Estimated Z        : {estimated_z_cm:.4f} cm")
    print(f"Error              : {error_cm:+.4f} cm")
    print(f"Saved disparity to : {disparity_out_path}")
    print(f"Saved click vis to : {clicked_out_path}")
    print(f"Logged CSV row to  : {log_csv_path}")


if __name__ == "__main__":
    main()