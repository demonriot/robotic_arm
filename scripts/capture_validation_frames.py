import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a synchronized stereo frame pair for depth validation."
    )
    parser.add_argument(
        "--left-cam-index",
        type=int,
        default=None,
        help="OpenCV camera index for the left camera. Default loops to yaml",
    )
    parser.add_argument(
        "--right-cam-index",
        type=int,
        default=None,
        help="OpenCV camera index for the right camera. Default loops to yaml",
    )
    parser.add_argument(
        "--trial-name",
        type=str,
        required=True,
        help='Trial name, e.g. "158cm_trial1"',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory. Defaults to stereo_cam_validation/captured_validation_frames",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Requested capture width. Default loops to yaml.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Requested capture height. Default loops to yaml.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Number of warmup frames before capture. Default: 20",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown seconds before capture after pressing 'c'. Default: 3",
    )
    return parser.parse_args()


def create_output_dirs(base_dir: Path) -> tuple[Path, Path]:
    left_dir = base_dir / "left"
    right_dir = base_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return left_dir, right_dir


def configure_camera(cap: cv2.VideoCapture, width: int, height: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def grab_frame(cap: cv2.VideoCapture, cam_name: str) -> any:
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame from {cam_name}.")
    return frame


def put_overlay(
    left_frame,
    right_frame,
    trial_name: str,
    message: str,
) -> tuple:
    left_vis = left_frame.copy()
    right_vis = right_frame.copy()

    cv2.putText(
        left_vis,
        f"LEFT | Trial: {trial_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        right_vis,
        f"RIGHT | Trial: {trial_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        left_vis,
        message,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        right_vis,
        message,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return left_vis, right_vis


def stack_frames(left_frame, right_frame):
    if left_frame.shape[:2] != right_frame.shape[:2]:
        right_frame = cv2.resize(
            right_frame, (left_frame.shape[1], left_frame.shape[0])
        )
    return cv2.hconcat([left_frame, right_frame])


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    import yaml
    config_path = base_dir / 'configs' / 'stereo.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    cam_l_id = args.left_cam_index if args.left_cam_index is not None else config['camera']['left_id']
    cam_r_id = args.right_cam_index if args.right_cam_index is not None else config['camera']['right_id']
    cam_w = args.width if args.width is not None else config['camera']['width']
    cam_h = args.height if args.height is not None else config['camera']['height']
    base_output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir.parent / "outputs" / "validation" / "captured_frames"
    )

    left_dir, right_dir = create_output_dirs(base_output_dir)

    left_path = left_dir / f"left_{args.trial_name}.png"
    right_path = right_dir / f"right_{args.trial_name}.png"

    print("[INFO] Opening cameras...")
    cap_left = cv2.VideoCapture(cam_l_id)
    cap_right = cv2.VideoCapture(cam_r_id)

    if not cap_left.isOpened():
        raise RuntimeError(
            f"Could not open left camera at index {cam_l_id}."
        )
    if not cap_right.isOpened():
        raise RuntimeError(
            f"Could not open right camera at index {cam_r_id}."
        )

    configure_camera(cap_left, cam_w, cam_h)
    configure_camera(cap_right, cam_w, cam_h)

    print("[INFO] Warming up cameras...")
    for _ in range(args.warmup_frames):
        _ = cap_left.read()
        _ = cap_right.read()

    print("[INFO] Press 'c' to capture, or 'q' / ESC to quit.")

    try:
        while True:
            left_frame = grab_frame(cap_left, "left camera")
            right_frame = grab_frame(cap_right, "right camera")

            left_vis, right_vis = put_overlay(
                left_frame,
                right_frame,
                args.trial_name,
                "Press 'c' to capture | 'q' to quit",
            )

            combined = stack_frames(left_vis, right_vis)
            cv2.namedWindow("Stereo Validation Capture", cv2.WINDOW_NORMAL)
            cv2.imshow("Stereo Validation Capture", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                print("[INFO] Exiting without saving.")
                break

            if key == ord("c"):
                for sec in range(args.countdown, 0, -1):
                    left_frame = grab_frame(cap_left, "left camera")
                    right_frame = grab_frame(cap_right, "right camera")

                    left_vis, right_vis = put_overlay(
                        left_frame,
                        right_frame,
                        args.trial_name,
                        f"Capturing in {sec}...",
                    )
                    combined = stack_frames(left_vis, right_vis)
                    cv2.imshow("Stereo Validation Capture", combined)
                    cv2.waitKey(1)
                    time.sleep(1)

                # Grab frames back-to-back as the saved pair
                final_left = grab_frame(cap_left, "left camera")
                final_right = grab_frame(cap_right, "right camera")

                cv2.imwrite(str(left_path), final_left)
                cv2.imwrite(str(right_path), final_right)

                print("[INFO] Saved stereo validation pair:")
                print(f"       Left : {left_path}")
                print(f"       Right: {right_path}")

                saved_left = final_left.copy()
                saved_right = final_right.copy()

                cv2.putText(
                    saved_left,
                    "SAVED",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    saved_right,
                    "SAVED",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                combined_saved = stack_frames(saved_left, saved_right)
                cv2.imshow("Stereo Validation Capture", combined_saved)
                cv2.waitKey(1200)
                break

    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()