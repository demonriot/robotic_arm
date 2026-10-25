
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np
import requests
import yaml


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "stereo.yaml"

LEFT_DIR = PROJECT_ROOT / "data" / "calibration" / "left"
RIGHT_DIR = PROJECT_ROOT / "data" / "calibration" / "right"

LEFT_DIR.mkdir(parents=True, exist_ok=True)
RIGHT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

REQUEST_TIMEOUT = 10
PREVIEW_WIDTH = 1200

WINDOW_NAME = "Stereo Calibration - Live Preview"

# ==========================================
# LOAD CONFIG
# ==========================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

LEFT_URL = config["camera"]["left_id"]
RIGHT_URL = config["camera"]["right_id"]

EXPECTED_WIDTH = int(config["camera"]["width"])
EXPECTED_HEIGHT = int(config["camera"]["height"])


# ==========================================
# CAPTURE HELPERS
# ==========================================

def fetch_image(url):
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"Cache-Control": "no-cache"},
    )

    response.raise_for_status()

    image_array = np.frombuffer(
        response.content,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to decode image from {url}"
        )

    if image.shape[:2] != (
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
    ):
        raise RuntimeError(
            f"Unexpected image size: "
            f"{image.shape[1]}x{image.shape[0]}"
        )

    return image


def capture_pair(executor):
    """
    Request both ESP32 cameras concurrently.
    """

    left_future = executor.submit(
        fetch_image,
        LEFT_URL,
    )

    right_future = executor.submit(
        fetch_image,
        RIGHT_URL,
    )

    left_img = left_future.result()
    right_img = right_future.result()

    return left_img, right_img


# ==========================================
# PAIR NUMBERING
# ==========================================

def next_pair_number():
    existing = set()

    for directory in (LEFT_DIR, RIGHT_DIR):
        for path in directory.glob("pair_*.png"):
            try:
                number = int(
                    path.stem.split("_")[-1]
                )
                existing.add(number)
            except ValueError:
                pass

    return max(existing, default=0) + 1


# ==========================================
# PREVIEW
# ==========================================

def make_preview(
    left_img,
    right_img,
    pair_number,
    paused=False,
):
    preview = np.hstack(
        (left_img, right_img)
    )

    scale = min(
        1.0,
        PREVIEW_WIDTH / preview.shape[1],
    )

    if scale < 1.0:
        preview = cv2.resize(
            preview,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    height, width = preview.shape[:2]

    # Dark header strip
    overlay = preview.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 48),
        (0, 0, 0),
        -1,
    )

    preview = cv2.addWeighted(
        overlay,
        0.65,
        preview,
        0.35,
        0,
    )

    status = "PAUSED" if paused else "LIVE"

    text = (
        f"{status} | Next pair: {pair_number:03d} | "
        "SPACE/ENTER: Save | P: Pause | Q: Quit"
    )

    cv2.putText(
        preview,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        "LEFT",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        preview,
        "RIGHT",
        (width // 2 + 15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    return preview


# ==========================================
# SAVE
# ==========================================

def save_pair(left_img, right_img, pair_number):
    left_path = (
        LEFT_DIR / f"pair_{pair_number:03d}.png"
    )

    right_path = (
        RIGHT_DIR / f"pair_{pair_number:03d}.png"
    )

    if left_path.exists() or right_path.exists():
        raise FileExistsError(
            f"Pair {pair_number:03d} already exists."
        )

    left_ok = cv2.imwrite(
        str(left_path),
        left_img,
    )

    right_ok = cv2.imwrite(
        str(right_path),
        right_img,
    )

    if not (left_ok and right_ok):
        # Avoid leaving behind a partial stereo pair.
        if left_path.exists():
            left_path.unlink()

        if right_path.exists():
            right_path.unlink()

        raise RuntimeError(
            "Failed to save stereo pair."
        )

    print(
        f"Saved pair {pair_number:03d} "
        f"({EXPECTED_WIDTH}x{EXPECTED_HEIGHT})"
    )


# ==========================================
# MAIN
# ==========================================

def main():
    print("===================================")
    print("LIVE STEREO CALIBRATION CAPTURE")
    print("===================================")

    print(f"LEFT:  {LEFT_URL}")
    print(f"RIGHT: {RIGHT_URL}")

    print(
        f"Resolution: "
        f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
    )

    print("\nControls:")
    print("SPACE / ENTER : Capture and save")
    print("P             : Pause/resume")
    print("Q / ESC       : Quit")

    pair_number = next_pair_number()

    paused = False
    last_left = None
    last_right = None

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:

                if not paused:
                    try:
                        left_img, right_img = capture_pair(
                            executor
                        )

                        last_left = left_img
                        last_right = right_img

                    except Exception as exc:
                        print(f"Preview error: {exc}")

                if last_left is not None:
                    preview = make_preview(
                        last_left,
                        last_right,
                        pair_number,
                        paused,
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        preview,
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("p"):
                    paused = not paused
                    print(
                        "Preview paused."
                        if paused
                        else "Preview resumed."
                    )

                if key in (32, 13, 10):
                    if last_left is None:
                        print(
                            "No valid camera frames yet."
                        )
                        continue

                    # Acquire a fresh pair rather than
                    # saving potentially stale preview
                    # frames.
                    try:
                        print(
                            f"Capturing pair "
                            f"{pair_number:03d}..."
                        )

                        left_img, right_img = capture_pair(
                            executor
                        )

                        save_pair(
                            left_img,
                            right_img,
                            pair_number,
                        )

                        last_left = left_img
                        last_right = right_img

                        pair_number += 1

                    except Exception as exc:
                        print(
                            f"Capture failed: {exc}"
                        )

        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
