
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "configs" / "stereo.yaml"

LEFT_DIR = BASE_DIR / "data" / "calibration" / "left"
RIGHT_DIR = BASE_DIR / "data" / "calibration" / "right"
PREVIEW_DIR = BASE_DIR / "outputs" / "calibration"

LEFT_DIR.mkdir(parents=True, exist_ok=True)
RIGHT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def capture_image(url):
    # Accept either /capture or /stream in the YAML file
    base_url = url.rsplit("/", 1)[0]
    capture_url = base_url + "/capture"

    response = requests.get(capture_url, timeout=10)
    response.raise_for_status()

    image = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(f"Invalid image from {capture_url}")

    return image


def capture_pair():
    left_url = config["camera"]["left_id"]
    right_url = config["camera"]["right_id"]

    left = capture_image(left_url)
    right = capture_image(right_url)

    if left.shape != right.shape:
        raise ValueError(
            f"Resolution mismatch: {left.shape} vs {right.shape}"
        )

    return left, right


def main():
    print("ESP32-CAM Stereo Calibration Capture")
    print("-----------------------------------")
    print("Press ENTER to capture a pair.")
    print("Type 'a' to automatically capture every 3 seconds.")
    print("Type 'q' to quit.")

    auto_mode = False

    existing = sorted(LEFT_DIR.glob("pair_*.png"))
    pair_number = len(existing) + 1

    while True:
        command = input("\nCommand [ENTER/a/q]: ").strip().lower()

        if command == "q":
            break

        if command == "a":
            auto_mode = not auto_mode
            print(f"Auto mode: {'ON' if auto_mode else 'OFF'}")

        if auto_mode:
            print("Capturing automatically. Press Ctrl+C to stop.")

        try:
            while True:
                left, right = capture_pair()

                filename = f"pair_{pair_number:03d}.png"

                cv2.imwrite(str(LEFT_DIR / filename), left)
                cv2.imwrite(str(RIGHT_DIR / filename), right)

                preview = cv2.hconcat([left, right])
                cv2.imwrite(
                    str(PREVIEW_DIR / "capture_preview.jpg"),
                    preview
                )

                print(f"Saved stereo pair #{pair_number}")
                pair_number += 1

                if not auto_mode:
                    break

                time.sleep(3)

        except KeyboardInterrupt:
            auto_mode = False
            print("\nAuto capture stopped.")

        except Exception as e:
            auto_mode = False
            print(f"Capture error: {e}")

    print("Capture session finished.")


if __name__ == "__main__":
    main()
