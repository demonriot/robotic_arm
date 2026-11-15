
import cv2
import yaml
import requests
import numpy as np

from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "monocular.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

CAMERA_URL = config["camera"]["source"]
CONNECTION_TYPE = config["camera"]["connection_type"].lower()

CAPTURE_DIR = PROJECT_ROOT / config["paths"]["capture_dir"]
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Camera connection
# --------------------------------------------------

def open_camera():
    if CONNECTION_TYPE == "mjpeg":
        cap = cv2.VideoCapture(CAMERA_URL)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot connect to camera: {CAMERA_URL}"
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    if CONNECTION_TYPE == "snapshot":
        return None

    raise ValueError(
        "connection_type must be 'mjpeg' or 'snapshot'"
    )


def read_frame(cap):
    if CONNECTION_TYPE == "mjpeg":
        ret, frame = cap.read()

        if not ret or frame is None:
            return None

        return frame

    try:
        response = requests.get(
            CAMERA_URL,
            timeout=5,
            headers={"Cache-Control": "no-cache"}
        )
        response.raise_for_status()

        data = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    except requests.RequestException as exc:
        print(f"Camera request failed: {exc}")
        return None


# --------------------------------------------------
# Capture images
# --------------------------------------------------

def save_image(frame):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = CAPTURE_DIR / f"calib_{timestamp}.png"

    if not cv2.imwrite(str(filename), frame):
        print("Failed to save image")
        return

    print(f"Saved: {filename}")


# --------------------------------------------------
# Main loop
# --------------------------------------------------

def main():
    print(f"Connecting to: {CAMERA_URL}")

    cap = open_camera()

    print("\nControls:")
    print("SPACE / ENTER : Capture image")
    print("Q / ESC       : Quit\n")

    cv2.namedWindow(
        "Monocular IP Camera",
        cv2.WINDOW_NORMAL
    )

    try:
        while True:
            frame = read_frame(cap)

            if frame is None:
                print("Frame unavailable")
                key = cv2.waitKey(100) & 0xFF

                if key in (ord("q"), 27):
                    break

                continue

            display = frame.copy()

            cv2.putText(
                display,
                "SPACE: Capture | Q: Quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Monocular IP Camera", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (32, 13):
                save_image(frame)

            elif key in (ord("q"), 27):
                break

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
