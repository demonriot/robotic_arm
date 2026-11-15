
import cv2
import yaml
import numpy as np
import requests

from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs/monocular.yaml", "r") as f:
    config = yaml.safe_load(f)

CAMERA_URL = config["camera"]["source"]
CONNECTION_TYPE = config["camera"]["connection_type"].lower()

PARAMS_PATH = ROOT / "outputs/monocular/homography.npz"

params = np.load(PARAMS_PATH)

H = params["H"]
CALIB_WIDTH, CALIB_HEIGHT = map(int, params["image_size"])

WINDOW_NAME = "EGG_ROBOT - Object Detection"

MIN_AREA = 300


# --------------------------------------------------
# Pixel to world coordinates
# --------------------------------------------------

def pixel_to_world(u, v):
    point = np.array(
        [[[u, v]]],
        dtype=np.float32
    )

    world = cv2.perspectiveTransform(point, H)[0, 0]

    return float(world[0]), float(world[1])


# --------------------------------------------------
# Red object detection
# --------------------------------------------------

def detect_red_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around the HSV hue scale.
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(
        hsv, lower_red1, upper_red1
    )

    mask2 = cv2.inRange(
        hsv, lower_red2, upper_red2
    )

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    if area < MIN_AREA:
        return None, mask

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None, mask

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return {
        "center": (cx, cy),
        "contour": contour,
        "area": area
    }, mask


# --------------------------------------------------
# Camera
# --------------------------------------------------

def read_snapshot():
    try:
        response = requests.get(
            CAMERA_URL,
            timeout=5,
            headers={"Cache-Control": "no-cache"}
        )

        response.raise_for_status()

        data = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

    except requests.RequestException as exc:
        print(f"Snapshot error: {exc}")
        return None


def main():
    cap = None

    if CONNECTION_TYPE == "mjpeg":
        cap = cv2.VideoCapture(CAMERA_URL)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot connect to {CAMERA_URL}"
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE
    )

    print("EGG_ROBOT Object Detection")
    print("Q / ESC : Quit")
    print("M       : Toggle mask window")

    show_mask = False

    try:
        while True:
            if CONNECTION_TYPE == "mjpeg":
                ret, frame = cap.read()

                if not ret:
                    print("Camera frame unavailable")
                    break
            else:
                frame = read_snapshot()

                if frame is None:
                    continue

            height, width = frame.shape[:2]

            if (width, height) != (
                CALIB_WIDTH, CALIB_HEIGHT
            ):
                raise RuntimeError(
                    f"Resolution mismatch: {width}x{height}. "
                    f"Expected {CALIB_WIDTH}x{CALIB_HEIGHT}"
                )

            detection, mask = detect_red_object(frame)

            display = frame.copy()

            if detection is not None:
                cx, cy = detection["center"]
                contour = detection["contour"]

                X, Y = pixel_to_world(cx, cy)

                cv2.drawContours(
                    display,
                    [contour],
                    -1,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    display,
                    (cx, cy),
                    5,
                    (255, 255, 0),
                    -1
                )

                label = f"X={X:.2f} Y={Y:.2f} cm"

                cv2.putText(
                    display,
                    label,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"Pixel=({cx},{cy})",
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    display,
                    "No red object detected",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(WINDOW_NAME, display)

            if show_mask:
                cv2.imshow("Red Mask", mask)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("m"):
                show_mask = not show_mask

                if not show_mask:
                    cv2.destroyWindow("Red Mask")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
