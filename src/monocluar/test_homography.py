
import cv2
import yaml
import requests
import numpy as np

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

with open(ROOT / "configs/monocular.yaml", "r") as f:
    config = yaml.safe_load(f)

CAMERA_URL = config["camera"]["source"]
CONNECTION_TYPE = config["camera"]["connection_type"].lower()

PARAMS_PATH = ROOT / "outputs/monocular/homography.npz"

params = np.load(PARAMS_PATH)

H = params["H"]
calib_width, calib_height = map(int, params["image_size"])

WINDOW_NAME = "Monocular - World Coordinates"

selected_point = None


def pixel_to_world(u, v):
    point = np.array(
        [[[u, v]]],
        dtype=np.float32
    )

    world = cv2.perspectiveTransform(point, H)[0, 0]

    return float(world[0]), float(world[1])


def mouse_callback(event, x, y, flags, userdata):
    global selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = pixel_to_world(x, y)

        selected_point = (x, y, X, Y)

        print(
            f"Pixel: ({x}, {y}) "
            f"-> World: ({X:.2f}, {Y:.2f}) cm"
        )


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
                f"Could not open {CAMERA_URL}"
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    print("Left-click to display world coordinates.")
    print("Press Q or ESC to quit.")

    try:
        while True:
            if CONNECTION_TYPE == "mjpeg":
                ret, frame = cap.read()

                if not ret:
                    print("Could not read frame")
                    break
            else:
                frame = read_snapshot()

                if frame is None:
                    continue

            height, width = frame.shape[:2]

            if (width, height) != (
                calib_width, calib_height
            ):
                raise RuntimeError(
                    "Camera resolution changed! "
                    f"Expected {calib_width}x{calib_height}, "
                    f"got {width}x{height}."
                )

            display = frame.copy()

            if selected_point is not None:
                u, v, X, Y = selected_point

                cv2.circle(
                    display,
                    (u, v),
                    6,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    display,
                    f"X={X:.2f} cm, Y={Y:.2f} cm",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
