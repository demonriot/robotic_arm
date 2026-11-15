
import cv2
import yaml
import requests
import numpy as np

from pathlib import Path
from datetime import datetime

# =====================================================
# PATHS AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "monocular.yaml"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "monocular"

HOMOGRAPHY_PATH = OUTPUT_ROOT / "homography.npz"
MAPPING_PATH = OUTPUT_ROOT / "camera_to_robot.npz"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

WINDOW = "EGG_ROBOT - Camera to Robot Mapping"

MIN_POINTS = 3

# =====================================================
# LOAD CONFIGURATION
# =====================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CAMERA_URL = config["camera"]["source"]

CONNECTION_TYPE = config["camera"]["connection_type"].lower()

# =====================================================
# LOAD EXISTING HOMOGRAPHY
# =====================================================

if not HOMOGRAPHY_PATH.exists():
    raise FileNotFoundError(
        f"Homography not found:\n{HOMOGRAPHY_PATH}"
    )

data = np.load(HOMOGRAPHY_PATH)

H = data["H"].astype(np.float64)

CALIBRATION_IMAGE_SIZE = tuple(
    int(x) for x in data["image_size"]
)

print("\nLoaded homography:")
print(H)

print("Calibration image size:", CALIBRATION_IMAGE_SIZE)

# =====================================================
# CAMERA
# =====================================================

def open_camera():

    if CONNECTION_TYPE == "mjpeg":

        cap = cv2.VideoCapture(CAMERA_URL)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera: {CAMERA_URL}"
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    if CONNECTION_TYPE == "snapshot":
        return None

    raise ValueError(
        f"Unsupported connection type: {CONNECTION_TYPE}"
    )


def get_frame(cap):

    if CONNECTION_TYPE == "mjpeg":

        ok, frame = cap.read()

        if not ok:
            return None

        return frame

    try:
        response = requests.get(
            CAMERA_URL,
            timeout=5,
            headers={"Cache-Control": "no-cache"}
        )

        response.raise_for_status()

        image_bytes = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        return cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

    except requests.RequestException as exc:

        print("Camera error:", exc)

        return None


# =====================================================
# COORDINATE TRANSFORMS
# =====================================================

def pixel_to_checkerboard(u, v):

    point = np.array(
        [[[float(u), float(v)]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        H
    )

    return transformed[0, 0].astype(np.float64)


def checkerboard_to_robot(xc, yc, affine_matrix):

    point = np.array(
        [float(xc), float(yc), 1.0],
        dtype=np.float64
    )

    result = affine_matrix @ point

    return float(result[0]), float(result[1])


def pixel_to_robot(u, v, affine_matrix):

    xc, yc = pixel_to_checkerboard(u, v)

    return checkerboard_to_robot(
        xc,
        yc,
        affine_matrix
    )


# =====================================================
# RED OBJECT DETECTION
# =====================================================

def detect_red(frame):

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask1 = cv2.inRange(
        hsv,
        np.array([0, 100, 80]),
        np.array([10, 255, 255])
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([170, 100, 80]),
        np.array([179, 255, 255])
    )

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(contour) < 300:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    u = int(
        moments["m10"] / moments["m00"]
    )

    v = int(
        moments["m01"] / moments["m00"]
    )

    return (u, v)


# =====================================================
# REFERENCE POINT COLLECTION
# =====================================================

camera_points = []
robot_points = []

pending_clicks = []

mapping_matrix = None


def mouse_callback(event, x, y, flags, userdata):

    if event == cv2.EVENT_LBUTTONDOWN:
        pending_clicks.append((x, y))


def fit_mapping():

    if len(camera_points) < MIN_POINTS:
        print("At least three points are required.")
        return None

    source = np.array(
        camera_points,
        dtype=np.float64
    )

    target = np.array(
        robot_points,
        dtype=np.float64
    )

    # Fit:
    #
    # X = a*Xc + b*Yc + tx
    # Y = c*Xc + d*Yc + ty

    design = np.column_stack(
        [
            source[:, 0],
            source[:, 1],
            np.ones(len(source))
        ]
    )

    if np.linalg.matrix_rank(design) < 3:
        print(
            "Reference points are collinear or too close."
        )
        return None

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        target,
        rcond=None
    )

    affine = coefficients.T

    predictions = design @ coefficients

    errors = np.linalg.norm(
        predictions - target,
        axis=1
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    print("\nMapping matrix:")
    print(affine)

    print("\nPoint errors (cm):")
    for i, error in enumerate(errors, start=1):
        print(f"  Point {i}: {error:.3f}")

    print(f"\nTraining RMSE: {rmse:.3f} cm")

    if len(source) == 3:
        print(
            "Note: with exactly three points, the affine "
            "model can fit perfectly even when measurements "
            "are inaccurate. RMSE is not independent validation."
        )

    return affine


def save_mapping(affine):

    np.savez(
        MAPPING_PATH,
        A=affine,
        H=H,
        camera_points=np.array(
            camera_points,
            dtype=np.float64
        ),
        robot_points=np.array(
            robot_points,
            dtype=np.float64
        ),
        image_size=np.array(
            CALIBRATION_IMAGE_SIZE
        ),
        timestamp=datetime.now().isoformat()
    )

    print("\nMapping saved:")
    print(MAPPING_PATH)


# =====================================================
# DISPLAY
# =====================================================

def draw_display(frame):

    display = frame.copy()

    for i, (u, v) in enumerate(
        pending_reference_pixels,
        start=1
    ):

        cv2.circle(
            display,
            (u, v),
            5,
            (255, 255, 0),
            -1
        )

        cv2.putText(
            display,
            str(i),
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

    cv2.putText(
        display,
        f"Points: {len(camera_points)} | "
        "Click marker to add",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    h, w = display.shape[:2]

    cv2.putText(
        display,
        "F: fit/save | U: undo | Q: quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    if mapping_matrix is not None:

        object_pixel = detect_red(frame)

        if object_pixel is not None:

            u, v = object_pixel

            x, y = pixel_to_robot(
                u,
                v,
                mapping_matrix
            )

            cv2.circle(
                display,
                (u, v),
                7,
                (0, 255, 0),
                2
            )

            label = (
                f"Robot X={x:.1f} Y={y:.1f} cm"
            )

            text_y = max(50, v - 15)

            cv2.putText(
                display,
                label,
                (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    return display


# Separate list for drawing clicked points
pending_reference_pixels = []


# =====================================================
# MAIN
# =====================================================

def main():

    global mapping_matrix

    print("\nEGG_ROBOT - Camera to Robot Calibration")
    print("--------------------------------------")

    print("Camera:", CAMERA_URL)
    print("Homography:", HOMOGRAPHY_PATH)

    print("\nInstructions:")
    print("1. Place at least 3 markers on the tabletop.")
    print("2. Measure each marker's robot X/Y in cm.")
    print("3. Click each marker in the camera window.")
    print("4. Enter the measured X/Y in the terminal.")
    print("5. Press F to calculate and save mapping.")
    print("6. Press Q to exit.")

    cap = open_camera()

    try:

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            WINDOW,
            mouse_callback
        )

        while True:

            frame = get_frame(cap)

            if frame is None:
                print("No camera frame.")
                continue

            h, w = frame.shape[:2]

            # OpenCV reports frame size as width, height.
            if (w, h) != CALIBRATION_IMAGE_SIZE:
                raise RuntimeError(
                    "Camera resolution changed.\n"
                    f"Calibration: {CALIBRATION_IMAGE_SIZE}\n"
                    f"Current: {(w, h)}"
                )

            display = draw_display(frame)

            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("u"):

                if camera_points:

                    camera_points.pop()
                    robot_points.pop()
                    pending_reference_pixels.pop()

                    mapping_matrix = None

                    print("Last reference point removed.")

            elif key == ord("f"):

                affine = fit_mapping()

                if affine is not None:

                    mapping_matrix = affine
                    save_mapping(affine)

                    print(
                        "\nMapping active. "
                        "Red object robot coordinates "
                        "will now appear in the image."
                    )

            while pending_clicks:

                u, v = pending_clicks.pop(0)

                xc, yc = pixel_to_checkerboard(
                    u,
                    v
                )

                print(
                    f"\nClicked pixel: ({u}, {v})"
                )

                print(
                    f"Checkerboard coordinates: "
                    f"({xc:.2f}, {yc:.2f}) cm"
                )

                try:

                    x = float(
                        input("Robot X (cm): ")
                    )

                    y = float(
                        input("Robot Y (cm): ")
                    )

                except ValueError:

                    print("Invalid values. Point ignored.")
                    continue

                camera_points.append(
                    (xc, yc)
                )

                robot_points.append(
                    (x, y)
                )

                pending_reference_pixels.append(
                    (u, v)
                )

                mapping_matrix = None

                print(
                    f"Point {len(camera_points)} recorded:"
                    f" pixel=({u},{v}), "
                    f"robot=({x:.2f},{y:.2f})"
                )

    finally:

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
