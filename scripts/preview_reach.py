
from pathlib import Path
import math

import cv2
import numpy as np
import requests
import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "monocular.yaml"
OUTPUT_DIR = ROOT / "outputs" / "monocular"

HOMOGRAPHY_PATH = OUTPUT_DIR / "homography.npz"
MAPPING_PATH = OUTPUT_DIR / "camera_to_robot.npz"


# ============================================================
# ROBOT GEOMETRY (cm)
# Provisional model, calibrated at one measured pose.
# ============================================================

L1 = 12.0
L2 = 12.0

SHOULDER_HEIGHT = 15.0
GRIPPER_RADIAL_OFFSET = 4.12
GRIPPER_VERTICAL_OFFSET = -7.72

TARGET_Z = 8.0

STEPS_PER_DEGREE = 35.5555556

# Last physically measured reference:
# Gripper = (25, 3.3, 4.5) cm
# Shoulder = +20 degrees
# Forearm = -35 degrees
# Relative elbow = -55 degrees

REF_BASE = math.degrees(math.atan2(3.3, 25.0))
REF_SHOULDER = 20.0
REF_ELBOW = -55.0


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():
    with np.load(HOMOGRAPHY_PATH) as data:
        H = np.asarray(data["H"], dtype=np.float64)
        image_size = tuple(int(v) for v in data["image_size"])

    with np.load(MAPPING_PATH) as data:
        A = np.asarray(data["A"], dtype=np.float64)

        if "H" in data:
            saved_H = np.asarray(data["H"], dtype=np.float64)

            if not np.allclose(saved_H, H):
                raise RuntimeError(
                    "Mapping and homography do not match."
                )

    return H, A, image_size


def pixel_to_robot(u, v, H, A):
    pixel = np.array([[[u, v]]], dtype=np.float64)

    checkerboard_xy = cv2.perspectiveTransform(
        pixel, H
    )[0, 0]

    point = np.array([
        checkerboard_xy[0],
        checkerboard_xy[1],
        1.0
    ])

    result = A @ point

    return float(result[0]), float(result[1])


# ============================================================
# RED OBJECT DETECTION
# ============================================================

def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 300:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    u = int(moments["m10"] / moments["m00"])
    v = int(moments["m01"] / moments["m00"])

    return u, v, contour


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def inverse_kinematics(x, y, z):
    base = math.degrees(math.atan2(y, x))

    radial = math.hypot(x, y)

    wrist_r = radial - GRIPPER_RADIAL_OFFSET

    wrist_z = (
        z
        - SHOULDER_HEIGHT
        - GRIPPER_VERTICAL_OFFSET
    )

    D = (
        wrist_r**2
        + wrist_z**2
        - L1**2
        - L2**2
    ) / (2 * L1 * L2)

    if D < -1.0 or D > 1.0:
        return None

    # Elbow-down branch, matching the measured pose.
    elbow = -math.acos(max(-1.0, min(1.0, D)))

    shoulder = (
        math.atan2(wrist_z, wrist_r)
        - math.atan2(
            L2 * math.sin(elbow),
            L1 + L2 * math.cos(elbow)
        )
    )

    return {
        "base": base,
        "shoulder": math.degrees(shoulder),
        "elbow": math.degrees(elbow)
    }


def angle_changes_to_steps(angles):
    # Relative to the measured reference pose.
    # These are DELTAS, not absolute session targets.

    shoulder_steps = round(
        -(angles["shoulder"] - REF_SHOULDER)
        * STEPS_PER_DEGREE
    )

    elbow_steps = round(
        (angles["elbow"] - REF_ELBOW)
        * STEPS_PER_DEGREE
    )

    base_steps = round(
        (angles["base"] - REF_BASE)
        * STEPS_PER_DEGREE
    )

    return shoulder_steps, elbow_steps, base_steps


# ============================================================
# CAMERA
# ============================================================

class Camera:
    def __init__(self, camera_config):
        self.source = camera_config["source"]
        self.mode = camera_config.get(
            "connection_type", "mjpeg"
        ).lower()

        self.cap = None
        self.session = None

        if self.mode == "snapshot":
            self.session = requests.Session()
        else:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise RuntimeError("Camera stream unavailable.")

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self.mode == "snapshot":
            try:
                response = self.session.get(
                    self.source, timeout=3
                )
                response.raise_for_status()

                array = np.frombuffer(
                    response.content, np.uint8
                )

                frame = cv2.imdecode(
                    array, cv2.IMREAD_COLOR
                )

                return frame is not None, frame

            except requests.RequestException:
                return False, None

        return self.cap.read()

    def close(self):
        if self.cap is not None:
            self.cap.release()

        if self.session is not None:
            self.session.close()


# ============================================================
# MAIN
# ============================================================

def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    H, A, image_size = load_calibration()
    camera = Camera(config["camera"])

    print("\nEGG_ROBOT - IK Reach Preview")
    print("--------------------------------")
    print("No Arduino connection. No motor movement.")
    print(f"Target height: {TARGET_Z:.1f} cm")
    print("Press P to print current target.")
    print("Press Q to quit.")

    latest = None

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                key = cv2.waitKey(20) & 0xFF

                if key == ord("q"):
                    break

                continue

            h, w = frame.shape[:2]

            if (w, h) != image_size:
                raise RuntimeError(
                    f"Expected {image_size}, got {(w, h)}"
                )

            detection = detect_red(frame)
            latest = None

            cv2.rectangle(
                frame, (0, 0), (w, 125),
                (25, 25, 25), -1
            )

            cv2.putText(
                frame,
                "EGG_ROBOT - IK PREVIEW ONLY",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )

            if detection is not None:
                u, v, contour = detection

                x, y = pixel_to_robot(u, v, H, A)

                angles = inverse_kinematics(
                    x, y, TARGET_Z
                )

                cv2.drawContours(
                    frame, [contour], -1,
                    (0, 255, 0), 2
                )

                cv2.circle(
                    frame, (u, v), 5,
                    (255, 255, 255), -1
                )

                cv2.putText(
                    frame,
                    f"Object X={x:.2f} Y={y:.2f} cm",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (100, 255, 100), 2
                )

                if angles is not None:
                    steps = angle_changes_to_steps(angles)

                    latest = (x, y, angles, steps)

                    cv2.putText(
                        frame,
                        "IK: REACHABLE (geometrically)",
                        (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (100, 255, 100), 2
                    )

                    cv2.putText(
                        frame,
                        "Press P for joint targets",
                        (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1
                    )

                else:
                    cv2.putText(
                        frame,
                        "IK: UNREACHABLE",
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 100, 255), 2
                    )

            else:
                cv2.putText(
                    frame,
                    "Red object not detected",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 150, 255), 2
                )

            cv2.imshow("EGG_ROBOT - Reach Preview", frame)

            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("p"):
                if latest is None:
                    print("\nNo reachable target detected.")
                    continue

                x, y, angles, steps = latest

                print("\n" + "=" * 45)
                print("REACH PREVIEW")
                print("=" * 45)

                print(
                    f"Target XYZ: "
                    f"({x:.2f}, {y:.2f}, {TARGET_Z:.2f}) cm"
                )

                print("\nCalculated joint angles:")
                print(f"  Base:     {angles['base']:.2f} deg")
                print(
                    f"  Shoulder: {angles['shoulder']:.2f} deg"
                )
                print(f"  Elbow:    {angles['elbow']:.2f} deg")

                print("\nMotor-step changes from measured pose:")
                print(f"  Shoulder X: {steps[0]:+d}")
                print(f"  Elbow Y:    {steps[1]:+d}")
                print(f"  Base Z:     {steps[2]:+d}")

                print(
                    "\nNOTE: These are relative to the "
                    "measured reference pose, not the "
                    "current Arduino session zero."
                )

                print(
                    "Geometric reachability does not "
                    "guarantee collision-free movement."
                )

    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
