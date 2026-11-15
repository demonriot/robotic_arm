
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import serial
import yaml


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "monocular.yaml"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "monocular"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

HOMOGRAPHY_PATH = OUTPUT_ROOT / "homography.npz"
MAPPING_PATH = OUTPUT_ROOT / "camera_to_robot.npz"
RECORD_PATH = OUTPUT_ROOT / "robot_samples.csv"

SERIAL_PORT = "COM9"
SERIAL_BAUD = 115200

DEFAULT_JOG_STEPS = 50
MAX_SESSION_STEPS = 1400

WINDOW_NAME = "EGG_ROBOT - Camera and Robot Control"

# Existing, tested control directions
COMMANDS = {
    "UP": ("X", -1),
    "DOWN": ("X", +1),
    "LEFT": ("Z", +1),
    "RIGHT": ("Z", -1),
    "E+": ("Y", +1),
    "E-": ("Y", -1),
}

# Red object detection
RED_LOW_1 = np.array([0, 100, 80], dtype=np.uint8)
RED_HIGH_1 = np.array([10, 255, 255], dtype=np.uint8)

RED_LOW_2 = np.array([170, 100, 80], dtype=np.uint8)
RED_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 300


# ============================================================
# CAMERA
# ============================================================

class Camera:
    def __init__(self, config):
        camera_config = config["camera"]

        self.source = camera_config["source"]
        self.connection_type = camera_config.get(
            "connection_type", "mjpeg"
        ).lower()

        self.cap = None
        self.session = None

        if self.connection_type == "snapshot":
            self.session = requests.Session()
        else:
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera stream: {self.source}"
                )

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self.connection_type == "snapshot":
            try:
                response = self.session.get(
                    self.source,
                    timeout=3
                )
                response.raise_for_status()

                array = np.frombuffer(
                    response.content,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

                return frame is not None, frame

            except requests.RequestException:
                return False, None

        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

        if self.session is not None:
            self.session.close()


# ============================================================
# CALIBRATED CAMERA -> ROBOT MAPPING
# ============================================================

class CameraRobotMapping:
    def __init__(self):
        if not HOMOGRAPHY_PATH.exists():
            raise FileNotFoundError(
                f"Homography not found: {HOMOGRAPHY_PATH}"
            )

        if not MAPPING_PATH.exists():
            raise FileNotFoundError(
                f"Camera-to-robot mapping not found: {MAPPING_PATH}"
            )

        with np.load(HOMOGRAPHY_PATH) as data:
            self.H = np.asarray(data["H"], dtype=np.float64)
            self.image_size = tuple(
                int(v) for v in data["image_size"]
            )

        with np.load(MAPPING_PATH) as data:
            self.A = np.asarray(data["A"], dtype=np.float64)

            if "image_size" in data:
                mapping_image_size = tuple(
                    int(v) for v in data["image_size"]
                )

                if mapping_image_size != self.image_size:
                    raise RuntimeError(
                        "Mapping and homography image sizes differ."
                    )

            if "H" in data:
                mapping_H = np.asarray(
                    data["H"],
                    dtype=np.float64
                )

                if not np.allclose(
                    mapping_H,
                    self.H,
                    rtol=1e-5,
                    atol=1e-7
                ):
                    raise RuntimeError(
                        "The mapping was created using a different "
                        "homography. Refit the camera-to-robot mapping."
                    )

        if self.H.shape != (3, 3):
            raise ValueError("Homography H must be 3x3.")

        if self.A.shape != (2, 3):
            raise ValueError("Affine mapping A must be 2x3.")

    def pixel_to_robot(self, u, v):
        pixel = np.array(
            [[[float(u), float(v)]]],
            dtype=np.float64
        )

        checkerboard_point = cv2.perspectiveTransform(
            pixel,
            self.H
        )[0, 0]

        xc, yc = checkerboard_point

        robot_point = self.A @ np.array(
            [xc, yc, 1.0],
            dtype=np.float64
        )

        return float(robot_point[0]), float(robot_point[1])

    def validate_frame(self, frame):
        height, width = frame.shape[:2]

        if (width, height) != self.image_size:
            raise RuntimeError(
                f"Camera resolution is {width}x{height}, "
                f"but calibration expects "
                f"{self.image_size[0]}x{self.image_size[1]}."
            )


# ============================================================
# RED OBJECT DETECTION
# ============================================================

def detect_red_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(
        hsv,
        RED_LOW_1,
        RED_HIGH_1
    )

    mask_2 = cv2.inRange(
        hsv,
        RED_LOW_2,
        RED_HIGH_2
    )

    mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((5, 5), dtype=np.uint8)

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

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < MIN_CONTOUR_AREA:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    u = int(moments["m10"] / moments["m00"])
    v = int(moments["m01"] / moments["m00"])

    return {
        "pixel": (u, v),
        "contour": contour,
        "area": area,
    }


# ============================================================
# ARDUINO COMMUNICATION
# ============================================================

class RobotController:
    def __init__(self):
        self.ser = serial.Serial(
            SERIAL_PORT,
            SERIAL_BAUD,
            timeout=0.5,
            write_timeout=1
        )

        # Opening the serial port resets many Arduino Mega boards.
        time.sleep(2.5)
        self.ser.reset_input_buffer()

        self.steps = {"X": 0, "Y": 0, "Z": 0}
        self.armed = False
        self.busy = False
        self.stopped = False

        print("\nConnected to Arduino:", SERIAL_PORT)

        self.send("ARM")
        self.wait_for_idle(timeout=5)

        if not self.armed:
            raise RuntimeError(
                "Arduino did not confirm ARMED status."
            )

    def send(self, command):
        print("TX:", command)

        self.ser.write(
            (command + "\n").encode("ascii")
        )

        self.ser.flush()

    def parse_line(self, line):
        if not line:
            return

        print("RX:", line)

        if line.startswith("STATUS "):
            values = {}

            for token in line.split()[1:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    values[key] = value

            for axis in ("X", "Y", "Z"):
                if axis in values:
                    self.steps[axis] = int(values[axis])

            if "armed" in values:
                self.armed = values["armed"] == "1"

            if "busy" in values:
                self.busy = values["busy"] == "1"

        elif line.startswith("OK STOPPING"):
            self.stopped = True

    def poll(self):
        # Avoid blocking the camera UI.
        count = 0

        while self.ser.in_waiting and count < 30:
            line = self.ser.readline().decode(
                "ascii",
                errors="replace"
            ).strip()

            self.parse_line(line)
            count += 1

    def request_status(self):
        self.send("STATUS")

    def wait_for_idle(self, timeout=15):
        deadline = time.monotonic() + timeout
        last_status = 0

        while time.monotonic() < deadline:
            self.poll()

            now = time.monotonic()

            if now - last_status >= 0.25:
                self.request_status()
                last_status = now

            self.poll()

            if self.armed and not self.busy:
                return True

            time.sleep(0.05)

        return False

    def jog(self, command_name, amount):
        if self.stopped:
            print("Movement locked after STOP.")
            return False

        if not self.armed or self.busy:
            print("Robot is not ready.")
            return False

        axis, direction = COMMANDS[command_name]

        target = self.steps.copy()
        target[axis] += direction * amount

        if any(
            abs(value) > MAX_SESSION_STEPS
            for value in target.values()
        ):
            print(
                "Rejected: target exceeds session step limit."
            )
            return False

        self.send(
            f"MOVE3 {target['X']} "
            f"{target['Y']} {target['Z']}"
        )

        self.busy = True
        return True

    def stop(self):
        self.send("STOP")
        self.stopped = True

    def close(self):
        if self.ser.is_open:
            self.ser.close()


# ============================================================
# RECORDING
# ============================================================

def record_sample(robot, detection, object_xy):
    new_file = not RECORD_PATH.exists()

    with RECORD_PATH.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if new_file:
            writer.writerow([
                "timestamp",
                "X_steps",
                "Y_steps",
                "Z_steps",
                "object_u",
                "object_v",
                "object_x_cm",
                "object_y_cm",
            ])

        u, v = detection["pixel"]
        x_cm, y_cm = object_xy

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            robot.steps["X"],
            robot.steps["Y"],
            robot.steps["Z"],
            u,
            v,
            round(x_cm, 3),
            round(y_cm, 3),
        ])

    print(
        f"Recorded: steps={robot.steps}, "
        f"object=({x_cm:.2f}, {y_cm:.2f}) cm"
    )


# ============================================================
# USER INTERFACE
# ============================================================

BUTTONS = {
    "UP": (90, 365, 170, 405),
    "DOWN": (90, 420, 170, 460),
    "LEFT": (10, 420, 85, 460),
    "RIGHT": (175, 420, 265, 460),
    "E+": (280, 365, 350, 405),
    "E-": (280, 420, 350, 460),
    "REC": (370, 365, 445, 405),
    "STOP": (460, 365, 555, 405),
}

UI_STATE = {
    "clicked": None,
}


def mouse_callback(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for name, (x1, y1, x2, y2) in BUTTONS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            UI_STATE["clicked"] = name
            break


def draw_button(frame, name, rect, enabled=True):
    x1, y1, x2, y2 = rect

    if name == "STOP":
        color = (40, 40, 190)
    elif name == "REC":
        color = (80, 140, 60)
    else:
        color = (70, 70, 70)

    if not enabled:
        color = (50, 50, 50)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (210, 210, 210),
        1
    )

    cv2.putText(
        frame,
        name,
        (x1 + 9, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


def draw_interface(
    frame,
    robot,
    detection,
    object_xy,
    jog_steps,
    message
):
    display = frame.copy()
    height, width = display.shape[:2]

    # Dark top information panel
    cv2.rectangle(
        display,
        (0, 0),
        (width, 115),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        display,
        "EGG_ROBOT | CALIBRATED CAMERA",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    status = (
        "STOPPED" if robot.stopped
        else "MOVING" if robot.busy
        else "READY" if robot.armed
        else "NOT ARMED"
    )

    cv2.putText(
        display,
        f"Robot: {status}  |  Jog: {jog_steps} steps",
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1
    )

    cv2.putText(
        display,
        f"Shoulder X={robot.steps['X']}  "
        f"Elbow Y={robot.steps['Y']}  "
        f"Base Z={robot.steps['Z']}",
        (10, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1
    )

    if object_xy is not None:
        x_cm, y_cm = object_xy

        object_text = (
            f"RED OBJECT: X={x_cm:.2f} cm  "
            f"Y={y_cm:.2f} cm"
        )

        text_color = (90, 255, 90)
    else:
        object_text = "RED OBJECT: Not detected"
        text_color = (120, 180, 255)

    cv2.putText(
        display,
        object_text,
        (10, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        text_color,
        2
    )

    if detection is not None:
        cv2.drawContours(
            display,
            [detection["contour"]],
            -1,
            (0, 255, 0),
            2
        )

        u, v = detection["pixel"]

        cv2.circle(
            display,
            (u, v),
            5,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            display,
            f"({u},{v})",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    for name, rect in BUTTONS.items():
        enabled = (
            name in ("STOP", "REC")
            or (
                robot.armed
                and not robot.busy
                and not robot.stopped
            )
        )

        draw_button(
            display,
            name,
            rect,
            enabled
        )

    cv2.putText(
        display,
        message[:75],
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return display


# ============================================================
# MAIN
# ============================================================

def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    mapping = CameraRobotMapping()

    print("\nLoaded camera-to-robot mapping:")
    print(mapping.A)

    camera = None
    robot = None

    try:
        camera = Camera(config)
        robot = RobotController()

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            mouse_callback
        )

        jog_steps = DEFAULT_JOG_STEPS
        last_status_time = 0
        last_frame = None
        message = "Manual control + calibrated object XY"

        print("\nControls:")
        print("W/S: shoulder up/down")
        print("A/F: base left/right")
        print("E/D: elbow +/-")
        print("1/2/3: jog 25/50/100 steps")
        print("R: record")
        print("SPACE: controlled STOP")
        print("Q: quit")
        print("\nIMPORTANT: ARM establishes a new session zero.")
        print("It does NOT physically home the robot.")

        while True:
            robot.poll()

            now = time.monotonic()

            if now - last_status_time > 0.5:
                robot.request_status()
                last_status_time = now

            # Freeze camera while motors are moving, matching
            # the previous manual-control workflow.
            if robot.busy and last_frame is not None:
                frame = last_frame.copy()
                camera_ok = True
            else:
                camera_ok, frame = camera.read()

                if camera_ok and frame is not None:
                    last_frame = frame.copy()

            if not camera_ok or frame is None:
                message = "Camera frame unavailable"

                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), 27):
                    break

                continue

            mapping.validate_frame(frame)

            detection = detect_red_object(frame)
            object_xy = None

            if detection is not None:
                u, v = detection["pixel"]

                object_xy = mapping.pixel_to_robot(u, v)

            display = draw_interface(
                frame,
                robot,
                detection,
                object_xy,
                jog_steps,
                message
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(20) & 0xFF

            action = UI_STATE["clicked"]
            UI_STATE["clicked"] = None

            if key in (ord("q"), 27):
                break

            if key == ord("1"):
                jog_steps = 25
            elif key == ord("2"):
                jog_steps = 50
            elif key == ord("3"):
                jog_steps = 100

            keyboard_actions = {
                ord("w"): "UP",
                ord("s"): "DOWN",
                ord("a"): "LEFT",
                ord("f"): "RIGHT",
                ord("e"): "E+",
                ord("d"): "E-",
                ord("r"): "REC",
                ord(" "): "STOP",
            }

            if action is None:
                action = keyboard_actions.get(key)

            if action in COMMANDS:
                if robot.jog(action, jog_steps):
                    message = (
                        f"Jog {action}: {jog_steps} steps"
                    )

            elif action == "REC":
                if detection is not None and object_xy is not None:
                    if not robot.busy:
                        record_sample(
                            robot,
                            detection,
                            object_xy
                        )
                        message = "Sample recorded"
                    else:
                        message = "Wait until robot stops"
                else:
                    message = "No red object detected"

            elif action == "STOP":
                robot.stop()
                message = "STOP requested; movement locked"

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        if camera is not None:
            camera.release()

        if robot is not None:
            robot.close()

        cv2.destroyAllWindows()
        print("Controller closed.")


if __name__ == "__main__":
    main()
