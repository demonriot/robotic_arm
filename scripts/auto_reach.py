
"""
EGG_ROBOT - One-shot red-object reach

Requires:
    preview_reach.py in the same directory

Camera:
    Uses existing homography and camera-to-robot mapping.

Arduino:
    COM9, 115200 baud

Reference:
    Gripper (25, 3.3, 4.5) cm
    Shoulder +20 deg
    Forearm -35 deg
    Relative elbow -55 deg

Target:
    8 cm above the tabletop
"""

import math
import re
import time

import cv2
import serial
import yaml

from preview_reach import (
    CONFIG_PATH,
    load_calibration,
    pixel_to_robot,
    detect_red,
    inverse_kinematics,
    angle_changes_to_steps,
    Camera,
    TARGET_Z,
)


# ============================================================
# SETTINGS
# ============================================================

PORT = "COM9"
BAUD = 115200

STEPS_PER_DEG = 35.5555556

# Must match Arduino firmware.
# Provisional limits, not verified mechanical limits.
LIMITS = {
    "base": (-40.0, 40.0),
    "shoulder": (0.0, 65.0),
    "elbow": (-100.0, -15.0),
}

REF = {
    "base": math.degrees(math.atan2(3.3, 25.0)),
    "shoulder": 20.0,
    "elbow": -55.0,
}


# ============================================================
# ARDUINO COMMUNICATION
# ============================================================

def request(ser, command, timeout=5):
    ser.reset_input_buffer()

    print("TX:", command)

    ser.write(
        (command + "\n").encode("ascii")
    )
    ser.flush()

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        response = ser.readline().decode(
            errors="replace"
        ).strip()

        if response:
            print("RX:", response)

            if response.startswith("ERR"):
                raise RuntimeError(response)

            if (
                response.startswith("OK")
                or response.startswith("STATUS ")
            ):
                return response

    raise TimeoutError(
        f"No response to {command}; "
        "controller state unknown."
    )


def status(ser):
    response = request(ser, "STATUS")

    values = {
        key: int(value)
        for key, value in re.findall(
            r"(armed|busy|X|Y|Z)=(-?\d+)",
            response
        )
    }

    required = {
        "armed", "busy", "X", "Y", "Z"
    }

    if set(values) != required:
        raise RuntimeError(
            "Unexpected STATUS: " + response
        )

    return values


# ============================================================
# CAMERA DETECTION
# ============================================================

def acquire():
    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    H, A, size = load_calibration()

    camera = Camera(config["camera"])

    print("\nCamera controls:")
    print("P = capture red object")
    print("Q = cancel")

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), 27):
                    return None

                continue

            frame_size = (
                frame.shape[1],
                frame.shape[0]
            )

            if frame_size != size:
                raise RuntimeError(
                    "Camera resolution changed; "
                    "calibration invalid."
                )

            found = detect_red(frame)
            candidate = None

            if found is not None:
                u, v, contour = found

                x, y = pixel_to_robot(
                    u, v, H, A
                )

                angles = inverse_kinematics(
                    x, y, TARGET_Z
                )

                cv2.drawContours(
                    frame,
                    [contour],
                    -1,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (u, v),
                    5,
                    (255, 255, 255),
                    -1
                )

                if angles is not None:
                    candidate = (
                        x, y, angles
                    )

                label = (
                    "OK"
                    if angles is not None
                    else "UNREACHABLE"
                )

                cv2.putText(
                    frame,
                    f"Object ({x:.1f}, {y:.1f}) "
                    f"IK: {label}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "P: capture  Q: cancel",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "EGG_ROBOT - Reach Object",
                frame
            )

            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                return None

            if key == ord("p"):
                if candidate is not None:
                    return candidate

                print(
                    "No reachable red object."
                )

    finally:
        camera.close()
        cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nEGG_ROBOT - FULL REACH")
    print("=" * 50)

    print(
        "Reference gripper: "
        "(25, 3.3, 4.5) cm"
    )

    print(
        "Shoulder +20 deg, "
        "forearm -35 deg, "
        "base +7.52 deg"
    )

    print(
        "\nKeep the arm stationary "
        "at the reference pose."
    )

    print(
        "Close Arduino Serial Monitor "
        "before connecting to COM9."
    )

    confirmation = input(
        "\nType REFERENCE to confirm: "
    ).strip()

    if confirmation != "REFERENCE":
        print("Cancelled.")
        return

    # --------------------------------------------------------
    # Acquire target
    # --------------------------------------------------------

    result = acquire()

    if result is None:
        print("Cancelled.")
        return

    x, y, angles = result

    sx, sy, sz = angle_changes_to_steps(
        angles
    )

    print("\n" + "=" * 50)
    print("FULL REACH TARGET")
    print("=" * 50)

    print(
        f"Target XYZ: "
        f"({x:.2f}, {y:.2f}, "
        f"{TARGET_Z:.2f}) cm"
    )

    print("\nJoint angles:")

    for name, angle in angles.items():
        print(
            f"  {name}: {angle:.2f} deg"
        )

    print("\nArduino command:")
    print(
        f"MOVE3 {sx} {sy} {sz}"
    )

    # --------------------------------------------------------
    # Check provisional joint limits
    # --------------------------------------------------------

    for axis, angle in angles.items():
        low, high = LIMITS[axis]

        if not low <= angle <= high:
            print(
                f"REJECTED: {axis} "
                f"{angle:.2f} deg outside "
                f"[{low}, {high}]"
            )
            return

    # --------------------------------------------------------
    # Verify step conversion
    # --------------------------------------------------------

    reconstructed = {
        "shoulder": (
            REF["shoulder"]
            - sx / STEPS_PER_DEG
        ),
        "elbow": (
            REF["elbow"]
            + sy / STEPS_PER_DEG
        ),
        "base": (
            REF["base"]
            + sz / STEPS_PER_DEG
        ),
    }

    for axis in angles:
        difference = abs(
            reconstructed[axis]
            - angles[axis]
        )

        if difference > 0.04:
            raise RuntimeError(
                "Motor conversion mismatch."
            )

    print(
        "\nWARNING: Joint limits are "
        "provisional."
    )

    print(
        "There is no physical homing "
        "or collision detection."
    )

    print(
        "Check the entire arm path "
        "and keep a physical power "
        "cutoff accessible."
    )

    approval = input(
        "\nType MOVE to authorize "
        "the full movement: "
    ).strip()

    if approval != "MOVE":
        print("Cancelled.")
        return

    # --------------------------------------------------------
    # Connect to Arduino
    # --------------------------------------------------------

    with serial.Serial(
        PORT,
        BAUD,
        timeout=0.5,
        write_timeout=2
    ) as ser:

        # Opening COM9 typically resets Arduino Mega.
        time.sleep(2.5)

        ser.reset_input_buffer()

        initial = status(ser)

        if (
            initial["armed"] != 0
            or initial["busy"] != 0
        ):
            raise RuntimeError(
                "Controller unexpectedly "
                "armed or busy."
            )

        # Establish current pose as zero.
        request(ser, "ARM")

        state = status(ser)

        expected = {
            "armed": 1,
            "busy": 0,
            "X": 0,
            "Y": 0,
            "Z": 0,
        }

        if state != expected:
            raise RuntimeError(
                f"Reference not zeroed: {state}"
            )

        # ----------------------------------------------------
        # Execute full movement
        # ----------------------------------------------------

        command = (
            f"MOVE3 {sx} {sy} {sz}"
        )

        request(ser, command)

        print("\nMovement started.")
        print(
            "Ctrl+C requests controlled STOP "
            "(not an emergency stop)."
        )

        deadline = (
            time.monotonic() + 120
        )

        try:
            while time.monotonic() < deadline:
                state = status(ser)

                if state["busy"] == 0:
                    print(
                        "\nFinal status:",
                        state
                    )

                    actual = (
                        state["X"],
                        state["Y"],
                        state["Z"]
                    )

                    target = (
                        sx, sy, sz
                    )

                    if actual != target:
                        print(
                            "WARNING: Final steps "
                            "differ from target."
                        )

                    return

                time.sleep(0.4)

            print(
                "WARNING: Movement timed out. "
                "State unknown."
            )

        except KeyboardInterrupt:
            print(
                "\nSTOP requested."
            )

            request(ser, "STOP")

            end = (
                time.monotonic() + 15
            )

            while time.monotonic() < end:
                state = status(ser)

                if state["busy"] == 0:
                    print(
                        "Stopped at:",
                        state
                    )
                    return

                time.sleep(0.3)

            print(
                "WARNING: STOP not confirmed."
            )


if __name__ == "__main__":
    main()
