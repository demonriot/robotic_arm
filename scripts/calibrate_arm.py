
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import serial


# ============================================================
# CONFIGURATION
# ============================================================

PORT = "COM9"
BAUD = 115200

STEPS_PER_DEG = 35.5555556

# Axis mapping and signs
# X: shoulder, positive steps lower shoulder angle
# Y: elbow, positive steps increase relative elbow angle
# Z: base, positive steps increase base angle

SIGNS = {
    "X": -1,
    "Y": +1,
    "Z": +1,
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "arm_calibration"
OUT.mkdir(parents=True, exist_ok=True)

STATE_FILE = OUT / "robot_state.json"
LOG_FILE = OUT / "movement_log.csv"
CALIB_FILE = OUT / "robot_calibration.json"

FIELDS = [
    "timestamp", "event", "command",
    "X", "Y", "Z",
    "base_deg", "shoulder_deg",
    "elbow_deg", "forearm_deg",
    "busy",
]

REFERENCE = None
SESSION_STEPS = {"X": 0, "Y": 0, "Z": 0}


# ============================================================
# UTILITIES
# ============================================================

def timestamp():
    return datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )


def save_json(path, data):
    temporary = path.with_suffix(".tmp")

    with temporary.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=2)

    temporary.replace(path)


def ask_number(label):
    while True:
        try:
            return float(input(label))
        except ValueError:
            print("Enter a valid number.")


def ask_pose(name):
    print(f"\n===== {name} =====")
    print("Base: 0 deg along +X.")
    print("Shoulder: positive above horizontal.")
    print("Forearm: absolute angle from horizontal.")
    print("XYZ: gripper tip relative to robot base, cm.")

    base = ask_number("Base angle (deg): ")
    shoulder = ask_number("Shoulder angle (deg): ")
    forearm = ask_number("Forearm angle (deg): ")

    x = ask_number("Gripper X (cm): ")
    y = ask_number("Gripper Y (cm): ")
    z = ask_number("Gripper Z (cm): ")

    return {
        "base": base,
        "shoulder": shoulder,
        "forearm": forearm,
        "elbow": forearm - shoulder,
        "xyz": [x, y, z],
    }


# ============================================================
# ANGLE TRACKING
# ============================================================

def calculate_angles(steps):
    base = (
        REFERENCE["base"]
        + SIGNS["Z"]
        * (steps["Z"] - SESSION_STEPS["Z"])
        / STEPS_PER_DEG
    )

    shoulder = (
        REFERENCE["shoulder"]
        + SIGNS["X"]
        * (steps["X"] - SESSION_STEPS["X"])
        / STEPS_PER_DEG
    )

    elbow = (
        REFERENCE["elbow"]
        + SIGNS["Y"]
        * (steps["Y"] - SESSION_STEPS["Y"])
        / STEPS_PER_DEG
    )

    return {
        "base": base,
        "shoulder": shoulder,
        "elbow": elbow,
        "forearm": shoulder + elbow,
    }


def record(event, command, status):
    angles = calculate_angles(status)

    row = {
        "timestamp": timestamp(),
        "event": event,
        "command": command,
        "X": status["X"],
        "Y": status["Y"],
        "Z": status["Z"],
        "base_deg": round(angles["base"], 4),
        "shoulder_deg": round(angles["shoulder"], 4),
        "elbow_deg": round(angles["elbow"], 4),
        "forearm_deg": round(angles["forearm"], 4),
        "busy": status["busy"],
    }

    with LOG_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(row)

    save_json(STATE_FILE, {
        "updated_at": row["timestamp"],
        "reference": REFERENCE,
        "reference_steps": SESSION_STEPS,
        "motor_steps": {
            axis: status[axis]
            for axis in ("X", "Y", "Z")
        },
        "estimated_angles": angles,
        "armed": status["armed"],
        "busy": status["busy"],
        "last_event": event,
        "last_command": command,
        "note": (
            "Angles are open-loop estimates. "
            "No encoders or homing sensors."
        ),
    })


# ============================================================
# SERIAL COMMUNICATION
# ============================================================

def send(ser, command, timeout=4):
    ser.reset_input_buffer()

    ser.write(
        (command + "\n").encode("ascii")
    )
    ser.flush()

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        line = ser.readline().decode(
            errors="replace"
        ).strip()

        if not line:
            continue

        if line.startswith("ERR"):
            raise RuntimeError(line)

        if (
            line.startswith("OK")
            or line.startswith("STATUS ")
        ):
            return line

    raise TimeoutError(
        f"No response to {command}"
    )


def get_status(ser):
    line = send(ser, "STATUS")

    values = {
        key: int(value)
        for key, value in re.findall(
            r"(armed|busy|X|Y|Z)=(-?\d+)",
            line
        )
    }

    if set(values) != {
        "armed", "busy", "X", "Y", "Z"
    }:
        raise RuntimeError(
            f"Invalid STATUS: {line}"
        )

    return values


def monitor_motion(ser, command):
    print("\nMovement started.")
    print("Ctrl+C requests controlled STOP.")

    try:
        while True:
            status = get_status(ser)

            record("motion_sample", command, status)

            angles = calculate_angles(status)

            print(
                f"\rSteps: "
                f"X={status['X']:6d} "
                f"Y={status['Y']:6d} "
                f"Z={status['Z']:6d} | "
                f"Base={angles['base']:+7.2f} "
                f"Shoulder={angles['shoulder']:+7.2f} "
                f"Forearm={angles['forearm']:+7.2f}",
                end="",
                flush=True
            )

            if status["busy"] == 0:
                print("\nMovement complete.")
                record("movement_complete", command, status)
                return status

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nRequesting controlled STOP...")
        send(ser, "STOP")

        # Keep polling while the motors decelerate.
        while True:
            status = get_status(ser)
            record("stopping", "STOP", status)

            if status["busy"] == 0:
                print("Stopped.")
                record("stopped", "STOP", status)
                return status

            time.sleep(0.2)


# ============================================================
# CALIBRATION DATA
# ============================================================

def save_calibration(pose_a, pose_b, steps_b):
    predicted = calculate_angles(steps_b)

    differences = {
        key: round(
            pose_b[key] - predicted[key], 3
        )
        for key in (
            "base", "shoulder",
            "elbow", "forearm"
        )
    }

    data = {
        "created_at": timestamp(),
        "geometry_cm": {
            "upper_arm": 12.0,
            "forearm": 12.0,
            "shoulder_height": 15.0,
            "nominal_gripper_radial_offset": 6.0,
            "nominal_gripper_vertical_offset": -5.0,
        },
        "steps_per_degree": STEPS_PER_DEG,
        "motor_signs": SIGNS,
        "target_z_cm": 8.0,
        "pose_a": {
            "measurements": pose_a,
            "steps": {"X": 0, "Y": 0, "Z": 0},
        },
        "pose_b": {
            "measurements": pose_b,
            "steps": {
                axis: steps_b[axis]
                for axis in ("X", "Y", "Z")
            },
            "predicted_angles": predicted,
            "measurement_minus_prediction_deg": differences,
        },
    }

    save_json(CALIB_FILE, data)

    print("\nCalibration saved:")
    print(CALIB_FILE)

    print("\nAngle differences (measured - estimated):")
    for key, value in differences.items():
        print(f"  {key}: {value:+.2f} deg")

    print(
        "\nThese differences help check the "
        "motor-angle model. They are not "
        "automatically treated as corrections."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    global REFERENCE, SESSION_STEPS

    print("=" * 55)
    print("EGG_ROBOT - CALIBRATION AND MOVEMENT LOGGER")
    print("=" * 55)

    print(
        "\nThe arm must remain stationary "
        "until ARM establishes its zero."
    )

    print(
        "A new session needs a known physical "
        "reference because there are no encoders."
    )

    pose_a = ask_pose("POSE A - INITIAL")

    input(
        "\nKeep the arm stationary. "
        "Press Enter to connect and ARM..."
    )

    with serial.Serial(
        PORT,
        BAUD,
        timeout=0.5,
        write_timeout=2
    ) as ser:

        time.sleep(2.5)
        ser.reset_input_buffer()

        initial = get_status(ser)

        if initial["armed"] or initial["busy"]:
            raise RuntimeError(
                "Controller is not idle and unarmed."
            )

        send(ser, "ARM")

        status = get_status(ser)

        if (
            status["armed"] != 1
            or status["busy"] != 0
            or any(
                status[a] != 0
                for a in ("X", "Y", "Z")
            )
        ):
            raise RuntimeError(
                "Could not establish reference zero."
            )

        REFERENCE = pose_a
        SESSION_STEPS = {
            "X": 0, "Y": 0, "Z": 0
        }

        record("reference_set", "ARM", status)

        print("\nReference established.")
        print("Commands:")
        print("  x -100   shoulder jog")
        print("  y 100    elbow jog")
        print("  z -100   base jog")
        print("  status   show current position")
        print("  pose     record Pose B")
        print("  quit     exit")
        print(
            "\nJog values are signed motor "
            "microsteps, not degrees."
        )

        while True:
            try:
                command = input(
                    "\nrobot> "
                ).strip().lower()

                if not command:
                    continue

                if command == "quit":
                    status = get_status(ser)
                    record("session_end", "quit", status)
                    break

                if command == "status":
                    status = get_status(ser)
                    record("status", "STATUS", status)

                    print(
                        json.dumps(
                            {
                                "steps": {
                                    a: status[a]
                                    for a in ("X", "Y", "Z")
                                },
                                "angles": calculate_angles(
                                    status
                                ),
                            },
                            indent=2
                        )
                    )
                    continue

                if command == "pose":
                    status = get_status(ser)

                    if status["busy"]:
                        print(
                            "Wait until movement stops."
                        )
                        continue

                    record("pose_b_before_measurement",
                           "POSE", status)

                    print(
                        "\nMeasure the current gripper "
                        "position and joint angles."
                    )

                    pose_b = ask_pose(
                        "POSE B - NEW POSITION"
                    )

                    save_calibration(
                        pose_a, pose_b, status
                    )

                    record("pose_b_recorded", "POSE", status)

                    print(
                        "\nYou can keep jogging, "
                        "or type quit."
                    )
                    continue

                parts = command.split()

                if (
                    len(parts) == 2
                    and parts[0] in ("x", "y", "z")
                ):
                    axis = parts[0].upper()
                    delta = int(parts[1])

                    if delta == 0:
                        print("Delta cannot be zero.")
                        continue

                    before = get_status(ser)

                    if before["busy"]:
                        print("Robot is busy.")
                        continue

                    cmd = f"MOVE {axis} {delta}"

                    record("command_requested", cmd, before)
                    print("Sending:", cmd)

                    send(ser, cmd)
                    monitor_motion(ser, cmd)

                    continue

                print("Unknown command.")

            except ValueError:
                print("Use an integer number of steps.")

            except RuntimeError as error:
                print("Controller error:", error)

            except KeyboardInterrupt:
                print(
                    "\nInterrupted. Checking status..."
                )

                try:
                    status = get_status(ser)

                    if status["busy"]:
                        send(ser, "STOP")
                        monitor_motion(ser, "STOP")
                    else:
                        record("interrupted", "CTRL+C", status)

                except Exception as error:
                    print(
                        "Could not verify controller state:",
                        error
                    )
                    raise

    print("\nSession ended.")
    print("State and movement history saved.")


if __name__ == "__main__":
    main()
