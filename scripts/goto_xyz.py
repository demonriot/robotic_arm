
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import serial

PORT = "COM9"
BAUD = 115200

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "arm_calibration"
OUT.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT / "pose_d_movements.csv"
STATE_FILE = OUT / "pose_d_state.json"

L1 = 12.0
L2 = 12.0
SHOULDER_HEIGHT = 15.0
STEPS_PER_DEG = 35.5555556
DEFAULT_Z = 8.0

# Original measured pose, corrected forearm sign
POSE_A = {
    "base": 90.0,
    "shoulder": 65.0,
    "forearm": -65.0,
    "xyz": (0.0, 18.0, 8.0),
}

# Current physical position
POSE_D = {
    "base": 90.0,
    "shoulder": 74.11,
    "forearm": -75.0,
    "xyz": (0.0, 12.0, 7.2),
}

REFERENCE = {
    "base": POSE_D["base"],
    "shoulder": POSE_D["shoulder"],
    "forearm": POSE_D["forearm"],
}

REFERENCE["elbow"] = (
    REFERENCE["forearm"] - REFERENCE["shoulder"]
)

# Provisional motor model:
# X positive -> shoulder decreases
# Y positive -> relative elbow increases
# Z positive -> base increases
#
# Forearm = shoulder + relative elbow.
# Actual linkage coupling remains unverified.

AXES = ("X", "Y", "Z")


def timestamp():
    return datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )


# ============================================================
# GEOMETRY
# ============================================================

def pose_offsets(pose):
    s = math.radians(pose["shoulder"])
    f = math.radians(pose["forearm"])

    x, y, z = pose["xyz"]
    radius = math.hypot(x, y)

    dr = (
        radius
        - L1 * math.cos(s)
        - L2 * math.cos(f)
    )

    dz = (
        z
        - SHOULDER_HEIGHT
        - L1 * math.sin(s)
        - L2 * math.sin(f)
    )

    return dr, dz


a_dr, a_dz = pose_offsets(POSE_A)
d_dr, d_dz = pose_offsets(POSE_D)

# Least-squares fixed-offset estimate
RADIAL_OFFSET = (a_dr + d_dr) / 2
VERTICAL_OFFSET = (a_dz + d_dz) / 2


def fk(angles):
    b = math.radians(angles["base"])
    s = math.radians(angles["shoulder"])
    f = math.radians(angles["forearm"])

    radius = (
        L1 * math.cos(s)
        + L2 * math.cos(f)
        + RADIAL_OFFSET
    )

    height = (
        SHOULDER_HEIGHT
        + L1 * math.sin(s)
        + L2 * math.sin(f)
        + VERTICAL_OFFSET
    )

    return (
        radius * math.cos(b),
        radius * math.sin(b),
        height,
    )


def ik(x, y, z, current):
    base = math.degrees(math.atan2(y, x))

    r = math.hypot(x, y) - RADIAL_OFFSET
    h = z - SHOULDER_HEIGHT - VERTICAL_OFFSET

    D = (
        r*r + h*h - L1*L1 - L2*L2
    ) / (2 * L1 * L2)

    if D < -1.000001 or D > 1.000001:
        raise ValueError(
            "Target outside modeled workspace."
        )

    D = max(-1.0, min(1.0, D))

    candidates = []

    for e in (math.acos(D), -math.acos(D)):
        s = (
            math.atan2(h, r)
            - math.atan2(
                L2 * math.sin(e),
                L1 + L2 * math.cos(e)
            )
        )

        shoulder = math.degrees(s)
        elbow = math.degrees(e)
        forearm = shoulder + elbow

        angles = {
            "base": base,
            "shoulder": shoulder,
            "forearm": forearm,
            "elbow": elbow,
        }

        score = (
            (shoulder - current["shoulder"])**2
            + (elbow - current["elbow"])**2
        )

        candidates.append((score, angles))

    return min(candidates, key=lambda item: item[0])[1]


# ============================================================
# MOTOR CONVERSION
# ============================================================

def angles_from_steps(steps):
    base = (
        REFERENCE["base"]
        + steps["Z"] / STEPS_PER_DEG
    )

    shoulder = (
        REFERENCE["shoulder"]
        - steps["X"] / STEPS_PER_DEG
    )

    elbow = (
        REFERENCE["elbow"]
        + steps["Y"] / STEPS_PER_DEG
    )

    return {
        "base": base,
        "shoulder": shoulder,
        "elbow": elbow,
        "forearm": shoulder + elbow,
    }


def steps_from_angles(angles):
    return {
        "X": round(
            -(angles["shoulder"]
              - REFERENCE["shoulder"])
            * STEPS_PER_DEG
        ),
        "Y": round(
            (angles["elbow"]
             - REFERENCE["elbow"])
            * STEPS_PER_DEG
        ),
        "Z": round(
            (angles["base"]
             - REFERENCE["base"])
            * STEPS_PER_DEG
        ),
    }


# ============================================================
# SERIAL
# ============================================================

def send(ser, command, timeout=5):
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

        if line.startswith("ERR"):
            raise RuntimeError(line)

        if line.startswith(("OK", "STATUS ")):
            return line

    raise TimeoutError(
        "No response to " + command
    )


def status(ser):
    line = send(ser, "STATUS")

    values = {
        k: int(v)
        for k, v in re.findall(
            r"(armed|busy|X|Y|Z)=(-?\d+)",
            line
        )
    }

    if set(values) != {
        "armed", "busy", "X", "Y", "Z"
    }:
        raise RuntimeError(
            "Invalid Arduino STATUS: " + line
        )

    return values


def steps_of(s):
    return {axis: s[axis] for axis in AXES}


# ============================================================
# LOGGING
# ============================================================

FIELDS = [
    "timestamp", "event",
    "X", "Y", "Z",
    "base_deg", "shoulder_deg",
    "forearm_deg", "elbow_deg",
    "predicted_x", "predicted_y", "predicted_z",
    "measured_x", "measured_y", "measured_z",
    "measured_forearm_deg",
]


def log(event, steps, measured=None,
        measured_forearm=None):

    angles = angles_from_steps(steps)
    xyz = fk(angles)

    row = {
        "timestamp": timestamp(),
        "event": event,
        "X": steps["X"],
        "Y": steps["Y"],
        "Z": steps["Z"],
        "base_deg": angles["base"],
        "shoulder_deg": angles["shoulder"],
        "forearm_deg": angles["forearm"],
        "elbow_deg": angles["elbow"],
        "predicted_x": xyz[0],
        "predicted_y": xyz[1],
        "predicted_z": xyz[2],
        "measured_x": measured[0] if measured else "",
        "measured_y": measured[1] if measured else "",
        "measured_z": measured[2] if measured else "",
        "measured_forearm_deg": (
            measured_forearm
            if measured_forearm is not None
            else ""
        ),
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

    state = {
        "timestamp": row["timestamp"],
        "reference": REFERENCE,
        "motor_steps": steps,
        "estimated_angles": angles,
        "predicted_xyz": xyz,
        "event": event,
    }

    temp = STATE_FILE.with_suffix(".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    temp.replace(STATE_FILE)


# ============================================================
# MOVEMENT
# ============================================================

def move(ser, target):
    command = (
        f"MOVE3 {target['X']} "
        f"{target['Y']} {target['Z']}"
    )

    print("\nSending:", command)
    send(ser, command)

    deadline = time.monotonic() + 180

    while time.monotonic() < deadline:
        s = status(ser)
        steps = steps_of(s)

        log("motion_sample", steps)

        angles = angles_from_steps(steps)

        print(
            f"\rBase={angles['base']:+7.2f} "
            f"Shoulder={angles['shoulder']:+7.2f} "
            f"Forearm={angles['forearm']:+7.2f}",
            end="",
            flush=True
        )

        if not s["busy"]:
            print()

            if steps != target:
                log("incomplete", steps)
                raise RuntimeError(
                    "Movement ended before target."
                )

            log("complete", steps)
            print("Movement finished.")
            return

        time.sleep(0.2)

    raise TimeoutError("Movement timed out.")


def stop(ser):
    send(ser, "STOP")

    deadline = time.monotonic() + 20

    while time.monotonic() < deadline:
        s = status(ser)
        steps = steps_of(s)
        log("stopping", steps)

        if not s["busy"]:
            log("stopped", steps)
            print("\nStopped.")
            return

        time.sleep(0.2)

    raise TimeoutError(
        "Could not confirm stop."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 50)
    print("EGG_ROBOT - CORRECTED IK CONTROLLER")
    print("=" * 50)

    print("\nPhysical reference: Pose D")
    print("Base:      +90.00 deg")
    print("Shoulder:  +74.11 deg")
    print("Forearm:   -75.00 deg")
    print("Gripper:   (0, 12, 7.2) cm")

    print("\nFitted offsets:")
    print(f"Radial:   {RADIAL_OFFSET:.2f} cm")
    print(f"Vertical: {VERTICAL_OFFSET:.2f} cm")

    print(
        "\nThe model is approximate and "
        "joint coupling is not yet verified."
    )

    confirm = input(
        "\nType POSE_D if arm is still "
        "at this position: "
    ).strip()

    if confirm != "POSE_D":
        print("Cancelled.")
        return

    with serial.Serial(
        PORT, BAUD,
        timeout=0.5,
        write_timeout=2
    ) as ser:

        time.sleep(2.5)
        ser.reset_input_buffer()

        s = status(ser)

        if s["armed"] or s["busy"]:
            raise RuntimeError(
                "Arduino must be idle and unarmed."
            )

        send(ser, "ARM")

        s = status(ser)

        if (
            s["armed"] != 1
            or s["busy"] != 0
            or any(s[a] != 0 for a in AXES)
        ):
            raise RuntimeError(
                "Could not establish reference."
            )

        log("pose_d_reference", {
            a: 0 for a in AXES
        })

        print("\nReady.")
        print("Enter target XYZ.")
        print("Type home to return to Pose D.")
        print("Type q to exit.")

        while True:
            try:
                raw = input(
                    "\nTarget X (cm): "
                ).strip().lower()

                if raw == "q":
                    break

                if raw == "home":
                    move(ser, {
                        a: 0 for a in AXES
                    })
                    continue

                x = float(raw)
                y = float(
                    input("Target Y (cm): ")
                )

                raw_z = input(
                    "Target Z (default 8): "
                ).strip()

                z = (
                    float(raw_z)
                    if raw_z
                    else DEFAULT_Z
                )

                s = status(ser)

                if s["busy"]:
                    print("Arm is moving.")
                    continue

                current = angles_from_steps(
                    steps_of(s)
                )

                target_angles = ik(
                    x, y, z, current
                )

                target_steps = steps_from_angles(
                    target_angles
                )

                print("\nTarget joint angles:")
                for key, value in target_angles.items():
                    print(f"  {key}: {value:.2f} deg")

                print("\nTarget motor steps:")
                print(target_steps)

                print(
                    "\nCheck physical joint clearance "
                    "before executing."
                )

                if input(
                    "Type MOVE to execute: "
                ).strip() != "MOVE":
                    print("Cancelled.")
                    continue

                move(ser, target_steps)

                answer = input(
                    "\nRecord measured position? (y/n): "
                ).strip().lower()

                if answer == "y":
                    measured = [
                        float(input("Measured X: ")),
                        float(input("Measured Y: ")),
                        float(input("Measured Z: ")),
                    ]

                    raw_f = input(
                        "Forearm angle in degrees "
                        "(negative if downward; "
                        "blank to skip): "
                    ).strip()

                    measured_f = (
                        float(raw_f)
                        if raw_f
                        else None
                    )

                    s = status(ser)

                    log(
                        "physical_measurement",
                        steps_of(s),
                        measured,
                        measured_f
                    )

                    print("Measurement saved.")

            except KeyboardInterrupt:
                print("\nSTOP requested.")
                stop(ser)
                break

            except ValueError as error:
                print("Invalid target:", error)

            except (
                RuntimeError,
                TimeoutError,
                serial.SerialException
            ) as error:
                print("Controller error:", error)
                print(
                    "Position may be uncertain. "
                    "Verify before restarting."
                )
                break

    print("\nSession ended.")


if __name__ == "__main__":
    main()
