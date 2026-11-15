
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import serial

PORT = "COM9"
BAUD = 115200

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "empirical_calibration"
OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT / "calibration_points.csv"
STATE_PATH = OUT / "session_state.json"

AXES = ("X", "Y", "Z")
FIELDS = [
    "timestamp", "point",
    "motor_x", "motor_y", "motor_z",
    "gripper_x_cm", "gripper_y_cm", "gripper_z_cm",
    "notes",
]

def save_state(steps, status_text=""):
    data = {
        "timestamp": datetime.now().isoformat(),
        "steps": steps,
        "status": status_text,
    }
    STATE_PATH.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

def save_point(point, steps, xyz, notes=""):
    row = {
        "timestamp": datetime.now().isoformat(),
        "point": point,
        "motor_x": steps["X"],
        "motor_y": steps["Y"],
        "motor_z": steps["Z"],
        "gripper_x_cm": xyz[0],
        "gripper_y_cm": xyz[1],
        "gripper_z_cm": xyz[2],
        "notes": notes,
    }

    with CSV_PATH.open(
        "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved {point}: steps={steps}, XYZ={xyz}")

def command(ser, text, timeout=5):
    ser.reset_input_buffer()
    ser.write((text + "\n").encode("ascii"))
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

    raise TimeoutError(f"No response to: {text}")

def get_status(ser):
    response = command(ser, "STATUS")

    values = {
        k: int(v)
        for k, v in re.findall(
            r"(armed|busy|X|Y|Z)=(-?\d+)",
            response
        )
    }

    required = {"armed", "busy", "X", "Y", "Z"}
    if set(values) != required:
        raise RuntimeError(
            "Unexpected STATUS: " + response
        )

    return values

def get_steps(status):
    return {a: status[a] for a in AXES}

def wait_for_motion(ser, target):
    deadline = time.monotonic() + 180

    while time.monotonic() < deadline:
        status = get_status(ser)
        steps = get_steps(status)

        save_state(steps, "moving" if status["busy"] else "idle")

        print(
            f"\rX={steps['X']:>7} "
            f"Y={steps['Y']:>7} "
            f"Z={steps['Z']:>7}",
            end="", flush=True
        )

        if not status["busy"]:
            print()
            if steps != target:
                raise RuntimeError(
                    "Motion stopped before reaching target."
                )
            print("Movement complete.")
            return

        time.sleep(0.2)

    raise TimeoutError("Movement timeout.")

def move_absolute(ser, target):
    status = get_status(ser)

    if status["busy"]:
        raise RuntimeError("Robot is already moving.")

    current = get_steps(status)

    if target == current:
        print("Already at requested motor position.")
        return

    print(f"Current: {current}")
    print(f"Target:  {target}")

    approval = input(
        "Type MOVE to execute: "
    ).strip()

    if approval != "MOVE":
        print("Cancelled.")
        return

    command(
        ser,
        f"MOVE3 {target['X']} {target['Y']} {target['Z']}"
    )

    wait_for_motion(ser, target)

def jog(ser, axis, delta):
    status = get_status(ser)

    if status["busy"]:
        raise RuntimeError("Robot is already moving.")

    target = get_steps(status)
    target[axis] += delta

    move_absolute(ser, target)

def record(ser, point):
    status = get_status(ser)

    if status["busy"]:
        print("Wait until the robot stops.")
        return

    steps = get_steps(status)

    print("\nMeasure the gripper tip relative")
    print("to the robot's fixed XYZ origin.")

    x = float(input("Gripper X (cm): "))
    y = float(input("Gripper Y (cm): "))
    z = float(input("Gripper Z (cm): "))
    notes = input("Notes (optional): ").strip()

    print("\nRecording:")
    print("Motor steps:", steps)
    print("Gripper XYZ:", (x, y, z))

    if input("Save? (y/n): ").lower() == "y":
        save_point(point, steps, (x, y, z), notes)
        save_state(steps, "point_recorded")
    else:
        print("Not saved.")

def show_help():
    print("""
COMMANDS

  x 200          Jog motor X +200 steps
  x -200         Jog motor X -200 steps
  y 200          Jog motor Y +200 steps
  z -100         Jog motor Z -100 steps

  goto 0 0 0     Move to initial motor reference
  goto 200 300 0 Move to absolute motor steps

  mark P1         Record measured XYZ for P1
  status          Show current motor steps
  help            Show this menu
  quit            Exit

All motor steps are relative to the
initial pose established for this session.
""")

def main():
    print("=" * 55)
    print("EGG ROBOT - EMPIRICAL CALIBRATION")
    print("=" * 55)

    if CSV_PATH.exists() and CSV_PATH.stat().st_size:
        raise RuntimeError(
            "Calibration CSV already exists. "
            "Archive or rename it before starting "
            "a new zero-reference session."
        )

    print("\nPlace the arm at your new initial pose.")
    print("Do not move the arm manually after zeroing.")
    print("Check clearance before each movement.")
    print("Keep the motor power enabled during calibration.")

    confirm = input(
        "\nType START when ready: "
    ).strip()

    if confirm != "START":
        return

    with serial.Serial(
        PORT, BAUD,
        timeout=0.5,
        write_timeout=2
    ) as ser:

        time.sleep(2.5)
        ser.reset_input_buffer()

        status = get_status(ser)

        if status["armed"] or status["busy"]:
            raise RuntimeError(
                "Arduino must be unarmed and idle "
                "before establishing a new reference."
            )

        command(ser, "ARM")

        status = get_status(ser)

        if (
            status["armed"] != 1
            or status["busy"] != 0
            or any(status[a] != 0 for a in AXES)
        ):
            raise RuntimeError(
                "Could not establish motor zero."
            )

        save_state(
            {"X": 0, "Y": 0, "Z": 0},
            "initial_reference"
        )

        print("\nInitial motor zero established.")
        print("First, record the initial pose:")
        record(ser, "P0")

        show_help()

        while True:
            try:
                raw = input(
                    "\ncalibration> "
                ).strip()

                if not raw:
                    continue

                parts = raw.split()
                action = parts[0].lower()

                if action in ("quit", "exit", "q"):
                    break

                elif action == "help":
                    show_help()

                elif action == "status":
                    s = get_status(ser)
                    print("Motor steps:", get_steps(s))
                    print("Busy:", bool(s["busy"]))

                elif action in ("x", "y", "z"):
                    if len(parts) != 2:
                        print("Usage: x 200")
                        continue

                    axis = action.upper()
                    delta = int(parts[1])
                    jog(ser, axis, delta)

                elif action == "goto":
                    if len(parts) != 4:
                        print("Usage: goto X Y Z")
                        continue

                    target = dict(zip(
                        AXES,
                        map(int, parts[1:])
                    ))

                    move_absolute(ser, target)

                elif action == "mark":
                    if len(parts) != 2:
                        print("Usage: mark P1")
                        continue

                    record(ser, parts[1])

                else:
                    print(
                        "Unknown command. Type help."
                    )

            except KeyboardInterrupt:
                print("\nStopping motion...")
                try:
                    command(ser, "STOP")
                    deadline = time.monotonic() + 20

                    while time.monotonic() < deadline:
                        s = get_status(ser)
                        save_state(
                            get_steps(s), "stopping"
                        )
                        if not s["busy"]:
                            break
                        time.sleep(0.2)

                    print("Stop requested.")
                except Exception as e:
                    print("STOP could not be confirmed:", e)
                break

            except ValueError:
                print("Invalid numeric input.")

            except (
                RuntimeError,
                TimeoutError,
                serial.SerialException
            ) as e:
                print("Controller error:", e)
                print(
                    "Verify motor position before continuing."
                )
                break

    print("\nCalibration session ended.")
    print("CSV:", CSV_PATH)
    print("State:", STATE_PATH)

if __name__ == "__main__":
    main()
