
import argparse
import csv
import os
import time
from datetime import datetime

import serial

STEPS_PER_DEG = 12800 / 360

AXIS = {
    "base": "Z",
    "shoulder": "X",
    "elbow": "Y",
}

DIRECTION = {
    "base": 1,
    "shoulder": -1,
    "elbow": 1,
}

LOG_FILE = "arm_measurements.csv"


class ManualJog:
    def __init__(self, port):
        self.ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()

    def send(self, command):
        self.ser.write((command + "\n").encode())
        self.ser.flush()
        response = self.ser.readline().decode(errors="replace").strip()
        return response

    def status(self):
        response = self.send("STATUS")

        if not response.startswith("STATUS "):
            raise RuntimeError(response)

        state = {}

        for item in response.split()[1:]:
            key, value = item.split("=")
            state[key] = int(value)

        return state

    def deltas(self):
        state = self.status()

        return {
            joint: (
                DIRECTION[joint]
                * state[AXIS[joint]]
                / STEPS_PER_DEG
            )
            for joint in AXIS
        }

    def show(self):
        d = self.deltas()
        print("\nJoint changes from session start:")
        print(f"  Base:     {d['base']:+.2f}°")
        print(f"  Shoulder: {d['shoulder']:+.2f}°")
        print(f"  Elbow:    {d['elbow']:+.2f}°")
        print()

    def wait(self):
        while True:
            state = self.status()

            if not state["busy"]:
                return

            time.sleep(0.1)

    def jog(self, joint, degrees):
        if joint not in AXIS:
            print("Unknown joint.")
            return

        if abs(degrees) > 1.0:
            print("Maximum 1 degree per command.")
            return

        state = self.status()

        if not state["armed"]:
            print("Type 'arm' first.")
            return

        if state["busy"]:
            print("Motor is busy.")
            return

        response = self.send(f"JOG {joint} {degrees:.6f}")
        print(response)

        if response.startswith("OK JOGGING"):
            self.wait()
            self.show()

    def record(self):
        self.wait()
        d = self.deltas()

        print("Enter the measured object/gripper coordinates.")
        x = float(input("X from base (cm): "))
        y = float(input("Y from base (cm): "))
        z = float(input("Gripping point Z above table (cm): "))

        print("Enter measured absolute joint angles if known.")
        print("Leave blank if not measured.")

        def optional_angle(prompt):
            value = input(prompt).strip()
            return float(value) if value else ""

        base_angle = optional_angle("Base absolute angle (deg): ")
        shoulder_angle = optional_angle("Shoulder absolute angle (deg): ")
        elbow_angle = optional_angle("Elbow relative angle (deg): ")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "x_cm": x,
            "y_cm": y,
            "z_cm": z,
            "base_delta_deg": round(d["base"], 3),
            "shoulder_delta_deg": round(d["shoulder"], 3),
            "elbow_delta_deg": round(d["elbow"], 3),
            "base_abs_deg": base_angle,
            "shoulder_abs_deg": shoulder_angle,
            "elbow_abs_deg": elbow_angle,
        }

        exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not exists:
                writer.writeheader()

            writer.writerow(row)

        print(f"Saved measurement to {LOG_FILE}")

    def close(self):
        self.ser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    args = parser.parse_args()

    robot = ManualJog(args.port)

    print("\nEGG_ROBOT Manual Exploration")
    print("Current pose = session origin.")
    print("No confirmation required.")
    print()
    print("Commands:")
    print("  arm")
    print("  jog shoulder -0.5")
    print("  jog elbow 0.5")
    print("  jog base 1")
    print("  angles")
    print("  record")
    print("  status")
    print("  stop")
    print("  quit")
    print()

    try:
        while True:
            try:
                command = input("jog> ").strip()
            except EOFError:
                break

            if not command:
                continue

            parts = command.split()
            action = parts[0].lower()

            try:
                if action == "quit":
                    break

                elif action == "arm":
                    print(robot.send("ARM"))

                elif action == "status":
                    print(robot.status())

                elif action == "angles":
                    robot.show()

                elif action == "stop":
                    print(robot.send("STOP"))

                elif action == "record":
                    robot.record()

                elif action == "jog" and len(parts) == 3:
                    robot.jog(parts[1].lower(), float(parts[2]))

                else:
                    print("Unknown command.")

            except (ValueError, RuntimeError, serial.SerialException) as e:
                print("Error:", e)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        robot.close()


if __name__ == "__main__":
    main()
