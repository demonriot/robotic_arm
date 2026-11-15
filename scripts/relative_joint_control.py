
import time
import serial

# ---------------- Configuration ----------------

PORT = "COM9"
BAUD = 115200

# Existing Arduino MOVE3 firmware limits
MAX_SESSION_STEPS = 1400

# Small relative movements for initial testing
DEFAULT_STEP = 100

# Axis mapping:
# X = shoulder
# Y = elbow
# Z = base
AXES = {
    "shoulder": "X",
    "elbow": "Y",
    "base": "Z",
}

# ------------------------------------------------

class Robot:
    def __init__(self, port=PORT):
        self.ser = serial.Serial(
            port,
            BAUD,
            timeout=2
        )

        # Arduino Mega usually resets when serial opens.
        time.sleep(2)

        self.positions = {
            "X": 0,
            "Y": 0,
            "Z": 0
        }

    def send(self, command):
        self.ser.reset_input_buffer()
        self.ser.write((command + "\n").encode())
        self.ser.flush()

        response = self.ser.readline().decode(
            errors="replace"
        ).strip()

        print(f"Arduino: {response}")
        return response

    def status(self):
        return self.send("STATUS")

    def arm(self):
        print(
            "\nWARNING: ARM resets all motor "
            "step counters to zero."
        )
        print(
            "Ensure the arm is supported and "
            "the workspace is clear."
        )

        confirm = input("Type ARM to continue: ")

        if confirm != "ARM":
            return

        response = self.send("ARM")

        if response.startswith("OK"):
            self.positions = {
                "X": 0,
                "Y": 0,
                "Z": 0
            }

    def move_relative(self, joint, delta):
        axis = AXES[joint]

        current = self.positions[axis]
        target = current + delta

        if abs(target) > MAX_SESSION_STEPS:
            print("Movement rejected: session limit")
            return

        target_positions = self.positions.copy()
        target_positions[axis] = target

        command = (
            f"MOVE3 "
            f"{target_positions['X']} "
            f"{target_positions['Y']} "
            f"{target_positions['Z']}"
        )

        print(
            f"\nMoving {joint}: "
            f"{current} -> {target} steps"
        )

        response = self.send(command)

        if not response.startswith("OK"):
            print("Move not accepted")
            return

        # Wait for coordinated movement to finish.
        while True:
            time.sleep(0.15)
            response = self.status()

            if "busy=0" in response:
                break

        self.positions = target_positions

    def close(self):
        self.ser.close()


def main():
    robot = Robot()

    print("\nEGG_ROBOT - Relative Joint Control")
    print("--------------------------------")
    print("status")
    print("arm")
    print("move base 100")
    print("move shoulder -100")
    print("move elbow 100")
    print("stop")
    print("quit")

    try:
        while True:
            command = input("\nrobot> ").strip().lower()

            if command == "quit":
                break

            elif command == "status":
                robot.status()

            elif command == "arm":
                robot.arm()

            elif command == "stop":
                robot.send("STOP")

            elif command.startswith("move "):
                parts = command.split()

                if len(parts) != 3:
                    print("Usage: move JOINT STEPS")
                    continue

                joint = parts[1]

                if joint not in AXES:
                    print("Unknown joint")
                    continue

                try:
                    delta = int(parts[2])
                except ValueError:
                    print("Steps must be an integer")
                    continue

                if abs(delta) > DEFAULT_STEP:
                    print(
                        f"Maximum per command: "
                        f"{DEFAULT_STEP} steps"
                    )
                    continue

                robot.move_relative(joint, delta)

            else:
                print("Unknown command")

    finally:
        robot.close()


if __name__ == "__main__":
    main()
