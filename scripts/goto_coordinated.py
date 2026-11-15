
import argparse
import math
import time

import serial


# ============================================================
# ARM GEOMETRY (cm)
# ============================================================

H = 15.0
L1 = 12.0
L2 = 12.0

GRIP_X = 6.0
GRIP_Z = -5.0

STEPS_PER_DEG = 12800.0 / 360.0


# ============================================================
# CALIBRATION
#
# Latest estimated physical angles:
#
# Base     = 34.325 degrees
# Shoulder = 38.340625 degrees
# Elbow    = -76.6875 degrees
#
# IMPORTANT:
# These angles must match the actual physical pose
# when ARM establishes the new zero-step reference.
# ============================================================

CAL_ANGLES = {
    "base": 34.325,
    "shoulder": 38.340625,
    "elbow": -76.6875,
}

CAL_STEPS = {
    "base": 0,
    "shoulder": 0,
    "elbow": 0,
}


# ============================================================
# MOTOR CONFIGURATION
#
# RAMPS X = shoulder
# RAMPS Y = elbow
# RAMPS Z = base
# ============================================================

AXIS = {
    "base": "Z",
    "shoulder": "X",
    "elbow": "Y",
}

SIGN = {
    "base": 1,
    "shoulder": -1,
    "elbow": 1,
}


# ============================================================
# SAFETY LIMITS
# ============================================================

# Must match the Arduino firmware.
MAX_SESSION_STEPS = 1400

# Allows approximately 35-degree base movement.
MAX_CHANGE_DEG = 36.0

MAX_RADIUS = 26.0
MIN_HEIGHT = 7.0
MAX_HEIGHT = 18.0

# Provisional joint limits.
# Only use base 0 if physically verified safe.
JOINT_LIMITS = {
    "base": (0.0, 50.0),
    "shoulder": (25.0, 50.0),
    "elbow": (-90.0, -60.0),
}


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def fk(angles):

    b = math.radians(angles["base"])
    s = math.radians(angles["shoulder"])
    e = math.radians(angles["elbow"])

    r = (
        L1 * math.cos(s)
        + L2 * math.cos(s + e)
        + GRIP_X
    )

    z = (
        H
        + L1 * math.sin(s)
        + L2 * math.sin(s + e)
        + GRIP_Z
    )

    x = r * math.cos(b)
    y = r * math.sin(b)

    return x, y, z


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def ik(x, y, z):

    r = math.hypot(x, y)

    rw = r - GRIP_X
    zw = z - H - GRIP_Z

    d = (
        rw**2 + zw**2
        - L1**2 - L2**2
    ) / (2 * L1 * L2)

    if not -1.0 <= d <= 1.0:
        raise ValueError(
            "Target outside geometric reach"
        )

    e = -math.acos(
        max(-1.0, min(1.0, d))
    )

    s = (
        math.atan2(zw, rw)
        - math.atan2(
            L2 * math.sin(e),
            L1 + L2 * math.cos(e)
        )
    )

    b = math.atan2(y, x)

    return {
        "base": math.degrees(b),
        "shoulder": math.degrees(s),
        "elbow": math.degrees(e),
    }


# ============================================================
# ANGLE / STEP CONVERSIONS
# ============================================================

def angles_from_steps(steps):

    return {
        j: (
            CAL_ANGLES[j]
            + SIGN[j]
            * (steps[j] - CAL_STEPS[j])
            / STEPS_PER_DEG
        )
        for j in CAL_ANGLES
    }


def steps_from_angles(angles):

    return {
        j: (
            CAL_STEPS[j]
            + round(
                (angles[j] - CAL_ANGLES[j])
                * STEPS_PER_DEG
                * SIGN[j]
            )
        )
        for j in CAL_ANGLES
    }


# ============================================================
# SAFETY VALIDATION
# ============================================================

def check_angles(angles):

    for joint, angle in angles.items():

        low, high = JOINT_LIMITS[joint]

        if not low <= angle <= high:
            raise ValueError(
                f"{joint} angle {angle:.3f} "
                f"outside [{low}, {high}]"
            )


def check_workspace(angles):

    x, y, z = fk(angles)

    radius = math.hypot(x, y)

    if radius > MAX_RADIUS:
        raise ValueError(
            f"Radius {radius:.2f} exceeds "
            f"{MAX_RADIUS} cm"
        )

    if not MIN_HEIGHT <= z <= MAX_HEIGHT:
        raise ValueError(
            f"Height {z:.2f} outside "
            f"[{MIN_HEIGHT}, {MAX_HEIGHT}] cm"
        )


def check_joint_path(start_steps, target_steps):

    largest = max(
        abs(target_steps[j] - start_steps[j])
        for j in start_steps
    )

    samples = max(30, largest // 10)

    for i in range(samples + 1):

        t = i / samples

        intermediate = {
            j: round(
                start_steps[j]
                + t * (
                    target_steps[j] - start_steps[j]
                )
            )
            for j in start_steps
        }

        angles = angles_from_steps(intermediate)

        check_angles(angles)
        check_workspace(angles)


# ============================================================
# ROBOT CONTROLLER
# ============================================================

class Robot:

    def __init__(self, port):

        self.ser = serial.Serial(
            port,
            115200,
            timeout=2
        )

        # Arduino Mega commonly resets when serial opens.
        time.sleep(2)

        self.ser.reset_input_buffer()

    def send(self, command):

        self.ser.write(
            (command + "\n").encode()
        )

        self.ser.flush()

        reply = self.ser.readline().decode(
            errors="replace"
        ).strip()

        if not reply:
            raise RuntimeError(
                f"No response to {command}"
            )

        return reply

    def status(self):

        reply = self.send("STATUS")

        if not reply.startswith("STATUS "):
            raise RuntimeError(reply)

        data = {}

        for item in reply.split()[1:]:

            key, value = item.split("=")

            data[key] = int(value)

        return data

    def steps(self):

        state = self.status()

        return {
            joint: state[AXIS[joint]]
            for joint in AXIS
        }

    def pose(self):

        angles = angles_from_steps(
            self.steps()
        )

        return angles, fk(angles)

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    def calibrate(self):

        state = self.status()

        print("Arduino:", state)

        if state["busy"]:
            raise RuntimeError(
                "Cannot calibrate while moving"
            )

        if state["armed"]:
            raise RuntimeError(
                "Arduino already armed"
            )

        print("\nCalibration reference:")

        for joint, angle in CAL_ANGLES.items():
            print(f"  {joint}: {angle:.4f} degrees")

        print(
            "\nVerify these match the ACTUAL "
            "physical joint angles."
        )

        answer = input(
            "Type CALIBRATE to arm: "
        ).strip()

        if answer != "CALIBRATE":
            raise RuntimeError(
                "Calibration cancelled"
            )

        reply = self.send("ARM")

        if reply != "OK ARMED":
            raise RuntimeError(reply)

        print("Calibration established.")

    # --------------------------------------------------------
    # WAIT FOR MOVEMENT
    # --------------------------------------------------------

    def wait_idle(self):

        while True:

            state = self.status()

            if not state["armed"]:
                raise RuntimeError(
                    "Arduino became disarmed"
                )

            if not state["busy"]:
                return

            time.sleep(0.1)

    # --------------------------------------------------------
    # COMMON MOVEMENT VALIDATION
    # --------------------------------------------------------

    def validate_move(self, target_angles):

        self.wait_idle()

        current_steps = self.steps()

        current_angles = angles_from_steps(
            current_steps
        )

        target_steps = steps_from_angles(
            target_angles
        )

        print("\n--- MOVEMENT PREVIEW ---")

        print("Current angles:", current_angles)
        print("Target angles:", target_angles)

        print("Current steps:", current_steps)
        print("Target steps:", target_steps)

        print("Current XYZ:", fk(current_angles))
        print("Target XYZ:", fk(target_angles))

        for joint in current_angles:

            change = (
                target_angles[joint]
                - current_angles[joint]
            )

            print(
                f"{joint}: {change:+.3f} degrees"
            )

            if abs(change) > MAX_CHANGE_DEG:
                raise ValueError(
                    f"{joint} change exceeds "
                    f"{MAX_CHANGE_DEG} degrees"
                )

            if (
                abs(target_steps[joint])
                > MAX_SESSION_STEPS
            ):
                raise ValueError(
                    f"{joint} exceeds "
                    "session travel limit"
                )

        check_joint_path(
            current_steps,
            target_steps
        )

        print("\nPreliminary checks: PASSED")
        print(
            "Check the real arm's clearance "
            "before executing."
        )

        return target_steps

    # --------------------------------------------------------
    # CARTESIAN MOVEMENT
    # --------------------------------------------------------

    def preview(self, x, y, z):

        if not all(
            math.isfinite(v)
            for v in (x, y, z)
        ):
            raise ValueError(
                "XYZ coordinates must be finite"
            )

        target_angles = ik(x, y, z)

        return self.validate_move(
            target_angles
        )

    def goto(self, x, y, z):

        target_steps = self.preview(
            x, y, z
        )

        self.execute_move(target_steps)

    # --------------------------------------------------------
    # DIRECT JOINT MOVEMENT
    # --------------------------------------------------------

    def preview_joint(self, joint, angle):

        joint = joint.lower()

        if joint not in CAL_ANGLES:
            raise ValueError(
                "Joint must be base, shoulder, or elbow"
            )

        if not math.isfinite(angle):
            raise ValueError(
                "Angle must be finite"
            )

        current_angles, _ = self.pose()

        target_angles = current_angles.copy()

        target_angles[joint] = angle

        return self.validate_move(
            target_angles
        )

    def joint(self, joint, angle):

        target_steps = self.preview_joint(
            joint, angle
        )

        self.execute_move(target_steps)

    # --------------------------------------------------------
    # EXECUTE MOVE3
    # --------------------------------------------------------

    def execute_move(self, target_steps):

        answer = input(
            "\nType MOVE to execute: "
        ).strip()

        if answer != "MOVE":
            print("Movement cancelled.")
            return

        command = (
            f"MOVE3 "
            f"{target_steps['shoulder']} "
            f"{target_steps['elbow']} "
            f"{target_steps['base']}"
        )

        reply = self.send(command)

        print("Arduino:", reply)

        if reply == "OK ALREADY_THERE":
            return

        if reply != "OK MOVING":
            raise RuntimeError(reply)

        self.wait_idle()

        final_steps = self.steps()

        final_angles = angles_from_steps(
            final_steps
        )

        print("\nMovement completed.")
        print("Final steps:", final_steps)
        print("Final angles:", final_angles)
        print("Predicted XYZ:", fk(final_angles))

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        self.ser.close()


# ============================================================
# INTERACTIVE CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        required=True
    )

    args = parser.parse_args()

    robot = Robot(args.port)

    try:

        print("\nEGG_ROBOT Joint + XYZ Controller")

        print("\nAvailable commands:")
        print("  status")
        print("  pose")
        print("  preview X Y Z")
        print("  goto X Y Z")
        print("  preview_joint JOINT ANGLE")
        print("  joint JOINT ANGLE")
        print("  stop")
        print("  quit")

        robot.calibrate()

        while True:

            line = input("\nxyz> ").strip()

            parts = line.split()

            if not parts:
                continue

            command = parts[0].lower()

            try:

                if command == "quit":
                    break

                elif command == "status":
                    print(robot.status())

                elif command == "pose":

                    angles, xyz = robot.pose()

                    print("Angles:", angles)
                    print("Predicted XYZ:", xyz)

                elif (
                    command in ("preview", "goto")
                    and len(parts) == 4
                ):

                    x, y, z = map(
                        float,
                        parts[1:]
                    )

                    if command == "preview":
                        robot.preview(x, y, z)

                    else:
                        robot.goto(x, y, z)

                elif (
                    command in (
                        "preview_joint",
                        "joint"
                    )
                    and len(parts) == 3
                ):

                    joint_name = parts[1]

                    target_angle = float(
                        parts[2]
                    )

                    if command == "preview_joint":

                        robot.preview_joint(
                            joint_name,
                            target_angle
                        )

                    else:

                        robot.joint(
                            joint_name,
                            target_angle
                        )

                elif command == "stop":
                    print(robot.send("STOP"))

                else:
                    print("Unknown command")

            except (
                ValueError,
                RuntimeError,
                serial.SerialException
            ) as exc:

                print("ERROR:", exc)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        robot.close()


if __name__ == "__main__":
    main()
