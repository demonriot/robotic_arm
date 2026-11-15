
import math

# ============================================================
# ROBOT GEOMETRY (cm)
# ============================================================

L1 = 12.0
L2 = 12.0

SHOULDER_HEIGHT = 15.0

GRIPPER_RADIAL_OFFSET = 6.0
GRIPPER_VERTICAL_OFFSET = -5.0

# ============================================================
# MOTOR CALIBRATION
# ============================================================

STEPS_PER_DEGREE = 35.5555556

# Motor directions established during manual testing
BASE_SIGN = +1
SHOULDER_SIGN = -1
ELBOW_SIGN = +1

# ============================================================
# NEW MEASURED REFERENCE POSE
# ============================================================

REF_STEPS = {
    "X": 500,     # Shoulder
    "Y": -400,    # Elbow
    "Z": 1150,    # Base
}

REF_ANGLES = {
    "base": math.degrees(math.atan2(3.3, 25.0)),
    "shoulder": 20.0,
    "elbow": -55.0,
}

REF_POSITION = {
    "x": 25.0,
    "y": 3.3,
    "z": 4.5,
}


# ============================================================
# STEP COUNTS -> JOINT ANGLES
# ============================================================

def steps_to_angles(x_steps, y_steps, z_steps):
    shoulder_delta = (
        SHOULDER_SIGN
        * (x_steps - REF_STEPS["X"])
        / STEPS_PER_DEGREE
    )

    elbow_delta = (
        ELBOW_SIGN
        * (y_steps - REF_STEPS["Y"])
        / STEPS_PER_DEGREE
    )

    base_delta = (
        BASE_SIGN
        * (z_steps - REF_STEPS["Z"])
        / STEPS_PER_DEGREE
    )

    return {
        "base": REF_ANGLES["base"] + base_delta,
        "shoulder": REF_ANGLES["shoulder"] + shoulder_delta,
        "elbow": REF_ANGLES["elbow"] + elbow_delta,
    }


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(base_deg, shoulder_deg, elbow_deg):
    base = math.radians(base_deg)
    shoulder = math.radians(shoulder_deg)
    elbow = math.radians(elbow_deg)

    forearm = shoulder + elbow

    radial_distance = (
        L1 * math.cos(shoulder)
        + L2 * math.cos(forearm)
        + GRIPPER_RADIAL_OFFSET
    )

    z = (
        SHOULDER_HEIGHT
        + L1 * math.sin(shoulder)
        + L2 * math.sin(forearm)
        + GRIPPER_VERTICAL_OFFSET
    )

    x = radial_distance * math.cos(base)
    y = radial_distance * math.sin(base)

    return x, y, z


# ============================================================
# REPORT
# ============================================================

def print_report(x_steps, y_steps, z_steps):
    angles = steps_to_angles(
        x_steps,
        y_steps,
        z_steps
    )

    x, y, z = forward_kinematics(
        angles["base"],
        angles["shoulder"],
        angles["elbow"]
    )

    print("\n" + "=" * 52)
    print("EGG_ROBOT - FORWARD KINEMATICS")
    print("=" * 52)

    print(
        f"\nMotor steps: "
        f"X={x_steps}, Y={y_steps}, Z={z_steps}"
    )

    print("\nCalculated joint angles:")
    print(f"  Base:     {angles['base']:.2f} deg")
    print(f"  Shoulder: {angles['shoulder']:.2f} deg")
    print(f"  Elbow:    {angles['elbow']:.2f} deg")

    print(
        f"  Forearm:  "
        f"{angles['shoulder'] + angles['elbow']:.2f} deg"
    )

    print("\nPredicted gripper position:")
    print(f"  X = {x:.2f} cm")
    print(f"  Y = {y:.2f} cm")
    print(f"  Z = {z:.2f} cm")

    if (
        x_steps == REF_STEPS["X"]
        and y_steps == REF_STEPS["Y"]
        and z_steps == REF_STEPS["Z"]
    ):
        dx = x - REF_POSITION["x"]
        dy = y - REF_POSITION["y"]
        dz = z - REF_POSITION["z"]

        error = math.sqrt(
            dx**2 + dy**2 + dz**2
        )

        print("\nMeasured reference position:")
        print(
            f"  X = {REF_POSITION['x']:.2f} cm"
        )
        print(
            f"  Y = {REF_POSITION['y']:.2f} cm"
        )
        print(
            f"  Z = {REF_POSITION['z']:.2f} cm"
        )

        print("\nPrediction error:")
        print(f"  dX = {dx:+.2f} cm")
        print(f"  dY = {dy:+.2f} cm")
        print(f"  dZ = {dz:+.2f} cm")
        print(f"  Total = {error:.2f} cm")

        if error > 2.0:
            print(
                "\nWARNING: Geometry needs validation "
                "before autonomous movement."
            )

    print("=" * 52)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nEGG_ROBOT - Kinematics Test")
    print("No Arduino connection. No motor movement.")

    print_report(
        REF_STEPS["X"],
        REF_STEPS["Y"],
        REF_STEPS["Z"]
    )

    while True:
        print(
            "\nEnter X Y Z motor steps "
            "to predict gripper position."
        )
        print("Press Enter to exit.")

        raw = input("Steps: ").strip()

        if not raw:
            break

        try:
            values = raw.replace(",", " ").split()

            if len(values) != 3:
                raise ValueError(
                    "Enter exactly three step counts."
                )

            x_steps, y_steps, z_steps = map(
                int,
                values
            )

            print_report(
                x_steps,
                y_steps,
                z_steps
            )

        except ValueError as error:
            print("Invalid input:", error)


if __name__ == "__main__":
    main()
