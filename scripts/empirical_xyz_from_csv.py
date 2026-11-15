
"""
ROBO ARM - EMPIRICAL XYZ CONTROLLER

Uses NEW calibration data from:
outputs/empirical_calibration/calibration_points.csv

Default: prediction only
Optional: --move for supervised robot movement

No minimum CSV row count.
No 3 cm nearest-point limit.

IMPORTANT:
- The latest CSV row is NOT necessarily the current robot pose.
- session_state.json stores the last recorded motor counters.
- Arduino STATUS must agree with session_state.json before movement.
- This controller has no homing or collision avoidance.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "outputs"
    / "empirical_calibration"
    / "calibration_points.csv"
)

STATE_PATH = (
    ROOT
    / "outputs"
    / "empirical_calibration"
    / "session_state.json"
)

AXES = ("X", "Y", "Z")

COLUMNS = (
    "motor_x",
    "motor_y",
    "motor_z",
    "gripper_x_cm",
    "gripper_y_cm",
    "gripper_z_cm",
)


# --------------------------------------------------
# LOAD CALIBRATION DATA
# --------------------------------------------------

def load_samples(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        missing = set(COLUMNS) - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"CSV missing columns: {sorted(missing)}"
            )

        rows = [
            [float(row[c]) for c in COLUMNS]
            for row in reader
        ]

    if not rows:
        raise ValueError("Calibration CSV is empty")

    data = np.asarray(rows, dtype=float)

    if not np.isfinite(data).all():
        raise ValueError(
            "Calibration CSV contains invalid values"
        )

    motors = data[:, :3]
    positions = data[:, 3:]

    return motors, positions


# --------------------------------------------------
# CURRENT MOTOR POSITION
# --------------------------------------------------

def get_reference_position(state_path, motors):
    """
    Returns a motor-position estimate and its source.

    session_state.json is preferred.
    Latest CSV row is only a prediction fallback.
    Neither alone verifies physical motor position.
    """

    if state_path.exists():
        try:
            saved = json.loads(
                state_path.read_text(encoding="utf-8")
            )

            steps = saved.get("steps", {})

            if all(a in steps for a in AXES):
                current = np.array(
                    [float(steps[a]) for a in AXES]
                )

                if np.isfinite(current).all():
                    return current, "session_state.json"

        except (OSError, ValueError, TypeError, KeyError):
            pass

    return motors[-1].copy(), "latest CSV row (unverified)"


# --------------------------------------------------
# EMPIRICAL MODEL
# --------------------------------------------------

class EmpiricalModel:

    def __init__(self, motors, positions):

        self.m = np.asarray(motors, dtype=float)
        self.p = np.asarray(positions, dtype=float)

        self.lo = self.m.min(axis=0)
        self.hi = self.m.max(axis=0)

        self.scale = np.maximum(
            np.ptp(self.m, axis=0),
            100.0
        )

        self.neighbors = min(12, len(self.m))

        # A full 3D affine relationship needs four
        # affinely independent motor configurations.
        design = np.column_stack((
            np.ones(len(self.m)),
            self.m - self.m[0]
        ))

        self.rank = np.linalg.matrix_rank(design)

        self.motion_ready = (
            self.rank == 4
            and np.unique(self.m, axis=0).shape[0] >= 4
        )

    def forward(self, motor):
        """
        Estimate XYZ from motor steps using
        locally weighted affine regression.
        """

        motor = np.asarray(motor, dtype=float)

        distances = np.linalg.norm(
            (self.m - motor) / self.scale,
            axis=1
        )

        ids = np.argsort(distances)[:self.neighbors]

        weights = 1.0 / (
            0.06 + distances[ids] ** 2
        )

        A = np.column_stack((
            np.ones(len(ids)),
            (self.m[ids] - motor) / self.scale
        ))

        regularization = np.diag([
            1e-8,
            0.015,
            0.015,
            0.015
        ])

        lhs = (
            A.T @ (weights[:, None] * A)
            + regularization
        )

        rhs = (
            A.T
            @ (weights[:, None] * self.p[ids])
        )

        coefficients = np.linalg.solve(lhs, rhs)

        return coefficients[0]

    def predict(
        self,
        xyz,
        current=None,
        max_error_cm=1.0
    ):
        """
        Predict motor steps for target XYZ.

        No nearest-point distance rejection.

        Motor targets remain inside the
        measured motor-coordinate ranges.
        """

        xyz = np.asarray(xyz, dtype=float)

        if xyz.shape != (3,):
            raise ValueError(
                "Enter exactly three values: X Y Z"
            )

        if not np.isfinite(xyz).all():
            raise ValueError(
                "XYZ must contain finite numbers"
            )

        distances = np.linalg.norm(
            self.p - xyz,
            axis=1
        )

        nearest = float(distances.min())

        nearest_ids = np.argsort(distances)[
            :min(8, len(self.m))
        ]

        seeds = self.m[nearest_ids]

        if current is not None:
            seeds = np.vstack((
                seeds,
                np.clip(current, self.lo, self.hi)
            ))

        # Handle motor axes that did not vary in
        # the collected calibration measurements.
        active = np.flatnonzero(
            self.hi > self.lo
        )

        candidates = []

        for seed in seeds:

            if len(active) == 0:
                steps = self.m[0].astype(int)

            else:
                def residual(active_steps):
                    full = seed.copy()
                    full[active] = active_steps

                    return self.forward(full) - xyz

                result = least_squares(
                    residual,
                    seed[active],
                    bounds=(
                        self.lo[active],
                        self.hi[active]
                    ),
                    max_nfev=160,
                    diff_step=0.005
                )

                full = seed.copy()
                full[active] = result.x

                steps = np.rint(full).astype(int)

            estimated = self.forward(steps)

            error = float(
                np.linalg.norm(estimated - xyz)
            )

            travel = 0.0

            if current is not None:
                travel = (
                    0.02
                    * np.linalg.norm(
                        (steps - current) / self.scale
                    )
                )

            candidates.append((
                error + travel,
                steps,
                estimated,
                error
            ))

        _, steps, estimated, error = min(
            candidates,
            key=lambda item: item[0]
        )

        if error > max_error_cm:
            raise ValueError(
                f"Prediction residual {error:.2f} cm "
                f"exceeds {max_error_cm:.2f} cm. "
                "Target may be outside the modeled workspace."
            )

        return steps, estimated, nearest, error


# --------------------------------------------------
# ARDUINO COMMUNICATION
# --------------------------------------------------

def read_status(ser):

    ser.reset_input_buffer()
    ser.write(b"STATUS\n")
    ser.flush()

    deadline = time.monotonic() + 4

    while time.monotonic() < deadline:

        line = ser.readline().decode(
            errors="replace"
        ).strip()

        if line.startswith("ERR"):
            raise RuntimeError(line)

        if line.startswith("STATUS"):

            values = {
                k: int(v)
                for k, v in re.findall(
                    r"(armed|busy|X|Y|Z)=(-?\d+)",
                    line
                )
            }

            if set(values) == {
                "armed", "busy", "X", "Y", "Z"
            }:
                return values

    raise RuntimeError(
        "No valid Arduino STATUS response"
    )


# --------------------------------------------------
# SUPERVISED MOVEMENT
# --------------------------------------------------

def move_once(port, baud, steps, state_path):

    import serial

    if not state_path.exists():
        raise RuntimeError(
            "Missing session_state.json. "
            "Movement requires a verified reference."
        )

    saved = json.loads(
        state_path.read_text(encoding="utf-8")
    )

    expected = saved.get("steps", {})

    if not all(a in expected for a in AXES):
        raise RuntimeError(
            "Invalid session_state.json"
        )

    expected_steps = {
        a: int(expected[a])
        for a in AXES
    }

    print("\n--- MOVEMENT SAFETY CHECK ---")
    print("Opening serial may reset the Arduino.")
    print("No homing, collision checking or path planning.")
    print("Ensure the arm has a clear path.")
    print("Verify the calibration zero is unchanged.")

    confirmation = input(
        "Type REFERENCE VERIFIED to connect: "
    ).strip()

    if confirmation != "REFERENCE VERIFIED":
        print("Cancelled")
        return

    with serial.Serial(
        port,
        baud,
        timeout=0.5,
        write_timeout=2
    ) as ser:

        time.sleep(2.5)

        status = read_status(ser)

        actual = {
            a: status[a]
            for a in AXES
        }

        if (
            not status["armed"]
            or status["busy"]
            or actual != expected_steps
        ):
            raise RuntimeError(
                "Arduino motor reference mismatch.\n"
                f"Arduino: {status}\n"
                f"Saved: {expected_steps}\n"
                "Do not move until re-referenced."
            )

        print("Current motor steps:", actual)
        print("Target motor steps:", steps.tolist())

        confirmation = input(
            "Type MOVE to execute: "
        ).strip()

        if confirmation != "MOVE":
            print("Cancelled")
            return

        command = (
            "MOVE3 %d %d %d\n" % tuple(steps)
        )

        ser.reset_input_buffer()
        ser.write(command.encode("ascii"))
        ser.flush()

        deadline = time.monotonic() + 5
        acknowledged = False

        while time.monotonic() < deadline:

            line = ser.readline().decode(
                errors="replace"
            ).strip()

            if line.startswith("ERR"):
                raise RuntimeError(line)

            if line.startswith("OK"):
                acknowledged = True
                break

        if not acknowledged:
            raise RuntimeError(
                "MOVE3 was not acknowledged. "
                "Inspect the robot before retrying."
            )

        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:

            status = read_status(ser)

            if not status["busy"]:

                actual_steps = np.array([
                    status[a] for a in AXES
                ])

                if not np.array_equal(
                    actual_steps, steps
                ):
                    raise RuntimeError(
                        "Movement stopped short. "
                        "Reference may be invalid."
                    )

                new_state = {
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                    "steps": {
                        a: int(v)
                        for a, v in zip(
                            AXES,
                            actual_steps
                        )
                    },
                    "status": "move_complete"
                }

                state_path.write_text(
                    json.dumps(new_state, indent=2),
                    encoding="utf-8"
                )

                print(
                    "Movement complete:",
                    actual_steps.tolist()
                )

                return

            time.sleep(0.25)

        raise TimeoutError(
            "Movement timed out. "
            "Inspect the arm before retrying."
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=STATE_PATH
    )

    parser.add_argument(
        "--port",
        default="COM9"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200
    )

    parser.add_argument(
        "--move",
        action="store_true"
    )

    parser.add_argument(
        "--max-error-cm",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    motors, positions = load_samples(args.csv)

    model = EmpiricalModel(
        motors,
        positions
    )

    print("\n========== ROBO ARM ==========")
    print("EMPIRICAL XYZ CONTROLLER")
    print("================================")

    print(f"Calibration points: {len(motors)}")
    print(f"Calibration file: {args.csv}")

    print(
        "Latest CSV motor steps:",
        motors[-1].astype(int).tolist()
    )

    current, source = get_reference_position(
        args.state,
        motors
    )

    print(
        "Reference motor steps:",
        current.astype(int).tolist()
    )
    print("Reference source:", source)

    print("Calibration design rank:", model.rank)

    if not model.motion_ready:
        print(
            "\nWARNING: Insufficient independent "
            "3D calibration configurations."
        )
        print(
            "Predictions may be unreliable. "
            "Physical movement is disabled."
        )

        if args.move:
            raise RuntimeError(
                "Movement requires four affinely "
                "independent motor configurations."
            )

    if args.move:
        print("\nMODE: SUPERVISED MOVEMENT")
    else:
        print("\nMODE: PREDICTION ONLY")

    print(
        "\nNo 3 cm nearest-point limit is applied."
    )
    print(
        "Targets outside the measured workspace "
        "may still be inaccurate or unreachable."
    )

    while True:

        try:
            raw = input(
                "\nTarget XYZ in cm (X Y Z), or q: "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            break

        if raw.lower() in ("q", "quit", "exit"):
            break

        try:
            xyz = np.array([
                float(v)
                for v in raw.replace(",", " ").split()
            ])

            # Reload last recorded motor position.
            current, source = get_reference_position(
                args.state,
                motors
            )

            predicted, estimated, nearest, error = (
                model.predict(
                    xyz,
                    current=current,
                    max_error_cm=args.max_error_cm
                )
            )

            print("\n---------- PREDICTION ----------")

            print(
                "Target XYZ:",
                xyz.tolist()
            )

            print(
                "Estimated XYZ:",
                np.round(estimated, 3).tolist()
            )

            print(
                "Predicted motor steps:",
                predicted.tolist()
            )

            print(
                "Reference motor steps:",
                current.astype(int).tolist()
            )

            print("Reference source:", source)

            print(
                "Motor step differences:",
                (
                    predicted
                    - current.astype(int)
                ).tolist()
            )

            print(
                "Nearest measured XYZ distance:",
                round(nearest, 3),
                "cm (informational only)"
            )

            print(
                "Model residual:",
                round(error, 3),
                "cm"
            )

            if args.move:
                move_once(
                    args.port,
                    args.baud,
                    predicted,
                    args.state
                )

        except (
            ValueError,
            RuntimeError,
            TimeoutError,
            OSError,
            np.linalg.LinAlgError
        ) as e:
            print("\nREJECTED:", e)


if __name__ == "__main__":
    main()

'''
timestamp,point,motor_x,motor_y,motor_z,gripper_x_cm,gripper_y_cm,gripper_z_cm,notes
2026-11-15T13:06:22.213688,P0,0,0,0,0.0,19.0,7.0,
2026-11-15T13:11:27.622305,P1,200,0,0,0.0,20.0,6.5,
2026-11-15T13:13:04.120061,P2,400,0,0,0.0,20.7,5.5,
2026-11-15T13:13:58.293536,P3,400,200,0,0.0,22.0,6.2,
2026-11-15T13:15:24.749971,P4,400,400,0,0.0,23.2,7.0,
2026-11-15T13:16:38.260646,P5,800,400,0,0.0,24.5,5.0,
2026-11-15T13:17:38.819392,P6,800,800,0,0.0,26.0,7.2,
2026-11-15T13:19:45.728004,P7,800,800,-200,2.5,25.0,7.2,
2026-11-15T13:21:22.682011,P8,800,800,400,-6.0,25.0,7.2,
'''