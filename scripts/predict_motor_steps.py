
import numpy as np

# Each row:
# [global motor X, global motor Y, global motor Z,
#  gripper X, gripper Y, gripper Z]

DATA = np.array([
    [0,    0,    0,     0,   6,    5],
    [200,  0,    0,     0,   7.5,  4.5],
    [400,  0,    0,     0,   9,    3.2],
    [400,  200,  0,     0,   10,   3.5],
    [400,  400,  0,     0,   11.5, 4],
    [400,  400,  200,  -2,   11,   4],
    [400,  400, -200,   2,   11,   4],
    [800,  400,  0,     0,   13,   1.7],
    [400,  800,  0,     0,   14,   5.1],
    [800,  800,  0,     0,   15.2, 3],
    [800,  800,  350,  -4,   15,   2.8],

    # Second session converted to global coordinates
    [200,  490, -250,   2.5, 12,   5.5],
    [-200, 490, -650,   5.7, 8.7,  7],
    [-200, 790,  550,  -5.7, 10.5, 7.8],
    [300, 1790, -250,   2.5, 17.5, 10.2],
], dtype=float)

MOTOR = DATA[:, :3]
XYZ = DATA[:, 3:]

# Current position in original global coordinates.
CURRENT_GLOBAL = np.array([300, 1290, -250])

# New session's P0 = original P10.
SESSION_OFFSET = np.array([800, 800, 350])

# Scale XYZ dimensions for nearest-neighbor selection.
# Units are centimeters.
XYZ_SCALE = np.array([3.0, 4.0, 3.0])

NEIGHBORS = 4


def predict(target):
    distances = np.linalg.norm(
        (XYZ - target) / XYZ_SCALE,
        axis=1
    )

    indices = np.argsort(distances)[:NEIGHBORS]

    # Exact calibration point
    if distances[indices[0]] < 1e-8:
        return MOTOR[indices[0]].copy(), indices

    # Inverse-distance weighted interpolation.
    weights = 1 / np.maximum(
        distances[indices], 1e-6
    )**2

    weights /= weights.sum()

    predicted = (
        weights[:, None] * MOTOR[indices]
    ).sum(axis=0)

    return predicted, indices


def main():
    print("=" * 55)
    print("EMPIRICAL MOTOR POSITION PREDICTOR")
    print("=" * 55)
    print("No serial connection will be opened.")
    print("Current session motor steps: -500, 490, -600")
    print("Type q to exit.")

    while True:
        raw = input(
            "\nTarget XYZ (e.g. 2 12 6): "
        ).strip()

        if raw.lower() in ("q", "quit"):
            break

        try:
            target = np.array(
                [float(v) for v in raw.split()]
            )

            if target.shape != (3,):
                raise ValueError(
                    "Enter exactly three numbers."
                )

            predicted, neighbors = predict(target)

            global_steps = np.rint(
                predicted
            ).astype(int)

            session_steps = (
                global_steps - SESSION_OFFSET
            )

            print("\nNearest measured points:")

            for idx in neighbors:
                print(
                    f"  XYZ={XYZ[idx]} "
                    f"motor={MOTOR[idx].astype(int)}"
                )

            nearest_distance = np.linalg.norm(
                XYZ[neighbors[0]] - target
            )

            print(
                "\nDistance to nearest measured XYZ: "
                f"{nearest_distance:.2f} cm"
            )

            print("\nEstimated GLOBAL motor steps:")
            print(global_steps.tolist())

            print("\nEstimated CURRENT SESSION steps:")
            print(session_steps.tolist())

            print("\nEstimated step changes from current:")
            print(
                (session_steps -
                 (CURRENT_GLOBAL - SESSION_OFFSET)
                ).tolist()
            )

            print("\nCommand for existing calibration script:")
            print(
                "goto "
                + " ".join(
                    str(v) for v in session_steps
                )
            )

            print(
                "\nCAUTION: This is interpolation, "
                "not a verified collision-free path."
            )

            if nearest_distance > 3:
                print(
                    "WARNING: Target is far from "
                    "the nearest measured position."
                )

        except ValueError as e:
            print("Invalid input:", e)


if __name__ == "__main__":
    main()

#goto -437 -256 -396