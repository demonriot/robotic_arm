"""3-DOF EGG_ROBOT kinematics. Units: cm, degrees, motor microsteps."""
from dataclasses import dataclass
from math import acos, atan2, cos, degrees, hypot, radians, sin, sqrt


@dataclass(frozen=True)
class ArmGeometry:
    shoulder_height_cm: float = 9.5
    upper_arm_cm: float = 12.0
    forearm_cm: float = 16.0
    gripper_horizontal_cm: float = 6.0
    gripper_down_cm: float = 5.0
    motor_microsteps_per_rev: int = 3200
    gear_ratio: float = 4.0

    @property
    def joint_microsteps_per_rev(self) -> float:
        return self.motor_microsteps_per_rev * self.gear_ratio


@dataclass(frozen=True)
class JointAngles:
    base_deg: float
    shoulder_deg: float
    elbow_deg: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


class UnreachableTarget(ValueError):
    pass


def forward_kinematics(angles: JointAngles, g: ArmGeometry = ArmGeometry()) -> Point3D:
    base, shoulder, elbow = map(radians, (angles.base_deg, angles.shoulder_deg, angles.elbow_deg))
    reach = (g.upper_arm_cm * cos(shoulder)
             + g.forearm_cm * cos(shoulder + elbow)
             + g.gripper_horizontal_cm)
    z = (g.shoulder_height_cm + g.upper_arm_cm * sin(shoulder)
         + g.forearm_cm * sin(shoulder + elbow) - g.gripper_down_cm)
    return Point3D(reach * cos(base), reach * sin(base), z)


def inverse_kinematics(target: Point3D, g: ArmGeometry = ArmGeometry(),
                       elbow_branch: str = 'up') -> JointAngles:
    """Return one idealized solution. Branch names select elbow-angle sign.

    Does not enforce joint limits, collisions, or physical motor zero offsets.
    Assumes wrist/gripper horizontal offset is radially outward and the
    reachable solution has nonnegative wrist radial distance.
    """
    if elbow_branch not in ('up', 'down'):
        raise ValueError("elbow_branch must be 'up' or 'down'")
    radial_target = hypot(target.x, target.y)
    rw = radial_target - g.gripper_horizontal_cm
    zw = target.z + g.gripper_down_cm - g.shoulder_height_cm
    if rw < -1e-9:
        raise UnreachableTarget('Target lies inside 6 cm radial gripper offset for this model')
    L1, L2 = g.upper_arm_cm, g.forearm_cm
    D = (rw * rw + zw * zw - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if D < -1 - 1e-9 or D > 1 + 1e-9:
        raise UnreachableTarget(f'Target outside arm reach (cosine term={D:.5f})')
    D = max(-1.0, min(1.0, D))
    elbow = acos(D) * (1 if elbow_branch == 'up' else -1)
    shoulder = atan2(zw, rw) - atan2(L2 * sin(elbow), L1 + L2 * cos(elbow))
    base = atan2(target.y, target.x)
    return JointAngles(degrees(base), degrees(shoulder), degrees(elbow))


def angle_delta_to_microsteps(delta_deg: float, g: ArmGeometry = ArmGeometry()) -> int:
    """Signed relative joint motion; physical direction sign is configured later."""
    return round(delta_deg * g.joint_microsteps_per_rev / 360.0)


def joint_deltas_to_microsteps(start: JointAngles, end: JointAngles,
                                g: ArmGeometry = ArmGeometry()) -> tuple[int, int, int]:
    return tuple(angle_delta_to_microsteps(b - a, g)
                 for a, b in zip((start.base_deg, start.shoulder_deg, start.elbow_deg),
                                 (end.base_deg, end.shoulder_deg, end.elbow_deg)))
