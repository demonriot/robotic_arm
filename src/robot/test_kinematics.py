"""Run from project root: python -m src.robot.test_kinematics"""
from math import isclose
from .kinematics import (ArmGeometry, JointAngles, Point3D, UnreachableTarget,
                         angle_delta_to_microsteps, forward_kinematics,
                         inverse_kinematics, joint_deltas_to_microsteps)


def test_roundtrip():
    for angles in (JointAngles(0, 0, 0), JointAngles(30, 20, -45),
                   JointAngles(-45, -20, 75), JointAngles(90, 30, -90)):
        target = forward_kinematics(angles)
        for branch in ('up', 'down'):
            recovered = inverse_kinematics(target, elbow_branch=branch)
            result = forward_kinematics(recovered)
            for actual, expected in zip((result.x, result.y, result.z),
                                        (target.x, target.y, target.z)):
                assert isclose(actual, expected, abs_tol=1e-8), (angles, branch, result, target)


def test_known_values():
    p = forward_kinematics(JointAngles(0, 0, 0))
    assert isclose(p.x, 34, abs_tol=1e-9)
    assert isclose(p.y, 0, abs_tol=1e-9)
    assert isclose(p.z, 4.5, abs_tol=1e-9)
    assert angle_delta_to_microsteps(90) == 3200
    assert joint_deltas_to_microsteps(JointAngles(0, 0, 0), JointAngles(90, 45, -45)) == (3200, 1600, -1600)
    try:
        inverse_kinematics(Point3D(100, 0, 0))
    except UnreachableTarget:
        pass
    else:
        raise AssertionError('Unreachable target must raise')


if __name__ == '__main__':
    test_roundtrip()
    test_known_values()
    target = Point3D(20, 10, 8)
    print('Target:', target)
    for branch in ('up', 'down'):
        angles = inverse_kinematics(target, elbow_branch=branch)
        print(f'{branch:4s}: angles={angles}; FK={forward_kinematics(angles)}')
    print('All kinematics tests passed.')
