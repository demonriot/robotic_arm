import math

def fk(theta1_deg, theta2_deg, theta3_deg, L1=4.0, L2=8.0, L3=6.0):
    """
    Forward kinematics for 3-DOF arm.

    Angles:
        theta1_deg: base yaw (degrees)
        theta2_deg: shoulder pitch (degrees)
        theta3_deg: elbow pitch (degrees)

    Lengths:
        L1: base to shoulder vertical offset (cm)
        L2: shoulder to elbow (cm)
        L3: elbow to wrist (cm)

    Returns:
        (x, y, z) in cm in robot base frame
    """

    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    t3 = math.radians(theta3_deg)

    r = L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    z = L1 + L2 * math.sin(t2) + L3 * math.sin(t2 + t3)

    x = r * math.cos(t1)
    y = r * math.sin(t1)

    return x, y, z

if __name__ == "__main__":
    # Example usage
    theta1 = 0  # degrees
    theta2 = -3  # degrees
    theta3 = -50  # degrees

    x, y, z = fk(theta1, theta2, theta3)
    print(f"End effector position: x={x:.2f} cm, y={y:.2f} cm, z={z:.2f} cm")