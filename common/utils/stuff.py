import numpy as np


def wrap_angle(angle: float):
    """
    Args:
        angle (float): angle in [rad]

    Returns:
        float: Angle in [rad] that is wrapped to [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle 
