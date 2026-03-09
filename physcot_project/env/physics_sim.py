"""
Base physics simulation utilities for PhysCoT experiments.
Pure Python / NumPy 2D rigid body dynamics.
"""

import numpy as np


# Physical constants
G = 9.81  # gravity (m/s^2)


def rotate_point(px, py, cx, cy, angle):
    """Rotate point (px,py) around center (cx,cy) by angle (radians)."""
    s, c = np.sin(angle), np.cos(angle)
    px -= cx; py -= cy
    return cx + c * px - s * py, cy + s * px + c * py


def segment_intersects(p1, p2, p3, p4):
    """Test if segment p1-p2 intersects segment p3-p4. Returns bool."""
    def cross2d(v, w):
        return v[0] * w[1] - v[1] * w[0]
    d1 = np.array(p2) - np.array(p1)
    d2 = np.array(p4) - np.array(p3)
    cross = cross2d(d1, d2)
    if abs(cross) < 1e-10:
        return False  # parallel
    t = cross2d(np.array(p3) - np.array(p1), d2) / cross
    u = cross2d(np.array(p3) - np.array(p1), d1) / cross
    return 0 <= t <= 1 and 0 <= u <= 1
