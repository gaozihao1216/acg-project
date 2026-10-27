# -*- coding: utf-8 -*-
"""基础数学工具（四元数）。"""

from .quaternion import (
    quat_conjugate,
    quat_dot,
    quat_from_axis_angle,
    quat_from_euler_degrees,
    quat_from_euler_xyz,
    quat_identity,
    quat_integrate,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
    quat_to_matrix,
)

__all__ = [
    "quat_identity",
    "quat_normalize",
    "quat_multiply",
    "quat_conjugate",
    "quat_dot",
    "quat_from_axis_angle",
    "quat_from_euler_xyz",
    "quat_from_euler_degrees",
    "quat_rotate_vector",
    "quat_to_matrix",
    "quat_integrate",
]
