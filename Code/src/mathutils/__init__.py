# -*- coding: utf-8 -*-
"""基础数学工具：四元数（``quaternion``）、三维向量（``vec3``）、
二维凸多边形（``poly2d``）。

子模块整体导出，而不是逐个函数列进 ``__all__``：
:func:`quat_rotate_vector`、:func:`vec3.madd` 这类名字本身已经带了前缀，
再摊平到包顶层反而看不出出处。
"""

from . import poly2d, quaternion, vec3
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
    "quaternion",
    "vec3",
    "poly2d",
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
