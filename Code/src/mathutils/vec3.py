# -*- coding: utf-8 -*-
"""三维向量基础运算。

这里的函数都很短，但**统一命名和参数顺序**能显著减少求解器里的低级错误
（例如把 ``cross(a, b)`` 写成 ``cross(b, a)`` 会让力矩方向整体反号）。

全部使用元组 ``(x, y, z)`` 作为输入输出，不引入 numpy 依赖：
本项目的仿真规模（数百个刚体）用纯 Python 完全够，而且便于在 Blender
内部直接复用。若日后规模上升，只需把这一个模块换成 numpy 实现即可。

命名约定
--------
- ``add / sub / scale`` : 逐分量运算
- ``dot / cross``       : 点积与叉积
- ``length``            : 向量模长
- ``normalize``         : 归一化；零向量会抛异常而不是静默返回零
"""

from __future__ import annotations

import math
from typing import Sequence

Vec3 = Sequence[float]


def add(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    """向量加法 ``a + b``。"""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    """向量减法 ``a - b``。

    注意参数顺序：``sub(b_position, a_position)`` 得到"从 A 指向 B"的向量，
    这正是接触法向所需的约定。
    """
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, s: float) -> tuple[float, float, float]:
    """向量数乘 ``s * a``。"""
    return (a[0] * s, a[1] * s, a[2] * s)


def madd(a: Vec3, b: Vec3, s: float) -> tuple[float, float, float]:
    """返回 ``a + s * b``。

    物理里最常见的组合（积分位置、施加冲量）都是这个形式，单独提供可以
    少建一次中间元组。
    """
    return (a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    """点积 ``a · b``。"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    """叉积 ``a × b``。

    顺序很关键：``cross(r, F)`` 是力矩，``cross(r, n)`` 出现在有效质量公式里。
    反过来写会得到方向相反的力矩，表现为物体被撞后朝反方向旋转。
    """
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a: Vec3) -> float:
    """向量模长。"""
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def length_squared(a: Vec3) -> float:
    """模长平方。比较长度时用它可以省一次开方。"""
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]


def normalize(a: Vec3) -> tuple[float, float, float]:
    """归一化。

    异常
    ----
    ValueError : 输入是零向量（或接近零），此时方向没有定义。

    故意抛异常而不是返回单位向量：静默返回会让"两球质心完全重合"这类
    退化情形变成难以追踪的错误结果，而不是当场暴露。
    """
    n = length(a)
    if n < 1e-12:
        raise ValueError(f"无法归一化零向量：{tuple(a)}")
    inv = 1.0 / n
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def zeros() -> tuple[float, float, float]:
    """返回零向量，避免到处写字面量。"""
    return (0.0, 0.0, 0.0)


def orthogonal_basis(n: Vec3) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """给定单位法向 ``n``，构造两个与之正交的单位切向。

    用于把摩擦力分解到两个切向方向上。

    实现说明：不能简单地取 ``cross(n, (1,0,0))``——当 ``n`` 恰好接近 X 轴时
    叉积会退化成零向量。这里先挑一个与 ``n`` **最不平行**的坐标轴再叉乘，
    保证数值稳定。
    """
    ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
    if ax <= ay and ax <= az:
        helper = (1.0, 0.0, 0.0)
    elif ay <= az:
        helper = (0.0, 1.0, 0.0)
    else:
        helper = (0.0, 0.0, 1.0)

    t1 = normalize(cross(n, helper))
    t2 = cross(n, t1)          # 已是单位向量，无需再归一化
    return t1, t2
