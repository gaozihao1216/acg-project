# -*- coding: utf-8 -*-
"""旋转体的质量属性：**闭式**积分，不采样。

为什么要有这一层
----------------
圆锥、圆台、球冠都是"把一条半径曲线 ``r(z)`` 绕局部 Z 轴转一圈"得到的实心体
（球冠也是：``r(z) = sqrt(R² − z²)``，它同时被平面截断）。它们的体积、质心、
惯量属于同一个积分族：

======================  ==================================================
量                      积分
======================  ==================================================
体积 ``V``              ``π ∫ r² dz``
一阶矩                  ``π ∫ r² z dz``
绕轴惯量 ``I_zz``       ``(π/2) ∫ r⁴ dz``
绕直径惯量 ``I_xx``     ``π ∫ (r⁴/4 + r² z²) dz``
======================  ==================================================

积分里只出现 ``r²`` 和它的平方，所以只要把 ``s(z) = r(z)²`` 写成**多项式系数**，
四个量就都能精确算出来——球冠也是多项式（``s = R² − z²``），圆锥与圆台更是
（``s = (a + bz)²``）。

于是"数值积分"这个担心不成立：多项式积分是闭式的，误差只有浮点舍入。这一点
必须在代码里说清楚，否则后来的人会以为这里是个近似，进而去找"更精确的实现"。

关于惯量的参照点
----------------
``I_xx`` 先按"关于 z=0 平面上的原点"积分，再用平行轴定理平移到质心；
``I_zz`` 不需要平移——它绕的是旋转轴，而旋转轴穿过质心，沿轴平移不改变它。

返回的量都是**单位密度**的：密度取 1 时质量等于体积，所以"关于质心"的
平移项直接写 ``− V·z̄²``，不必再乘质量。调用方按 ``mass / volume`` 缩放即可。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RevolutionMoments:
    """旋转体的质量属性（单位密度，惯量关于**质心**）。"""

    volume: float
    #: 质心在**输入区间坐标系**下的 z 坐标。调用方据此把原点挪到质心。
    centroid_z: float
    #: 绕垂直于轴的直径的惯量（单位密度，关于质心）
    ixx: float
    #: 绕自身对称轴的惯量（单位密度，关于质心）
    izz: float


def _poly_mul(a: "tuple[float, ...] | list[float]", b) -> list[float]:
    """多项式相乘，系数按**低次在前**排列。"""
    result = [0.0] * (len(a) + len(b) - 1)
    for i, coefficient_a in enumerate(a):
        if coefficient_a == 0.0:
            continue
        for j, coefficient_b in enumerate(b):
            result[i + j] += coefficient_a * coefficient_b
    return result


def _poly_integral(coefficients, z0: float, z1: float) -> float:
    """多项式在 ``[z0, z1]`` 上的定积分（低次在前）。"""
    total = 0.0
    for power, coefficient in enumerate(coefficients):
        if coefficient == 0.0:
            continue
        total += coefficient * (z1 ** (power + 1) - z0 ** (power + 1)) / (power + 1)
    return total


def solid_of_revolution(squared_radius, z0: float, z1: float) -> RevolutionMoments:
    """半径曲线平方 ``s(z) = r(z)²`` 绕 z 轴旋转所得实心体的质量属性。

    参数
    ----
    squared_radius : ``r(z)²`` 的多项式系数（低次在前）
    z0, z1         : 旋转体的 z 区间，要求 ``z1 > z0``

    返回
    ----
    RevolutionMoments
        ``centroid_z`` 用的是**输入坐标系**；``ixx`` / ``izz`` 已经平移到质心。

    异常
    ----
    ValueError : 区间为空，或算出的体积非正（半径曲线恒零）
    """
    if z1 <= z0:
        raise ValueError(f"旋转体的 z 区间必须满足 z1 > z0，收到：[{z0}, {z1}]")

    s = tuple(float(value) for value in squared_radius)
    s_squared = _poly_mul(s, s)

    volume = math.pi * _poly_integral(s, z0, z1)
    if volume <= 0.0:
        raise ValueError(f"旋转体体积必须为正，收到：{volume:.6g}（半径曲线恒为零？）")

    first_moment = math.pi * _poly_integral(_poly_mul(s, (0.0, 1.0)), z0, z1)
    centroid_z = first_moment / volume

    izz = 0.5 * math.pi * _poly_integral(s_squared, z0, z1)

    # 关于原点的 I_xx = π ∫ (r⁴/4 + r² z²) dz，再平移到质心
    integral_r4 = _poly_integral(s_squared, z0, z1)
    integral_r2_z2 = _poly_integral(_poly_mul(s, (0.0, 0.0, 1.0)), z0, z1)
    ixx_origin = math.pi * (0.25 * integral_r4 + integral_r2_z2)
    ixx = ixx_origin - volume * centroid_z * centroid_z

    return RevolutionMoments(
        volume=volume,
        centroid_z=centroid_z,
        ixx=ixx,
        izz=izz,
    )
