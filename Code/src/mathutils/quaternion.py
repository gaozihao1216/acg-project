# -*- coding: utf-8 -*-
"""四元数工具模块。

本模块只做纯数学计算，**不依赖 numpy、不依赖 bpy**，因此可以同时被
Blender 渲染脚本和后续 W2 的刚体求解器使用。

约定
----
1. 四元数统一表示为 ``(w, x, y, z)``，其中 ``w`` 是实部（标量部分）。
2. 只有单位四元数才表示"旋转"；本模块所有旋转相关函数内部都会先做归一化。
3. 向量统一表示为长度为 3 的序列 ``(x, y, z)``。
4. 旋转矩阵采用**列向量右乘**约定：``v' = R @ v``。

为什么用四元数而不是欧拉角
--------------------------
刚体姿态积分时必须避免万向节死锁（gimbal lock），并且要能对任意旋转做
稳定的插值与归一化。四元数只有 4 个分量、一个约束（模长为 1），
积分后只需重新归一化即可，是刚体动力学中的标准做法。
"""

from __future__ import annotations

import math
from typing import Sequence

# 类型别名，仅用于提升可读性；运行时不产生额外开销。
Vec3 = Sequence[float]
Quat = Sequence[float]


# ---------------------------------------------------------------------------
# 基础构造与规范化
# ---------------------------------------------------------------------------

def quat_identity() -> tuple[float, float, float, float]:
    """返回单位四元数 (1, 0, 0, 0)，它表示"不旋转"。

    场景 JSON 中出现 ``"orientation": [1, 0, 0, 0]`` 即表示物体朝向与
    世界坐标轴对齐。
    """
    return (1.0, 0.0, 0.0, 0.0)


def quat_normalize(q: Quat) -> tuple[float, float, float, float]:
    """把任意四元数归一化为单位四元数。

    数值积分之后四元数的模长会缓慢漂离 1，必须每一步都归一化，
    否则旋转会逐渐被缩放（表现为物体慢慢缩小或形变）。

    参数
    ----
    q : 任意非零四元数 (w, x, y, z)

    异常
    ----
    ValueError : 输入模长过小（接近零），无法确定旋转方向。
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError(f"四元数模长过小，无法归一化：{tuple(q)}")
    inv = 1.0 / norm
    return (w * inv, x * inv, y * inv, z * inv)


# ---------------------------------------------------------------------------
# 四元数代数运算
# ---------------------------------------------------------------------------

def quat_multiply(a: Quat, b: Quat) -> tuple[float, float, float, float]:
    """四元数乘法（Hamilton 积），返回 ``a ⊗ b``。

    几何含义：结果表示"先施加 b 旋转，再施加 a 旋转"。

    注意乘法**不可交换**，``a ⊗ b != b ⊗ a``。姿态积分
    ``q_new = Δq ⊗ q_old`` 中增量必须放在左边，表示在**世界坐标系**下
    叠加旋转；放在右边则表示在**局部坐标系**下叠加。
    """
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q: Quat) -> tuple[float, float, float, float]:
    """返回共轭四元数 (w, -x, -y, -z)。

    对单位四元数而言，共轭等于逆，因此本函数可直接用于"反向旋转"。
    """
    return (float(q[0]), -float(q[1]), -float(q[2]), -float(q[3]))


def quat_dot(a: Quat, b: Quat) -> float:
    """四元数点积，用于判断两个姿态的接近程度（W2 睡眠判定会用到）。

    返回值的绝对值越接近 1，表示两个旋转越接近。
    """
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])


# ---------------------------------------------------------------------------
# 从角度构造旋转
# ---------------------------------------------------------------------------

def quat_from_axis_angle(axis: Vec3, angle: float) -> tuple[float, float, float, float]:
    """绕任意单位轴 ``axis`` 旋转 ``angle`` 弧度，返回对应四元数。

    公式：``q = (cos(θ/2), sin(θ/2) * axis)``。
    注意是**半角**，这是四元数表示旋转的标准形式。

    参数
    ----
    axis  : 旋转轴方向，内部会自动归一化；若长度过小则抛出异常。
    angle : 旋转角，单位为**弧度**（不是角度）。
    """
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length < 1e-12:
        raise ValueError(f"旋转轴长度过小：{tuple(axis)}")
    ax, ay, az = ax / length, ay / length, az / length

    half = 0.5 * float(angle)
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


def quat_from_euler_xyz(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """由 XYZ 顺序的欧拉角构造四元数。

    旋转顺序：先绕 X 轴 ``rx``，再绕 Y 轴 ``ry``，最后绕 Z 轴 ``rz``
    （即 ``q = qz ⊗ qy ⊗ qx``）。

    这个函数主要是为了**方便手写场景 JSON**：让人可以用直观的角度描述
    物体朝向，而不是手算四元数分量。刚体求解器内部仍应使用四元数。

    参数单位：弧度。
    """
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), rx)
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), ry)
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), rz)
    return quat_normalize(quat_multiply(qz, quat_multiply(qy, qx)))


def quat_from_euler_degrees(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """``quat_from_euler_xyz`` 的角度制版本，方便直接在 JSON 里写角度值。"""
    return quat_from_euler_xyz(math.radians(rx), math.radians(ry), math.radians(rz))


# ---------------------------------------------------------------------------
# 旋转应用
# ---------------------------------------------------------------------------

def quat_rotate_vector(q: Quat, v: Vec3) -> tuple[float, float, float]:
    """用单位四元数旋转向量，返回旋转后的新向量。

    使用无矩阵的等价公式：

        t = 2 * cross(q.xyz, v)
        v' = v + q.w * t + cross(q.xyz, t)

    相比先转 3x3 矩阵再相乘，这个写法运算量更小，也不会因为矩阵
    正交性退化而引入数值误差。
    """
    w, x, y, z = quat_normalize(q)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

    # t = 2 * (q.xyz × v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)

    # v' = v + w * t + q.xyz × t
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_to_matrix(q: Quat) -> tuple[tuple[float, float, float], ...]:
    """把单位四元数转换为 3x3 旋转矩阵（行优先嵌套元组）。

    返回值 ``R`` 满足 ``R @ v`` 等于 ``quat_rotate_vector(q, v)``。

    刚体动力学里用到它的地方主要有两处：
    1. 把局部坐标下的惯量张量旋转到世界坐标：``I_world = R @ I_local @ R.T``
    2. 计算接触点相对质心的力臂向量。
    """
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


# ---------------------------------------------------------------------------
# 姿态积分（W2 刚体求解器使用）
# ---------------------------------------------------------------------------

def quat_integrate(q: Quat, angular_velocity: Vec3, dt: float) -> tuple[float, float, float, float]:
    """按角速度积分一个时间步，返回新的姿态四元数。

    使用一阶近似 ``dq/dt = 0.5 * ω ⊗ q``，离散化为：

        q_new = normalize( q + 0.5 * dt * (0, ω) ⊗ q )

    这里角速度 ``ω`` 必须是**世界坐标系**下的角速度，且增量乘以在左侧。
    积分后必须归一化，否则模长漂移会让旋转逐渐失真。

    参数
    ----
    q                : 当前姿态
    angular_velocity : 世界坐标系下的角速度，单位 rad/s
    dt               : 时间步长，单位秒
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    ox, oy, oz = float(angular_velocity[0]), float(angular_velocity[1]), float(angular_velocity[2])
    half_dt = 0.5 * float(dt)

    # (0, ω) ⊗ q 的展开
    dw = -(ox * x + oy * y + oz * z)
    dx = ox * w + oy * z - oz * y
    dy = oy * w + oz * x - ox * z
    dz = oz * w + ox * y - oy * x

    return quat_normalize((
        w + half_dt * dw,
        x + half_dt * dx,
        y + half_dt * dy,
        z + half_dt * dz,
    ))
