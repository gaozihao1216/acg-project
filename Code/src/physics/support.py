# -*- coding: utf-8 -*-
"""支撑映射（support mapping）：GJK / EPA 唯一的几何输入。

为什么要有这一层
----------------
窄相原先按"几何对"手写函数：球-球、球-盒、盒-盒、盒-凸包、凸包-凸包……
每加一种几何，就要补 O(N) 个函数，再加 O(N) 个 TOI 函数。这条路在
"球 + 盒 + 凸包"时刚好走得通，但柱与胶囊一进来就要补 9 个窄相 + 9 个 TOI，
而且曲面与多面体的组合根本没有干净的 SAT 轴集合可用。

支撑映射把问题一次性收口：**任意凸体都可以用一个函数描述**

    ``support(d) = 该物体上沿方向 d 最远的点``

有了它，GJK 负责"离多远 / 碰没碰"，EPA 负责"陷多深 / 沿哪个方向推出"，
于是"几何对"不再是代码分派维度，而是数据维度——加一种几何只需要写
这一个函数，不需要碰任何碰撞代码。

局部 vs 世界
------------
所有 ``support_local`` 都在**物体自身的主轴系**里工作（球心/质心在原点，
柱与胶囊的轴是 +Z）。``support_world`` 负责先转到局部、算完再转回世界，
这样每种几何的支撑函数都只有几行、且与姿态无关。
"""

from __future__ import annotations

import math

from ..geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    ConvexHullGeometry,
    CylinderGeometry,
    SphereGeometry,
)
from ..mathutils import quaternion as quat
from ..mathutils import vec3

#: 方向向量长度小于它就当作"没有方向"，退回物体自身的中心/轴线端点。
_EPS = 1e-12

#: 支撑特征采样：方向与轴线夹角余弦超过它，认为"正对端面"（约 3°）。
#:
#: 用几度的容差而不是严格的 ``cos == ±1``：立着的柱被求解器微调几毫弧度后，
#: 严格判据会让流形从 4 点塌成 1 点，物体随即开始摇摆。容差内的"近似端面"
#: 取 4 个圆周点，深度逐点按投影算，多出来的点会被自动裁掉，不会引入错误。
_FACE_COS_TOL = 0.9986

#: 同上：余弦落在这个带宽内认为"接近侧面接触"（约 3°），取 2 个母线端点。
_SIDE_COS_TOL = 0.05


def support_local(geometry, direction) -> tuple[float, float, float]:
    """几何体在**局部坐标**下沿 ``direction`` 的最远点。

    实现已经下沉到 ``GeometrySpec.support``：每种几何最清楚自己的支撑点怎么算，
    这里只做一次委托。这样新增一种几何**不需要碰本模块**——"加几何就要改碰撞
    代码"正是支撑映射要消灭的成本。

    参数
    ----
    geometry  : 任一种几何对象
    direction : 搜索方向（不必归一化，长度为零时退回中心附近的确定点）

    返回
    ----
    tuple[float, float, float] : 局部坐标下的支撑点

    异常
    ----
    TypeError : 该几何没有支撑映射（例如组合体，它靠逐部件展开参与碰撞）
    """
    return geometry.support(direction)


def support_world(body, direction) -> tuple[float, float, float]:
    """刚体在世界坐标下沿 ``direction`` 的最远点。"""
    inverse = quat.quat_conjugate(body.orientation)
    local_direction = quat.quat_rotate_vector(inverse, direction)
    local_point = support_local(body.geometry, local_direction)
    world = quat.quat_rotate_vector(body.orientation, local_point)
    return (world[0] + body.position[0], world[1] + body.position[1], world[2] + body.position[2])


def world_axis(body) -> tuple[float, float, float]:
    """柱/胶囊的轴线在世界坐标下的方向（局部 +Z 的像）。"""
    return quat.quat_rotate_vector(body.orientation, (0.0, 0.0, 1.0))


def feature_world(
    body, direction, tol: float = 1e-4
) -> list[tuple[float, float, float]]:
    """沿 ``direction`` 的**支撑特征**：一组"几乎同样远"的表面点。

    为什么需要它
    ------------
    GJK/EPA 只给一个法向和一个深度，对应的接触点是**一个**。对平面-平面接触，
    单点会让物体绕该点自由翻转——盒-地面当初必须返回 4 个点，正是这个原因。
    曲面上同样存在"成片接触"：立着的圆柱底面贴着别的面时，接触区是一整圈；
    躺着的胶囊贴着平面时，是一条线段。

    所以这里按几何形状解析地枚举"朝 ``direction`` 看过去最凸的那一片"：

    - 球：恒定 1 个点
    - 盒 / 凸包：支撑值落在最大值 ``tol`` 内的顶点（面贴面时最多 4 个）
    - 圆柱：正对端面 → 该端面圆周上 4 个点；正对侧面 → 两条母线的端点（2 个）
    - 胶囊：正对侧面 → 两端球心各一点（2 个）

    返回的点都**在表面上**（不是中点），接触点由调用方在两侧之间取中点。
    """
    geometry = body.geometry

    defect = math.sqrt(
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    if defect <= _EPS:
        return [support_world(body, direction)]

    if isinstance(geometry, SphereGeometry):
        return [support_world(body, direction)]

    if isinstance(geometry, BoxGeometry):
        size = geometry.size
        half = (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])
        local_direction = quat.quat_rotate_vector(
            quat.quat_conjugate(body.orientation), direction
        )
        best = (
            abs(local_direction[0]) * half[0]
            + abs(local_direction[1]) * half[1]
            + abs(local_direction[2]) * half[2]
        )
        points = []
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    vertex = (sign_x * half[0], sign_y * half[1], sign_z * half[2])
                    score = (
                        vertex[0] * local_direction[0]
                        + vertex[1] * local_direction[1]
                        + vertex[2] * local_direction[2]
                    )
                    if best - score <= tol:
                        points.append(_local_to_world(body, vertex))
        return points

    if isinstance(geometry, ConvexHullGeometry):
        local_direction = quat.quat_rotate_vector(
            quat.quat_conjugate(body.orientation), direction
        )
        scored = [
            (
                vertex[0] * local_direction[0]
                + vertex[1] * local_direction[1]
                + vertex[2] * local_direction[2],
                vertex,
            )
            for vertex in geometry.vertices
        ]
        best = max(item[0] for item in scored)
        return [
            _local_to_world(body, vertex)
            for score, vertex in scored
            if best - score <= tol
        ]

    axis = world_axis(body)
    cos_axis = (
        direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]
    ) / defect

    if isinstance(geometry, CylinderGeometry):
        radius = geometry.radius
        half_height = 0.5 * geometry.height
        if abs(cos_axis) >= _FACE_COS_TOL:
            # 端面接触：圆周上等角取 4 点，足以撑住"平放不翻"
            center = vec3.madd(body.position, axis, half_height if cos_axis > 0.0 else -half_height)
            e1, e2 = vec3.orthogonal_basis(axis)
            return [
                vec3.add(center, _ring_offset(e1, e2, angle, radius))
                for angle in (
                    math.pi * 0.25,
                    math.pi * 0.75,
                    math.pi * 1.25,
                    math.pi * 1.75,
                )
            ]
        if abs(cos_axis) <= _SIDE_COS_TOL:
            # 侧面接触：两条母线的端点
            radial = _radial_direction(direction, axis)
            return [
                vec3.madd(
                    vec3.madd(body.position, axis, sign * half_height), radial, radius
                )
                for sign in (-1.0, 1.0)
            ]
        return [support_world(body, direction)]

    if isinstance(geometry, CapsuleGeometry):
        radius = geometry.radius
        half_height = 0.5 * geometry.height
        if abs(cos_axis) <= _SIDE_COS_TOL:
            radial = _radial_direction(direction, axis)
            return [
                vec3.madd(
                    vec3.madd(body.position, axis, sign * half_height), radial, radius
                )
                for sign in (-1.0, 1.0)
            ]
        return [support_world(body, direction)]

    raise TypeError(f"不支持支撑特征采样的几何类型：{type(geometry).__name__}")


def _local_to_world(body, point) -> tuple[float, float, float]:
    """局部坐标点 → 世界坐标点。"""
    world = quat.quat_rotate_vector(body.orientation, point)
    return (world[0] + body.position[0], world[1] + body.position[1], world[2] + body.position[2])


def _radial_direction(direction, axis) -> tuple[float, float, float]:
    """方向里垂直于轴线的那一份（单位化）。"""
    projection = direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]
    radial = (
        direction[0] - axis[0] * projection,
        direction[1] - axis[1] * projection,
        direction[2] - axis[2] * projection,
    )
    length = math.sqrt(radial[0] ** 2 + radial[1] ** 2 + radial[2] ** 2)
    if length <= _EPS:
        # 方向与轴线平行时没有径向分量：任取一个正交方向，保证确定性
        e1, _e2 = vec3.orthogonal_basis(axis)
        return e1
    return (radial[0] / length, radial[1] / length, radial[2] / length)


def _ring_offset(e1, e2, angle, radius) -> tuple[float, float, float]:
    """圆周上角度 ``angle`` 处、半径 ``radius`` 的偏移向量。"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        (e1[0] * cos_a + e2[0] * sin_a) * radius,
        (e1[1] * cos_a + e2[1] * sin_a) * radius,
        (e1[2] * cos_a + e2[2] * sin_a) * radius,
    )
