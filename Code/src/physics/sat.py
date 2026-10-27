# -*- coding: utf-8 -*-
"""盒-盒碰撞：分离轴定理（SAT）+ 接触流形裁剪。

为什么盒子比球难一个量级
------------------------
球是「曲率处处相同」的凸体，任意方向只能有 1 个接触点，法向就是质心连线。
盒子是「平面拼接」的凸体：

- **面贴面**：理想情况是整块面积接触，离散后必须给 4 个点才能表达支撑多边形
- **棱碰棱**：只有 1 个点，而且这个点的位置与两个盒子的棱都有关
- **顶点碰面**：1 个点

单个接触点约束不住盒子的旋转，堆叠会持续抖动、慢慢倾倒。所以流形必须是多点的。

三段式算法
----------
1. **SAT 找最小穿透轴**（15 条：A 的 3 个面法向、B 的 3 个面法向、9 个棱叉积）
2. **按轴的类型生成流形**
   - 面轴 → 参考面/入射面裁剪，得到 1–4 个点
   - 棱轴 → 求两条最近棱的最近点，得到 1 个点
3. **记稳定特征 ID**，为暖启动提供跨帧比对依据

为什么对称盒可以用 ``|d·n|``
----------------------------
盒子关于自身中心对称，因此沿轴 ``n`` 的分离量只需考虑轴的**方向线**而非朝向：

    separation = |dot(n, c_B − c_A)| − (r_A(n) + r_B(n))

其中 ``r(n) = Σ|dot(axis_i, n)|·h_i`` 是盒子在该轴上的投影半径。
只要有一条轴的 separation > 0，两盒必然分离。

法向约定
--------
本模块返回的 ``ContactPoint.normal`` 一律**从 A 指向 B**（与全局约定一致）。
参考面在 B 上时，内部计算用的是「从 B 指向 A」的朝向，最后统一翻回。
"""

from __future__ import annotations

import math

from ..mathutils import quaternion as quat
from ..mathutils import vec3
from .body import RigidBody
from .contact import ContactPoint

#: 两个棱方向近似平行时，叉积轴退化，跳过该轴
_PARALLEL_EPS = 1e-6

#: 一个流形最多保留的接触点数
MAX_MANIFOLD_POINTS = 4

#: 特征 ID 的编码基数：face_id * _FEATURE_BASE + vertex_id
_FEATURE_BASE = 100


# ---------------------------------------------------------------------------
# 几何辅助
# ---------------------------------------------------------------------------

def box_axes(body: RigidBody) -> list[tuple[float, float, float]]:
    """盒子的三个局部轴在世界坐标下的方向（单位向量）。"""
    orientation = body.orientation
    return [
        quat.quat_rotate_vector(orientation, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]


def box_half_extents(body: RigidBody) -> tuple[float, float, float]:
    """半尺寸（``size`` 存的是全长）。"""
    size = body.geometry.size
    return (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])


def _project_radius(axes, half, direction) -> float:
    """盒子在给定方向上的投影半径 ``Σ|dot(axis_i, n)|·h_i``。

    这个量是 SAT 的核心：它描述盒子在该方向"伸出多远"，
    对任意朝向都成立，不需要枚举 8 个顶点。
    """
    return (
        abs(vec3.dot(axes[0], direction)) * half[0]
        + abs(vec3.dot(axes[1], direction)) * half[1]
        + abs(vec3.dot(axes[2], direction)) * half[2]
    )


def _face_vertices(center, axes, half, axis_index: int, sign: float):
    """返回某个面的 4 个顶点（世界坐标）与该面的两个切向轴序号。

    **顶点必须按环序排列**，否则多边形是自交的"蝴蝶形"，裁剪会得出错误结果。
    正确的环绕顺序是四边形的一圈：

        (-1,-1) → (+1,-1) → (+1,+1) → (-1,+1)

    而不是按 ``for su: for sv:`` 自然产生的 Z 字序
    ``(-,-) → (-,+) → (+,-) → (+,+)``——后者两条对角边会交叉。
    这个坑实测过一次：平放堆叠本该得到 4 个接触点，结果只得到 2 个。
    """
    face_center = vec3.madd(center, axes[axis_index], sign * half[axis_index])
    tangent = [i for i in range(3) if i != axis_index]

    ring = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))

    vertices = []
    for su, sv in ring:
        point = vec3.madd(face_center, axes[tangent[0]], su * half[tangent[0]])
        point = vec3.madd(point, axes[tangent[1]], sv * half[tangent[1]])
        vertices.append(point)
    return vertices, tangent, face_center


def _clip_polygon(polygon, normal, offset, counter):
    """用半空间 ``dot(normal, p) <= offset`` 裁剪凸多边形。

    多边形顶点带整数标签（``(position, vid)``），标签会传下去：
    原来的顶点保持自己的标签，被切出来的新顶点拿一个新标签。
    这样"哪些点是同一个点"在相邻帧之间可以比对——暖启动要用。

    参数
    ----
    polygon : list[(position, vid)]
    normal  : 裁剪平面法向
    offset  : 平面偏移
    counter : 单元素列表 ``[next_vid]``，用于分配新标签
    """
    if not polygon:
        return []

    output = []
    count = len(polygon)

    for index in range(count):
        current, current_id = polygon[index]
        following, following_id = polygon[(index + 1) % count]

        d_current = vec3.dot(normal, current) - offset
        d_following = vec3.dot(normal, following) - offset

        if d_current <= 0.0:
            output.append((current, current_id))

        # 跨越平面：插入交点
        if (d_current < 0.0 < d_following) or (d_following < 0.0 < d_current):
            t = d_current / (d_current - d_following)
            delta = vec3.sub(following, current)
            point = vec3.madd(current, delta, t)
            counter[0] += 1
            output.append((point, counter[0]))

    return output


def _closest_points_on_segments(p1, d1, half1, p2, d2, half2):
    """两条线段的最近点（参数化求解后夹紧到线段范围内）。

    线段 1：``p1 + s·d1``，``s ∈ [−half1, +half1]``
    线段 2：``p2 + t·d2``，``t ∈ [−half2, +half2]``
    """
    r = vec3.sub(p1, p2)
    a = vec3.dot(d1, d1)
    e = vec3.dot(d2, d2)
    f = vec3.dot(d2, r)
    c = vec3.dot(d1, r)
    b = vec3.dot(d1, d2)
    denominator = a * e - b * b

    if denominator > 1e-12:
        s = (b * f - c * e) / denominator
    else:
        # 两条棱平行（理论上 SAT 已跳过该轴，这里兜底）
        s = 0.0

    s = max(-half1, min(half1, s))

    # 固定 s 后求最优 t
    t = (b * s + f) / e if e > 1e-12 else 0.0
    t = max(-half2, min(half2, t))

    # t 被夹紧后重新求 s
    s = (b * t - c) / a if a > 1e-12 else 0.0
    s = max(-half1, min(half1, s))

    point1 = vec3.madd(p1, d1, s)
    point2 = vec3.madd(p2, d2, t)
    return point1, point2


def _support_edge(center, axes, half, direction, edge_axis: int):
    """返回沿 ``edge_axis`` 方向、且在 ``direction`` 上最靠前的那条棱。

    棱的其余两个分量按与 ``direction`` 的点积符号取正负，
    因此得到的是"最靠近对方盒子"的那条棱。
    """
    edge_center = center
    for axis_index in range(3):
        if axis_index == edge_axis:
            continue
        sign = 1.0 if vec3.dot(axes[axis_index], direction) >= 0.0 else -1.0
        edge_center = vec3.madd(edge_center, axes[axis_index], sign * half[axis_index])
    return edge_center, axes[edge_axis], half[edge_axis]


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def collide_box_box(body_a: RigidBody, body_b: RigidBody) -> list[ContactPoint]:
    """两个有向包围盒之间的碰撞检测。

    返回
    ----
    list[ContactPoint]
        法向从 A 指向 B。未接触时返回空列表（而不是 None），
        因为面接触天然是 0–4 个点。
    """
    center_a = body_a.position
    center_b = body_b.position
    axes_a = box_axes(body_a)
    axes_b = box_axes(body_b)
    half_a = box_half_extents(body_a)
    half_b = box_half_extents(body_b)

    delta = vec3.sub(center_b, center_a)

    # --- SAT：找最小穿透轴 ---
    best_separation = -float("inf")
    best_kind = None      # 'face_a' / 'face_b' / 'edge'
    best_i = -1
    best_j = -1
    best_normal = (0.0, 0.0, 0.0)

    def consider(normal, kind, i, j):
        """评估一条候选轴，更新最优记录。返回该轴的分离量。"""
        nonlocal best_separation, best_kind, best_i, best_j, best_normal
        radius = _project_radius(axes_a, half_a, normal) + _project_radius(
            axes_b, half_b, normal
        )
        # 盒子中心对称，故用 |d·n|；同时把法向朝向定为「A 指向 B」
        projection = vec3.dot(delta, normal)
        separation = abs(projection) - radius
        oriented = normal if projection >= 0.0 else vec3.scale(normal, -1.0)
        if separation > best_separation:
            best_separation = separation
            best_kind = kind
            best_i = i
            best_j = j
            best_normal = oriented
        return separation

    # A 的 3 个面法向
    for i in range(3):
        if consider(axes_a[i], "face_a", i, -1) > 0.0:
            return []

    # B 的 3 个面法向
    for j in range(3):
        if consider(axes_b[j], "face_b", -1, j) > 0.0:
            return []

    # 9 条棱叉积轴
    for i in range(3):
        for j in range(3):
            cross = vec3.cross(axes_a[i], axes_b[j])
            length = vec3.length(cross)
            if length < _PARALLEL_EPS:
                continue
            if consider(vec3.scale(cross, 1.0 / length), "edge", i, j) > 0.0:
                return []

    if best_kind is None:
        return []

    penetration_total = -best_separation

    # --- 棱-棱：单点接触 ---
    if best_kind == "edge":
        return _edge_contact(
            center_a, axes_a, half_a, center_b, axes_b, half_b,
            best_i, best_j, best_normal, penetration_total,
        )

    # --- 面接触：参考面 / 入射面裁剪 ---
    return _face_contacts(
        center_a, axes_a, half_a, center_b, axes_b, half_b,
        best_kind, best_i, best_j, best_normal,
    )


def _edge_contact(
    center_a, axes_a, half_a, center_b, axes_b, half_b,
    edge_i: int, edge_j: int, normal, penetration: float,
) -> list[ContactPoint]:
    """棱-棱接触：取两条最近棱上最近点的中点。"""
    # n 从 A 指向 B：A 上最靠 B 的棱、B 上最靠 A 的棱
    point_a, direction_a, reach_a = _support_edge(
        center_a, axes_a, half_a, normal, edge_i
    )
    reversed_normal = vec3.scale(normal, -1.0)
    point_b, direction_b, reach_b = _support_edge(
        center_b, axes_b, half_b, reversed_normal, edge_j
    )

    closest_a, closest_b = _closest_points_on_segments(
        point_a, direction_a, reach_a, point_b, direction_b, reach_b
    )
    middle = vec3.scale(vec3.add(closest_a, closest_b), 0.5)

    feature_id = (20 + edge_i * 3 + edge_j) * _FEATURE_BASE
    return [
        ContactPoint(
            position=middle,
            normal=normal,
            penetration=max(0.0, penetration),
            feature_id=feature_id,
        )
    ]


def _face_contacts(
    center_a, axes_a, half_a, center_b, axes_b, half_b,
    kind: str, axis_i: int, axis_j: int, normal_ab,
) -> list[ContactPoint]:
    """面接触：把入射面裁剪到参考面范围内，收集穿透点。"""
    if kind == "face_a":
        ref_center, ref_axes, ref_half = center_a, axes_a, half_a
        ref_axis = axis_i
        ref_normal = normal_ab                    # A 的面朝外，指向 B
        inc_center, inc_axes, inc_half = center_b, axes_b, half_b
        ref_is_a = True
    else:
        ref_center, ref_axes, ref_half = center_b, axes_b, half_b
        ref_axis = axis_j
        ref_normal = vec3.scale(normal_ab, -1.0)  # B 的面朝外，指向 A
        inc_center, inc_axes, inc_half = center_a, axes_a, half_a
        ref_is_a = False

    ref_sign = 1.0 if vec3.dot(ref_axes[ref_axis], ref_normal) >= 0.0 else -1.0
    _, ref_tangents, ref_face_center = _face_vertices(
        ref_center, ref_axes, ref_half, ref_axis, ref_sign
    )

    # 入射面：法向与参考面法向**最反向**的那个面。
    #
    # 注意每个轴有正负两个面，所以必须比较 |dot| 再定符号。
    # 第一版只比较了正方向（`-dot(axis, ref_normal)` 取最大），
    # 结果对 +z 参考法向选中的是 x 轴（点积 0）而不是 -z 面（点积 -1）：
    # 入射面整个选错，裁剪出的是侧面，平放堆叠只拿到 2 个点。
    best_axis = 0
    best_alignment = -1.0
    for index in range(3):
        alignment = abs(vec3.dot(inc_axes[index], ref_normal))
        if alignment > best_alignment:
            best_alignment = alignment
            best_axis = index

    # 取与参考法向反向的那一侧，使入射面正对参考面
    inc_sign = -1.0 if vec3.dot(inc_axes[best_axis], ref_normal) >= 0.0 else 1.0

    inc_vertices, _, _ = _face_vertices(
        inc_center, inc_axes, inc_half, best_axis, inc_sign
    )

    # 用参考面的 4 个侧平面裁剪入射面
    polygon = [(vertex, index) for index, vertex in enumerate(inc_vertices)]
    counter = [len(inc_vertices)]
    for tangent_index in ref_tangents:
        for sign in (1.0, -1.0):
            plane_normal = vec3.scale(ref_axes[tangent_index], sign)
            offset = vec3.dot(plane_normal, ref_face_center) + ref_half[tangent_index]
            polygon = _clip_polygon(polygon, plane_normal, offset, counter)
            if not polygon:
                return []

    # 收集穿透参考面的点
    face_id = (0 if ref_is_a else 3) + ref_axis
    contacts: list[ContactPoint] = []

    for vertex, vertex_id in polygon:
        # 相对参考面的有向距离：负值表示陷入
        separation = vec3.dot(ref_normal, vec3.sub(vertex, ref_face_center))
        if separation > 0.0:
            continue

        # 接触点取「入射顶点」与「它在参考面上的投影」的中点
        position = vec3.madd(vertex, ref_normal, -0.5 * separation)
        contacts.append(
            ContactPoint(
                position=position,
                normal=normal_ab,
                penetration=-separation,
                feature_id=face_id * _FEATURE_BASE + vertex_id,
            )
        )

    if len(contacts) > MAX_MANIFOLD_POINTS:
        contacts.sort(key=lambda p: p.penetration, reverse=True)
        del contacts[MAX_MANIFOLD_POINTS:]

    return contacts
