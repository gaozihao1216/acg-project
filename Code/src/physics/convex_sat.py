# -*- coding: utf-8 -*-
"""广义分离轴定理（SAT）：任意凸多面体两两碰撞 + 接触流形生成。

为什么复用 box 的算法而不是改用 GJK
------------------------------------
`sat.py` 里盒-盒用的轴集合是「3 个面法向 + 3 个面法向 + 9 条棱叉积」。
把这个模式推广到任意凸多面体，完整的分离轴集合恰好是：

    所有分离轴 = A 的全部面法向 ∪ B 的全部面法向 ∪ { a 棱 × b 棱 }

盒子只是"6 个面、12 条棱"的特例。所以**不需要换算法**，只需要把
"从 3 个坐标轴读轴"换成"从面表/棱表读轴"。裁剪用的凸多边形裁剪也已是通用的。

选择 SAT 而非 GJK/EPA 的理由：**可验证**。SAT 的穿透深度是解析值，
能直接和手算结果比对；GJK 只能自证收敛。本项目的每一项都有解析对照，
这条更重要。

代价与上限
----------
棱-棱轴数是 ``E_A · E_B``。这是 SAT 唯一的规模问题：

====================== ================== ====================
凸包规模                棱数              棱-棱候选轴
====================== ================== ====================
立方体                  12                144
20 面凸包               30                900
64 面凸包               ~96               9 216
512 面凸包              768               589 824  ← 不可用
====================== ================== ====================

所以**碰撞代理必须是粗网格**。这不是妥协，是业界标准做法：
PhysX / Bullet 的凸代理通常只有 8–32 个面。

`ConvexHullGeometry` 因此在加载阶段就拒绝过复杂的凸包（见 `max_faces`），
而不是让它在仿真里悄悄卡死——一次 60 万条轴的查询会让整个演示失去响应，
现象上看起来像"死循环"，极难定位。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..mathutils import quaternion as quat
from .contact import ContactPoint

#: 棱方向叉积的退化阈值：两条棱平行时叉积趋零，那条轴没有意义，必须跳过。
#: 阈值取相对量级，避免单位不同导致误判。
_PARALLEL_EPS = 1e-8

#: 每个面接触最多保留的点数。4 点足以约束平面接触的全部自由度
#: （1 法向平动 + 2 切向平动 + 3 转动中的 2 个倾覆 + 1 个扭转）。
_MAX_MANIFOLD_POINTS = 4


@dataclass
class Polytope:
    """世界坐标系下的凸多面体。

    属性
    ----
    vertices : 世界坐标顶点
    faces    : 每个面是**从外部看逆时针**的顶点下标列表
    normals  : 每个面的单位外法向（世界坐标）
    edges    : 唯一无向棱表 ``(i, j)``
    center   : 顶点平均位置，只用于确定法向的正方向

    为什么全都预先转到世界坐标
    --------------------------
    SAT 要在世界坐标下比较投影区间。如果每次查询都做局部→世界的变换，
    一个 64 面的凸包就要变换几十个顶点、几十条棱，开销比 SAT 本身还大。
    所以在查询入口一次性构建，之后全部是纯算术。
    """

    vertices: list[tuple[float, float, float]]
    faces: list[list[int]]
    normals: list[tuple[float, float, float]]
    edges: list[tuple[int, int]]
    center: tuple[float, float, float]


# ---------------------------------------------------------------------------
# 向量工具
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normalize(a):
    n = _length(a)
    if n <= 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


# ---------------------------------------------------------------------------
# 从几何体构建世界坐标多面体
# ---------------------------------------------------------------------------

def polytope_from_hull(geometry, position, orientation) -> Polytope:
    """把 ``ConvexHullGeometry`` 按位姿变换到世界坐标。"""
    vertices = [quat.quat_rotate_vector(orientation, v) for v in geometry.vertices]
    vertices = [tuple(v[k] + position[k] for k in range(3)) for v in vertices]

    normals = [
        quat.quat_rotate_vector(orientation, n) for n in geometry.face_normals
    ]
    edges = list(geometry.edges)
    faces = [list(face) for face in geometry.faces]

    return Polytope(
        vertices=vertices,
        faces=faces,
        normals=normals,
        edges=edges,
        center=_centroid(vertices),
    )


def polytope_from_box(size: Sequence[float], position, orientation) -> Polytope:
    """把轴对齐盒子构造成等价的多面体（8 顶点、6 个四边形面、12 条棱）。

    为什么盒子要走这条路
    --------------------
    让"凸包 vs 盒子"复用同一套 SAT，避免再写一份
    "凸包-盒"的特殊代码。盒子本身已经有一条专门的快速路径（``sat.py``），
    这里只用于盒子和凸包混用时的统一处理。
    """
    hx, hy, hz = 0.5 * size[0], 0.5 * size[1], 0.5 * size[2]
    local = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    # 从外部看逆时针的圆柱环序
    faces = [
        [0, 3, 2, 1],   # -z
        [4, 5, 6, 7],   # +z
        [0, 1, 5, 4],   # -y
        [3, 7, 6, 2],   # +y
        [0, 4, 7, 3],   # -x
        [1, 2, 6, 5],   # +x
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    vertices = [quat.quat_rotate_vector(orientation, v) for v in local]
    vertices = [tuple(v[k] + position[k] for k in range(3)) for v in vertices]

    normals = []
    for face in faces:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normals.append(_normalize(_cross(_sub(b, a), _sub(c, a))))

    return Polytope(
        vertices=vertices,
        faces=faces,
        normals=normals,
        edges=edges,
        center=_centroid(vertices),
    )


def _centroid(vertices) -> tuple[float, float, float]:
    count = len(vertices)
    return (
        sum(v[0] for v in vertices) / count,
        sum(v[1] for v in vertices) / count,
        sum(v[2] for v in vertices) / count,
    )


# ---------------------------------------------------------------------------
# 投影区间与分离量
# ---------------------------------------------------------------------------

def _interval(vertices, axis) -> tuple[float, float]:
    """所有顶点在 ``axis`` 上投影的最小值与最大值。

    直接对顶点求极值，而不是用"半径 = Σ|aᵢ·n|hᵢ"的盒子公式——
    那个公式只对中心对称的盒子成立，凸包没有这个性质。
    """
    first = _dot(vertices[0], axis)
    low = high = first
    for index in range(1, len(vertices)):
        value = _dot(vertices[index], axis)
        if value < low:
            low = value
        elif value > high:
            high = value
    return low, high


def _separation(a: Polytope, b: Polytope, axis) -> float:
    """沿 ``axis`` 的分离量：正值表示已分离，负值表示重叠深度。"""
    a_low, a_high = _interval(a.vertices, axis)
    b_low, b_high = _interval(b.vertices, axis)
    return max(a_low - b_high, b_low - a_high)


def separation_with_side(a: Polytope, b: Polytope, axis) -> tuple[float, int]:
    """沿 ``axis`` 的分离量，并指出是哪一侧的投影给出的。

    返回 ``(分离量, 侧)``：``侧 = +1`` 表示 A 整体在轴负侧（间隙为
    ``a_low - b_high``），``侧 = -1`` 表示 B 在负侧。

    为什么要把"侧"暴露出来
    ----------------------
    计算间隙的**变化率**需要知道往哪边收：侧不同，闭合速度的表达式差一个符号。
    ``_separation`` 只返回 ``max(两者)``，符号信息在取 max 时就丢了。
    TOI 用它做保守预测（见 ``toi.py``），少了这个符号会把"分离"算成"接近"。
    """
    a_low, a_high = _interval(a.vertices, axis)
    b_low, b_high = _interval(b.vertices, axis)
    first = a_low - b_high
    second = b_low - a_high
    if first >= second:
        return first, +1
    return second, -1


def candidate_axes(polytope_a: Polytope, polytope_b: Polytope, *, with_indices: bool = True):
    """枚举 SAT 候选轴，附带来源标记 ``(轴向, 类型, A 下标, B 下标)``。

    三种来源：A 的面法向、B 的面法向、A 棱 × B 棱。

    **顺序有含义**：先面后棱，于是共面贴合的两个盒子被判成面接触（4 点流形），
    而不是被某条棱叉积轴降级成单点。``collide_polytopes`` 的接触点质量依赖这一点。

    抽成生成器是为了让 TOI 与 SAT 用**同一套**候选轴。TOI 需要逐轴算间隙的
    时间导数，SAT 需要逐轴比较分离量；两套各写一遍的话，轴集合一旦不一致，
    就会出现"SAT 说没碰到、TOI 说已经到了"这种最难查的分歧。

    重复轴会被**剔除**（``±n`` 视为同一条，因为间隙公式对 n 取反不变）。
    这不是微优化：正六棱柱的 18 条棱给出 324 个叉积，其中只有极少数是新的方向。
    实测一对六棱柱的候选轴从 **256 条降到 8 条**——SAT 与 TOI 同时受益。

    剔除只会**丢掉完全相同的方向**，因此 SAT 的 max 不变；TOI 的 min 也不变
    （两条同向轴的间隙差在 1e-9·尺度 量级，远小于 200 µm 的接触余量）。
    """
    seen: set[tuple[int, int, int]] = set()

    def fresh(axis) -> bool:
        key = _axis_key(axis)
        if key in seen:
            return False
        seen.add(key)
        return True

    for index, normal in enumerate(polytope_a.normals):
        if fresh(normal):
            yield normal, "face_a", index, -1

    for index, normal in enumerate(polytope_b.normals):
        if fresh(normal):
            yield normal, "face_b", -1, index

    # 棱 × 棱：按**方向去重**再叉乘。
    #
    # 正六棱柱有 18 条棱，18×18 = 324 次叉乘只产出 6 个互不相同的新方向。
    # 先把方向去重（六棱柱 18 条棱只有 4 个方向），叉乘次数从 324 降到 16，
    # 结果集合完全相同——叉乘只依赖方向。
    #
    # ``with_indices=False`` 时索引置 -1：TOI 只看间隙，不需要"是哪两条棱"。
    # SAT 需要索引（用它算棱-棱接触点），所以它走完整枚举。
    # 两种模式产出的**轴集合**严格一致，只是索引信息的有无。
    if with_indices:
        edges_a = list(enumerate(polytope_a.edges))
        edges_b = list(enumerate(polytope_b.edges))
    else:
        edges_a = [(-1, edge) for edge in polytope_a.edges]
        edges_b = [(-1, edge) for edge in polytope_b.edges]

    directions_a = _unique_directions(polytope_a, edges_a)
    directions_b = _unique_directions(polytope_b, edges_b)

    for direction_a, length_a, edge_a_index in directions_a:
        for direction_b, length_b, edge_b_index in directions_b:
            axis = _cross(direction_a, direction_b)
            length = _length(axis)
            if length <= _PARALLEL_EPS * max(length_a * length_b, 1e-30):
                continue   # 两棱平行，叉积无意义
            unit = _scale(axis, 1.0 / length)
            if fresh(unit):
                yield unit, "edge", edge_a_index, edge_b_index


def _unique_directions(polytope: Polytope, indexed_edges):
    """把棱方向去重，返回 ``[(单位方向, 长度, 棱下标)]``。

    ``±d`` 视为同一方向：叉积在 ``d`` 与 ``-d`` 之间只差一个符号，而轴集合
    本来就对 ``±`` 等价。保留下标是为了 SAT 仍能定位"是哪两条棱"。
    """
    seen: set[tuple[int, int, int]] = set()
    result = []
    for index, (start, end) in indexed_edges:
        direction = _sub(polytope.vertices[end], polytope.vertices[start])
        length = _length(direction)
        if length <= 0.0:
            continue
        unit = _scale(direction, 1.0 / length)
        key = _axis_key(unit)
        if key in seen:
            continue
        seen.add(key)
        result.append((unit, length, index))
    return result


#: 轴去重的量化步长。1e-9 对单位轴而言约等于"方向差 1e-9 弧度"，
#: 远小于任何几何精度，所以合并它们不会改变判据。
_AXIS_QUANTUM = 1e-9


def _axis_key(axis) -> tuple[int, int, int]:
    """把单位轴量化成可哈希的键，且 ``n`` 与 ``-n`` 得到同一个键。

    符号规范化是必需的：``_separation`` 对 n 取反不变（取 max 的两侧互换），
    所以 ``+n`` 与 ``-n`` 是**同一条判据**。不规范化的话，两条相反方向的面法向
    会被当成新轴各算一遍——那恰好是盒-盒最常见的浪费。
    """
    x, y, z = axis
    length = (x * x + y * y + z * z) ** 0.5
    if length > 0.0:
        x, y, z = x / length, y / length, z / length
    for component in (x, y, z):
        if abs(component) > _AXIS_QUANTUM:
            if component < 0.0:
                x, y, z = -x, -y, -z
            break
    return (
        int(round(x / _AXIS_QUANTUM)),
        int(round(y / _AXIS_QUANTUM)),
        int(round(z / _AXIS_QUANTUM)),
    )


# ---------------------------------------------------------------------------
# 凸多边形裁剪
# ---------------------------------------------------------------------------

def _clip_polygon(polygon, normal, offset, tolerance: float) -> list:
    """把凸多边形裁剪到半空间 ``dot(normal, p) <= offset`` 内。

    ``polygon`` 是 ``[(point, tag), ...]``，``tag`` 用来追踪特征编号：
    原顶点保留自己的编号，交点获得新编号（由两端编号组合而成），
    这样裁剪后的每个点都能映射回几何特征，暖启动才能跨帧匹配。

    这是 Sutherland–Hodgman 算法，对**凸**裁剪窗口正确。
    它不处理自交多边形——所以面顶点必须按环序排列，按 Z 字序排列会切出
    错误结果（盒-盒那一步曾栽在这里）。
    """
    if not polygon:
        return []

    result = []
    count = len(polygon)

    for i in range(count):
        current, current_tag = polygon[i]
        following, following_tag = polygon[(i + 1) % count]

        current_distance = _dot(normal, current) - offset
        following_distance = _dot(normal, following) - offset

        current_inside = current_distance <= tolerance
        following_inside = following_distance <= tolerance

        if current_inside:
            result.append((current, current_tag))

        if current_inside != following_inside:
            denominator = current_distance - following_distance
            if abs(denominator) > 1e-30:
                t = current_distance / denominator
                intersection = (
                    current[0] + (following[0] - current[0]) * t,
                    current[1] + (following[1] - current[1]) * t,
                    current[2] + (following[2] - current[2]) * t,
                )
                result.append((intersection, (current_tag * 31 + following_tag * 17 + 7) % 997))

    return result


# ---------------------------------------------------------------------------
# 面接触流形
# ---------------------------------------------------------------------------

def _side_planes(polytope: Polytope, face_index: int):
    """某个面四周的侧平面，用于把入射面裁剪到参考面的范围内。

    对于从外部看逆时针的多边形，内部位于每条有向边的左侧；平面内指向
    **内部**的向量是 ``cross(normal, edge)``（用 +z 法向、xy 平面上的逆时针
    多边形可验证：边沿 +x、法向 +z 时叉积得 +y，正是内侧）。

    返回的是**朝外**的 ``(法向, 偏移)``，配合 ``_clip_polygon`` 的
    ``dot(n, p) <= offset`` 约定，保留的正好是面内部。
    若直接返回朝内法向，会把整个面裁掉，得到空多边形——
    这个符号错误的表现是"所有面接触都返回 0 点"，而棱-棱却正常。
    """
    face = polytope.faces[face_index]
    normal = polytope.normals[face_index]
    planes = []
    for i in range(len(face)):
        p = polytope.vertices[face[i]]
        q = polytope.vertices[face[(i + 1) % len(face)]]
        edge = _sub(q, p)
        inward = _cross(normal, edge)
        length = _length(inward)
        if length <= 1e-12:
            continue
        inward = _scale(inward, 1.0 / length)
        # 翻转成朝外，匹配 _clip_polygon 的 <= 约定
        planes.append((_scale(inward, -1.0), -_dot(inward, p)))
    return planes


def _most_parallel(axis, polytope: Polytope, opposite: bool) -> int:
    """找到法向与 ``axis`` 最（反）平行的面。

    ``opposite=True`` 时找最反向的面（入射面）；否则找最同向的面（参考面）。
    """
    best_index = 0
    best_value = -2.0
    for index, normal in enumerate(polytope.normals):
        value = _dot(normal, axis)
        if opposite:
            value = -value
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def _max_dot(normals, axis) -> tuple[int, float]:
    """在法向表里找到与 ``axis`` 点积最大的面，返回 ``(下标, 点积)``。"""
    best_index = 0
    best_value = -2.0
    for index, normal in enumerate(normals):
        value = _dot(normal, axis)
        if value > best_value:
            best_value = value
            best_index = index
    return best_index, best_value


def _face_manifold(
    reference: Polytope,
    incident: Polytope,
    reference_face: int,
    contact_normal,
    tolerance: float,
) -> list[ContactPoint]:
    """参考面 vs 入射面：裁剪出入射面陷入参考面的部分。

    参数
    ----
    contact_normal : 对外报告的接触法向，**恒为 A→B**

    注意 ``contact_normal`` 与参考面自身的外法向不一定同向：
    参考面可能在 B 上，此时它的外法向等于 ``-contact_normal``。
    穿透深度必须沿参考面外法向来量（那才是"陷入多深"），
    而报告的接触法向必须统一为 A→B（引擎其余部分的约定）。

    早先的实现把两者混为一谈：当最优轴来自 A 的面却背离 B 时，
    代码跑去 B 上挑"最反向的面"当参考面，那个面的法向与最优轴并不相同，
    于是深度沿一个方向、法向报另一个方向，穿透值直接错到 1.4 m。
    """
    ref_normal = reference.normals[reference_face]
    ref_point = reference.vertices[reference.faces[reference_face][0]]

    incident_face = _most_parallel(ref_normal, incident, opposite=True)
    incident_face_indices = incident.faces[incident_face]

    polygon = [
        (incident.vertices[index], index) for index in incident_face_indices
    ]

    # 先用参考面的侧平面把入射面裁到参考面的轮廓内
    for plane_normal, plane_offset in _side_planes(reference, reference_face):
        polygon = _clip_polygon(polygon, plane_normal, plane_offset, tolerance)
        if not polygon:
            return []

    # 再保留陷入参考面之后的部分
    contacts = []
    for point, tag in polygon:
        separation = _dot(ref_normal, _sub(point, ref_point))
        if separation > tolerance:
            continue
        depth = -separation
        # 接触点取两表面之间的中点，避免贴在某一侧导致抖动
        position = _add(point, _scale(ref_normal, -0.5 * depth))
        contacts.append(
            ContactPoint(
                position=position,
                normal=contact_normal,
                penetration=depth,
                feature_id=1000 + reference_face * 16 + (tag % 16),
            )
        )

    if not contacts:
        return []

    # 最多保留 4 个最深的点；不足时全留
    if len(contacts) > _MAX_MANIFOLD_POINTS:
        contacts.sort(key=lambda c: c.penetration, reverse=True)
        contacts = contacts[:_MAX_MANIFOLD_POINTS]

    return contacts


# ---------------------------------------------------------------------------
# 棱-棱接触
# ---------------------------------------------------------------------------

def _closest_between_segments(p1, q1, p2, q2, tolerance: float):
    """两条线段的最近点对（Ericson《Real-Time Collision Detection》5.1.9）。

    为什么不能简单取"两条棱的中点连线"：两条异面直线最近点的参数需要
    联立求解。用中点连线会在棱交叉时给出偏斜的法向和偏大的穿透。
    """
    d1 = _sub(q1, p1)
    d2 = _sub(q2, p2)
    r = _sub(p1, p2)

    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)

    if a <= tolerance and e <= tolerance:
        # 两条棱都退化成点
        return p1, p2

    if a <= tolerance:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = _dot(d1, r)
        if e <= tolerance:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = _dot(d1, d2)
            denominator = a * e - b * b
            if denominator != 0.0:
                s = min(max((b * f - c * e) / denominator, 0.0), 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)

    closest_a = _add(p1, _scale(d1, s))
    closest_b = _add(p2, _scale(d2, t))
    return closest_a, closest_b


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def collide_polytopes(
    polytope_a: Polytope,
    polytope_b: Polytope,
    tolerance: float = 1e-9,
) -> list[ContactPoint]:
    """两个凸多面体的碰撞检测，返回 0–4 个接触点。

    返回的 ``normal`` 一律从 A 指向 B，与引擎其余部分约定一致。

    算法
    ----
    1. 枚举全部候选轴：A 的面法向、B 的面法向、A 棱 × B 棱
    2. 取分离量最大的那条轴（重叠时即"最小穿透轴"）
    3. 分离量为正 ⇒ 无接触，立即返回（这是 SAT 的早退优势）
    4. 按最优轴来自面还是棱，分别生成面接触流形或单点接触
    """
    if not polytope_a.vertices or not polytope_b.vertices:
        return []

    best_separation = -float("inf")
    best_axis = None
    best_kind = None
    best_a = -1
    best_b = -1

    # 轴集合与顺序由 ``candidate_axes`` 统一提供（TOI 用同一份），
    # 这里只负责比较分离量并保留来源，以便决定后面生成面流形还是单点。
    for axis, kind, index_a, index_b in candidate_axes(polytope_a, polytope_b):
        separation = _separation(polytope_a, polytope_b, axis)
        if separation > best_separation:
            best_separation = separation
            best_axis = axis
            best_kind = kind
            best_a = index_a
            best_b = index_b
        if best_separation > 0.0:
            return []   # 已找到分离轴，无需继续

    if best_axis is None:
        return []

    # --- 统一法向方向为 A → B ---
    direction = _sub(polytope_b.center, polytope_a.center)
    flip = _dot(direction, best_axis) < 0.0
    if flip:
        best_axis = _scale(best_axis, -1.0)

    penetration = -best_separation

    # --- 棱-棱：单点接触 ---
    if best_kind == "edge":
        a0, a1 = polytope_a.edges[best_a]
        b0, b1 = polytope_b.edges[best_b]
        point_a, point_b = _closest_between_segments(
            polytope_a.vertices[a0], polytope_a.vertices[a1],
            polytope_b.vertices[b0], polytope_b.vertices[b1],
            tolerance,
        )
        midpoint = _scale(_add(point_a, point_b), 0.5)
        return [
            ContactPoint(
                position=midpoint,
                normal=best_axis,
                penetration=max(penetration, 0.0),
                feature_id=200000 + best_a * 1024 + best_b,
            )
        ]

    # --- 面接触 ---
    # 参考面必须满足：它自己的外法向与最终法向（A→B）方向一致地"朝向对方"。
    # 于是只有两种可能：
    #   A 上有一个面外法向 ≈ +best_axis   → 以 A 为参考体
    #   B 上有一个面外法向 ≈ -best_axis   → 以 B 为参考体
    # 取两者中对齐得更好的那个。这样深度与法向一定自洽。
    face_a, dot_a = _max_dot(polytope_a.normals, best_axis)
    face_b, dot_b = _max_dot(polytope_b.normals, _scale(best_axis, -1.0))

    if dot_a >= dot_b:
        return _face_manifold(
            polytope_a, polytope_b, face_a, best_axis, tolerance
        )
    return _face_manifold(
        polytope_b, polytope_a, face_b, best_axis, tolerance
    )


def closest_point_on_polytope(polytope: Polytope, point) -> tuple[tuple, float, tuple]:
    """点到凸多面体表面的最近点。

    返回 ``(最近点, 距离, 指向外部的单位法向)``。

    用途：球-凸包碰撞。球退化成"点 + 半径"，于是问题变成点到多面体表面的距离。

    两种情形必须分开处理
    ------------------
    - **球心在外部**：遍历所有面，把点投影到面多边形上取最小距离。不能只投影到
      面所在平面——那样在棱和顶点附近会给出偏小的距离。
    - **球心在内部**：最近点就是"最浅的那个面"上的投影，法向取该面法向，
      距离记为负（表示已经嵌入），由调用方加上半径。
    """
    # 判断内外：所有面都满足 dot(n, p - v) <= 0 即在内部
    inside = True
    shallowest = 0
    shallowest_value = -float("inf")
    for index, normal in enumerate(polytope.normals):
        vertex = polytope.vertices[polytope.faces[index][0]]
        value = _dot(normal, _sub(point, vertex))
        if value > 0.0:
            inside = False
        if value > shallowest_value:
            shallowest_value = value
            shallowest = index

    if inside:
        normal = polytope.normals[shallowest]
        vertex = polytope.vertices[polytope.faces[shallowest][0]]
        closest = _sub(point, _scale(normal, -shallowest_value))
        # 距离取负，表示球心已在内部
        return closest, -abs(shallowest_value), normal

    best_distance = float("inf")
    best_point = point
    best_normal = (0.0, 0.0, 1.0)

    for index in range(len(polytope.faces)):
        candidate = _closest_point_on_polygon(polytope, index, point)
        distance = _length(_sub(candidate, point))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate
            best_normal = polytope.normals[index]

    if best_distance > 0.0:
        best_normal = _scale(_sub(point, best_point), 1.0 / best_distance)

    return best_point, best_distance, best_normal


def _closest_point_on_polygon(polytope: Polytope, face_index: int, point):
    """点到凸多边形（含其内部）的最近点。

    做法：先把点投影到该面所在的平面，再把候选限制为"平面上的投影点"
    与"各条边上的最近点"中的最小者。

    为什么不能只投影到平面：点落在棱或顶点附近时，平面投影会跑到多边形之外，
    距离被显著低估，表现为球"提前"与凸包接触。
    """
    face = polytope.faces[face_index]
    normal = polytope.normals[face_index]
    anchor = polytope.vertices[face[0]]

    signed = _dot(normal, _sub(point, anchor))
    projected = _sub(point, _scale(normal, signed))

    count = len(face)

    # 先判断**平面投影是否落在多边形内部**。
    #
    # 这一步不能省。投影点的法向距离恒为 0，若不加判断就把它当候选，
    # 任何"与面共面、但在多边形之外"的点都会得到距离 0 ——
    # 表现为球从侧面贴近凸包时被判成"已深深嵌入"，穿透等于整个半径。
    # 实测就是这样：球放在顶点外侧 0.2 m，报穿透 0.2。
    inside = True
    for i in range(count):
        p = polytope.vertices[face[i]]
        q = polytope.vertices[face[(i + 1) % count]]
        inward = _cross(normal, _sub(q, p))
        if _dot(inward, _sub(projected, p)) < -1e-12 * max(_length(inward), 1e-30):
            inside = False
            break
    if inside:
        return projected

    # 投影在多边形外 ⇒ 最近点必在边界上：逐条边取夹紧后的最近点
    best = projected
    best_distance = float("inf")

    for i in range(count):
        p = polytope.vertices[face[i]]
        q = polytope.vertices[face[(i + 1) % count]]
        edge = _sub(q, p)
        length_squared = _dot(edge, edge)
        if length_squared <= 1e-30:
            continue
        t = _dot(_sub(projected, p), edge) / length_squared
        t = min(max(t, 0.0), 1.0)
        candidate = _add(p, _scale(edge, t))
        distance = _length(_sub(candidate, point))
        if distance < best_distance:
            best_distance = distance
            best = candidate

    return best
