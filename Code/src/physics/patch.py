# -*- coding: utf-8 -*-
"""支撑 patch 的**语义层**：把"哪一片几何在接触"变成"哪些位置真的重叠"。

为什么要多这一层
----------------
``support.support_patch_world`` 只回答"沿这个方向哪一片是支撑集"，它是三维的、
连续的。接触流形要的是"两侧区域在**切平面上的交集**"——一个二维问题：

    两侧 patch ──投影──> 两个二维凸点集 ──求交──> 交叠区 ──稀疏──> ≤4 个接触点

P0-1 的旧实现跳过了中间三步，直接拿"沿法向最近的那四个采样点"当接触点：四个点
关于圆柱轴对称，净力矩为零，于是靠边搭接的物体被四个并不存在的支撑点托住。
这一层的作用就是把"切向覆盖"补回来。

退化是这个问题的常态
--------------------
圆盘侧视是线段，锥顶是点，两根杆交叉时两侧都是点。所以：

- **类别必须由投影后的二维结果决定**，不能沿用 ``SupportPatch.kind``（三维是
  多边形、投影后可能只剩一条线段）；
- 所有组合都由 :func:`mathutils.poly2d.convex_intersection` 统一处理，不在调用方
  按点数分派——那样一定会漏掉某种组合。

坐标与精度
----------
投影原点是两侧 patch 质心的中点（而不是世界原点）：接触点常常离原点很远，
"先减原点再投影"是唯一能保住有效数字的做法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..mathutils import poly2d, vec3
from .support import SupportPatch, support_patch_world

#: 接触路径的圆盘边数上限。
#:
#: 求交路径用 256 边是为了让多边形逼近圆盘（内缩 < 0.5 mm）；接触只需要"点落在
#: 交叠区里"，而结果最后会被稀疏到 4 点，256 边换来的精度会被直接丢掉，白付
#: O(N·M) 的求交代价。64 边在 r = 1 m 时内缩 1.2 mm，仍在 ``slop`` 以内。
MAX_CONTACT_SEGMENTS = 64

#: 一个流形最多保留 4 点：法向约束只需要撑开一个平面，多出来的点只会让
#: 求解器多做几组同样的约束。
MAX_CONTACT_POINTS = 4

#: 交集判定容差（米）。投影后坐标已经在"接触点附近"，所以这个量级足够。
_OVERLAP_EPS = 1e-9

#: 空交集退化成单点的次数。用于观测"离散误差导致的回退"是否频繁，
#: 它非零不代表出错，但突然变大说明边数上限或容差需要复查。
_fallback_count = 0


def fallback_count() -> int:
    """到目前为止，有多少次因为切向交集为空而退化成单点。"""
    return _fallback_count


def reset_fallback_count() -> None:
    """清零计数（测试用）。"""
    global _fallback_count
    _fallback_count = 0


def note_fallback() -> None:
    """记录一次单点回退。"""
    global _fallback_count
    _fallback_count += 1


@dataclass
class PatchOverlap:
    """两侧支撑 patch 在切平面上的交集。

    属性
    ----
    points : list[tuple[float, float]]
        交集中的点，二维（``origin + x·u + y·v`` 即世界坐标）。已经稀疏到
        :data:`MAX_CONTACT_POINTS` 以内；空列表表示没有交叠。
    origin : tuple[float, float, float]
        投影原点（世界坐标），两侧质心的中点。
    u, v : tuple[float, float, float]
        切向基，来自 :func:`poly2d.tangent_basis`，对 ``n`` 与 ``-n`` 逐位一致。
    empty_reason : str | None
        为空时的原因：``"missing_patch"``（有一侧没有顶点）/ ``"disjoint"``
        （有 patch 但切向不重叠）/ ``None``（非空）。
    """

    points: list
    origin: tuple
    u: tuple
    v: tuple
    empty_reason: str | None = None


@dataclass
class Anchor:
    """一个接触点在两侧表面上的锚点与合成结果。

    ``position`` 取两锚点中点，``penetration`` 是沿法向的间隙（正值表示嵌入）。
    """

    position: tuple
    normal: tuple
    penetration: float


def contact_patch_world(body, direction) -> SupportPatch:
    """接触路径的支撑 patch：与求交同源，但圆盘边数封顶 :data:`MAX_CONTACT_SEGMENTS`。"""
    return support_patch_world(body, direction, max_segments=MAX_CONTACT_SEGMENTS)


def overlap(patch_a: SupportPatch, patch_b: SupportPatch, normal) -> PatchOverlap:
    """两侧 patch 在法向 ``normal`` 的切平面上的交集。

    ``normal`` 必须是从 A 指向 B 的单位向量；两侧 patch 应分别是沿 ``normal`` /
    ``-normal`` 取到的支撑集。返回的点已稀疏到 4 点以内，顺序按 ``(x, y)`` 排序，
    因此交换 A/B 得到的是同一组点（``test_poly2d`` 的第 12 项守着这条性质）。
    """
    u, v = poly2d.tangent_basis(normal)
    if not patch_a.vertices or not patch_b.vertices:
        return PatchOverlap([], (0.0, 0.0, 0.0), u, v, "missing_patch")

    origin = _midpoint(_centroid(patch_a.vertices), _centroid(patch_b.vertices))
    flat_a = poly2d.project(patch_a.vertices, u, v, origin)
    flat_b = poly2d.project(patch_b.vertices, u, v, origin)
    points = poly2d.convex_intersection(flat_a, flat_b, _OVERLAP_EPS)
    reduced = reduce_to_four(points)
    return PatchOverlap(reduced, origin, u, v, None if reduced else "disjoint")


def reduce_to_four(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """把交集稀疏成最多 4 点，且尽量"铺开"。

    规则：从交集的**面积重心**沿 ``±x``、``±y`` 射四条射线，取边界交点。

    为什么不直接取"顶点里 x/y 的极值点"：交点都是内接多边形的顶点，物体稍微
    转一点，极值就可能换到相邻顶点上，位置于是跳一个**顶点间距**——半径 0.3 m、
    55 边形时实测跳 1.7 cm。射线求交拿到的是边上的插值点，同样的扰动只引起
    "多边形边界本身"的误差（内缩，0.5 mm 量级），而且四个方向固定，不会因为
    极值换人而整体重排。

    交集本身不足 4 点时**不补点**：1 点（点接触）、2 点（线段）、3 点（三角形）
    都是合法流形，硬凑到 4 点会制造出并不存在的支撑。
    """
    if len(points) <= MAX_CONTACT_POINTS:
        return list(points)

    # 射线求交需要**环序**；输入顺序不保证（可能是顺时针，也可能是乱序），
    # 先过一次凸包把它规范化。这一步同时也是确定性来源：凸包的输出与输入顺序
    # 无关，"正序 / 逆序输入给出同一组点"靠的就是它（省掉它会差在最后几位上）。
    polygon = [points[index] for index in poly2d.convex_hull_2d(points)]
    center = (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )
    picked: list[tuple[float, float]] = []
    for direction in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
        hit = _ray_to_boundary(center, direction, polygon)
        if hit is None:
            continue
        if not any(_same_point(hit, other) for other in picked):
            picked.append(hit)

    if len(picked) < MAX_CONTACT_POINTS:
        # 交集退化成一条线（或射线打空）：用最远点采样补齐，仍然是确定性规则
        remaining = [
            point for point in sorted(points)
            if not any(_same_point(point, other) for other in picked)
        ]
        while remaining and len(picked) < MAX_CONTACT_POINTS:
            farthest = max(
                remaining,
                key=lambda point: (
                    min(_distance_squared_2d(point, other) for other in picked),
                    point,
                ),
            )
            picked.append(farthest)
            remaining.remove(farthest)

    return sorted(picked)


def _ray_to_boundary(center, direction, polygon):
    """从 ``center`` 沿 ``direction`` 射出，返回与逆时针凸多边形的第一个交点。

    ``center`` 必须（近似）在多边形内部；打不到就返回 ``None``。交点用边的
    **插值**得到，而不是返回某个顶点——这正是"位置不随极值换人而跳变"的来源。
    """
    best = None
    count = len(polygon)
    for index in range(count):
        first = polygon[index]
        second = polygon[(index + 1) % count]
        edge_x = second[0] - first[0]
        edge_y = second[1] - first[1]
        denominator = direction[0] * edge_y - direction[1] * edge_x
        if abs(denominator) <= 1e-15:
            continue
        offset_x = first[0] - center[0]
        offset_y = first[1] - center[1]
        # 解 center + t·d = first + u·e：
        #   t = (w × e) / (d × e)，u = (w × d) / (d × e)，w = first - center
        t = (offset_x * edge_y - offset_y * edge_x) / denominator
        u = (offset_x * direction[1] - offset_y * direction[0]) / denominator
        if t < -1e-12 or u < -1e-9 or u > 1.0 + 1e-9:
            continue
        if best is None or t < best:
            best = t
    if best is None:
        return None
    return (center[0] + best * direction[0], center[1] + best * direction[1])


def lift_anchors(
    patch_overlap: PatchOverlap,
    patch_a: SupportPatch,
    patch_b: SupportPatch,
    normal,
) -> list[Anchor]:
    """把交集点抬回三维，投到两侧支撑平面上，得到锚点、位置与深度。

    支撑集天然是平面的：沿 ``normal`` 的支撑点都满足 ``x·n = h(n)``。所以两侧
    各自有一张垂直于 ``n`` 的平面，深度就是两张平面之间的距离——同一流形里
    每个点相同。逐点计算（而不是直接用 EPA 的全局深度）是为了让"点确实落在
    两侧表面上"这件事由构造保证，而不是靠外部传参。
    """
    if not patch_overlap.points:
        return []

    plane_a = _plane_offset(patch_a.vertices, normal)
    plane_b = _plane_offset(patch_b.vertices, normal)
    origin = patch_overlap.origin
    u, v = patch_overlap.u, patch_overlap.v

    anchors: list[Anchor] = []
    for x, y in patch_overlap.points:
        world = (
            origin[0] + x * u[0] + y * v[0],
            origin[1] + x * u[1] + y * v[1],
            origin[2] + x * u[2] + y * v[2],
        )
        along = vec3.dot(world, normal)
        anchor_a = vec3.madd(world, normal, plane_a - along)
        anchor_b = vec3.madd(world, normal, plane_b - along)
        penetration = vec3.dot(vec3.sub(anchor_a, anchor_b), normal)
        anchors.append(
            Anchor(
                position=_midpoint(anchor_a, anchor_b),
                normal=normal,
                penetration=penetration,
            )
        )
    return anchors


def match_tags(
    points: list[tuple[float, float, float]],
    tagged: list[tuple[tuple[float, float, float], int]],
    normal,
) -> list[int]:
    """给每个接触点分配一个**互不相同**的槽位标签。

    输入 ``tagged`` 是 ``support.feature_world_tagged`` 的四点（底面 ``0..3``、
    顶面 ``4..7``）。两条要求同时成立：

    1. **批内唯一**：``feature_id = tag_a*TAG_LIMIT + tag_b`` 不能撞号——撞了
       暖启动会把某处的冲量施给另一个接触点，症状是静置物体缓慢自旋；
    2. **连续**：物体转一点点，标签不该换人。用"各自取最近槽位"做不到这一点：
       交集的极值点正好落在两个槽位的中垂线上时（轴对齐的面接触就是这种情况），
       任意小的扰动都会让最近的槽位翻到对面。

    所以点数与槽位数相等时走**按极角排序后对齐**：点和槽位各自按切平面极角排序，
    第 k 个点配第 k 个槽位。这里**不做**"总角度偏移最小"的优化——那一步在两个
    槽位集合相距 45° 时会退化成精确平局（轴对齐面接触就是如此），平局两侧的代价
    差 1e-3 就能让整组配对轮转一位。固定用"排序后对齐"，配对就只依赖两边的**环形
    次序**，而次序在微扰下不变。

    数量不等时（例如 3 点流形对 4 个槽位、或单点接触）退化为"距离升序的贪心
    一对一"：唯一性仍然成立，只是不再保证扰动下的连续。
    """
    if not points or not tagged:
        return [0] * len(points)

    u, v = poly2d.tangent_basis(normal)
    origin = _centroid([point for point, _tag in tagged])
    angles = [_angle(point, origin, u, v) for point in points]
    slot_angles = [_angle(point, origin, u, v) for point, _tag in tagged]

    if None in angles or None in slot_angles or len(points) != len(tagged):
        return _greedy_tags(points, tagged)

    point_order = sorted(range(len(points)), key=lambda index: angles[index])
    slot_order = sorted(range(len(tagged)), key=lambda index: slot_angles[index])
    assigned = [0] * len(points)
    for position, point_index in enumerate(point_order):
        assigned[point_index] = tagged[slot_order[position]][1]
    return assigned


def _greedy_tags(points, tagged) -> list[int]:
    """距离升序的一对一分配（点数与槽位数不相等时的退化路径）。"""
    remaining = list(range(len(tagged)))
    assigned: list[int | None] = [None] * len(points)
    pairs = sorted(
        (distance_squared(point, tagged[index][0]), point_index, index)
        for point_index, point in enumerate(points)
        for index in range(len(tagged))
    )
    for _squared, point_index, index in pairs:
        if assigned[point_index] is not None or index not in remaining:
            continue
        assigned[point_index] = tagged[index][1]
        remaining.remove(index)
        if all(item is not None for item in assigned):
            break

    fallback = tagged[-1][1]
    return [fallback if tag is None else tag for tag in assigned]


def _angle(point, origin, u, v):
    """点在切平面上的极角；落在原点（角度无定义）时返回 ``None``。"""
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    dz = point[2] - origin[2]
    planar_u = dx * u[0] + dy * u[1] + dz * u[2]
    planar_v = dx * v[0] + dy * v[1] + dz * v[2]
    if math.hypot(planar_u, planar_v) <= 1e-12:
        return None
    return math.atan2(planar_v, planar_u)





def distance_squared(a, b) -> float:
    """两点距离的平方（避免开方，配对时只需要比较大小）。"""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _centroid(points) -> tuple[float, float, float]:
    """点列的平均位置。"""
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _midpoint(a, b) -> tuple[float, float, float]:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]))


def _plane_offset(points, normal) -> float:
    """支撑集所在平面沿 ``normal`` 的偏移 ``h(n)``（支撑集共面，取均值抗噪声）。"""
    total = 0.0
    for point in points:
        total += vec3.dot(point, normal)
    return total / len(points)


def _same_point(a, b, eps: float = 1e-12) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps





def _distance_squared_2d(a, b) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy
