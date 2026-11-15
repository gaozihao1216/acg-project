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

#: 空交集退化成单点的**原因计数**。非零不代表出错（离散误差、窄搭接都会触发），
#: 但某一种原因突然变大，说明对应的环节需要复查：``"disjoint"`` 变多通常是边数
#: 上限或容差，``"separated"`` 变多则说明法向/深度与 patch 不自洽。
_fallback_reasons: dict[str, int] = {}


def fallback_count() -> int:
    """到目前为止，有多少次因为切向交集为空而退化成单点。"""
    return sum(_fallback_reasons.values())


def fallback_reasons() -> dict[str, int]:
    """按原因分类的回退次数（``disjoint`` / ``missing_patch`` / ``separated``）。"""
    return dict(_fallback_reasons)


def reset_fallback_count() -> None:
    """清零计数（测试用）。"""
    _fallback_reasons.clear()


def note_fallback(reason: str = "unknown") -> None:
    """记录一次单点回退及其原因。"""
    _fallback_reasons[reason] = _fallback_reasons.get(reason, 0) + 1


@dataclass
class PatchOverlap:
    """两侧支撑 patch 在切平面上的交集（**完整**交集，未稀疏）。

    属性
    ----
    points : list[tuple[float, float]]
        交集多边形的顶点，二维（``origin + x·u + y·v`` 即世界坐标）。空列表表示
        没有交叠。稀疏到 4 点是下一步（:func:`contact_anchors`）的事——必须先按
        真实表面算出每点的间隙，否则倾斜端面上"本已分离"的点会在缩减时被当成
        候选，甚至挤掉真正接触的点。
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
    """两侧 patch 在法向 ``normal`` 的切平面上的交集（不稀疏）。

    ``normal`` 必须是从 A 指向 B 的单位向量；两侧 patch 应分别是沿 ``normal`` /
    ``-normal`` 取到的支撑集。求交在规范化（逆时针）坐标下进行，因此交换 A/B
    得到同一个多边形（``test_patch`` 的第 3 项守着这条性质）。
    """
    u, v = poly2d.tangent_basis(normal)
    if not patch_a.vertices or not patch_b.vertices:
        return PatchOverlap([], (0.0, 0.0, 0.0), u, v, "missing_patch")

    origin = _midpoint(_centroid(patch_a.vertices), _centroid(patch_b.vertices))
    flat_a = poly2d.project(patch_a.vertices, u, v, origin)
    flat_b = poly2d.project(patch_b.vertices, u, v, origin)
    points = poly2d.convex_intersection(flat_a, flat_b, _OVERLAP_EPS)
    return PatchOverlap(points, origin, u, v, None if points else "disjoint")


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


def contact_anchors(
    patch_overlap: PatchOverlap,
    patch_a: SupportPatch,
    patch_b: SupportPatch,
    normal,
) -> list[Anchor]:
    """交集 → 按**真实表面**逐点算间隙、裁掉分离点 → 稀疏到 ≤4 点 → 抬成锚点。

    为什么不能"先把交集缩到 4 点，再用一张垂直于 ``normal`` 的平面算深度"
    ------------------------------------------------------------------
    ``support_patch_world`` 对端面留了约 3° 的容差（这是为了不让求解器几毫弧度的
    抖动把 4 点流形塌成 1 点），于是在倾斜端面上，patch 是**整片倾斜的圆盘**，
    而垂直于 ``normal`` 的平均平面并不在物体表面上：半径 0.3 m 的圆柱倾斜 2° 时，
    那个平面与真实端面在边缘处差 ``0.3·sin2° ≈ 10.5 mm``。

    后果是双向的：低侧少报深度、高侧给**本已分离**的点建立约束（实测四点都报
    ``9.70 mm``，真实值是 ``20.16 mm`` 与 ``−0.77 mm``）。凭空多出来的支撑点会
    改变力矩，看起来像"该倒的不倒"。

    所以间隙用两侧 patch **各自的拟合平面**（倾斜端面就是端面本身，平面件就是
    面本身）与法向直线的交点来算，并且**先裁掉间隙为负的点再稀疏**——否则高侧
    的点可能在射线取极值时把低侧的点挤下去。
    """
    if not patch_overlap.points:
        return []

    surface_a = _surface(patch_a, patch_overlap, normal)
    surface_b = _surface(patch_b, patch_overlap, normal)
    values = _gap_values(patch_overlap.points, surface_a, surface_b, normal)

    # 间隙沿切平面是仿射的（平面件显然；母线是端点线性插值，也是仿射），
    # 所以"≥0"可以在二维里精确裁剪，而不是只丢掉负值顶点——那会漏掉边界交点。
    candidates = _clip_positive(patch_overlap.points, values)
    if not candidates:
        return []

    selected = reduce_to_four(candidates)
    anchors: list[Anchor] = []
    for point in selected:
        anchor_a = surface_a.point_at(point)
        anchor_b = surface_b.point_at(point)
        # 裁剪线（间隙=0）上的点重新求值可能差 -1e-17 m：夹到 0，
        # 既避免 ContactPoint 的"深度必须非负"检查，也符合"相切"的物理含义。
        penetration = max(0.0, vec3.dot(vec3.sub(anchor_a, anchor_b), normal))
        anchors.append(
            Anchor(
                position=_midpoint(anchor_a, anchor_b),
                normal=normal,
                penetration=penetration,
            )
        )
    return anchors


class _Surface:
    """一侧 patch 的"表面"：给定切平面坐标，回答"这里的表面点在哪"。

    三种退化各有各的求法，混用会出错：

    - **多边形**（端面圆盘、盒面、凸包面）：顶点共面，Newell 拟合平面，沿碰撞
      法向求交；
    - **线段**（圆柱/胶囊侧面的母线）：**没有**唯一平面。侧面选取留了容差，倾斜
      母线会被整条交下来，此时它的两个端点法向坐标相差可达数十毫米——用"过顶点
      均值的垂直平面"就等于把整条线压平，深度全报中位数（实测两端真实是
      −12.633 mm / +22.267 mm，却都报 4.817 mm）。正确做法是按切向投影参数
      **插值三个端点**，间隙沿母线线性变化；
    - **点**（严格凸体的支撑点）：表面点就是它自己。
    """

    __slots__ = ("kind", "vertices", "plane", "flat", "origin", "u", "v", "normal")

    def __init__(self, patch: SupportPatch, overlap: PatchOverlap, normal) -> None:
        self.vertices = patch.vertices
        self.origin = overlap.origin
        self.u = overlap.u
        self.v = overlap.v
        self.normal = normal
        self.plane = None
        self.flat = None

        if len(self.vertices) >= 3:
            self.kind = "plane"
            self.plane = _patch_plane(self.vertices, normal)
        elif len(self.vertices) == 2:
            self.kind = "segment"
            self.flat = poly2d.project(self.vertices, self.u, self.v, self.origin)
        else:
            self.kind = "point"

    def point_at(self, point):
        """切平面坐标 → 该侧表面上的点（世界坐标）。"""
        if self.kind == "point":
            return self.vertices[0]
        if self.kind == "segment":
            return self._segment_point(point)
        world = (
            self.origin[0] + point[0] * self.u[0] + point[1] * self.v[0],
            self.origin[1] + point[0] * self.u[1] + point[1] * self.v[1],
            self.origin[2] + point[0] * self.u[2] + point[1] * self.v[2],
        )
        return _project_to_plane(world, self.plane, self.normal)

    def _segment_point(self, point):
        (x0, y0), (x1, y1) = self.flat
        dx = x1 - x0
        dy = y1 - y0
        denominator = dx * dx + dy * dy
        if denominator <= 1e-18:
            return self.vertices[0]
        parameter = ((point[0] - x0) * dx + (point[1] - y0) * dy) / denominator
        parameter = min(1.0, max(0.0, parameter))
        first, second = self.vertices
        return (
            first[0] + parameter * (second[0] - first[0]),
            first[1] + parameter * (second[1] - first[1]),
            first[2] + parameter * (second[2] - first[2]),
        )


def _surface(patch: SupportPatch, overlap: PatchOverlap, normal) -> _Surface:
    return _Surface(patch, overlap, normal)


def _gap_values(points, surface_a, surface_b, normal):
    """逐点的法向间隙，用**交集多边形内**的三角形拟合仿射形式加速。

    间隙沿切平面是仿射的（平面投影、母线端点插值都是），但母线插值在超出端点范围
    时会夹到端点，**在交集之外不再仿射**。所以三个采样点必须取交集内部的顶点，
    不能用固定的 (0,0)/(1,0)/(0,1)——那可能落在母线延长线上，拟合出错误的系数。

    55 边形时这把 110 次表面投影降到 6 次。
    """

    def gap(point):
        return vec3.dot(
            vec3.sub(surface_a.point_at(point), surface_b.point_at(point)), normal
        )

    count = len(points)
    if count < 3:
        return [gap(point) for point in points]

    first = points[0]
    second = max(
        points,
        key=lambda point: (point[0] - first[0]) ** 2 + (point[1] - first[1]) ** 2,
    )
    if _same_point(first, second):
        return [gap(point) for point in points]
    third = max(
        points,
        key=lambda point: abs(
            (second[0] - first[0]) * (point[1] - first[1])
            - (second[1] - first[1]) * (point[0] - first[0])
        ),
    )
    dx1, dy1 = second[0] - first[0], second[1] - first[1]
    dx2, dy2 = third[0] - first[0], third[1] - first[1]
    cross = dx1 * dy2 - dy1 * dx2
    if abs(cross) <= 1e-18:
        return [gap(point) for point in points]

    base = gap(first)
    rise1 = gap(second) - base
    rise2 = gap(third) - base
    along_u = (rise1 * dy2 - dy1 * rise2) / cross
    along_v = (dx1 * rise2 - rise1 * dx2) / cross
    return [
        base + along_u * (point[0] - first[0]) + along_v * (point[1] - first[1])
        for point in points
    ]


def _patch_plane(vertices, normal) -> tuple[tuple, float]:
    """多边形的平面：``(单位法向, 偏移)``，法向与外 ``normal`` 同向。"""
    nx = ny = nz = 0.0
    count = len(vertices)
    for index in range(count):
        first = vertices[index]
        second = vertices[(index + 1) % count]
        nx += (first[1] - second[1]) * (first[2] + second[2])
        ny += (first[2] - second[2]) * (first[0] + second[0])
        nz += (first[0] - second[0]) * (first[1] + second[1])
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return tuple(normal), _plane_offset(vertices, normal)
    plane_normal = (nx / length, ny / length, nz / length)
    if vec3.dot(plane_normal, normal) < 0.0:
        plane_normal = (-plane_normal[0], -plane_normal[1], -plane_normal[2])
    return plane_normal, _plane_offset(vertices, plane_normal)


def _project_to_plane(world, plane, normal) -> tuple:
    """把世界点沿 ``normal`` 推到平面上（平面与 normal 平行时退回平均偏移）。"""
    plane_normal, offset = plane
    denominator = vec3.dot(normal, plane_normal)
    if abs(denominator) <= 1e-9:
        # 平面与法向平行：没有交点，退回"垂直平面"近似
        return vec3.madd(world, normal, offset - vec3.dot(world, normal))
    return vec3.madd(
        world, normal, (offset - vec3.dot(world, plane_normal)) / denominator
    )


def _clip_positive(points: list[tuple[float, float]], values: list[float]):
    """按 ``values >= 0`` 裁剪凸点集。

    点数分派而不是一律走多边形循环：两点输入当成"闭合二边形"遍历时，正向和反向
    是同一条边，**交点会被插入两次**——实测返回三个锚点、其中两个位置重合。
    两点按参数区间裁一次就够。
    """
    count = len(points)
    if count == 2:
        (first, second), (value_first, value_second) = points, values
        if value_first >= 0.0 and value_second >= 0.0:
            return [first, second]
        if value_first < 0.0 and value_second < 0.0:
            return []
        parameter = value_first / (value_first - value_second)
        crossing = (
            first[0] + parameter * (second[0] - first[0]),
            first[1] + parameter * (second[1] - first[1]),
        )
        return [first, crossing] if value_first >= 0.0 else [crossing, second]

    output: list[tuple[float, float]] = []
    for index in range(count):
        current = points[index]
        following = points[(index + 1) % count]
        current_value = values[index]
        following_value = values[(index + 1) % count]
        if current_value >= 0.0:
            output.append(current)
        if (current_value > 0.0 and following_value < 0.0) or (
            current_value < 0.0 and following_value > 0.0
        ):
            parameter = current_value / (current_value - following_value)
            output.append(
                (
                    current[0] + parameter * (following[0] - current[0]),
                    current[1] + parameter * (following[1] - current[1]),
                )
            )

    # 顶点恰好落在裁剪线上时，"保留顶点"与"插入交点"会产生重合点，去掉
    deduped: list[tuple[float, float]] = []
    for point in output:
        if not any(_same_point(point, other) for other in deduped):
            deduped.append(point)
    return deduped


def _to_world(x: float, y: float, patch_overlap: PatchOverlap) -> tuple:
    """切平面二维坐标 → 世界坐标。"""
    origin = patch_overlap.origin
    u, v = patch_overlap.u, patch_overlap.v
    return (
        origin[0] + x * u[0] + y * v[0],
        origin[1] + x * u[1] + y * v[1],
        origin[2] + x * u[2] + y * v[2],
    )


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
