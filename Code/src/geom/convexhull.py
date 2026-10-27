# -*- coding: utf-8 -*-
"""凸包构造（增量法）与凸多面体表示。

为什么碰撞需要凸包
------------------
所有窄相算法（SAT、GJK）都要求**凸性**。凹体没有数学上良定义的
"最近点 / 最小穿透轴"，直接做凹-凹碰撞是没有意义的。所以任意网格进入
碰撞系统之前，必须先得到它的凸代理：凸包，或若干凸块的并集。

本模块解决第一件事：从一团点（或一个网格的顶点）构造凸包。

算法：增量构建
--------------
1. 先找到 4 个不共面的点，构成初始四面体
2. 对其余每个点，找出它能"看见"的面（点在该面外侧）
3. 看不见任何面 ⇒ 点在包内，丢弃
4. 否则找出**地平线**（可见面与不可见面之间的一圈边），删掉可见面，
   用地平线连到该点生成新面

这是 QuickHull 的核心思想（先取极端点，通常几步就能把大部分点排除），
实现上用朴素扫描代替维护可见性图——点数量在几百到几千时够用，
代码量却少一半。

为什么要合并共面三角形
----------------------
增量法天然产生三角形。一个立方体会得到 12 个三角形，但它们只落在 6 个平面上。
如果不合并，SAT 会把同一平面数三遍：面法向轴从 6 条变成 12 条，
棱数从 12 条变成 18 条，棱-棱候选轴从 144 变成 324，**慢 6 倍**。

所以构造完要做一次合并：按平面分组，把每组三角形的边界环提出来作为一个多边形面。
这个过程同时给出了 SAT 需要的面法向与棱表。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

#: 几何容差。所有判断（共面、可见性、去重）都按"特征尺寸 × 该系数"缩放，
#: 避免绝对阈值在不同尺度下失效。
_TOLERANCE = 1e-9


@dataclass
class ConvexPolyhedron:
    """一个凸多面体：顶点表 + 多边形面表 + 派生的棱与面法向。

    属性
    ----
    vertices      : 顶点坐标（局部坐标系，质心不一定在原点）
    faces         : 每个面是**逆时针（从外部看）**的顶点下标列表
    face_normals  : 每个面的单位外法向
    face_offsets  : 每个面的平面常数 ``d``，满足 ``n · x = d`` 对面上任意点成立
    edges         : 唯一无向棱表 ``(i, j)``，``i < j``
    triangles     : 三角化后的面表，用于体积/惯量计算

    顶点必须按上述约定排布，否则体积会正负抵消、SAT 法向会整体反向。
    """

    vertices: list[tuple[float, float, float]]
    faces: list[list[int]]
    face_normals: list[tuple[float, float, float]] = field(default_factory=list)
    face_offsets: list[float] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)

    def face_center(self, index: int) -> tuple[float, float, float]:
        """某个面的顶点平均位置。"""
        face = self.faces[index]
        count = len(face)
        return (
            sum(self.vertices[i][0] for i in face) / count,
            sum(self.vertices[i][1] for i in face) / count,
            sum(self.vertices[i][2] for i in face) / count,
        )

    def face_radius(self, index: int) -> float:
        """某个面到其中心的最大距离，用于裁剪时做数值缩放。"""
        center = self.face_center(index)
        best = 0.0
        for i in self.faces[index]:
            v = self.vertices[i]
            dx, dy, dz = v[0] - center[0], v[1] - center[1], v[2] - center[2]
            best = max(best, math.sqrt(dx * dx + dy * dy + dz * dz))
        return best

    def scale(self) -> float:
        """特征尺寸：顶点到包围盒中心的最大距离。"""
        if not self.vertices:
            return 1.0
        low = [min(v[k] for v in self.vertices) for k in range(3)]
        high = [max(v[k] for v in self.vertices) for k in range(3)]
        center = [(low[k] + high[k]) * 0.5 for k in range(3)]
        best = 0.0
        for v in self.vertices:
            dx, dy, dz = v[0] - center[0], v[1] - center[1], v[2] - center[2]
            best = max(best, math.sqrt(dx * dx + dy * dy + dz * dz))
        return best if best > 0.0 else 1.0

    def bounding_radius(self) -> float:
        """包围球半径（球心取顶点包围盒中心）。

        注意：与碰撞用的"以质心为球心"不同，这里只用于宽阶段的快速否定测试，
        取包围盒中心即可，保证能包住所有顶点。
        """
        return self.scale()


# ---------------------------------------------------------------------------
# 基础向量运算（本模块自足，不引入 physics 层依赖）
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normalize(a):
    n = _length(a)
    if n <= 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


# ---------------------------------------------------------------------------
# 初始四面体
# ---------------------------------------------------------------------------

def _initial_simplex(points, tolerance):
    """找到 4 个构成非退化四面体的点，返回下标 ``(i0,i1,i2,i3)``。

    逐级取"最远点"：先最远两点，再离直线最远，最后离平面最远。
    任一级退化（距离低于容差）就说明所有点共线或共面，无法构成三维凸包。
    """
    count = len(points)

    # 1) 取 x 最小的点作为起点
    i0 = min(range(count), key=lambda i: points[i][0])

    # 2) 离 i0 最远的点
    i1 = max(
        range(count),
        key=lambda i: _length(_sub(points[i], points[i0])),
    )
    if _length(_sub(points[i1], points[i0])) <= tolerance:
        raise ValueError("所有点重合，无法构造凸包")

    # 3) 离直线 i0-i1 最远的点
    direction = _normalize(_sub(points[i1], points[i0]))
    best_i2, best_area = -1, tolerance
    for i in range(count):
        if i in (i0, i1):
            continue
        offset = _sub(points[i], points[i0])
        perpendicular = _sub(offset, tuple(direction[k] * _dot(offset, direction) for k in range(3)))
        area = _length(perpendicular)
        if area > best_area:
            best_area, best_i2 = area, i
    if best_i2 < 0:
        raise ValueError("所有点共线，无法构造三维凸包")
    i2 = best_i2

    # 4) 离平面 i0-i1-i2 最远的点
    normal = _normalize(_cross(_sub(points[i1], points[i0]), _sub(points[i2], points[i0])))
    best_i3, best_height = -1, tolerance
    for i in range(count):
        if i in (i0, i1, i2):
            continue
        height = abs(_dot(_sub(points[i], points[i0]), normal))
        if height > best_height:
            best_height, best_i3 = height, i
    if best_i3 < 0:
        raise ValueError("所有点共面，无法构造三维凸包（退化凸包）")

    return (i0, i1, i2, best_i3)


def _make_face(vertices, a, b, c, interior):
    """构造三角形面，并保证外法向背向 ``interior``。

    统一在这里定朝向，比在每个调用点推导绕向可靠得多：
    ``interior`` 是初始四面体的重心，必然严格位于凸包内部。
    """
    normal = _cross(_sub(vertices[b], vertices[a]), _sub(vertices[c], vertices[a]))
    if _length(normal) <= 0.0:
        return None
    face_center = tuple((vertices[a][k] + vertices[b][k] + vertices[c][k]) / 3.0 for k in range(3))
    if _dot(normal, _sub(face_center, interior)) < 0.0:
        b, c = c, b
    return (a, b, c)


# ---------------------------------------------------------------------------
# 共面三角形合并
# ---------------------------------------------------------------------------

def _merge_coplanar(vertices, triangles, tolerance):
    """把落在同一平面上的三角形合并成多边形面。

    做法：按 ``(法向, 平面常数)`` 分组，然后对每组收集**有向边**——
    组内共享的内部边会同时以正反两个方向出现，互相抵消；
    剩下的就是该多边形的边界环。

    凸包上同一平面只可能有一块连通区域，所以边界恰好是一个环。
    """
    scale = 1.0
    if vertices:
        scale = max(
            max(abs(v[k]) for v in vertices) for k in range(3)
        ) or 1.0
    plane_tolerance = tolerance * scale

    groups: list[tuple[tuple[float, float, float], float, list[tuple[int, int, int]]]] = []

    for tri in triangles:
        a, b, c = tri
        normal = _cross(_sub(vertices[b], vertices[a]), _sub(vertices[c], vertices[a]))
        length = _length(normal)
        if length <= 0.0:
            continue   # 退化三角形，丢弃
        unit = (normal[0] / length, normal[1] / length, normal[2] / length)
        offset = _dot(unit, vertices[a])

        placed = False
        for index, (group_normal, group_offset, group_tris) in enumerate(groups):
            if (
                abs(_dot(unit, group_normal) - 1.0) <= 1e-7
                and abs(offset - group_offset) <= plane_tolerance * 10.0
            ):
                group_tris.append(tri)
                placed = True
                break
        if not placed:
            groups.append((unit, offset, [tri]))

    faces: list[list[int]] = []
    face_normals: list[tuple[float, float, float]] = []
    face_offsets: list[float] = []

    for normal, offset, group_tris in groups:
        # 收集有向边，内部共享边成对抵消
        directed: dict[tuple[int, int], int] = {}
        for a, b, c in group_tris:
            for edge in ((a, b), (b, c), (c, a)):
                reverse = (edge[1], edge[0])
                if directed.get(reverse):
                    directed[reverse] -= 1
                    if directed[reverse] == 0:
                        del directed[reverse]
                else:
                    directed[edge] = directed.get(edge, 0) + 1

        boundary = [(u, v) for (u, v), _ in directed.items()]
        if len(boundary) < 3:
            continue

        # 按有向边串成环：next[u] = v。
        # 若同一顶点出现两条出边，说明该平面组不连通（凸包上不会发生），
        # 直接丢弃这个面，避免产生自交多边形。
        following: dict[int, int] = {}
        broken = False
        for u, v in boundary:
            if u in following:
                broken = True
                break
            following[u] = v
        if broken:
            continue

        start = boundary[0][0]
        loop = [start]
        cursor = following.get(start, -1)
        guard = 0
        while cursor != -1 and cursor != start and guard <= len(boundary):
            loop.append(cursor)
            cursor = following.get(cursor, -1)
            guard += 1

        if len(loop) != len(boundary) or len(loop) < 3:
            continue   # 成环失败，丢弃（不会发生，但防御住）

        faces.append(loop)
        face_normals.append(normal)
        face_offsets.append(offset)

    return faces, face_normals, face_offsets


def _unique_edges(faces) -> list[tuple[int, int]]:
    """从未重复的面顶点对中提取唯一无向棱。"""
    seen = set()
    result = []
    for face in faces:
        count = len(face)
        for i in range(count):
            u, v = face[i], face[(i + 1) % count]
            key = (u, v) if u < v else (v, u)
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _fan_triangles(faces) -> list[tuple[int, int, int]]:
    """把多边形面扇形三角化（凸多边形保证扇形完全落在面内）。"""
    result = []
    for face in faces:
        for i in range(1, len(face) - 1):
            result.append((face[0], face[i], face[i + 1]))
    return result


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def convex_hull(points: Sequence[Sequence[float]]) -> ConvexPolyhedron:
    """从点集构造凸包。

    参数
    ----
    points : 任意点集（不必是凸的，内部点会被自动剔除）

    返回
    ----
    ConvexPolyhedron : 顶点经过去重、面按逆时针朝外排列

    异常
    -----
    ValueError : 点数不足 4、全部重合、共线、或共面（退化凸包）
    """
    raw = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if len(raw) < 4:
        raise ValueError(f"构造凸包至少需要 4 个点，收到 {len(raw)}")

    # --- 去重（按特征尺度设阈值，否则绝对阈值在不同尺度下会失效）---
    low = [min(p[k] for p in raw) for k in range(3)]
    high = [max(p[k] for p in raw) for k in range(3)]
    extent = max(high[k] - low[k] for k in range(3))
    if extent <= 0.0:
        raise ValueError("所有点重合，无法构造凸包")
    tolerance = _TOLERANCE * max(extent, 1.0)

    unique: list[tuple[float, float, float]] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    inverse = 1.0 / tolerance
    for point in raw:
        key = (
            int(math.floor(point[0] * inverse)),
            int(math.floor(point[1] * inverse)),
            int(math.floor(point[2] * inverse)),
        )
        duplicate = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        candidate = unique[other]
                        if _length(_sub(candidate, point)) <= tolerance:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if not duplicate:
            buckets.setdefault(key, []).append(len(unique))
            unique.append(point)

    if len(unique) < 4:
        raise ValueError("去重后有效点少于 4 个，无法构造凸包")

    i0, i1, i2, i3 = _initial_simplex(unique, tolerance)
    interior = tuple(
        (unique[i0][k] + unique[i1][k] + unique[i2][k] + unique[i3][k]) / 4.0
        for k in range(3)
    )

    triangles: list[tuple[int, int, int]] = []
    for a, b, c in (
        (i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3),
    ):
        face = _make_face(unique, a, b, c, interior)
        if face is not None:
            triangles.append(face)

    if len(triangles) < 4:
        raise ValueError("初始四面体退化，无法构造凸包")

    used = {i0, i1, i2, i3}
    face_tolerance = tolerance * 10.0

    for index, point in enumerate(unique):
        if index in used:
            continue

        # 找出该点能看见的面（点在平面外侧）
        visible = []
        for face in triangles:
            a, b, c = face
            normal = _cross(_sub(unique[b], unique[a]), _sub(unique[c], unique[a]))
            if _dot(normal, _sub(point, unique[a])) > face_tolerance * _length(normal):
                visible.append(face)

        if not visible:
            continue    # 点在包内

        # 地平线：可见面的有向边中，反向边不在可见集合里的那些
        visible_edges = {}
        for a, b, c in visible:
            for u, v in ((a, b), (b, c), (c, a)):
                visible_edges[(u, v)] = True

        horizon = []
        for (u, v) in visible_edges:
            if (v, u) not in visible_edges:
                horizon.append((u, v))

        visible_set = set(visible)
        triangles = [face for face in triangles if face not in visible_set]

        for u, v in horizon:
            face = _make_face(unique, u, v, index, interior)
            if face is not None:
                triangles.append(face)

        used.add(index)

    faces, normals, offsets = _merge_coplanar(unique, triangles, tolerance)
    if len(faces) < 4:
        raise ValueError("合并共面面片后剩余面不足 4 个，凸包退化")

    # 只保留被面引用到的顶点，压缩编号
    referenced = sorted({index for face in faces for index in face})
    remap = {old: new for new, old in enumerate(referenced)}
    vertices = [unique[old] for old in referenced]
    faces = [[remap[index] for index in face] for face in faces]

    return ConvexPolyhedron(
        vertices=vertices,
        faces=faces,
        face_normals=normals,
        face_offsets=offsets,
        edges=_unique_edges(faces),
        triangles=_fan_triangles(faces),
    )
