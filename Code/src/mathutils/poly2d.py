# -*- coding: utf-8 -*-
"""二维凸多边形工具：接触平面的切向基、凸包、包含测试与求交。

为什么单列一个模块
------------------
接触流形要在**切平面**上判断"两侧支撑区域是否真的重叠"。这类计算纯粹是二维
几何，与刚体、几何体都不相干。放在 ``mathutils`` 里可以让 ``support``（给
支撑 patch 排序）与窄相（求交）共用，又不产生 physics 内部循环依赖。

统一约定
--------
- 点用 ``(x, y)``；
- 多边形顶点按**逆时针**给出（:func:`convex_hull_2d` 的输出保证这一点）；
- 所有判定带 ``eps`` 容差，避免"恰好共边"被判成不重叠。

关于切向基的符号
----------------
:func:`tangent_basis` 先把法向规范化到"绝对值最大的分量为正"，再建正交基。
这一步是 A/B 对称的前提：``normal`` 与 ``-normal`` 必须给出**同一组**切向，
否则两侧投影到两个不同的平面里，求交结果没有可比性。
"""

from __future__ import annotations

import math

Point2 = tuple[float, float]

#: 当作"零"的长度阈值。法向、边向量小于它就不该再拿去向量积。
_NORMAL_EPS = 1e-12


def canonical_normal(normal):
    """把 ``n`` 与 ``-n`` 规范到同一个方向（绝对值最大的分量为正）。"""
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if ax >= ay and ax >= az:
        flip = normal[0] < 0.0
    elif ay >= az:
        flip = normal[1] < 0.0
    else:
        flip = normal[2] < 0.0
    if not flip:
        return (normal[0], normal[1], normal[2])
    return (-normal[0], -normal[1], -normal[2])


def tangent_basis(normal):
    """接触平面的两个切向 ``(u, v)``，对 ``n`` 与 ``-n`` 稳定一致。

    入口先把法向**归一化**：``v = n × u`` 只有在 ``|n| = 1`` 且 ``u ⊥ n`` 时才是
    单位向量。传进来一个未归一化的法向（GJK 的搜索方向、EPA 的原始面法向都可能
    带着长度）过去会得到 ``|v| = |n|``，投影到切平面时 y 方向被整体拉伸——那种
    错误不会报错，只会让交集看起来"怪异地扁"。

    零法向没有切平面可言，直接拒绝而不是除以零：调用方此时应该先检查接触是否
    成立，而不是来求切向基。
    """
    nx, ny, nz = canonical_normal(normal)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= _NORMAL_EPS:
        raise ValueError(f"tangent_basis 需要非零法向，收到：{normal!r}")
    nx, ny, nz = nx / length, ny / length, nz / length

    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax <= ay and ax <= az:
        helper = (1.0, 0.0, 0.0)
    elif ay <= az:
        helper = (0.0, 1.0, 0.0)
    else:
        helper = (0.0, 0.0, 1.0)

    ux = ny * helper[2] - nz * helper[1]
    uy = nz * helper[0] - nx * helper[2]
    uz = nx * helper[1] - ny * helper[0]
    length = math.sqrt(ux * ux + uy * uy + uz * uz)
    u = (ux / length, uy / length, uz / length)
    # 注意用**归一化后**的 u 做叉积：|n × u| = |n||u|sin∠ = 1 只在 u 是单位
    # 向量时成立。早先这里用的是未归一化的 ux/uy/uz，于是 |v| = |n × helper|，
    # 法向越"斜"（helper 轴与 n 越不垂直）错得越多——(1,2,3) 时 |v| = 0.964。
    v = (
        ny * u[2] - nz * u[1],
        nz * u[0] - nx * u[2],
        nx * u[1] - ny * u[0],
    )
    return u, v


def project(points, u, v, origin):
    """世界点 → 切平面二维坐标。"""
    projected = []
    for point in points:
        dx, dy, dz = point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]
        projected.append((dx * u[0] + dy * u[1] + dz * u[2],
                          dx * v[0] + dy * v[1] + dz * v[2]))
    return projected


def cross2(origin: Point2, a: Point2, b: Point2) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def _same_point(a: Point2, b: Point2, eps: float = 1e-12) -> bool:
    """两点在 ``eps`` 内重合（逐分量比较，不做开方）。"""
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def convex_hull_2d(points: list[Point2], eps: float = 1e-12) -> list[int]:
    """Andrew monotone chain，返回**下标**（逆时针、去重、去共线）。

    返回下标而不是点：调用方要保留"这个 2D 点原本是哪个 3D 顶点 / 哪个标签"，
    索引是最省事且不失精度的做法。

    去重对**所有**输入规模都生效（早先 ``len(points) <= 2`` 直接原样返回，于是
    两个重合的点会给出 ``[0, 1]``——一个零面积的"线段"）。退化情形逐条明确：

    - 空输入 → ``[]``
    - 全部重合 → 一个下标（只保留排序后的第一个）
    - 一条线段（含共线多点）→ 两个端点下标
    - 其余 → 逆时针凸包，不含共线的中间点
    """
    if not points:
        return []

    order = sorted(range(len(points)), key=lambda index: (points[index][0], points[index][1]))
    unique: list[int] = []
    for index in order:
        if unique and _same_point(points[unique[-1]], points[index], eps):
            continue
        unique.append(index)
    if len(unique) <= 2:
        return unique

    def build(sequence):
        hull: list[int] = []
        for index in sequence:
            while len(hull) >= 2 and cross2(
                points[hull[-2]], points[hull[-1]], points[index]
            ) <= eps:
                hull.pop()
            hull.append(index)
        return hull

    lower = build(unique)
    upper = build(reversed(unique))
    return lower[:-1] + upper[:-1]


def point_in_convex_2d(point: Point2, polygon: list[Point2], eps: float = 1e-9) -> bool:
    """点是否在逆时针凸多边形内（含边界）。"""
    count = len(polygon)
    if count == 0:
        return False
    if count == 1:
        return abs(point[0] - polygon[0][0]) <= eps and abs(point[1] - polygon[0][1]) <= eps
    if count == 2:
        return _point_on_segment(point, polygon[0], polygon[1], eps)

    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        if cross2(a, b, point) < -eps:
            return False
    return True


def _point_on_segment(point: Point2, a: Point2, b: Point2, eps: float) -> bool:
    if abs(cross2(a, b, point)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= point[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= point[1] <= max(a[1], b[1]) + eps
    )


def segment_intersection(a1: Point2, a2: Point2, b1: Point2, b2: Point2, eps: float = 1e-12):
    """两条线段的**严格**交点；平行、共线、只碰到端点时返回 ``None``。

    端点相碰不在这里处理：那种点会以"顶点落在对方内部"的形式被收录，来源也更
    清楚（是某个顶点，而不是某条边上的点）。
    """
    r = (a2[0] - a1[0], a2[1] - a1[1])
    s = (b2[0] - b1[0], b2[1] - b1[1])
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) <= eps:
        return None

    dx, dy = b1[0] - a1[0], b1[1] - a1[1]
    t = (dx * s[1] - dy * s[0]) / denominator
    u = (dx * r[1] - dy * r[0]) / denominator
    if not (-eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps):
        return None
    # 端点不算"严格相交"：让顶点路径去收它
    if min(t, 1.0 - t) <= 1e-9 or min(u, 1.0 - u) <= 1e-9:
        return None
    return (a1[0] + t * r[0], a1[1] + t * r[1])


def dedupe(items, eps: float = 1e-9):
    """按二维距离去重，保留首次出现的元素。``items`` 是 ``(point2, payload)``。"""
    kept = []
    for point, payload in items:
        duplicate = False
        for existing_point, _existing_payload in kept:
            if (abs(point[0] - existing_point[0]) <= eps
                    and abs(point[1] - existing_point[1]) <= eps):
                duplicate = True
                break
        if not duplicate:
            kept.append((point, payload))
    return kept


def polygon_area_2d(polygon: list[Point2]) -> float:
    """有向面积，逆时针为正。"""
    total = 0.0
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return 0.5 * total
