# -*- coding: utf-8 -*-
"""二维凸多边形工具：接触平面的切向基、凸包、包含测试与求交。

为什么单列一个模块
------------------
接触流形要在**切平面**上判断"两侧支撑区域是否真的重叠"。这类计算纯粹是二维
几何，与刚体、几何体都不相干。放在 ``mathutils`` 里可以让 ``support``（给支撑
patch 排序）与将来的切平面求交共用，又不产生 physics 内部循环依赖。

当前使用方只有 ``physics.support._patch_from_points``（投影 + 凸包排序）；
点包含、线段求交、去重、面积这几件是 P0-1 第三步（退化 patch 求交）预置的
工具，**尚未接线**，只被 ``tests/test_poly2d.py`` 覆盖。别误以为窄相已经在用。

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


# ---------------------------------------------------------------------------
# 凸点集的交集：点 / 线段 / 多边形统一入口
# ---------------------------------------------------------------------------

def convex_intersection(
    a: list[Point2], b: list[Point2], eps: float = 1e-9
) -> list[Point2]:
    """两个**凸**点集的交集，返回规范化的点列（``[]`` 表示不相交）。

    为什么一个函数吃掉九种组合
    --------------------------
    支撑 patch 在投影后退化成什么，事先并不知道：圆盘侧视是一条线段，锥顶是一个
    点，两者相遇又是另一个退化。调用方要是自己按"点数"分派，就会出现"某几种组合
    没人写"的漏洞——P0-1 里"四个接触点全在接触区外"正是这种漏洞的产物。

    所以这里统一契约：**输入是任意 1 / 2 / ≥3 个共面点，输出是同一套规范形式**
    （1 点、2 点（线段）、≥3 点（逆时针凸多边形）），空交集返回 ``[]``。

    退化组合与处理方式：

    | A | B | 处理 |
    |---|---|---|
    | 点 | 任意 | 点是否属于 B（1 点相等 / 线段含点 / 多边形含点） |
    | 线段 | 线段 | 严格相交取交点；共线取重叠区间；端点相碰取该点 |
    | 线段 | 多边形 | 按每条边的半平面做参数区间裁剪（精确，非采样） |
    | 多边形 | 多边形 | Sutherland–Hodgman 逐边裁剪 |

    两侧都是凸集，所以裁剪是**精确**的：不需要"顶点包含 + 边求交 + 再求凸包"那套
    通用多边形流程（那套只在凹多边形下才必要），也不会因为"先裁谁"而得到不同结果。

    容差 ``eps`` 是**长度**量纲（米），内部把所有叉积都除以边长换成有符号距离，
    于是"恰好共边"不会因为边长不同而时对时错。
    """
    left = _canonical_points(a, eps)
    right = _canonical_points(b, eps)
    if not left or not right:
        return []

    if len(left) == 1 or len(right) == 1:
        point = left[0] if len(left) == 1 else right[0]
        other = right if len(left) == 1 else left
        return [point] if _point_in_set(point, other, eps) else []

    if len(left) == 2 and len(right) == 2:
        return _segment_overlap(left, right, eps)
    if len(left) == 2:
        return _clip_segment(left, right, eps)
    if len(right) == 2:
        return _clip_segment(right, left, eps)
    return _clip_polygon(left, right, eps)


def _canonical_points(points: list[Point2], eps: float) -> list[Point2]:
    """去重、去共线，按 1 点 / 2 点 / 逆时针多边形三种形式返回。

    已经是凸多边形（无论顺逆时针）且没有相邻重复点时走快路径：只做一次 O(n) 的
    朝向检查与反转，省掉排序 + monotone chain。接触流形的输入正是这种形式
    （圆盘采样点、盒面顶点），单次流形构造里这一段占了可观的常数。
    """
    if not points:
        return []
    if len(points) >= 3:
        orientation = _convex_orientation(points, eps)
        if orientation > 0:
            return list(points)
        if orientation < 0:
            return list(reversed(points))
    return [points[index] for index in convex_hull_2d(points, eps)]


def _convex_orientation(points: list[Point2], eps: float = 1e-12) -> int:
    """``+1`` 逆时针凸、``-1`` 顺时针凸、``0`` 有凹点 / 相邻重合 / 点数不足。"""
    count = len(points)
    if count < 3:
        return 0
    sign = 0
    for index in range(count):
        first = points[index]
        second = points[(index + 1) % count]
        third = points[(index + 2) % count]
        if _same_point(first, second, eps) or _same_point(second, third, eps):
            return 0
        cross = (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])
        if cross > eps:
            current = 1
        elif cross < -eps:
            current = -1
        else:
            continue   # 共线三点不决定朝向
        if sign == 0:
            sign = current
        elif sign != current:
            return 0
    return sign


def _signed_distance(a: Point2, b: Point2, point: Point2) -> float:
    """``point`` 到有向边 ``a→b`` 左侧为正的距离（逆时针多边形内部为正）。"""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-15:
        return 0.0
    return cross2(a, b, point) / length


def _point_in_set(point: Point2, points: list[Point2], eps: float) -> bool:
    """点是否属于 1 点 / 线段 / 凸多边形形式的点集。"""
    if len(points) == 1:
        return _same_point(point, points[0], eps)
    if len(points) == 2:
        return _point_on_segment(point, points[0], points[1], eps)
    return point_in_convex_2d(point, points, eps)


def _segment_overlap(a: list[Point2], b: list[Point2], eps: float) -> list[Point2]:
    """线段 ∩ 线段：不平行取交点（含端点相碰），共线取重叠区间。"""
    dx1 = a[1][0] - a[0][0]
    dy1 = a[1][1] - a[0][1]
    dx2 = b[1][0] - b[0][0]
    dy2 = b[1][1] - b[0][1]
    length1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    length2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
    if length1 <= eps or length2 <= eps:
        return []

    cross = dx1 * dy2 - dy1 * dx2
    if abs(cross) > eps * length1 * length2:
        point = segment_intersection(a[0], a[1], b[0], b[1], eps)
        if point is not None:
            return [point]
        # 严格相交之外只剩"端点落在对方上"这一种接触
        hits = [p for p in (a[0], a[1]) if _point_on_segment(p, b[0], b[1], eps)]
        hits += [p for p in (b[0], b[1]) if _point_on_segment(p, a[0], a[1], eps)]
        return _canonical_points(hits, eps) if hits else []

    # 平行：不共线就没有交集
    if abs(_signed_distance(a[0], a[1], b[0])) > eps:
        return []

    # 共线：把 b 的两端投影到 a 的参数轴上，取重叠区间
    denominator = length1 * length1
    parameters = []
    for point in b:
        t = ((point[0] - a[0][0]) * dx1 + (point[1] - a[0][1]) * dy1) / denominator
        parameters.append(t)
    low = max(0.0, min(parameters))
    high = min(1.0, max(parameters))
    if high - low < -eps / length1:
        return []
    if high - low <= eps / length1:
        t = 0.5 * (low + high)
        return [(a[0][0] + t * dx1, a[0][1] + t * dy1)]
    return [
        (a[0][0] + low * dx1, a[0][1] + low * dy1),
        (a[0][0] + high * dx1, a[0][1] + high * dy1),
    ]


def _clip_segment(
    segment: list[Point2], polygon: list[Point2], eps: float
) -> list[Point2]:
    """线段 ∩ 凸多边形：把线段参数化成 ``p0 + t·d``，逐边收紧 ``t`` 的区间。"""
    p0, p1 = segment
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if math.hypot(dx, dy) <= eps:
        return [p0] if _point_in_set(p0, polygon, eps) else []

    low, high = 0.0, 1.0
    count = len(polygon)
    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        first = _signed_distance(a, b, p0)
        second = _signed_distance(a, b, p1)
        slope = second - first
        if slope > 1e-15:
            low = max(low, -first / slope)
        elif slope < -1e-15:
            high = min(high, -first / slope)
        elif first < -eps:
            # 整条线段都在这条边的外侧
            return []
        if low > high + 1e-12:
            return []

    if high - low <= 1e-12:
        t = 0.5 * (low + high)
        return [(p0[0] + t * dx, p0[1] + t * dy)]
    return [
        (p0[0] + low * dx, p0[1] + low * dy),
        (p0[0] + high * dx, p0[1] + high * dy),
    ]


def _clip_polygon(
    subject: list[Point2], clipper: list[Point2], eps: float
) -> list[Point2]:
    """凸多边形 ∩ 凸多边形：用 ``clipper`` 的每条边裁 ``subject``（Sutherland–Hodgman）。"""
    output = list(subject)
    count = len(clipper)
    for index in range(count):
        if not output:
            return []
        a = clipper[index]
        b = clipper[(index + 1) % count]
        output = _clip_edge(output, a, b, eps)
    if not output:
        return []
    return _canonical_points(output, eps)


def _clip_edge(
    points: list[Point2], a: Point2, b: Point2, eps: float
) -> list[Point2]:
    """保留有向边 ``a→b`` 左侧（含边界）的部分，跨边处插入交点。

    每条边的长度与每个点的有符号距离都**只算一次**：这是接触流形每帧都要走的
    热路径（一个 55 边形乘 4 条盒边），逐个点现算 ``hypot`` 会让单次流形构造
    从 0.1 ms 涨到 0.5 ms。
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-15:
        return list(points)

    count = len(points)
    distances = [cross2(a, b, points[index]) / length for index in range(count)]

    output: list[Point2] = []
    for index in range(count):
        current = points[index]
        following = points[(index + 1) % count]
        current_distance = distances[index]
        following_distance = distances[(index + 1) % count]
        if current_distance >= -eps:
            output.append(current)
        if (current_distance > eps and following_distance < -eps) or (
            current_distance < -eps and following_distance > eps
        ):
            t = current_distance / (current_distance - following_distance)
            output.append(
                (
                    current[0] + t * (following[0] - current[0]),
                    current[1] + t * (following[1] - current[1]),
                )
            )
    return output
