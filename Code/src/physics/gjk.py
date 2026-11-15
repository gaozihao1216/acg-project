# -*- coding: utf-8 -*-
"""GJK + EPA：任意两个凸体之间的统一碰撞查询。

分工
----
- **GJK**（Gilbert-Johnson-Keerthi）回答"离多远"：在闵可夫斯基差
  ``A ⊖ B = { p_A - p_B }`` 上迭代求"离原点最近的点"。原点在差集里
  ⇔ 两物体重叠。
- **EPA**（Expanding Polytope Algorithm）回答"陷多深、往哪推"：从 GJK 留下的
  单纯形出发，沿"离原点最近的那个面"的法向不断把多面体撑大，直到底面再也
  撑不动。此时的法向就是最小平移向量（MTV）方向，面距就是穿透深度。

为什么值得替换掉"按几何对写函数"
--------------------------------
原先每种组合都要一条窄相 + 一条 TOI，加柱/胶囊就要补 18 个函数，而且
曲面与多面体之间根本没有可用的有限 SAT 轴集合。GJK/EPA 只依赖支撑映射
（见 ``support.py``），于是"几何对"从代码维度变成数据维度：**新增几何体
不需要碰任何碰撞代码**。

与既有解析解的关系
------------------
球-球、球-盒、球-凸包、盒-盒、盒-凸包、凸包-凸包仍走各自的解析/SAT 实现
——它们更精确也更快，且结果已被大量测试钉住。GJK 只负责"含柱或胶囊"的
那 9 种组合，以及将来新增的几何体。

保守性
------
GJK 给出的是**精确**的分离距离（迭代收敛到 1e-9 米）；EPA 给出的是**近似**
最小平移量（支撑展开，收敛到 1e-4 米），对光滑曲面会略微低估，方向误差
在 1e-3 弧度量级——远小于求解器的 slop，不影响稳定性。
"""

from __future__ import annotations

import math

from ..mathutils import vec3
from . import patch as patch_module
from .contact import ContactPoint
from .support import TAG_LIMIT, feature_world_tagged, support_world

#: GJK 迭代上限。凸体问题通常 5–10 次收敛，40 次是防御性的兜底。
_MAX_GJK_ITERATIONS = 40

#: EPA 迭代上限。
_MAX_EPA_ITERATIONS = 48

#: 多面体顶点数上限，防止病态输入把内存吃穿。
_MAX_EPA_VERTICES = 128

#: GJK 收敛阈值（米）。低于它认为"已到最近点"。
#:
#: 取 1 µm 而不是机器精度：TOI 只用 gap/closing，1 µm 的间隙误差在 10 m/s 下
#: 折算成 1e-7 秒，远小于任何切分粒度；而放宽两个数量级能把迭代次数砍掉近半
#: （实测曲面对单次查询 0.64 ms → 0.35 ms）。
_GJK_TOL = 1e-7

#: EPA 收敛阈值（米）。支撑面与当前面距之差小于它即认为撑不动了。
_EPA_TOL = 2e-4

#: 判定两个顶点重合的阈值（米），用于单纯形去重。
_DUPLICATE_TOL = 1e-12


# ---------------------------------------------------------------------------
# 闵可夫斯基差与单纯形
# ---------------------------------------------------------------------------

def _minkowski(body_a, body_b, direction):
    """差集 ``A ⊖ B`` 沿 ``direction`` 的支撑点，同时返回两侧的见证点。"""
    point_a = support_world(body_a, direction)
    point_b = support_world(body_b, (-direction[0], -direction[1], -direction[2]))
    return (
        (point_a[0] - point_b[0], point_a[1] - point_b[1], point_a[2] - point_b[2]),
        point_a,
        point_b,
    )


def _solve_linear(matrix, rhs):
    """高斯消元（部分主元）解小型线性方程组；奇异时返回 ``None``。"""
    size = len(matrix)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]

    for column in range(size):
        pivot = column
        best = abs(augmented[column][column])
        for row in range(column + 1, size):
            value = abs(augmented[row][column])
            if value > best:
                best = value
                pivot = row
        if best < 1e-14:
            return None
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        inverse = 1.0 / augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] * inverse
            if factor == 0.0:
                continue
            for rest in range(column, size + 1):
                augmented[row][rest] -= factor * augmented[column][rest]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size]
        for column in range(row + 1, size):
            total -= augmented[row][column] * solution[column]
        solution[row] = total / augmented[row][row]
    return solution


def _distance_sq(point) -> float:
    """点到原点距离的平方（避免开方，比较时够用）。"""
    return point[0] * point[0] + point[1] * point[1] + point[2] * point[2]


def _segment_closest(points, indices):
    """线段上离原点最近的点。"""
    first, second = indices
    head = points[first]
    tail = points[second]
    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    dz = tail[2] - head[2]
    denominator = dx * dx + dy * dy + dz * dz
    if denominator <= 1e-24:
        return head, [first], [1.0]

    t = -(head[0] * dx + head[1] * dy + head[2] * dz) / denominator
    if t <= 0.0:
        return head, [first], [1.0]
    if t >= 1.0:
        return tail, [second], [1.0]
    closest = (head[0] + dx * t, head[1] + dy * t, head[2] + dz * t)
    return closest, [first, second], [1.0 - t, t]


def _triangle_closest(points, indices):
    """三角形上离原点最近的点：先在平面内解重心坐标，落在外面就退到棱上。"""
    first, second, third = indices
    head = points[first]
    edge1 = (
        points[second][0] - head[0],
        points[second][1] - head[1],
        points[second][2] - head[2],
    )
    edge2 = (
        points[third][0] - head[0],
        points[third][1] - head[1],
        points[third][2] - head[2],
    )

    aa = edge1[0] * edge1[0] + edge1[1] * edge1[1] + edge1[2] * edge1[2]
    ab = edge1[0] * edge2[0] + edge1[1] * edge2[1] + edge1[2] * edge2[2]
    bb = edge2[0] * edge2[0] + edge2[1] * edge2[1] + edge2[2] * edge2[2]
    pa = -(head[0] * edge1[0] + head[1] * edge1[1] + head[2] * edge1[2])
    pb = -(head[0] * edge2[0] + head[1] * edge2[1] + head[2] * edge2[2])

    determinant = aa * bb - ab * ab
    if abs(determinant) > 1e-18:
        weight_b = (pa * bb - pb * ab) / determinant
        weight_c = (aa * pb - ab * pa) / determinant
        weight_a = 1.0 - weight_b - weight_c
        if weight_a >= 0.0 and weight_b >= 0.0 and weight_c >= 0.0:
            closest = (
                head[0] * weight_a + points[second][0] * weight_b + points[third][0] * weight_c,
                head[1] * weight_a + points[second][1] * weight_b + points[third][1] * weight_c,
                head[2] * weight_a + points[second][2] * weight_b + points[third][2] * weight_c,
            )
            return closest, [first, second, third], [weight_a, weight_b, weight_c]

    best = None
    for pair in ((first, second), (second, third), (third, first)):
        candidate = _segment_closest(points, pair)
        squared = _distance_sq(candidate[0])
        if best is None or squared < best[0]:
            best = (squared, candidate)
    return best[1]


def _tetrahedron_closest(points, indices):
    """四面体上离原点最近的点；原点在内部时距离为 0（即两物体重叠）。"""
    first, second, third, fourth = indices
    head = points[first]

    axes = []
    for index in (second, third, fourth):
        axes.append(
            (
                points[index][0] - head[0],
                points[index][1] - head[1],
                points[index][2] - head[2],
            )
        )
    edge_a, edge_b, edge_c = axes

    matrix = [
        [
            edge_a[0] * edge_a[0] + edge_a[1] * edge_a[1] + edge_a[2] * edge_a[2],
            edge_a[0] * edge_b[0] + edge_a[1] * edge_b[1] + edge_a[2] * edge_b[2],
            edge_a[0] * edge_c[0] + edge_a[1] * edge_c[1] + edge_a[2] * edge_c[2],
        ],
        [
            edge_b[0] * edge_a[0] + edge_b[1] * edge_a[1] + edge_b[2] * edge_a[2],
            edge_b[0] * edge_b[0] + edge_b[1] * edge_b[1] + edge_b[2] * edge_b[2],
            edge_b[0] * edge_c[0] + edge_b[1] * edge_c[1] + edge_b[2] * edge_c[2],
        ],
        [
            edge_c[0] * edge_a[0] + edge_c[1] * edge_a[1] + edge_c[2] * edge_a[2],
            edge_c[0] * edge_b[0] + edge_c[1] * edge_b[1] + edge_c[2] * edge_b[2],
            edge_c[0] * edge_c[0] + edge_c[1] * edge_c[1] + edge_c[2] * edge_c[2],
        ],
    ]
    rhs = [
        -(head[0] * edge_a[0] + head[1] * edge_a[1] + head[2] * edge_a[2]),
        -(head[0] * edge_b[0] + head[1] * edge_b[1] + head[2] * edge_b[2]),
        -(head[0] * edge_c[0] + head[1] * edge_c[1] + head[2] * edge_c[2]),
    ]
    solution = _solve_linear(matrix, rhs)
    if solution is not None:
        weight_b, weight_c, weight_d = solution
        weight_a = 1.0 - weight_b - weight_c - weight_d
        if (
            weight_a >= 0.0
            and weight_b >= 0.0
            and weight_c >= 0.0
            and weight_d >= 0.0
        ):
            return (
                (0.0, 0.0, 0.0),
                [first, second, third, fourth],
                [weight_a, weight_b, weight_c, weight_d],
            )

    best = None
    for face in (
        (first, second, third),
        (first, second, fourth),
        (first, third, fourth),
        (second, third, fourth),
    ):
        candidate = _triangle_closest(points, face)
        squared = _distance_sq(candidate[0])
        if best is None or squared < best[0]:
            best = (squared, candidate)
    return best[1]


def closest_point_on_simplex(points):
    """单纯形（点集凸包）上离原点最近的点。

    按顶点数走分支：线段解一元二次、三角形解 2×2、四面体解 3×3；重心坐标
    出现负值时退到低维面（棱 / 三角形）上再找。

    为什么不"枚举所有子集"
    ----------------------
    那个写法更短，但每次调用要做十几次高斯消元。实测在含柱/胶囊的场景里，
    光这一项就吃掉单步 70% 的时间（单次 GJK 查询 2.2 ms）。分支写法每次只解
    一个小系统，快一个数量级。

    返回 ``(closest, indices, weights)``：``indices`` 是真正用到的点下标，
    ``weights`` 是对应的重心坐标（和为 1），用于回算两侧见证点。
    """
    count = len(points)
    if count <= 1:
        return points[0], [0], [1.0]
    if count == 2:
        closest, indices, weights = _segment_closest(points, (0, 1))
    elif count == 3:
        closest, indices, weights = _triangle_closest(points, (0, 1, 2))
    else:
        closest, indices, weights = _tetrahedron_closest(points, (0, 1, 2, 3))

    # 丢掉权重为 0 的顶点：否则单纯形不会收缩，迭代会白白多跑几轮
    kept_indices = []
    kept_weights = []
    for index, weight in zip(indices, weights):
        if weight > 0.0:
            kept_indices.append(index)
            kept_weights.append(weight)
    if not kept_indices:
        return points[indices[0]], [indices[0]], [1.0]

    total = sum(kept_weights)
    if abs(total - 1.0) > 1e-12:
        kept_weights = [weight / total for weight in kept_weights]
    return closest, kept_indices, kept_weights


# ---------------------------------------------------------------------------
# GJK
# ---------------------------------------------------------------------------

def _dedupe(items: list[tuple], fresh: tuple) -> list[tuple]:
    """把一个点加入单纯形；若已存在则原样返回（用于终止判断）。"""
    for item in items:
        if (
            abs(item[0][0] - fresh[0][0]) <= _DUPLICATE_TOL
            and abs(item[0][1] - fresh[0][1]) <= _DUPLICATE_TOL
            and abs(item[0][2] - fresh[0][2]) <= _DUPLICATE_TOL
        ):
            return items
    return items + [fresh]


def _witnesses(simplex, indices, weights):
    """由重心坐标回算两侧的见证点（A 上一点、B 上一点）。"""
    x = y = z = 0.0
    u = v = w = 0.0
    for weight, index in zip(weights, indices):
        point_a = simplex[index][1]
        point_b = simplex[index][2]
        x += weight * point_a[0]
        y += weight * point_a[1]
        z += weight * point_a[2]
        u += weight * point_b[0]
        v += weight * point_b[1]
        w += weight * point_b[2]
    return (x, y, z), (u, v, w)


def distance(body_a, body_b):
    """两凸体的最近距离（GJK）。

    返回
    ----
    ``(gap, point_a, point_b, simplex)``

    - ``gap > 0``：分离，``point_a``/``point_b`` 是各自的最近点
    - ``gap == 0``：重叠，``simplex`` 是包围原点的单纯形，交给 EPA
    """
    initial = (
        body_b.position[0] - body_a.position[0],
        body_b.position[1] - body_a.position[1],
        body_b.position[2] - body_a.position[2],
    )
    if initial[0] * initial[0] + initial[1] * initial[1] + initial[2] * initial[2] <= 1e-24:
        initial = (1.0, 0.0, 0.0)

    # 差集在"B 相对 A 的反方向"一侧，所以先朝 -initial 取支撑
    simplex = [_minkowski(body_a, body_b, (-initial[0], -initial[1], -initial[2]))]

    def result(gap_value):
        """用当前单纯形打包返回值。"""
        points = [item[0] for item in simplex]
        closest_point, kept, kept_weights = closest_point_on_simplex(points)
        point_a, point_b = _witnesses(simplex, kept, kept_weights)
        return (
            gap_value,
            point_a,
            point_b,
            [simplex[index] for index in kept],
        )

    for _ in range(_MAX_GJK_ITERATIONS):
        points = [item[0] for item in simplex]
        closest, indices, weights = closest_point_on_simplex(points)
        gap = math.sqrt(
            closest[0] * closest[0] + closest[1] * closest[1] + closest[2] * closest[2]
        )

        if gap <= _GJK_TOL:
            # 原点被单纯形包住 ⇒ 重叠
            return result(0.0)

        # 朝原点方向继续搜索
        direction = (-closest[0] / gap, -closest[1] / gap, -closest[2] / gap)
        new_vertex = _minkowski(body_a, body_b, direction)

        # 支撑面已经"够不着"原点：这就是最终距离
        support_projection = (
            new_vertex[0][0] * direction[0]
            + new_vertex[0][1] * direction[1]
            + new_vertex[0][2] * direction[2]
        )
        if support_projection + gap <= _GJK_TOL:
            return result(gap)

        reduced = _dedupe([simplex[index] for index in indices], new_vertex)
        if len(reduced) == len(indices):
            # 新点与已有顶点重合：无法再靠近
            return result(gap)
        simplex = reduced

    # 迭代用尽：用当前最近点作为答案（保守，不会漏接触）
    points = [item[0] for item in simplex]
    closest, _indices, _weights = closest_point_on_simplex(points)
    return result(
        math.sqrt(closest[0] * closest[0] + closest[1] * closest[1] + closest[2] * closest[2])
    )


# ---------------------------------------------------------------------------
# EPA
# ---------------------------------------------------------------------------

def _face(normal, distance, i, j, k):
    """构造一个面记录（法向朝外、距离是原点到该面的距离）。"""
    return (normal, distance, i, j, k)


def _make_face(vertices, i, j, k, reference):
    """由三个顶点构造面，法向朝多面体外侧。

    "外侧"的判据是**背离多面体内部的参考点**，而不是"背离原点"。
    这个区别在退化情形下是致命的：两个球重叠时原点可能正好落在初始四面体
    的棱上，那一圈面的平面**过原点**，`dot(n, p) == 0`，靠原点的符号根本
    定不出朝外还是朝里，法向随机翻——EPA 随即朝错误方向撑开，深度算成
    几百毫米。
    """
    point_i = vertices[i]
    edge1 = vec3.sub(vertices[j], point_i)
    edge2 = vec3.sub(vertices[k], point_i)
    normal = vec3.cross(edge1, edge2)
    length = vec3.length(normal)
    if length <= 1e-18:
        return None
    normal = (normal[0] / length, normal[1] / length, normal[2] / length)
    if vec3.dot(normal, vec3.sub(point_i, reference)) < 0.0:
        normal = (-normal[0], -normal[1], -normal[2])
    return _face(normal, vec3.dot(normal, point_i), i, j, k)


def _initial_tetrahedron(body_a, body_b, simplex):
    """把 GJK 的单纯形补成一个四面体，作为 EPA 的初始多面体。

    单纯形通常已经是四面体（原点被包住时 GJK 会保留 4 个顶点）；
    只剩 1–3 个顶点时（原点正好落在边或面上），沿正交方向补支撑点。
    """
    vertices = []
    for item in simplex:
        point = item[0]
        duplicate = False
        for existing in vertices:
            if (
                abs(existing[0] - point[0]) <= _DUPLICATE_TOL
                and abs(existing[1] - point[1]) <= _DUPLICATE_TOL
                and abs(existing[2] - point[2]) <= _DUPLICATE_TOL
            ):
                duplicate = True
                break
        if not duplicate:
            vertices.append(point)

    # 兜底方向：坐标轴 + 对角线，够撑起任意退化情形
    fallback_directions = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
    )

    candidates = list(vertices)
    for direction in fallback_directions:
        point = _minkowski(body_a, body_b, direction)[0]
        duplicate = False
        for existing in candidates:
            if (
                abs(existing[0] - point[0]) <= _DUPLICATE_TOL
                and abs(existing[1] - point[1]) <= _DUPLICATE_TOL
                and abs(existing[2] - point[2]) <= _DUPLICATE_TOL
            ):
                duplicate = True
                break
        if not duplicate:
            candidates.append(point)

    if not candidates:
        return []

    # 贪心：p0 离原点最远 → p1 离 p0 最远 → p2 三角形面积最大 → p3 体积最大
    p0 = max(candidates, key=lambda point: vec3.length_squared(point))

    p1 = max(candidates, key=lambda point: vec3.length_squared(vec3.sub(point, p0)))

    best_area = -1.0
    p2 = None
    for point in candidates:
        area = vec3.length_squared(vec3.cross(vec3.sub(p1, p0), vec3.sub(point, p0)))
        if area > best_area:
            best_area = area
            p2 = point
    if p2 is None or best_area <= 1e-24:
        return []

    best_volume = -1.0
    p3 = None
    normal = vec3.cross(vec3.sub(p1, p0), vec3.sub(p2, p0))
    for point in candidates:
        volume = abs(vec3.dot(normal, vec3.sub(point, p0)))
        if volume > best_volume:
            best_volume = volume
            p3 = point
    if p3 is None or best_volume <= 1e-24:
        return []

    return [p0, p1, p2, p3]


def penetration(body_a, body_b, simplex):
    """EPA：返回 ``(normal_a_to_b, depth)``；两体重叠时才有意义。

    法向是**从 A 指向 B** 的单位向量，深度是最小平移量（米）。
    """
    vertices = _initial_tetrahedron(body_a, body_b, simplex)
    if len(vertices) < 4:
        return None

    # 内部参考点：顶点均值一定在凸多面体内部，用它定"哪边是外"
    reference = (
        sum(point[0] for point in vertices) / len(vertices),
        sum(point[1] for point in vertices) / len(vertices),
        sum(point[2] for point in vertices) / len(vertices),
    )

    faces = []
    for triple in ((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)):
        face = _make_face(vertices, *triple, reference=reference)
        if face is not None:
            faces.append(face)
    if not faces:
        return None

    best_normal = faces[0][0]
    best_distance = faces[0][1]

    for _ in range(_MAX_EPA_ITERATIONS):
        # 取离原点最近的面（有符号：原点跑到面外时会是负值，正好优先修正）
        closest_index = 0
        closest_distance = faces[0][1]
        for index in range(1, len(faces)):
            if faces[index][1] < closest_distance:
                closest_distance = faces[index][1]
                closest_index = index

        normal = faces[closest_index][0]
        support_point = _minkowski(body_a, body_b, normal)[0]
        support_distance = vec3.dot(support_point, normal)

        best_normal = normal
        best_distance = min(closest_distance, support_distance)

        if support_distance - closest_distance <= _EPA_TOL:
            # 撑不动了：这个面就是最近边界。取内接面与支撑面的中点——
            # 真实边界夹在两者之间，取中点把一阶误差抵消掉
            best_distance = 0.5 * (support_distance + closest_distance)
            break

        if len(vertices) >= _MAX_EPA_VERTICES:
            break

        # 只有被新顶点"看到"的面才删，其余保留
        new_index = len(vertices)
        vertices.append(support_point)

        kept = []
        removed = []
        for face in faces:
            if vec3.dot(face[0], vec3.sub(support_point, vertices[face[2]])) > 1e-12:
                removed.append(face)
            else:
                kept.append(face)

        if not removed:
            # 数值病态：新点没有落在任何面的外侧，直接收敛
            best_normal = normal
            best_distance = closest_distance
            break

        # 收集地平线：只被一个被删面拥有的无向边
        edge_count: dict[tuple[int, int], int] = {}
        for face in removed:
            for edge in ((face[2], face[3]), (face[3], face[4]), (face[4], face[2])):
                key = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
                edge_count[key] = edge_count.get(key, 0) + 1

        faces = kept
        for (first, second), count in edge_count.items():
            if count != 1:
                continue
            face = _make_face(vertices, first, second, new_index, reference=reference)
            if face is not None:
                faces.append(face)

        if not faces:
            break

    if best_distance < 0.0:
        best_distance = 0.0
    return _refine_normal(body_a, body_b, best_normal)


def _extent(body_a, body_b, normal) -> float:
    """沿 ``normal`` 方向的重叠量 ``h_W(n) = (w_A - w_B)·n``。

    这是差集 ``W = A ⊖ B`` 的支撑函数取值。**最小平移量 = min h_W(n)**，
    与"原点到 W 边界的距离"是同一件事——这正是 EPA 在逼近的那个量。
    """
    difference, _point_a, _point_b = _minkowski(body_a, body_b, normal)
    return vec3.dot(difference, normal)


def _refine_normal(body_a, body_b, normal):
    """把 EPA 给的方向校准到真正的"最小平移"方向。

    为什么 EPA 之后还要这一步
    ------------------------
    EPA 的面距是多面体（内接于真实差集）的**下界**，方向也只精确到"最近的那个
    三角面"。两个球重叠时，EPA 常要几十次迭代才能把方向磨到 1° 以内，
    而这里只需要几次支撑查询。

    依据
    ----
    最优法向处，两侧支撑点之差必与法向平行：``w_A(n) - w_B(-n) = depth · n``。
    于是 ``n ← normalize(w_A - w_B)`` 是**不动点迭代**，每步用 ``h_W`` 重新打分，
    **只接受变好的**——这样"面贴面"（最优解是一整片平台）的情形不会被从面法向
    上拽走：任何倾斜都会让 ``h_W`` 变大，直接被拒。

    为什么要多个起点
    ----------------
    这个迭代有两个不动点：真正的最小平移方向，以及它的**反方向**（那是最大值，
    不是最小值）。球-球重叠时 EPA 若落在反侧，单起点会牢牢卡在错误的一侧
    （实测法向偏差 148°）。所以这里给几个起点、先粗评一次、只精修最好的两个。
    """
    seeds = [normal, (-normal[0], -normal[1], -normal[2])]

    offset = (
        body_b.position[0] - body_a.position[0],
        body_b.position[1] - body_a.position[1],
        body_b.position[2] - body_a.position[2],
    )
    offset_length = vec3.length(offset)
    if offset_length > 1e-12:
        center = (
            offset[0] / offset_length,
            offset[1] / offset_length,
            offset[2] / offset_length,
        )
        seeds.append(center)
        seeds.append((-center[0], -center[1], -center[2]))

    # 先粗评，只精修最有希望的两个起点
    scored = sorted(
        ((_extent(body_a, body_b, seed), seed) for seed in seeds),
        key=lambda item: item[0],
    )

    best_depth, best_normal = scored[0]
    for _depth, seed in scored[:2]:
        direction = seed
        for _step in range(10):
            difference, _point_a, _point_b = _minkowski(body_a, body_b, direction)
            length = vec3.length(difference)
            if length <= 1e-12:
                break
            candidate = (
                difference[0] / length,
                difference[1] / length,
                difference[2] / length,
            )
            if vec3.dot(candidate, direction) > 1.0 - 1e-12:
                break   # 已是不动点
            depth = _extent(body_a, body_b, candidate)
            if depth < best_depth:
                best_depth = depth
                best_normal = candidate
            direction = candidate

    if best_depth < 0.0:
        best_depth = 0.0
    return (best_normal, best_depth)


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------

def collide(body_a, body_b) -> list[ContactPoint]:
    """任意两个凸体之间的接触点列表（法向 A→B），无接触时返回空列表。"""
    gap, point_a, point_b, simplex = distance(body_a, body_b)
    if gap > _GJK_TOL:
        return []

    result = penetration(body_a, body_b, simplex)
    if result is None:
        return []
    normal, depth = result
    if depth <= 0.0:
        return []

    return _build_manifold(body_a, body_b, normal, depth, (point_a, point_b))


def _build_manifold(body_a, body_b, normal, depth, witnesses=None) -> list[ContactPoint]:
    """由法向 + 深度生成接触点：**在切平面上求两侧支撑集的交集**。

    为什么不再用"最近采样点配对"
    ----------------------------
    旧实现拿两侧的 4 点采样逐点配对、只在法向上判深度，完全不看切向覆盖。
    0.6 m 宽的圆柱搭在盒子边上 0.2 m 时，四个 45°/135°/225°/315° 采样点里有两个
    在盒子投影外 1.5 m、两个在投影左侧，却被当成真实接触点。更糟的是这四点关于
    圆柱轴对称，净力矩为零——症状不是"翻倒"，而是**该倒的不倒**。

    现在的流程（P0-1）：

    1. 两侧各取高密度支撑 patch（``patch.contact_patch_world``，圆盘最多 64 边）；
    2. 投影到法向的切平面，求两个二维凸点集的交集（``poly2d.convex_intersection``）；
    3. 按两侧**各自拟合出的表面平面**逐点算间隙，把分离点裁掉；
    4. 剩下的交集稀疏到 ≤4 点，再抬成两侧锚点、取中点。

    为什么第 3 步必须在稀疏之前：端面选取留了约 3° 容差，倾斜端面会把整片倾斜
    圆盘交给这一层；先缩到 4 点再用"垂直于法向的平均平面"算深度，会给高侧本已
    分离的点建立约束（实测 0.3 m 圆柱倾斜 2°：四点都报 9.70 mm，真实是
    20.16 mm / −0.77 mm）。细节见 ``patch.contact_anchors``。

    空交集时不回退到旧采样
    ----------------------
    GJK 认为接触、但离散后的两个多边形切向不相交（圆盘内接多边形比真实圆盘小
    一圈）时，退化成**单点**。这个点用 ``distance()`` 留下的**成对见证点**的中点：
    ``point_a − point_b ≈ 0``，两点对应同一个位置，所以不会像"各自查支撑点再配对"
    那样把力臂甩到几十厘米外（实测近边缘搭接、盒心 x=1.49999 时，旧回退交换 A/B
    给出 ``y = −0.250 m``，见证点给出 ``y ≈ +0.002 m``）。

    这个界要收窄：见证点只保证**位置落在接触处附近**。重叠时 GJK 的距离见证点可以
    落在交叠体积内部，不保证是沿最终 EPA 法向的一对表面锚点；也正因为如此，A/B
    对调的完整一致性**没有**被这条兜住——近边缘案例反向仍是 1 点、深度 ~51 µm，
    正向 2 点、~10 µm。A/B 对称性要靠 patch 求交那条主路径，回退只求"不偏"。

    特征 ID
    -------
    ``feature_id = tag_a * TAG_LIMIT + tag_b``，标签仍来自
    ``support.feature_world_tagged``（0..7 的体坐标槽位），分配方式是"两侧各按
    极角排序后逐个对齐"：批内唯一由构造保证，且微扰下不换人。
    """
    opposite = (-normal[0], -normal[1], -normal[2])

    patch_a = patch_module.contact_patch_world(body_a, normal)
    patch_b = patch_module.contact_patch_world(body_b, opposite)
    overlap = patch_module.overlap(patch_a, patch_b, normal)
    anchors = patch_module.contact_anchors(overlap, patch_a, patch_b, normal)

    if anchors:
        positions = [anchor.position for anchor in anchors]
        tags_a = patch_module.match_tags(
            positions, feature_world_tagged(body_a, normal), normal
        )
        tags_b = patch_module.match_tags(
            positions, feature_world_tagged(body_b, opposite), normal
        )
        return [
            ContactPoint(
                position=anchor.position,
                normal=normal,
                penetration=anchor.penetration,
                feature_id=tag_a * TAG_LIMIT + tag_b,
            )
            for anchor, tag_a, tag_b in zip(anchors, tags_a, tags_b)
        ]

    # 回退：优先用 GJK 留下的成对见证点（两侧同一点），它一定在接触位置上
    patch_module.note_fallback(overlap.empty_reason or "separated")
    if witnesses is not None:
        point_a, point_b = witnesses
        position = (
            0.5 * (point_a[0] + point_b[0]),
            0.5 * (point_a[1] + point_b[1]),
            0.5 * (point_a[2] + point_b[2]),
        )
    else:
        support_a = support_world(body_a, normal)
        support_b = support_world(body_b, opposite)
        position = (
            0.5 * (support_a[0] + support_b[0]),
            0.5 * (support_a[1] + support_b[1]),
            0.5 * (support_a[2] + support_b[2]),
        )
    return [
        ContactPoint(position=position, normal=normal, penetration=depth),
    ]
