# -*- coding: utf-8 -*-
"""网格与点云生成工具：给质量属性模块提供可解析验证的输入。

这些函数的唯一目的是**构造有已知解析答案的测试对象**。它们不是渲染用的
模型加载器——真实的网格加载（OBJ/STL）是另一件事，等做碰撞代理时再写。

为什么所有网格都要手写缠绕方向
------------------------------
散度定理依赖**有符号体积**。面朝外 ⇒ 体积为正；缠绕方向不一致 ⇒ 正负抵消，
体积接近零，惯量完全是垃圾数据。而错误的惯量不会报错，只会让物体转得"怪"。

所以这里每个面都按"从外部看逆时针"手工排好，并在测试里用
"体积 == 解析体积"来反向确认缠绕没写反。

凸多面体的三角形面用扇形三角化即可——凸包的面上任意扇形都落在面内，
不会产生退化三角形。
"""

from __future__ import annotations

import math
from typing import Sequence

#: 通用的 8 个顶点下标约定（与 ``box_mesh`` 内部一致）
#: 0:(x0,y0,z0) 1:(x1,y0,z0) 2:(x1,y1,z0) 3:(x0,y1,z0)
#: 4:(x0,y0,z1) 5:(x1,y0,z1) 6:(x1,y1,z1) 7:(x0,y1,z1)


def box_mesh(
    size: Sequence[float],
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """生成轴对齐长方体的闭合三角网格（12 个三角形，法向朝外）。

    参数
    ----
    size   : 三个方向的长度
    center : 几何中心

    返回
    ----
    (vertices, faces)
    """
    hx, hy, hz = 0.5 * size[0], 0.5 * size[1], 0.5 * size[2]
    cx, cy, cz = center

    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz

    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]

    # 每个面的两个三角形都按"从外部看逆时针"排列
    faces = [
        (0, 3, 2), (0, 2, 1),   # -z
        (4, 5, 6), (4, 6, 7),   # +z
        (0, 1, 5), (0, 5, 4),   # -y
        (3, 7, 6), (3, 6, 2),   # +y
        (0, 4, 7), (0, 7, 3),   # -x
        (1, 2, 6), (1, 6, 5),   # +x
    ]
    return vertices, faces


def shell_mesh(
    outer: Sequence[float],
    inner: Sequence[float],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """生成空心长方体（外层盒 + 反向的内层盒）的闭合网格。

    内层盒的面**反向缠绕**，于是它的有符号体积为负，在散度定理里自动被减掉。
    这正是"空腔不需要特殊代码"的演示——同一套累加逻辑同时处理了实体与空腔。

    参数
    ----
    outer : 外层尺寸
    inner : 内腔尺寸（必须严格小于 ``outer``）
    """
    for axis in range(3):
        if inner[axis] >= outer[axis]:
            raise ValueError("shell_mesh 的内腔必须严格小于外形")

    vertices, faces = box_mesh(outer)
    offset = len(vertices)

    inner_vertices, inner_faces = box_mesh(inner)
    vertices.extend(inner_vertices)

    # 翻转每个三角形的缠绕方向：交换后两个顶点
    for a, b, c in inner_faces:
        faces.append((a + offset, c + offset, b + offset))

    return vertices, faces


def prism_mesh(
    polygon: Sequence[Sequence[float]],
    height: float,
    z0: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """把二维多边形（逆时针）沿 z 方向拉伸成闭合棱柱。

    参数
    ----
    polygon : 逆时针多边形顶点 ``[(x, y), ...]``，不得自交
    height  : 拉伸高度
    z0      : 底面高度

    返回
    ----
    (vertices, faces)

    说明
    ----
    用棱柱而不是拼盒子，是因为**拼盒子会产生内部面**（两块盒子共用面上
    有两个朝向相反的面，有符号体积互相抵消，虽然结果仍正确，但不再是
    封闭流形）。棱柱是真正的封闭流形，更适合验证"封闭网格"这条路径。

    L 形、T 形、带缺口的多边形都可以直接喂进来——这正是"任意形状"的入口。
    """
    count = len(polygon)
    if count < 3:
        raise ValueError("prism_mesh 的多边形至少需要 3 个顶点")

    z1 = z0 + height

    vertices: list[tuple[float, float, float]] = []
    for x, y in polygon:
        vertices.append((float(x), float(y), z0))
    for x, y in polygon:
        vertices.append((float(x), float(y), z1))

    faces: list[tuple[int, int, int]] = []

    # 底面（法向 -z）：逆时针多边形反转即得到 -z 朝向
    for i in range(1, count - 1):
        faces.append((0, i + 1, i))

    # 顶面（法向 +z）
    for i in range(1, count - 1):
        faces.append((count, count + i, count + i + 1))

    # 侧面：每条边一个四边形，拆成两个三角形
    for i in range(count):
        j = (i + 1) % count
        bottom_i, bottom_j = i, j
        top_i, top_j = count + i, count + j
        faces.append((bottom_i, bottom_j, top_j))
        faces.append((bottom_i, top_j, top_i))

    return vertices, faces


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """鞋带公式求多边形面积（逆时针为正），用于独立校验体积。"""
    total = 0.0
    count = len(polygon)
    for i in range(count):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % count]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


# ---------------------------------------------------------------------------
# 点云采样
# ---------------------------------------------------------------------------

def lattice_points_in_box(
    size: Sequence[float],
    cells_per_axis: int,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[tuple[float, float, float]]:
    """在长方体内部按**均匀立方点阵**采样，每个格子取中心。

    这条路径的特殊之处：它是唯一能让修正公式**精确到机器精度**的采样方式。
    因为中点法则对二次多项式（惯量正是坐标的二次泛函）在加回
    ``h²/12`` 项后没有截断误差。

    ``cells_per_axis`` 为 n 时点数 ``N = n³``，格子边长 ``h = size / n``。
    """
    n = int(cells_per_axis)
    if n < 1:
        raise ValueError("cells_per_axis 必须 ≥ 1")

    h = [size[k] / n for k in range(3)]
    origin = [center[k] - 0.5 * size[k] for k in range(3)]

    points = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                points.append(
                    (
                        origin[0] + (ix + 0.5) * h[0],
                        origin[1] + (iy + 0.5) * h[1],
                        origin[2] + (iz + 0.5) * h[2],
                    )
                )
    return points


def lattice_points_in_shell(
    outer: Sequence[float],
    inner: Sequence[float],
    cells_per_axis: int,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[tuple[float, float, float]]:
    """在空心长方体的**壳体区域**内按乱序点阵采样（跳过内腔）。

    这是"点云能表达空腔"的直接演示：内腔里没有点，它就不贡献质量。
    """
    n = int(cells_per_axis)
    h = [outer[k] / n for k in range(3)]
    origin = [center[k] - 0.5 * outer[k] for k in range(3)]
    half_inner = [0.5 * inner[k] for k in range(3)]

    points = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                x = origin[0] + (ix + 0.5) * h[0] - center[0]
                y = origin[1] + (iy + 0.5) * h[1] - center[1]
                z = origin[2] + (iz + 0.5) * h[2] - center[2]
                if abs(x) < half_inner[0] and abs(y) < half_inner[1] and abs(z) < half_inner[2]:
                    continue   # 内腔，不放点
                points.append((x + center[0], y + center[1], z + center[2]))
    return points


def random_points_in_box(
    size: Sequence[float],
    count: int,
    rng,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[tuple[float, float, float]]:
    """在长方体内部均匀随机采样（等体积概率）。

    与点阵采样对比：点阵的间距是各向同性的常数，随机点的局部间距有涨落，
    所以修正项的残差会更大。这个对比能说明修正公式的适用边界。
    """
    points = []
    for _ in range(count):
        points.append(
            tuple(
                center[k] + rng.uniform(-0.5, 0.5) * size[k]
                for k in range(3)
            )
        )
    return points


def random_points_in_sphere(
    radius: float,
    count: int,
    rng,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[tuple[float, float, float]]:
    """在球内均匀采样（用立方根修正半径分布，否则会向球心聚集）。"""
    points = []
    for _ in range(count):
        # 方向：单位球面均匀
        while True:
            x = rng.uniform(-1.0, 1.0)
            y = rng.uniform(-1.0, 1.0)
            z = rng.uniform(-1.0, 1.0)
            length_squared = x * x + y * y + z * z
            if 1e-12 < length_squared <= 1.0:
                break
        scale = radius * (rng.random() ** (1.0 / 3.0)) / math.sqrt(length_squared)
        points.append(
            (
                center[0] + x * scale,
                center[1] + y * scale,
                center[2] + z * scale,
            )
        )
    return points
