# -*- coding: utf-8 -*-
"""宽阶段（broadphase）：把 O(N²) 的两两检测降到近似 O(N)。

问题
----
窄相（球-球、球-盒、盒-盒）逐个物体对都很便宜，但"枚举所有物体对"是 O(N²)。
100 个物体 → 4950 对，每秒 240 步 × 3 子步 → 每秒 356 万次窄相调用。
实际上任何时刻真正在接触的物体对只有 O(N) 量级，其余全是白算。

解决
----
先用**廉价的包围盒**筛掉绝大多数明显不相交的物体对，只把通过的候选对交给窄相。
这一步就叫宽阶段。

两种实现
--------
- ``BruteForceBroadphase``：不做任何剔除，就是 O(N²)。**保留它不是为了用，
  而是为了给加速比提供基线**——没有基线的"优化"无法证明有效。
- ``SpatialHashBroadphase``：均匀网格空间哈希。

空间哈希怎么工作
----------------
1. 选一个格子边长 ``cell``
2. 每个物体算出世界坐标 AABB，把它**塞进所有与 AABB 相交的格子**
3. 遍历格子，同一个格子里（或相邻格子里）的物体就是候选对
4. 再用 AABB 精确判断一次，过滤掉"同格但不相交"的

为什么用哈希而不是八叉树
------------------------
八叉树需要维护层级结构、插入删除、节点分裂合并，代码量是空间哈希的三到五倍。
物体分布比较均匀时（本项目的堆叠、散落场景），均匀网格的剔除率已经很好，
而且实现简单、缓存友好、每帧可以直接清空重建，不需要增量维护。

代价是**格子边长选得不合适会退化**：
- 太小 → 一个大物体跨越很多格子，插入开销爆炸
- 太大 → 每个格子里挤满物体，剔除效果消失

所以下面的实现带两个保护：按物体尺寸自动估算边长，以及给跨格数超限的
大物体单独走"全局列表"通道。

AABB 的保守性
-------------
包围盒相交是窄相接触的**必要条件**，因此用 AABB 剔除**不会漏掉任何接触**。
反过来 AABB 相交不保证真的接触，所以候选对还要过一遍窄相。
这点很重要：宽阶段可以保守，不能激进。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..geom.geometry import (
    BoxGeometry,
    CompoundGeometry,
    ConvexHullGeometry,
    SphereGeometry,
)
from ..mathutils import quaternion as quat
from .body import RigidBody, iter_parts

#: 自动估算格子边长时，相对"平均包围球半径"的倍数。
#:
#: 取 2 表示格子大致等于一个物体的直径。太小会让物体跨很多格子，
#: 太大则每个格子塞满物体、剔除率下降。实测 1.5–3 之间差别不大。
_CELL_SIZE_FACTOR = 2.0

#: 单个物体允许占用的格子数上限。超过就当作"大物体"，
#: 放进全局列表，避免一个巨大物体把哈希表撑爆。
_MAX_CELLS_PER_BODY = 64


@dataclass(frozen=True)
class AABB:
    """轴对齐包围盒（世界坐标）。"""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def overlaps(self, other: "AABB") -> bool:
        """两个 AABB 是否相交（含边界相接）。

        相接算相交是有意为之：恰好相切的两个物体下一步就可能产生穿透，
        宽阶段在这里稍微保守一点更安全。
        """
        return (
            self.lower[0] <= other.upper[0] and self.upper[0] >= other.lower[0]
            and self.lower[1] <= other.upper[1] and self.upper[1] >= other.lower[1]
            and self.lower[2] <= other.upper[2] and self.upper[2] >= other.lower[2]
        )

    def center(self) -> tuple[float, float, float]:
        """包围盒中心，用于体积估计与调试。"""
        return (
            0.5 * (self.lower[0] + self.upper[0]),
            0.5 * (self.lower[1] + self.upper[1]),
            0.5 * (self.lower[2] + self.upper[2]),
        )


def world_aabb(body: RigidBody) -> AABB:
    """计算刚体的世界坐标 AABB。

    盒子必须按**朝向**算，不能直接用半尺寸：
    一个 45° 倾斜的立方体，其世界 AABB 的边长是原来的 √2 倍。
    用局部半尺寸会得到一个过小的盒子，导致漏掉真实接触。

    盒子的世界轴 ``a_i = R·e_i``，则沿世界轴 ``k`` 的半跨度是

        Σ_i |a_i[k]| · h_i

    这正是"把半尺寸投影到世界轴"的标准做法。
    """
    geometry = body.geometry
    center = body.position

    if isinstance(geometry, CompoundGeometry):
        # 复合体：逐部件算 AABB 再取并集。部件的世界位姿由 PartView 给出，
        # 与窄相看到的位姿完全同源，不会出现"宽相按整体、窄相按部件"的错位。
        lower = [float("inf"), float("inf"), float("inf")]
        upper = [float("-inf"), float("-inf"), float("-inf")]
        for _index, view in iter_parts(body):
            box = world_aabb(view)
            for axis in range(3):
                lower[axis] = min(lower[axis], box.lower[axis])
                upper[axis] = max(upper[axis], box.upper[axis])
        return AABB(
            (lower[0], lower[1], lower[2]),
            (upper[0], upper[1], upper[2]),
        )

    if isinstance(geometry, SphereGeometry):
        radius = geometry.radius
        return AABB(
            (center[0] - radius, center[1] - radius, center[2] - radius),
            (center[0] + radius, center[1] + radius, center[2] + radius),
        )

    if isinstance(geometry, BoxGeometry):
        size = geometry.size
        half = (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])
        axes = [
            quat.quat_rotate_vector(body.orientation, axis)
            for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ]
        extent = [
            abs(axes[0][k]) * half[0]
            + abs(axes[1][k]) * half[1]
            + abs(axes[2][k]) * half[2]
            for k in range(3)
        ]
        return AABB(
            (center[0] - extent[0], center[1] - extent[1], center[2] - extent[2]),
            (center[0] + extent[0], center[1] + extent[1], center[2] + extent[2]),
        )

    if isinstance(geometry, ConvexHullGeometry):
        # 凸包顶点数很少（碰撞代理通常 ≤ 32），直接变换后取极值就是**精确** AABB。
        # 用包围球兜底虽然也不错对，但会很松：一个细长凸包的包围球体积可能是
        # 它自身 AABB 的好几倍，白白放进大量候选对，白跑窄相。
        world_vertices = [
            quat.quat_rotate_vector(body.orientation, v) for v in geometry.vertices
        ]
        low = [
            center[k] + min(v[k] for v in world_vertices) for k in range(3)
        ]
        high = [
            center[k] + max(v[k] for v in world_vertices) for k in range(3)
        ]
        return AABB(tuple(low), tuple(high))

    # 其余几何（柱、胶囊、圆锥、圆台、椭球、球冠）用包围球兜底：曲面体算精确
    # AABB 要旋转椭球/端面圆，收益不大，而兜底保证**保守不漏**——宽阶段多放几个
    # 候选对只是多跑几次窄相，漏掉才是错。
    radius = body.bounding_radius
    return AABB(
        (center[0] - radius, center[1] - radius, center[2] - radius),
        (center[0] + radius, center[1] + radius, center[2] + radius),
    )


class BruteForceBroadphase:
    """O(N²) 基线：枚举所有物体对，不做任何剔除。

    **这个类存在的意义是提供对照组**。加速比 = 基线耗时 / 优化后耗时，
    没有基线就无法证明优化有效，也无法说明"优化在哪里生效"。
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        """用于日志与对照表的短名称。"""
        return "brute"

    def candidate_pairs(self, bodies: list[RigidBody]) -> list[tuple[int, int]]:
        """返回所有可能的物体对（含静态-静态，由调用方过滤）。"""
        count = len(bodies)
        return [(i, j) for i in range(count) for j in range(i + 1, count)]


class SpatialHashBroadphase:
    """均匀网格空间哈希宽阶段。

    用法::

        broadphase = SpatialHashBroadphase(bodies)
        pairs = broadphase.candidate_pairs(bodies)   # 每步调用一次

    同一步内可以反复调用；内部容器会复用而不是每次重新分配。
    """

    __slots__ = ("cell_size", "margin", "_grid", "_large", "_stats")

    def __init__(
        self,
        bodies: list[RigidBody] | None = None,
        cell_size: float | None = None,
        margin: float = 0.0,
    ) -> None:
        """配置宽阶段。

        参数
        ----
        bodies    : 用于自动估算格子边长；为 None 时用 ``cell_size``
        cell_size : 格子边长；None 表示按物体尺寸自动估算
        margin    : 给 AABB 加的余量（米）。默认 0 就足够——
                    AABB 相交是接触的必要条件，不加余量也不会漏。
                    加余量只会让候选对变多，一般用于"提前一帧预判"。
        """
        self.margin = max(0.0, float(margin))
        self._grid: dict[tuple[int, int, int], list[int]] = {}
        self._large: list[int] = []
        self._stats = {"cells": 0, "large": 0, "pairs": 0}

        if cell_size is not None:
            value = float(cell_size)
            if value <= 0.0:
                raise ValueError(f"broadphase cell_size 必须为正，收到：{cell_size}")
            self.cell_size = value
        elif bodies:
            self.cell_size = self.estimate_cell_size(bodies)
        else:
            # 没有物体时给一个占位值，首次调用 candidate_pairs 时会重估
            self.cell_size = 1.0

    @property
    def name(self) -> str:
        """用于日志与对照表的短名称。"""
        return "spatial_hash"

    @property
    def stats(self) -> dict:
        """最近一次调用的统计量（格子数、大物体数、候选对数）。"""
        return dict(self._stats)

    @staticmethod
    def estimate_cell_size(bodies: list[RigidBody]) -> float:
        """按物体尺寸估算格子边长。

        用**平均包围球半径**的固定倍数。选平均值而不是最大值，
        是为了让"少数几个大物体"不至于把整张网格撑粗——
        大物体另有"跨格超限"通道兜底。
        """
        if not bodies:
            return 1.0
        total = sum(body.bounding_radius for body in bodies)
        average = total / len(bodies)
        if average <= 0.0:
            return 1.0
        return _CELL_SIZE_FACTOR * average

    # -- 内部 -------------------------------------------------------------

    def _cell_index(self, coordinate: float) -> int:
        """把世界坐标映射到格子下标。用 floor 保证负坐标也正确。"""
        return int(math.floor(coordinate / self.cell_size))

    def _cell_range(self, lower: float, upper: float) -> range:
        """某个轴上 AABB 覆盖的格子下标区间。"""
        return range(self._cell_index(lower), self._cell_index(upper) + 1)

    def _keys_for(self, box: AABB):
        """枚举 AABB 覆盖的所有格子键；跨格过多时返回 None。"""
        xs = self._cell_range(box.lower[0], box.upper[0])
        ys = self._cell_range(box.lower[1], box.upper[1])
        zs = self._cell_range(box.lower[2], box.upper[2])

        if len(xs) * len(ys) * len(zs) > _MAX_CELLS_PER_BODY:
            return None

        return [
            (x, y, z)
            for x in xs
            for y in ys
            for z in zs
        ]

    # -- 主流程 -----------------------------------------------------------

    def candidate_pairs(
        self,
        bodies: list[RigidBody],
        inflation: list[float] | None = None,
    ) -> list[tuple[int, int]]:
        """返回可能相交的物体对 ``(i, j)``，``i < j``，按字典序排序。

        排序是刻意的：哈希表的遍历顺序依赖插入顺序，不同配置下可能不同。
        若不加排序，求解器的接触处理顺序会随物体数量浮动，
        导致同一场景在不同规模下给出略有差异的结果——
        这对"可复现"和"与基线对比"都是致命的。

        参数
        ----
        inflation : 可选，逐物体的 AABB 膨胀半径（米）。给 TOI 用：它要回答的是
            "这个子步内有没有可能碰上"，所以 AABB 必须按**本子步最大位移**膨胀
            （``|v|h + |ω|Rh``），否则高速物体在穿过的过程中根本没有重叠的 AABB，
            宽阶段会把真正要切的碰撞漏掉。默认 ``None`` 表示不膨胀，用于窄相。
        """
        count = len(bodies)
        if count < 2:
            self._stats = {"cells": 0, "large": 0, "pairs": 0}
            return []

        boxes = [world_aabb(body) for body in bodies]
        if inflation is not None:
            boxes = [
                AABB(
                    (box.lower[0] - extra, box.lower[1] - extra, box.lower[2] - extra),
                    (box.upper[0] + extra, box.upper[1] + extra, box.upper[2] + extra),
                ) if extra > 0.0 else box
                for box, extra in zip(boxes, inflation)
            ]
        if self.margin > 0.0:
            boxes = [
                AABB(
                    (box.lower[0] - self.margin, box.lower[1] - self.margin,
                     box.lower[2] - self.margin),
                    (box.upper[0] + self.margin, box.upper[1] + self.margin,
                     box.upper[2] + self.margin),
                )
                for box in boxes
            ]

        grid = self._grid
        grid.clear()
        large = self._large
        large.clear()

        for index, box in enumerate(boxes):
            keys = self._keys_for(box)
            if keys is None:
                # 跨格太多：放进全局列表，下面所有物体都会与它比较
                large.append(index)
                continue
            for key in keys:
                bucket = grid.get(key)
                if bucket is None:
                    grid[key] = [index]
                else:
                    bucket.append(index)

        pairs: set[tuple[int, int]] = set()

        # --- 同一格子内的物体互为候选 ---
        for bucket in grid.values():
            size = len(bucket)
            if size < 2:
                continue
            for a in range(size):
                index_a = bucket[a]
                box_a = boxes[index_a]
                for b in range(a + 1, size):
                    index_b = bucket[b]
                    if index_a == index_b:
                        continue
                    low, high = (index_a, index_b) if index_a < index_b else (index_b, index_a)
                    if (low, high) in pairs:
                        continue
                    if box_a.overlaps(boxes[index_b]):
                        pairs.add((low, high))

        # --- 大物体与全体比较 ---
        for index_a in large:
            box_a = boxes[index_a]
            for index_b in range(count):
                if index_a == index_b:
                    continue
                low, high = (index_a, index_b) if index_a < index_b else (index_b, index_a)
                if (low, high) in pairs:
                    continue
                if box_a.overlaps(boxes[index_b]):
                    pairs.add((low, high))

        self._stats = {"cells": len(grid), "large": len(large), "pairs": len(pairs)}
        return sorted(pairs)
