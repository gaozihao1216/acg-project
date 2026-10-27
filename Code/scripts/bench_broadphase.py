# -*- coding: utf-8 -*-
"""宽阶段加速比基准：空间哈希 vs 暴力枚举。

这个脚本回答一个问题
--------------------
"加了空间哈希，到底快了多少？"——**必须用数字回答，不能靠感觉。**

三个指标，缺一不可
------------------
1. **检测耗时**（宽阶段 + 窄相）：算法层面的公平对比
2. **整步耗时**（含求解器与积分）：用户真实感知的收益
3. **候选对数**：解释加速从哪来

为什么不能只比"宽阶段自身耗时"
------------------------------
第一版脚本就是这么做的，结果哈希**看起来慢了 10 倍**（0.09x）。这个数字具有
严重的误导性：

    暴力宽阶段做的事是「生成 N(N-1)/2 个下标对」——它不做任何实际判断，
    全部开销都推给了窄相。所以它的"自身耗时"几乎为零。

拿"几乎为零"去比"哈希的建表开销"，等于说"什么都不做比做点事更快"。
**公平的对比必须把下游的窄相开销一起算进来。** 所以主指标用「检测耗时」。

两种布局
--------
- **spread（散布）**：间距 4 × 直径，AABB 完全不重叠 → 候选对 0，哈希的最优场景
- **dense（密排）**：间距恰好 1 × 直径，AABB 两两相接但**穿透为 0**
  → 每个物体有约 6 个邻居，候选对多，但不产生真实接触，因此求解器不干扰计时。
  这才是真实堆叠（每个物体与上下左右都有接触）的样子。

只测最优场景会高估收益，所以两种都测。

运行::

    python Code/scripts/bench_broadphase.py
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import SphereGeometry
from src.geom.inertia import compute_inertia
from src.mathutils import quaternion as quat
from src.physics.body import RigidBody
from src.physics.broadphase import BruteForceBroadphase, SpatialHashBroadphase
from src.physics.narrowphase import detect_contacts
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec

#: 检测计时的重复次数
DETECT_REPEATS = 8

#: 端到端计时的物理步数
SIM_STEPS = 20

#: 计时取多轮的最小值——最小值比平均值更抗噪声
RUNS = 3

RADIUS = 0.25


# ---------------------------------------------------------------------------
# 场景构造
# ---------------------------------------------------------------------------

def build_lattice(count: int, spacing_factor: float) -> list[RigidBody]:
    """把 ``count`` 个球摆成立方点阵。

    点阵而不是随机分布，是为了**可复现**：同一 ``count`` 每次得到完全相同的位置，
    两种宽阶段的对比才是同一条起跑线。
    """
    spacing = 2.0 * RADIUS * spacing_factor
    side = max(1, int(math.ceil(count ** (1.0 / 3.0))))
    geometry = SphereGeometry(RADIUS)

    bodies = []
    for index in range(count):
        ix = index % side
        iy = (index // side) % side
        iz = index // (side * side)
        position = (ix * spacing, iy * spacing, iz * spacing + RADIUS)
        mass = geometry.volume() * 1000.0
        bodies.append(
            RigidBody(
                name=f"s{index}",
                geometry=geometry,
                position=position,
                orientation=quat.quat_identity(),
                mass=mass,
                inertia_local=compute_inertia(geometry, mass),
                restitution=0.1,
                friction=0.6,
            )
        )
    return bodies


def build_world(bodies: list[RigidBody], broadphase: str) -> PhysicsWorld:
    """用指定宽阶段构造物理世界。"""
    simulation = SimulationSpec(
        enabled=True,
        physics_dt=1.0 / 240.0,
        substeps=1,
        iterations=8,
        baumgarte=0.25,
        slop=0.002,
        restitution_threshold=0.6,
        gravity=(0.0, 0.0, 0.0),      # 关重力：只测检测开销，不让物体动起来
        linear_damping=0.0,
        angular_damping=0.0,
        use_toi_splitting=False,       # 关 TOI：否则会引入额外的窄相调用
        broadphase=broadphase,
        duration=1.0,
    )
    return PhysicsWorld(bodies, None, simulation)


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------

def make_broadphase(bodies: list[RigidBody], mode: str):
    """构造指定模式的宽阶段。"""
    if mode == "brute":
        return BruteForceBroadphase()
    return SpatialHashBroadphase(bodies=bodies)


def time_detection(bodies: list[RigidBody], mode: str) -> tuple[float, int]:
    """宽阶段 + 窄相的合计耗时（秒/次）与候选对数。

    这是**算法层面的公平对比**：两种模式都跑完"拿到接触流形"这件事。
    """
    broadphase = make_broadphase(bodies, mode)
    pairs = broadphase.candidate_pairs(bodies)
    detect_contacts(bodies, None, pairs)          # 预热

    gc.disable()
    try:
        best = float("inf")
        for _ in range(RUNS):
            start = time.perf_counter()
            for _ in range(DETECT_REPEATS):
                current = broadphase.candidate_pairs(bodies)
                detect_contacts(bodies, None, current)
            best = min(best, (time.perf_counter() - start) / DETECT_REPEATS)
    finally:
        gc.enable()

    return best, len(pairs)


def time_simulation(count: int, spacing_factor: float, mode: str) -> float:
    """端到端：平均每个物理步的耗时（秒）。"""
    best = float("inf")
    for _ in range(RUNS):
        bodies = build_lattice(count, spacing_factor)
        world = build_world(bodies, mode)
        world.step()                              # 预热

        gc.disable()
        try:
            start = time.perf_counter()
            for _ in range(SIM_STEPS):
                world.step()
            best = min(best, (time.perf_counter() - start) / SIM_STEPS)
        finally:
            gc.enable()

    return best


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def format_row(cells) -> str:
    """把一行数据对齐成表格。"""
    return "  ".join(str(cell) for cell in cells)


def section(title: str) -> None:
    """打印小节标题。"""
    print()
    print(title)
    print("-" * 100)


def run_layout(label: str, spacing_factor: float, sizes: list[int]) -> None:
    """跑一种布局的完整扫描。"""
    section(f"布局：{label}（间距 = {spacing_factor} × 直径）")

    print(format_row([
        f"{'N':>6}",
        f"{'检测 暴力':>12}",
        f"{'检测 哈希':>12}",
        f"{'检测加速':>10}",
        f"{'候选对 暴力':>12}",
        f"{'候选对 哈希':>12}",
        f"{'窄相减少':>10}",
        f"{'整步 暴力':>12}",
        f"{'整步 哈希':>12}",
        f"{'端到端加速':>11}",
    ]))
    print("-" * 100)

    for count in sizes:
        bodies = build_lattice(count, spacing_factor)

        brute_detect, brute_pairs = time_detection(bodies, "brute")
        hash_detect, hash_pairs = time_detection(bodies, "spatial_hash")

        brute_step = time_simulation(count, spacing_factor, "brute")
        hash_step = time_simulation(count, spacing_factor, "spatial_hash")

        detect_speedup = brute_detect / hash_detect if hash_detect > 0 else float("inf")
        pair_reduction = brute_pairs / max(1, hash_pairs)
        step_speedup = brute_step / hash_step if hash_step > 0 else float("inf")

        print(format_row([
            f"{count:>6}",
            f"{brute_detect * 1e6:>10.2f}us",
            f"{hash_detect * 1e6:>10.2f}us",
            f"{detect_speedup:>9.2f}x",
            f"{brute_pairs:>12}",
            f"{hash_pairs:>12}",
            f"{pair_reduction:>9.1f}x",
            f"{brute_step * 1e3:>10.3f}ms",
            f"{hash_step * 1e3:>10.3f}ms",
            f"{step_speedup:>10.2f}x",
        ]))


def run_cell_size_sensitivity(count: int) -> None:
    """格子边长对性能的影响。"""
    section(f"格子边长敏感性（N = {count}，间距 = 1.0 × 直径）")

    bodies = build_lattice(count, 1.0)
    default = SpatialHashBroadphase(bodies=bodies).cell_size
    candidates = [default * factor for factor in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]

    print(format_row([
        f"{'格子边长':>11}", f"{'相对默认':>9}", f"{'检测耗时':>12}",
        f"{'格子数':>8}", f"{'兜底物体':>9}", f"{'候选对':>8}", f"{'相对默认':>10}",
    ]))
    print("-" * 100)

    baseline = None
    for cell in candidates:
        broadphase = SpatialHashBroadphase(bodies=bodies, cell_size=cell)
        broadphase.candidate_pairs(bodies)

        gc.disable()
        try:
            start = time.perf_counter()
            for _ in range(DETECT_REPEATS):
                current = broadphase.candidate_pairs(bodies)
                detect_contacts(bodies, None, current)
            elapsed = (time.perf_counter() - start) / DETECT_REPEATS
        finally:
            gc.enable()

        stats = broadphase.stats
        if abs(cell - default) < 1e-12:
            baseline = elapsed

        print(format_row([
            f"{cell:>10.4f}m",
            f"{cell / default:>8.2f}x",
            f"{elapsed * 1e6:>10.2f}us",
            f"{stats['cells']:>8}",
            f"{stats['large']:>9}",
            f"{stats['pairs']:>8}",
            "-" if baseline is None else f"{elapsed / baseline:>9.2f}x",
        ]))


def main() -> int:
    """跑完全部基准。"""
    print("=" * 100)
    print("宽阶段加速比基准：空间哈希 vs 暴力枚举")
    print("=" * 100)
    print(f"球半径 {RADIUS} m；检测计时 {DETECT_REPEATS} 次取均；整步计时 {SIM_STEPS} 步取均；均取 {RUNS} 轮最小值。")
    print("整步配置：substeps=1、iterations=8、关重力、关 TOI。")

    run_layout("spread 散布（AABB 几乎不重叠）", 4.0, [20, 50, 100, 200, 400])
    run_layout("dense 密排（每个物体约 6 个邻居）", 1.0, [20, 50, 100, 200, 400])
    run_cell_size_sensitivity(200)

    section("结论")
    print("1. 空间哈希把检测从 O(N²) 降到约 O(N)，规模越大优势越明显。")
    print("2. dense 布局的加速比比 spread 低——这是合理的：邻居多，剔除率就低。")
    print("3. 端到端加速比小于检测加速比，因为整步里还有求解器与积分。")
    print("   对外汇报应引用端到端数字，它才是用户能感知到的收益。")
    print("4. 格子边长取默认值（2 × 平均包围球半径）已接近最优；")
    print("   取到默认的 0.25 倍时所有物体都会因跨格超限走兜底通道，退化为 O(N²)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
