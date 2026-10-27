# -*- coding: utf-8 -*-
"""求解器策略决策实验：多迭代（PGS） vs 小步长（TGS 风格）。

要回答的问题
------------
Erin Catto（Box2D 作者）在 ``Solver2D`` 里给过明确结论：

    "It has long been known that smaller time steps are more effective than
     more iterations. This is based on the Taylor Series."

但对**一次性高速撞击**，我们此前实测的结论相反（接触只在第一个子步存在，
后续子步无约束可解）。两者不矛盾，只是适用域不同：

- 一次性撞击   → 迭代次数是主导
- 持续接触/堆叠 → 子步是主导（每个子步都能在更新后的位置上重新线性化）

本脚本用**持续接触**场景来判定：在我们的实现与尺度上，哪种分配更划算。

为什么要用"解析平衡位置"做基准
------------------------------
只比"看起来稳不稳"没有说服力。这里构造**平衡位置可手算**的构型：
一个球搁在两个静态球形成的凹槽里，由几何得

    z_eq = z_base + sqrt((R + r)² - d²)

于是"下沉了多少毫米"是一个确定数字，不同配置可直接比较。

为什么要对多组初始扰动取平均
----------------------------
单次跑出来的下沉量会受到接触构型与释放高度的混沌影响，
首轮实验里 (3,4) 组合明显偏离趋势。因此每个配置跑 3 个释放高度，
取平均，避免从噪声里读结论。

场景
----
1. **凹槽静置**：两静态球 + 一动态球，2 个接触，有解析平衡高度
2. **金字塔**：三静态球 + 两动态球 + 顶球，多接触链；
   以"高精度解"为参照测量偏差（绝对下沉量会被 slop 掩盖，不可用）

运行::

    python Code/scripts/bench_solver_strategy.py
"""

from __future__ import annotations

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
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec

#: 场景几何常量
BASE_R = 0.4
BASE_D = 0.5
BALL_R = 0.3

#: 仿真时长（秒）与对应步数
DURATION = 2.0

#: 释放高度扰动比例（相对 0.25 m 基准抬升量）
PERTURBATIONS = (0.2, 0.5, 1.0)


def make_sphere(name, radius, position, mass=None, restitution=0.10,
                friction=0.75, is_static=False) -> RigidBody:
    """造一个球刚体。静态球质量无意义，但仍需有限值参与惯量计算。"""
    geometry = SphereGeometry(radius)
    if mass is None:
        mass = geometry.volume() * 1000.0
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name, geometry=geometry, position=position,
        mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction, is_static=is_static,
    )
    body.linear_velocity = (0.0, 0.0, 0.0)
    return body


def make_world(bodies, substeps: int, iterations: int, dt: float = 1.0 / 240.0) -> PhysicsWorld:
    """按指定策略组一个世界。

    ``substeps * iterations`` 固定时，两者是不同的"工作量分配方式"：
    求解器遍历次数相同，但窄相检测次数等于 substeps。
    """
    sim = SimulationSpec(
        enabled=True,
        physics_dt=dt,
        substeps=substeps,
        iterations=iterations,
        baumgarte=0.25,
        slop=0.002,
        restitution_threshold=1.0,
        gravity=(0.0, 0.0, -9.81),
        ground_z=0.0,
        ground_friction=0.75,
        ground_restitution=0.05,
        linear_damping=0.02,
        angular_damping=1.0,
        # 本实验比较"迭代 vs 子步"。TOI 分段会额外增加窄相调用次数、
        # 污染对比，因此统一关闭。
        use_toi_splitting=False,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), GroundPlane(0.0, 0.75, 0.05), sim)


def simulate(world: PhysicsWorld) -> float:
    """跑完整个时长，返回总耗时（秒）。"""
    steps = int(round(DURATION / world.simulation.physics_dt))
    start = time.perf_counter()
    for _ in range(steps):
        world.step()
    return time.perf_counter() - start, steps


# ---------------------------------------------------------------------------
# 场景一：凹槽静置（有精确解析平衡高度）
# ---------------------------------------------------------------------------

def valley_equilibrium_z() -> float:
    """凹槽中动态球的解析平衡高度（球心 z 坐标）。"""
    vertical = math.sqrt((BASE_R + BALL_R) ** 2 - BASE_D ** 2)
    return BASE_R + vertical


def build_valley(perturbation: float):
    """两静态球形成的凹槽 + 一个动态球。

    ``perturbation`` 控制释放高度（1.0 对应抬高 0.25 m）。
    """
    z_eq = valley_equilibrium_z()
    bodies = [
        make_sphere("base_l", BASE_R, (-BASE_D, 0.0, BASE_R), is_static=True),
        make_sphere("base_r", BASE_R, (+BASE_D, 0.0, BASE_R), is_static=True),
        make_sphere("ball", BALL_R, (0.0, 0.0, z_eq + 0.25 * perturbation)),
    ]
    return bodies, z_eq


def run_valley(substeps: int, iterations: int, perturbation: float) -> dict:
    """跑一次凹槽场景，返回下沉量与耗时。"""
    bodies, z_eq = build_valley(perturbation)
    world = make_world(bodies, substeps, iterations)
    ball = bodies[2]

    elapsed, steps = simulate(world)

    return {
        "sag_mm": (ball.position[2] - z_eq) * 1000.0,
        "drift_mm": abs(ball.position[0]) * 1000.0,
        "residual_v": ball.speed(),
        "ms_per_step": elapsed / steps * 1000.0,
    }


# ---------------------------------------------------------------------------
# 场景二：金字塔（多接触链）
# ---------------------------------------------------------------------------

def build_pyramid():
    """三静态球 + 两动态球 + 顶球，形成一条多接触链。"""
    bodies = [
        make_sphere("base_0", BASE_R, (-0.9, 0.0, BASE_R), is_static=True),
        make_sphere("base_1", BASE_R, (0.0, 0.0, BASE_R), is_static=True),
        make_sphere("base_2", BASE_R, (+0.9, 0.0, BASE_R), is_static=True),
    ]
    valley_z = BASE_R + math.sqrt((BASE_R + BALL_R) ** 2 - 0.45 ** 2)
    for idx, x in enumerate((-0.45, 0.45)):
        bodies.append(make_sphere(f"low_{idx}", BALL_R, (x, 0.0, valley_z + 0.02)))
    top_z = valley_z + math.sqrt((2.0 * BALL_R) ** 2 - 0.45 ** 2)
    bodies.append(make_sphere("top", BALL_R, (0.0, 0.0, top_z + 0.02)))
    return bodies


def run_pyramid(substeps: int, iterations: int, reference=None):
    """跑金字塔场景。

    ``reference`` 为高精度解的动态球位置；给出时计算相对偏差。
    """
    bodies = build_pyramid()
    world = make_world(bodies, substeps, iterations)
    elapsed, steps = simulate(world)

    dynamic = [b for b in bodies if not b.is_static]
    result = {
        "residual_v": max(b.speed() for b in dynamic),
        "ms_per_step": elapsed / steps * 1000.0,
        "positions": [tuple(b.position) for b in dynamic],
    }
    if reference is not None:
        worst = 0.0
        for got, ref in zip(result["positions"], reference):
            dist = math.sqrt(sum((got[i] - ref[i]) ** 2 for i in range(3)))
            worst = max(worst, dist)
        result["deviation_mm"] = worst * 1000.0
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

#: (substeps, iterations) 组合。首尾是"纯多迭代"与"纯小步长"两端，
#: 中间围绕当前默认值 (2, 12) 加密，用于判断现状是否已落在最优点附近。
STRATEGIES = [
    (1, 12),
    (2, 6),
    (2, 8),
    (2, 12),
    (3, 6),
    (3, 8),
    (4, 6),
    (6, 2),
    (12, 1),
]


def main() -> int:
    """跑完全部策略并打印对照表。"""
    print("=" * 100)
    print("求解器策略决策实验：多迭代（PGS） vs 小步长（TGS 风格）")
    print("=" * 100)
    print(f"凹槽解析平衡高度 z_eq = {valley_equilibrium_z():.6f} m")
    print(f"每个配置跑 {len(PERTURBATIONS)} 个释放高度取平均，仿真 {DURATION} s\n")

    print("-" * 100)
    print("场景一：凹槽静置（2 个接触，有解析解）")
    print("-" * 100)
    print(f"{'substeps':>8} {'iters':>6} {'遍历':>5} {'平均下沉mm':>11} {'最差mm':>9} "
          f"{'侧移mm':>8} {'余速':>8} {'ms/步':>8}")
    print("-" * 100)

    valley_rows = []
    for substeps, iterations in STRATEGIES:
        runs = [run_valley(substeps, iterations, p) for p in PERTURBATIONS]
        mean_sag = sum(r["sag_mm"] for r in runs) / len(runs)
        worst_sag = max(abs(r["sag_mm"]) for r in runs)
        mean_drift = sum(r["drift_mm"] for r in runs) / len(runs)
        worst_v = max(r["residual_v"] for r in runs)
        mean_ms = sum(r["ms_per_step"] for r in runs) / len(runs)

        valley_rows.append({
            "substeps": substeps, "iterations": iterations,
            "sag": mean_sag, "worst": worst_sag, "drift": mean_drift,
            "residual": worst_v, "ms": mean_ms,
        })
        print(
            f"{substeps:>8} {iterations:>6} {substeps * iterations:>5} "
            f"{mean_sag:>11.3f} {worst_sag:>9.3f} {mean_drift:>8.3f} "
            f"{worst_v:>8.4f} {mean_ms:>8.3f}"
        )

    # ---- 场景二：以高精度解为参照 ----
    print("\n" + "-" * 100)
    print("场景二：金字塔（多接触链）——以 substeps=16/iters=8 为参照解")
    print("-" * 100)

    ref = run_pyramid(16, 8)
    print(f"参照解耗时 {ref['ms_per_step']:.2f} ms/步（仅用于生成基准，不参与排名）\n")
    print(f"{'substeps':>8} {'iters':>6} {'遍历':>5} {'偏离参照mm':>12} {'余速':>8} {'ms/步':>8}")
    print("-" * 100)

    pyramid_rows = []
    for substeps, iterations in STRATEGIES:
        r = run_pyramid(substeps, iterations, reference=ref["positions"])
        pyramid_rows.append({
            "substeps": substeps, "iterations": iterations,
            "deviation": r["deviation_mm"], "residual": r["residual_v"],
            "ms": r["ms_per_step"],
        })
        print(
            f"{substeps:>8} {iterations:>6} {substeps * iterations:>5} "
            f"{r['deviation_mm']:>12.3f} {r['residual_v']:>8.4f} {r['ms_per_step']:>8.3f}"
        )

    # ---- 判定 ----
    print("\n" + "=" * 100)
    print("判定")
    print("=" * 100)

    big_step = next(r for r in valley_rows if r["substeps"] == 1)
    # 与 substeps=1 遍历次数相同的那一组
    same_work = [r for r in valley_rows if r["substeps"] * r["iterations"] == 12]
    best_accuracy = min(same_work, key=lambda r: r["worst"])

    print(f"凹槽（遍历次数均为 12）：")
    print(f"  substeps=1,  iterations=12 → 平均下沉 {big_step['sag']:+.3f} mm，"
          f"耗时 {big_step['ms']:.3f} ms/步")
    print(f"  substeps={best_accuracy['substeps']}, iterations={best_accuracy['iterations']}"
          f" → 平均下沉 {best_accuracy['sag']:+.3f} mm，耗时 {best_accuracy['ms']:.3f} ms/步")
    ratio = abs(big_step["sag"]) / max(abs(best_accuracy["sag"]), 1e-9)
    print(f"  精度提升 {ratio:.1f} 倍，代价 {best_accuracy['ms'] / big_step['ms']:.2f} 倍时间")

    print(f"\n金字塔（以高精度解为参照）：")
    for row in pyramid_rows:
        print(f"  substeps={row['substeps']:>2}, iters={row['iterations']:>2} → "
              f"偏离 {row['deviation']:>7.3f} mm，{row['ms']:.2f} ms/步")

    # 估算：若接触点可跨子步复用（不重跑窄相），子步方案的成本会降到多少
    slope = (best_accuracy["ms"] - big_step["ms"]) / (best_accuracy["substeps"] - 1)
    print(f"\n每多一个子步的额外开销约 {slope:.4f} ms（主要是重复窄相检测）。")
    print(f"若像 Box2D 那样把接触点存在局部坐标里跨子步复用，"
          f"子步方案的成本可降到约 {big_step['ms']:.3f} ms/步，即与现在持平。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
