# -*- coding: utf-8 -*-
"""多体同时接触（simultaneous contacts）测试。

要回答的问题
------------
顺序冲量求解器是**逐个**处理接触的。当三个物体在同一时刻互相挤压
（例如中间的球被左右两侧同时撞击），逐个施加冲量会破坏对称性。
那么：

1. 结果会不会错？
2. 迭代次数够不够把它"补"回来？
3. 物体的**列表顺序**会不会改变结果？

为什么这个场景重要
------------------
- 球堆叠（一个球被多个球同时支撑）
- 盒-盒堆叠（每个流形 4 个点，且多个流形同时存在）
- 自碰撞密集的瞬间

如果求解器在这类构型下表现不稳，"能静置"和"不炸"就都保不住。

"同时"的真实含义
----------------
离散仿真里没有真正的"同时"。所谓同时接触，只是指**在同一个子步内，
多个接触流形同时存在**。所以问题等价于：如何求解一个多点接触系统
在同一时刻的速度。

理论背景（为什么这件事本质上没有唯一答案）
------------------------------------------
刚体的多体同时碰撞在连续时间意义上是**不适定（ill-posed）**问题：
彼此接触的 n 个刚体，碰撞后的速度依赖于本构假设（牛顿恢复系数假设、
泊松冲量假设、能量守恒假设给出不同结果），并不存在唯一解。
所以工程上的目标是：**稳定、可复现、不注入能量、对称构型得到对称结果**，
而不是"物理上绝对正确"。

运行::

    python Code/tests/test_multicontact.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import SphereGeometry
from src.geom.inertia import compute_inertia
from src.physics.body import RigidBody
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_sphere(name, radius, position, velocity=(0.0, 0.0, 0.0),
                mass=1.0, restitution=0.7, friction=0.0) -> RigidBody:
    """造一个球刚体，默认 1 kg 便于手算。"""
    geometry = SphereGeometry(radius)
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name, geometry=geometry, position=position,
        mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(bodies, iterations: int = 12, substeps: int = 1) -> PhysicsWorld:
    """纯净世界：无重力、无阻尼、无 Baumgarte，只保留接触求解本身。"""
    sim = SimulationSpec(
        enabled=True,
        physics_dt=1.0 / 240.0,
        substeps=substeps,
        iterations=iterations,
        baumgarte=0.0,
        slop=0.0,
        restitution_threshold=0.05,
        gravity=(0.0, 0.0, 0.0),
        linear_damping=0.0,
        angular_damping=0.0,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), None, sim)


def build_squeeze(reverse_order: bool = False, restitution: float = 0.7):
    """构造"中间球被左右两球同时挤压"的对称构型。

    三球半径 0.3，球心间距 0.598（略小于 0.6），因此**两个接触在第一步同时存在**。
    左球以 +v 向右、右球以 -v 向左，中间球静止。

    解析解（等质量、对称、无摩擦）：
        由对称性 v_B = 0；两个接触的恢复目标都是 +e·v
        ⇒ v_B - v_A = e·v 且 v_C - v_B = e·v
        ⇒ v_A = -e·v, v_B = 0, v_C = +e·v
    """
    radius = 0.3
    half_gap = 0.299
    speed = 3.0

    left = make_sphere("left", radius, (-half_gap, 0.0, 0.0), (speed, 0.0, 0.0),
                       restitution=restitution)
    middle = make_sphere("middle", radius, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                         restitution=restitution)
    right = make_sphere("right", radius, (half_gap, 0.0, 0.0), (-speed, 0.0, 0.0),
                        restitution=restitution)

    bodies = [left, middle, right]
    if reverse_order:
        # 只改列表顺序，几何与物理完全不变
        bodies = list(reversed(bodies))
    return bodies, speed


def settle_one_step(bodies, iterations: int = 12):
    """走一步，返回三步各自的速度。"""
    world = make_world(bodies, iterations=iterations)
    world.step()
    return world


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------

def test_simultaneous_squeeze() -> tuple[bool, str]:
    """对称三体挤压：结果必须对称，且与解析解一致。"""
    bodies, speed = build_squeeze(restitution=0.7)
    world = settle_one_step(bodies, iterations=12)

    left, middle, right = bodies
    e = 0.7

    # 解析解
    exp_left = -e * speed
    exp_middle = 0.0
    exp_right = e * speed

    err = max(
        abs(left.linear_velocity[0] - exp_left),
        abs(middle.linear_velocity[0] - exp_middle),
        abs(right.linear_velocity[0] - exp_right),
    )

    return err < 0.05, (
        f"对称挤压 e={e}（12 次迭代）："
        f"left={left.linear_velocity[0]:+.4f}（期望 {exp_left:+.4f}），"
        f"middle={middle.linear_velocity[0]:+.4f}（期望 0），"
        f"right={right.linear_velocity[0]:+.4f}（期望 {exp_right:+.4f}），"
        f"最大误差={err:.2e}"
    )


def test_convergence_vs_iterations() -> tuple[bool, str]:
    """迭代次数扫描：这是回答"该怎么解"的核心数据。"""
    speed = 3.0
    e = 0.7

    rows = []
    for iterations in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48):
        bodies, _ = build_squeeze(restitution=e)
        settle_one_step(bodies, iterations=iterations)
        left, middle, right = bodies
        err = max(
            abs(left.linear_velocity[0] - (-e * speed)),
            abs(middle.linear_velocity[0]),
            abs(right.linear_velocity[0] - e * speed),
        )
        rows.append((iterations, middle.linear_velocity[0], err))

    summary = "  ".join(f"it{n}:mid={vm:+.3f},err={er:.3f}" for n, vm, er in rows)
    converged = rows[-1][2] < 1e-6
    return converged, f"迭代扫描 → {summary}"


def test_order_independence() -> tuple[bool, str]:
    """物体列表顺序不同，结果必须相同。

    这是检测"顺序依赖"的直接手段：如果结果随列表顺序变化，
    说明求解结果取决于遍历次序而不是物理本身。
    """
    bodies_a, speed = build_squeeze(restitution=0.7)
    settle_one_step(bodies_a, iterations=12)
    by_name_a = {b.name: b.linear_velocity[0] for b in bodies_a}

    bodies_b, _ = build_squeeze(restitution=0.7, reverse_order=True)
    settle_one_step(bodies_b, iterations=12)
    by_name_b = {b.name: b.linear_velocity[0] for b in bodies_b}

    drift = max(abs(by_name_a[k] - by_name_b[k]) for k in by_name_a)
    return drift < 1e-6, (
        f"正序 left={by_name_a['left']:+.6f} / 逆序 left={by_name_b['left']:+.6f}；"
        f"middle {by_name_a['middle']:+.6f} / {by_name_b['middle']:+.6f}；"
        f"right {by_name_a['right']:+.6f} / {by_name_b['right']:+.6f}；最大差异={drift:.2e}"
    )


def test_momentum_and_energy() -> tuple[bool, str]:
    """三体同时接触：动量守恒、能量不增加。"""
    bodies, _ = build_squeeze(restitution=0.7)
    world = make_world(bodies, iterations=12)

    p_before = sum(b.mass * b.linear_velocity[0] for b in bodies)
    e_before = world.total_kinetic_energy()

    world.step()

    p_after = sum(b.mass * b.linear_velocity[0] for b in bodies)
    e_after = world.total_kinetic_energy()

    momentum_ok = abs(p_after - p_before) < 1e-9
    energy_ok = e_after <= e_before * 1.000001

    return momentum_ok and energy_ok, (
        f"动量 {p_before:+.6f} → {p_after:+.6f}；"
        f"动能 {e_before:.6f} J → {e_after:.6f} J"
        f"（涨幅 {(e_after/e_before-1)*100:+.4f}%）"
    )


def test_four_body_chain_squeeze() -> tuple[bool, str]:
    """四个球同时挤压：检验迭代次数随接触数增长时的鲁棒性。

    四球排成一排，中间两个球被两侧同时压缩，接触数 3。
    只检查"不炸、不增能量、动量守恒"，不追求解析解
    （四体情形的连续时间解本身就不唯一）。
    """
    radius = 0.3
    gap = 0.598
    speed = 2.0

    bodies = []
    for i in range(4):
        x = (i - 1.5) * gap
        velocity = (speed, 0.0, 0.0) if i == 0 else (
            (-speed, 0.0, 0.0) if i == 3 else (0.0, 0.0, 0.0)
        )
        bodies.append(make_sphere(f"b{i}", radius, (x, 0.0, 0.0), velocity,
                                  restitution=0.7))

    world = make_world(bodies, iterations=12)
    p_before = sum(b.linear_velocity[0] for b in bodies)
    e_before = world.total_kinetic_energy()

    for _ in range(40):
        world.step()

    p_after = sum(b.linear_velocity[0] for b in bodies)
    e_after = world.total_kinetic_energy()
    finite = all(math.isfinite(c) for b in bodies for c in b.linear_velocity)

    ok = (
        finite
        and abs(p_after - p_before) < 1e-8
        and e_after <= e_before * 1.000001
    )
    return ok, (
        f"四体挤压 40 步：动量 {p_before:+.4f} → {p_after:+.4f}；"
        f"动能 {e_before:.4f} → {e_after:.4f} J；有限={finite}；"
        f"终态 vx={[round(b.linear_velocity[0],3) for b in bodies]}"
    )


def test_small_steps_vs_iterations() -> tuple[bool, str]:
    """"多次迭代"与"缩小步长"的对照实验。

    这是回答"多体同时接触该怎么解"最关键的一组数据。

    现代引擎（Box2D v3、MuJoCo）倾向于**小步长 × 少迭代**，理由是：
    每个子步位置都会更新，接触几何变得更"新"，后续子步能在此基础上继续修正。

    但对**一次性的高速撞击**，这个优势可能不存在——碰撞只在第一个子步里发生，
    之后的子步物体已经分离、没有接触可解，于是"小步长"退化成了"只迭代一次"。

    下面的实验把总工作量固定为 ``substeps × iterations = 12``，只改变分配方式。
    """
    speed = 3.0
    e = 0.7

    rows = []
    for substeps, iterations in ((1, 12), (2, 6), (3, 4), (4, 3), (6, 2), (12, 1)):
        bodies, _ = build_squeeze(restitution=e)
        world = make_world(bodies, iterations=iterations, substeps=substeps)
        world.step()

        left, middle, right = bodies
        err = max(
            abs(left.linear_velocity[0] - (-e * speed)),
            abs(middle.linear_velocity[0]),
            abs(right.linear_velocity[0] - e * speed),
        )
        rows.append((substeps, iterations, err))

    summary = "  ".join(f"sub{n}/it{m}:err={er:.4f}" for n, m, er in rows)

    # 该断言记录一个结论而不是物理性质：单次撞击下，迭代次数是主导因素。
    # 若将来换成 TGS-soft 或直接求解器，这个断言需要重新评估。
    best_is_many_iterations = rows[0][2] < rows[-1][2]
    return best_is_many_iterations, f"总工作量固定为 12 → {summary}"


def build_chain(n: int, restitution: float = 0.7, speed: float = 3.0):
    """构造 n 个球排成一排、左右两端同时内挤的构型。

    间距 0.598 略小于半径和 0.6，因此**全部 n-1 个接触在第一步同时存在**。

    解析解：只有两端的球有初始接近速度，内部接触的 ``vn0 = 0``（恢复目标为 0）。
    由动量守恒与对称性可解得：两端球速度为 ``∓e·v``，所有内部球速度**恰好为 0**。
    这个结论与 n 无关，因此可以直接作为任意 n 的参考解。
    """
    radius = 0.3
    gap = 0.598

    bodies = []
    for i in range(n):
        x = (i - (n - 1) / 2.0) * gap
        if i == 0:
            velocity = (speed, 0.0, 0.0)
        elif i == n - 1:
            velocity = (-speed, 0.0, 0.0)
        else:
            velocity = (0.0, 0.0, 0.0)
        bodies.append(make_sphere(f"c{i}", radius, (x, 0.0, 0.0), velocity,
                                  restitution=restitution))
    return bodies


def _chain_error(bodies, e: float, speed: float) -> float:
    """与解析解的最大偏差。"""
    worst = 0.0
    last = len(bodies) - 1
    for i, body in enumerate(bodies):
        if i == 0:
            target = -e * speed
        elif i == last:
            target = e * speed
        else:
            target = 0.0
        worst = max(worst, abs(body.linear_velocity[0] - target))
    return worst


def test_convergence_vs_contact_count() -> tuple[bool, str]:
    """接触数量增加时，需要多少次迭代才能收敛。

    这是判断"迭代次数该设成多少"的依据。盒-盒堆叠的接触数会远多于 3，
    如果收敛速度随接触数急剧恶化，就必须提前换方案（暖启动 / 直接求解器）。
    """
    e = 0.7
    speed = 3.0
    tolerance = 0.01

    rows = []
    for n in (3, 5, 7, 9):
        needed = None
        for iterations in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            bodies = build_chain(n, restitution=e, speed=speed)
            world = make_world(bodies, iterations=iterations)
            world.step()
            if _chain_error(bodies, e, speed) < tolerance:
                needed = iterations
                break
        rows.append((n, n - 1, needed))

    summary = "  ".join(
        f"{n}球({c}接触):需{n_it if n_it else '>32'}次迭代" for n, c, n_it in rows
    )

    needed = [r[2] if r[2] else 999 for r in rows]

    # 断言一：所需的迭代次数随接触数**单调不减**（收敛确实在恶化，不是噪声）
    grows_with_contacts = all(needed[i] <= needed[i + 1] for i in range(len(needed) - 1))
    # 断言二：当前默认的 12 次迭代足以覆盖**两三个接触**的日常场景，
    #         但不足以覆盖 6 个以上接触的一维链——这正是盒-盒堆叠要面对的形态，
    #         因此加暖启动（warm starting）之前不要贸然堆 4 层以上的盒子。
    enough_for_small_scenes = needed[0] <= 12

    return grows_with_contacts and enough_for_small_scenes, (
        f"收敛到误差<{tolerance} → {summary}"
    )


def test_overlap_selects_regime() -> tuple[bool, str]:
    """初始重叠量如何决定三球对撞的结果：这是理解"同时性"的关键实验。

    三球等质量、半径 0.3、恢复系数 1，左球以 3 m/s 撞向静止的中间球与右球。

    刚性模型下这个问题**本身就是不适定的**：以下两个解都严格满足
    动量守恒与能量守恒，谁"对"取决于材料。

        解 A（波传播式，钢球的真实行为）: v = (0, 0, 3)
        解 B（同时式，完全塑性压缩后再弹开）: v = (-1, 2, 2)

    本实验改变两球之间的初始重叠量，观察求解器落在哪个解上：

    - 恰好相切（重叠 0）时，接触**不会同时存在**——A-B 先解，B 动起来之后
      才与 C 相切。顺序冲量因此自然给出解 A，也就是钢球的真实行为。
    - 一旦存在重叠，两个接触在第一步就同时有效，结果转向解 B。

    结论：**离散化本身在选择物理区间。** 小步长 + 不互相嵌入的初始状态，
    会让"同时"退化成"先后"，而后者才是刚度高的物体的真实物理。
    """
    radius = 0.3
    contact_distance = 2.0 * radius
    speed = 3.0

    rows = []
    for overlap in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3):
        gap = contact_distance - overlap
        bodies = [
            make_sphere("a", radius, (-gap, 0.0, 0.0), (speed, 0.0, 0.0), restitution=1.0),
            make_sphere("b", radius, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
            make_sphere("c", radius, (gap, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
        ]
        world = make_world(bodies, iterations=12)
        # 走 4 步：足够让"先后"传播完成
        for _ in range(4):
            world.step()

        values = [round(b.linear_velocity[0], 4) for b in bodies]
        rows.append((overlap, values))

    summary = "  ".join(f"δ={ov:.0e}:{v}" for ov, v in rows)

    # 断言：完全不重叠时必须给出波传播解 (0, 0, 3)
    touching = rows[0][1]
    wave_like = abs(touching[0]) < 1e-6 and abs(touching[1]) < 1e-6 and abs(touching[2] - speed) < 1e-6
    return wave_like, f"重叠量 → 终态 vx（期望 δ=0 为 [0,0,3]）→ {summary}"


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 对称三体同时挤压", test_simultaneous_squeeze),
    ("2. 迭代次数收敛扫描", test_convergence_vs_iterations),
    ("3. 物体顺序无关性", test_order_independence),
    ("4. 动量守恒与能量不增", test_momentum_and_energy),
    ("5. 四体同时挤压鲁棒性", test_four_body_chain_squeeze),
    ("6. 多迭代 vs 小步长", test_small_steps_vs_iterations),
    ("7. 接触数量对收敛速度的影响", test_convergence_vs_contact_count),
    ("8. 初始重叠量决定物理区间", test_overlap_selects_regime),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 94)
    print("多体同时接触（simultaneous contacts）测试")
    print("=" * 94)

    passed = 0
    for title, func in CHECKS:
        try:
            ok, message = func()
        except Exception as exc:  # noqa: BLE001
            ok, message = False, f"抛出异常：{type(exc).__name__}: {exc}"

        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {message}")
        if ok:
            passed += 1

    print("-" * 94)
    print(f"结果：{passed}/{len(CHECKS)} 通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
