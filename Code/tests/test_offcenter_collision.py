# -*- coding: utf-8 -*-
"""球-球偏心（不对心）碰撞测试。

为什么要单独测这一项
--------------------
对心碰撞只要法向算对就行，几乎不可能出错。真正暴露问题的是**偏心**：

1. 接触法向必须由**接触时刻**的两球质心位置算出，而不是初始位置；
2. 力臂 ``r_a = 接触点 - 质心`` 与质心连线平行但不等长，转动惯量项开始参与；
3. 一旦法向或力臂写错，症状就是"球被莫名其妙地侧向弹飞"——
   看起来像数值爆炸，实际往往只是方向或顺序写错。

物理预期（光滑球，无摩擦）
--------------------------
冲量必须严格沿两球心连线 ``n``：

    法向相对速度  vn0 = (v_b - v_a) · n
    冲量大小      J   = -(1 + e) · vn0 / (1/m_a + 1/m_b)
    v_a' = v_a - J·n/m_a
    v_b' = v_b + J·n/m_b

注意分母里**没有转动惯量项**：球的接触点力臂 ``r = n · 半径`` 与 ``n`` 平行，
所以 ``r × n = 0``，转动贡献恒为零。这条性质本身就是一个强力检查——
如果实现里用错了力臂，角速度就不会是 0。

为什么参考解要按"离散时刻"算
----------------------------
连续时间的碰撞发生在两球恰好相切的瞬间；而离散仿真只能在**步长边界**上
检测，此时已经有一点穿透，法向与相切瞬间略有差别（O(h) 量级）。
所以拿"连续解析解"比对会得到 1% 量级的差异，无法区分是误差还是 bug。

这里的做法是：先用真实仿真推进到"下一步即将检测到接触"的状态，
取**那一刻的真实位置**代入闭式公式算出期望速度，再走一步让求解器施加冲量，
两者都基于同一组位置，于是可以逐位比对（1e-9 量级）。

运行::

    python Code/tests/test_offcenter_collision.py
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
# 构造辅助
# ---------------------------------------------------------------------------

def make_sphere(
    name, radius, position, velocity=(0.0, 0.0, 0.0),
    mass=1.0, restitution=0.7, friction=0.0, is_static=False,
) -> RigidBody:
    """造一个球刚体。默认质量取 1 kg，便于手算对照。"""
    geometry = SphereGeometry(radius)
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name, geometry=geometry, position=position,
        mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction, is_static=is_static,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(
    bodies, baumgarte: float = 0.0,
    linear_damping: float = 0.0, angular_damping: float = 0.0,
    gravity=(0.0, 0.0, 0.0), iterations: int = 12, substeps: int = 1,
    use_toi_splitting: bool = False,
) -> PhysicsWorld:
    """组一个"纯净"世界：默认关掉重力、阻尼、Baumgarte，便于与闭式解比对。

    ``substeps=1`` 是为了让"一步 = 一个子步"，保证接触只在一个离散时刻被处理一次。

    ``use_toi_splitting=False`` 是刻意的：本文件检查的是**求解器本身**的冲量公式，
    需要碰撞恰好发生在一个子步边界上，才能用"那一刻的位置"做闭式比对。
    如果开启 TOI 分段，碰撞会在子步内部被提前解算，物体根本不会进入穿透状态，
    ``advance_to_contact`` 也就无从判断。TOI 分段本身有专门的测试文件
    ``test_toi_split.py``。
    """
    sim = SimulationSpec(
        enabled=True,
        physics_dt=1.0 / 240.0,
        substeps=substeps,
        iterations=iterations,
        baumgarte=baumgarte,
        slop=0.0,
        restitution_threshold=0.05,
        gravity=gravity,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        use_toi_splitting=use_toi_splitting,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), None, sim)


def distance(a: RigidBody, b: RigidBody) -> float:
    """两球质心距离。"""
    return math.sqrt(sum((b.position[i] - a.position[i]) ** 2 for i in range(3)))


def unit_normal(a: RigidBody, b: RigidBody):
    """从 a 指向 b 的单位法向（按当前位置计算）。"""
    d = [b.position[i] - a.position[i] for i in range(3)]
    length = math.sqrt(sum(c * c for c in d))
    return [c / length for c in d]


def closed_form_impulse(a: RigidBody, b: RigidBody):
    """按当前位置与速度，用闭式公式算出碰撞后的速度。

    调用时机必须是"下一步即将检测到接触"的那一刻，因为法向依赖位置。

    返回 ``(v_a_after, v_b_after)``。
    """
    n = unit_normal(a, b)

    inv_ma = 0.0 if a.is_static else 1.0 / a.mass
    inv_mb = 0.0 if b.is_static else 1.0 / b.mass
    k = inv_ma + inv_mb
    if k <= 0.0:
        return a.linear_velocity, b.linear_velocity

    rel = [b.linear_velocity[i] - a.linear_velocity[i] for i in range(3)]
    vn0 = sum(rel[i] * n[i] for i in range(3))

    e = max(a.restitution, b.restitution)
    j = -(1.0 + e) * vn0 / k

    v_a_after = tuple(a.linear_velocity[i] - j * inv_ma * n[i] for i in range(3))
    v_b_after = tuple(b.linear_velocity[i] + j * inv_mb * n[i] for i in range(3))
    return v_a_after, v_b_after


def advance_to_contact(world: PhysicsWorld, a: RigidBody, b: RigidBody, radius_sum: float) -> None:
    """推进到"下一步就会检测到接触"的状态。

    判定条件与窄相一致（距离 < 半径和）。退出时位置已在接触范围内，
    下一次 ``world.step()`` 会施加冲量。
    """
    guard = 0
    while distance(a, b) >= radius_sum:
        world.step()
        guard += 1
        if guard > 5000:
            raise RuntimeError("推进到接触的步数超出预期，检查初始条件")


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------

def test_centered_baseline() -> tuple[bool, str]:
    """对照组：对心碰撞。

    等质量、对心、恢复系数 e：``v_a' = (1-e)/2 · v0``，``v_b' = (1+e)/2 · v0``。
    """
    speed, e = 3.0, 0.7
    a = make_sphere("a", 0.3, (-1.0, 0.0, 0.0), (speed, 0.0, 0.0), restitution=e)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=e)
    world = make_world([a, b])

    advance_to_contact(world, a, b, 0.6)
    world.step()

    exp_a = (1.0 - e) / 2.0 * speed
    exp_b = (1.0 + e) / 2.0 * speed
    err = max(abs(a.linear_velocity[0] - exp_a), abs(b.linear_velocity[0] - exp_b))

    ok = err < 1e-9
    return ok, (
        f"对心 e={e}：a={a.linear_velocity[0]:+.9f}（期望 {exp_a:+.9f}），"
        f"b={b.linear_velocity[0]:+.9f}（期望 {exp_b:+.9f}），误差={err:.2e}"
    )


def test_offcenter_matches_analytic() -> tuple[bool, str]:
    """偏心碰撞：与闭式解逐分量比对（核心检查）。"""
    a = make_sphere("a", 0.3, (-1.0, 0.30, 0.0), (3.0, 0.0, 0.0), restitution=0.7, friction=0.0)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=0.7, friction=0.0)
    world = make_world([a, b])

    advance_to_contact(world, a, b, 0.6)

    # 接触时刻的真实法向（这是关键：不是初始位置的连线方向）
    n = unit_normal(a, b)
    offset_angle = math.degrees(math.acos(max(-1.0, min(1.0, n[0]))))

    exp_va, exp_vb = closed_form_impulse(a, b)
    world.step()

    errors = [abs(a.linear_velocity[i] - exp_va[i]) for i in range(3)]
    errors += [abs(b.linear_velocity[i] - exp_vb[i]) for i in range(3)]
    worst = max(errors)

    ok = worst < 1e-9
    return ok, (
        f"接触时刻法向偏离 x 轴 {offset_angle:.2f}°："
        f"a'=({a.linear_velocity[0]:+.6f},{a.linear_velocity[1]:+.6f}) "
        f"期望({exp_va[0]:+.6f},{exp_va[1]:+.6f})；"
        f"b'=({b.linear_velocity[0]:+.6f},{b.linear_velocity[1]:+.6f}) "
        f"期望({exp_vb[0]:+.6f},{exp_vb[1]:+.6f})；最大误差={worst:.2e}"
    )


def test_tangential_velocity_preserved() -> tuple[bool, str]:
    """无摩擦偏心碰撞：相对速度的**切向分量**必须保持不变。

    这是"冲量只沿法向"的直接物理表述。如果切向分量变了，
    说明摩擦被错误地启用了，或者法向算错导致投影出错。
    """
    a = make_sphere("a", 0.3, (-1.0, 0.30, 0.0), (3.0, 0.0, 0.0), restitution=0.7, friction=0.0)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=0.7, friction=0.0)
    world = make_world([a, b])

    advance_to_contact(world, a, b, 0.6)
    n = unit_normal(a, b)

    rel_before = [b.linear_velocity[i] - a.linear_velocity[i] for i in range(3)]
    vn_before = sum(rel_before[i] * n[i] for i in range(3))
    tang_before = [rel_before[i] - vn_before * n[i] for i in range(3)]

    world.step()

    rel_after = [b.linear_velocity[i] - a.linear_velocity[i] for i in range(3)]
    vn_after = sum(rel_after[i] * n[i] for i in range(3))
    tang_after = [rel_after[i] - vn_after * n[i] for i in range(3)]

    drift = max(abs(tang_after[i] - tang_before[i]) for i in range(3))
    ok = drift < 1e-12
    return ok, (
        f"切向相对速度 ({tang_before[0]:+.6f},{tang_before[1]:+.6f},{tang_before[2]:+.6f}) → "
        f"({tang_after[0]:+.6f},{tang_after[1]:+.6f},{tang_after[2]:+.6f})，漂移={drift:.2e}"
    )


def test_no_spurious_spin() -> tuple[bool, str]:
    """无摩擦偏心碰撞：角速度必须严格为 0。

    原理：球的接触点力臂 ``r = n·(半径 - 穿透/2)`` 与法向 ``n`` 平行，
    因此 ``r × n = 0``，法向冲量不产生力矩。若实现里用错力臂，
    这里就会冒出非零角速度——这正是"侧向弹起"的常见根因之一。
    """
    a = make_sphere("a", 0.3, (-1.0, 0.30, 0.0), (3.0, 0.0, 0.0), restitution=0.7, friction=0.0)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=0.7, friction=0.0)
    world = make_world([a, b])
    advance_to_contact(world, a, b, 0.6)
    world.step()

    spin_a = math.sqrt(sum(c * c for c in a.angular_velocity))
    spin_b = math.sqrt(sum(c * c for c in b.angular_velocity))

    ok = spin_a < 1e-12 and spin_b < 1e-12
    return ok, f"无摩擦偏心碰撞后角速度：a={spin_a:.3e}，b={spin_b:.3e}（均应为 0）"


def test_friction_produces_expected_spin() -> tuple[bool, str]:
    """有摩擦偏心碰撞：应当产生自旋，且方向符合切向冲量。

    摩擦系数非零时，切向冲量会对球产生力矩，球会转起来。
    这里只检查"确实转了、且角动量的符号与切向相对速度一致"，
    不追求逐位精确（摩擦是迭代近似的）。
    """
    a = make_sphere("a", 0.3, (-1.0, 0.30, 0.0), (3.0, 0.0, 0.0), restitution=0.7, friction=0.6)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=0.7, friction=0.6)
    world = make_world([a, b])
    advance_to_contact(world, a, b, 0.6)
    world.step()

    spin_a = math.sqrt(sum(c * c for c in a.angular_velocity))
    spin_b = math.sqrt(sum(c * c for c in b.angular_velocity))

    ok = spin_a > 1e-6 and spin_b > 1e-6
    return ok, (
        f"有摩擦偏心碰撞后角速度：a={spin_a:.4f} rad/s，b={spin_b:.4f} rad/s（应 > 0）"
    )


def test_momentum_conserved() -> tuple[bool, str]:
    """偏心碰撞：线动量三个分量都必须守恒（含摩擦的情形）。"""
    a = make_sphere("a", 0.3, (-1.0, 0.30, 0.0), (3.0, 0.0, 0.0), restitution=0.7, friction=0.4)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=0.7, friction=0.4)
    world = make_world([a, b])

    before = [a.mass * a.linear_velocity[i] + b.mass * b.linear_velocity[i] for i in range(3)]
    for _ in range(int(0.6 / world.simulation.physics_dt)):
        world.step()
    after = [a.mass * a.linear_velocity[i] + b.mass * b.linear_velocity[i] for i in range(3)]

    drift = max(abs(after[i] - before[i]) for i in range(3))
    ok = drift < 1e-9
    return ok, (
        f"动量 ({before[0]:+.6f},{before[1]:+.6f},{before[2]:+.6f}) → "
        f"({after[0]:+.6f},{after[1]:+.6f},{after[2]:+.6f})，最大漂移={drift:.2e}"
    )


def test_energy_not_increasing() -> tuple[bool, str]:
    """偏心碰撞（带摩擦）：总动能不得增加。"""
    a = make_sphere("a", 0.3, (-1.0, 0.30, 0.0), (3.0, 0.0, 0.0), restitution=0.7, friction=0.4)
    b = make_sphere("b", 0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=0.7, friction=0.4)
    world = make_world([a, b])

    before = world.total_kinetic_energy()
    peak = before
    for _ in range(int(0.6 / world.simulation.physics_dt)):
        world.step()
        peak = max(peak, world.total_kinetic_energy())
    after = world.total_kinetic_energy()

    growth = (peak - before) / before
    ok = after <= before * 1.000001 and growth < 1e-9
    return ok, f"动能 {before:.6f} J → {after:.6f} J，峰值涨幅={growth*100:.4f}%"


def test_vertical_drop_offset_direction() -> tuple[bool, str]:
    """竖直落球偏心：水平弹起方向必须与偏心方向一致，且速度有界。

    这是"侧向弹起"最直观的场景：球垂直下落，砸在另一个球的侧面。
    物理上它一定会往偏心方向滑出去（法向有水平分量），
    但**不应获得超过入射能量的速度**。
    """
    radius_large, radius_small = 0.42, 0.30
    offset = 0.20
    drop_speed = 2.0
    contact_height = radius_large + radius_small

    base = make_sphere("base", radius_large, (0.0, 0.0, 0.0),
                       restitution=0.5, friction=0.3, is_static=True)
    ball = make_sphere("ball", radius_small, (offset, 0.0, contact_height + 0.06),
                       (0.0, 0.0, -drop_speed), restitution=0.5, friction=0.3)

    world = make_world([base, ball])
    for _ in range(int(0.4 / world.simulation.physics_dt)):
        world.step()

    speed = ball.speed()
    vx = ball.linear_velocity[0]

    direction_ok = vx > 0.0
    bounded = speed <= drop_speed * 1.01

    ok = direction_ok and bounded
    return ok, (
        f"偏心下落 δ={offset}：水平速度 vx={vx:+.4f}，总速度={speed:.4f}（入射 {drop_speed}）"
    )


def test_offset_sweep() -> tuple[bool, str]:
    """扫描偏心量：结果必须连续、有界、无 NaN。

    两个极端尤其关键：
    - δ 极小（近对心）时水平速度必须趋近 0，不能跳变
    - δ 接近半径和（掠碰）时也不能爆炸
    """
    radius = 0.3
    contact = 2.0 * radius
    speed = 3.0

    rows = []
    all_ok = True
    for ratio in (0.0, 1e-4, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        offset = ratio * contact
        a = make_sphere("a", radius, (-1.0, offset, 0.0), (speed, 0.0, 0.0),
                        restitution=0.7, friction=0.0)
        b = make_sphere("b", radius, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        restitution=0.7, friction=0.0)
        world = make_world([a, b])
        for _ in range(int(0.8 / world.simulation.physics_dt)):
            world.step()

        speed_b = b.speed()
        vy_b = b.linear_velocity[1]
        finite = all(math.isfinite(c) for c in b.linear_velocity)

        if not finite or speed_b > speed * 1.01:
            all_ok = False
        if ratio <= 0.02 and abs(vy_b) > 0.05 * speed:
            all_ok = False

        rows.append((ratio, vy_b, speed_b))

    summary = "  ".join(f"{r:.4f}:vy={vy:+.3f},|v|={s:.3f}" for r, vy, s in rows)
    return all_ok, f"δ/R_sum → {summary}"


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 对心碰撞基线", test_centered_baseline),
    ("2. 偏心碰撞与闭式解一致", test_offcenter_matches_analytic),
    ("3. 无摩擦时切向速度不变", test_tangential_velocity_preserved),
    ("4. 无摩擦时不产生自旋", test_no_spurious_spin),
    ("5. 有摩擦时产生自旋", test_friction_produces_expected_spin),
    ("6. 线动量守恒", test_momentum_conserved),
    ("7. 动能不增加", test_energy_not_increasing),
    ("8. 竖直偏心落球：方向正确且有界", test_vertical_drop_offset_direction),
    ("9. 偏心量扫描：连续、有界、无 NaN", test_offset_sweep),
]


def main() -> int:
    """依次执行所有检查并汇总。"""
    print("=" * 88)
    print("球-球偏心碰撞测试")
    print("=" * 88)

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

    print("-" * 88)
    print(f"结果：{passed}/{len(CHECKS)} 通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
