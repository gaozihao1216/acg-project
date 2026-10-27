# -*- coding: utf-8 -*-
"""W2 第一步的数值验证：球-球 / 球-地面 + 顺序冲量求解器。

为什么先做数值验证再做渲染
--------------------------
渲染只能告诉你"看起来怪"，但看不出是穿透、能量注入、还是法向反号。
下面每一项都把物理性质写成可断言的条件，出问题时能立刻定位到是哪一条不满足。

直接用系统 Python 运行（不需要 Blender）::

    python Code/tests/test_rigid_sphere.py

覆盖的性质
----------
1. 自由下落与解析解一致          —— 验证积分器
2. 落地后能真正静置              —— 验证 Baumgarte + 恢复系数阈值
3. 高速下落不会穿透地面          —— 验证子步与检测顺序
4. 总机械能不会凭空增加          —— 验证冲量符号与摩擦限幅
5. 弹性碰撞的动量守恒与速度交换  —— 验证恢复系数与有效质量公式
6. 多球场景不炸、不穿、能收敛    —— 综合回归
"""

from __future__ import annotations

import os
import sys

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


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------

def make_sphere(
    name: str,
    radius: float,
    position,
    velocity=(0.0, 0.0, 0.0),
    mass: float | None = None,
    restitution: float = 0.2,
    friction: float = 0.5,
    is_static: bool = False,
) -> RigidBody:
    """按给定参数造一个球刚体，质量缺省按水密度估算。"""
    geometry = SphereGeometry(radius)
    if mass is None:
        mass = geometry.volume() * 1000.0
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name,
        geometry=geometry,
        position=position,
        mass=mass,
        inertia_local=inertia,
        restitution=restitution,
        friction=friction,
        is_static=is_static,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(
    bodies,
    ground: GroundPlane | None = None,
    dt: float = 1.0 / 240.0,
    substeps: int = 2,
    iterations: int = 12,
    baumgarte: float = 0.25,
    slop: float = 0.002,
    restitution_threshold: float = 1.0,
    linear_damping: float = 0.05,
    angular_damping: float = 1.0,
) -> PhysicsWorld:
    """按指定参数组一个物理世界。"""
    sim = SimulationSpec(
        enabled=True,
        physics_dt=dt,
        substeps=substeps,
        iterations=iterations,
        baumgarte=baumgarte,
        slop=slop,
        restitution_threshold=restitution_threshold,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), ground, sim)


def advance(world: PhysicsWorld, seconds: float) -> None:
    """把世界推进指定时长（按 physics_dt 逐步走）。"""
    steps = int(round(seconds / world.simulation.physics_dt))
    for _ in range(steps):
        world.step()


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------

def test_free_fall() -> tuple[bool, str]:
    """自由下落：与解析解比较。

    半隐式欧拉在常加速度下的位移误差约为 ``0.5 * g * h * t``，
    在 h = 1/480、t = 0.5 时约 5 毫米，因此容差取 2 厘米。
    """
    z0 = 10.0
    # 关掉阻尼，否则解析解（不含阻力）对不上
    world = make_world(
        [make_sphere("faller", 0.3, (0.0, 0.0, z0))],
        ground=None,
        linear_damping=0.0,
        angular_damping=0.0,
    )
    t = 0.5
    advance(world, t)

    body = world.bodies[0]
    expected = z0 - 0.5 * 9.81 * t * t
    error = abs(body.position[2] - expected)

    ok = error < 0.02
    return ok, f"下落 t={t}s：z={body.position[2]:.4f}，解析解={expected:.4f}，误差={error*1000:.1f} mm"


def test_rest_on_ground() -> tuple[bool, str]:
    """落地静置：速度收敛到接近零，且停在半径高度。"""
    radius = 0.3
    world = make_world(
        [make_sphere("ball", radius, (0.0, 0.0, 1.0), restitution=0.2, friction=0.6)],
        ground=GroundPlane(z=0.0, friction=0.6, restitution=0.2),
    )
    advance(world, 4.0)

    body = world.bodies[0]
    speed = body.speed()
    height_error = body.position[2] - radius
    penetration = world.max_penetration()

    ok = speed < 0.05 and abs(height_error) < 0.01 and penetration < 0.02
    return ok, (
        f"静置 4s：速度={speed:.4f} m/s，z={body.position[2]:.4f}（期望 {radius}），"
        f"高度误差={height_error*1000:.2f} mm，最大穿透={penetration*1000:.2f} mm"
    )


def test_no_tunneling() -> tuple[bool, str]:
    """高速下落：任何时刻都不应穿到地面以下。"""
    radius = 0.25
    world = make_world(
        [make_sphere("bullet", radius, (0.0, 0.0, 3.0), velocity=(0.0, 0.0, -15.0),
                     restitution=0.3, friction=0.4)],
        ground=GroundPlane(z=0.0, friction=0.4, restitution=0.3),
    )

    min_z = world.bodies[0].position[2]
    for _ in range(int(2.0 / world.simulation.physics_dt)):
        world.step()
        min_z = min(min_z, world.bodies[0].position[2])

    # 允许不到 1 厘米的瞬时穿透（slop 量级），但不允许"穿过去"
    ok = min_z > -0.01
    return ok, f"初速 -15 m/s 撞地：全程最低球心 z={min_z:.5f}（下限 -0.01）"


def test_energy_not_increasing() -> tuple[bool, str]:
    """总机械能不得凭空增加。

    恢复系数 0.5 意味着每次反弹都损失能量，所以终态能量必须低于初态。
    Baumgarte 会把穿透"推出来"，理论上会注入一点能量，因此允许极小的
    相对增量，但绝不允许整体上升。
    """
    world = make_world(
        [make_sphere("ball", 0.3, (0.0, 0.0, 2.0), restitution=0.5, friction=0.5)],
        ground=GroundPlane(z=0.0, friction=0.5, restitution=0.5),
    )

    initial = world.total_energy()
    peak = initial
    for _ in range(int(4.0 / world.simulation.physics_dt)):
        world.step()
        peak = max(peak, world.total_energy())
    final = world.total_energy()

    growth = (peak - initial) / max(1e-9, abs(initial))
    ok = final <= initial * 1.001 and growth < 0.05
    return ok, (
        f"能量：初={initial:.3f} J，峰={peak:.3f} J，终={final:.3f} J，"
        f"峰值涨幅={growth*100:.2f}%"
    )


def test_elastic_swap() -> tuple[bool, str]:
    """等质量弹性对撞：速度应完全交换，动量守恒。

    这个用例专门关掉 Baumgarte（``baumgarte=0``），因为位置修正会额外注入
    分离速度，掩盖恢复系数本身的正确性。属于"分离变量"的调试手法。
    """
    radius = 0.3
    speed = 3.0
    world = make_world(
        [
            make_sphere("left", radius, (-1.0, 0.0, 0.0), velocity=(speed, 0.0, 0.0),
                        restitution=1.0, friction=0.0),
            make_sphere("right", radius, (1.0, 0.0, 0.0), velocity=(-speed, 0.0, 0.0),
                        restitution=1.0, friction=0.0),
        ],
        ground=None,
        baumgarte=0.0,
        restitution_threshold=0.1,
        # 弹性碰撞要求速度严格交换，任何阻尼都会让结果偏离，必须关掉
        linear_damping=0.0,
        angular_damping=0.0,
    )

    momentum_before = sum(b.mass * b.linear_velocity[0] for b in world.bodies)
    advance(world, 0.6)

    left, right = world.bodies
    momentum_after = sum(b.mass * b.linear_velocity[0] for b in world.bodies)

    # 等质量一维弹性碰撞后速度交换：left 应为 -3，right 应为 +3
    left_ok = abs(left.linear_velocity[0] - (-speed)) < 0.15
    right_ok = abs(right.linear_velocity[0] - speed) < 0.15
    momentum_ok = abs(momentum_after - momentum_before) < 1e-6 * abs(momentum_before) + 1e-6

    ok = left_ok and right_ok and momentum_ok
    return ok, (
        f"弹性对撞：left vx={left.linear_velocity[0]:+.4f}（期望 {-speed:+.1f}），"
        f"right vx={right.linear_velocity[0]:+.4f}（期望 {speed:+.1f}），"
        f"动量 {momentum_before:+.6f} → {momentum_after:+.6f}"
    )


def test_pile_settles() -> tuple[bool, str]:
    """多球综合回归：不炸、不穿、能收敛。"""
    bodies = [
        make_sphere("b0", 0.32, (0.35, -0.25, 2.20), restitution=0.55, friction=0.35),
        make_sphere("b1", 0.32, (-0.35, -0.15, 2.00), restitution=0.10, friction=0.70),
        make_sphere("b2", 0.28, (0.00, 0.30, 1.70), restitution=0.40, friction=0.40),
        make_sphere("b3", 0.20, (-0.60, 0.35, 2.40), restitution=0.35, friction=0.50),
        make_sphere("b4", 0.20, (0.65, 0.40, 2.50), restitution=0.35, friction=0.50),
    ]
    world = make_world(
        bodies,
        ground=GroundPlane(z=0.0, friction=0.55, restitution=0.15),
        iterations=12,
    )

    min_z = min(b.position[2] for b in bodies)
    for _ in range(int(5.0 / world.simulation.physics_dt)):
        world.step()
        min_z = min(min_z, *(b.position[2] for b in bodies))

    world.check_finite()

    # 每个球都不应沉到地面以下（允许 slop 量级）
    lowest_surface = min(b.position[2] - b.geometry.radius for b in bodies)
    max_speed = world.max_speed()
    penetration = world.max_penetration()

    ok = lowest_surface > -0.01 and max_speed < 0.6 and penetration < 0.02
    return ok, (
        f"5 球 5 秒：最低表面 z={lowest_surface*1000:.2f} mm，"
        f"最大速度={max_speed:.4f} m/s，最大穿透={penetration*1000:.2f} mm，"
        f"全程最低球心 z={min_z:.4f}"
    )


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 自由下落与解析解一致", test_free_fall),
    ("2. 落地后能静置", test_rest_on_ground),
    ("3. 高速不穿透地面", test_no_tunneling),
    ("4. 总机械能不增加", test_energy_not_increasing),
    ("5. 弹性碰撞速度交换", test_elastic_swap),
    ("6. 多球场景收敛", test_pile_settles),
]


def main() -> int:
    """依次执行所有检查并汇总结果。"""
    print("=" * 82)
    print("W2 第一步 · 刚体球碰撞与顺序冲量求解器 · 数值验证")
    print("=" * 82)

    passed = 0
    for title, func in CHECKS:
        try:
            ok, message = func()
        except Exception as exc:  # noqa: BLE001
            ok, message = False, f"抛出异常：{type(exc).__name__}: {exc}"

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {title}")
        print(f"       {message}")
        if ok:
            passed += 1

    print("-" * 82)
    print(f"结果：{passed}/{len(CHECKS)} 通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
