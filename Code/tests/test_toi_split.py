# -*- coding: utf-8 -*-
"""TOI 分段求解与初始状态校验的专项测试。

背景
----
窄相只看得到"此刻已穿透"的物体。一个子步内若先后发生多次碰撞
（A 撞 B、B 再撞 C），一次性检测 + 一次联立求解会把它们当作"同时接触"，
于是得到"整体同时变形"的解——那是软材料的物理。硬物（钢球等）的正确行为
是**接触逐个传递**。

本文件验证两件事：

1. ``toi`` 模块的预测是否正确（与解析解比对）
2. 开启分段后，深穿透是否被消除、链式碰撞是否按正确次序传递
3. 初始就互相嵌入的场景是否被拒绝

运行::

    python Code/tests/test_toi_split.py
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
from src.geom.transform import TransformSpec
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.toi import sphere_ground_toi, sphere_sphere_toi
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import MaterialSpec, ObjectSpec, SceneSpec, SimulationSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_sphere(name, radius, position, velocity=(0.0, 0.0, 0.0),
                mass=1.0, restitution=0.7, friction=0.0) -> RigidBody:
    """造一个球刚体。"""
    geometry = SphereGeometry(radius)
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name, geometry=geometry, position=position,
        mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(bodies, ground=None, dt: float = 1.0 / 240.0, substeps: int = 1,
               use_toi_splitting: bool = True, max_events: int = 8,
               iterations: int = 12, gravity=(0.0, 0.0, 0.0)) -> PhysicsWorld:
    """组一个可开关 TOI 分段的物理世界。"""
    sim = SimulationSpec(
        enabled=True,
        physics_dt=dt,
        substeps=substeps,
        iterations=iterations,
        baumgarte=0.0,
        slop=0.0,
        restitution_threshold=0.05,
        gravity=gravity,
        linear_damping=0.0,
        angular_damping=0.0,
        use_toi_splitting=use_toi_splitting,
        max_events_per_substep=max_events,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), ground, sim)


def max_overlap_depth(world: PhysicsWorld) -> float:
    """当前场景中最深的穿透（米）。"""
    manifolds = detect_contacts(world.bodies, world.ground)
    if not manifolds:
        return 0.0
    return max(m.deepest.penetration for m in manifolds)


def build_high_speed_chain():
    """构造"一个子步内连续撞三次"的链式场景。

    半径 0.3，表面间隙 0.1 m，左球以 60 m/s 撞入。
    子步 h = 1/240 ≈ 4.17 ms，而 60 m/s 在 4.17 ms 内走 0.25 m ——
    远大于 0.1 m 的间隙，因此**两次碰撞都发生在同一个子步内部**。

    解析：
        第 1 次接触时刻 t₁ = 0.1 / 60 = 1.667 ms
        第 2 次接触时刻 t₂ = 2t₁ = 3.333 ms   （两者都 < 4.17 ms）
        恢复系数 1、等质量 ⇒ 速度依次传递，终态应为 [0, 0, 60]
    """
    radius = 0.3
    speed = 60.0
    positions = (-0.7, 0.0, 0.7)

    bodies = [
        make_sphere("a", radius, (positions[0], 0.0, 0.0), (speed, 0.0, 0.0), restitution=1.0),
        make_sphere("b", radius, (positions[1], 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
        make_sphere("c", radius, (positions[2], 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
    ]
    return bodies, speed


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------

def test_toi_matches_analytic() -> tuple[bool, str]:
    """TOI 预测应与解析解一致。"""
    radius = 0.3
    speed = 6.0
    height = 2.0

    body = make_sphere("ball", radius, (0.0, 0.0, height), (0.0, 0.0, -speed))
    ground = GroundPlane(z=0.0)

    predicted = sphere_ground_toi(body, ground, 1.0)
    expected = (height - radius) / speed

    err = abs(predicted - expected) if predicted is not None else 999.0
    return err < 1e-12, (
        f"球-地面 TOI：预测={predicted}，解析={expected:.9f}，误差={err:.2e}"
    )


def test_sphere_sphere_toi_matches_analytic() -> tuple[bool, str]:
    """球-球 TOI 应与二次方程解析解一致。"""
    radius = 0.3
    speed = 40.0

    a = make_sphere("a", radius, (-1.0, 0.0, 0.0), (speed, 0.0, 0.0))
    b = make_sphere("b", radius, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    predicted = sphere_sphere_toi(a, b, 1.0)
    expected = (1.0 - 2.0 * radius) / speed        # 需要靠近 1.0 - 0.6 = 0.4 m

    err = abs(predicted - expected) if predicted is not None else 999.0
    return err < 1e-12, (
        f"球-球 TOI：预测={predicted}，解析={expected:.9f}，误差={err:.2e}"
    )


def test_chain_within_one_substep() -> tuple[bool, str]:
    """一个子步内的两次碰撞必须按次序分别传递速度。"""
    bodies, speed = build_high_speed_chain()
    world = make_world(bodies, use_toi_splitting=True)
    world.step()

    values = [round(b.linear_velocity[0], 3) for b in bodies]
    events = world.last_event_count

    # 期望：速度完全传递到最右端的球
    ok = (
        abs(values[0]) < 0.5
        and abs(values[1]) < 0.5
        and abs(values[2] - speed) < 1.5
    )
    return ok, (
        f"单子步内两次碰撞：终态 vx={values}（期望 [0, 0, {speed:.0f}]），"
        f"触发 {events} 次 TOI 事件"
    )


def test_penetration_reduction() -> tuple[bool, str]:
    """开启 TOI 分段后，链式碰撞的最大穿透应大幅下降。

    这是分段求解最直接的收益：不开分段时，物体必须先"陷进"对方，
    窄相才看得见它，穿透深度会达到步长位移量级（本例 0.25 m，接近一个半径）。
    """
    results = {}
    for enabled in (False, True):
        bodies, _ = build_high_speed_chain()
        world = make_world(bodies, use_toi_splitting=enabled)
        deepest = 0.0
        for _ in range(6):
            world.step()
            deepest = max(deepest, max_overlap_depth(world))
        results[enabled] = deepest

    without = results[False]
    with_toi = results[True]
    ok = with_toi < without * 0.1
    return ok, (
        f"链式碰撞最大穿透：不分段={without * 1000:.2f} mm，"
        f"分段={with_toi * 1000:.3f} mm（降低到 {with_toi / max(without, 1e-12) * 100:.2f}%）"
    )


def test_event_cap_no_hang() -> tuple[bool, str]:
    """事件数上限必须能防住病态场景（否则可能死循环）。"""
    radius = 0.3
    speed = 200.0
    bodies = [
        make_sphere("a", radius, (-0.75, 0.0, 0.0), (speed, 0.0, 0.0), restitution=1.0),
        make_sphere("b", radius, (-0.05, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
        make_sphere("c", radius, (0.65, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
        make_sphere("d", radius, (1.35, 0.0, 0.0), (0.0, 0.0, 0.0), restitution=1.0),
    ]
    world = make_world(bodies, use_toi_splitting=True, max_events=1)

    for _ in range(20):
        world.step()

    finite = all(math.isfinite(c) for b in bodies for c in b.linear_velocity)
    capped = world.last_event_count <= 1
    ok = finite and capped
    return ok, (
        f"max_events=1、四球 200 m/s：有限={finite}，末步事件数={world.last_event_count}，"
        f"终态 vx={[round(b.linear_velocity[0], 2) for b in bodies]}"
    )


def test_resting_has_no_events() -> tuple[bool, str]:
    """静置物体不应持续触发 TOI 事件（否则会白白拖慢仿真）。"""
    radius = 0.3
    body = make_sphere("ball", radius, (0.0, 0.0, 1.2), restitution=0.2, friction=0.5)
    ground = GroundPlane(z=0.0, friction=0.5, restitution=0.2)
    world = make_world(
        [body], ground=ground, gravity=(0.0, 0.0, -9.81), substeps=2
    )

    # 先让它落定
    for _ in range(int(2.0 / world.simulation.physics_dt)):
        world.step()

    # 统计静置阶段的平均事件数
    total_events = 0
    samples = 200
    for _ in range(samples):
        world.step()
        total_events += world.last_event_count

    average = total_events / samples
    ok = average < 0.2
    return ok, (
        f"静置球 {samples} 步：平均每步 {average:.3f} 次 TOI 事件，"
        f"z={body.position[2]:.6f}（半径 {radius}）"
    )


def test_initial_overlap_rejected() -> tuple[bool, str]:
    """初始就互相嵌入的场景必须被拒绝，且可以被显式放行。"""
    def build(allow: bool) -> SceneSpec:
        objects = [
            ObjectSpec(
                name="a",
                geometry=SphereGeometry(0.3),
                transform=TransformSpec.from_dict({"position": [-0.1, 0.0, 0.0]}),
                material=MaterialSpec(),
            ),
            ObjectSpec(
                name="b",
                geometry=SphereGeometry(0.3),
                transform=TransformSpec.from_dict({"position": [0.1, 0.0, 0.0]}),
                material=MaterialSpec(),
            ),
        ]
        return SceneSpec(
            name="overlap",
            objects=objects,
            simulation=SimulationSpec(enabled=True, allow_initial_overlap=allow),
        )

    rejected = False
    message = ""
    try:
        PhysicsWorld.from_scene(build(allow=False))
    except ValueError as exc:
        rejected = True
        message = str(exc).splitlines()[0]

    allowed_ok = False
    try:
        PhysicsWorld.from_scene(build(allow=True))
        allowed_ok = True
    except ValueError:
        allowed_ok = False

    ok = rejected and allowed_ok
    return ok, (
        f"重叠 0.4 m 的两球：默认被拒绝={rejected}（{message}）；"
        f"显式 allow_initial_overlap 后放行={allowed_ok}"
    )


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 球-地面 TOI 与解析解一致", test_toi_matches_analytic),
    ("2. 球-球 TOI 与解析解一致", test_sphere_sphere_toi_matches_analytic),
    ("3. 单子步内链式碰撞次序正确", test_chain_within_one_substep),
    ("4. 分段大幅降低穿透深度", test_penetration_reduction),
    ("5. 事件上限防死循环", test_event_cap_no_hang),
    ("6. 静置不产生多余事件", test_resting_has_no_events),
    ("7. 初始嵌入被拒绝", test_initial_overlap_rejected),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("TOI 分段求解与初始状态校验")
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
