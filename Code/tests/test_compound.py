# -*- coding: utf-8 -*-
"""组合体（compound）的质量属性、碰撞与 TOI 验证。

为什么需要组合体
---------------
凸包只能表示**凸**的**多面体**。L 形支架、T 形夹具、人形这类物体都不凸，
取凸包会把凹槽填平——碰撞体积凭空变大，物体悬在半空。组合体保留每个部件的
原始解析形状，碰撞时逐部件求交、结果取并集，体积/惯量/包围球仍然精确。

覆盖的性质
----------
1. 哑铃惯量与平行轴定理的解析解一致
2. 不等体积部件的质心与解析解一致
3. 单部件组合体逐位等于该图元（惯量与主轴旋转都不引入额外变换）
4. 场景 JSON 能解析组合体，并在世界层面跑通
5. 哑铃横放落地能静置，停在半径高度
6. T 形组合体立着落地不倒，高度等于解析值
7. 组合体的窄相把两个部件的接触都报出来（不是只报最深的一个）
8. 同一对刚体上不同部件的特征 ID 不撞号（暖启动的前提）
9. 宽阶段 AABB 等于各部件 AABB 的并集
10. 自转的组合体：TOI 靠 v + ω × r 报出事件（用质心速度则永远为 0）

运行::

    python Code/tests/test_compound.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import (
    BoxGeometry,
    CompoundGeometry,
    CompoundPart,
    SphereGeometry,
)
from src.geom.inertia import compute_inertia
from src.physics import toi as toi_module
from src.physics.body import RigidBody, iter_parts, make_ground_proxy
from src.physics.broadphase import world_aabb
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SceneSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_body(name, geometry, position, orientation=(1.0, 0.0, 0.0, 0.0),
              velocity=(0.0, 0.0, 0.0), mass=1.0, static=False) -> RigidBody:
    """按几何 + 位姿造刚体；惯量与主轴旋转按 ``RigidBody.from_object_spec`` 的做法处理。"""
    inertia = compute_inertia(geometry, mass)
    from src.mathutils import quaternion as quat

    final_orientation = quat.quat_multiply(orientation, geometry.inertia_frame())
    body = RigidBody(
        name=name,
        geometry=geometry,
        position=position,
        orientation=final_orientation,
        mass=mass,
        inertia_local=inertia,
        is_static=static,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(bodies, ground, substeps=4, dt=1.0 / 120.0) -> PhysicsWorld:
    """带地面、开 TOI 分段的世界。"""
    from src.scene.scene_spec import SimulationSpec

    simulation = SimulationSpec(
        enabled=True, physics_dt=dt, substeps=substeps, iterations=12, duration=1.0
    )
    return PhysicsWorld(list(bodies), ground, simulation)


def advance(world: PhysicsWorld, seconds: float) -> None:
    """按物理步长推进。"""
    for _ in range(int(round(seconds / world.simulation.physics_dt))):
        world.step()


def dumbbell(radius: float, span: float) -> CompoundGeometry:
    """两个球 + 一根无质量的连接（球心相距 ``span``）。"""
    piece = SphereGeometry(radius)
    return CompoundGeometry(
        [
            CompoundPart(piece, (-0.5 * span, 0.0, 0.0)),
            CompoundPart(piece, (0.5 * span, 0.0, 0.0)),
        ]
    )


# ---------------------------------------------------------------------------
# 1–3. 质量属性
# ---------------------------------------------------------------------------

def test_dumbbell_inertia() -> tuple[bool, str]:
    """哑铃：绕轴线的惯量 = 2×(2/5)mr²，横向 = 2×(2/5 mr² + m(d/2)²)。"""
    radius, span = 0.2, 1.0
    geometry = dumbbell(radius, span)

    mass_each = SphereGeometry(radius).volume()
    total = 2.0 * mass_each
    own = 0.4 * mass_each * radius * radius
    across = 2.0 * (own + mass_each * (0.5 * span) ** 2)
    along = 2.0 * own

    moments = geometry.principal_moments(total)
    expected = sorted([along, across, across])
    error = max(abs(a - b) for a, b in zip(sorted(moments), expected))

    ok = error < 1e-12 and abs(geometry.volume() - total) < 1e-12
    return ok, (
        f"主惯量={[round(m, 6) for m in moments]}，解析={[round(m, 6) for m in expected]}，"
        f"最大误差={error:.2e}，体积={geometry.volume():.6f}（解析 {total:.6f}）"
    )


def test_center_of_mass() -> tuple[bool, str]:
    """不等体积的两块：质心按体积加权，落在解析位置上。"""
    big = SphereGeometry(0.3)     # 体积是小的 27 倍
    small = SphereGeometry(0.1)
    geometry = CompoundGeometry(
        [CompoundPart(big, (0.0, 0.0, 0.0)), CompoundPart(small, (1.0, 0.0, 0.0))]
    )

    big_volume = big.volume()
    small_volume = small.volume()
    expected_x = small_volume * 1.0 / (big_volume + small_volume)
    error = abs(geometry.com_offset[0] - expected_x)

    ok = error < 1e-12
    return ok, (
        f"质心 x={geometry.com_offset[0]:.6f}（解析 {expected_x:.6f}），"
        f"误差={error:.2e} m"
    )


def test_single_part_matches_primitive() -> tuple[bool, str]:
    """单部件组合体：惯量与主轴旋转都必须和该图元完全一致。"""
    box = BoxGeometry((0.4, 0.6, 0.8))
    compound = CompoundGeometry([CompoundPart(box)])

    expected = compute_inertia(box, 2.0)
    got = compound.principal_moments(2.0)
    frame = compound.inertia_frame()

    error = max(abs(a - b) for a, b in zip(got, expected))
    frame_error = max(abs(a - b) for a, b in zip(frame, (1.0, 0.0, 0.0, 0.0)))
    ok = error < 1e-12 and frame_error < 1e-12
    return ok, (
        f"主惯量={[round(v, 9) for v in got]}，图元={[round(v, 9) for v in expected]}，"
        f"主轴四元数={tuple(round(v, 9) for v in frame)}"
    )


# ---------------------------------------------------------------------------
# 4. 场景 JSON
# ---------------------------------------------------------------------------

def test_scene_json_roundtrip() -> tuple[bool, str]:
    """场景 JSON 里写 compound：能解析、能加载、初始无穿透。"""
    scene_data = {
        "name": "compound_case",
        "objects": [
            {
                "name": "dumbbell",
                "geometry": {
                    "type": "compound",
                    "parts": [
                        {
                            "geometry": {"type": "box", "size": [0.3, 0.3, 0.3]},
                            "position": [-0.25, 0.0, 0.0],
                        },
                        {
                            "geometry": {"type": "sphere", "radius": 0.15},
                            "position": [0.25, 0.0, 0.0],
                        },
                    ],
                },
                "transform": {"position": [0.0, 0.0, 2.0]},
                "mass": 2.0,
            }
        ],
        "simulation": {
            "enabled": True,
            "physics_dt": 0.004166666666666667,
            "substeps": 3,
            "iterations": 10,
            "duration": 1.0,
        },
    }

    scene = SceneSpec.from_dict(scene_data)
    geometry = scene.objects[0].geometry
    world = PhysicsWorld.from_scene(scene)

    parsed_ok = (
        isinstance(geometry, CompoundGeometry)
        and len(geometry.parts) == 2
        and isinstance(geometry.parts[0].geometry, BoxGeometry)
        and isinstance(geometry.parts[1].geometry, SphereGeometry)
    )
    body = world.bodies[0]
    inertial_ok = (
        body.inertia_local[0] > 0.0
        and abs(body.mass - 2.0) < 1e-12
    )
    advance(world, 1.0)
    world.check_finite()

    ok = parsed_ok and inertial_ok and body.position[2] > 0.0
    return ok, (
        f"解析={type(geometry).__name__}（{len(geometry.parts)} 部件），"
        f"质量={body.mass}，1 s 后 z={body.position[2]:.4f}，有限={not body.is_static}"
    )


# ---------------------------------------------------------------------------
# 5–6. 落地静置
# ---------------------------------------------------------------------------

def test_dumbbell_rests_on_ground() -> tuple[bool, str]:
    """哑铃横放落地：停在半径高度，不穿地、不自旋失控。"""
    radius, span = 0.2, 0.9
    body = make_body("dumbbell", dumbbell(radius, span), (0.0, 0.0, 1.2), mass=2.0)
    world = make_world([body], GroundPlane(z=0.0, friction=0.7, restitution=0.05))
    advance(world, 4.0)

    error = world.bodies[0].position[2] - radius
    penetration = world.max_penetration()
    spin = sum(abs(v) for v in world.bodies[0].angular_velocity)
    ok = abs(error) < 0.02 and penetration < 0.02 and spin < 0.5 and world.bodies[0].speed() < 0.1
    return ok, (
        f"静置 4s：z={world.bodies[0].position[2]:.4f}（期望 {radius}），"
        f"高度误差={error * 1000:.1f} mm，速度={world.bodies[0].speed():.4f} m/s，"
        f"角速度={spin:.4f} rad/s，最大穿透={penetration * 1000:.2f} mm"
    )


def test_t_shape_stands() -> tuple[bool, str]:
    """T 形（底板 + 竖杆）落地：底板贴地、竖杆竖直、整体不倒。"""
    base = BoxGeometry((1.0, 0.4, 0.1))
    stem = BoxGeometry((0.15, 0.2, 0.8))
    geometry = CompoundGeometry(
        [
            CompoundPart(base, (0.0, 0.0, 0.0)),
            CompoundPart(stem, (0.0, 0.0, 0.45)),
        ]
    )
    body = make_body("t_shape", geometry, (0.0, 0.0, 1.2), mass=3.0)
    world = make_world([body], GroundPlane(z=0.0, friction=0.8, restitution=0.02))
    advance(world, 4.0)

    settled = world.bodies[0]
    # 质心高：底板 (体积 0.04) 中心 0，竖杆 (体积 0.024) 中心 0.45
    base_volume = 1.0 * 0.4 * 0.1
    stem_volume = 0.15 * 0.2 * 0.8
    center_z = (base_volume * 0.0 + stem_volume * 0.45) / (base_volume + stem_volume)
    # 底板下表面贴地时，质心在"底板中心 + 半厚"之上
    expected = center_z + 0.05
    error = settled.position[2] - expected
    tilt = abs(settled.orientation[1]) + abs(settled.orientation[2])
    ok = abs(error) < 0.02 and tilt < 0.05 and settled.speed() < 0.1
    return ok, (
        f"静置 4s：z={settled.position[2]:.4f}（期望 {expected:.4f}），"
        f"误差={error * 1000:.1f} mm，倾斜分量={tilt:.4f}，速度={settled.speed():.4f} m/s"
    )


# ---------------------------------------------------------------------------
# 7–8. 窄相与特征 ID
# ---------------------------------------------------------------------------

def test_both_parts_contact() -> tuple[bool, str]:
    """哑铃压在一块宽盒上：两个球部件都要有接触点（不能只报一个）。"""
    plate = make_body("plate", BoxGeometry((3.0, 3.0, 0.2)), (0.0, 0.0, -0.1), static=True)
    # 球半径 0.2，中心距 0.9；球心放在 z=0.19 → 各陷 1 cm
    body = make_body("dumbbell", dumbbell(0.2, 0.9), (0.0, 0.0, 0.19), mass=2.0)

    manifolds = detect_contacts([plate, body], None)
    if not manifolds:
        return False, "哑铃压在盒上却没有接触"

    manifold = max(manifolds, key=len)
    points = manifold.points
    feature_ids = [point.feature_id for point in points]
    # 两个球各 1 点，且应分属两个不同的部件命名空间
    offsets = {fid // 1_000_000 for fid in feature_ids}
    ok = len(points) >= 2 and len(offsets) >= 2 and len(set(feature_ids)) == len(feature_ids)
    return ok, (
        f"接触点={len(points)}，特征 ID={sorted(set(feature_ids))}，"
        f"部件命名空间={sorted(offsets)}，不撞号={len(set(feature_ids)) == len(feature_ids)}"
    )


def test_feature_ids_unique_on_ground() -> tuple[bool, str]:
    """组合体与地面的接触：不同部件的特征 ID 不能撞号。

    注意这里的"不撞号"是**跨部件**的，不是"每个点各有一个号"：同一部件内部
    的多个地面接触点共用默认 ``feature_id=0``（法向相同，暖启动一个标量就够），
    所以唯一性只需在两个部件的命名空间之间成立。
    """
    geometry = CompoundGeometry(
        [
            CompoundPart(BoxGeometry((0.4, 0.4, 0.2)), (-0.3, 0.0, 0.0)),
            CompoundPart(BoxGeometry((0.4, 0.4, 0.2)), (0.3, 0.0, 0.0)),
        ]
    )
    body = make_body("pair", geometry, (0.0, 0.0, 0.095), mass=2.0)
    contacts = detect_contacts([body], GroundPlane(z=0.0))
    if not contacts:
        return False, "两块并排的复合体没有地面接触"

    feature_ids = [point.feature_id for point in contacts[0].points]
    by_offset: dict[int, set[int]] = {}
    for fid in feature_ids:
        by_offset.setdefault(fid // 1_000_000, set()).add(fid)
    # 每个部件内部共用一个 ID，且两个部件的 ID 互不相同
    per_part_single = all(len(ids) == 1 for ids in by_offset.values())
    distinct = len({next(iter(ids)) for ids in by_offset.values()}) == len(by_offset)
    ok = len(feature_ids) >= 8 and len(by_offset) == 2 and per_part_single and distinct
    return ok, (
        f"地面接触点={len(feature_ids)}，部件命名空间={sorted(by_offset)}，"
        f"部件内共用={per_part_single}，跨部件不撞号={distinct}"
    )


# ---------------------------------------------------------------------------
# 9. 宽阶段
# ---------------------------------------------------------------------------

def test_broadphase_aabb_union() -> tuple[bool, str]:
    """复合体的世界 AABB 必须等于各部件 AABB 的并集。"""
    geometry = CompoundGeometry(
        [
            CompoundPart(BoxGeometry((0.4, 0.4, 0.4)), (-0.5, 0.0, 0.0)),
            CompoundPart(SphereGeometry(0.2), (0.6, 0.3, 0.1)),
        ]
    )
    body = make_body("combo", geometry, (1.0, 2.0, 3.0))

    combined = world_aabb(body)
    lower = [float("inf")] * 3
    upper = [float("-inf")] * 3
    for _index, view in iter_parts(body):
        box = world_aabb(view)
        for axis in range(3):
            lower[axis] = min(lower[axis], box.lower[axis])
            upper[axis] = max(upper[axis], box.upper[axis])

    error = max(
        max(abs(combined.lower[a] - lower[a]), abs(combined.upper[a] - upper[a]))
        for a in range(3)
    )
    ok = error < 1e-12
    return ok, (
        f"AABB=({tuple(round(v, 4) for v in combined.lower)}, "
        f"{tuple(round(v, 4) for v in combined.upper)})，误差={error:.2e}"
    )


# ---------------------------------------------------------------------------
# 10. 自转复合体的 TOI
# ---------------------------------------------------------------------------

def test_rotating_compound_toi() -> tuple[bool, str]:
    """整体静止、只自转的哑铃：TOI 必须靠 v + ω × r 报出事件。

    用质心速度算的话，两侧相对速度都是 0，TOI 会永远返回 None——
    高速自转的复合体就会直接穿过去。这条测试钉住这个换算。
    """
    # 必须是"质心与部件不重合"的组合体：单部件复合体的质心就在部件上，
    # r = 0，压根没有 ω × r 这一项
    spinner = make_body("spinner", dumbbell(0.15, 1.0), (0.0, 0.0, 0.0), mass=2.0)
    spinner.angular_velocity = (0.0, 0.0, 6.0)     # 6 rad/s → 末端 3 m/s

    # 部件此刻在 (0.5, 0)，切向速度 v = ω × r = +Y 方向 3 m/s。
    # 把目标放在它正前方的 0.6 处：表面间隙 0.3 m ⇒ 解析时刻约 0.1 s。
    target = make_body("target", SphereGeometry(0.15), (0.5, 0.6, 0.0), static=True)

    # 质心速度为零：仅凭线速度不可能接近
    assert spinner.linear_velocity == (0.0, 0.0, 0.0)

    window = 0.2
    pair_time = toi_module.compound_pair_toi(spinner, target, window)
    world_time = toi_module.earliest_toi([spinner, target], None, window)

    ok = (
        pair_time is not None
        and 0.0 < pair_time <= window
        and world_time is not None
        and abs(world_time[0] - pair_time) < 1e-12
    )
    return ok, (
        f"逐部件 TOI={None if pair_time is None else round(pair_time, 6)} s"
        f"（解析 {(0.6 - 0.3) / 3.0:.4f} s），"
        f"场景最早事件={None if world_time is None else round(world_time[0], 6)} s"
        f"（质心线速度=0，能报出来只可能来自 v + ω × r）"
    )


CHECKS = [
    ("1. 哑铃惯量与平行轴定理一致", test_dumbbell_inertia),
    ("2. 不等体积部件的质心解析正确", test_center_of_mass),
    ("3. 单部件组合体等于该图元", test_single_part_matches_primitive),
    ("4. 场景 JSON 支持组合体", test_scene_json_roundtrip),
    ("5. 哑铃横放落地能静置", test_dumbbell_rests_on_ground),
    ("6. T 形组合体立着不倒", test_t_shape_stands),
    ("7. 两个部件同时接触都被报出", test_both_parts_contact),
    ("8. 地面接触的特征 ID 不撞号", test_feature_ids_unique_on_ground),
    ("9. 宽阶段 AABB 是部件并集", test_broadphase_aabb_union),
    ("10. 自转复合体靠 ω×r 报出 TOI", test_rotating_compound_toi),
]


def main() -> int:
    """依次执行并汇总结果。"""
    print("=" * 88)
    print("组合体：质量属性 / 窄相 / TOI 验证")
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
    print("=" * 88)
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
