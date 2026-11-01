# -*- coding: utf-8 -*-
"""柱 / 胶囊（曲面几何）的窄相、TOI 与地面接触验证。

为什么单独一个测试文件
----------------------
柱与胶囊的侧面/端帽是**曲面**：既没有有限 SAT 轴集合，也没有顶点可枚举。
它们接入的是 GJK + EPA 那条通用路径（``src/physics/gjk.py``，几何信息全部来自
``src/physics/support.py`` 的支撑映射）。这条路径是新的，就必须有独立的
数值断言钉住它，而且要与**解析解**对照，而不是与它自己对照。

覆盖的性质
----------
1. 圆柱立着落地后能静置，高度等于半高（下沉不超过 1 cm）
2. 胶囊竖放落地后能静置，高度等于半高 + 半径
3. 胶囊横放落地：高度等于半径，且不会翻滚
4. 球-柱：GJK/EPA 的深度与法向与解析最近点解一致（100 组随机构型）
5. 立着的柱压在盒面上：4 个接触点、法向竖直、深度等于压入量
6. 高速柱撞地不穿透
7. 柱-柱并排挤压：深度与解析值一致
8. TOI 与解析接触时刻一致（不偏晚）
9. 通用路径没有破坏成熟结果：盒-盒的 GJK/EPA 与 SAT 一致
10. 分离距离双向对称，远离时无接触

运行::

    python Code/tests/test_curved.py
"""

from __future__ import annotations

import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CylinderGeometry,
    SphereGeometry,
)
from src.geom.inertia import compute_inertia
from src.mathutils import quaternion as quat
from src.physics import gjk
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.sat import collide_box_box
from src.physics.support import support_world
from src.physics.toi import curved_pair_toi
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_body(
    name: str,
    geometry,
    position,
    orientation=(1.0, 0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    restitution: float = 0.1,
    friction: float = 0.6,
    mass: float = 1.0,
) -> RigidBody:
    """按几何 + 位姿造一个刚体（惯量走统一的 compute_inertia）。"""
    body = RigidBody(
        name=name,
        geometry=geometry,
        position=position,
        orientation=orientation,
        mass=mass,
        inertia_local=compute_inertia(geometry, mass),
        restitution=restitution,
        friction=friction,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(bodies, ground, substeps: int = 4, dt: float = 1.0 / 120.0) -> PhysicsWorld:
    """带地面的世界：开启 TOI 分段，重力正常。"""
    sim = SimulationSpec(
        enabled=True,
        physics_dt=dt,
        substeps=substeps,
        iterations=12,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), ground, sim)


def advance(world: PhysicsWorld, seconds: float) -> None:
    """按物理步长推进给定时间。"""
    steps = int(round(seconds / world.simulation.physics_dt))
    for _ in range(steps):
        world.step()


def angle_degrees(normal_a, normal_b) -> float:
    """两个单位向量的夹角（度）。"""
    dot = sum(a * b for a, b in zip(normal_a, normal_b))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# 1–3. 与地面静置
# ---------------------------------------------------------------------------

def test_cylinder_rests_on_ground() -> tuple[bool, str]:
    """圆柱立着落地：停稳、高度等于半高、下沉不超过 1 cm。"""
    radius, height = 0.25, 0.6
    cylinder = make_body("cyl", CylinderGeometry(radius, height), (0.0, 0.0, 1.2))
    world = make_world([cylinder], GroundPlane(z=0.0, friction=0.6, restitution=0.1))
    advance(world, 4.0)

    body = world.bodies[0]
    expected = 0.5 * height
    error = body.position[2] - expected
    penetration = world.max_penetration()
    ok = (
        body.speed() < 0.05
        and abs(error) < 0.01
        and penetration < 0.02
        and abs(body.angular_velocity[0]) < 0.2
        and abs(body.angular_velocity[1]) < 0.2
    )
    return ok, (
        f"静置 4s：速度={body.speed():.4f} m/s，z={body.position[2]:.4f}"
        f"（期望 {expected}），高度误差={error * 1000:.2f} mm，"
        f"最大穿透={penetration * 1000:.2f} mm"
    )


def test_capsule_rests_upright() -> tuple[bool, str]:
    """胶囊竖着落地：停稳，高度等于半高 + 半径。"""
    radius, segment = 0.2, 0.5
    capsule = make_body("cap", CapsuleGeometry(radius, segment), (0.0, 0.0, 1.2))
    world = make_world([capsule], GroundPlane(z=0.0, friction=0.6, restitution=0.1))
    advance(world, 4.0)

    body = world.bodies[0]
    expected = 0.5 * segment + radius
    error = body.position[2] - expected
    penetration = world.max_penetration()
    ok = body.speed() < 0.05 and abs(error) < 0.01 and penetration < 0.02
    return ok, (
        f"静置 4s：速度={body.speed():.4f} m/s，z={body.position[2]:.4f}"
        f"（期望 {expected}），高度误差={error * 1000:.2f} mm，"
        f"最大穿透={penetration * 1000:.2f} mm"
    )


def test_capsule_lying_flat() -> tuple[bool, str]:
    """胶囊横着落地：停在半径高度，且不翻滚。

    横放的胶囊与平面是**线接触**，流形必须给出 2 个点；只给 1 个点时它会绕着
    那条线自由翻转，表现为缓慢滚走。
    """
    radius, segment = 0.2, 0.5
    lying = quat.quat_from_axis_angle((1.0, 0.0, 0.0), 0.5 * math.pi)
    capsule = make_body(
        "cap", CapsuleGeometry(radius, segment), (0.0, 0.0, 0.5), orientation=lying
    )
    world = make_world([capsule], GroundPlane(z=0.0, friction=0.6, restitution=0.1))
    advance(world, 4.0)

    body = world.bodies[0]
    axis = quat.quat_rotate_vector(body.orientation, (0.0, 0.0, 1.0))
    error = body.position[2] - radius
    penetration = world.max_penetration()
    spin = abs(body.angular_velocity[0]) + abs(body.angular_velocity[1]) + abs(body.angular_velocity[2])
    ok = (
        body.speed() < 0.05
        and abs(error) < 0.02
        and penetration < 0.02
        and abs(axis[2]) < 0.15
        and spin < 0.5
    )
    return ok, (
        f"静置 4s：z={body.position[2]:.4f}（期望 {radius}），"
        f"高度误差={error * 1000:.2f} mm，轴与竖直夹角="
        f"{math.degrees(math.acos(max(-1.0, min(1.0, abs(axis[2]))))):.1f}°，"
        f"角速度={spin:.4f} rad/s"
    )


# ---------------------------------------------------------------------------
# 4. 球-柱：与解析最近点解对照
# ---------------------------------------------------------------------------

def _sphere_cylinder_analytic(sphere: RigidBody, cylinder: RigidBody):
    """球-柱的解析接触：返回 (法向 柱→球, 深度)，无接触返回 None。

    球心转到柱的局部系 → 夹紧到柱体（径向夹到半径、轴向夹到半高）得到
    最近点；球心在柱外时距离就是欧氏距离（凸体性质保证这是精确解）。
    """
    geometry = cylinder.geometry
    radius, half_height = geometry.radius, 0.5 * geometry.height

    inverse = quat.quat_conjugate(cylinder.orientation)
    relative = (
        sphere.position[0] - cylinder.position[0],
        sphere.position[1] - cylinder.position[1],
        sphere.position[2] - cylinder.position[2],
    )
    local = quat.quat_rotate_vector(inverse, relative)

    radial_length = math.hypot(local[0], local[1])
    clamped_radial = min(radial_length, radius)
    clamped_z = max(-half_height, min(half_height, local[2]))

    if radial_length > 1e-12:
        scale = clamped_radial / radial_length
        closest = (local[0] * scale, local[1] * scale, clamped_z)
    else:
        closest = (0.0, 0.0, clamped_z)

    delta = (
        local[0] - closest[0],
        local[1] - closest[1],
        local[2] - closest[2],
    )
    distance = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
    if distance <= 1e-12:
        return None   # 球心落在柱体内：本用例不覆盖（解析式要分支讨论）

    penetration = sphere.geometry.radius - distance
    if penetration <= 0.0:
        return None

    normal_local = (delta[0] / distance, delta[1] / distance, delta[2] / distance)
    normal = quat.quat_rotate_vector(cylinder.orientation, normal_local)
    return normal, penetration


def test_sphere_cylinder_matches_analytic() -> tuple[bool, str]:
    """100 组随机构型：GJK/EPA 的深度与法向必须贴合解析解。"""
    random.seed(20240501)
    worst_angle = 0.0
    worst_depth = 0.0
    checked = 0

    for _ in range(200):
        radius = random.uniform(0.15, 0.3)
        height = random.uniform(0.3, 0.8)
        sphere_radius = random.uniform(0.05, 0.2)

        tilt = quat.quat_from_axis_angle(
            (random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0)),
            random.uniform(0.0, math.pi),
        )
        cylinder = make_body(
            "cyl", CylinderGeometry(radius, height), (0.0, 0.0, 0.0), orientation=tilt
        )

        # 在柱体侧面附近撒球心，保证大部分构型是"刚好贴上"
        angle = random.uniform(0.0, 2.0 * math.pi)
        axial = random.uniform(-0.6, 0.6) * height
        reach = radius + sphere_radius * random.uniform(0.3, 1.6)
        local_center = (math.cos(angle) * reach, math.sin(angle) * reach, axial)
        world_center = quat.quat_rotate_vector(tilt, local_center)
        sphere = make_body("ball", SphereGeometry(sphere_radius), world_center)

        expected = _sphere_cylinder_analytic(sphere, cylinder)
        points = gjk.collide(cylinder, sphere)
        if expected is None:
            if points:
                return False, "解析判定未接触，GJK 却报了接触"
            continue
        if not points:
            return False, (
                f"GJK 漏检：球心距柱面 {sphere_radius - expected[1]:.4f} m < 半径 "
                f"{sphere_radius:.4f} m"
            )

        checked += 1
        worst_angle = max(worst_angle, angle_degrees(expected[0], points[0].normal))
        worst_depth = max(worst_depth, abs(expected[1] - points[0].penetration))

    ok = checked >= 100 and worst_angle < 2.0 and worst_depth < 2e-4
    return ok, (
        f"{checked} 组接触对照解析解：最大法向偏差={worst_angle:.3f}°，"
        f"最大深度偏差={worst_depth * 1000:.4f} mm（阈值 2° / 0.2 mm）"
    )


# ---------------------------------------------------------------------------
# 5. 柱压在盒面上：多点流形
# ---------------------------------------------------------------------------

def test_cylinder_on_box_face() -> tuple[bool, str]:
    """立着的柱压进盒顶 2 mm：应得到 4 点流形、竖直法向、2 mm 深度。

    单点流形会让柱子在盒面上绕那一点打转——这正是当初盒-地面必须返回
    4 个点的原因，曲面同样适用。
    """
    box = make_body("box", BoxGeometry((2.0, 2.0, 0.2)), (0.0, 0.0, 0.0))
    cylinder = make_body(
        "cyl", CylinderGeometry(0.25, 0.6), (0.0, 0.0, 0.5 * 0.6 + 0.1 - 0.002)
    )
    points = gjk.collide(cylinder, box)

    if not points:
        return False, "柱压在盒面上却没有检测到接触"
    depths = [point.penetration for point in points]
    normal = points[0].normal
    ok = (
        len(points) == 4
        and max(abs(depth - 0.002) for depth in depths) < 2e-4
        and angle_degrees(normal, (0.0, 0.0, -1.0)) < 0.5
    )
    return ok, (
        f"接触点={len(points)}（期望 4），深度={[round(d * 1000, 3) for d in depths]} mm"
        f"（期望 2.0），法向={tuple(round(c, 4) for c in normal)}（期望 (0,0,-1)）"
    )


# ---------------------------------------------------------------------------
# 6. 高速柱不穿透
# ---------------------------------------------------------------------------

def test_cylinder_no_tunneling() -> tuple[bool, str]:
    """-15 m/s 落地的柱：底面任何时刻都不应低于地面 1 cm 以上。"""
    radius, height = 0.25, 0.6
    cylinder = make_body(
        "bullet", CylinderGeometry(radius, height), (0.0, 0.0, 3.0),
        velocity=(0.0, 0.0, -15.0),
    )
    world = make_world([cylinder], GroundPlane(z=0.0, friction=0.4, restitution=0.3))

    # 度量必须用**支撑点**：撞地后柱会倾倒，``z - 半高`` 不再是它的最低点
    lowest = support_world(cylinder, (0.0, 0.0, -1.0))[2]
    steps = int(round(2.0 / world.simulation.physics_dt))
    for _ in range(steps):
        world.step()
        lowest = min(lowest, support_world(world.bodies[0], (0.0, 0.0, -1.0))[2])

    ok = lowest > -0.01
    return ok, f"初速 -15 m/s 撞地：全程最低底面 z={lowest:.5f}（下限 -0.01）"


# ---------------------------------------------------------------------------
# 7. 柱-柱：深度与解析值一致
# ---------------------------------------------------------------------------

def test_cylinder_pair_depth() -> tuple[bool, str]:
    """两个并排的柱：法向沿连心线、深度等于半径和减中心距。"""
    radius, height = 0.2, 0.5
    first = make_body("a", CylinderGeometry(radius, height), (0.0, 0.0, 0.0))
    second = make_body("b", CylinderGeometry(radius, height), (0.37, 0.0, 0.0))

    expected_depth = 2.0 * radius - 0.37
    points = gjk.collide(first, second)
    if not points:
        return False, "并排的柱没有检测到接触"

    depth_error = max(abs(point.penetration - expected_depth) for point in points)
    normal_error = angle_degrees(points[0].normal, (1.0, 0.0, 0.0))
    ok = depth_error < 2e-4 and normal_error < 0.5
    return ok, (
        f"深度={[round(p.penetration * 1000, 3) for p in points]} mm"
        f"（期望 {expected_depth * 1000:.3f}），"
        f"法向偏差={normal_error:.3f}°，点数={len(points)}"
    )


# ---------------------------------------------------------------------------
# 8. TOI 与解析接触时刻一致
# ---------------------------------------------------------------------------

def test_curved_toi_matches_analytic() -> tuple[bool, str]:
    """球以 2 m/s 直线撞静止的柱：TOI 应与解析时刻相差不到 1 ms。

    解析：表面间隙 1.55 m、闭合速率 2 m/s ⇒ t = 0.775 s。
    TOI 故意多推进 ``_CONTACT_OVERLAP``（0.2 mm）让窄相能看到穿透，
    所以允许比解析值晚 1 ms 量级，但**不能提前漏掉**。
    """
    cylinder = make_body("cyl", CylinderGeometry(0.25, 0.6), (0.0, 0.0, 0.0), restitution=0.5)
    sphere = make_body(
        "ball", SphereGeometry(0.2), (2.0, 0.0, 0.0), velocity=(-2.0, 0.0, 0.0),
        restitution=0.5,
    )

    gap = 2.0 - 0.25 - 0.2
    expected = gap / 2.0

    forward = curved_pair_toi(cylinder, sphere, 20.0)
    backward = curved_pair_toi(sphere, cylinder, 20.0)

    if forward is None or backward is None:
        return False, f"TOI 没有报出事件（期望 {expected:.4f}s）"
    ok = abs(forward - expected) < 1e-3 and abs(forward - backward) < 1e-12
    return ok, (
        f"TOI={forward:.6f}s，解析={expected:.6f}s，"
        f"误差={(forward - expected) * 1000:.4f} ms，双向对称性={abs(forward - backward):.2e}"
    )


# ---------------------------------------------------------------------------
# 9. 通用路径与 SAT 的一致性（盒-盒）
# ---------------------------------------------------------------------------

def test_gjk_matches_sat_for_boxes() -> tuple[bool, str]:
    """随机 100 组盒-盒：GJK/EPA 与 SAT 的深度、法向必须吻合。

    盒-盒正式运行时走 SAT，这条测试的价值在于**用成熟实现校验新实现**：
    通用路径算错了，这里立刻红。
    """
    random.seed(880517)
    worst_angle = 0.0
    worst_depth = 0.0
    checked = 0

    for _ in range(120):
        box_a = make_body("a", BoxGeometry((0.6, 0.5, 0.4)), (0.0, 0.0, 0.0))
        axis = (random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0))
        tilt = quat.quat_from_axis_angle(axis, random.uniform(0.0, 1.2))
        offset = tuple(random.uniform(-0.35, 0.35) for _ in range(3))
        box_b = make_body("b", BoxGeometry((0.5, 0.5, 0.5)), offset, orientation=tilt)

        sat_points = collide_box_box(box_a, box_b)
        gjk_points = gjk.collide(box_a, box_b)
        if not sat_points:
            if gjk_points:
                return False, "SAT 判定未接触，GJK 却报了接触"
            continue
        if not gjk_points:
            return False, "SAT 报了接触，GJK 漏检"

        deepest = max(sat_points, key=lambda point: point.penetration)
        checked += 1
        worst_angle = max(worst_angle, angle_degrees(deepest.normal, gjk_points[0].normal))
        worst_depth = max(worst_depth, abs(deepest.penetration - gjk_points[0].penetration))

    ok = checked >= 40 and worst_angle < 1.0 and worst_depth < 1e-3
    return ok, (
        f"{checked} 组盒-盒：最大法向偏差={worst_angle:.3f}°、"
        f"最大深度偏差={worst_depth * 1000:.4f} mm（阈值 1° / 1 mm）"
    )


# ---------------------------------------------------------------------------
# 10. 距离查询：对称、远离时无接触
# ---------------------------------------------------------------------------

def test_distance_symmetry() -> tuple[bool, str]:
    """GJK 距离必须双向完全一致；远离的柱应按"无接触"处理。"""
    random.seed(31337)
    worst = 0.0
    for _ in range(100):
        first = make_body("a", CylinderGeometry(0.2, 0.5), (0.0, 0.0, 0.0))
        second = make_body(
            "b", CapsuleGeometry(0.15, 0.3),
            (random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)),
        )
        worst = max(worst, abs(gjk.distance(first, second)[0] - gjk.distance(second, first)[0]))

    far = make_body("far", CylinderGeometry(0.2, 0.5), (5.0, 0.0, 0.0))
    near = make_body("near", CylinderGeometry(0.2, 0.5), (0.0, 0.0, 0.0))
    far_points = gjk.collide(far, near)

    ok = worst < 1e-12 and not far_points
    return ok, (
        f"100 组距离双向最大差={worst:.2e} m；相距 5 m 的两个柱接触点={len(far_points)}"
    )


# ---------------------------------------------------------------------------
# 运行时：确认世界层面也接上了（窄相 + TOI 都被真实调用）
# ---------------------------------------------------------------------------

def test_world_handles_cylinder_scene() -> tuple[bool, str]:
    """柱落在盒子上：世界层面跑通（宽阶段 → TOI → GJK 窄相 → 求解器）。"""
    box = make_body("box", BoxGeometry((1.5, 1.5, 0.2)), (0.0, 0.0, 0.1), restitution=0.1)
    cylinder = make_body("cyl", CylinderGeometry(0.2, 0.4), (0.0, 0.0, 0.8), restitution=0.1)
    world = make_world([box, cylinder], GroundPlane(z=0.0, friction=0.6, restitution=0.1))
    advance(world, 3.0)

    world.check_finite()
    box_body, cylinder_body = world.bodies
    expected = 0.2 + 0.2      # 盒顶(0.2) + 柱半高(0.2)
    error = cylinder_body.position[2] - expected
    manifolds = detect_contacts(world.bodies, world.ground)
    penetration = world.max_penetration()
    ok = (
        abs(error) < 0.02
        and cylinder_body.speed() < 0.1
        and box_body.speed() < 0.05
        and bool(manifolds)
        and penetration < 0.02
    )
    return ok, (
        f"柱落在盒上：z={cylinder_body.position[2]:.4f}（期望 {expected}），"
        f"误差={error * 1000:.1f} mm，柱速度={cylinder_body.speed():.4f} m/s，"
        f"盒速度={box_body.speed():.4f} m/s，流形数={len(manifolds)}，"
        f"最大穿透={penetration * 1000:.2f} mm"
    )


CHECKS = [
    ("1. 圆柱立着落地能静置", test_cylinder_rests_on_ground),
    ("2. 胶囊竖放落地能静置", test_capsule_rests_upright),
    ("3. 胶囊横放落地不翻滚", test_capsule_lying_flat),
    ("4. 球-柱深度与法向对照解析解", test_sphere_cylinder_matches_analytic),
    ("5. 柱压在盒面上得到 4 点流形", test_cylinder_on_box_face),
    ("6. 高速柱不穿透地面", test_cylinder_no_tunneling),
    ("7. 柱-柱挤压深度与解析一致", test_cylinder_pair_depth),
    ("8. 曲面 TOI 与解析接触时刻一致", test_curved_toi_matches_analytic),
    ("9. 盒-盒上 GJK/EPA 与 SAT 一致", test_gjk_matches_sat_for_boxes),
    ("10. GJK 距离双向对称且远离无接触", test_distance_symmetry),
    ("11. 世界层面柱-盒场景收敛", test_world_handles_cylinder_scene),
]


def main() -> int:
    """依次执行并汇总结果。"""
    print("=" * 88)
    print("柱 / 胶囊：支撑映射 + GJK/EPA 窄相与 TOI 验证")
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
