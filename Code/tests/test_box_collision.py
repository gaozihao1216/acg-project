# -*- coding: utf-8 -*-
"""盒-球 与 盒-地面 碰撞测试。

这一步为什么重要
----------------
球-球碰撞里，接触点力臂 ``r`` 永远与法向平行（``r × n = 0``），因此
**法向冲量从不产生力矩**——球的旋转只能由摩擦产生。这意味着在此之前的
所有测试都没有真正考验"姿态积分 + 世界坐标逆惯量 + 力矩"这条链路。

盒-球不同：球撞在盒子的边缘时，力臂与法向不平行，``r × n ≠ 0``，
盒子必须转起来。这一步正是把**转动动力学第一次接进回路**。

测试分四组
----------
1. **窄相几何**：法向、穿透深度、接触点位置（含球心在盒内的退化情形）
2. **力矩耦合**：偏心撞击必须产生旋转，对心撞击必须不产生
3. **盒-地面流形**：平放的盒子必须得到 4 个接触点，且能静止不翻转
4. **守恒律**：动量守恒、能量不增、盒-盒被明确拒绝

运行::

    python Code/tests/test_box_collision.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import BoxGeometry, SphereGeometry
from src.geom.inertia import compute_inertia
from src.geom.transform import TransformSpec
from src.mathutils import quaternion as quat
from src.physics.body import RigidBody
from src.physics.contact import GROUND_BODY
from src.physics.narrowphase import (
    GroundPlane,
    collide_box_ground,
    collide_sphere_box,
    detect_contacts,
)
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import MaterialSpec, ObjectSpec, SceneSpec, SimulationSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_box(name, size, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0),
             velocity=(0.0, 0.0, 0.0), mass=None, restitution=0.25,
             friction=0.5, is_static=False) -> RigidBody:
    """造一个盒子刚体。质量缺省按体积 × 1000 反推。"""
    geometry = BoxGeometry(size)
    if mass is None:
        mass = geometry.volume() * 1000.0
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name, geometry=geometry, position=position,
        orientation=orientation, mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction, is_static=is_static,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_sphere(name, radius, position, velocity=(0.0, 0.0, 0.0), mass=None,
                restitution=0.25, friction=0.5, is_static=False) -> RigidBody:
    """造一个球刚体。"""
    geometry = SphereGeometry(radius)
    if mass is None:
        mass = geometry.volume() * 1000.0
    inertia = compute_inertia(geometry, mass)
    body = RigidBody(
        name=name, geometry=geometry, position=position,
        mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction, is_static=is_static,
    )
    body.linear_velocity = tuple(velocity)
    return body


def make_world(bodies, ground=None, gravity=(0.0, 0.0, 0.0), substeps: int = 3,
               iterations: int = 8, use_toi_splitting: bool = False) -> PhysicsWorld:
    """组一个物理世界。默认关重力、关 TOI，便于孤立验证单次交互。"""
    sim = SimulationSpec(
        enabled=True,
        physics_dt=1.0 / 240.0,
        substeps=substeps,
        iterations=iterations,
        baumgarte=0.25,
        slop=0.002,
        restitution_threshold=0.3,
        gravity=gravity,
        linear_damping=0.0,
        angular_damping=0.0,
        use_toi_splitting=use_toi_splitting,
        duration=1.0,
    )
    return PhysicsWorld(list(bodies), ground, sim)


def advance(world: PhysicsWorld, seconds: float) -> None:
    """推进指定时长。"""
    steps = int(round(seconds / world.simulation.physics_dt))
    for _ in range(steps):
        world.step()


def spin(body: RigidBody) -> float:
    """角速度模长。"""
    return math.sqrt(sum(c * c for c in body.angular_velocity))


def box_local_axes(body: RigidBody):
    """盒子三个局部轴在世界坐标下的方向。"""
    return [
        quat.quat_rotate_vector(body.orientation, axis)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]


def lowest_vertex_z(body: RigidBody) -> float:
    """盒子 8 个顶点中的最低 z（世界坐标）。"""
    half = [0.5 * s for s in body.geometry.size]
    lowest = float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (sx * half[0], sy * half[1], sz * half[2])
                world = quat.quat_rotate_vector(body.orientation, local)
                lowest = min(lowest, world[2] + body.position[2])
    return lowest


# ---------------------------------------------------------------------------
# 组一：窄相几何
# ---------------------------------------------------------------------------

def test_sphere_box_face() -> tuple[bool, str]:
    """球撞盒面：法向应沿坐标轴，穿透深度精确可算。

    盒中心在原点、边长 0.8（半尺寸 0.4）。
    球半径 0.2，球心在 x 方向 -0.55 处 → 到 -x 面的距离是 0.15，
    小于半径 0.2，因此穿透 0.05。
    """
    box = make_box("box", (0.8, 0.8, 0.8))
    sphere = make_sphere("ball", 0.2, (-0.55, 0.0, 0.0))

    point = collide_sphere_box(sphere, box)
    if point is None:
        return False, "应当检测到碰撞，但返回了 None"

    expected_penetration = 0.05
    normal_from_box = point.normal
    ok = (
        abs(point.penetration - expected_penetration) < 1e-9
        and abs(normal_from_box[0] + 1.0) < 1e-9
        and abs(normal_from_box[1]) < 1e-9
        and abs(normal_from_box[2]) < 1e-9
    )
    return ok, (
        f"球撞 -x 面：法向=({normal_from_box[0]:+.3f},{normal_from_box[1]:+.3f},"
        f"{normal_from_box[2]:+.3f})（期望 (-1,0,0)），穿透={point.penetration:.6f}"
        f"（期望 {expected_penetration}）"
    )


def test_sphere_box_edge() -> tuple[bool, str]:
    """球撞盒棱：法向应为斜向，且是单位向量。

    球心在 (x, 0, z) 平面上、正对 -x 与 -z 两条面交成的棱。
    """
    box = make_box("box", (0.8, 0.8, 0.8))
    # 棱位于 x=-0.4, z=-0.4；把球心放在该棱向外 (1,0,1)/√2 方向 0.15 处
    offset = 0.15 / math.sqrt(2.0)
    sphere = make_sphere("ball", 0.2, (-0.4 - offset, 0.0, -0.4 - offset))

    point = collide_sphere_box(sphere, box)
    if point is None:
        return False, "应当检测到碰撞，但返回了 None"

    length = math.sqrt(sum(c * c for c in point.normal))
    # 法向应是 (-1,0,-1)/√2
    expected = -1.0 / math.sqrt(2.0)
    ok = (
        abs(length - 1.0) < 1e-9
        and abs(point.normal[0] - expected) < 1e-6
        and abs(point.normal[2] - expected) < 1e-6
        and abs(point.penetration - 0.05) < 1e-6
    )
    return ok, (
        f"球撞棱：|n|={length:.9f}，n=({point.normal[0]:+.4f},0,{point.normal[2]:+.4f})"
        f"（期望 ({expected:+.4f},0,{expected:+.4f})），穿透={point.penetration:.6f}"
    )


def test_sphere_box_inside() -> tuple[bool, str]:
    """球心在盒内（深度穿透）：法向应指向最近的面。

    球心放在盒中心偏 +z 处，因此最近面是 +z 面。
    """
    box = make_box("box", (1.0, 1.0, 1.0))       # 半尺寸 0.5
    sphere = make_sphere("ball", 0.1, (0.0, 0.0, 0.3))

    point = collide_sphere_box(sphere, box)
    if point is None:
        return False, "球心在盒内时应产生接触，但返回了 None"

    # 到 +z 面距离 0.2，穿透 = r + 0.2 = 0.3
    ok = (
        abs(point.normal[2] - 1.0) < 1e-9
        and abs(point.penetration - 0.3) < 1e-9
    )
    return ok, (
        f"球心在盒内：法向=({point.normal[0]:+.3f},{point.normal[1]:+.3f},"
        f"{point.normal[2]:+.3f})（期望 (0,0,1)），穿透={point.penetration:.6f}（期望 0.3）"
    )


def test_sphere_box_no_contact() -> tuple[bool, str]:
    """未接触时必须返回 None（不能"总觉得快要碰上了"）。"""
    box = make_box("box", (0.8, 0.8, 0.8))
    far = make_sphere("ball", 0.2, (-1.2, 0.0, 0.0))
    just_outside = make_sphere("ball2", 0.2, (-0.61, 0.0, 0.0))

    far_point = collide_sphere_box(far, box)
    outside_point = collide_sphere_box(just_outside, box)

    ok = far_point is None and outside_point is None
    return ok, (
        f"远离={far_point is None}（期望 True），"
        f"恰好在外侧(间隙 0.01)={outside_point is None}（期望 True）"
    )


def test_box_ground_manifold() -> tuple[bool, str]:
    """平放的盒子陷入地面 1 cm：应恰好得到 4 个接触点。"""
    box = make_box("box", (0.6, 0.6, 0.6), position=(0.0, 0.0, 0.29))
    points = collide_box_ground(box, GroundPlane(z=0.0))

    penetrations = sorted(p.penetration for p in points)
    all_up = all(abs(p.normal[2] - 1.0) < 1e-9 for p in points)

    ok = len(points) == 4 and all_up and abs(penetrations[0] - 0.01) < 1e-9
    return ok, (
        f"平放盒子陷入 1 cm：接触点 {len(points)} 个（期望 4），"
        f"法向全为 +Z={all_up}，穿透={[round(x, 6) for x in penetrations]}"
    )


# ---------------------------------------------------------------------------
# 组二：力矩耦合（本步的核心价值）
# ---------------------------------------------------------------------------

def test_offcenter_hit_creates_spin() -> tuple[bool, str]:
    """球以偏心方式撞盒子：盒子必须转起来。

    这是"力臂 → 力矩 → 世界坐标逆惯量 → 姿态积分"整条链路是否接通的判据。
    在球-球阶段这条链路从未被真正考验过（球的力臂恒与法向平行）。
    """
    box = make_box("box", (0.8, 0.8, 0.8), restitution=0.4, friction=0.0)
    # 沿 +x 飞行，高度偏到 z=+0.3（半尺寸 0.4，所以打在面靠上的位置）
    sphere = make_sphere("ball", 0.15, (-1.0, 0.0, 0.30), (6.0, 0.0, 0.0),
                         restitution=0.4, friction=0.0)

    world = make_world([box, sphere])
    advance(world, 0.2)

    angular = spin(box)
    # 力矩 ∝ r × F：r ≈ (-0.4, 0, +0.3)，盒子受到的冲量沿 +x，
    # 叉乘得 (0, +0.3J, 0) → ω_y 为正。
    ok = angular > 1e-3 and box.angular_velocity[1] > 0.0
    return ok, (
        f"偏心撞击：盒子角速度模长={angular:.4f} rad/s（应 > 0），"
        f"ω=({box.angular_velocity[0]:+.4f},{box.angular_velocity[1]:+.4f},"
        f"{box.angular_velocity[2]:+.4f})（ω_y 应为正）"
    )


def test_centered_hit_no_spin() -> tuple[bool, str]:
    """球正对盒面中心撞击：不应产生任何旋转。

    此时力臂与法向平行，``r × n = 0``，法向冲量力矩为零；
    摩擦也关掉，因此总力矩必须严格为 0。
    如果这条测试失败，说明力臂算错了（例如用了质心到质心的方向）。
    """
    box = make_box("box", (0.8, 0.8, 0.8), restitution=0.4, friction=0.0)
    sphere = make_sphere("ball", 0.15, (-1.0, 0.0, 0.0), (6.0, 0.0, 0.0),
                         restitution=0.4, friction=0.0)

    world = make_world([box, sphere])
    advance(world, 0.2)

    angular = spin(box)
    ok = angular < 1e-9
    return ok, (
        f"对心撞击：盒子角速度模长={angular:.3e} rad/s（应为 0），"
        f"盒子 vx={box.linear_velocity[0]:+.4f}（应不为 0，说明确实撞上了）"
    )


def test_inertia_tensor_used() -> tuple[bool, str]:
    """转动惯量必须是按形状算出的张量，且世界逆惯量随姿态正确旋转。

    这是"转动惯量真正接入引擎"的直接判据，比看角速度更精确：
    直接对逆惯量算符施以单位力矩，结果的模长必须等于 ``1/I``。

    注意一个容易踩的坑：盒子的 ``Iyy = m/12·(sx² + sz²)``。
    如果拿 ``(0.9, 0.9, 0.3)`` 和 ``(0.3, 0.3, 0.9)`` 对比，两者互为转置，
    ``Iyy`` **完全相同**——我第一版测试就是栽在这里，看起来像代码错了，
    其实是测试选错了形状。
    """
    size = (0.9, 0.9, 0.3)
    mass = 100.0
    box = make_box("box", size, mass=mass)

    # --- 1. 解析式 Ixx = m/12·(sy²+sz²)，其余同理 ---
    sx, sy, sz = size
    expected = (
        mass / 12.0 * (sy * sy + sz * sz),
        mass / 12.0 * (sx * sx + sz * sz),
        mass / 12.0 * (sx * sx + sy * sy),
    )
    actual = box.inertia_local
    formula_ok = all(abs(actual[i] - expected[i]) < 1e-12 for i in range(3))

    # --- 2. 各向异性：扁盒的 Ixx 与 Izz 必须不同 ---
    anisotropic = abs(actual[0] - actual[2]) > 1e-6

    # --- 3. 未旋转时：沿局部 x 施加单位力矩，ω 应为 1/Ixx 且沿 x ---
    omega = box.inverse_inertia_world((1.0, 0.0, 0.0))
    expect_x = 1.0 / actual[0]
    axis_ok = abs(omega[0] - expect_x) < 1e-12 and abs(omega[1]) < 1e-12

    # --- 4. 旋转后：沿世界方向 R·x 施加力矩，ω 仍应沿同一方向、比例 1/Ixx ---
    rotated = make_box(
        "rot", size, mass=mass,
        orientation=quat.quat_from_axis_angle((0.0, 0.0, 1.0), 0.7),
    )
    world_axis = quat.quat_rotate_vector(rotated.orientation, (1.0, 0.0, 0.0))
    omega_rot = rotated.inverse_inertia_world(world_axis)
    projection = sum(omega_rot[i] * world_axis[i] for i in range(3))
    rotated_ok = abs(projection - expect_x) < 1e-12

    ok = formula_ok and anisotropic and axis_ok and rotated_ok
    return ok, (
        f"惯量 (Ixx,Iyy,Izz)=({actual[0]:.4f},{actual[1]:.4f},{actual[2]:.4f})，"
        f"解析值=({expected[0]:.4f},{expected[1]:.4f},{expected[2]:.4f})，"
        f"各向异性={anisotropic}，局部逆惯量={axis_ok}，"
        f"旋转后逆惯量={rotated_ok}（沿世界轴投影 {projection:.9f}，期望 {expect_x:.9f}）"
    )


# ---------------------------------------------------------------------------
# 组三：盒-地面静置
# ---------------------------------------------------------------------------

def test_box_rests_on_ground() -> tuple[bool, str]:
    """轴对齐盒子落到平地：应静止、姿态不变、不翻转、不下陷。"""
    half = 0.3
    box = make_box("box", (0.6, 0.6, 0.6), position=(0.0, 0.0, 0.8),
                   restitution=0.1, friction=0.7)
    ground = GroundPlane(z=0.0, friction=0.7, restitution=0.05)
    world = make_world([box], ground=ground, gravity=(0.0, 0.0, -9.81))
    advance(world, 3.0)

    height_error = abs(box.position[2] - half)
    angular = spin(box)
    axes = box_local_axes(box)
    # 姿态应仍与坐标轴对齐（三个局部轴各自只有一个分量接近 ±1）
    alignment = max(max(abs(a[0]), abs(a[1]), abs(a[2])) for a in axes)

    ok = (
        height_error < 0.005
        and box.speed() < 0.02
        and angular < 0.02
        and alignment > 0.999
    )
    return ok, (
        f"落定后 z={box.position[2]:.4f}（期望 {half}，误差 {height_error * 1000:.2f} mm），"
        f"|v|={box.speed():.4f}，|ω|={angular:.4f}，姿态对齐度={alignment:.5f}"
    )


def test_tilted_box_settles_flat() -> tuple[bool, str]:
    """倾斜 20° 落下的盒子：应翻到某个面贴地并静止。

    这条考验的是 4 点流形能否真正约束旋转——单点接触时盒子会
    绕着接触点缓慢转动而停不下来。
    """
    tilt = quat.quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(20.0))
    box = make_box("box", (0.6, 0.6, 0.6), position=(0.0, 0.0, 0.75),
                   orientation=tilt, restitution=0.1, friction=0.8)
    ground = GroundPlane(z=0.0, friction=0.8, restitution=0.05)
    world = make_world([box], ground=ground, gravity=(0.0, 0.0, -9.81))
    advance(world, 4.0)

    axes = box_local_axes(box)
    # 落定后必有一个局部轴与竖直方向近乎平行（面贴地）
    best = max(abs(a[2]) for a in axes)
    lowest = lowest_vertex_z(box)

    ok = box.speed() < 0.03 and spin(box) < 0.03 and best > 0.995 and lowest > -0.006
    return ok, (
        f"倾斜 20° 落下：|v|={box.speed():.4f}，|ω|={spin(box):.4f}，"
        f"最低顶点 z={lowest * 1000:.2f} mm（不应明显穿地），"
        f"某局部轴与竖直的贴合度={best:.5f}（应 > 0.995）"
    )


# ---------------------------------------------------------------------------
# 组四：守恒律与拒绝
# ---------------------------------------------------------------------------

def test_box_sphere_momentum_energy() -> tuple[bool, str]:
    """盒-球偏心碰撞：线动量守恒、总动能不增加。"""
    box = make_box("box", (0.8, 0.8, 0.8), restitution=0.4, friction=0.4)
    sphere = make_sphere("ball", 0.15, (-1.0, 0.0, 0.25), (6.0, 0.0, 0.0),
                         restitution=0.4, friction=0.4)
    world = make_world([box, sphere])

    p_before = [
        box.mass * box.linear_velocity[i] + sphere.mass * sphere.linear_velocity[i]
        for i in range(3)
    ]
    e_before = world.total_kinetic_energy()
    peak = e_before

    for _ in range(int(0.3 / world.simulation.physics_dt)):
        world.step()
        peak = max(peak, world.total_kinetic_energy())

    p_after = [
        box.mass * box.linear_velocity[i] + sphere.mass * sphere.linear_velocity[i]
        for i in range(3)
    ]
    e_after = world.total_kinetic_energy()

    momentum_drift = max(abs(p_after[i] - p_before[i]) for i in range(3))
    growth = (peak - e_before) / e_before

    ok = momentum_drift < 1e-6 and growth < 1e-9
    return ok, (
        f"动量漂移={momentum_drift:.2e}，动能 {e_before:.4f} J → {e_after:.4f} J，"
        f"峰值涨幅={growth * 100:.4f}%"
    )


def test_box_box_accepted() -> tuple[bool, str]:
    """盒-盒场景现在应当被接受（此前是明确拒绝的）。

    这是一条**回归护栏**：盒-盒 SAT 实现之后，world 里那条"组合未支持就报错"
    的校验不应该再拦下两个盒子。如果哪天它又开始拒绝，说明 `supported_pair`
    被改坏了，而症状会是"场景加载不了"而不是"碰撞不对"，很容易误判。
    """
    objects = [
        ObjectSpec(
            name=f"box{i}",
            geometry=BoxGeometry((0.5, 0.5, 0.5)),
            transform=TransformSpec.from_dict({"position": [0.0, 0.0, 0.25 + i * 0.9]}),
            material=MaterialSpec(),
        )
        for i in range(2)
    ]
    scene = SceneSpec(
        name="two_boxes",
        objects=objects,
        simulation=SimulationSpec(enabled=True),
    )

    accepted = True
    message = ""
    try:
        world = PhysicsWorld.from_scene(scene)
        world.step()
        message = f"步进正常，bodies={len(world.bodies)}"
    except Exception as exc:  # noqa: BLE001
        accepted = False
        message = f"{type(exc).__name__}: {exc}"

    return accepted, f"两个盒子的场景被接受={accepted}（{message}）"


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 球撞盒面：法向与穿透", test_sphere_box_face),
    ("2. 球撞盒棱：斜向法向", test_sphere_box_edge),
    ("3. 球心在盒内：最近面方向", test_sphere_box_inside),
    ("4. 未接触时返回 None", test_sphere_box_no_contact),
    ("5. 盒-地面：4 点流形", test_box_ground_manifold),
    ("6. 偏心撞击产生旋转", test_offcenter_hit_creates_spin),
    ("7. 对心撞击不产生旋转", test_centered_hit_no_spin),
    ("8. 惯量按形状计算", test_inertia_tensor_used),
    ("9. 盒子落定静置", test_box_rests_on_ground),
    ("10. 倾斜盒子翻至面贴地", test_tilted_box_settles_flat),
    ("11. 盒-球动量与能量", test_box_sphere_momentum_energy),
    ("12. 盒-盒场景被接受", test_box_box_accepted),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 92)
    print("盒-球 / 盒-地面 碰撞测试")
    print("=" * 92)

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

    print("-" * 92)
    print(f"结果：{passed}/{len(CHECKS)} 通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
