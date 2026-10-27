# -*- coding: utf-8 -*-
"""盒-盒 SAT 与接触流形裁剪测试。

为什么盒-盒值得单独一套测试
--------------------------
球-球的正确性可以直接和闭式解比对。盒-盒不行：15 条候选轴、参考面/入射面选择、
多边形裁剪、4 点裁减，任何一步写错都会给出"看起来合理"的少数点——
比如平放堆叠只得到 2 个点，仿真仍然能跑，只是盒子会慢慢倾斜。

本次实现过程中就踩到两个只有靠这类测试才能发现的 bug：

1. 面顶点按 Z 字序排列（`for su: for sv:` 的自然结果），四边形其实是自交的
   蝴蝶形，裁剪结果全错。
2. 入射面选择只看了每个轴的**正方向**，对 +z 参考法向选中的是 x 轴而不是 −z 面。

两者都不会抛异常，只是一个数字不对。所以判据必须精确。

测试分四组
----------
1. **构型正确性**：面贴面 4 点、侧撞 4 点、分离 0 点、棱-棱 1 点、退化重合
2. **法向约定**：法向必须始终从 A 指向 B（交换顺序应取反）
3. **特征 ID 稳定**：静置时接触点 ID 集合跨帧不变（暖启动的前提）
4. **随机抽样**：用内切球/包围球给出可靠的单侧判据，独立于 SAT 内部实现

运行::

    python Code/tests/test_box_box.py
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

from src.geom.geometry import BoxGeometry
from src.geom.inertia import compute_inertia
from src.mathutils import quaternion as quat
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane
from src.physics.sat import collide_box_box, box_axes, box_half_extents
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_box(name, size, position=(0.0, 0.0, 0.0), orientation=None,
             restitution=0.1, friction=0.7, is_static=False) -> RigidBody:
    """造一个盒子刚体。"""
    geometry = BoxGeometry(size)
    mass = geometry.volume() * 1000.0
    inertia = compute_inertia(geometry, mass)
    if orientation is None:
        orientation = quat.quat_identity()
    return RigidBody(
        name=name, geometry=geometry, position=position, orientation=orientation,
        mass=mass, inertia_local=inertia,
        restitution=restitution, friction=friction, is_static=is_static,
    )


def make_world(bodies, ground=None, gravity=(0.0, 0.0, -9.81),
               substeps: int = 3, iterations: int = 8) -> PhysicsWorld:
    """组一个物理世界（默认开重力，用于静置类测试）。"""
    sim = SimulationSpec(
        enabled=True,
        physics_dt=1.0 / 240.0,
        substeps=substeps,
        iterations=iterations,
        baumgarte=0.25,
        slop=0.002,
        restitution_threshold=0.6,
        gravity=gravity,
        linear_damping=0.05,
        angular_damping=1.0,
        use_toi_splitting=False,
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


def describe(points) -> str:
    """把流形整理成一行易读文本。"""
    if not points:
        return "n=0"
    return (
        f"n={len(points)} normal=({points[0].normal[0]:+.3f},{points[0].normal[1]:+.3f},"
        f"{points[0].normal[2]:+.3f}) pen={[round(p.penetration, 4) for p in points]}"
    )


# ---------------------------------------------------------------------------
# 组一：构型正确性
# ---------------------------------------------------------------------------

def test_face_to_face_stacked() -> tuple[bool, str]:
    """平放堆叠必须给出 4 个接触点。

    这是最容易写错的一格：单点或两点都能"跑起来"，但盒子会绕接触点缓慢倾倒。
    穿透量可精确手算：中心距 0.9、半尺寸各 0.5 → 重叠 0.1。
    """
    low = make_box("low", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    high = make_box("high", (1.0, 1.0, 1.0), (0.0, 0.0, 0.9))
    points = collide_box_box(low, high)

    ok = (
        len(points) == 4
        and abs(points[0].normal[2] - 1.0) < 1e-9
        and all(abs(p.penetration - 0.1) < 1e-9 for p in points)
    )
    return ok, f"平放堆叠：{describe(points)}（期望 4 点、法向 +z、穿透 0.1）"


def test_face_to_face_side() -> tuple[bool, str]:
    """侧向面贴面：同样 4 点，法向沿 +x。"""
    left = make_box("left", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    right = make_box("right", (1.0, 1.0, 1.0), (0.9, 0.0, 0.0))
    points = collide_box_box(left, right)

    ok = (
        len(points) == 4
        and abs(points[0].normal[0] - 1.0) < 1e-9
        and all(abs(p.penetration - 0.1) < 1e-9 for p in points)
    )
    return ok, f"侧向面贴面：{describe(points)}（期望 4 点、法向 +x、穿透 0.1）"


def test_separated() -> tuple[bool, str]:
    """分离时必须返回 0 个点（不能"总觉得快碰上了"）。"""
    left = make_box("left", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    right = make_box("right", (1.0, 1.0, 1.0), (1.3, 0.0, 0.0))
    slant = make_box(
        "slant", (1.0, 1.0, 1.0), (1.3, 0.0, 0.0),
        quat.quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(45.0)),
    )

    a = collide_box_box(left, right)
    b = collide_box_box(left, slant)

    ok = len(a) == 0 and len(b) == 0
    return ok, f"分离(正对)={len(a)} 点，分离(斜置)={len(b)} 点（都应为 0）"


def test_edge_edge() -> tuple[bool, str]:
    """棱碰棱：应为 1 个点，法向为两条棱叉积方向（斜的、单位长度）。"""
    first = make_box(
        "e1", (1.2, 0.3, 0.3), (0.0, 0.0, 0.0),
        quat.quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(40.0)),
    )
    second = make_box(
        "e2", (0.3, 1.2, 0.3), (0.0, 0.0, 0.0),
        quat.quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(40.0)),
    )
    points = collide_box_box(first, second)

    if not points:
        return False, "棱-棱构型应当接触，但返回 0 个点"

    length = math.sqrt(sum(c * c for c in points[0].normal))
    oblique = abs(points[0].normal[0]) > 0.1 and abs(points[0].normal[2]) > 0.1
    ok = len(points) == 1 and abs(length - 1.0) < 1e-9 and oblique
    return ok, f"棱-棱：{describe(points)}，|n|={length:.9f}，斜向={oblique}"


def test_vertex_face() -> tuple[bool, str]:
    """一个盒子的**单个顶点**扎进另一个盒子的面：应恰好 1 个点。

    构造：让立方体的体对角线竖直。此时 8 个顶点的 z 坐标分为四档
    ``−0.866 / −0.289 / +0.289 / +0.866``，只有最低的那个顶点能扎进去。

    注意绕 z 轴转 45° **不行**——那样落下来的是棱（2 个点），不是顶点。
    这也是一个容易写错测试的地方：看起来"斜着放"，实际考的是棱-面而不是顶点-面。
    """
    base = make_box("base", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    # 把体对角线 (1,1,1)/√3 转到 (0,0,-1)
    axis = (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
    angle = math.acos(-1.0 / math.sqrt(3.0))
    orientation = quat.quat_from_axis_angle(axis, angle)

    # 体对角线竖直时，最低顶点到中心正好是 √3/2。
    # 基座顶面在 z=0.5，要让这个顶点落到 0.45（即穿透 0.05），中心应在 0.45+√3/2。
    # 这里必须用表达式而不是手写的 1.316：手写四位小数会带来 2.5e-5 的偏差，
    # 足以让 1e-6 的断言失败，看起来像"穿透算错了"。
    tip_center = 0.45 + math.sqrt(3.0) / 2.0
    tip = make_box("tip", (1.0, 1.0, 1.0), (0.0, 0.0, tip_center), orientation)
    points = collide_box_box(base, tip)

    ok = len(points) == 1 and abs(points[0].penetration - 0.05) < 1e-9
    return ok, (
        f"单顶点扎入：{describe(points)}（期望 1 点、穿透 0.05，"
        f"实测 {points[0].penetration:.12f}）"
    )


def test_full_overlap_degenerate() -> tuple[bool, str]:
    """完全重合是退化构型：不能崩溃、不能产生 NaN/负穿透。"""
    first = make_box("a", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    second = make_box("b", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    points = collide_box_box(first, second)

    finite = all(
        math.isfinite(p.penetration)
        and all(math.isfinite(c) for c in p.position)
        for p in points
    )
    ok = len(points) > 0 and finite and all(p.penetration >= 0.0 for p in points)
    return ok, f"完全重合：{describe(points)}，数值有限={finite}"


# ---------------------------------------------------------------------------
# 组二：法向约定
# ---------------------------------------------------------------------------

def test_normal_direction_swaps() -> tuple[bool, str]:
    """交换 A/B 顺序后法向必须严格取反：这是冲量符号正确的前提。

    如果这条失败，表现是"两个盒子碰撞时互相吸引"，而且很难从画面上看出。
    """
    first = make_box("a", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    second = make_box("b", (0.8, 0.8, 0.8), (0.85, 0.3, 0.2))

    forward = collide_box_box(first, second)
    backward = collide_box_box(second, first)

    if not forward or not backward:
        return False, "应当接触却返回 0 个点"

    dot = sum(forward[0].normal[i] * backward[0].normal[i] for i in range(3))
    ok = abs(dot + 1.0) < 1e-9
    return ok, (
        f"顺序交换：A→B 法向=({forward[0].normal[0]:+.4f},{forward[0].normal[1]:+.4f},"
        f"{forward[0].normal[2]:+.4f})，B→A 法向点积={dot:+.9f}（应为 −1）"
    )


# ---------------------------------------------------------------------------
# 组三：特征 ID 稳定性
# ---------------------------------------------------------------------------

def test_feature_ids_stable() -> tuple[bool, str]:
    """静置时接触点的特征 ID 集合必须跨帧不变。

    暖启动要把上一帧的累积冲量喂给这一帧"同一个"接触点。
    如果 ID 每帧都在变，冲量就会喂错对象——轻则无效，重则注入能量。
    """
    low = make_box("low", (0.8, 0.8, 0.8), (0.0, 0.0, 0.4), is_static=True)
    # 底面留 1 cm 间隙，落下后稳定贴合
    high = make_box("high", (0.8, 0.8, 0.8), (0.0, 0.0, 1.21))
    world = make_world([low, high], ground=GroundPlane(z=0.0, friction=0.7))

    snapshots = []
    for _ in range(8):
        advance(world, 0.1)
        ids = frozenset(p.feature_id for p in collide_box_box(low, high))
        snapshots.append(ids)

    # 注意：必须断言集合**非空**。第一版把上盒放在 z=0.82（已经深深嵌入），
    # 盒子被 Baumgarte 推开后彻底分离，8 次采样全是空集，
    # "全部相同"于是平凡成立 —— 一次假通过。
    non_empty = bool(snapshots[0])
    stable = all(s == snapshots[0] for s in snapshots)
    ok = non_empty and stable

    return ok, (
        f"8 次采样的 ID 集合：{sorted(snapshots[0])} / "
        f"非空={non_empty}，全部相同={stable}"
    )


# ---------------------------------------------------------------------------
# 组四：随机抽样（独立于 SAT 内部实现的判据）
# ---------------------------------------------------------------------------

def test_random_sampling() -> tuple[bool, str]:
    """随机姿态抽样，用内切球/包围球给出可靠的单侧判据。

    这两个判据都不依赖 SAT 的实现细节：

    - **内切球相交 ⇒ 必然重叠**：盒内能放下的最大球半径是 ``min(half)``。
      两个内切球相交，两个盒子必然相交。
    - **包围球分离 ⇒ 必然不重叠**：球的半径是 ``|half|``。

    落在两者之间的构型是"说不准"，跳过不计——这是为了让判据保持**充分**，
    而不是拿一个可能同样错的参考实现去对比。
    """
    random.seed(20261023)

    must_collide = 0
    must_separate = 0
    failures = []

    for _ in range(400):
        size_a = (0.6, 0.6, 0.6)
        size_b = (0.5, 0.5, 0.5)
        half_a = [0.5 * s for s in size_a]
        half_b = [0.5 * s for s in size_b]

        orientation_a = quat.quat_from_axis_angle(
            (_random_axis()), random.uniform(0.0, math.pi)
        )
        orientation_b = quat.quat_from_axis_angle(
            (_random_axis()), random.uniform(0.0, math.pi)
        )

        # 沿随机方向偏移，幅度覆盖"必重叠"到"必分离"的全区间
        direction = _random_axis()
        distance = random.uniform(0.0, 1.6)
        offset = tuple(direction[i] * distance for i in range(3))

        box_a = make_box("a", size_a, (0.0, 0.0, 0.0), orientation_a)
        box_b = make_box("b", size_b, offset, orientation_b)

        points = collide_box_box(box_a, box_b)
        detected = len(points) > 0

        inner_radius = min(half_a) + min(half_b)
        outer_radius = math.sqrt(sum(h * h for h in half_a)) + math.sqrt(
            sum(h * h for h in half_b)
        )

        if distance < inner_radius - 1e-9:
            must_collide += 1
            if not detected:
                failures.append(f"内切球相交却未检测到（d={distance:.4f}）")
        elif distance > outer_radius + 1e-9:
            must_separate += 1
            if detected:
                failures.append(f"包围球分离却报告了接触（d={distance:.4f}）")

    ok = not failures
    detail = "；".join(failures[:3]) if failures else "无异常"
    return ok, (
        f"400 次随机抽样：必重叠 {must_collide} 例、必分离 {must_separate} 例，{detail}"
    )


def _random_axis() -> tuple[float, float, float]:
    """随机单位轴（拒绝采样，避免退化）。"""
    while True:
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        z = random.uniform(-1.0, 1.0)
        length = math.sqrt(x * x + y * y + z * z)
        if length > 0.2:
            return (x / length, y / length, z / length)


# ---------------------------------------------------------------------------
# 组五：动力学（盒子堆叠）
# ---------------------------------------------------------------------------

def test_two_box_stack() -> tuple[bool, str]:
    """两个盒子叠放：必须稳定静止，不下陷、不倾倒、不抖动。"""
    ground = GroundPlane(z=0.0, friction=0.8, restitution=0.05)
    low = make_box("low", (0.8, 0.8, 0.8), (0.0, 0.0, 0.41), friction=0.8)
    high = make_box("high", (0.8, 0.8, 0.8), (0.0, 0.0, 1.24), friction=0.8)
    world = make_world([low, high], ground=ground)
    advance(world, 3.0)

    # 直立度：每个盒子取三个局部轴中**最接近竖直**的那个，再对两个盒子取较小值。
    # 第一版写成了对全部 6 个轴取 min —— 直立盒子的水平轴 z 分量为 0，
    # 于是"完全直立"反而得到 0.0002，直接把正确的物理判成失败。
    def uprightness(body) -> float:
        return max(abs(axis[2]) for axis in box_axes(body))

    upright = min(uprightness(low), uprightness(high))

    sink = abs(low.position[2] - 0.4)
    gap = high.position[2] - low.position[2]

    ok = (
        low.speed() < 0.03 and spin(low) < 0.03
        and high.speed() < 0.03 and spin(high) < 0.03
        and upright > 0.995
        and sink < 0.006
        and abs(gap - 0.8) < 0.01
    )
    return ok, (
        f"两层堆叠：低层 z={low.position[2]:.4f}（期望 0.4，偏差 {sink * 1000:.2f} mm），"
        f"层间距={gap:.4f}（期望 0.8），"
        f"|v|={low.speed():.4f}/{high.speed():.4f}，"
        f"|ω|={spin(low):.4f}/{spin(high):.4f}，直立度={upright:.5f}"
    )


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 平放堆叠：4 点流形", test_face_to_face_stacked),
    ("2. 侧向面贴面：4 点流形", test_face_to_face_side),
    ("3. 分离时为 0 点", test_separated),
    ("4. 棱-棱：单点斜法向", test_edge_edge),
    ("5. 顶点扎入面", test_vertex_face),
    ("6. 完全重合不崩溃", test_full_overlap_degenerate),
    ("7. 交换顺序法向取反", test_normal_direction_swaps),
    ("8. 特征 ID 跨帧稳定", test_feature_ids_stable),
    ("9. 随机抽样单侧判据", test_random_sampling),
    ("10. 两层堆叠稳定", test_two_box_stack),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 92)
    print("盒-盒 SAT 与接触流形测试")
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
