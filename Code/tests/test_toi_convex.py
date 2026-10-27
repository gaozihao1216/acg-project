# -*- coding: utf-8 -*-
"""凸包接触时刻预测（TOI）的专项测试。

背景
----
``toi.py`` 原先只对球有效：球-球是一元二次方程，球-地面是一元一次。凸包没有
解析式，于是**跳过子步切分**——不漏碰撞，但一个子步的位移会整个陷进去。
实测空心盒/棱柱的穿透比球大一个数量级（10.5 mm vs 0.29 mm）。

本文件验证四类性质：

1. **平移情形是精确的**：轴对齐盒子对盒子、盒子对地面，与解析解逐位比对。
2. **旋转情形由顶点速度决定**：纯自转的盒子，TOI 必须由下降最快的那个顶点给出，
   而不是由质心速度（为零）给出——后者会永远返回 None。
3. **保守性**：不存在"报得偏晚导致漏切分"的情形。设计上允许偏晚一点点
   （``_CONTACT_OVERLAP``，200 µm 的穿透余量），但必须有上界。
4. **不产生伪事件**：分离、远离、已重叠都不该报事件。

为什么第 3 条要单独测
--------------------
"偏早"的代价只是多切一刀（推进、检测、发现没接触、继续），"偏晚"的代价是
漏掉切分、深穿透，甚至让牛顿摆飞出两个球。两者不对等，所以要钉住上界。

运行::

    python Code/tests/test_toi_convex.py
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

from src.geom.geometry import BoxGeometry, ConvexHullGeometry
from src.geom.meshutil import prism_mesh
from src.geom.transform import TransformSpec
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane
from src.physics.toi import (
    _CONTACT_OVERLAP,
    hull_ground_toi,
    hull_hull_toi,
)
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import (
    BodyPhysicsSpec, GroundSpec, MaterialSpec, ObjectSpec, SceneSpec,
    SimulationSpec,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_body(
    name,
    geometry,
    position,
    velocity=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    static=False,
    mass=1.0,
):
    spec = ObjectSpec(
        name=name,
        geometry=geometry,
        transform=TransformSpec(position=position, orientation=(0.0, 0.0, 0.0, 1.0)),
        material=MaterialSpec(name="m"),
        mass=mass,
        physics=BodyPhysicsSpec(
            linear_velocity=velocity,
            angular_velocity=angular,
            is_static=static,
            restitution=0.0,
            friction=0.5,
        ),
    )
    return RigidBody.from_object_spec(spec)


def box(size, position, velocity=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0),
        static=False):
    return make_body("box", BoxGeometry(size), position, velocity, angular, static)


def hex_prism(radius=0.35, height=0.7):
    """正六棱柱的凸包（顶点以质心为原点）。"""
    polygon = [
        (radius * math.cos(2.0 * math.pi * i / 6.0),
         radius * math.sin(2.0 * math.pi * i / 6.0))
        for i in range(6)
    ]
    vertices, _faces = prism_mesh(polygon, height, z0=-height / 2.0)
    return ConvexHullGeometry(vertices, max_faces=32)


def aabb_toi(size_a, position_a, velocity_a, size_b, position_b, velocity_b):
    """轴对齐盒子（纯平移）的**解析**接触时刻。

    沿每条轴各自的投影区间开始重叠的时刻是 ``gap_i / closing_i``；
    三条轴要**同时**重叠才算碰上，所以真正的 TOI 是这三者的**最大值**。

    取最小值是常见错误：那意味着"任意一条轴重叠就算撞上"，
    对盒-盒会报得过早（侧面擦过时尤其明显）。

    返回 ``(时刻, 闭合速度)``；后者用于给 200 µm 余量折算时间容差
    （余量是**位移**，除以该轴的闭合速度才是时间）。
    """
    times = []
    for axis in range(3):
        delta = position_b[axis] - position_a[axis]
        gap = abs(delta) - (size_a[axis] + size_b[axis]) * 0.5
        if gap <= 0.0:
            times.append((0.0, float("inf")))
            continue
        # 沿该轴的分离速度：远离则永远不会撞
        separation_speed = velocity_b[axis] - velocity_a[axis]
        if delta < 0.0:
            separation_speed = -separation_speed
        if separation_speed >= 0.0:
            return None
        times.append((gap / (-separation_speed), -separation_speed))
    return max(times, key=lambda pair: pair[0])


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_ground_translation_exact() -> tuple[bool, str]:
    """盒子落地：``t = (最低点高度 - 地面) / 下降速度``。

    这是最朴素的一条，但它同时验证了三件事：多面体的顶点在 body 系与场景系
    之间没有错位（否则最低点算错）、速度符号没搞反、以及没有把"已经重叠"
    误判成事件。
    """
    cases = [
        ((0.5, 0.5, 0.5), 2.0, -8.0),
        ((1.0, 0.4, 0.6), 0.6, -2.0),
        ((0.2, 0.2, 0.2), 1.0, -0.5),
    ]
    ground = GroundPlane(z=0.0)
    worst = 0.0
    notes = []
    for size, height, speed in cases:
        bottom = height - size[2] * 0.5
        body = box(size, (0.0, 0.0, height), (0.0, 0.0, speed))
        t = hull_ground_toi(body, ground, 10.0)

        analytic = bottom / (-speed)
        expected = (bottom + _CONTACT_OVERLAP) / (-speed)
        if t is None:
            return False, f"盒子未预测到落地（{size}，h={height}，vz={speed}）"
        worst = max(worst, abs(t - expected))
        notes.append(f"{t:.6f} vs 解析 {analytic:.6f}")

    ok = worst < 1e-12
    return ok, f"与解析解最大偏差 {worst:.2e} s（{'；'.join(notes)}）"


def test_ground_rotation_dominates() -> tuple[bool, str]:
    """**纯自转**的盒子必须由顶点速度决定接触时刻。

    质心速度为 0，若只用质心速度就会返回 None——而实际上是某个角在往下扫。
    这道题专门排除"只考虑平移"的实现。

    参数：底面在 z=0.1，绕 x 轴 ω=5 rad/s。
    角点 r=(0, -0.25, -0.25) 的速度 (ω×r)_z = ω_x·r_y = 5×(-0.25) = -1.25 m/s，
    所以 t ≈ (0.1 + 2e-4) / 1.25。
    """
    body = box(
        (0.5, 0.5, 0.5), (0.0, 0.0, 0.35),
        velocity=(0.0, 0.0, 0.0), angular=(5.0, 0.0, 0.0),
    )
    t = hull_ground_toi(body, GroundPlane(z=0.0), 10.0)

    expected = (0.1 + _CONTACT_OVERLAP) / 1.25
    if t is None:
        return False, "纯自转的盒子被判定为永不触地（只看质心速度的实现会这样）"

    ok = abs(t - expected) < 1e-12
    return ok, (
        f"t={t:.6f}s，由角点下降速度 1.25 m/s 给出；"
        f"解析 {expected:.6f}s，偏差 {abs(t - expected):.2e}（质心速度为 0）"
    )


def test_hull_translation_exact() -> tuple[bool, str]:
    """盒-盒（纯平移）与解析解一致，容差只来自 ``_CONTACT_OVERLAP``。

    这是"精确性"的核心证据：面-面接近由面法向给出精确解，棱-棱由叉积轴给出。
    随机取 10 组位姿与速度，与三维解析解逐组比对。
    """
    rng = random.Random(20261025)
    worst = 0.0
    checked = 0
    for _ in range(10):
        size_a = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        size_b = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        pos_a = (0.0, 0.0, 0.0)
        pos_b = (
            rng.uniform(-0.6, 0.6),
            rng.uniform(-0.6, 0.6),
            rng.uniform(0.9, 1.6),
        )
        vel_b = (
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-4.0, -1.0),
        )

        analytic = aabb_toi(size_a, pos_a, (0.0, 0.0, 0.0), size_b, pos_b, vel_b)
        if analytic is None:
            continue   # 这组是分离运动，本来就不该接触
        exact, closing = analytic

        a = box(size_a, pos_a)
        b = box(size_b, pos_b, vel_b)
        t = hull_hull_toi(a, b, 10.0)
        if t is None:
            return False, f"静止盒与下落盒未预测到接触（解析 {exact:.6f}s）"

        # 200 µm 的余量是**位移**，要除以该轴的闭合速度才换算成时间
        tolerance = _CONTACT_OVERLAP / closing + 1e-12
        worst = max(worst, abs(t - exact) - tolerance)
        checked += 1

    ok = worst <= 0.0 and checked >= 6
    return ok, (
        f"{checked} 组随机位姿：|t - 解析| ≤ 200 µm / 闭合速度"
        f"（最多超出容差 {worst:.2e} s）"
    )


def test_no_false_events() -> tuple[bool, str]:
    """分离、远离、已重叠都不该报事件。

    第三条尤其重要：已经重叠的物体属于**持续接触**，交给常规求解器推出，
    如果同时又被当成"未来事件"，就会在同一个子步里反复切分、互相打架。
    """
    ground = GroundPlane(z=0.0)

    far = box((0.5, 0.5, 0.5), (0.0, 0.0, 5.0))
    other = box((0.5, 0.5, 0.5), (0.0, 0.0, 0.0), static=True)
    far_clear = hull_hull_toi(far, other, 0.5) is None

    # 上升：离地面越来越远
    rising = box((0.5, 0.5, 0.5), (0.0, 0.0, 1.0), (0.0, 0.0, 3.0))
    rising_clear = hull_ground_toi(rising, ground, 1.0) is None

    # 已经重叠 5 mm
    overlapped_a = box((0.5, 0.5, 0.5), (0.0, 0.0, 0.25))
    overlapped_b = box((0.5, 0.5, 0.5), (0.0, 0.0, 0.745))
    overlap_clear = hull_hull_toi(overlapped_a, overlapped_b, 1.0) is None

    sunk = box((0.5, 0.5, 0.5), (0.0, 0.0, 0.24), (0.0, 0.0, -1.0))
    sunk_clear = hull_ground_toi(sunk, ground, 1.0) is None

    ok = far_clear and rising_clear and overlap_clear and sunk_clear
    return ok, (
        f"远离={far_clear}，上升={rising_clear}，已重叠={overlap_clear}，"
        f"已陷入地面={sunk_clear}"
    )


def test_penetration_improves_in_simulation() -> tuple[bool, str]:
    """集成验证：盒与棱柱的穿透必须显著下降，而球保持原有的零穿透。

    这条同时是"开关仍然有效"的守门人——如果有人把 ``body_toi`` 的分派改坏，
    凸包会静默退化成不做切分，而单元测试仍然全绿。
    """

    def run(objects, toi_enabled):
        scene = SceneSpec(
            name="t", objects=objects, camera=None, lights=[],
            ground=GroundSpec(enabled=True),
            simulation=SimulationSpec(
                enabled=True, physics_dt=1.0 / 240.0, substeps=3, iterations=8,
                ground_z=0.0, duration=1.0, use_toi_splitting=toi_enabled,
            ),
            render=None,
        )
        world = PhysicsWorld.from_scene(scene)
        worst = 0.0
        for _ in range(int(scene.simulation.duration / scene.simulation.physics_dt)):
            world.step()
            worst = max(worst, world.max_penetration())
        return worst * 1000.0

    from src.geom.geometry import SphereGeometry

    cases = {
        "盒子": [ObjectSpec(
            "box", BoxGeometry((0.5, 0.5, 0.5)),
            TransformSpec(position=(0, 0, 2.0), orientation=(0, 0, 0, 1)),
            MaterialSpec(name="m"), mass=1.0,
            physics=BodyPhysicsSpec(linear_velocity=(0, 0, -8.0),
                                    restitution=0.0, friction=0.5))],
        "棱柱": [ObjectSpec(
            "prism", hex_prism(),
            TransformSpec(position=(0, 0, 1.6), orientation=(0, 0, 0, 1)),
            MaterialSpec(name="m"), mass=1.0,
            physics=BodyPhysicsSpec(linear_velocity=(0.3, 0.15, -6.0),
                                    restitution=0.0, friction=0.5))],
        "球": [ObjectSpec(
            "ball", SphereGeometry(0.3),
            TransformSpec(position=(0, 0, 2.0), orientation=(0, 0, 0, 1)),
            MaterialSpec(name="m"), mass=1.0,
            physics=BodyPhysicsSpec(linear_velocity=(0, 0, -8.0),
                                    restitution=0.0, friction=0.5))],
    }

    measured = {}
    for label, objects in cases.items():
        measured[label] = (run(objects, False), run(objects, True))

    box_off, box_on = measured["盒子"]
    prism_off, prism_on = measured["棱柱"]
    sphere_off, sphere_on = measured["球"]

    ok = (
        box_on < box_off * 0.3
        and prism_on < prism_off * 0.3
        and sphere_on <= sphere_off * 0.05
        and max(box_on, prism_on, sphere_on) < 1.0
    )
    return ok, (
        f"盒子 {box_off:.3f}→{box_on:.3f} mm，棱柱 {prism_off:.3f}→{prism_on:.3f} mm，"
        f"球 {sphere_off:.3f}→{sphere_on:.3f} mm（全部 < 1 mm）"
    )


def test_sphere_path_unchanged() -> tuple[bool, str]:
    """分派不能把球的**精确解**也套上 200 µm 余量。

    球-球、球-地面是一元二次/一次方程，本来就没有流形退化问题，必须保持
    逐位精确；``_CONTACT_OVERLAP`` 只应作用于多面体（那里才有裁剪多边形）。
    """
    from src.geom.geometry import SphereGeometry
    from src.physics.toi import sphere_ground_toi, sphere_sphere_toi

    ball = make_body(
        "ball", SphereGeometry(0.25), (0.0, 0.0, 2.0), (0.0, 0.0, -4.0)
    )
    t_ground = sphere_ground_toi(ball, GroundPlane(z=0.0), 10.0)
    ground_expected = (2.0 - 0.25) / 4.0

    a = make_body("a", SphereGeometry(0.5), (0.0, 0.0, 0.0))
    b = make_body("b", SphereGeometry(0.5), (0.0, 0.0, 3.0), (0.0, 0.0, -2.0))
    t_pair = sphere_sphere_toi(a, b, 10.0)
    pair_expected = (3.0 - 1.0) / 2.0

    ok = (
        t_ground is not None
        and t_pair is not None
        and abs(t_ground - ground_expected) < 1e-12
        and abs(t_pair - pair_expected) < 1e-12
    )
    return ok, (
        f"球-地面 {t_ground:.9f}s（解析 {ground_expected:.9f}），"
        f"球-球 {t_pair:.9f}s（解析 {pair_expected:.9f}）——无 200 µm 余量"
    )


CHECKS = [
    ("1. 盒子落地：顶点精确解", test_ground_translation_exact),
    ("2. 纯自转由顶点速度决定", test_ground_rotation_dominates),
    ("3. 盒-盒纯平移与解析解一致", test_hull_translation_exact),
    ("4. 不产生伪事件", test_no_false_events),
    ("5. 仿真中穿透显著下降", test_penetration_improves_in_simulation),
    ("6. 球走精确解、不吃余量", test_sphere_path_unchanged),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("凸包接触时刻预测（TOI）")
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
