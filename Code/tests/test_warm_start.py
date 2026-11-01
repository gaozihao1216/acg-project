# -*- coding: utf-8 -*-
"""暖启动（warm starting）的专项测试。

暖启动做什么
------------
把上一个子步的**法向累积冲量**作为本子步迭代的初值。静置接触每帧的冲量几乎
不变（就是支撑重力的那一份），冷启动则每帧都从 ``P = 0`` 重新"爬"上去，链越长
爬得越慢——这正是低迭代次数下塔整体下沉的直接原因。

这个文件钉住的五件事
--------------------
1. **开关与默认值**：默认开启，JSON 可关，且能往返序列化
2. **真的有用**：低迭代次数下，暖启动让静置穿透显著变小、残余速度显著变小
3. **与 TOI 共存不退化**：TOI 会把子步切段，暖启动不能在同一子步内重复施加
   （第一版就错在这里：棱柱落地穿透从 0.148 mm 退到 1.317 mm）
4. **确定性**：同一场景跑两遍，位姿必须逐位相同
5. **不注入能量**：静置 1 s 后总机械能不得高于初始值

运行::

    python Code/tests/test_warm_start.py
"""

from __future__ import annotations

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import BoxGeometry, ConvexHullGeometry
from src.geom.meshutil import prism_mesh
from src.geom.transform import TransformSpec
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import (
    BodyPhysicsSpec,
    GroundSpec,
    MaterialSpec,
    ObjectSpec,
    SceneSpec,
    SimulationSpec,
)

#: 塔的盒子边长（与 scripts/bench_warm_start.py 保持一致）
BOX = 0.5
#: 层间初始间隙
GAP = 0.002


def make_box(name: str, position, friction: float = 0.8) -> ObjectSpec:
    """造一个静止起步的盒子。"""
    return ObjectSpec(
        name=name,
        geometry=BoxGeometry((BOX, BOX, BOX)),
        transform=TransformSpec(position=position, orientation=(0.0, 0.0, 0.0, 1.0)),
        material=MaterialSpec(name="mat"),
        mass=1.0,
        physics=BodyPhysicsSpec(restitution=0.0, friction=friction),
    )


def tower_scene(layers: int, iterations: int, warm_start: bool, duration: float) -> SceneSpec:
    """竖直叠 ``layers`` 层的静置塔。"""
    objects = []
    z = BOX / 2.0
    for index in range(layers):
        objects.append(make_box(f"L{index}", (0.0, 0.0, z)))
        z += BOX + GAP

    return SceneSpec(
        name="tower",
        objects=objects,
        camera=None,
        lights=[],
        ground=GroundSpec(enabled=True),
        simulation=SimulationSpec(
            enabled=True,
            physics_dt=1.0 / 240.0,
            substeps=3,
            iterations=iterations,
            ground_z=0.0,
            duration=duration,
            # 堆叠场景不需要 TOI：这里要隔离的是暖启动本身的效果
            use_toi_splitting=False,
            warm_start=warm_start,
        ),
        render=None,
    )


def advance(world: PhysicsWorld, seconds: float) -> None:
    """按 1/240 的步长推进指定时长。"""
    for _ in range(int(round(seconds * 240.0))):
        world.step()


def poses(world: PhysicsWorld) -> str:
    """把全部位姿序列化成字符串，用于逐位比较。"""
    return json.dumps(
        [
            {"p": [round(v, 12) for v in body.position],
             "q": [round(v, 12) for v in body.orientation]}
            for body in world.bodies
        ],
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# 1. 开关与默认值
# ---------------------------------------------------------------------------

def test_default_and_switch() -> tuple[bool, str]:
    """默认开启、可显式关闭、能往返序列化。"""
    problems = []

    if SimulationSpec().warm_start is not True:
        problems.append("SimulationSpec() 默认应为 True")
    if SimulationSpec.from_dict({}).warm_start is not True:
        problems.append("空 simulation 段默认应为 True")

    partial = SimulationSpec.from_dict({"enabled": True})
    if partial.warm_start is not True:
        problems.append("只写 enabled 时 warm_start 应取默认 True")

    disabled = SimulationSpec.from_dict({"enabled": True, "warm_start": False})
    if disabled.warm_start is not False:
        problems.append("显式 false 未生效")

    if "warm_start" not in partial.to_dict():
        problems.append("to_dict 未记录 warm_start，资产将无法复现该配置")

    return not problems, (
        "；".join(problems) if problems
        else "默认 True，显式 false 生效，to_dict 含该键"
    )


# ---------------------------------------------------------------------------
# 2. 低迭代时的实际收益
# ---------------------------------------------------------------------------

def test_reduces_sinking_at_low_iterations() -> tuple[bool, str]:
    """3 层塔 + 每子步只迭代 1 次：暖启动必须明显减小下沉与残余速度。

    这是暖启动存在的理由：冷启动下每次迭代只从 0 往上爬一点，1 次迭代连
    支撑重力的冲量都凑不齐，塔会持续下沉。
    """
    cold = PhysicsWorld.from_scene(tower_scene(3, 1, False, 0.6))
    advance(cold, 0.6)
    cold_pen = cold.max_penetration() * 1000.0
    cold_speed = max(body.speed() for body in cold.bodies)

    warm = PhysicsWorld.from_scene(tower_scene(3, 1, True, 0.6))
    advance(warm, 0.6)
    warm_pen = warm.max_penetration() * 1000.0
    warm_speed = max(body.speed() for body in warm.bodies)

    ok = warm_pen < cold_pen * 0.75 and warm_speed < cold_speed * 0.5
    return ok, (
        f"穿透 {cold_pen:.3f} → {warm_pen:.3f} mm"
        f"（{cold_pen / warm_pen:.2f} 倍），"
        f"末速度 {cold_speed:.4f} → {warm_speed:.4f} m/s"
        f"（{cold_speed / max(warm_speed, 1e-9):.0f} 倍）"
    )


# ---------------------------------------------------------------------------
# 3. 与 TOI 共存
# ---------------------------------------------------------------------------

def test_no_double_apply_with_toi() -> tuple[bool, str]:
    """TOI 开启时不得重复施加同一笔冲量。

    TOI 把一个子步切成多段、每段各求解一次。若每段都暖启动，静置接触会在同一
    子步内被反复施加支撑冲量——实测棱柱落地峰值穿透 0.148 → 1.317 mm。
    这里用同一个高速棱柱落地场景守住这个上界。
    """
    radius, height = 0.35, 0.7
    polygon = [
        (radius * math.cos(2.0 * math.pi * i / 6.0),
         radius * math.sin(2.0 * math.pi * i / 6.0))
        for i in range(6)
    ]
    vertices, _faces = prism_mesh(polygon, height, z0=-height / 2.0)

    obj = ObjectSpec(
        "prism",
        ConvexHullGeometry(vertices, max_faces=32),
        TransformSpec(position=(0.0, 0.0, 1.6), orientation=(0.0, 0.0, 0.0, 1.0)),
        MaterialSpec(name="mat"),
        mass=1.0,
        physics=BodyPhysicsSpec(
            linear_velocity=(0.3, 0.15, -6.0), restitution=0.0, friction=0.5
        ),
    )
    scene = SceneSpec(
        name="prism_drop",
        objects=[obj],
        camera=None,
        lights=[],
        ground=GroundSpec(enabled=True),
        simulation=SimulationSpec(
            enabled=True, physics_dt=1.0 / 240.0, substeps=3, iterations=8,
            ground_z=0.0, duration=1.0, use_toi_splitting=True, warm_start=True,
        ),
        render=None,
    )

    world = PhysicsWorld.from_scene(scene)
    peak = 0.0
    for _ in range(240):
        world.step()
        peak = max(peak, world.max_penetration())

    peak_mm = peak * 1000.0
    return peak_mm < 1.0, f"高速棱柱落地峰值穿透 {peak_mm:.3f} mm（要求 < 1 mm）"


# ---------------------------------------------------------------------------
# 4. 确定性
# ---------------------------------------------------------------------------

def test_deterministic() -> tuple[bool, str]:
    """同一场景跑两遍，位姿必须逐位相同。

    暖启动引入了跨帧状态，一旦缓存里出现"依赖字典遍历顺序"的行为，
    确定性就会破掉——``reproduce.py`` 的逐字节比对会立刻失效。
    """
    first = PhysicsWorld.from_scene(tower_scene(3, 4, True, 0.5))
    advance(first, 0.5)
    second = PhysicsWorld.from_scene(tower_scene(3, 4, True, 0.5))
    advance(second, 0.5)

    same = poses(first) == poses(second)
    return same, (
        "两次运行位姿逐位相同（缓存未破坏确定性）" if same
        else "两次运行位姿不同：缓存引入了非确定性"
    )


# ---------------------------------------------------------------------------
# 5. 能量
# ---------------------------------------------------------------------------

def test_no_energy_injection() -> tuple[bool, str]:
    """静置 1 s 后总机械能不得高于初始值。

    暖启动最大的风险是"重复计费"——把上一帧的支撑冲量再施加一次，就等于凭空
    注入动量。能量单调不增是最直接的反证。
    """
    world = PhysicsWorld.from_scene(tower_scene(3, 1, True, 1.0))
    initial = world.total_energy()
    advance(world, 1.0)
    final = world.total_energy()

    ok = final <= initial + 1e-9
    return ok, f"E0={initial:.6f} J → E1={final:.6f} J，Δ={final - initial:+.3e} J"


CHECKS = [
    ("1. 默认开启、可关闭、可序列化", test_default_and_switch),
    ("2. 低迭代时显著减小下沉", test_reduces_sinking_at_low_iterations),
    ("3. 与 TOI 共存不重复施加", test_no_double_apply_with_toi),
    ("4. 跨帧缓存不破坏确定性", test_deterministic),
    ("5. 不注入能量", test_no_energy_injection),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("暖启动（warm starting）")
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
