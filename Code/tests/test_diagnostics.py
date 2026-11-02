# -*- coding: utf-8 -*-
"""诊断指标 ``PhysicsWorld.max_penetration()`` 的语义。

这个指标是几乎所有验收用例的判据（"峰值穿透 < 1 mm"），所以它本身必须先被测。
它要回答的问题是"**有没有陷进去**"，口径是**所有接触里最深的那个**：

- 地面接触
- 物体之间的接触
- 两者同时存在 → 取二者最大值

早先的实现里有一句 ``if self.ground is None: return 0.0``：自由空间场景调用它，
拿到的不是"没有穿透"，而是"根本没检测"。这种"恒 0"会静默地把断言变成永真，
比抛异常危险得多——所以四条用例分别钉住"自由空间能量出值""不接触才是 0"
"有地面时不退化""两类同时存在时取最大"。
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
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec

#: 球-球穿透是解析解（半径和减中心距），这里的容差只留给浮点误差
_EXACT = 1e-9


def _sphere(name: str, radius: float, position) -> RigidBody:
    return RigidBody(
        name=name,
        geometry=SphereGeometry(radius),
        mass=1.0,
        position=tuple(float(value) for value in position),
    )


def _world(bodies, ground):
    """手工构造世界：``from_scene`` 总会建地面，自由空间只能这样来。"""
    return PhysicsWorld(bodies=bodies, ground=ground, simulation=SimulationSpec())


def _reference(world) -> float:
    """直接从窄相流形汇总最大深度，作为**包装层**对照。

    它和 ``max_penetration()`` 调的是同一个 ``detect_contacts``，所以只验证
    "包装有没有把结果汇总对"，不验证窄相本身——窄相的正确性由本文件里球-球、
    球-地面的解析期望值，以及各自的专项测试负责。
    """
    manifolds = detect_contacts(world.bodies, world.ground)
    if not manifolds:
        return 0.0
    return max(manifold.deepest.penetration for manifold in manifolds)


# ---------------------------------------------------------------------------
# 1. 自由空间：两球重叠 0.05 m
# ---------------------------------------------------------------------------

def test_free_space_overlap() -> tuple[bool, str]:
    """没有地面时，球-球重叠必须被量出来。

    半径 0.1 + 0.2 = 0.3，中心距 0.25 → 穿透 0.05 m。
    """
    world = _world(
        [
            _sphere("a", 0.1, (0.0, 0.0, 10.0)),
            _sphere("b", 0.2, (0.0, 0.0, 10.25)),
        ],
        None,
    )
    penetration = world.max_penetration()
    ok = abs(penetration - 0.05) < _EXACT
    return ok, f"自由空间两球重叠 0.05 m → 指标={penetration:.9f}"


# ---------------------------------------------------------------------------
# 2. 自由空间：不接触就是 0
# ---------------------------------------------------------------------------

def test_free_space_no_contact() -> tuple[bool, str]:
    """不接触时必须真的是 0，而不是因为"没检测"才是 0。

    第 1 项已经证明自由空间**能**产出非零值，这里才能反过来确认 0 的含义。
    """
    world = _world(
        [
            _sphere("a", 0.1, (0.0, 0.0, 10.0)),
            _sphere("b", 0.2, (0.0, 0.0, 10.9)),
        ],
        None,
    )
    penetration = world.max_penetration()
    ok = penetration == 0.0
    return ok, f"自由空间相距 0.9 m（半径和 0.3）→ 指标={penetration}"


# ---------------------------------------------------------------------------
# 3. 有地面时结果不退化
# ---------------------------------------------------------------------------

def test_ground_still_reported() -> tuple[bool, str]:
    """带地面的场景里，地面穿透仍要被报出来，且与直接检测一致。"""
    world = _world([_sphere("ball", 0.1, (0.0, 0.0, 0.05))], GroundPlane(z=0.0))
    penetration = world.max_penetration()
    reference = _reference(world)
    ok = abs(penetration - 0.05) < _EXACT and abs(penetration - reference) < 1e-12
    return ok, (
        f"球半径 0.1、球心 z=0.05 → 陷地 0.05，指标={penetration:.9f}，"
        f"与直接检测一致={abs(penetration - reference) < 1e-12}"
    )


# ---------------------------------------------------------------------------
# 4. 地面与物-物同时存在：取最大值
# ---------------------------------------------------------------------------

def test_takes_max_of_both() -> tuple[bool, str]:
    """两类接触同时存在时，必须返回二者中**更深**的那个。

    两组数据互为对照，防止实现里"只取地面"或"只取物体对"：
    - 物-物 0.03 / 地面 0.06 → 期望 0.06
    - 物-物 0.06 / 地面 0.02 → 期望 0.06
    """
    # 物-物 0.03（0.1+0.1-0.17），地面 0.06（z=0.04）
    shallow_pair = _world(
        [
            _sphere("a", 0.1, (0.0, 0.0, 10.0)),
            _sphere("b", 0.1, (0.0, 0.0, 10.17)),
            _sphere("c", 0.1, (5.0, 0.0, 0.04)),
        ],
        GroundPlane(z=0.0),
    )
    # 物-物 0.06（0.1+0.1-0.14），地面 0.02（z=0.08）
    deep_pair = _world(
        [
            _sphere("a", 0.1, (0.0, 0.0, 10.0)),
            _sphere("b", 0.1, (0.0, 0.0, 10.14)),
            _sphere("c", 0.1, (5.0, 0.0, 0.08)),
        ],
        GroundPlane(z=0.0),
    )

    first = shallow_pair.max_penetration()
    second = deep_pair.max_penetration()
    ok = (
        abs(first - 0.06) < _EXACT
        and abs(second - 0.06) < _EXACT
        and abs(first - _reference(shallow_pair)) < 1e-12
        and abs(second - _reference(deep_pair)) < 1e-12
    )
    return ok, (
        f"物-物 0.03 + 地面 0.06 → {first:.9f}（期望 0.06）；"
        f"物-物 0.06 + 地面 0.02 → {second:.9f}（期望 0.06）"
    )


def test_multipoint_manifold_takes_deepest() -> tuple[bool, str]:
    """多点流形上必须取**最深**的那个点，不是第一个、也不是平均。

    前面几项都是球的单点接触，覆盖不到这一点。这里把盒子绕 x 轴倾 0.02 rad、
    球心落到 z=0.29：底面四角全部陷进地面，但深度分两档（约 15.9 mm / 3.9 mm）。
    指标必须等于这一档里的最大值，而不是最浅值或均值。
    """
    tilt = 0.02
    orientation = (math.cos(0.5 * tilt), math.sin(0.5 * tilt), 0.0, 0.0)
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((0.6, 0.6, 0.6)),
        mass=1.0,
        position=(0.0, 0.0, 0.29),
        orientation=orientation,
    )
    world = _world([box], GroundPlane(z=0.0))
    manifolds = detect_contacts(world.bodies, world.ground)
    depths = [point.penetration for manifold in manifolds for point in manifold.points]
    if len(depths) < 2:
        return False, f"倾斜盒应给出多点流形，实际 {len(depths)} 点"

    deepest = max(depths)
    shallowest = min(depths)
    reported = world.max_penetration()
    ok = (
        deepest - shallowest > 1e-3          # 确实有深浅两档，不是退化成一点
        and abs(reported - deepest) < 1e-12  # 取的是最大，不是最小/平均
    )
    return ok, (
        f"{len(depths)} 点，深度区间 [{shallowest * 1000:.3f}, {deepest * 1000:.3f}] mm，"
        f"指标={reported * 1000:.3f} mm（应等于最深点）"
    )


CHECKS = [
    ("1. 自由空间两球重叠 0.05 m", test_free_space_overlap),
    ("2. 自由空间不接触返回 0", test_free_space_no_contact),
    ("3. 有地面时仍报出穿透", test_ground_still_reported),
    ("4. 地面与物-物同时存在取最大", test_takes_max_of_both),
    ("5. 多点流形取最深点", test_multipoint_manifold_takes_deepest),
]


def main() -> int:
    print("=" * 88)
    print("诊断指标：max_penetration 覆盖地面与物体间两类接触")
    print("=" * 88)

    failures = 0
    for title, case in CHECKS:
        try:
            ok, detail = case()
        except Exception as exc:  # 单个用例崩溃不应中断整轮
            ok, detail = False, f"异常：{type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {detail}")
        if not ok:
            failures += 1

    print("-" * 88)
    print(f"结果：{len(CHECKS) - failures}/{len(CHECKS)} 通过")
    print("=" * 88)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
