# -*- coding: utf-8 -*-
"""宽阶段（空间哈希）测试。

宽阶段的正确性判据很特殊
------------------------
窄相只要漏一个接触，物体会穿模——这是肉眼可见的。宽阶段漏一对的后果**完全相同**
（窄相根本看不到那一对），但宽阶段为了性能**本来就会漏掉绝大多数对**。

所以不能问"哈希和暴力结果一样吗"，那会退化成"不许优化"。
正确的判据是两条：

1. **不漏**：凡是 AABB 相交的物体对，必须出现在哈希结果里。
   因为 AABB 相交是窄相接触的**必要条件**，保证了这一条就不可能漏接触。
2. **不虚增**：结果必须是暴力枚举的子集。这条几乎自动成立（哈希自己也做 AABB 过滤），
   但如果将来加入"相邻格子"之类的前置筛选，就可能引入不存在的对。

这两条合起来等价于：**哈希结果 == 所有 AABB 相交的物体对**。
测试就是直接验证这个等式。

另一组重点是退化情形
--------------------
格子边长选得太小，一个巨大物体会横跨成百上千个格子，插入开销爆炸。
实现里有"跨格超限 → 走全局列表"的兜底通道，这条通道必须单独测——
它在正常场景里永远不触发，一旦写错就会静默漏掉大物体与所有物体的接触。

运行::

    python Code/tests/test_broadphase.py
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

from src.geom.geometry import BoxGeometry, SphereGeometry
from src.geom.inertia import compute_inertia
from src.mathutils import quaternion as quat
from src.physics.body import RigidBody
from src.physics.broadphase import (
    BruteForceBroadphase,
    SpatialHashBroadphase,
    world_aabb,
)
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def make_sphere(name, radius, position) -> RigidBody:
    """造一个球刚体。"""
    geometry = SphereGeometry(radius)
    mass = geometry.volume() * 1000.0
    return RigidBody(
        name=name, geometry=geometry, position=position,
        orientation=quat.quat_identity(), mass=mass,
        inertia_local=compute_inertia(geometry, mass),
    )


def make_box(name, size, position, orientation=None) -> RigidBody:
    """造一个盒子刚体。"""
    geometry = BoxGeometry(size)
    mass = geometry.volume() * 1000.0
    return RigidBody(
        name=name, geometry=geometry, position=position,
        orientation=orientation or quat.quat_identity(), mass=mass,
        inertia_local=compute_inertia(geometry, mass),
    )


def overlapping_pairs(bodies) -> set[tuple[int, int]]:
    """暴力求出所有 AABB 相交的物体对——这是宽阶段应当给出的**精确答案**。"""
    boxes = [world_aabb(body) for body in bodies]
    expected = set()
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if boxes[i].overlaps(boxes[j]):
                expected.add((i, j))
    return expected


def random_scene(seed: int, count: int, spread: float = 1.2) -> list[RigidBody]:
    """生成一堆随机摆放、随机朝向的球与盒子。"""
    rng = random.Random(seed)
    bodies = []
    for index in range(count):
        position = (
            rng.uniform(-spread, spread),
            rng.uniform(-spread, spread),
            rng.uniform(0.0, 1.2),
        )
        if rng.random() < 0.5:
            bodies.append(make_sphere(f"s{index}", rng.uniform(0.1, 0.35), position))
        else:
            axis = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
            length = math.sqrt(sum(c * c for c in axis)) or 1.0
            axis = tuple(c / length for c in axis)
            orientation = quat.quat_from_axis_angle(axis, rng.uniform(0.0, math.pi))
            side = rng.uniform(0.2, 0.7)
            bodies.append(
                make_box(f"b{index}", (side, side, side), position, orientation)
            )
    return bodies


# ---------------------------------------------------------------------------
# 组一：AABB
# ---------------------------------------------------------------------------

def test_aabb_sphere() -> tuple[bool, str]:
    """球的 AABB 就是中心 ± 半径。"""
    body = make_sphere("s", 0.25, (1.0, -2.0, 0.5))
    box = world_aabb(body)
    ok = (
        abs(box.lower[0] - 0.75) < 1e-12 and abs(box.upper[0] - 1.25) < 1e-12
        and abs(box.lower[1] + 2.25) < 1e-12 and abs(box.upper[2] - 0.75) < 1e-12
    )
    return ok, f"球 AABB lower={tuple(round(c, 4) for c in box.lower)} upper={tuple(round(c, 4) for c in box.upper)}"


def test_aabb_rotated_box() -> tuple[bool, str]:
    """倾斜盒子的 AABB 必须**按朝向**算，不能用局部半尺寸。

    45° 倾斜的单位立方体，其世界 AABB 半宽是 √2/2 ≈ 0.7071 而不是 0.5。
    如果用局部半尺寸，AABB 会小 30%，直接漏掉大量真实接触。
    """
    upright = make_box("u", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    tilted = make_box(
        "t", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
        quat.quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(45.0)),
    )
    box_upright = world_aabb(upright)
    box_tilted = world_aabb(tilted)

    expected = math.sqrt(2.0) / 2.0
    ok = (
        abs(box_upright.upper[0] - 0.5) < 1e-12
        and abs(box_tilted.upper[0] - expected) < 1e-9
        and abs(box_tilted.upper[2] - expected) < 1e-9
        and abs(box_tilted.upper[1] - 0.5) < 1e-9   # 绕 y 转不改变 y 方向跨度
    )
    return ok, (
        f"正放半宽={box_upright.upper[0]:.6f}（期望 0.5），"
        f"倾斜 45° 半宽={box_tilted.upper[0]:.6f}（期望 {expected:.6f}）"
    )


# ---------------------------------------------------------------------------
# 组二：哈希结果 == 所有 AABB 相交对
# ---------------------------------------------------------------------------

def test_no_missing_pairs() -> tuple[bool, str]:
    """核心判据：哈希结果必须**恰好等于**所有 AABB 相交的物体对。

    少一个 → 漏接触（物体穿模）；多一个 → 白跑窄相（性能下降）。
    """
    trials = 120
    total_expected = 0
    mismatches = []

    for trial in range(trials):
        bodies = random_scene(seed=trial, count=random.Random(trial).randint(2, 16))
        expected = overlapping_pairs(bodies)
        broadphase = SpatialHashBroadphase(bodies=bodies)
        got = set(broadphase.candidate_pairs(bodies))
        total_expected += len(expected)

        if got != expected:
            missing = expected - got
            extra = got - expected
            mismatches.append(
                f"trial{trial}: 漏 {len(missing)} 多 {len(extra)}"
            )

    ok = not mismatches
    detail = "；".join(mismatches[:3]) if mismatches else "完全一致"
    return ok, f"{trials} 组随机场景，共 {total_expected} 个相交对，{detail}"


def test_cell_size_robust() -> tuple[bool, str]:
    """格子边长在很宽的范围内变化，结果都必须是同一组对。

    这条防的是"某些边长下哈希退化、靠运气才没漏"。
    """
    bodies = random_scene(seed=99, count=20)
    expected = overlapping_pairs(bodies)

    sizes = [0.08, 0.15, 0.3, 0.5, 0.9, 1.6, 3.0]
    results = []
    all_ok = True
    for size in sizes:
        broadphase = SpatialHashBroadphase(bodies=bodies, cell_size=size)
        got = set(broadphase.candidate_pairs(bodies))
        results.append((size, got == expected))
        if got != expected:
            all_ok = False

    return all_ok, "  ".join(f"h={s}:{'OK' if ok else 'FAIL'}" for s, ok in results)


def test_result_is_sorted() -> tuple[bool, str]:
    """输出必须按 ``(i, j)`` 升序排序。

    不排序的话，求解器的接触遍历顺序会随哈希表的插入顺序浮动，
    同一场景在不同规模下给出略有差异的结果——对可复现性是致命的。
    """
    bodies = random_scene(seed=5, count=18)
    broadphase = SpatialHashBroadphase(bodies=bodies)
    pairs = broadphase.candidate_pairs(bodies)

    ok = pairs == sorted(pairs) and all(i < j for i, j in pairs)
    return ok, f"{len(pairs)} 个候选对，已排序={pairs == sorted(pairs)}，i<j 全部成立={all(i < j for i, j in pairs)}"


# ---------------------------------------------------------------------------
# 组三：大物体兜底通道
# ---------------------------------------------------------------------------

def test_large_body_fallback() -> tuple[bool, str]:
    """一个巨大物体横跨成百上千个格子时，必须走兜底通道且**不能漏对**。

    兜底通道（``_large`` 列表）在正常场景里永远不触发，
    一旦写错就会静默漏掉大物体与所有其他物体的接触——
    症状是"小球穿过大平面"，而且只在物体尺寸悬殊时出现。
    """
    # 格子边长被强行设得很小 → 大盒子必定跨格超限
    bodies = [make_box("floor_like", (20.0, 20.0, 0.4), (0.0, 0.0, 0.2))]  # AABB 100x100x0.4
    for index in range(10):
        bodies.append(make_sphere(f"s{index}", 0.2, (index * 0.5 - 2.5, 0.0, 0.35)))

    broadphase = SpatialHashBroadphase(bodies=bodies, cell_size=0.25)
    pairs = broadphase.candidate_pairs(bodies)
    stats = broadphase.stats

    expected = overlapping_pairs(bodies)
    got = set(pairs)
    ok = (got == expected) and stats["large"] >= 1

    return ok, (
        f"格子 0.25 m、大物体 20×20 m：走兜底通道的大物体 {stats['large']} 个"
        f"（应 ≥ 1），格子数 {stats['cells']}，"
        f"候选对 {len(got)}（应为 {len(expected)}），"
        f"结果一致={got == expected}"
    )


# ---------------------------------------------------------------------------
# 组四：与窄相串联
# ---------------------------------------------------------------------------

def test_contact_set_identical() -> tuple[bool, str]:
    """同一场景下，两种宽阶段产出的**接触流形集合完全相同**。

    这是端到端判据：宽阶段只是加速手段，不应改变任何物理结果。
    """
    bodies_hash = random_scene(seed=31, count=14)
    bodies_brute = random_scene(seed=31, count=14)

    hash_pairs = SpatialHashBroadphase(bodies=bodies_hash).candidate_pairs(bodies_hash)
    brute_pairs = BruteForceBroadphase().candidate_pairs(bodies_brute)

    hash_manifolds = detect_contacts(bodies_hash, None, hash_pairs)
    brute_manifolds = detect_contacts(bodies_brute, None, brute_pairs)

    def signature(manifolds):
        return sorted(
            (m.body_a, m.body_b, len(m.points), round(m.deepest.penetration, 12))
            for m in manifolds
        )

    equal = signature(hash_manifolds) == signature(brute_manifolds)
    return equal, (
        f"哈希候选 {len(hash_pairs)} 对 → {len(hash_manifolds)} 个流形；"
        f"暴力 {len(brute_pairs)} 对 → {len(brute_manifolds)} 个流形；相同={equal}"
    )


def test_world_uses_broadphase() -> tuple[bool, str]:
    """物理世界接线正确：默认走哈希，切到 brute 时候选对数应变成 N(N-1)/2。"""
    def build(mode: str):
        bodies = [
            make_sphere(f"s{i}", 0.2, (i * 2.0, 0.0, 0.5))  # 彼此间距 2 m，互不相交
            for i in range(12)
        ]
        sim = SimulationSpec(
            enabled=True, physics_dt=1.0 / 240.0, substeps=1, iterations=4,
            gravity=(0.0, 0.0, 0.0), use_toi_splitting=False,
            broadphase=mode, duration=1.0,
        )
        return PhysicsWorld(bodies, None, sim)

    world_hash = build("spatial_hash")
    world_hash.step()
    count_hash = world_hash.last_candidate_count

    world_brute = build("brute")
    world_brute.step()
    count_brute = world_brute.last_candidate_count

    expected_brute = 12 * 11 // 2
    ok = count_brute == expected_brute and count_hash == 0
    return ok, (
        f"12 个相距 2 m 的球：暴力候选 {count_brute}（期望 {expected_brute}），"
        f"哈希候选 {count_hash}（期望 0）"
    )


def test_ground_contacts_unaffected() -> tuple[bool, str]:
    """地面接触不经过宽阶段，切换模式后必须完全一致。

    地面是无限平面，无法塞进哈希格子。它走 O(N) 的独立通道，
    这条测试防止将来有人"顺手"把地面也塞进网格里，导致接触丢失。
    """
    def build(mode: str):
        bodies = [
            make_sphere(f"s{i}", 0.25, (i * 0.4 - 1.2, 0.0, 0.25))
            for i in range(6)
        ]
        sim = SimulationSpec(
            enabled=True, physics_dt=1.0 / 240.0, substeps=1, iterations=4,
            gravity=(0.0, 0.0, -9.81), use_toi_splitting=False,
            broadphase=mode, duration=1.0,
        )
        return PhysicsWorld(bodies, GroundPlane(z=0.0), sim)

    results = []
    for mode in ("spatial_hash", "brute"):
        world = build(mode)
        for _ in range(120):
            world.step()
        results.append([round(body.position[2], 9) for body in world.bodies])

    ok = results[0] == results[1]
    return ok, (
        f"落定高度：哈希={results[0][:3]}…  暴力={results[1][:3]}…  完全一致={ok}"
    )


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. 球的 AABB", test_aabb_sphere),
    ("2. 倾斜盒的 AABB 按朝向计算", test_aabb_rotated_box),
    ("3. 哈希结果 == 所有 AABB 相交对", test_no_missing_pairs),
    ("4. 格子边长鲁棒性", test_cell_size_robust),
    ("5. 输出排序稳定", test_result_is_sorted),
    ("6. 大物体兜底通道不漏对", test_large_body_fallback),
    ("7. 接触流形集合与暴力一致", test_contact_set_identical),
    ("8. 物理世界接线正确", test_world_uses_broadphase),
    ("9. 地面接触不受影响", test_ground_contacts_unaffected),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 92)
    print("宽阶段（空间哈希）测试")
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
