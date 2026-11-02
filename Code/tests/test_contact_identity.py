# -*- coding: utf-8 -*-
"""接触点身份（``ContactPoint.feature_id``）的契约测试。

契约（写在 ``contact.ContactPoint`` 的文档里）
--------------------------------------------
1. **同一物体对内唯一**：一次求解里的各个接触点，``feature_id`` 互不相同。
2. **由几何特征决定**：盒用顶点序号、凸包用顶点下标、回转体用"端面序号 × 4 +
   圆周序号"、GJK 流形用 ``tag_a * TAG_LIMIT + tag_b``。**不是**列表下标。

为什么这两条值得单独一个文件
--------------------------
暖启动的缓存键是 ``(body_a, body_b, feature_id)``。键重复时后写的覆盖先写的，
下一帧再把那个值施给所有共用该键的点；键用列表下标时，倾斜裁掉一个浅点就会
让后面所有点的身份整体前移。两种错法的症状都很隐蔽——不是穿透爆炸，而是
静置物体缓慢自旋、偏心负载下支撑点悄悄换位。

这个文件用可复现的数字把两条要求钉住：地面四点、通用流形四点、偏心堆叠的
缓存条目数。
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
    CapsuleGeometry,
    ConeGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    SphereGeometry,
    SphericalCapGeometry,
)
from src.physics.body import RigidBody
from src.physics.narrowphase import (
    GroundPlane,
    _collide_points,
    collide_box_ground,
    collide_convex_ground,
    collide_cylinder_ground,
    detect_contacts,
)
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SceneSpec

GROUND = GroundPlane(z=0.0)


def make_body(geometry, z: float, orientation=(1.0, 0.0, 0.0, 0.0), mass: float = 1.0):
    return RigidBody(
        name="probe",
        geometry=geometry,
        mass=mass,
        position=(0.0, 0.0, z),
        orientation=orientation,
    )


def _ids(points) -> list[int]:
    return [point.feature_id for point in points]


# ---------------------------------------------------------------------------
# 1. 地面：每种几何的接触点 ID 都唯一
# ---------------------------------------------------------------------------

def test_ground_ids_unique() -> tuple[bool, str]:
    """所有几何的地面接触点，ID 必须在流形内互不相同。

    覆盖三条地面路径：盒（专属枚举）、柱/胶囊（专属解析）、
    圆锥/圆台/球冠/椭球/球（支撑特征通用解）。
    """
    sideways = (math.cos(math.pi * 0.25), math.sin(math.pi * 0.25), 0.0, 0.0)
    cases = (
        ("盒", make_body(BoxGeometry((0.6, 0.6, 0.6)), 0.25)),
        ("柱", make_body(CylinderGeometry(0.3, 1.0), 0.4)),
        ("胶囊横放", make_body(CapsuleGeometry(0.2, 0.6), 0.15, sideways)),
        ("圆锥", make_body(ConeGeometry(0.3, 1.2), 0.2)),
        ("圆台", make_body(FrustumGeometry(0.4, 0.15, 0.9), 0.25)),
        ("球冠", make_body(SphericalCapGeometry(0.5, 0.5), 0.15)),
        ("椭球", make_body(EllipsoidGeometry((0.5, 0.4, 0.3)), 0.25)),
        ("球", make_body(SphereGeometry(0.3), 0.2)),
    )

    failures = []
    details = []
    for label, body in cases:
        points = detect_contacts([body], GROUND)
        if not points:
            failures.append(f"{label} 没有地面接触")
            continue
        ids = _ids(points[0].points)
        details.append(f"{label} {len(ids)} 点 {ids}")
        if len(ids) != len(set(ids)):
            failures.append(f"{label} ID 重复：{ids}")

    return not failures, (
        "；".join(details) if not failures else "；".join(failures)
    )


# ---------------------------------------------------------------------------
# 2. 地面：ID 确实来自几何特征，而不是列表序号
# ---------------------------------------------------------------------------

def test_ground_ids_match_features() -> tuple[bool, str]:
    """盒底四角必须是 {0, 2, 4, 6}，柱端面必须是 {0, 1, 2, 3}。

    这条防的是"唯一但随机"：只要 ID 与顶点/圆周序号对应，就能解释、能比对，
    也才能保证下一帧同一个角拿到同一个键。
    """
    box = make_body(BoxGeometry((0.6, 0.6, 0.6)), 0.25)
    box_ids = _ids(collide_box_ground(box, GROUND))

    cylinder = make_body(CylinderGeometry(0.3, 1.0), 0.4)
    cylinder_ids = _ids(collide_cylinder_ground(cylinder, GROUND))

    ok = box_ids == [0, 2, 4, 6] and cylinder_ids == [0, 1, 2, 3]
    return ok, (
        f"盒底四角 ID={box_ids}（应 [0, 2, 4, 6]，即 iz=0 的四个顶点）；"
        f"柱底圆周 ID={cylinder_ids}（应 [0, 1, 2, 3]）"
    )


# ---------------------------------------------------------------------------
# 3. 裁掉浅点，剩余点的身份不变
# ---------------------------------------------------------------------------

def test_ids_survive_point_pruning() -> tuple[bool, str]:
    """盒子倾斜后只剩两个角接触，剩下的 ID 必须是原子集的子集。

    这是"不能用列表下标"的直接判据：按下标编号时，裁掉两点后剩下的是
    ``[0, 1]``；按顶点序号编号时剩下的是 ``{0, 2, 4, 6}`` 里的某两个。
    """
    tilt = 0.15
    orientation = (math.cos(0.5 * tilt), 0.0, math.sin(0.5 * tilt), 0.0)
    body = make_body(BoxGeometry((0.6, 0.6, 0.6)), 0.32, orientation)
    ids = _ids(collide_box_ground(body, GROUND))

    full = {0, 2, 4, 6}
    subset = set(ids) < full
    not_renumbered = set(ids) != set(range(len(ids)))
    ok = bool(ids) and subset and not_renumbered
    return ok, (
        f"倾斜 {math.degrees(tilt):.1f}° 后 {len(ids)} 点，ID={ids}；"
        f"是底面四角 {sorted(full)} 的真子集={subset}；"
        f"未被重新编号={not_renumbered}"
    )


# ---------------------------------------------------------------------------
# 4. 通用流形（GJK/EPA）：ID 唯一
# ---------------------------------------------------------------------------

def test_gjk_manifold_ids_unique() -> tuple[bool, str]:
    """走 GJK 的物体对（柱-盒）给出的四点流形，ID 必须互不相同。

    这里用的是审阅文档里那个偏置案例：接触点的**位置**目前仍不对（那是下一个
    里程碑要修的），但身份必须是四个互不相同的键——否则修好位置之后，缓存
    依然只存得下一个角的冲量。
    """
    cylinder = make_body(CylinderGeometry(1.0, 1.0), 0.0)
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((1.0, 1.0, 1.0)),
        mass=1.0,
        position=(1.3, 0.0, 0.9),
    )
    points = _collide_points(cylinder, box)
    ids = _ids(points)
    ok = len(ids) >= 2 and len(ids) == len(set(ids))
    return ok, (
        f"{len(ids)} 点，ID={ids}，唯一={len(ids) == len(set(ids))}；"
        f"编码形式 tag_a*512+tag_b，故都 ≥512（单点才会是 0）"
    )


# ---------------------------------------------------------------------------
# 5. 通用流形：微小运动不改变身份
# ---------------------------------------------------------------------------

def test_gjk_ids_stable_under_small_motion() -> tuple[bool, str]:
    """柱体绕自身轴微小转动后，同一个接触角应保持同一个 ID。

    判据不看 ID 序列（顺序会随深度排序变），而是**按位置配对**：对每个点在
    另一帧里找最近的点，比较两者的 ID。序号编号会在这里成片错位。
    """
    base = make_body(CylinderGeometry(0.3, 0.8), 0.45)
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((0.9, 0.9, 0.4)),
        mass=1.0,
        position=(0.0, 0.0, 0.2),
    )
    before = _collide_points(base, box)
    if len(before) < 2:
        return False, f"基准姿态只有 {len(before)} 点，无法比较身份"

    angle = 1e-3
    turned = make_body(
        CylinderGeometry(0.3, 0.8),
        0.45,
        (math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle)),
    )
    after = _collide_points(turned, box)

    mismatches = 0
    for point in before:
        nearest = min(
            after,
            key=lambda other: sum(
                (point.position[axis] - other.position[axis]) ** 2 for axis in range(3)
            ),
        )
        if nearest.feature_id != point.feature_id:
            mismatches += 1

    ok = len(after) == len(before) and mismatches == 0
    return ok, (
        f"扰动 {math.degrees(angle):.3f}°：前 {len(before)} 点 / 后 {len(after)} 点，"
        f"按位置配对后 ID 不一致的点数={mismatches}"
    )


# ---------------------------------------------------------------------------
# 6-7. 偏心堆叠：缓存按点分开存
# ---------------------------------------------------------------------------

def _offcenter_stack() -> PhysicsWorld:
    scene = SceneSpec.from_dict(
        {
            "name": "offcenter_stack",
            "objects": [
                {
                    "name": "base",
                    "geometry": {"type": "box", "size": [0.6, 0.6, 0.6]},
                    "transform": {"position": [0.0, 0.0, 0.3]},
                    "mass": 2.0,
                },
                {
                    "name": "load",
                    "geometry": {"type": "box", "size": [0.3, 0.3, 0.3]},
                    "transform": {"position": [0.22, 0.0, 0.75]},
                    "mass": 2.0,
                },
            ],
            "simulation": {
                "enabled": True,
                "physics_dt": 1.0 / 240.0,
                "substeps": 2,
                "iterations": 16,
                "duration": 1.5,
                "use_toi_splitting": False,
            },
        }
    )
    return PhysicsWorld.from_scene(scene)


def test_cache_keys_per_contact_point() -> tuple[bool, str]:
    """偏心负载下，地面四个接触点必须各占一个缓存键。

    旧实现四点共用 ``feature_id=0``，缓存里只剩一条：实测四角累积冲量是
    0.011 / 0.015 / 0.030 / 0.026（相差近 3 倍），写回时互相覆盖。
    """
    world = _offcenter_stack()
    for _ in range(360):
        world.step()

    manifolds = detect_contacts(world.bodies, world.ground)
    ground_points = [
        point
        for manifold in manifolds
        if manifold.body_a < 0
        for point in manifold.points
    ]
    cache_keys = [key for key in world.solver._impulse_cache if key[0] < 0]
    ids = _ids(ground_points)

    ok = (
        len(ground_points) >= 4
        and len(set(ids)) == len(ground_points)
        and len(cache_keys) == len(ground_points)
    )
    return ok, (
        f"地面接触点={len(ground_points)}，ID={ids}，"
        f"缓存键={len(cache_keys)}（应等于点数）"
    )


def test_impulses_stored_separately() -> tuple[bool, str]:
    """偏心负载下四角的累积冲量本来就不相同，缓存里也必须保留这个差异。

    若缓存只存一个标量，下一帧会把同一个值施给四个角：净冲量偏大、分布抹平。
    """
    world = _offcenter_stack()
    for _ in range(360):
        world.step()

    values = [
        round(entry[0], 6)
        for key, entry in world.solver._impulse_cache.items()
        if key[0] < 0
    ]
    ok = len(values) >= 4 and len(set(values)) >= 2 and min(values) >= 0.0
    return ok, (
        f"地面缓存条目={len(values)}，互不相同的冲量={len(set(values))} 个："
        f"{sorted(values)}"
    )


CHECKS = [
    ("1. 各几何地面接触的 ID 唯一", test_ground_ids_unique),
    ("2. 地面 ID 与几何特征对应", test_ground_ids_match_features),
    ("3. 倾斜裁点不改变剩余点身份", test_ids_survive_point_pruning),
    ("4. GJK 流形 ID 唯一", test_gjk_manifold_ids_unique),
    ("5. GJK 流形 ID 微小运动下稳定", test_gjk_ids_stable_under_small_motion),
    ("6. 偏心负载：缓存键数 = 接触点数", test_cache_keys_per_contact_point),
    ("7. 偏心负载：四点冲量分别存储", test_impulses_stored_separately),
]


def main() -> int:
    print("=" * 88)
    print("接触点身份：唯一 + 特征驱动")
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
