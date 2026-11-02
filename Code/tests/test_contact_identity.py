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

from src.mathutils import quaternion as quat
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
from src.physics.support import TAG_LIMIT, feature_world, feature_world_tagged
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

def test_gjk_ids_stable_under_axis_spin() -> tuple[bool, str]:
    """柱体绕**自身轴**自转后，接触点的 ID 与位置配对一致。

    这条只证明一个具体情形：轴在世界空间不动，于是圆周采样基也不动，采样槽位
    随体一起转，标签自然跟上。

    它**不**证明倾斜稳定，更不证明跨越基底分界时稳定——那两件当前做不到，
    见第 8 项与 ``contact.py`` 里"跨帧稳定 = best effort"的措辞。
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
# 6. 接口边界：零方向与标签范围
# ---------------------------------------------------------------------------

def test_zero_direction_returns_tagged() -> tuple[bool, str]:
    """零方向必须返回 ``[(point, tag)]``，而不是裸点。

    这里曾经漏包标签：``feature_world_tagged`` 的零方向分支直接返回
    ``[support_world(...)]``，于是 ``feature_world()`` 解包时抛
    ``ValueError: too many values to unpack``。零向量在 GJK 的退化情形
    （两体中心重合、法向退化为零）下真的会出现，属于会崩的路径。
    """
    body = make_body(CylinderGeometry(0.3, 1.0), 0.0)
    tagged = feature_world_tagged(body, (0.0, 0.0, 0.0))
    shape_ok = len(tagged) == 1 and len(tagged[0]) == 2
    plain = feature_world(body, (0.0, 0.0, 0.0))   # 薄包装：当初就是这里崩的
    ok = shape_ok and len(plain) == 1 and len(plain[0]) == 3
    return ok, f"零方向 → {tagged}；feature_world → {len(plain)} 个裸点"


def test_all_tags_within_limit() -> tuple[bool, str]:
    """所有几何、所有方向下，标签都必须落在 ``[0, TAG_LIMIT)``。

    ``TAG_LIMIT`` 是 ``gjk`` 编码 ``tag_a * TAG_LIMIT + tag_b`` 的隐含宽度。
    编码本身不做检查，越界只能由 ``feature_world_tagged`` 的校验兜住；
    这条用例保证校验确实在跑，且现有几何都用得下 512 的宽度。
    """
    directions = (
        (0.0, 0.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.31, 0.22, -0.92),
        (0.0, 0.0, 0.0),
    )
    geometries = (
        ("盒", BoxGeometry((0.6, 0.6, 0.6))),
        ("柱", CylinderGeometry(0.3, 1.0)),
        ("胶囊", CapsuleGeometry(0.2, 0.6)),
        ("圆锥", ConeGeometry(0.3, 1.2)),
        ("圆台", FrustumGeometry(0.4, 0.15, 0.9)),
        ("球冠", SphericalCapGeometry(0.5, 0.5)),
        ("椭球", EllipsoidGeometry((0.5, 0.4, 0.3))),
        ("球", SphereGeometry(0.3)),
    )
    highest = -1
    for label, geometry in geometries:
        body = make_body(geometry, 0.0)
        for direction in directions:
            for _point, tag in feature_world_tagged(body, direction):
                if not 0 <= tag < TAG_LIMIT:
                    return False, f"{label} 在方向 {direction} 上给出越界标签 {tag}"
                highest = max(highest, tag)
    return True, (
        f"{len(geometries)} 种几何 × {len(directions)} 个方向，"
        f"最大标签={highest}（上限 {TAG_LIMIT}）"
    )


# ---------------------------------------------------------------------------
# 7. 圆盘采样槽位：固定在体坐标系
# ---------------------------------------------------------------------------

def _tilt(axis, angle):
    """绕单位轴转 ``angle`` 的四元数 ``(w, x, y, z)``。"""
    half = 0.5 * angle
    s = math.sin(half)
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def _to_body_frame(body, point):
    """世界点 → 体坐标：``R⁻¹ (p − x)``。"""
    offset = (
        point[0] - body.position[0],
        point[1] - body.position[1],
        point[2] - body.position[2],
    )
    return quat.quat_rotate_vector(quat.quat_conjugate(body.orientation), offset)


def test_ring_tags_stay_in_body_frame() -> tuple[bool, str]:
    """端面类支撑特征：同一个 tag 恒对应同一个**体坐标**槽位。

    覆盖三条入口——圆柱自己的端面分支，以及 ``_face_ring()`` 服务的圆锥底面、
    圆台底面/顶面、球冠底面。它们共用 ``_ring_basis()``，但入口不同，逐个跑一遍。

    验收口径是局部坐标，不是世界坐标——物体转了，采样点本来就该跟着转：

        p_local = R⁻¹ (p_world − x)

    改造前圆周起点取自 ``vec3.orthogonal_basis(世界轴)``，它按"轴的最小分量"挑
    辅助坐标轴，轴方向跨越该分界时基底不连续：竖直圆柱绕 y 轴倾斜 ``1e-6`` rad，
    同一个 tag 的世界位置位移 **0.424 m**（盘面直径 0.6 m）。改到体坐标系后
    这个跳变消失。

    这只解决了"基底跳变"，**不等于**完整跨帧持久性：端面 / 侧面 / 单点三种
    模式之间仍会切换（``_FACE_COS_TOL`` 约 3°），流形点数仍会变，P0-1 重建
    接触区域后采样点本身还会变。那些要靠局部锚点 + 有效性校验（`docs/07` §4.2）。
    """
    # 倾斜角必须留在端面模式的容差内（``_FACE_COS_TOL = 0.9986``，约 3°）：
    # 超过就会被判成侧面/单点接触，那是另一种模式，不归这条用例管。
    # 而基底跳变在 1e-6 rad 量级就会发生，所以这点角度足够暴露问题。
    poses = (
        ("恒等", (1.0, 0.0, 0.0, 0.0)),
        ("绕 x 1e-6", _tilt((1.0, 0.0, 0.0), 1e-6)),
        ("绕 y 1e-6", _tilt((0.0, 1.0, 0.0), 1e-6)),
        ("绕 x 1e-3", _tilt((1.0, 0.0, 0.0), 1e-3)),
        ("绕 y 1e-3", _tilt((0.0, 1.0, 0.0), 1e-3)),
        ("绕 (0.6,0.8,0) 2.5e-3", _tilt((0.6, 0.8, 0.0), 2.5e-3)),
    )
    cases = (
        ("圆柱端面", CylinderGeometry(0.3, 1.0), (0.0, 0.0, -1.0)),
        ("圆锥底面", ConeGeometry(0.3, 1.2), (0.0, 0.0, -1.0)),
        ("圆台底面", FrustumGeometry(0.4, 0.15, 0.9), (0.0, 0.0, -1.0)),
        ("圆台顶面", FrustumGeometry(0.4, 0.15, 0.9), (0.0, 0.0, 1.0)),
        ("球冠底面", SphericalCapGeometry(0.5, 0.5), (0.0, 0.0, -1.0)),
    )

    failures: list[str] = []
    report: list[str] = []
    for label, geometry, direction in cases:
        reference: dict[int, tuple[float, float, float]] = {}
        worst_local = 0.0
        worst_pose = ""
        unique_ok = True
        point_count = 0

        for name, orientation in poses:
            body = make_body(geometry, 0.0, orientation)
            tagged = feature_world_tagged(body, direction)
            point_count = len(tagged)
            tags = [tag for _point, tag in tagged]
            unique_ok = unique_ok and len(tags) == len(set(tags))
            if point_count != 4:
                failures.append(f"{label}/{name}：{point_count} 点（期望 4）")
                break

            for point, tag in tagged:
                local = _to_body_frame(body, point)
                if tag not in reference:
                    reference[tag] = local
                    continue
                deviation = math.sqrt(
                    sum((local[axis] - reference[tag][axis]) ** 2 for axis in range(3))
                )
                if deviation > worst_local:
                    worst_local = deviation
                    worst_pose = name

        if point_count != 4:
            continue
        if not unique_ok or worst_local >= 1e-9:
            failures.append(
                f"{label}：唯一={unique_ok}，体坐标偏差={worst_local:.1e}（{worst_pose}）"
            )
        report.append(f"{label} 4 点/偏差 {worst_local:.1e}")

    # 姿态确实改变了采样点的世界位置（否则这些用例等于没转）
    flat_body = make_body(CylinderGeometry(0.3, 1.0), 0.0)
    turned_body = make_body(
        CylinderGeometry(0.3, 1.0), 0.0, _tilt((0.0, 1.0, 0.0), 1e-3)
    )
    flat_zero = dict((tag, point) for point, tag in feature_world_tagged(flat_body, (0.0, 0.0, -1.0)))[0]
    turned_zero = dict((tag, point) for point, tag in feature_world_tagged(turned_body, (0.0, 0.0, -1.0)))[0]
    world_moved = math.sqrt(
        sum((flat_zero[axis] - turned_zero[axis]) ** 2 for axis in range(3))
    )
    if world_moved <= 1e-5:
        failures.append(f"倾斜 1e-3 rad 后世界位置没变（{world_moved:.1e} m），用例空转")

    if failures:
        return False, "；".join(failures)
    return True, (
        f"{len(cases)} 种端面 × {len(poses)} 个姿态：" + "；".join(report)
        + f"；倾斜 1e-3 rad 世界位置移动 {world_moved:.2e} m"
    )


# ---------------------------------------------------------------------------
# 8-9. 偏心堆叠：缓存按点分开存
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
    ("5. 绕自身轴自转时 ID 与位置配对一致", test_gjk_ids_stable_under_axis_spin),
    ("6. 零方向返回带标签", test_zero_direction_returns_tagged),
    ("7. 所有标签都在 TAG_LIMIT 内", test_all_tags_within_limit),
    ("8. 圆盘标签在体坐标系里稳定", test_ring_tags_stay_in_body_frame),
    ("9. 偏心负载：缓存键数 = 接触点数", test_cache_keys_per_contact_point),
    ("10. 偏心负载：四点冲量分别存储", test_impulses_stored_separately),
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
