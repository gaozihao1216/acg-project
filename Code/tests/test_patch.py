# -*- coding: utf-8 -*-
"""支撑 patch 语义层（``physics.patch``）的契约测试。

这一层负责把"哪一片几何在接触"变成"哪些位置真的重叠"，P0-1 的四个误接触点
就是跳过它直接配对采样点的结果。这里钉四件事：

1. 退化由**投影后**的二维形状决定（三维的多边形压扁成线段也要正确处理）；
2. 交集点铺开在交叠区里、稀疏到 4 点以内、而且确定性；
3. 两侧锚点确实落在各自表面上，深度沿法向一致；
4. 交换 A/B 得到同一组位置——这是 P0-1 第 4 项验收的前提。

运行::

    python Code/tests/test_patch.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import BoxGeometry, CylinderGeometry
from src.mathutils import quaternion as quat
from src.physics import gjk, patch
from src.physics.body import RigidBody
from src.physics.support import (
    PATCH_POLYGON,
    PATCH_SEGMENT,
    SupportPatch,
    support_patch_world,
)

#: P0-1 的场景常量，与 ``test_contact_region.py`` 保持一致
CYLINDER_RADIUS = 1.0
BOX_HALF = 0.5
BOX_CENTER = (1.3, 0.0, 0.9)
OVERLAP_MIN_X = BOX_CENTER[0] - BOX_HALF      # 0.8
OVERLAP_MAX_X = CYLINDER_RADIUS               # 1.0
NORMAL = (0.0, 0.0, 1.0)
_OPPOSITE = (0.0, 0.0, -1.0)


def _scene():
    cylinder = RigidBody(
        name="cylinder", geometry=CylinderGeometry(CYLINDER_RADIUS, 1.0), mass=1.0
    )
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF)),
        mass=1.0,
        position=BOX_CENTER,
    )
    return cylinder, box


def test_overlap_in_p01_scene() -> tuple[bool, str]:
    """P0-1 场景：交集非空、点数 ≤4、且落在 0.8≤x≤1.0 的搭接带上。"""
    cylinder, box = _scene()
    patch_a = patch.contact_patch_world(cylinder, NORMAL)
    patch_b = patch.contact_patch_world(box, _OPPOSITE)
    result = patch.overlap(patch_a, patch_b, NORMAL)

    problems = []
    if not result.points:
        problems.append(f"交集为空（reason={result.empty_reason}）")
        return False, "；".join(problems)
    if result.empty_reason is not None:
        problems.append(f"非空却带着 empty_reason={result.empty_reason}")

    anchors = patch.contact_anchors(result, patch_a, patch_b, NORMAL)
    if not anchors:
        problems.append("交集非空但一个锚点都没留下")
    if len(anchors) > patch.MAX_CONTACT_POINTS:
        problems.append(f"{len(anchors)} 点，超过上限 {patch.MAX_CONTACT_POINTS}")
    outside = [
        (round(anchor.position[0], 4), round(anchor.position[1], 4))
        for anchor in anchors
        if not (
            OVERLAP_MIN_X - 1e-6 <= anchor.position[0] <= OVERLAP_MAX_X + 1e-6
            and abs(anchor.position[1]) <= BOX_HALF + 1e-6
        )
    ]
    if outside:
        problems.append(f"落在交叠区外的点={outside}")
    depths = [anchor.penetration for anchor in anchors]
    if max(abs(d - 0.1) for d in depths) > 1e-9:
        problems.append(f"深度={[round(d, 6) for d in depths]}，应为 0.1")
    if not all(abs(anchor.position[2] - 0.45) < 1e-9 for anchor in anchors):
        problems.append("接触点不在两表面中点（z 应为 0.45）")

    # 端到端：同一条路径接进 _build_manifold 后，**不应该**走到单点回退
    patch.reset_fallback_count()
    contacts = gjk.collide(cylinder, box)
    if patch.fallback_count() != 0:
        problems.append(f"流形走了单点回退 {patch.fallback_count()} 次（交集不该为空）")
    if len(contacts) != len(anchors):
        problems.append(f"流形 {len(contacts)} 点，patch 层 {len(anchors)} 点，不一致")
    if len({point.feature_id for point in contacts}) != len(contacts):
        problems.append("流形上的 feature_id 有重复")

    return not problems, ("；".join(problems) if problems else
                          f"{len(anchors)} 点，x∈"
                          f"[{min(a.position[0] for a in anchors):.4f}, "
                          f"{max(a.position[0] for a in anchors):.4f}]，深度 0.1，"
                          f"回退 0 次")


def test_projected_kind_wins() -> tuple[bool, str]:
    """三维写着 POLYGON、投影后退化成线段的 patch 必须按线段处理。

    构造：三个点其实共线（沿 y 轴），但 ``kind`` 谎称多边形；上方一个盒子的
    底面去和它求交，正确的交集是那条线段被裁掉之后的部分，而不是"零面积多边形"。
    """
    line_like = SupportPatch(
        vertices=[(0.0, -1.0, 0.5), (0.0, 0.0, 0.5), (0.0, 1.0, 0.5)],
        kind=PATCH_POLYGON,
        normal=(0.0, 0.0, 1.0),
    )
    # 盒子底面在 z = 0（中心 z = 0.5），于是与那条 z = 0.5 的"线段 patch"
    # 沿法向嵌入 0.5 m——切向交叠与法向深度是分开算的，正好用来核对深度。
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((1.0, 1.0, 1.0)),
        mass=1.0,
        position=(0.0, 0.0, 0.5),
    )
    patch_b = patch.contact_patch_world(box, _OPPOSITE)
    result = patch.overlap(line_like, patch_b, NORMAL)

    problems = []
    if len(result.points) != 2:
        problems.append(f"交集应是 2 点（线段），得到 {len(result.points)} 点")
    else:
        span = math.dist(result.points[0], result.points[1])
        # 线段 y ∈ [-1, 1] 被盒面裁成 y ∈ [-0.5, 0.5]，长度 1
        if abs(span - 1.0) > 1e-9:
            problems.append(f"线段长度={span:.6f}，应为 1.0")
    anchors = patch.contact_anchors(result, line_like, patch_b, NORMAL)
    if len(anchors) != 2:
        problems.append(f"{len(anchors)} 个锚点")
    elif max(abs(anchor.penetration - 0.5) for anchor in anchors) > 1e-9:
        problems.append(f"深度={[round(a.penetration, 6) for a in anchors]}，应为 0.5")

    return not problems, ("；".join(problems) if problems else
                          f"退化成 {len(result.points)} 点线段，深度 0.5")


def test_swap_matches() -> tuple[bool, str]:
    """交换 A/B：交集点集一致，锚点位置一致（法向同时取反，与 GJK 一致）。"""
    cylinder, box = _scene()
    patch_a = patch.contact_patch_world(cylinder, NORMAL)
    patch_b = patch.contact_patch_world(box, _OPPOSITE)
    forward = patch.contact_anchors(
        patch.overlap(patch_a, patch_b, NORMAL), patch_a, patch_b, NORMAL
    )
    backward = patch.contact_anchors(
        patch.overlap(patch_b, patch_a, _OPPOSITE), patch_b, patch_a, _OPPOSITE
    )

    problems = []
    if len(forward) != len(backward):
        problems.append(f"点数不同：{len(forward)} vs {len(backward)}")
    else:
        worst = max(
            min(math.dist(a.position, b.position) for b in backward) for a in forward
        )
        if worst > 1e-9:
            problems.append(f"位置最大偏差 {worst:.2e} m")

    return not problems, ("；".join(problems) if problems else
                          f"{len(forward)} 点，交换后位置偏差 ≤ 1e-9 m")


def test_reduce_is_deterministic_and_spread() -> tuple[bool, str]:
    """100 点圆盘：稀疏到 4 点、两个方向都铺开、两次调用完全相同。"""
    disc = [
        (math.cos(2.0 * math.pi * index / 100), math.sin(2.0 * math.pi * index / 100))
        for index in range(100)
    ]
    first = patch.reduce_to_four(disc)
    second = patch.reduce_to_four(list(reversed(disc)))

    problems = []
    if len(first) != patch.MAX_CONTACT_POINTS:
        problems.append(f"{len(first)} 点，应为 4")
    if first != second:
        problems.append("输入顺序改变后结果不同（不是确定性算法）")
    if first:
        span_x = max(p[0] for p in first) - min(p[0] for p in first)
        span_y = max(p[1] for p in first) - min(p[1] for p in first)
        if span_x < 1.9 or span_y < 1.9:
            problems.append(f"铺开不足：跨度 {span_x:.3f} × {span_y:.3f}")
    if first != sorted(first):
        problems.append("返回顺序不是按 (x, y) 排序")

    short = patch.reduce_to_four([(0.0, 0.0), (1.0, 0.0)])
    if len(short) != 2:
        problems.append(f"2 点输入被补成了 {len(short)} 点")

    return not problems, ("；".join(problems) if problems else
                          f"4 点，跨度 {span_x:.3f} × {span_y:.3f}，输入反向结果不变")


def test_empty_overlap_reasons() -> tuple[bool, str]:
    """分离给 ``disjoint``，没有 patch 给 ``missing_patch``，两种情况都是空列表。"""
    box_a = RigidBody(
        name="a", geometry=BoxGeometry((1.0, 1.0, 1.0)), mass=1.0, position=(0.0, 0.0, 0.0)
    )
    box_b = RigidBody(
        name="b", geometry=BoxGeometry((1.0, 1.0, 1.0)), mass=1.0, position=(5.0, 0.0, 0.0)
    )
    patch_a = patch.contact_patch_world(box_a, NORMAL)
    patch_b = patch.contact_patch_world(box_b, _OPPOSITE)
    disjoint = patch.overlap(patch_a, patch_b, NORMAL)
    missing = patch.overlap(SupportPatch([], PATCH_SEGMENT, NORMAL), patch_b, NORMAL)

    problems = []
    if disjoint.points or disjoint.empty_reason != "disjoint":
        problems.append(f"分离：points={disjoint.points}, reason={disjoint.empty_reason}")
    if missing.points or missing.empty_reason != "missing_patch":
        problems.append(f"缺 patch：points={missing.points}, reason={missing.empty_reason}")
    if patch.contact_anchors(disjoint, patch_a, patch_b, NORMAL):
        problems.append("空交集仍然产生了锚点")

    return not problems, ("；".join(problems) if problems else
                          "disjoint / missing_patch 两种空集都正确且不产生锚点")


def test_match_tags_unique() -> tuple[bool, str]:
    """标签分配一对一：4 点用满 4 个槽位，3 点用 3 个，且不撞号。"""
    tagged = [
        ((1.0, 1.0, 0.0), 0),
        ((-1.0, 1.0, 0.0), 1),
        ((-1.0, -1.0, 0.0), 2),
        ((1.0, -1.0, 0.0), 3),
    ]
    problems = []
    corners = [(1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
    four = patch.match_tags(corners, tagged, NORMAL)
    if sorted(four) != [0, 1, 2, 3]:
        problems.append(f"4 点标签={four}，应为 0..3 各一次")

    # 同一位置重复三次：数量与槽位不等，走贪心一对一，仍必须互不相同
    three = patch.match_tags([(1.0, 1.0, 0.0)] * 3, tagged, NORMAL)
    if len(set(three)) != 3:
        problems.append(f"3 个重合点拿到标签 {three}，应互不相同")

    empty = patch.match_tags([], tagged, NORMAL)
    if empty:
        problems.append("空输入应返回空列表")

    return not problems, ("；".join(problems) if problems else
                          f"4 点→{four}，3 重合点→{three}")


def test_contact_segment_budget() -> tuple[bool, str]:
    """接触路径的边数封顶 64；求交路径仍按 0.5 mm 误差取更多边。"""
    cylinder, _box = _scene()
    contact_patch = patch.contact_patch_world(cylinder, NORMAL)
    fine_patch = support_patch_world(cylinder, NORMAL)

    problems = []
    if len(contact_patch.vertices) > patch.MAX_CONTACT_SEGMENTS:
        problems.append(
            f"接触 patch {len(contact_patch.vertices)} 边，超过 {patch.MAX_CONTACT_SEGMENTS}"
        )
    if len(fine_patch.vertices) <= patch.MAX_CONTACT_SEGMENTS:
        problems.append(
            f"求交 patch 只有 {len(fine_patch.vertices)} 边，应比接触路径更密"
        )

    return not problems, ("；".join(problems) if problems else
                          f"接触 {len(contact_patch.vertices)} 边 / 求交 "
                          f"{len(fine_patch.vertices)} 边")


def test_tilted_cap_per_point_depth() -> tuple[bool, str]:
    """倾斜 2° 的端面：深度必须按**真实端面**逐点算，高侧不给穿透约束。

    这是端面选取容差（约 3°）的直接后果：patch 交出的是一整片**倾斜**圆盘，
    而旧实现用"垂直于碰撞法向的平均平面"算深度，于是整片共用一个值。半径 0.3 m
    倾斜 2° 时端面高度在边缘处差 ``0.3·sin2° ≈ 10.5 mm``：低侧少报一半，高侧
    把本已分离的点也建了约束（实测四点都报 9.70 mm，真实是 20.16 mm / −0.77 mm）。
    """
    radius, height = 0.3, 1.0
    angle = math.radians(2.0)
    orientation = quat.quat_from_axis_angle((1.0, 0.0, 0.0), angle)
    box_top = 0.4
    cap_center_z = box_top - 0.0097          # 端面中心嵌入 9.7 mm
    position_z = cap_center_z + 0.5 * height * math.cos(angle)
    cylinder = RigidBody(
        name="cylinder",
        geometry=CylinderGeometry(radius, height),
        mass=1.0,
        position=(0.0, 0.0, position_z),
        orientation=orientation,
    )
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((2.0, 2.0, 0.8)),
        mass=1.0,
        position=(0.0, 0.0, 0.0),
    )
    contacts = gjk.collide(cylinder, box)

    axis = quat.quat_rotate_vector(orientation, (0.0, 0.0, 1.0))
    cap_center = (
        cylinder.position[0] - axis[0] * 0.5 * height,
        cylinder.position[1] - axis[1] * 0.5 * height,
        cylinder.position[2] - axis[2] * 0.5 * height,
    )

    def true_penetration(point) -> float:
        """该点正下方端面与盒顶之间的嵌入量（负值表示分离）。"""
        offset = (
            point.position[0] - cap_center[0],
            point.position[1] - cap_center[1],
            point.position[2] - cap_center[2],
        )
        along = sum(offset[i] * axis[i] for i in range(3))
        surface_z = point.position[2] - along / axis[2]
        return box_top - surface_z

    problems = []
    if not contacts:
        return False, "没有产生接触点"
    for point in contacts:
        expected = true_penetration(point)
        if abs(point.penetration - expected) > 1e-6:
            problems.append(
                f"x={point.position[0]:.4f} y={point.position[1]:.4f}："
                f"报 {point.penetration * 1e3:.3f} mm，真实 {expected * 1e3:.3f} mm"
            )
        if expected < -1e-9:
            problems.append(
                f"y={point.position[1]:.4f} 处已分离 {(-expected) * 1e3:.3f} mm，"
                f"不该建立约束"
            )
    deepest = max(true_penetration(point) for point in contacts)
    if abs(deepest - 0.0202) > 5e-4:
        problems.append(f"最深点 {deepest * 1e3:.3f} mm，应约 20.2 mm")

    return not problems, ("；".join(problems) if problems else
                          f"{len(contacts)} 点，深度 "
                          f"{min(true_penetration(p) for p in contacts) * 1e3:.3f}"
                          f"–{deepest * 1e3:.3f} mm（都 ≥ 0）")


def test_near_edge_swap_keeps_position() -> tuple[bool, str]:
    """近边缘搭接（切向重叠 1e-5 m）：交换 A/B 后即使在回退路径上，也不能偏到角上。

    旧回退自己查两侧支撑点再取中点。平面上有一整片等价支撑点时，两次查询可能
    挑到相距很远的两个点：实测交换后给出 ``y = −0.250 m`` 的单点，而真实接触在
    ``y ≈ 0``——0.25 m 的力臂足以改变冲量分布。现在回退用 ``distance()`` 留下的
    成对见证点，两者本来就重合在接触位置上。
    """
    cylinder = RigidBody(
        name="cylinder", geometry=CylinderGeometry(1.0, 1.0), mass=1.0
    )
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((1.0, 1.0, 1.0)),
        mass=1.0,
        position=(1.49999, 0.0, 0.9),
    )
    forward = gjk.collide(cylinder, box)
    patch.reset_fallback_count()
    backward = gjk.collide(box, cylinder)
    reasons = patch.fallback_reasons()

    problems = []
    for label, points in (("正向", forward), ("反向", backward)):
        if not points:
            problems.append(f"{label}没有接触点")
            continue
        for point in points:
            if abs(point.position[0] - 1.0) > 1e-3:
                problems.append(f"{label}：x={point.position[0]:.5f}，应在 1.0 附近")
            if abs(point.position[1]) > 0.02:
                problems.append(f"{label}：y={point.position[1]:.5f}，偏离接触区")
    unknown = set(reasons) - {"disjoint", "missing_patch", "separated"}
    if unknown:
        problems.append(f"回退原因未分类：{unknown}")

    return not problems, ("；".join(problems) if problems else
                          f"反向 {len(backward)} 点、回退 {reasons or '0 次'}，"
                          f"位置都在 x≈1.0、|y|≤0.02")


def test_tilted_generatrix_per_point_depth() -> tuple[bool, str]:
    """倾斜母线的深度必须按**端点插值**算，不能压成平均平面。

    圆柱绕 Y 轴转 2° 侧抵盒面（x = 0.295 m）时，侧面选取把整条母线交了下来，
    它的两个端点沿 X 方向一个是分离 12.6 mm、一个是嵌入 22.3 mm。旧实现用"过顶点
    均值的垂直平面"，两点都报 4.817 mm——分离的那端凭空多了约束，嵌入的那端少报
    了 17 mm。现在按切向投影参数插值三维端点，间隙沿母线线性变化。
    """
    radius, height = 0.3, 1.0
    angle = math.radians(2.0)
    orientation = quat.quat_from_axis_angle((0.0, 1.0, 0.0), angle)
    cylinder = RigidBody(
        name="cylinder",
        geometry=CylinderGeometry(radius, height),
        mass=1.0,
        position=(0.0, 0.0, 0.0),
        orientation=orientation,
    )
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((1.0, 1.0, 1.0)),
        mass=1.0,
        position=(0.795, 0.0, 0.0),     # 左表面在 x = 0.295
    )
    wall_x = 0.295
    contacts = gjk.collide(cylinder, box)

    axis = quat.quat_rotate_vector(orientation, (0.0, 0.0, 1.0))
    radial = quat.quat_rotate_vector(orientation, (1.0, 0.0, 0.0))

    def surface_x(parameter: float) -> float:
        """母线上参数 s ∈ [-1, 1] 处的世界 x 坐标（解析，不用 patch 的中间量）。"""
        return (
            cylinder.position[0]
            + parameter * 0.5 * height * axis[0]
            + radius * radial[0]
        )

    def parameter_of(point) -> float:
        """把接触点还原到母线上：位置是两锚点中点，沿 +法向退回 A 侧表面。

        ``position = (a + b)/2``、``penetration = (a − b)·n``，所以
        ``position + n·penetration/2 = a``（圆柱这一侧的锚点）。
        """
        surface = [
            point.position[i] + 0.5 * point.penetration * point.normal[i]
            for i in range(3)
        ]
        return sum(
            (surface[i] - cylinder.position[i]) * axis[i] for i in range(3)
        ) / (0.5 * height)

    problems = []
    if not contacts:
        return False, "没有产生接触点"
    for point in contacts:
        expected = surface_x(parameter_of(point)) - wall_x
        if abs(point.penetration - expected) > 1e-6:
            problems.append(
                f"z={point.position[2]:.4f}：报 {point.penetration * 1e3:.3f} mm，"
                f"真实 {expected * 1e3:.3f} mm"
            )
        if expected < -1e-9:
            problems.append(f"z={point.position[2]:.4f} 处已分离，不该建约束")
    deepest = max(surface_x(parameter_of(point)) - wall_x for point in contacts)
    if abs(deepest - 0.022267) > 5e-5:
        problems.append(f"最深点 {deepest * 1e3:.3f} mm，应约 22.27 mm")

    return not problems, ("；".join(problems) if problems else
                          f"{len(contacts)} 点，深度 {deepest * 1e3:.3f} mm 以内变化，"
                          f"分离端已被裁掉")


def test_segment_clip_has_no_duplicate_anchor() -> tuple[bool, str]:
    """两点 patch 跨过零间隙：结果必须恰好是"保留端点 + 一个边界交点"。

    把两点输入当成闭合二边形遍历时，正向与反向是同一条边，交点会被插入两次——
    实测返回三个锚点、其中两个位置重合。重合的锚点会让求解器解两遍同一约束。
    """
    segment = SupportPatch(
        vertices=[(0.0, -0.5, 0.52), (0.0, 0.5, 0.48)],
        kind=PATCH_SEGMENT,
        normal=(0.0, 0.0, 1.0),
    )
    face = SupportPatch(
        vertices=[
            (-0.6, -0.6, 0.5),
            (0.6, -0.6, 0.5),
            (0.6, 0.6, 0.5),
            (-0.6, 0.6, 0.5),
        ],
        kind=PATCH_POLYGON,
        normal=(0.0, 0.0, -1.0),
    )
    normal = (0.0, 0.0, 1.0)
    overlap = patch.overlap(segment, face, normal)
    anchors = patch.contact_anchors(overlap, segment, face, normal)

    problems = []
    if len(anchors) != 2:
        problems.append(f"{len(anchors)} 个锚点，应为 2（一个端点 + 一个边界交点）")
    else:
        first, second = anchors
        if math.dist(first.position, second.position) < 1e-6:
            problems.append("两个锚点位置重合")
        depths = sorted(round(anchor.penetration, 9) for anchor in anchors)
        if abs(depths[0]) > 1e-9:
            problems.append(f"边界交点的深度应为 0，得到 {depths[0]}")
        if abs(depths[1] - 0.02) > 1e-9:
            problems.append(f"保留端点的深度应为 0.02，得到 {depths[1]}")

    return not problems, ("；".join(problems) if problems else
                          "2 个锚点，深度 0 / 20 mm，无重合")


CHECKS = [
    ("1. P0-1 场景的交集落在搭接带内", test_overlap_in_p01_scene),
    ("2. 退化类别以投影结果为准", test_projected_kind_wins),
    ("3. 交换 A/B 得到同一组位置", test_swap_matches),
    ("4. 稀疏：确定性 + 铺开 + 不补点", test_reduce_is_deterministic_and_spread),
    ("5. 空交集的原因可区分", test_empty_overlap_reasons),
    ("6. 标签一对一分配", test_match_tags_unique),
    ("7. 接触路径的边数预算", test_contact_segment_budget),
    ("8. 倾斜端面逐点深度、高侧不建约束", test_tilted_cap_per_point_depth),
    ("9. 近边缘交换后回退不偏力臂", test_near_edge_swap_keeps_position),
    ("10. 倾斜母线按端点插值算深度", test_tilted_generatrix_per_point_depth),
    ("11. 两点裁剪不产生重合锚点", test_segment_clip_has_no_duplicate_anchor),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("支撑 patch 语义层（投影 / 求交 / 稀疏 / 锚点 / 身份）")
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
