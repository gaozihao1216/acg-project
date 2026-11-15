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
    if len(result.points) > patch.MAX_CONTACT_POINTS:
        problems.append(f"{len(result.points)} 点，超过上限 {patch.MAX_CONTACT_POINTS}")
    if result.empty_reason is not None:
        problems.append(f"非空却带着 empty_reason={result.empty_reason}")

    anchors = patch.lift_anchors(result, patch_a, patch_b, NORMAL)
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
    anchors = patch.lift_anchors(result, line_like, patch_b, NORMAL)
    if len(anchors) != 2:
        problems.append(f"{len(anchors)} 个锚点")
    elif max(abs(anchor.penetration - 0.5) for anchor in anchors) > 1e-9:
        problems.append(f"深度={[round(a.penetration, 6) for a in anchors]}，应为 0.5")

    return not problems, ("；".join(problems) if problems else
                          f"退化成 {len(result.points)} 点线段，深度 0.5")


def test_swap_matches() -> tuple[bool, str]:
    """交换 A/B：交集点集一致，锚点位置一致（法向由调用方取反）。"""
    cylinder, box = _scene()
    patch_a = patch.contact_patch_world(cylinder, NORMAL)
    patch_b = patch.contact_patch_world(box, _OPPOSITE)
    forward = patch.lift_anchors(patch.overlap(patch_a, patch_b, NORMAL), patch_a, patch_b, NORMAL)
    backward = patch.lift_anchors(
        patch.overlap(patch_b, patch_a, NORMAL), patch_b, patch_a, NORMAL
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
    if patch.lift_anchors(disjoint, patch_a, patch_b, NORMAL):
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


CHECKS = [
    ("1. P0-1 场景的交集落在搭接带内", test_overlap_in_p01_scene),
    ("2. 退化类别以投影结果为准", test_projected_kind_wins),
    ("3. 交换 A/B 得到同一组位置", test_swap_matches),
    ("4. 稀疏：确定性 + 铺开 + 不补点", test_reduce_is_deterministic_and_spread),
    ("5. 空交集的原因可区分", test_empty_overlap_reasons),
    ("6. 标签一对一分配", test_match_tags_unique),
    ("7. 接触路径的边数预算", test_contact_segment_budget),
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
