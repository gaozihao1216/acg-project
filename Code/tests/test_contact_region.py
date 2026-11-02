# -*- coding: utf-8 -*-
"""P0-1：接触点必须落在两侧接触区域的**切向交集**内。

场景（审阅文档里的偏置柱-盒搭接）
--------------------------------
圆柱半径 1、高 1，轴竖直，中心在原点 → 顶面是 ``z = 0.5`` 的圆盘 ``x²+y² ≤ 1``。
盒子边长 1，中心 ``(1.3, 0, 0.9)`` → 底面 ``z = 0.4``，投影区域
``x ∈ [0.8, 1.8]``、``y ∈ [-0.5, 0.5]``。两者沿 z 重叠 0.1 m，于是**真实的**切向
接触区是两者的交集：

    矩形 ∩ 圆盘 = { 0.8 ≤ x ≤ 1.0,  |y| ≤ min(0.5, √(1 − x²)) }

当前 ``gjk._build_manifold`` 只在法向上判深度（"最近采样点配对 + 沿法向投影"），
完全不看切向覆盖：四个点来自圆柱端面 45°/135°/225°/315° 的采样，x 全是 ±0.707
——两个在盒子 1.5 m 之外，两个在盒子投影的左侧。它们被当成接触点，力臂完全错误。

为什么这比"多点流形不准"更严重：四个点关于圆柱轴**对称**，净力矩为零，于是
症状不是"翻倒"，而是**该倒的不倒**——靠边搭接时物体被四个不存在的支撑点托住。

本文件是 P0-1 的验收用例
------------------------
修复前它们预期失败（见 ``KNOWN_FAILING``），所以 ``run_all`` 不会因此变红；
修好后请从集合里删掉对应条目，它们会转成正常 PASS。若某项在标记期间意外通过，
``main`` 会报 ``XPASS`` 并计入失败，提醒移除标记。

验收不要求"恰好 4 点"：只要**非空、不超过 4 点、且覆盖了交叠区**即可。
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
from src.physics.body import RigidBody
from src.physics.narrowphase import _collide_points

# -- 场景常量（改动前请先确认所有断言仍然成立）-----------------------------
CYLINDER_RADIUS = 1.0
BOX_HALF = 0.5
BOX_CENTER = (1.3, 0.0, 0.9)

BOX_MIN_X = BOX_CENTER[0] - BOX_HALF      # 0.8
BOX_MAX_X = BOX_CENTER[0] + BOX_HALF      # 1.8
BOX_MIN_Y = BOX_CENTER[1] - BOX_HALF      # -0.5
BOX_MAX_Y = BOX_CENTER[1] + BOX_HALF      # 0.5
OVERLAP_MAX_X = CYLINDER_RADIUS           # 1.0：圆盘右边界

_TOL = 1e-6


def _scene():
    cylinder = RigidBody(
        name="cylinder",
        geometry=CylinderGeometry(CYLINDER_RADIUS, 1.0),
        mass=1.0,
        position=(0.0, 0.0, 0.0),
    )
    box = RigidBody(
        name="box",
        geometry=BoxGeometry((2 * BOX_HALF, 2 * BOX_HALF, 2 * BOX_HALF)),
        mass=1.0,
        position=BOX_CENTER,
    )
    return cylinder, box


def _points():
    cylinder, box = _scene()
    return _collide_points(cylinder, box)


def _tangent(point) -> tuple[float, float]:
    """取切向 (x, y)：这个场景的法向是 ±z，所以切平面就是 xy。"""
    return (point.position[0], point.position[1])


# ---------------------------------------------------------------------------
# 1. 点必须在盒子底面的投影区域内
# ---------------------------------------------------------------------------

def test_points_inside_box_face() -> tuple[bool, str]:
    points = _points()
    if not points:
        return False, "没有产生任何接触点"

    outside = [
        (round(x, 4), round(y, 4))
        for point in points
        for x, y in [_tangent(point)]
        if not (BOX_MIN_X - _TOL <= x <= BOX_MAX_X + _TOL
                and BOX_MIN_Y - _TOL <= y <= BOX_MAX_Y + _TOL)
    ]
    ok = not outside
    return ok, (
        f"{len(points)} 点，x∈[{min(_tangent(p)[0] for p in points):.4f}, "
        f"{max(_tangent(p)[0] for p in points):.4f}]；"
        f"落在盒子投影 x∈[{BOX_MIN_X}, {BOX_MAX_X}]、y∈[{BOX_MIN_Y}, {BOX_MAX_Y}] 之外的点={outside}"
    )


# ---------------------------------------------------------------------------
# 2. 点必须在圆柱端面圆盘内
# ---------------------------------------------------------------------------

def test_points_inside_disc() -> tuple[bool, str]:
    points = _points()
    if not points:
        return False, "没有产生任何接触点"

    outside = []
    worst_radius = 0.0
    for point in points:
        x, y = _tangent(point)
        radius = math.hypot(x, y)
        worst_radius = max(worst_radius, radius)
        if radius > CYLINDER_RADIUS + _TOL:
            outside.append((round(x, 4), round(y, 4), round(radius, 4)))
    ok = not outside
    return ok, (
        f"{len(points)} 点，最大半径={worst_radius:.4f}"
        f"（圆盘半径 {CYLINDER_RADIUS}）；越界点={outside}"
    )


# ---------------------------------------------------------------------------
# 3. 点必须落在两侧区域的**交集**里
# ---------------------------------------------------------------------------

def test_points_inside_tangential_overlap() -> tuple[bool, str]:
    """只看"在盒子投影内"是不够的：交叠区的上界是圆盘，右界才是盒子。

    这个场景里交叠区是 ``x ∈ [0.8, 1.0]``，不是整个盒子投影 ``[0.8, 1.8]``。
    """
    points = _points()
    if not points:
        return False, "没有产生任何接触点"

    outside = []
    for point in points:
        x, y = _tangent(point)
        in_box = BOX_MIN_X - _TOL <= x <= BOX_MAX_X + _TOL and abs(y) <= BOX_MAX_Y + _TOL
        in_disc = math.hypot(x, y) <= CYLINDER_RADIUS + _TOL
        in_cap = in_box and in_disc and x <= OVERLAP_MAX_X + _TOL
        if not in_cap:
            outside.append((round(x, 4), round(y, 4)))

    ok = not outside
    return ok, (
        f"{len(points)} 点；落在交叠区 0.8≤x≤1.0、|y|≤min(0.5,√(1−x²)) 之外的点={outside}"
    )


# ---------------------------------------------------------------------------
# 4. 交换 A/B：位置集合基本相同，法向相反
# ---------------------------------------------------------------------------

def test_swapped_pair_is_symmetric() -> tuple[bool, str]:
    cylinder, box = _scene()
    forward = _collide_points(cylinder, box)
    backward = _collide_points(box, cylinder)
    if not forward or not backward:
        return False, f"正向 {len(forward)} 点 / 反向 {len(backward)} 点"

    worst_distance = 0.0
    for point in forward:
        nearest = min(
            backward,
            key=lambda other: math.dist(tuple(point.position), tuple(other.position)),
        )
        worst_distance = max(worst_distance, math.dist(tuple(point.position), tuple(nearest.position)))

    worst_dot = min(
        sum(a * b for a, b in zip(forward[0].normal, backward[0].normal)), 0.0
    )
    ok = len(forward) == len(backward) and worst_distance < 1e-6 and worst_dot < -1.0 + 1e-6
    return ok, (
        f"正向 {len(forward)} 点 / 反向 {len(backward)} 点；"
        f"按位置配对的最大距离={worst_distance:.3e} m；法向点积={worst_dot:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. 点数与覆盖：非空、≤4、且铺开在交叠区上
# ---------------------------------------------------------------------------

def test_manifold_covers_overlap() -> tuple[bool, str]:
    """不要求恰好 4 点，但要求"铺开"——不能全挤在交叠区外或一个角落。"""
    points = _points()
    if not points:
        return False, "没有产生任何接触点"
    if len(points) > 4:
        return False, f"{len(points)} 点，超过上限 4"

    xs = [_tangent(point)[0] for point in points]
    ys = [_tangent(point)[1] for point in points]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    inside = [
        point for point in points
        if BOX_MIN_X - _TOL <= _tangent(point)[0] <= OVERLAP_MAX_X + _TOL
        and abs(_tangent(point)[1]) <= BOX_MAX_Y + _TOL
    ]
    ok = len(inside) >= 1 and span_x > 0.05 and span_y > 0.05
    return ok, (
        f"{len(points)} 点，交叠区内的点={len(inside)}；"
        f"x 跨度={span_x:.4f}、y 跨度={span_y:.4f}（应铺开在 0.8≤x≤1.0 上）"
    )


CHECKS = [
    ("1. 点在盒子底面投影内", test_points_inside_box_face),
    ("2. 点在圆柱端面圆盘内", test_points_inside_disc),
    ("3. 点在两侧区域的切向交集内", test_points_inside_tangential_overlap),
    ("4. 交换 A/B 后位置对称、法向相反", test_swapped_pair_is_symmetric),
    ("5. 点数非空且铺开在交叠区上", test_manifold_covers_overlap),
]

#: P0-1 修复前**预期失败**的条目。修好后请从这里删掉，它们会转成正常 PASS。
KNOWN_FAILING = {
    "1. 点在盒子底面投影内",
    "3. 点在两侧区域的切向交集内",
    "4. 交换 A/B 后位置对称、法向相反",
    "5. 点数非空且铺开在交叠区上",
}


def main() -> int:
    print("=" * 88)
    print("P0-1：接触点必须落在两侧接触区域的切向交集内")
    print("=" * 88)

    failures = 0
    xfailed = 0
    for title, case in CHECKS:
        try:
            ok, detail = case()
        except Exception as exc:  # 单个用例崩溃不应中断整轮
            ok, detail = False, f"异常：{type(exc).__name__}: {exc}"

        if title in KNOWN_FAILING:
            if ok:
                print(f"[XPASS] {title}")
                print("        已通过——请从 KNOWN_FAILING 中移除本项")
                failures += 1
            else:
                print(f"[XFAIL] {title}（P0-1 未修复，预期失败）")
                print(f"        {detail}")
                xfailed += 1
            continue

        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {detail}")
        if not ok:
            failures += 1

    print("-" * 88)
    print(
        f"结果：{len(CHECKS) - failures - xfailed}/{len(CHECKS) - len(KNOWN_FAILING)} 通过，"
        f"{xfailed} 项待修（P0-1）"
    )
    print("=" * 88)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
