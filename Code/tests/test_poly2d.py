# -*- coding: utf-8 -*-
"""二维几何工具（``mathutils.poly2d``）的契约测试。

这些函数是 P0-1 接触区域求交的地基，而它们出错的方式都很**安静**：

- 切向基没归一化 → 交集"看起来扁了一点"（``|v| = |n|``，y 方向整体被拉伸）；
- 凸包没去重 → 多边形多了个零长度的边，后面的环序、面积全跟着偏；
- 点包含在边界上判反 → 贴着边缘的接触点偶尔消失，表现为"偶尔穿透一下"。

没有一条会抛异常，所以这里逐条钉住数值，不测"大致正确"。

运行::

    python Code/tests/test_poly2d.py
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.mathutils import poly2d


def _length(vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(vector):
    length = _length(vector)
    return tuple(component / length for component in vector)


def _max_diff(a, b) -> float:
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def test_tangent_basis_unit_and_orthogonal() -> tuple[bool, str]:
    """切向基必须是单位正交的，**非单位法向也一样**。

    回归点：``v`` 曾直接取未归一化的叉积分量，于是 ``normal=(0,0,2)`` 得到
    ``|v| = 4``——投影到切平面时 y 被拉伸 4 倍，交集形状没有意义。
    """
    problems = []
    worst = 0.0
    for normal in (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (0.0, 1e-3, 1.0),
        (-3.0, 0.5, -0.25),
        (0.0, 0.0, 2.0),
        (5.0, -7.0, 11.0),
    ):
        u, v = poly2d.tangent_basis(normal)
        unit_normal = _unit(normal)
        worst = max(
            worst,
            abs(_length(u) - 1.0),
            abs(_length(v) - 1.0),
            abs(_dot(unit_normal, u)),
            abs(_dot(unit_normal, v)),
            abs(_dot(u, v)),
        )
        if abs(_length(u) - 1.0) > 1e-12 or abs(_length(v) - 1.0) > 1e-12:
            problems.append(f"{normal}：|u|={_length(u):.9g} |v|={_length(v):.9g}")
        elif abs(_dot(unit_normal, u)) > 1e-12 or abs(_dot(unit_normal, v)) > 1e-12:
            problems.append(f"{normal}：切向不与法向正交")
        elif abs(_dot(u, v)) > 1e-12:
            problems.append(f"{normal}：u 与 v 不正交")

    return not problems, ("；".join(problems) if problems else
                          f"8 个法向（含未归一化）最大偏差 {worst:.2e}")


def test_tangent_basis_sign_stable() -> tuple[bool, str]:
    """``n`` 与 ``-n`` 必须给出**逐位相同**的切向基。

    这是 A/B 对称的前提：两侧投影到不同的平面里，交集就没有可比性。这里要求
    逐位相同而不只是"张成同一个平面"——交换两侧后连切向顺序都不能变，否则
    "交换 A/B 得到同一组点"只能靠排序去补救。
    """
    problems = []
    for normal in (
        (0.0, 0.0, 1.0),
        (0.3, -0.4, 0.5),
        (7.0, 0.0, -2.0),
        (-1.0, 1e-8, 0.0),
    ):
        u, v = poly2d.tangent_basis(normal)
        flipped_u, flipped_v = poly2d.tangent_basis(tuple(-c for c in normal))
        if _max_diff(u, flipped_u) > 1e-15 or _max_diff(v, flipped_v) > 1e-15:
            problems.append(
                f"{normal}：n 与 -n 的切向不同 "
                f"(Δu={_max_diff(u, flipped_u):.3e}, Δv={_max_diff(v, flipped_v):.3e})"
            )
    return not problems, ("；".join(problems) if problems else "4 个法向双向一致")


def test_tangent_basis_rejects_zero() -> tuple[bool, str]:
    """零法向没有切平面：必须显式拒绝，而不是除零。"""
    problems = []
    for normal in ((0.0, 0.0, 0.0), (1e-15, 0.0, 0.0), (0.0, -1e-14, 0.0)):
        try:
            poly2d.tangent_basis(normal)
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{normal}：抛出 {type(exc).__name__} 而不是 ValueError")
        else:
            problems.append(f"{normal}：没有拒绝")
    return not problems, ("；".join(problems) if problems else "3 个退化法向均被 ValueError 拒绝")


def test_hull_degenerate() -> tuple[bool, str]:
    """空、单点、重合点、共线多点：不许返回零面积"多边形"。

    回归点：早先 ``len(points) <= 2`` 直接原样返回，两个重合点会给出 ``[0, 1]``，
    于是一个点被当成一条零长度线段。
    """
    cases = [
        ([], 0),
        ([(2.0, 3.0)], 1),
        ([(2.0, 3.0), (2.0, 3.0)], 1),
        ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], 2),
        ([(0.0, 0.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 2.0)], 2),
    ]
    problems = []
    notes = []
    for points, expected in cases:
        hull = poly2d.convex_hull_2d(points)
        notes.append(f"{len(points)} 点 → {len(hull)}")
        if len(hull) != expected:
            problems.append(f"{points}：期望 {expected} 个下标，得到 {hull}")
        elif len(set(hull)) != len(hull):
            problems.append(f"{points}：下标重复 {hull}")
        elif any(not 0 <= index < len(points) for index in hull):
            problems.append(f"{points}：下标越界 {hull}")
    return not problems, ("；".join(problems) if problems else "，".join(notes))


def test_hull_orders_and_drops_collinear() -> tuple[bool, str]:
    """方块 + 边中点 + 内部点：只留 4 个角，且逆时针。"""
    points = [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),   # 四个角
        (0.5, 0.0), (1.0, 0.5), (0.5, 0.5),               # 边中点 + 内部点
    ]
    hull = poly2d.convex_hull_2d(points)
    polygon = [points[index] for index in hull]
    area = poly2d.polygon_area_2d(polygon)
    problems = []
    if set(hull) != {0, 1, 2, 3}:
        problems.append(f"凸包下标={hull}，应为四个角 {{0,1,2,3}}")
    if area <= 0.0:
        problems.append(f"有向面积={area}，应为正（逆时针）")
    if abs(abs(area) - 1.0) > 1e-12:
        problems.append(f"面积={area}，应为 1")
    return not problems, ("；".join(problems) if problems else
                          f"下标={hull}，面积={area:.6g}（逆时针）")


def test_point_in_convex() -> tuple[bool, str]:
    """点包含：角点、边中点、内部判真；略外侧判假。"""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    truth = [
        ((0.0, 0.0), True, "角点"),
        ((1.0, 1.0), True, "对角角点"),
        ((0.5, 0.0), True, "边中点"),
        ((0.5, 0.5), True, "内部"),
        ((0.5, 1.5), False, "上方外侧"),
        ((-1e-6, 0.5), False, "左侧外侧（超过 eps）"),
    ]
    problems = []
    for point, expected, label in truth:
        got = poly2d.point_in_convex_2d(point, square)
        if got != expected:
            problems.append(f"{label} {point}：得到 {got}，应为 {expected}")
    return not problems, ("；".join(problems) if problems else
                          f"{len(truth)} 个点位判定正确")


def test_segment_intersection_cases() -> tuple[bool, str]:
    """严格交叉给出交点；共边、端点相碰、平行、不相交都返回 ``None``。

    端点/共边**必须**返回 ``None``：那些点由"顶点包含"路径收录，来源更清楚。
    若这里也返回一个点，同一个接触点会被两条路径各收一次。
    """
    cases = [
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.5, 0.5), "十字交叉"),
        ((0.0, 0.0), (1.0, 0.0), (0.5, -0.5), (0.5, 0.5), (0.5, 0.0), "T 形交叉"),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0), None, "共端点"),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0), None, "完全共线"),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), None, "平行"),
        ((0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (2.0, 1.0), None, "不相交"),
    ]
    problems = []
    for a1, a2, b1, b2, expected, label in cases:
        got = poly2d.segment_intersection(a1, a2, b1, b2)
        if expected is None:
            if got is not None:
                problems.append(f"{label}：得到 {got}，应为 None")
        elif got is None or _max_diff(got, expected) > 1e-12:
            problems.append(f"{label}：得到 {got}，应为 {expected}")
    return not problems, ("；".join(problems) if problems else f"{len(cases)} 个相交情形正确")


def test_project_and_area() -> tuple[bool, str]:
    """投影保持距离与有向面积；法向取反后投影逐位不变。"""
    normal = (0.0, 0.0, 3.0)          # 未归一化，专门覆盖"忘了归一化"的写法
    u, v = poly2d.tangent_basis(normal)
    origin = (10.0, -2.0, 4.0)        # 远离原点：覆盖大坐标下的相减
    points = [
        (10.0, -2.0, 4.0),
        (11.0, -2.0, 4.0),
        (11.0, -1.0, 4.0),
        (10.0, -1.0, 4.0),
    ]
    flat = poly2d.project(points, u, v, origin)
    area = poly2d.polygon_area_2d(flat)
    flipped_u, flipped_v = poly2d.tangent_basis((0.0, 0.0, -1.0))
    flipped = poly2d.project(points, flipped_u, flipped_v, origin)

    problems = []
    # 边在切平面内，长度必须原样保留：沿 x 的边 1、对角线 √2
    for index, expected in ((1, 1.0), (2, math.sqrt(2.0))):
        got = _length(flat[index])
        if abs(got - expected) > 1e-12:
            problems.append(f"第 {index} 点投影后距原点 {got:.9g}，应为 {expected:.9g}")
    if abs(abs(area) - 1.0) > 1e-12:
        problems.append(f"面积={area}，应为 ±1")
    if _max_diff(flat[1], flipped[1]) > 1e-15:
        problems.append(f"n 与 -n 的投影不同：{flat[1]} vs {flipped[1]}")
    return not problems, ("；".join(problems) if problems else
                          f"面积={area:.6g}，边长=1，n/-n 投影逐位一致")


def test_swap_symmetry() -> tuple[bool, str]:
    """两侧交换后，投影点列与凸包必须完全一致。

    P0-1 的验收方式就是"交换 A/B，位置集合一致"。这里把四个点**构造在**切平面
    里（±u ±v），于是凸包是一个真正的四边形——如果点集恰好共线，这条测试会
    用一个 2 点凸包假通过。
    """
    normal = (0.6, -0.8, 0.3)
    u, v = poly2d.tangent_basis(normal)
    flipped_u, flipped_v = poly2d.tangent_basis(tuple(-c for c in normal))
    origin = (0.1, 0.2, 0.3)
    points = [
        tuple(origin[i] + sign_u * u[i] + sign_v * v[i] for i in range(3))
        for sign_u, sign_v in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
    ]
    a_flat = poly2d.project(points, u, v, origin)
    b_flat = poly2d.project(points, flipped_u, flipped_v, origin)

    problems = []
    if any(_max_diff(a, b) > 1e-15 for a, b in zip(a_flat, b_flat)):
        problems.append("两侧投影点列不同")
    a_hull = poly2d.convex_hull_2d(a_flat)
    b_hull = poly2d.convex_hull_2d(b_flat)
    if a_hull != b_hull:
        problems.append(f"两侧凸包下标不同：{a_hull} vs {b_hull}")
    if len(a_hull) != 4:
        problems.append(f"切平面内的正方形凸包应含 4 个点，得到 {a_hull}")
    area = poly2d.polygon_area_2d([a_flat[index] for index in a_hull])
    if abs(abs(area) - 4.0) > 1e-12:
        problems.append(f"面积={area}，应为 ±4（边长 2 的正方形）")
    return not problems, ("；".join(problems) if problems else
                          f"凸包下标={a_hull}，面积={area:.6g}，两侧逐位一致")


def test_dedupe_keeps_first() -> tuple[bool, str]:
    """去重保留首次出现的元素，且不改变剩余项的顺序。"""
    items = [
        ((0.0, 0.0), "a"),
        ((1.0, 0.0), "b"),
        ((0.0, 5e-10), "a'"),
        ((1.0, 0.0), "b'"),
        ((2.0, 0.0), "c"),
    ]
    kept = poly2d.dedupe(items, eps=1e-9)
    payloads = [payload for _point, payload in kept]
    ok = payloads == ["a", "b", "c"]
    return ok, f"保留={payloads}（应为 ['a', 'b', 'c']）"


CHECKS = [
    ("1. 切向基单位正交（含未归一化法向）", test_tangent_basis_unit_and_orthogonal),
    ("2. 切向基对 ±n 逐位一致", test_tangent_basis_sign_stable),
    ("3. 零法向显式拒绝", test_tangent_basis_rejects_zero),
    ("4. 退化点集的凸包", test_hull_degenerate),
    ("5. 凸包去共线且逆时针", test_hull_orders_and_drops_collinear),
    ("6. 点包含（含边界）", test_point_in_convex),
    ("7. 线段求交的六种情形", test_segment_intersection_cases),
    ("8. 投影保持长度与面积", test_project_and_area),
    ("9. A/B 交换后几何一致", test_swap_symmetry),
    ("10. 去重保留首次出现", test_dedupe_keeps_first),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("二维几何工具（切向基 / 凸包 / 包含 / 求交）")
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
