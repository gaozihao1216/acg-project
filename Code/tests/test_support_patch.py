# -*- coding: utf-8 -*-
"""支撑 patch（``support_patch_world``）与身份采样分离的契约测试。

这两条路径共用"沿方向看哪片最凸"的知识，但**用途和类型都不同**：

- :func:`feature_world_tagged` → 固定 4 点 + 固定槽位标签，供暖启动跨帧匹配；
- :func:`support_patch_world` → ``count`` 点（按半径自适应）+ 无标签，供切平面求交。

混在一起的代价在 P0-1 里已经吃过一次：4 点内接四边形比圆盘小 29%，窄搭接时
被算成"不相交"，而高密度标签又会和 ``_TAG_SIDE`` / ``_TAG_SINGLE`` 撞号。
所以这里除了几何正确性，还要钉住"高密度路径**拿不到**标签"这件事。

运行::

    python Code/tests/test_support_patch.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import fields

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import (
    BoxGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    SphereGeometry,
)
from src.mathutils import poly2d
from src.physics.body import RigidBody
from src.physics.support import (
    PATCH_POINT,
    PATCH_POLYGON,
    PATCH_SEGMENT,
    SupportPatch,
    _PATCH_MAX_SEGMENTS,
    _PATCH_MESH_ERROR,
    _ring_inradius_error,
    _segments_for_radius,
    feature_world_tagged,
    support_patch_world,
)


def _make(geometry) -> RigidBody:
    return RigidBody(name="probe", geometry=geometry, mass=1.0)


def _length(vector) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _radial(point) -> float:
    """点到局部 Z 轴的距离（物体未旋转，世界系即体坐标系）。"""
    return math.hypot(point[0], point[1])


def test_cylinder_cap_polygon() -> tuple[bool, str]:
    """圆柱端面：多边形、点数由半径与目标误差决定、内缩达标、法向朝外。"""
    radius, height = 0.3, 1.0
    body = _make(CylinderGeometry(radius, height))
    patch = support_patch_world(body, (0.0, 0.0, -1.0))

    expected_count = _segments_for_radius(radius, _PATCH_MESH_ERROR)
    problems = []
    if patch.kind != PATCH_POLYGON:
        problems.append(f"kind={patch.kind}，应为 {PATCH_POLYGON}")
    if len(patch.vertices) != expected_count:
        problems.append(f"{len(patch.vertices)} 点，应为 {expected_count}")
    if len(patch.vertices) != len({(round(p[0], 12), round(p[1], 12), round(p[2], 12))
                                   for p in patch.vertices}):
        problems.append("顶点有重复")
    if max(abs(_radial(p) - radius) for p in patch.vertices) > 1e-12:
        problems.append("顶点不在端面圆周上")
    if max(abs(p[2] + 0.5 * height) for p in patch.vertices) > 1e-12:
        problems.append("顶点不在底面上")
    if max(abs(patch.normal[i] - (0.0, 0.0, -1.0)[i]) for i in range(3)) > 1e-12:
        problems.append(f"法向={patch.normal}，应为 (0, 0, -1)")
    error = _ring_inradius_error(radius, len(patch.vertices))
    if error > _PATCH_MESH_ERROR:
        problems.append(f"内缩 {error * 1e3:.4f} mm 超过目标 {_PATCH_MESH_ERROR * 1e3} mm")

    return not problems, ("；".join(problems) if problems else
                          f"{len(patch.vertices)} 边，内缩 {error * 1e3:.4f} mm ≤ "
                          f"{_PATCH_MESH_ERROR * 1e3} mm，法向 (0, 0, -1)")


def test_frustum_both_faces() -> tuple[bool, str]:
    """圆台两端各自按自己的半径取边数：顶面更小，边数也应该更少。"""
    body = _make(FrustumGeometry(0.4, 0.15, 0.9))
    bottom = support_patch_world(body, (0.0, 0.0, -1.0))
    top = support_patch_world(body, (0.0, 0.0, 1.0))
    problems = []

    if len(bottom.vertices) != _segments_for_radius(0.4, _PATCH_MESH_ERROR):
        problems.append(f"底面 {len(bottom.vertices)} 点，应为半径 0.4 对应的边数")
    if len(top.vertices) != _segments_for_radius(0.15, _PATCH_MESH_ERROR):
        problems.append(f"顶面 {len(top.vertices)} 点，应为半径 0.15 对应的边数")
    if not len(bottom.vertices) > len(top.vertices):
        problems.append("底面半径更大，边数却没有更多")
    if max(abs(_radial(p) - 0.4) for p in bottom.vertices) > 1e-12:
        problems.append("底面顶点半径不是 0.4")
    if max(abs(_radial(p) - 0.15) for p in top.vertices) > 1e-12:
        problems.append("顶面顶点半径不是 0.15")
    if max(abs(top.normal[i] - (0.0, 0.0, 1.0)[i]) for i in range(3)) > 1e-12:
        problems.append(f"顶面法向={top.normal}，应为 (0, 0, 1)")

    return not problems, ("；".join(problems) if problems else
                          f"底面 {len(bottom.vertices)} 点 / 顶面 {len(top.vertices)} 点，"
                          f"半径各为 0.4 / 0.15")


def test_box_face_is_ordered() -> tuple[bool, str]:
    """盒的端面要按**环序**给出，否则求交会把对角线当边。

    盒顶点按 ``4*ix + 2*iy + iz`` 枚举，那个顺序在 xy 投影上是"Z 字形"，
    所以 patch 必须重新按凸包排序。
    """
    body = _make(BoxGeometry((1.0, 1.0, 1.0)))
    patch = support_patch_world(body, (0.0, 0.0, 1.0))
    problems = []

    if patch.kind != PATCH_POLYGON or len(patch.vertices) != 4:
        problems.append(f"kind={patch.kind}, 点数={len(patch.vertices)}，应为 4 点多边形")
    else:
        u, v = poly2d.tangent_basis(patch.normal)
        origin = (0.0, 0.0, 0.5)
        flat = poly2d.project(patch.vertices, u, v, origin)
        area = poly2d.polygon_area_2d(flat)
        if abs(abs(area) - 1.0) > 1e-12:
            problems.append(f"环序面积={area}，应为 ±1（若是 Z 序会是自交的 0）")
        # 相邻边必须垂直：环序的对角线连接会让点积明显不为 0
        for index in range(4):
            a = flat[(index + 1) % 4]
            b = flat[index]
            c = flat[(index - 1) % 4]
            e1 = (a[0] - b[0], a[1] - b[1])
            e2 = (c[0] - b[0], c[1] - b[1])
            if abs(e1[0] * e2[0] + e1[1] * e2[1]) > 1e-12:
                problems.append(f"第 {index} 个顶点的两条边不垂直")
                break

    return not problems, ("；".join(problems) if problems else
                          "4 点、面积 1、相邻边正交")


def test_point_and_segment_kinds() -> tuple[bool, str]:
    """严格凸体 → 单点；立着看圆柱侧面 → 线段；斜着看圆柱 → 单点。"""
    problems = []
    notes = []

    sphere = support_patch_world(_make(SphereGeometry(0.5)), (1.0, 0.0, 0.0))
    notes.append(f"球={sphere.kind}")
    if sphere.kind != PATCH_POINT or len(sphere.vertices) != 1:
        problems.append(f"球：{sphere.kind} / {len(sphere.vertices)} 点")
    if abs(sphere.vertices[0][0] - 0.5) > 1e-12:
        problems.append(f"球的支撑点={sphere.vertices[0]}，应为 (0.5, 0, 0)")

    ellipsoid = support_patch_world(_make(EllipsoidGeometry((0.5, 0.3, 0.2))), (0.0, 1.0, 0.0))
    notes.append(f"椭球={ellipsoid.kind}")
    if ellipsoid.kind != PATCH_POINT:
        problems.append(f"椭球：{ellipsoid.kind}")

    side = support_patch_world(_make(CylinderGeometry(0.3, 1.0)), (1.0, 0.0, 0.0))
    notes.append(f"柱侧={side.kind}/{len(side.vertices)}")
    if side.kind != PATCH_SEGMENT or len(side.vertices) != 2:
        problems.append(f"柱侧：{side.kind} / {len(side.vertices)} 点")
    elif max(abs(abs(p[0]) - 0.3) for p in side.vertices) > 1e-12:
        problems.append("母线端点没有落在半径 0.3 上")
    elif max(abs(abs(p[2]) - 0.5) for p in side.vertices) > 1e-12:
        problems.append("母线端点没有落在 ±h/2 上")

    tilted = support_patch_world(_make(CylinderGeometry(0.3, 1.0)), (1.0, 0.0, 1.0))
    notes.append(f"柱斜={tilted.kind}")
    if tilted.kind != PATCH_POINT:
        problems.append(f"倾斜圆柱：{tilted.kind}，斜撑时支撑集只有一个点")

    return not problems, ("；".join(problems) if problems else "，".join(notes))


def test_patch_has_no_identity() -> tuple[bool, str]:
    """高密度路径**在类型上**拿不到标签，身份路径仍是固定 4 点。"""
    names = [field.name for field in fields(SupportPatch)]
    problems = []
    if names != ["vertices", "kind", "normal"]:
        problems.append(f"SupportPatch 字段={names}，不许多出身份字段")
    if hasattr(support_patch_world(_make(CylinderGeometry(0.3, 1.0)), (0.0, 0.0, -1.0)), "tag"):
        problems.append("patch 上有 tag 属性")

    body = _make(CylinderGeometry(0.3, 1.0))
    bottom = feature_world_tagged(body, (0.0, 0.0, -1.0))
    top = feature_world_tagged(body, (0.0, 0.0, 1.0))
    if [tag for _point, tag in bottom] != [0, 1, 2, 3]:
        problems.append(f"底面槽位={[tag for _p, tag in bottom]}，应为 0..3")
    if [tag for _point, tag in top] != [4, 5, 6, 7]:
        problems.append(f"顶面槽位={[tag for _p, tag in top]}，应为 4..7")

    return not problems, ("；".join(problems) if problems else
                          "patch 无身份字段；身份路径底面 0..3、顶面 4..7")


def test_mesh_error_takes_effect() -> tuple[bool, str]:
    """``mesh_error`` 必须是**实参**：不同误差要给出不同边数。

    回归点：早先函数体里写死 ``_PATCH_MESH_ERROR``，传 1e-4 和 1e-3 结果一样，
    参数退化成了布尔开关。
    """
    radius = 1.0
    coarse = _segments_for_radius(radius, 1e-3)
    default = _segments_for_radius(radius, _PATCH_MESH_ERROR)
    fine = _segments_for_radius(radius, 1e-4)
    problems = []
    if not coarse < default < fine:
        problems.append(f"边数未随误差单调变化：1e-3→{coarse}, 5e-4→{default}, 1e-4→{fine}")

    for bad in (0.0, -1e-6, float("nan")):
        try:
            _segments_for_radius(radius, bad)
        except ValueError:
            continue
        problems.append(f"mesh_error={bad}：没有拒绝")
    for bad in (-0.1, -1e-3):
        try:
            _segments_for_radius(bad, _PATCH_MESH_ERROR)
        except ValueError:
            continue
        problems.append(f"radius={bad}：没有拒绝")
    if _segments_for_radius(0.0, _PATCH_MESH_ERROR) < 3:
        problems.append("退化端面的边数小于 3")

    return not problems, ("；".join(problems) if problems else
                          f"1e-3→{coarse}, 5e-4→{default}, 1e-4→{fine}；非法值均被拒绝")


def test_cap_is_documented_boundary() -> tuple[bool, str]:
    """边数上限会打破误差承诺——这是**已知边界**，必须显式记录。

    半径 6.6 m 以上要满足 0.5 mm 内缩需要超过 :data:`_PATCH_MAX_SEGMENTS` 条边，
    触顶后实际内缩会大于目标。测试钉住"触顶确实超标"这件事本身：如果哪天有人
    把上限调高到够用，这条会失败，提醒同步更新注释与文档。
    """
    radius = 20.0
    count = _segments_for_radius(radius, _PATCH_MESH_ERROR)
    error = _ring_inradius_error(radius, count)
    if count != _PATCH_MAX_SEGMENTS:
        return False, f"r={radius} 的边数={count}，未按预期触顶 {_PATCH_MAX_SEGMENTS}"
    if not error > _PATCH_MESH_ERROR:
        return False, f"r={radius} 触顶后内缩 {error * 1e3:.4f} mm 仍未超标，边界描述已过时"
    return True, (f"r={radius} 触顶 {count} 边，实际内缩 {error * 1e3:.4f} mm > "
                  f"目标 {_PATCH_MESH_ERROR * 1e3} mm（已知边界）")


def test_zero_direction() -> tuple[bool, str]:
    """零方向：单点 + 零法向，不抛异常也不猜法向。"""
    patch = support_patch_world(_make(CylinderGeometry(0.3, 1.0)), (0.0, 0.0, 0.0))
    ok = (patch.kind == PATCH_POINT and len(patch.vertices) == 1
          and patch.normal == (0.0, 0.0, 0.0))
    return ok, f"kind={patch.kind}，点数={len(patch.vertices)}，法向={patch.normal}"


CHECKS = [
    ("1. 圆柱端面：多边形 + 内缩达标", test_cylinder_cap_polygon),
    ("2. 圆台两端各按自身半径取边数", test_frustum_both_faces),
    ("3. 盒端面按环序给出", test_box_face_is_ordered),
    ("4. 单点 / 线段分类", test_point_and_segment_kinds),
    ("5. 高密度 patch 不带身份", test_patch_has_no_identity),
    ("6. mesh_error 是实参", test_mesh_error_takes_effect),
    ("7. 边数上限是已知边界", test_cap_is_documented_boundary),
    ("8. 零方向退化", test_zero_direction),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("支撑 patch（求交几何）与身份采样的分离")
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
