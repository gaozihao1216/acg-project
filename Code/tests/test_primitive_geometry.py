# -*- coding: utf-8 -*-
"""参数化图元测试：体积 / 惯量 / 支撑映射 / 包含测试。

验证策略
--------
这里**不**用"再写一遍同样的公式然后比对"，那只能证明两处抄得一样。用的是三类
独立证据：

1. **退化情形**。圆台取 ``R₁ = R₂`` 必须逐位等于圆柱，取 ``R₂ = 0`` 必须等于
   圆锥；椭球取 ``a = b = c`` 必须等于球；球冠取 ``h = R`` 是半球、``h = 2R``
   是整球。这些等式把"新写的积分器"和"早已验证过的闭式解"钉在一起。
2. **常识公式**。圆锥的 ``3/10 m r²``、``83/320 m R²``（半球绕直径）都是独立
   来源的已知结果。
3. **暴力采样下界**。支撑映射声称"沿方向最远"，就在包围球里随机撒点，取体内
   点里最大投影作为下界，验证解析支撑点不劣于它。这条能抓出方向分支写错
   （比如圆锥该取底面圆周却取了顶点）。

惯量的容差用相对 1e-9：闭式积分之间的一致性应该是浮点级的，若某项只有 1e-3
量级的一致，说明公式本身就不对，不该被容差盖住。
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

from src.geom.geometry import (
    BoxGeometry,
    ConeGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    GeometrySpec,
    SphereGeometry,
    SphericalCapGeometry,
)
from src.geom.inertia import compute_inertia

MASS = 2.0


def _close(got: float, want: float, tol: float = 1e-9) -> bool:
    return abs(got - want) <= tol * max(1.0, abs(want))


# ---------------------------------------------------------------------------
# 1-2. 圆锥
# ---------------------------------------------------------------------------

def test_cone_centroid_offset() -> tuple[bool, str]:
    """圆锥的质心偏移：底面在 -h/4、顶点在 +3h/4。

    质心距底面 h/4、距顶点 3h/4，所以把质心挪到原点后，顶点自然落在 +3h/4。
    这个换算极易写反，写反后支撑映射、包围球、contains 会一起错，但惯量仍然
    是对的——正是那种"跑起来只是有点怪"的错。
    """
    radius, height = 0.3, 1.2
    cone = ConeGeometry(radius, height)
    ok = (
        _close(cone._base_z, -0.25 * height)
        and _close(cone._apex_z, 0.75 * height)
        and _close(cone.volume(), math.pi * radius ** 2 * height / 3.0)
    )
    return ok, (
        f"底面 z={cone._base_z:.6f}（应 {-0.25 * height:.6f}），"
        f"顶点 z={cone._apex_z:.6f}（应 {0.75 * height:.6f}），"
        f"体积={cone.volume():.6f}"
    )


def test_cone_inertia() -> tuple[bool, str]:
    """圆锥主惯量对上常识公式 ``Izz = 3/10 m r²``、``Ixx = m(3r²/20 + 3h²/80)``。"""
    radius, height = 0.3, 1.2
    ixx, iyy, izz = compute_inertia(ConeGeometry(radius, height), MASS)
    want_izz = 0.3 * MASS * radius ** 2
    want_ixx = MASS * (3.0 * radius ** 2 / 20.0 + 3.0 * height ** 2 / 80.0)
    ok = _close(izz, want_izz) and _close(ixx, want_ixx) and _close(ixx, iyy)
    return ok, (
        f"Ixx={ixx:.8f}（应 {want_ixx:.8f}），Izz={izz:.8f}（应 {want_izz:.8f}），"
        f"Ixx==Iyy={_close(ixx, iyy)}"
    )


# ---------------------------------------------------------------------------
# 3-4. 圆台的两种退化
# ---------------------------------------------------------------------------

def test_frustum_degenerates_to_cylinder() -> tuple[bool, str]:
    """``R₁ = R₂`` 时圆台必须**逐位**等于圆柱（体积与三个惯量）。"""
    radius, height = 0.3, 1.2
    frustum = FrustumGeometry(radius, radius, height)
    cylinder = CylinderGeometry(radius, height)
    differences = [
        abs(frustum.volume() - cylinder.volume()),
        *[
            abs(a - b)
            for a, b in zip(
                compute_inertia(frustum, MASS), compute_inertia(cylinder, MASS)
            )
        ],
    ]
    worst = max(differences)
    return worst <= 1e-12, (
        f"体积={frustum.volume():.10f}（圆柱 {cylinder.volume():.10f}），"
        f"惯量最大偏差={worst:.3e}"
    )


def test_frustum_degenerates_to_cone() -> tuple[bool, str]:
    """``R₂ = 0`` 时圆台必须等于圆锥——两条完全不同的代码路径给出同一结果。"""
    radius, height = 0.3, 1.2
    frustum = FrustumGeometry(radius, 0.0, height)
    cone = ConeGeometry(radius, height)
    differences = [
        abs(frustum.volume() - cone.volume()),
        *[
            abs(a - b)
            for a, b in zip(compute_inertia(frustum, MASS), compute_inertia(cone, MASS))
        ],
    ]
    worst = max(differences)
    return worst <= 1e-12, (
        f"体积={frustum.volume():.10f}（圆锥 {cone.volume():.10f}），"
        f"惯量最大偏差={worst:.3e}"
    )


# ---------------------------------------------------------------------------
# 5-7. 椭球与球冠的退化
# ---------------------------------------------------------------------------

def test_ellipsoid_degenerates_to_sphere() -> tuple[bool, str]:
    """``a = b = c = R`` 时椭球必须等于球。"""
    radius = 0.4
    ellipsoid = EllipsoidGeometry((radius, radius, radius))
    sphere = SphereGeometry(radius)
    differences = [
        abs(ellipsoid.volume() - sphere.volume()),
        *[
            abs(a - b)
            for a, b in zip(
                compute_inertia(ellipsoid, MASS), compute_inertia(sphere, MASS)
            )
        ],
    ]
    worst = max(differences)
    return worst <= 1e-12, (
        f"体积={ellipsoid.volume():.10f}（球 {sphere.volume():.10f}），"
        f"惯量最大偏差={worst:.3e}"
    )


def test_spherical_cap_hemisphere() -> tuple[bool, str]:
    """``h = R`` 的球冠是半球：``Izz = 2/5 m R²``、``Ixx = 83/320 m R²``。

    后一个数与 ``inertia.capsule_inertia`` 里半球那一段是同一个结果，两条
    独立路径互相印证。
    """
    radius = 0.5
    cap = SphericalCapGeometry(radius, radius)
    ixx, iyy, izz = compute_inertia(cap, MASS)
    want_izz = 0.4 * MASS * radius ** 2
    want_ixx = 83.0 / 320.0 * MASS * radius ** 2
    ok = (
        _close(cap.volume(), 2.0 / 3.0 * math.pi * radius ** 3)
        and _close(cap._base_z, -0.375 * radius)
        and _close(izz, want_izz)
        and _close(ixx, want_ixx)
        and _close(ixx, iyy)
    )
    return ok, (
        f"体积={cap.volume():.8f}，底面 z={cap._base_z:.6f}（应 {-0.375 * radius:.6f}），"
        f"Ixx={ixx:.8f}（应 {want_ixx:.8f}），Izz={izz:.8f}（应 {want_izz:.8f}）"
    )


def test_spherical_cap_full_sphere() -> tuple[bool, str]:
    """``h = 2R`` 的球冠退化为整球，三个惯量都必须等于球的。"""
    radius = 0.5
    cap = SphericalCapGeometry(radius, 2.0 * radius)
    sphere = SphereGeometry(radius)
    differences = [
        abs(cap.volume() - sphere.volume()),
        *[
            abs(a - b)
            for a, b in zip(compute_inertia(cap, MASS), compute_inertia(sphere, MASS))
        ],
    ]
    worst = max(differences)
    return worst <= 1e-9, (
        f"体积={cap.volume():.10f}（球 {sphere.volume():.10f}），"
        f"惯量最大偏差={worst:.3e}"
    )


# ---------------------------------------------------------------------------
# 8. 支撑映射：不劣于暴力采样
# ---------------------------------------------------------------------------

def _sampled_support(geometry, direction, samples: int = 20000) -> float:
    """在包围球里随机撒点，返回体内点中最大的投影（一个下界）。"""
    rng = random.Random(20261101)
    radius = geometry.bounding_radius()
    best = -math.inf
    for _ in range(samples):
        point = (
            rng.uniform(-radius, radius),
            rng.uniform(-radius, radius),
            rng.uniform(-radius, radius),
        )
        if not geometry.contains(point):
            continue
        score = (
            point[0] * direction[0]
            + point[1] * direction[1]
            + point[2] * direction[2]
        )
        if score > best:
            best = score
    return best


def test_support_matches_brute_force() -> tuple[bool, str]:
    """解析支撑点在任意方向上的投影都不低于采样结果。

    采样只能给出下界（撒不到精确极值点），所以判据是"解析值 ≥ 采样值 -
    很小量"。若方向分支写错（例如圆锥该取底面圆周却取了顶点），解析值会明显
    低于采样值，这条立刻失败。
    """
    geometries = {
        "圆锥": ConeGeometry(0.3, 1.2),
        "圆台": FrustumGeometry(0.4, 0.15, 0.9),
        "椭球": EllipsoidGeometry((0.5, 0.4, 0.3)),
        "球冠": SphericalCapGeometry(0.5, 0.2),
    }
    directions = ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.3, -0.5, 0.8))
    failures = []
    #: 解析值超出采样下界的最小余量。负值越大说明采样越没逼近极值点，
    #: 但只要不为负就说明解析支撑点没有漏掉更远的点。
    tightest = math.inf
    for label, geometry in geometries.items():
        for direction in directions:
            support = geometry.support(direction)
            analytic = sum(a * b for a, b in zip(support, direction))
            sampled = _sampled_support(geometry, direction)
            margin = analytic - sampled
            tightest = min(tightest, margin)
            if -margin > 1e-3 * geometry.bounding_radius():
                failures.append(f"{label} d={direction}: 解析 {analytic:.6f} < 采样 {sampled:.6f}")
    return not failures, (
        "全部方向解析值 ≥ 采样下界；最小余量 "
        f"{tightest:.3e}（采样只会低估，余量非负即通过）"
        if not failures
        else "；".join(failures)
    )


# ---------------------------------------------------------------------------
# 9. 包含测试
# ---------------------------------------------------------------------------

def test_contains_primitive() -> tuple[bool, str]:
    """内部点为真、外部点为假，四种新原语逐一验证。

    只在"明显内部/明显外部"取点，不测边界：边界归属依赖容差，属于实现细节。
    """
    cases = [
        ("圆锥", ConeGeometry(0.3, 1.2), (0.0, 0.0, 0.2), (0.0, 0.0, 0.95)),
        ("圆台", FrustumGeometry(0.4, 0.15, 0.9), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)),
        ("椭球", EllipsoidGeometry((0.5, 0.4, 0.3)), (0.1, 0.1, 0.1), (0.6, 0.0, 0.0)),
        ("球冠", SphericalCapGeometry(0.5, 0.2), (0.0, 0.0, 0.05), (0.45, 0.0, -0.3)),
    ]
    failures = []
    for label, geometry, inside, outside in cases:
        if not geometry.contains(inside):
            failures.append(f"{label} 漏判内部点 {inside}")
        if geometry.contains(outside):
            failures.append(f"{label} 误判外部点 {outside}")
    return not failures, (
        "四种原语的内外判定全部正确" if not failures else "；".join(failures)
    )


def test_contains_existing_geometry() -> tuple[bool, str]:
    """现有 5 种几何的 ``contains`` 已补齐（支撑映射重构后它们也得有包含测试）。"""
    cases = [
        ("盒", BoxGeometry((2.0, 2.0, 2.0)), (0.9, 0.9, 0.9), (1.2, 0.0, 0.0)),
        ("球", SphereGeometry(0.5), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0)),
        ("柱", CylinderGeometry(0.3, 1.0), (0.2, 0.0, 0.4), (0.0, 0.0, 0.6)),
        # 胶囊：轴向两端之外仍属于球冠区，所以内部的取点在半高之上
        ("胶囊", None, (0.0, 0.0, 0.7), (0.0, 0.0, 1.2)),
    ]
    from src.geom.geometry import CapsuleGeometry

    failures = []
    for label, geometry, inside, outside in cases:
        if geometry is None:
            geometry = CapsuleGeometry(0.25, 1.0)
        if not geometry.contains(inside):
            failures.append(f"{label} 漏判内部点 {inside}")
        if geometry.contains(outside):
            failures.append(f"{label} 误判外部点 {outside}")
    return not failures, (
        "盒/球/柱/胶囊的内外判定全部正确" if not failures else "；".join(failures)
    )


# ---------------------------------------------------------------------------
# 10. JSON 往返与注册表
# ---------------------------------------------------------------------------

def test_json_roundtrip() -> tuple[bool, str]:
    """四种新原语都能从字典重建，且参数与体积一致。"""
    samples = [
        ConeGeometry(0.3, 1.2),
        FrustumGeometry(0.4, 0.15, 0.9),
        EllipsoidGeometry((0.5, 0.4, 0.3)),
        SphericalCapGeometry(0.5, 0.2),
    ]
    failures = []
    for geometry in samples:
        rebuilt = GeometrySpec.from_dict(geometry.to_dict())
        if type(rebuilt) is not type(geometry):
            failures.append(f"{geometry.type_name} 重建类型错误")
        elif not _close(rebuilt.volume(), geometry.volume(), 1e-12):
            failures.append(
                f"{geometry.type_name} 体积不一致：{rebuilt.volume():.12f} vs {geometry.volume():.12f}"
            )
    return not failures, (
        "4 种原语的 to_dict → from_dict 往返一致"
        if not failures
        else "；".join(failures)
    )


def test_compound_has_no_support() -> tuple[bool, str]:
    """组合体没有整体支撑映射：它靠逐部件展开参与碰撞。

    这条测试是为了把"有意为之"钉住——如果哪天有人给 CompoundGeometry 补了
    一个"近似凸包"的 support，GJK 会把整个组合体当成凸体，凹槽被填平，
    碰撞体积凭空变大而不会报错。
    """
    from src.geom.geometry import CompoundGeometry, CompoundPart

    compound = CompoundGeometry(
        [
            CompoundPart(SphereGeometry(0.2), (-0.3, 0.0, 0.0)),
            CompoundPart(SphereGeometry(0.2), (0.3, 0.0, 0.0)),
        ]
    )
    try:
        compound.support((0.0, 0.0, 1.0))
    except TypeError as exc:
        return "支撑映射" in str(exc), f"按预期抛出 TypeError：{exc}"
    return False, "组合体竟然返回了支撑点——它不该被当作凸体"


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------

CASES = [
    ("1. 圆锥质心偏移与体积", test_cone_centroid_offset),
    ("2. 圆锥主惯量", test_cone_inertia),
    ("3. 圆台退化为圆柱", test_frustum_degenerates_to_cylinder),
    ("4. 圆台退化为圆锥", test_frustum_degenerates_to_cone),
    ("5. 椭球退化为球", test_ellipsoid_degenerates_to_sphere),
    ("6. 球冠 h=R 为半球", test_spherical_cap_hemisphere),
    ("7. 球冠 h=2R 为整球", test_spherical_cap_full_sphere),
    ("8. 支撑映射不劣于采样", test_support_matches_brute_force),
    ("9. 新原语的包含测试", test_contains_primitive),
    ("10. 现有几何的包含测试", test_contains_existing_geometry),
    ("11. JSON 往返", test_json_roundtrip),
    ("12. 组合体没有整体支撑映射", test_compound_has_no_support),
]


def main() -> int:
    print("=" * 88)
    print("参数化图元：体积 / 惯量 / 支撑映射 / 包含测试")
    print("=" * 88)

    failures = 0
    for title, case in CASES:
        try:
            ok, detail = case()
        except Exception as exc:  # 单个用例崩溃不应中断整轮
            ok, detail = False, f"异常：{type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {detail}")
        if not ok:
            failures += 1

    print("-" * 88)
    print(f"结果：{len(CASES) - failures}/{len(CASES)} 通过")
    print("=" * 88)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
