# -*- coding: utf-8 -*-
"""组合体惯量的两种模型：``disjoint``（解析求和）与 ``overlap``（采样并集）。

要回答的问题
------------
1. 采样模式的**绝对**精度够不够？——拿单个球、单个盒与解析解比，而不是拿
   它跟解析模式"互相印证"（那样两个都错也看不出来）。
2. 部件真的重叠时，两种模式差多少？——重叠两球的并集体积有解析式，可以直接比。
3. 采样是不是**确定性**的？同一份输入两次构造必须逐位相同（不涉及随机种子，
   因为用的是 Halton 低差异序列）。
4. 场景 JSON 缺 ``overlap`` 字段时必须报错——这个假设不能默认。

精度实测（10 万采样点）：单球惯量误差 0.10%、单盒 0.001%、重叠两球并集体积
0.022%。下面的容差按"实测值的一个宽松倍数"取，防止把偶发的采样波动当成回归。
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
    CompoundGeometry,
    CompoundPart,
    GeometrySpec,
    SphereGeometry,
)
from src.geom.inertia import compute_inertia

MASS = 2.0


def _relative_error(got, want) -> float:
    return abs(got - want) / abs(want) if want != 0.0 else abs(got)


def _max_relative_error(got, want) -> float:
    return max(_relative_error(a, b) for a, b in zip(got, want))


# ---------------------------------------------------------------------------
# 1-2. 采样模式的绝对精度：单个部件应与该图元的解析惯量一致
# ---------------------------------------------------------------------------

def test_single_sphere_sampled() -> tuple[bool, str]:
    """单球组合体在 overlap 模式下的惯量必须逼近 ``2/5 m R²``。

    球的包围盒是个立方体，填充率只有 52%，是采样条件较差的构型——这里能过，
    说明低差异序列在"盒子里装球"这种情形也够用。
    """
    radius = 0.25
    sphere = SphereGeometry(radius)
    compound = CompoundGeometry([CompoundPart(sphere)], overlap="overlap")

    sampled = compound.principal_moments(MASS)
    analytic = compute_inertia(sphere, MASS)
    error = _max_relative_error(sampled, analytic)
    volume_error = _relative_error(compound.volume(), sphere.volume())
    return error < 0.005 and volume_error < 0.005, (
        f"惯量采样={tuple(round(v, 6) for v in sampled)}，解析={analytic}，"
        f"最大相对误差 {error * 100:.3f}%；体积误差 {volume_error * 100:.3f}%"
    )


def test_single_box_sampled() -> tuple[bool, str]:
    """单盒组合体：包围盒与形状完全贴合，采样精度应该高一个量级。"""
    box = BoxGeometry((0.6, 0.4, 0.3))
    compound = CompoundGeometry([CompoundPart(box)], overlap="overlap")

    sampled = compound.principal_moments(MASS)
    analytic = compute_inertia(box, MASS)
    error = _max_relative_error(sampled, analytic)
    return error < 0.001, (
        f"惯量采样={tuple(round(v, 6) for v in sampled)}，解析={analytic}，"
        f"最大相对误差 {error * 100:.4f}%"
    )


# ---------------------------------------------------------------------------
# 3-4. 不重叠 / 重叠两种构型下，两种模式的对照
# ---------------------------------------------------------------------------

def test_disjoint_and_sampled_agree_when_parts_separate() -> tuple[bool, str]:
    """部件确实不重叠时，两种模式必须给出同一个结果。

    这是"采样模式没有系统性偏差"的核心证据：如果有偏差，这里会稳定地偏
    0.5% 以上，而不是随机抖动。
    """
    parts = [
        CompoundPart(SphereGeometry(0.2), (-0.3, 0.0, 0.0)),
        CompoundPart(SphereGeometry(0.2), (0.3, 0.0, 0.0)),
    ]
    exact = CompoundGeometry(parts, overlap="disjoint")
    sampled = CompoundGeometry(parts, overlap="overlap")

    inertia_error = _max_relative_error(
        sampled.principal_moments(MASS), exact.principal_moments(MASS)
    )
    volume_error = _relative_error(sampled.volume(), exact.volume())
    return inertia_error < 0.01 and volume_error < 0.01, (
        f"惯量最大相对偏差 {inertia_error * 100:.3f}%，"
        f"体积相对偏差 {volume_error * 100:.3f}%"
    )


def test_overlap_uses_union_volume() -> tuple[bool, str]:
    """两个明显重叠的球：``overlap`` 模式的体积必须等于**并集**的解析值。

    并集体积 = 两球之和 − 透镜体积，``V_lens = π(4R+d)(2R−d)²/12``。这条同时
    说明了 ``disjoint`` 错在哪：它把重叠区算了两遍，偏大 18%。
    """
    radius, distance = 0.25, 0.25
    parts = [
        CompoundPart(SphereGeometry(radius), (-0.5 * distance, 0.0, 0.0)),
        CompoundPart(SphereGeometry(radius), (0.5 * distance, 0.0, 0.0)),
    ]
    single = 4.0 / 3.0 * math.pi * radius ** 3
    lens = math.pi * (4.0 * radius + distance) * (2.0 * radius - distance) ** 2 / 12.0
    exact_union = 2.0 * single - lens

    sampled = CompoundGeometry(parts, overlap="overlap")
    summed = CompoundGeometry(parts, overlap="disjoint")

    error = _relative_error(sampled.volume(), exact_union)
    overshoot = (summed.volume() - exact_union) / exact_union
    return error < 0.005 and overshoot > 0.1, (
        f"并集体积 采样={sampled.volume():.6f}，解析={exact_union:.6f}，"
        f"误差 {error * 100:.3f}%；disjoint 相加={summed.volume():.6f}，"
        f"偏大 {overshoot * 100:.2f}%"
    )


# ---------------------------------------------------------------------------
# 5. 确定性
# ---------------------------------------------------------------------------

def test_sampled_mode_is_deterministic() -> tuple[bool, str]:
    """同一份输入两次构造，惯量必须**逐位相同**。

    采样用的是 Halton 低差异序列，不是伪随机数，所以这里不需要种子，也不存在
    "换个种子结果就变"的隐患。物理上这保证了同一份 JSON 每次加载得到同一个
    刚体，基准数据才可复现。
    """
    parts = [
        CompoundPart(BoxGeometry((0.5, 0.3, 0.2)), (-0.2, 0.0, 0.0)),
        CompoundPart(SphereGeometry(0.25), (0.25, 0.0, 0.1)),
    ]
    first = CompoundGeometry(parts, overlap="overlap")
    second = CompoundGeometry(parts, overlap="overlap")

    same_moments = first._unit_moments == second._unit_moments
    same_frame = first.inertia_frame() == second.inertia_frame()
    same_volume = first.volume() == second.volume()
    return same_moments and same_frame and same_volume, (
        f"惯量逐位相同={same_moments}，主轴逐位相同={same_frame}，"
        f"体积逐位相同={same_volume}（V={first.volume():.10f}）"
    )


# ---------------------------------------------------------------------------
# 6. 场景 JSON：overlap 是必填字段
# ---------------------------------------------------------------------------

def _compound_json(**extra) -> dict:
    payload = {
        "type": "compound",
        "parts": [
            {
                "geometry": {"type": "box", "size": [0.3, 0.3, 0.3]},
                "position": [-0.25, 0.0, 0.0],
            },
            {
                "geometry": {"type": "sphere", "radius": 0.15},
                "position": [0.25, 0.0, 0.0],
            },
        ],
    }
    payload.update(extra)
    return payload


def test_json_requires_overlap_field() -> tuple[bool, str]:
    """缺 ``overlap`` 必须报错，且错误信息要把两个合法值说清楚。"""
    try:
        GeometrySpec.from_dict(_compound_json())
    except ValueError as exc:
        message = str(exc)
        ok = "overlap" in message and "disjoint" in message
        return ok, f"按预期报错：{message}"
    return False, "缺少 overlap 字段却构造成功了——这个假设不能有默认值"


def test_json_rejects_unknown_overlap() -> tuple[bool, str]:
    """非法取值必须被拒绝，而不是被当成 disjoint 静默接受。"""
    try:
        GeometrySpec.from_dict(_compound_json(overlap="maybe"))
    except ValueError as exc:
        return "overlap" in str(exc), f"按预期报错：{exc}"
    return False, "非法 overlap 值被静默接受了"


def test_json_roundtrip_keeps_mode() -> tuple[bool, str]:
    """两种模式都要能往返：导出的 JSON 必须带着 ``overlap`` 与采样数。"""
    for mode in ("disjoint", "overlap"):
        original = GeometrySpec.from_dict(_compound_json(overlap=mode))
        exported = original.to_dict()
        rebuilt = GeometrySpec.from_dict(exported)

        if exported.get("overlap") != mode or rebuilt.overlap != mode:
            return False, f"{mode} 模式的 overlap 字段在往返中丢失"
        if not math.isclose(rebuilt.volume(), original.volume(), rel_tol=1e-12):
            return False, (
                f"{mode} 模式体积往返不一致："
                f"{rebuilt.volume():.12f} vs {original.volume():.12f}"
            )
    return True, "disjoint / overlap 两种模式的 JSON 往返都保留了模式与体积"


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------

CASES = [
    ("1. 单球采样惯量对上解析值", test_single_sphere_sampled),
    ("2. 单盒采样惯量对上解析值", test_single_box_sampled),
    ("3. 不重叠时两种模式一致", test_disjoint_and_sampled_agree_when_parts_separate),
    ("4. 重叠时体积等于并集", test_overlap_uses_union_volume),
    ("5. 采样模式是确定性的", test_sampled_mode_is_deterministic),
    ("6. JSON 缺少 overlap 必须报错", test_json_requires_overlap_field),
    ("7. JSON 拒绝非法 overlap", test_json_rejects_unknown_overlap),
    ("8. JSON 往返保留模式", test_json_roundtrip_keeps_mode),
]


def main() -> int:
    print("=" * 88)
    print("组合体惯量：disjoint（解析）与 overlap（采样并集）")
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
