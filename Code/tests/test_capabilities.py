# -*- coding: utf-8 -*-
"""能力矩阵的守卫测试：让"文档说的能力"变成断言。

为什么要有这个文件
------------------
这个仓库出现过一次成规模的文档漂移：``toi.py`` 在 W3 就覆盖了盒与凸包，但
``docs/06``、``docs/07``、``Code/README.md`` 里仍写着"TOI 只对球有效"，README
同一份文件里还前后矛盾。根源不是谁粗心，而是**没有任何东西会因为你写错而失败**。

本文件把三件事钉住：

1. **表自洽**：几何种类、物体对、与地面三张表互相一致，没有"支持但查不到"的格子
2. **表 == 代码行为**：逐格调用真实分派函数（``body_toi`` / ``ground_toi`` /
   ``is_supported`` / ``supported_pair``）比对；表里写的**函数名**必须真的存在
3. **表 == 文档**：扫描所有 ``.md`` 里的生成块，与 ``capabilities.block()`` 逐字比对

第 3 条是关键：改了能力表却忘了同步文档，这里直接红；反过来，文档被手改成
"看起来更完整"的样子，也会红。

运行::

    python Code/tests/test_capabilities.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CompoundGeometry,
    CompoundPart,
    ConvexHullGeometry,
    CylinderGeometry,
    SphereGeometry,
)
from src.physics import capabilities, narrowphase, toi as toi_module
from src.physics.body import RigidBody
from src.physics.narrowphase import GroundPlane, is_supported, supported_pair
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SceneSpec

#: 探测用的凸包（四面体）。选它是因为面少、构造便宜，且与球/盒都明显不同形。
_TETRA = [(0.0, 0.0, 0.0), (0.7, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, 0.0, 0.7)]


def make_geometry(kind: str):
    """按种类名造一个真实几何对象。"""
    if kind == "sphere":
        return SphereGeometry(0.3)
    if kind == "box":
        return BoxGeometry((0.6, 0.6, 0.6))
    if kind == "hull":
        return ConvexHullGeometry(_TETRA)
    if kind == "cylinder":
        return CylinderGeometry(0.25, 0.6)
    if kind == "capsule":
        return CapsuleGeometry(0.25, 0.4)
    if kind == "compound":
        # 两个球拼成的哑铃：与球/盒/凸包/柱/胶囊都明显不同形
        piece = SphereGeometry(0.2)
        return CompoundGeometry(
            [
                CompoundPart(piece, (-0.3, 0.0, 0.0)),
                CompoundPart(piece, (0.3, 0.0, 0.0)),
            ]
        )
    raise AssertionError(f"未知种类：{kind}")


def make_body(
    name: str,
    geometry,
    position,
    velocity=(0.0, 0.0, 0.0),
    static: bool = False,
) -> RigidBody:
    """构造一个只有位姿与速度的刚体（质量属性与本次探测无关）。"""
    body = RigidBody(
        name=name, geometry=geometry, mass=1.0, position=position, is_static=static
    )
    body.linear_velocity = velocity
    return body


# ---------------------------------------------------------------------------
# 1. 表自洽
# ---------------------------------------------------------------------------

def test_table_self_consistent() -> tuple[bool, str]:
    """三张表必须互相自洽，且没有多余的键。

    - ``SUPPORTED_KINDS`` 与 ``UNSUPPORTED_KINDS`` 不重叠，合起来等于 ``ALL_KINDS``
    - 每张表里的几何名都在 ``ALL_KINDS`` 里（防手滑写错名字后静默失效）
    - 物体对表的键都是 ``pair_key`` 规范序，不会出现两种写法
    - 窄相表与 TOI 表的键集合相等：**有碰撞就一定有 TOI**，缺一个就是配置错误
    """
    kinds = capabilities.ALL_KINDS
    problems = []

    if set(capabilities.SUPPORTED_KINDS) & set(capabilities.UNSUPPORTED_KINDS):
        problems.append("SUPPORTED 与 UNSUPPORTED 有交集")
    merged = capabilities.SUPPORTED_KINDS + capabilities.UNSUPPORTED_KINDS
    if sorted(set(merged)) != sorted(set(kinds)):
        problems.append("ALL_KINDS 与两个子集之和不一致")

    for table_name, table in (
        ("NARROWPHASE_IMPL", capabilities.NARROWPHASE_IMPL),
        ("TOI_IMPL", capabilities.TOI_IMPL),
    ):
        for key in table:
            if len(key) != 2:
                problems.append(f"{table_name} 有非二元组键：{key}")
                continue
            for kind in key:
                if kind not in kinds:
                    problems.append(f"{table_name} 出现未知几何 {kind!r}")
            if capabilities.pair_key(*key) != key:
                problems.append(f"{table_name} 的键 {key} 不是规范序")

    for kind in capabilities.GROUND_IMPL:
        if kind not in kinds:
            problems.append(f"GROUND_IMPL 出现未知几何 {kind!r}")

    if set(capabilities.NARROWPHASE_IMPL) != set(capabilities.TOI_IMPL):
        problems.append("窄相表与 TOI 表的键集合不同：有碰撞却没有 TOI 是配置错误")

    return not problems, (
        "；".join(problems) if problems
        else f"{len(kinds)} 种几何、{len(capabilities.all_pairs())} 个物体对、"
             f"{len(capabilities.GROUND_IMPL)} 种地面组合，全部有定义"
    )


# ---------------------------------------------------------------------------
# 2. 表 == 代码行为
# ---------------------------------------------------------------------------

def test_implementation_functions_exist() -> tuple[bool, str]:
    """表里写的函数名必须真的存在。

    这是"文档写着 `hull_hull_toi`，实际早就改名了"这类漂移的检测点。
    """
    missing = []
    for key, name in capabilities.NARROWPHASE_IMPL.items():
        if not hasattr(narrowphase, name):
            missing.append(f"narrowphase.{name}（窄相 {key}）")
    for key, entry in capabilities.TOI_IMPL.items():
        if not hasattr(toi_module, entry[0]):
            missing.append(f"toi.{entry[0]}（TOI {key}）")
    for kind, entry in capabilities.GROUND_IMPL.items():
        if not hasattr(narrowphase, entry[0]):
            missing.append(f"narrowphase.{entry[0]}（地面窄相 {kind}）")
        if not hasattr(toi_module, entry[1]):
            missing.append(f"toi.{entry[1]}（地面 TOI {kind}）")

    return not missing, (
        "缺失：" + "，".join(missing) if missing
        else f"{len(capabilities.NARROWPHASE_IMPL)} 个窄相 + "
             f"{len(capabilities.TOI_IMPL)} 个 TOI + "
             f"{len(capabilities.GROUND_IMPL)} 个地面函数，全部存在"
    )


def test_single_geometry_support() -> tuple[bool, str]:
    """``is_supported`` 必须与 ``SUPPORTED_KINDS`` 逐格一致。"""
    mismatches = []
    for kind in capabilities.ALL_KINDS:
        expected = kind in capabilities.SUPPORTED_KINDS
        actual = is_supported(make_geometry(kind))
        if expected != actual:
            mismatches.append(f"{kind}：表={expected} 实际={actual}")

    if is_supported(None):
        mismatches.append("is_supported(None) 应为 False")

    return not mismatches, (
        "，".join(mismatches) if mismatches
        else f"{len(capabilities.ALL_KINDS)} 种几何逐个核对，"
             f"支持 {len(capabilities.SUPPORTED_KINDS)} 种"
    )


def test_pair_support() -> tuple[bool, str]:
    """``supported_pair`` 必须与 ``NARROWPHASE_IMPL`` 逐格一致（含双向）。"""
    mismatches = []
    for kind_a, kind_b in capabilities.all_pairs():
        geometry_a, geometry_b = make_geometry(kind_a), make_geometry(kind_b)
        expected = capabilities.has_narrowphase(kind_a, kind_b)
        forward = supported_pair(geometry_a, geometry_b)
        backward = supported_pair(geometry_b, geometry_a)
        if forward != expected or backward != expected:
            mismatches.append(
                f"{kind_a}-{kind_b}：表={expected} 实际={forward}/{backward}"
            )

    return not mismatches, (
        "，".join(mismatches) if mismatches
        else f"{len(capabilities.all_pairs())} 个物体对双向一致"
    )


def test_dispatch_table_matches_capabilities() -> tuple[bool, str]:
    """窄相**分派表**与能力矩阵必须逐格对齐（两个方向都要查）。

    这是本文件里最要紧的一条。以前"支持哪一对几何"写在两个地方——分派代码里
    一串 ``isinstance``，和能力矩阵里一份声明——两份必须同步，而没人会记得同步。
    漏掉一处不会报错，只会让两个物体**静默互相穿过**，属于最难查的一类问题。

    现在分派表里每个条目自带"我在矩阵里叫什么"，本测试逐格验证：

    - 表里有的组合：函数名必须与矩阵一致
    - 矩阵里有、表里没有的组合：必须落到 GJK/EPA 兜底，且矩阵也必须这么写

    复合体不查：它在 ``_collide_pair`` 里逐部件展开，压根不经过这张表。
    """
    from src.physics.narrowphase import _PAIR_DISPATCH

    fallback_name = "collide_gjk"
    problems = []
    checked = 0

    for kind_a, kind_b in capabilities.all_pairs():
        declared = capabilities.narrowphase_function(kind_a, kind_b)
        if declared is None or "compound" in (kind_a, kind_b):
            continue
        # 双向都要成立：算法不应该依赖调用方把谁放在前面
        for first, second in ((kind_a, kind_b), (kind_b, kind_a)):
            entry = _PAIR_DISPATCH.get((first, second))
            resolved = entry[0] if entry is not None else fallback_name
            checked += 1
            if resolved != declared:
                problems.append(
                    f"{first}-{second}：矩阵={declared} 分派={resolved}"
                )

    return not problems, (
        "，".join(problems) if problems
        else f"{checked} 个方向的分派与矩阵名称一致（未覆盖的组合统一走 {fallback_name}）"
    )


def _pair_toi(kind_a: str, kind_b: str):
    """构造"必然相撞"的构型，返回 ``body_toi`` 的两个方向结果。

    A 静止在原点，B 在 +x 侧、初始表面间隙 1 m，以 2 m/s 沿 -x 靠近——
    只要实现支持的组合都会给出事件；不支持的组合必须返回 ``None``。
    后者同样重要：表说"没有"，代码却报了事件，会让人不敢删那行表。
    """
    body_a = make_body("a", make_geometry(kind_a), (0.0, 0.0, 0.0), static=True)
    body_b = make_body("b", make_geometry(kind_b), (0.0, 0.0, 0.0))

    distance = (
        body_a.geometry.bounding_radius()
        + body_b.geometry.bounding_radius()
        + 1.0
    )
    body_b.position = (distance, 0.0, 0.0)
    body_b.linear_velocity = (-2.0, 0.0, 0.0)

    return (
        toi_module.body_toi(body_a, body_b, 20.0),
        toi_module.body_toi(body_b, body_a, 20.0),
    )


def test_pair_toi_matches_table() -> tuple[bool, str]:
    """``body_toi`` 的支持情况必须与 ``TOI_IMPL`` 逐格一致（含双向对称）。"""
    problems = []
    reported = []
    for kind_a, kind_b in capabilities.all_pairs():
        expected = capabilities.toi_method(kind_a, kind_b) is not None
        forward, backward = _pair_toi(kind_a, kind_b)

        for label, value in (("正向", forward), ("反向", backward)):
            if (value is not None) != expected:
                problems.append(
                    f"{kind_a}-{kind_b} {label}：表={'有' if expected else '无'} "
                    f"实际={value}"
                )
        if forward is not None and backward is not None:
            if abs(forward - backward) > 1e-9:
                problems.append(
                    f"{kind_a}-{kind_b} 双向不对称：{forward:.6f} vs {backward:.6f}"
                )
            reported.append(f"{kind_a}-{kind_b}={forward:.3f}s")

    return not problems, (
        "，".join(problems) if problems
        else "支持的组合都报出事件：" + "，".join(reported)
    )


def test_ground_toi_matches_table() -> tuple[bool, str]:
    """``ground_toi`` 的支持情况必须与 ``GROUND_IMPL`` 逐格一致。"""
    ground = GroundPlane(z=0.0)
    problems = []
    reported = []

    for kind in capabilities.ALL_KINDS:
        body = make_body("falling", make_geometry(kind), (0.0, 0.0, 0.0))
        # 抬到"最低点离地 0.5 m"：多面体要保证没有顶点已经贴地，
        # 否则会被正确地判为"持续接触"而返回 None（见 hull_ground_toi 的说明）
        body.position = (0.0, 0.0, body.geometry.bounding_radius() + 0.5)
        body.linear_velocity = (0.0, 0.0, -2.0)

        value = toi_module.ground_toi(body, ground, 20.0)
        expected = capabilities.ground_toi_method(kind) is not None
        if (value is not None) != expected:
            problems.append(
                f"{kind}-地面：表={'有' if expected else '无'} 实际={value}"
            )
        if value is not None:
            reported.append(f"{kind}={value:.3f}s")

    return not problems, (
        "，".join(problems) if problems
        else "地面 TOI：" + "，".join(reported)
    )


def _geometry_dict(kind: str) -> dict:
    """``SceneSpec.from_dict`` 吃的几何字典（只覆盖未支持的种类）。"""
    if kind == "cylinder":
        return {"type": "cylinder", "radius": 0.25, "height": 0.6}
    if kind == "capsule":
        return {"type": "capsule", "radius": 0.25, "height": 0.4}
    raise AssertionError(f"本函数只用于未支持的几何：{kind}")


def test_unsupported_rejected_at_load() -> tuple[bool, str]:
    """没有窄相实现的几何必须在**加载阶段**被拒绝。

    这是能力表最实际的消费者：``PhysicsWorld.from_scene`` 用它把场景挡在门外。
    没有这道闸门，柱/胶囊会静默穿过一切，而这种症状极难归因。
    """
    if not capabilities.UNSUPPORTED_KINDS:
        return True, (
            "UNSUPPORTED_KINDS 为空：全部几何都有窄相与 TOI，"
            "闸门无事可做（保留此检查，将来再出现『只有几何』的类型时自动生效）"
        )

    messages = []
    ok = True
    for kind in capabilities.UNSUPPORTED_KINDS:
        payload = {
            "name": f"only_{kind}",
            "simulation": {"enabled": True, "duration": 0.1},
            "objects": [
                {
                    "name": kind,
                    "geometry": _geometry_dict(kind),
                    "transform": {"position": [0.0, 0.0, 1.0]},
                }
            ],
        }
        try:
            PhysicsWorld.from_scene(SceneSpec.from_dict(payload))
        except ValueError as exc:
            text = str(exc)
            rejected = "暂未" in text or "支持" in text
            ok = ok and rejected
            label = "被拒绝" if rejected else "报错但文案不含『暂未支持』"
            messages.append(f"{kind}：{label} → {text[:36]}")
        else:
            ok = False
            messages.append(f"{kind}：未报错（闸门失效）")

    return ok, "；".join(messages)


# ---------------------------------------------------------------------------
# 3. 表 == 文档
# ---------------------------------------------------------------------------

def _markdown_files(root: str) -> list[str]:
    """仓库里所有 .md（跳过版本库与缓存目录）。"""
    found = []
    skip = {".git", "__pycache__", "node_modules", ".venv", "venv"}
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            if name.endswith(".md"):
                found.append(os.path.join(current, name))
    return found


def _first_difference(actual: str, expected: str) -> str:
    """指出第一处不同的行，便于定位（整块贴出来太长）。"""
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    for index in range(max(len(actual_lines), len(expected_lines))):
        got = actual_lines[index] if index < len(actual_lines) else "<缺失>"
        want = expected_lines[index] if index < len(expected_lines) else "<多余>"
        if got != want:
            return f"第 {index + 1} 行 文档={got!r} 应为={want!r}"
    return "内容相同但首尾空白不同"


def test_docs_block_matches_table() -> tuple[bool, str]:
    """每个 ``.md`` 里的生成块必须与 ``capabilities.block()`` 逐字一致。

    不一致时的修法**不是手改文档**，而是::

        python Code/scripts/sync_docs.py

    手改会在下一次同步时被覆盖，而且中间那段时间文档与代码已经分叉。
    """
    root = os.path.dirname(_CODE_ROOT)
    expected = capabilities.markdown_table().strip()

    problems = []
    embedded = []
    for path in _markdown_files(root):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if capabilities.MARK_BEGIN not in text:
            continue

        begin = text.index(capabilities.MARK_BEGIN) + len(capabilities.MARK_BEGIN)
        end = text.index(capabilities.MARK_END, begin)
        actual = text[begin:end].strip()
        relative = os.path.relpath(path, root)
        if actual != expected:
            problems.append(f"{relative}：{_first_difference(actual, expected)}")
        embedded.append(relative)

    if not embedded:
        return False, (
            f"没有任何 .md 带生成标记（{capabilities.MARK_BEGIN}），文档失去了约束"
        )

    return not problems, (
        "；".join(problems) if problems
        else f"{len(embedded)} 处生成块与表一致（{'，'.join(embedded)}）"
    )


CHECKS = [
    ("1. 能力表自洽", test_table_self_consistent),
    ("2. 表里的函数名都存在", test_implementation_functions_exist),
    ("3. 单几何支持判据与表一致", test_single_geometry_support),
    ("4. 物体对窄相与表一致", test_pair_support),
    ("5. 分派表与矩阵逐格对齐", test_dispatch_table_matches_capabilities),
    ("6. 物体对 TOI 与表一致（双向）", test_pair_toi_matches_table),
    ("7. 地面 TOI 与表一致", test_ground_toi_matches_table),
    ("8. 未支持几何在加载阶段被拒绝", test_unsupported_rejected_at_load),
    ("9. 文档生成块与表逐字一致", test_docs_block_matches_table),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 88)
    print("能力矩阵（窄相 / TOI 的覆盖范围）")
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
