# -*- coding: utf-8 -*-
"""能力矩阵：哪种几何组合有窄相、哪种有 TOI，**只在这里定义一次**。

为什么需要这个模块
------------------
这个仓库出过一次很典型的文档漂移：``toi.py`` 在 W3 就覆盖了盒与凸包，但
``docs/06``、``docs/07``、``Code/README.md`` 里仍写着"TOI 只对球有效"，而且
README 同一份文件里前后自相矛盾。根因不是谁粗心，而是**散文写的能力清单
没法被验证**——它迟早会过期，而读者会相信更悲观的那一份。

所以能力现在只有这一份定义，并且被三处强制消费：

1. ``narrowphase.is_supported`` / ``supported_pair`` 从这里派生，不再是
   散落在各处的 ``isinstance`` 元组
2. ``tests/test_capabilities.py`` 逐格调用**真实分派函数**（``body_toi`` /
   ``ground_toi``）与本表比对，并检查表里的函数名确实存在——函数改名也会失败
3. 文档里的能力表由 :func:`block` **生成**；测试会把每个 ``.md`` 里的生成块
   与 :func:`block` 逐字比对。改了表却忘了改文档，测试直接红

新增一种几何因此只有三步：加进 :data:`SUPPORTED_KINDS`、补实现、
``python Code/scripts/sync_docs.py`` 重写文档块。漏掉任何一步都会在
测试里失败，而不是等到答辩时被问出来。

运行 ``python -m src.physics.capabilities`` 可直接打印当前矩阵。
"""

from __future__ import annotations

from ..geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CompoundGeometry,
    ConeGeometry,
    ConvexHullGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    GeometrySpec,
    SphereGeometry,
    SphericalCapGeometry,
)

#: 文档里生成块的分隔标记。测试只认这两个标记之间的内容，其余部分自由撰写。
MARK_BEGIN = "<!-- capabilities:begin -->"
MARK_END = "<!-- capabilities:end -->"

#: 几何种类标签（中文用于文档，英文键用于代码）
KIND_LABELS = {
    "sphere": "球",
    "box": "盒",
    "hull": "凸包",
    "cylinder": "柱",
    "capsule": "胶囊",
    "cone": "圆锥",
    "frustum": "圆台",
    "ellipsoid": "椭球",
    "spherical_cap": "球冠",
    "compound": "组合体",
}

#: 已有窄相实现的几何种类。顺序决定文档表格的行序，改动时同步看输出。
#:
#: 前五种是"棱柱/回转体"的老底子；中间四种是参数化曲面体（圆锥、圆台、椭球、
#: 球冠），它们全部走支撑映射，因此不需要写任何新的物体对代码；组合体放在最后，
#: 因为它是容器，键里带 compound 的格子都表示"展开成部件对"。
SUPPORTED_KINDS = (
    "sphere",
    "box",
    "hull",
    "cylinder",
    "capsule",
    "cone",
    "frustum",
    "ellipsoid",
    "spherical_cap",
    "compound",
)

#: 只有几何、没有碰撞的几何种类。它们连窄相都没有，因此也不可能有 TOI。
#: 现在为空：曲面体接入 GJK/EPA、参数化图元接入支撑特征之后，全部几何都有
#: 窄相、TOI 与地面三套实现。
UNSUPPORTED_KINDS = ()

ALL_KINDS = SUPPORTED_KINDS + UNSUPPORTED_KINDS

_GEOMETRY_KIND = {
    SphereGeometry: "sphere",
    BoxGeometry: "box",
    ConvexHullGeometry: "hull",
    CylinderGeometry: "cylinder",
    CapsuleGeometry: "capsule",
    ConeGeometry: "cone",
    FrustumGeometry: "frustum",
    EllipsoidGeometry: "ellipsoid",
    SphericalCapGeometry: "spherical_cap",
    CompoundGeometry: "compound",
}

# ---------------------------------------------------------------------------
# 物体对：窄相与 TOI
# ---------------------------------------------------------------------------
# 键是**按 ALL_KINDS 序号排好序的二元组**，所以只有一种写法，不存在
# ("box","sphere") 与 ("sphere","box") 各写一份、然后只改了一边的情况。

#: 窄相实现函数名。字符串而不是函数对象：本模块不该 import 物理模块
#: （否则与 narrowphase 互相依赖），由测试用 ``hasattr`` 验证它确实存在。
NARROWPHASE_IMPL = {
    ("sphere", "sphere"): "collide_sphere_sphere",
    ("sphere", "box"): "collide_sphere_box",
    ("sphere", "hull"): "collide_sphere_hull",
    ("box", "box"): "collide_box_box",
    ("box", "hull"): "collide_polytopes",
    ("hull", "hull"): "collide_polytopes",
    # 含侧面为曲面的几何：一律走 GJK + EPA 的通用路径（支撑映射驱动）
    ("sphere", "cylinder"): "collide_gjk",
    ("sphere", "capsule"): "collide_gjk",
    ("box", "cylinder"): "collide_gjk",
    ("box", "capsule"): "collide_gjk",
    ("hull", "cylinder"): "collide_gjk",
    ("hull", "capsule"): "collide_gjk",
    ("cylinder", "cylinder"): "collide_gjk",
    ("cylinder", "capsule"): "collide_gjk",
    ("capsule", "capsule"): "collide_gjk",
    # 参数化曲面体（圆锥、圆台、椭球、球冠）与所有几何之间：一律 GJK + EPA。
    # 它们没有有限 SAT 轴集合、也没有可枚举的顶点，但都有支撑映射，所以下面
    # 30 个格子是**零代码**——新增图元时完全不需要碰窄相分派。
    ("sphere", "cone"): "collide_gjk",
    ("sphere", "frustum"): "collide_gjk",
    ("sphere", "ellipsoid"): "collide_gjk",
    ("sphere", "spherical_cap"): "collide_gjk",
    ("box", "cone"): "collide_gjk",
    ("box", "frustum"): "collide_gjk",
    ("box", "ellipsoid"): "collide_gjk",
    ("box", "spherical_cap"): "collide_gjk",
    ("hull", "cone"): "collide_gjk",
    ("hull", "frustum"): "collide_gjk",
    ("hull", "ellipsoid"): "collide_gjk",
    ("hull", "spherical_cap"): "collide_gjk",
    ("cylinder", "cone"): "collide_gjk",
    ("cylinder", "frustum"): "collide_gjk",
    ("cylinder", "ellipsoid"): "collide_gjk",
    ("cylinder", "spherical_cap"): "collide_gjk",
    ("capsule", "cone"): "collide_gjk",
    ("capsule", "frustum"): "collide_gjk",
    ("capsule", "ellipsoid"): "collide_gjk",
    ("capsule", "spherical_cap"): "collide_gjk",
    ("cone", "cone"): "collide_gjk",
    ("cone", "frustum"): "collide_gjk",
    ("cone", "ellipsoid"): "collide_gjk",
    ("cone", "spherical_cap"): "collide_gjk",
    ("frustum", "frustum"): "collide_gjk",
    ("frustum", "ellipsoid"): "collide_gjk",
    ("frustum", "spherical_cap"): "collide_gjk",
    ("ellipsoid", "ellipsoid"): "collide_gjk",
    ("ellipsoid", "spherical_cap"): "collide_gjk",
    ("spherical_cap", "spherical_cap"): "collide_gjk",
    # 组合体：展开成部件对后复用上面所有路径
    ("sphere", "compound"): "collide_compound",
    ("box", "compound"): "collide_compound",
    ("hull", "compound"): "collide_compound",
    ("cylinder", "compound"): "collide_compound",
    ("capsule", "compound"): "collide_compound",
    ("cone", "compound"): "collide_compound",
    ("frustum", "compound"): "collide_compound",
    ("ellipsoid", "compound"): "collide_compound",
    ("spherical_cap", "compound"): "collide_compound",
    ("compound", "compound"): "collide_compound",
}

#: TOI 实现：(函数名, 精确性, 方法)
#:
#: 精确性只有两档，差别是有没有上界保证：
#:   * ``精确``   —— 匀速直线/定轴旋转假设下与解析解一致（可能带固定余量，见
#:                   ``toi._CONTACT_OVERLAP``，200 µm）
#:   * ``保守``   —— 可能偏早、绝不偏晚。偏早的代价是多切一刀，偏晚会漏切分
#: 这条区分必须保留：两者在"能不能拿解析解逐位比对"上完全不同。
TOI_IMPL = {
    ("sphere", "sphere"): ("sphere_sphere_toi", "精确", "一元二次方程取首次接触根"),
    ("sphere", "box"): ("sphere_hull_toi", "保守", "球心到多面体最近点 + 该处法向"),
    ("sphere", "hull"): ("sphere_hull_toi", "保守", "球心到多面体最近点 + 该处法向"),
    ("box", "box"): ("hull_hull_toi", "保守", "SAT 候选轴逐条推进"),
    ("box", "hull"): ("hull_hull_toi", "保守", "SAT 候选轴逐条推进"),
    ("hull", "hull"): ("hull_hull_toi", "保守", "SAT 候选轴逐条推进"),
    ("sphere", "cylinder"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("sphere", "capsule"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("box", "cylinder"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("box", "capsule"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("hull", "cylinder"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("hull", "capsule"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cylinder", "cylinder"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cylinder", "capsule"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("capsule", "capsule"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    # 参数化曲面体：同一条 GJK 距离 + 保守推进的路径，同样零代码
    ("sphere", "cone"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("sphere", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("sphere", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("sphere", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("box", "cone"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("box", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("box", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("box", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("hull", "cone"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("hull", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("hull", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("hull", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cylinder", "cone"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cylinder", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cylinder", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cylinder", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("capsule", "cone"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("capsule", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("capsule", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("capsule", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cone", "cone"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cone", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cone", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("cone", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("frustum", "frustum"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("frustum", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("frustum", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("ellipsoid", "ellipsoid"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("ellipsoid", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("spherical_cap", "spherical_cap"): ("curved_pair_toi", "保守", "GJK 距离 + 保守推进"),
    ("sphere", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("box", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("hull", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("cylinder", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("capsule", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("cone", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("frustum", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("ellipsoid", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("spherical_cap", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
    ("compound", "compound"): ("compound_pair_toi", "保守", "逐部件取最早"),
}

# ---------------------------------------------------------------------------
# 与地面
# ---------------------------------------------------------------------------
#: (窄相函数, TOI 函数, TOI 精确性, TOI 方法)
#:
#: 地面是无限平面，物体撞它时接触点必在顶点上，而顶点在"定速 + 定角速度"下
#: 走直线，所以这里三种几何的 TOI 都是**精确**的——与物体对之间的保守估计不同。
GROUND_IMPL = {
    "sphere": ("collide_sphere_ground", "sphere_ground_toi", "精确", "一元一次方程"),
    "box": ("collide_box_ground", "hull_ground_toi", "精确", "顶点解，含自转"),
    "hull": ("collide_hull_ground", "hull_ground_toi", "精确", "顶点解，含自转"),
    "cylinder": ("collide_cylinder_ground", "curved_ground_toi", "保守", "端面圆周最低点 + 自转上界"),
    "capsule": ("collide_capsule_ground", "curved_ground_toi", "保守", "端球最低点 + 自转上界"),
    # 参数化曲面体没有专属地面函数：它们统一走 collide_convex_ground——支撑特征
    # 裁剪 + 逐点深度，端面圆盘正对地面时自动给 4 点。TOI 复用保守的支撑点推进。
    "cone": ("collide_convex_ground", "curved_ground_toi", "保守", "支撑特征最低点 + 自转上界"),
    "frustum": ("collide_convex_ground", "curved_ground_toi", "保守", "支撑特征最低点 + 自转上界"),
    "ellipsoid": ("collide_convex_ground", "curved_ground_toi", "保守", "支撑特征最低点 + 自转上界"),
    "spherical_cap": ("collide_convex_ground", "curved_ground_toi", "保守", "支撑特征最低点 + 自转上界"),
    "compound": ("collide_compound_ground", "compound_ground_toi", "保守", "逐部件取最早"),
}


# ---------------------------------------------------------------------------
# 查询接口
# ---------------------------------------------------------------------------

def kind_of(geometry: GeometrySpec | None) -> str | None:
    """几何对象 → 种类标签；``None`` 或未知类型返回 ``None``。

    未知类型返回 ``None`` 而不是抛异常，是为了让调用方（加载阶段的校验）
    能给出"暂未支持"的业务错误，而不是一个 ``KeyError``。
    """
    if geometry is None:
        return None
    for cls, kind in _GEOMETRY_KIND.items():
        if isinstance(geometry, cls):
            return kind
    return None


#: 精确类型 -> 种类。``kind_of`` 用 ``isinstance`` 是为了容忍子类，代价是线性扫描；
#: TOI 的内层循环是 O(N²)，每对物体都扫一遍十项字典不合算，所以另开一张 O(1) 表。
_KIND_BY_TYPE = {cls: kind for cls, kind in _GEOMETRY_KIND.items()}


def kind_of_exact(geometry: GeometrySpec | None) -> str | None:
    """``kind_of`` 的 O(1) 版本：按**精确类型**查表，未命中再退回 ``isinstance``。

    内层循环专用。命中时零扫描；将来若出现几何子类，会由兜底分支正确处理，
    不会因为查表落空而静默变成"未支持"。
    """
    if geometry is None:
        return None
    kind = _KIND_BY_TYPE.get(type(geometry))
    if kind is not None:
        return kind
    return kind_of(geometry)


def pair_key(kind_a: str, kind_b: str) -> tuple[str, str]:
    """把一对几何种类排成唯一写法（按 :data:`ALL_KINDS` 的序号）。

    未知种类会在这里 ``ValueError``：能力表只认识 :data:`ALL_KINDS`，
    静默接受未知键会让新几何"看起来支持、实际查不到"。
    """
    try:
        index_a = ALL_KINDS.index(kind_a)
        index_b = ALL_KINDS.index(kind_b)
    except ValueError as exc:  # pragma: no cover - 防将来加几何时漏登记
        raise ValueError(
            f"未知几何种类：{kind_a!r} / {kind_b!r}；"
            f"请先加入 ALL_KINDS（当前：{ALL_KINDS}）"
        ) from exc
    return (kind_a, kind_b) if index_a <= index_b else (kind_b, kind_a)


def has_narrowphase(kind_a: str, kind_b: str) -> bool:
    """这一对几何之间是否已有窄相实现。"""
    return pair_key(kind_a, kind_b) in NARROWPHASE_IMPL


def narrowphase_function(kind_a: str, kind_b: str) -> str | None:
    """这一对的窄相实现函数名；不支持时返回 ``None``。"""
    return NARROWPHASE_IMPL.get(pair_key(kind_a, kind_b))


def has_ground_narrowphase(kind: str) -> bool:
    """该种几何是否有与地面的窄相实现。"""
    return kind in GROUND_IMPL


def toi_method(kind_a: str, kind_b: str):
    """这一对几何的 TOI：``(函数名, 精确性, 方法)``；不支持时返回 ``None``。"""
    return TOI_IMPL.get(pair_key(kind_a, kind_b))


def ground_toi_method(kind: str):
    """该种几何与地面的 TOI：``(窄相函数, TOI 函数, 精确性, 方法)`` 或 ``None``。"""
    return GROUND_IMPL.get(kind)


def all_pairs() -> list[tuple[str, str]]:
    """全部物体对（无序、无重复），顺序与 :data:`ALL_KINDS` 一致。"""
    pairs = []
    for index, kind_a in enumerate(ALL_KINDS):
        for kind_b in ALL_KINDS[index:]:
            pairs.append((kind_a, kind_b))
    return pairs


# ---------------------------------------------------------------------------
# 文档生成
# ---------------------------------------------------------------------------

def markdown_table() -> str:
    """生成能直接贴进 Markdown 的能力表（不含分隔标记）。

    表格里带**函数名**，这样文档不仅能回答"支不支持"，还能回答"在哪实现"。
    函数名由测试用 ``hasattr`` 验证，所以改名会立刻失败。
    """
    lines = [
        "| 几何 A | 几何 B | 窄相 | TOI | TOI 性质 |",
        "|---|---|---|---|---|",
    ]
    for kind_a, kind_b in all_pairs():
        narrow = NARROWPHASE_IMPL.get((kind_a, kind_b))
        narrow_cell = f"✅ `{narrow}`" if narrow else "✗"

        toi = TOI_IMPL.get((kind_a, kind_b))
        if toi:
            function, exactness, method = toi
            toi_cell = f"✅ `{function}`"
            method_cell = f"{exactness}（{method}）"
        else:
            toi_cell = "✗"
            method_cell = "—"

        lines.append(
            f"| {KIND_LABELS[kind_a]} | {KIND_LABELS[kind_b]} "
            f"| {narrow_cell} | {toi_cell} | {method_cell} |"
        )

    lines.append("")
    lines.append("| 几何 | 窄相 | TOI | TOI 性质 |")
    lines.append("|---|---|---|---|")
    for kind in ALL_KINDS:
        ground = GROUND_IMPL.get(kind)
        label = KIND_LABELS[kind]
        if ground:
            narrow, function, exactness, method = ground
            lines.append(
                f"| {label} - 地面 | ✅ `{narrow}` | ✅ `{function}` "
                f"| {exactness}（{method}） |"
            )
        else:
            lines.append(f"| {label} - 地面 | ✗ | ✗ | — |")

    return "\n".join(lines)


def block() -> str:
    """带分隔标记的生成块，供文档直接嵌入。"""
    return f"{MARK_BEGIN}\n{markdown_table()}\n{MARK_END}"


if __name__ == "__main__":  # pragma: no cover - 手工查看用
    print(block())
