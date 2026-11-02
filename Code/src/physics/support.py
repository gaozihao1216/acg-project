# -*- coding: utf-8 -*-
"""支撑映射（support mapping）：GJK / EPA 唯一的几何输入。

为什么要有这一层
----------------
窄相原先按"几何对"手写函数：球-球、球-盒、盒-盒、盒-凸包、凸包-凸包……
每加一种几何，就要补 O(N) 个函数，再加 O(N) 个 TOI 函数。这条路在
"球 + 盒 + 凸包"时刚好走得通，但柱与胶囊一进来就要补 9 个窄相 + 9 个 TOI，
而且曲面与多面体的组合根本没有干净的 SAT 轴集合可用。

支撑映射把问题一次性收口：**任意凸体都可以用一个函数描述**

    ``support(d) = 该物体上沿方向 d 最远的点``

有了它，GJK 负责"离多远 / 碰没碰"，EPA 负责"陷多深 / 沿哪个方向推出"，
于是"几何对"不再是代码分派维度，而是数据维度——加一种几何只需要写
这一个函数，不需要碰任何碰撞代码。

局部 vs 世界
------------
所有 ``support_local`` 都在**物体自身的主轴系**里工作（球心/质心在原点，
柱与胶囊的轴是 +Z）。``support_world`` 负责先转到局部、算完再转回世界，
这样每种几何的支撑函数都只有几行、且与姿态无关。
"""

from __future__ import annotations

import math

from ..geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    ConeGeometry,
    ConvexHullGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    SphereGeometry,
    SphericalCapGeometry,
)
from ..mathutils import quaternion as quat
from ..mathutils import vec3

#: 方向向量长度小于它就当作"没有方向"，退回物体自身的中心/轴线端点。
_EPS = 1e-12

#: 支撑特征采样：方向与轴线夹角余弦超过它，认为"正对端面"（约 3°）。
#:
#: 用几度的容差而不是严格的 ``cos == ±1``：立着的柱被求解器微调几毫弧度后，
#: 严格判据会让流形从 4 点塌成 1 点，物体随即开始摇摆。容差内的"近似端面"
#: 取 4 个圆周点，深度逐点按投影算，多出来的点会被自动裁掉，不会引入错误。
_FACE_COS_TOL = 0.9986

#: 同上：余弦落在这个带宽内认为"接近侧面接触"（约 3°），取 2 个母线端点。
_SIDE_COS_TOL = 0.05

#: 端面圆周采样角的序号是**标签**的一部分，所以这张表必须是常量：
#: 45°/135°/225°/315° 避开 0°，免得立着的柱在 x/y 轴上留下与坐标轴对齐的接触点
#: （那种对齐会让摩擦约束出现方向退化）。
_RING_ANGLES = (
    math.pi * 0.25,
    math.pi * 0.75,
    math.pi * 1.25,
    math.pi * 1.75,
)

#: 回转体"母线端点"类标签的起点。端面环占用 0..7，这里错开。
_TAG_SIDE = 8

#: 单点（严格凸体的锥顶、斜撑时的圆台侧面）与退化端面的标签。
#: 与端面环的 0..7 错开，是为了"端面接触 ↔ 单点接触"切换时不会撞上环上某一点。
_TAG_SINGLE = 16


def support_local(geometry, direction) -> tuple[float, float, float]:
    """几何体在**局部坐标**下沿 ``direction`` 的最远点。

    实现已经下沉到 ``GeometrySpec.support``：每种几何最清楚自己的支撑点怎么算，
    这里只做一次委托。这样新增一种几何**不需要碰本模块**——"加几何就要改碰撞
    代码"正是支撑映射要消灭的成本。

    参数
    ----
    geometry  : 任一种几何对象
    direction : 搜索方向（不必归一化，长度为零时退回中心附近的确定点）

    返回
    ----
    tuple[float, float, float] : 局部坐标下的支撑点

    异常
    ----
    TypeError : 该几何没有支撑映射（例如组合体，它靠逐部件展开参与碰撞）
    """
    return geometry.support(direction)


def support_world(body, direction) -> tuple[float, float, float]:
    """刚体在世界坐标下沿 ``direction`` 的最远点。"""
    inverse = quat.quat_conjugate(body.orientation)
    local_direction = quat.quat_rotate_vector(inverse, direction)
    local_point = support_local(body.geometry, local_direction)
    world = quat.quat_rotate_vector(body.orientation, local_point)
    return (world[0] + body.position[0], world[1] + body.position[1], world[2] + body.position[2])


def world_axis(body) -> tuple[float, float, float]:
    """柱/胶囊的轴线在世界坐标下的方向（局部 +Z 的像）。"""
    return quat.quat_rotate_vector(body.orientation, (0.0, 0.0, 1.0))


def feature_world(
    body, direction, tol: float = 1e-4
) -> list[tuple[float, float, float]]:
    """沿 ``direction`` 的**支撑特征**点列，不带身份标签。

    这是 :func:`feature_world_tagged` 的薄包装：只关心"哪里在接触"的调用方用它；
    需要给接触点分配 ``feature_id`` 的调用方（地面接触、GJK 流形）必须用带标签的
    版本，否则会退化成用数组下标当身份——那正是暖启动错配的根源。
    """
    return [point for point, _tag in feature_world_tagged(body, direction, tol)]


#: 标签上限。``gjk._build_manifold`` 用 ``tag_a * TAG_LIMIT + tag_b`` 编码两侧。
#:
#: 512 的来源：凸包顶点数受 ``ConvexHullGeometry(max_faces=64)`` 约束，欧拉公式
#: 给出 ``V = 2F - 4 ≈ 124`` 的上界，512 留了四倍余量。同时 511*512+511 = 262143
#: 远小于组合体的部件偏移 ``narrowphase._PART_FEATURE_STRIDE``（1e6），
#: 于是"部件对 × 局部 ID"的编码仍不会撞号。
TAG_LIMIT = 512


def feature_world_tagged(
    body, direction, tol: float = 1e-4
) -> list[tuple[tuple[float, float, float], int]]:
    """沿 ``direction`` 的**支撑特征**，每个点带一个几何特征标签。

    包装 :func:`_feature_branches` 并校验标签范围：``TAG_LIMIT`` 是
    ``gjk._build_manifold`` 编码两侧标签时的隐含宽度。将来新增几何若产出越界
    标签，必须在这里**立刻失败**，而不是在编码后静默撞键——那种错误只会表现为
    "某个角偶尔复用了别处的冲量"，没有任何报错。
    """
    points = _feature_branches(body, direction, tol)
    for _point, tag in points:
        if not 0 <= tag < TAG_LIMIT:
            raise ValueError(
                f"支撑特征标签 {tag} 超出 [0, {TAG_LIMIT})："
                f"{type(body.geometry).__name__} 的编码与 TAG_LIMIT 不一致"
            )
    return points


def _feature_branches(
    body, direction, tol: float = 1e-4
) -> list[tuple[tuple[float, float, float], int]]:
    """各几何的支撑特征分支（不校验标签，对外入口见 :func:`feature_world_tagged`）。

    沿 ``direction`` 的**支撑特征**：一组"几乎同样远"的表面点，各带一个标签。

    为什么需要它
    ------------
    GJK/EPA 只给一个法向和一个深度，对应的接触点是**一个**。对平面-平面接触，
    单点会让物体绕该点自由翻转——盒-地面当初必须返回 4 个点，正是这个原因。
    曲面上同样存在"成片接触"：立着的圆柱底面贴着别的面时，接触区是一整圈；
    躺着的胶囊贴着平面时，是一条线段。

    所以这里按几何形状解析地枚举"朝 ``direction`` 看过去最凸的那一片"：

    - 球 / 椭球：恒定 1 个点（严格凸）
    - 盒 / 凸包：支撑值落在最大值 ``tol`` 内的顶点（面贴面时最多 4 个）
    - 圆柱：正对端面 → 该端面圆周上 4 个点；正对侧面 → 两条母线的端点（2 个）
    - 胶囊：正对侧面 → 两端球心各一点（2 个）
    - 圆锥 / 圆台 / 球冠：端面圆盘正对方向 → 该圆周上 4 个点；否则 1 个点

    圆盘类几何"正对"的判据是方向与轴线的夹角余弦，用约 3° 的容差而不是严格相等：
    立着的柱被求解器微调几毫弧度后，严格判据会让流形从 4 点塌成 1 点，物体随即
    开始摇摆。

    返回的点都**在表面上**（不是中点），接触点由调用方在两侧之间取中点。

    标签（``tag``）的语义
    --------------------
    标签在**同一个物体内唯一**，且与"点在返回列表里的第几个"无关。这是硬要求，
    也是它能当暖启动键的前提。但"跨帧稳定"要分两种强度，不能一概而论：

    | 标签来源 | 性质 |
    |---|---|
    | 盒的顶点序号 ``4*ix + 2*iy + iz``、凸包的顶点下标 | **真特征**：体坐标系里固定的那个角 |
    | 圆柱 / 胶囊的母线端点序号 | **真特征**：体坐标系里固定的那一端 |
    | 圆盘四点（柱端面、锥/台/冠底面） | **采样槽位**：``_RING_ANGLES`` 固定的 4 个角，但圆周起点由 ``vec3.orthogonal_basis(世界轴)`` 决定 |
    | 严格凸体（球、椭球）与退化端面的单点 | 常量 ``_TAG_SINGLE``，只表示"本帧唯一的那个点" |

    圆盘那一行是**已知边界**：``orthogonal_basis`` 按"轴的最小分量"挑辅助坐标轴，
    轴方向跨越该分界时基底不连续，同一个圆周序号可能对应跳了很远的另一个位置
    （实测两轴夹角 2e-8 时首点切向差 114°）。批内唯一性不受影响——四点仍是四个
    不同标签——但它**不是**完整的持久接触身份。真正的跨帧匹配要靠两侧局部锚点 +
    有效性检查（见 `docs/07` §4.1）。

    裁点那条性质仍然成立：倾斜时浅点被裁掉，剩余点的标签**不变**（是原子集的
    子集），不会像数组下标那样整体前移。
    """
    geometry = body.geometry

    defect = math.sqrt(
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    if defect <= _EPS:
        # 零/极短方向：没有"最凸的那一片"可枚举，退化成单个支撑点。
        # 这里曾经漏包标签，直接返回了裸点，让 feature_world() 解包时抛
        # ValueError；零向量在 GJK 退化情形（两体中心重合）下真的会出现。
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, SphereGeometry):
        return [(support_world(body, direction), 0)]

    if isinstance(geometry, BoxGeometry):
        size = geometry.size
        half = (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])
        local_direction = quat.quat_rotate_vector(
            quat.quat_conjugate(body.orientation), direction
        )
        best = (
            abs(local_direction[0]) * half[0]
            + abs(local_direction[1]) * half[1]
            + abs(local_direction[2]) * half[2]
        )
        points = []
        # 顶点标签 = 4*ix + 2*iy + iz：与 collide_box_ground 的枚举顺序同源，
        # 于是"盒-地面"和"盒-盒"两条路径对同一个角给出同一个身份。
        for index_x, sign_x in enumerate((-1.0, 1.0)):
            for index_y, sign_y in enumerate((-1.0, 1.0)):
                for index_z, sign_z in enumerate((-1.0, 1.0)):
                    vertex = (sign_x * half[0], sign_y * half[1], sign_z * half[2])
                    score = (
                        vertex[0] * local_direction[0]
                        + vertex[1] * local_direction[1]
                        + vertex[2] * local_direction[2]
                    )
                    if best - score <= tol:
                        tag = 4 * index_x + 2 * index_y + index_z
                        points.append((_local_to_world(body, vertex), tag))
        return points

    if isinstance(geometry, ConvexHullGeometry):
        local_direction = quat.quat_rotate_vector(
            quat.quat_conjugate(body.orientation), direction
        )
        scored = [
            (
                vertex[0] * local_direction[0]
                + vertex[1] * local_direction[1]
                + vertex[2] * local_direction[2],
                vertex,
                index,
            )
            for index, vertex in enumerate(geometry.vertices)
        ]
        best = max(item[0] for item in scored)
        return [
            (_local_to_world(body, vertex), index)
            for score, vertex, index in scored
            if best - score <= tol
        ]

    axis = world_axis(body)
    cos_axis = (
        direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]
    ) / defect

    if isinstance(geometry, CylinderGeometry):
        radius = geometry.radius
        half_height = 0.5 * geometry.height
        if abs(cos_axis) >= _FACE_COS_TOL:
            # 端面接触：圆周上等角取 4 点，足以撑住"平放不翻"
            cap = 1 if cos_axis > 0.0 else 0
            center = vec3.madd(
                body.position, axis, half_height if cos_axis > 0.0 else -half_height
            )
            e1, e2 = vec3.orthogonal_basis(axis)
            return [
                (vec3.add(center, _ring_offset(e1, e2, angle, radius)), cap * 4 + step)
                for step, angle in enumerate(_RING_ANGLES)
            ]
        if abs(cos_axis) <= _SIDE_COS_TOL:
            # 侧面接触：两条母线的端点
            radial = _radial_direction(direction, axis)
            return [
                (
                    vec3.madd(
                        vec3.madd(body.position, axis, sign * half_height), radial, radius
                    ),
                    _TAG_SIDE + index,
                )
                for index, sign in enumerate((-1.0, 1.0))
            ]
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, CapsuleGeometry):
        radius = geometry.radius
        half_height = 0.5 * geometry.height
        if abs(cos_axis) <= _SIDE_COS_TOL:
            radial = _radial_direction(direction, axis)
            return [
                (
                    vec3.madd(
                        vec3.madd(body.position, axis, sign * half_height), radial, radius
                    ),
                    index,
                )
                for index, sign in enumerate((-1.0, 1.0))
            ]
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, EllipsoidGeometry):
        # 椭球是严格凸的：任意方向只有一个支撑点
        return [(support_world(body, direction), 0)]

    if isinstance(geometry, ConeGeometry):
        if cos_axis <= -_FACE_COS_TOL:
            # 底面正对方向：整个底面圆盘等深，圆周取 4 点才撑得住"平放不翻"
            return _face_ring(body, axis, geometry._base_z, geometry.radius, 0)
        # 其余方向（含顶点朝下）只有一个支撑点：圆锥是"圆盘 + 顶点"的凸包，
        # 支撑集要么是整块底面，要么是单点，不会出现线段
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, FrustumGeometry):
        if cos_axis <= -_FACE_COS_TOL:
            return _face_ring(body, axis, geometry._bottom_z, geometry.bottom_radius, 0)
        if cos_axis >= _FACE_COS_TOL:
            return _face_ring(body, axis, geometry._top_z, geometry.top_radius, 4)
        # 斜着支撑时最大值落在某条端面圆周上，是**一个**点（圆台侧面是直纹面，
        # 但只有沿母线方向才会退化成线段，而那条线段的深度并不相同）
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, SphericalCapGeometry):
        if cos_axis <= -_FACE_COS_TOL:
            # 平底正对方向：底面圆盘就是接触面
            return _face_ring(body, axis, geometry._base_z, geometry._rim_radius, 0)
        return [(support_world(body, direction), _TAG_SINGLE)]

    raise TypeError(f"不支持支撑特征采样的几何类型：{type(geometry).__name__}")


def _face_ring(body, axis, center_local_z: float, radius: float, base_tag: int):
    """端面接触：把该端面的圆周等角取 4 点（世界坐标），各带标签 ``base_tag + 序号``。

    取 4 点是"够用"而不是"精确"：端面是个圆盘，理论上无穷多点等深。4 个等分点
    足以让法向约束张成整个平面，物体不会绕单点翻转；倾斜时其余点深度更浅，
    会被逐点裁剪掉，自然退化成 1–2 点接触。

    角度取 45°、135°、225°、315°：避开 0°，免得"立得笔直的柱"在 x/y 轴上留下
    与坐标轴方向重合的接触点——那种对齐会让求解器的摩擦约束出现方向退化。

    标签用**固定的角度序号**而不是"当前列表里的第几个"：倾斜裁掉一个浅点后，
    剩下三个点的标签仍是它们原来那三个，暖启动不会串位。
    """
    if radius <= _EPS:
        # 退化端面（圆锥的顶点、圆台的 0 半径端）：没有圆周可取
        center = _local_to_world(body, (0.0, 0.0, center_local_z))
        return [(center, _TAG_SINGLE)]

    center = _local_to_world(body, (0.0, 0.0, center_local_z))
    e1, e2 = vec3.orthogonal_basis(axis)
    return [
        (vec3.add(center, _ring_offset(e1, e2, angle, radius)), base_tag + step)
        for step, angle in enumerate(_RING_ANGLES)
    ]


def _local_to_world(body, point) -> tuple[float, float, float]:
    """局部坐标点 → 世界坐标点。"""
    world = quat.quat_rotate_vector(body.orientation, point)
    return (world[0] + body.position[0], world[1] + body.position[1], world[2] + body.position[2])


def _radial_direction(direction, axis) -> tuple[float, float, float]:
    """方向里垂直于轴线的那一份（单位化）。"""
    projection = direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]
    radial = (
        direction[0] - axis[0] * projection,
        direction[1] - axis[1] * projection,
        direction[2] - axis[2] * projection,
    )
    length = math.sqrt(radial[0] ** 2 + radial[1] ** 2 + radial[2] ** 2)
    if length <= _EPS:
        # 方向与轴线平行时没有径向分量：任取一个正交方向，保证确定性
        e1, _e2 = vec3.orthogonal_basis(axis)
        return e1
    return (radial[0] / length, radial[1] / length, radial[2] / length)


def _ring_offset(e1, e2, angle, radius) -> tuple[float, float, float]:
    """圆周上角度 ``angle`` 处、半径 ``radius`` 的偏移向量。"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        (e1[0] * cos_a + e2[0] * sin_a) * radius,
        (e1[1] * cos_a + e2[1] * sin_a) * radius,
        (e1[2] * cos_a + e2[2] * sin_a) * radius,
    )
