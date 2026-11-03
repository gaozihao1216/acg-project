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
from dataclasses import dataclass

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
from ..mathutils import poly2d
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

#: 端面圆周采样：起始角固定 45°，其余等分。
#:
#: 45° 起步是为了避开 0°——立着的柱若在 x/y 轴上留下与坐标轴对齐的接触点，
#: 求解器的摩擦约束会出现方向退化。等分则让"4 点标签槽位"和"求交用的内接
#: 多边形"共用同一套角度定义：``segments=4`` 时正好是 45°/135°/225°/315°，
#: 与改造前逐位相同；求交路径按半径取更大的 ``segments``，多边形顶点仍是这套角。
def _ring_angles(segments: int):
    step = 2.0 * math.pi / segments
    return tuple(math.pi * 0.25 + index * step for index in range(segments))


#: 把圆盘离散成内接正多边形时容许的**内缩**（米），即 :func:`support_patch_world`
#: 的默认目标误差。边数按它反解：``N ≥ π / arccos(1 − ε/r)``。
#:
#: 0.5 mm 比 2 mm 的 slop 更小，窄搭接时不会因为"多边形比圆盘小一圈"而假性不相交。
_PATCH_MESH_ERROR = 5e-4

#: 自适应边数的上限。**这个上限会打破误差承诺**：半径超过约 6.6 m 时，要满足
#: 0.5 mm 内缩需要更多边，实际内缩就会大于 :data:`_PATCH_MESH_ERROR`
#: （实际值用 :func:`_ring_inradius_error` 核对）。取 256 是因为场景里最大的
#: 圆盘半径约 3 m，那个尺寸只需 173 边；再往上加，求交的 O(N·M) 代价涨得比
#: 收益快。早先这里的注释把方向写反了：边数随半径**增大**而增加。
_PATCH_MAX_SEGMENTS = 256


def _segments_for_radius(radius: float, mesh_error: float) -> int:
    """按**给定**内缩误差反解圆盘的离散边数。

    ``mesh_error`` 是实参而不是读模块常量：早先它写死在函数体里，传 1e-4 还是
    1e-3 结果完全一样，参数实际退化成了"要不要自适应"的布尔开关。

    返回值是**目标误差**下的边数；触到 :data:`_PATCH_MAX_SEGMENTS` 时实际内缩会
    超标，用 :func:`_ring_inradius_error` 核对，不要假定它一定达标。
    """
    if not mesh_error > 0.0:
        # NaN 也会走到这里：比较为假时一律拒绝
        raise ValueError(f"mesh_error 必须为正数，收到：{mesh_error!r}")
    if radius < 0.0:
        # 负半径没有几何意义；构造器已经拦了，这里拦住绕过构造器的调用
        raise ValueError(f"radius 不能为负，收到：{radius!r}")
    if radius <= _EPS:
        # 半径为 0 的退化端面：三角形就够表达"一个点"
        return 3
    ratio = max(-1.0, min(1.0, 1.0 - mesh_error / radius))
    count = math.ceil(math.pi / math.acos(ratio))
    return max(3, min(_PATCH_MAX_SEGMENTS, count))


def _ring_inradius_error(radius: float, count: int) -> float:
    """内接正 ``count`` 边形相对圆盘的**实际内缩**（正矢）``r·(1 − cos(π/N))``。

    把"目标误差"和"实际误差"分成两个可测量，边数是否够用就能被测试钉住，
    而不是靠注释里的承诺——触到边数上限时承诺本就不成立。
    """
    if radius <= _EPS or count < 3:
        return 0.0
    return radius * (1.0 - math.cos(math.pi / count))

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


#: 支撑 patch 的退化类别。求交按类别分派，不看点数——"两个点"既可能是线段，
#: 也可能是重合的重复点，靠点数猜会在退化情形下猜错。
PATCH_POINT = "point"
PATCH_SEGMENT = "segment"
PATCH_POLYGON = "polygon"


@dataclass
class SupportPatch:
    """沿某方向的**支撑几何**：只有形状，没有身份。

    与 :func:`feature_world_tagged` 的差别不是"点多点少"，而是**有没有身份**：
    暖启动要一个跨帧稳定的 ``feature_id``，那只有固定槽位的 4 点采样才有；
    切平面求交只关心几何，采样越密越好，点位与帧号无关。两者共用同一个返回类型
    时，防止误用的唯一手段是一句 docstring，而高密度标签还会与 ``_TAG_SIDE`` /
    ``_TAG_SINGLE`` 撞号——所以这里单独一个类型，它**没有** ``tag`` 字段。

    属性
    ----
    vertices : list[tuple[float, float, float]]
        世界坐标点列：``PATCH_POINT`` 1 个、``PATCH_SEGMENT`` 2 个、
        ``PATCH_POLYGON`` 至少 3 个且共面，按切平面上的角度序排列（凸包序）。
    kind : str
        ``PATCH_POINT`` / ``PATCH_SEGMENT`` / ``PATCH_POLYGON``。
    normal : tuple[float, float, float]
        支撑面的外法向（世界系、单位向量）；方向退化时是零向量。
    """

    vertices: list
    kind: str
    normal: tuple


def support_patch_world(
    body, direction, mesh_error: float = _PATCH_MESH_ERROR
) -> SupportPatch:
    """沿 ``direction`` 的支撑几何（世界坐标），供切平面求交使用。

    与 :func:`feature_world_tagged` 同源、不同用途：

    - 圆盘面按 ``mesh_error`` 由 :func:`_segments_for_radius` 反解边数（可达
      256 边），而不是固定 4 点——内接四边形比圆盘小 29%，窄搭接时会被算成
      "不相交"；
    - 返回 :class:`SupportPatch`，不带标签，不给暖启动留误用的余地。

    求交本身不在这里（见 ``docs/07`` 的 P0-1 计划）：本函数只回答"沿这个方向，
    哪一片几何是支撑集"。
    """
    geometry = body.geometry
    defect = math.sqrt(
        direction[0] * direction[0]
        + direction[1] * direction[1]
        + direction[2] * direction[2]
    )
    if defect <= _EPS:
        # 没有方向就没有法向，也没有"最凸的一片"：退回单点，法向留零向量
        return SupportPatch([support_world(body, direction)], PATCH_POINT, (0.0, 0.0, 0.0))

    unit = (direction[0] / defect, direction[1] / defect, direction[2] / defect)

    if isinstance(geometry, (SphereGeometry, EllipsoidGeometry)):
        # 严格凸：任意方向只有一个支撑点
        return SupportPatch([support_world(body, direction)], PATCH_POINT, unit)

    if isinstance(geometry, (BoxGeometry, ConvexHullGeometry)):
        return _patch_from_points(
            [point for point, _tag in _feature_branches(body, direction)], unit
        )

    axis = world_axis(body)
    cos_axis = unit[0] * axis[0] + unit[1] * axis[1] + unit[2] * axis[2]

    face = _cap_face(geometry, cos_axis)
    if face is not None:
        z_local, radius, sign = face
        count = _segments_for_radius(radius, mesh_error)
        vertices = _ring_points(body, z_local, radius, count)
        normal = axis if sign > 0.0 else (-axis[0], -axis[1], -axis[2])
        return SupportPatch(
            vertices, PATCH_POLYGON if len(vertices) >= 3 else PATCH_POINT, normal
        )

    if (isinstance(geometry, (CylinderGeometry, CapsuleGeometry))
            and abs(cos_axis) <= _SIDE_COS_TOL):
        # 侧面接触：两条母线的端点，是**一条线段**
        radial = _radial_direction(direction, axis)
        half_height = 0.5 * geometry.height
        vertices = [
            vec3.madd(
                vec3.madd(body.position, axis, sign * half_height), radial, geometry.radius
            )
            for sign in (-1.0, 1.0)
        ]
        return SupportPatch(vertices, PATCH_SEGMENT, radial)

    # 斜撑的锥/台/冠、倾斜圆柱的端面边缘：支撑集只是一个点
    return SupportPatch([support_world(body, direction)], PATCH_POINT, unit)


def _patch_from_points(points, normal) -> SupportPatch:
    """顶点集合 → :class:`SupportPatch`：按数量定类别，多边形先按切平面凸包重排。

    盒/凸包的顶点按 ``4*ix + 2*iy + iz`` 枚举，那是"角标顺序"，不是环序。求交
    需要环序（否则多边形的边会被连成对角线），所以这里投影到切平面、跑一次
    凸包拿回下标序列，再按序列取出三维点。顺带的收益是共线的中间点自动消失。
    """
    if len(points) <= 1:
        return SupportPatch(list(points), PATCH_POINT, normal)
    if len(points) == 2:
        return SupportPatch(list(points), PATCH_SEGMENT, normal)

    u, v = poly2d.tangent_basis(normal)
    flat = poly2d.project(points, u, v, _centroid(points))
    order = poly2d.convex_hull_2d(flat)
    return SupportPatch([points[index] for index in order], PATCH_POLYGON, normal)


def _centroid(points) -> tuple[float, float, float]:
    """点列的平均位置，用作切平面投影原点。"""
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _feature_branches(
    body, direction, tol: float = 1e-4
) -> list[tuple[tuple[float, float, float], int]]:
    """各几何的支撑特征分支（不校验标签，对外入口见 :func:`feature_world_tagged`）。

    这里只做**身份采样**：圆盘固定 4 点，标签是固定槽位，供暖启动跨帧匹配。
    需要高密度几何（切平面求交）的调用方走 :func:`support_patch_world`，它返回
    不带标签的 :class:`SupportPatch`——两条路径在**类型上**分开，而不是靠一句
    "这批标签别拿去匹配"的注释。

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
    | 圆盘四点（柱端面、锥/台/冠底面） | **体坐标槽位**：``_ring_angles(4)`` 固定的 4 个角，圆周起点由 :func:`_ring_basis` 固定在体坐标系 |
    | 严格凸体（球、椭球）与退化端面的单点 | 常量 ``_TAG_SINGLE``，只表示"本帧唯一的那个点" |

    圆盘这一行**曾经是**已知边界：圆周起点取自 ``vec3.orthogonal_basis(世界轴)``，
    而该基按"轴的最小分量"挑辅助坐标轴，轴方向跨越分界时不连续——竖直圆柱绕 y
    轴倾斜 ``1e-6`` rad，同一个 tag 的点就位移 0.424 m（盘面直径才 0.6 m）。现在
    基固定在体坐标系，同一 tag 恒对应同一个体坐标槽位（实测 6 个姿态下体坐标
    偏差 1.1e-16 m）。

    但"体坐标稳定"仍不等于完整跨帧持久：端面 / 侧面 / 单点三种模式会互相切换
    （标签整体换段），流形点数会变，P0-1 重建接触区域后采样点本身还会移动。
    真正的持久匹配要靠两侧局部锚点 + 有效性检查（见 `docs/07` §4.2）。

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

    if isinstance(geometry, (CylinderGeometry, CapsuleGeometry)):
        face = _cap_face(geometry, cos_axis)   # 胶囊没有平面端面，恒为 None
        if face is not None:
            # 端面接触：圆周上等角取 4 点，足以撑住"平放不翻"
            z_local, radius, sign = face
            return _face_ring_tagged(body, z_local, radius, 0 if sign < 0.0 else 4)
        if abs(cos_axis) <= _SIDE_COS_TOL:
            # 侧面接触：两条母线的端点
            radial = _radial_direction(direction, axis)
            half_height = 0.5 * geometry.height
            base_tag = _TAG_SIDE if isinstance(geometry, CylinderGeometry) else 0
            return [
                (
                    vec3.madd(
                        vec3.madd(body.position, axis, sign * half_height),
                        radial,
                        geometry.radius,
                    ),
                    base_tag + index,
                )
                for index, sign in enumerate((-1.0, 1.0))
            ]
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, EllipsoidGeometry):
        # 椭球是严格凸的：任意方向只有一个支撑点
        return [(support_world(body, direction), 0)]

    if isinstance(geometry, ConeGeometry):
        face = _cap_face(geometry, cos_axis)
        if face is not None:
            # 底面正对方向：整个底面圆盘等深，圆周取 4 点才撑得住"平放不翻"
            z_local, radius, sign = face
            return _face_ring_tagged(body, z_local, radius, 0 if sign < 0.0 else 4)
        # 其余方向（含顶点朝下）只有一个支撑点：圆锥是"圆盘 + 顶点"的凸包，
        # 支撑集要么是整块底面，要么是单点，不会出现线段
        return [(support_world(body, direction), _TAG_SINGLE)]

    if isinstance(geometry, (FrustumGeometry, SphericalCapGeometry)):
        face = _cap_face(geometry, cos_axis)
        if face is not None:
            # 圆台底面 0..3、顶面 4..7；球冠只有底面
            z_local, radius, sign = face
            return _face_ring_tagged(body, z_local, radius, 0 if sign < 0.0 else 4)
        # 斜着支撑时最大值落在某条端面圆周上，是**一个**点（圆台侧面是直纹面，
        # 但只有沿母线方向才会退化成线段，而那条线段的深度并不相同）
        return [(support_world(body, direction), _TAG_SINGLE)]

    raise TypeError(f"不支持支撑特征采样的几何类型：{type(geometry).__name__}")


def _ring_basis(body):
    """端面圆周采样的两个切向：**固定在体坐标系**里。

    为什么不能用 ``vec3.orthogonal_basis(世界轴)``：它按"轴的最小分量"挑辅助
    坐标轴，轴方向跨越该分界时基底不连续。实测竖直圆柱绕 y 轴倾斜 ``1e-6`` rad，
    同一个 tag 的点位移 **0.424 m**（盘面直径才 0.6 m）——标签没变，位置换了。

    固定在体坐标系之后，同一个 tag 恒对应同一个**体坐标**槽位；物体怎么转，
    槽位跟着转，这才是"几何特征"该有的行为。

    取 ``(0,1,0)`` / ``(-1,0,0)`` 而不是 ``(1,0,0)`` / ``(0,1,0)``，是因为
    ``orthogonal_basis((0,0,1))`` 恰好返回这两个方向——这样未旋转物体上的
    采样点与改造前逐位相同，把这次改动的影响限制在"带姿态的物体"上。
    """
    return (
        quat.quat_rotate_vector(body.orientation, (0.0, 1.0, 0.0)),
        quat.quat_rotate_vector(body.orientation, (-1.0, 0.0, 0.0)),
    )


def _cap_face(geometry, cos_axis):
    """方向正对某个**平面端面**时返回 ``(局部 z, 半径, 法向符号)``，否则 ``None``。

    把"哪个端面朝这个方向"收敛到一处：身份采样（4 点）与几何 patch（高密度）
    都从这里取端面，就不会出现两条路径对端面的判断不一致这种错误。

    符号 ``+1`` 表示端面朝局部 +Z（世界外法向 = 轴），``-1`` 表示朝 -Z。
    胶囊与椭球没有平面端面，恒返回 ``None``。
    """
    if isinstance(geometry, CylinderGeometry):
        if cos_axis >= _FACE_COS_TOL:
            return (0.5 * geometry.height, geometry.radius, 1.0)
        if cos_axis <= -_FACE_COS_TOL:
            return (-0.5 * geometry.height, geometry.radius, -1.0)
        return None
    if isinstance(geometry, ConeGeometry):
        # 顶点不是平面：圆锥只有底面
        if cos_axis <= -_FACE_COS_TOL:
            return (geometry._base_z, geometry.radius, -1.0)
        return None
    if isinstance(geometry, FrustumGeometry):
        if cos_axis >= _FACE_COS_TOL:
            return (geometry._top_z, geometry.top_radius, 1.0)
        if cos_axis <= -_FACE_COS_TOL:
            return (geometry._bottom_z, geometry.bottom_radius, -1.0)
        return None
    if isinstance(geometry, SphericalCapGeometry):
        # 球冠开口朝 +Z，平面只有底面
        if cos_axis <= -_FACE_COS_TOL:
            return (geometry._base_z, geometry._rim_radius, -1.0)
        return None
    return None


def _ring_points(body, center_local_z: float, radius: float, count: int):
    """端面圆周上 ``count`` 个等角点（世界坐标），**不带任何身份**。

    角度从 45° 起步等分：避开 0°，免得"立得笔直的柱"在 x/y 轴上留下与坐标轴
    方向重合的接触点——那种对齐会让求解器的摩擦约束出现方向退化。``count=4``
    时正好是 45°/135°/225°/315°，与改造前逐位相同。

    圆周起点由 :func:`_ring_basis` 固定在体坐标系，于是同一个角度序号恒对应同一个
    体坐标槽位；但这里的序号只是"第几个点"，要当身份用必须走
    :func:`_face_ring_tagged`（它固定 4 点，槽位才配得上暖启动的键）。
    """
    if radius <= _EPS:
        # 退化端面（圆锥的顶点、圆台的 0 半径端）：没有圆周可取
        return [_local_to_world(body, (0.0, 0.0, center_local_z))]

    center = _local_to_world(body, (0.0, 0.0, center_local_z))
    e1, e2 = _ring_basis(body)
    return [
        vec3.add(center, _ring_offset(e1, e2, angle, radius))
        for angle in _ring_angles(count)
    ]


def _face_ring_tagged(body, center_local_z: float, radius: float, base_tag: int):
    """身份路径：端面圆周固定 4 点，各带标签 ``base_tag + 角度序号``。

    端面是个圆盘，理论上无穷多点等深；4 个等分点足以让法向约束张成整个平面，
    物体不会绕单点翻转，倾斜时其余点深度更浅、会被逐点裁剪掉，自然退化成
    1–2 点接触。底面 ``base_tag=0``、顶面 ``base_tag=4``，环共占 ``0..7``，
    与 ``_TAG_SIDE`` / ``_TAG_SINGLE`` 不重叠。

    标签用**固定的角度序号**而不是"当前列表里的第几个"：倾斜裁掉一个浅点后，
    剩下三个点的标签仍是它们原来那三个，暖启动不会串位。

    注意：``narrowphase.collide_cylinder_ground`` **不用**这里——它按世界系求
    倾斜圆柱的真实最低点（``theta0``），那个最优性不能用固定槽位替代。
    """
    if radius <= _EPS:
        return [(_local_to_world(body, (0.0, 0.0, center_local_z)), _TAG_SINGLE)]
    return [
        (point, base_tag + step)
        for step, point in enumerate(_ring_points(body, center_local_z, radius, 4))
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
