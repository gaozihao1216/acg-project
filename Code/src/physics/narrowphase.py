# -*- coding: utf-8 -*-
"""窄相碰撞检测（narrowphase）。

当前实现范围
------------
================  ==========  ==================================
物体对            接触点数    实现
================  ==========  ==================================
球 - 球           1           解析（质心连线）
球 - 盒           1           解析（盒面最近点）
球 - 地面         1           解析（平面法向）
盒 - 盒           ≤4          SAT + 流形裁剪
盒/凸包 - 凸包    ≤4          广义 SAT
球 - 凸包         1           点到多面体最近点
盒/凸包 - 地面    ≤4          顶点解（每个陷入地面的顶点一点）
含柱 / 胶囊       1–4         **GJK + EPA**（支撑映射驱动）
柱 - 地面         ≤4          端面圆周解析最低点
胶囊 - 地面       ≤2          两端球解析最低点
================  ==========  ==================================

柱与胶囊的侧面是曲面：既没有有限 SAT 轴集合，也没有顶点可枚举，于是它们
（以及将来新增的任何凸体）统一走 ``gjk.collide``。要加一种几何，只需在
``support.py`` 里写它的支撑函数，不必在这里加分支——
"按几何对写函数"的组合爆炸到此为止。

任何"有几何但没有窄相"的组合都会在**加载阶段**被拒绝，不会静默穿模。

为什么球和盒的点数不一样
------------------------
球是**曲率处处相同**的凸体，任意方向只能提供 1 个接触点；
盒是**平面拼接**的，面贴面时理想情况下是整块面积的接触，
离散成有限个点才能表达"支撑多边形"，否则单个接触点会让盒子绕该点
自由翻转——表现是堆叠时盒子乱转、抖动。

这就是盒-地面必须返回 4 个点的原因，也是盒-盒那一步的核心难点。

法向约定
--------
``ContactPoint.normal`` 一律**从 ``body_a`` 指向 ``body_b``**。
求解器据此计算 ``v_rel = v_b - v_a`` 并沿法向施加冲量，
所以只要方向约定统一，谁做 a、谁做 b 都不影响结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..mathutils import quaternion as quat
from ..mathutils import vec3
from ..scene.scene_spec import BodyPhysicsSpec
from .body import RigidBody
from .contact import GROUND_BODY, ContactManifold, ContactPoint, merge_pair_materials
from ..geom.geometry import (
    BoxGeometry,
    CompoundGeometry,
    ConvexHullGeometry,
    GeometrySpec,
)
from .body import iter_parts
from .convex_sat import (
    Polytope,
    closest_point_on_polytope,
    collide_polytopes,
    polytope_from_box,
    polytope_from_hull,
)
from . import capabilities
from . import gjk as gjk_module
from . import support as support_module
from .sat import collide_box_box

#: 盒-地面接触点的数量上限。
#: 盒子有 8 个顶点，陷入地面的理论上最多 8 个；但对"支撑多边形"而言
#: 4 个点已经足够表达任意姿态下的接触面，多出来的点只会增加求解器负担、
#: 且彼此冗余（法向完全相同），因此按穿透深度取最深的 4 个。
MAX_BOX_GROUND_POINTS = 4

#: 复合体-地面的接触点数上限（按深度挑最深的）。见 collide_compound_ground。
_MAX_COMPOUND_GROUND_POINTS = 16


@dataclass
class GroundPlane:
    """静态水平地面（法向 +Z，无限大）。

    物理地面与场景里那块"视觉地面"是两回事：视觉地面只用于渲染，
    这里的地面只参与碰撞，两者的 ``z`` 通常相同，但互不依赖。
    """

    z: float = 0.0
    friction: float = 0.55
    restitution: float = 0.12

    @property
    def normal(self) -> tuple[float, float, float]:
        """地面法向恒为 +Z（朝上，指向地面外侧）。"""
        return (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 几何支持性
# ---------------------------------------------------------------------------

def is_supported(geometry: GeometrySpec | None) -> bool:
    """判断单个几何体是否已有窄相实现。

    判据来自 ``capabilities.SUPPORTED_KINDS``，**不在这里另列类型元组**：
    两处各写一遍，新增几何时总有一处会忘，而忘掉的那一处会静默地"不支持"
    一个已经能跑的几何。
    """
    return capabilities.kind_of(geometry) in capabilities.SUPPORTED_KINDS


def supported_pair(geometry_a: GeometrySpec | None, geometry_b: GeometrySpec | None) -> bool:
    """判断一对几何体之间是否已有窄相实现。

    单独判断**物体对**而不只是单个物体：两种几何各自都支持，不代表它们的组合
    就一定支持。只查单个体会让"组合未实现"的场景静默地不产生任何接触——
    物体互相穿过却毫无反应，是最难排查的一类症状。

    判据同样来自能力矩阵（``capabilities.NARROWPHASE_IMPL``），而不是
    "两次 ``is_supported``"：后者假定"单个支持 ⇒ 组合支持"，这个假定在
    球与盒并存的第一天就不成立过（曾经盒-盒未实现，但两者各自都支持）。
    """
    kind_a = capabilities.kind_of(geometry_a)
    kind_b = capabilities.kind_of(geometry_b)
    if kind_a is None or kind_b is None:
        return False
    return capabilities.has_narrowphase(kind_a, kind_b)


# ---------------------------------------------------------------------------
# 局部坐标与变换辅助
# ---------------------------------------------------------------------------

def _box_half_extents(box: RigidBody) -> tuple[float, float, float]:
    """盒子的半尺寸（``size`` 是全长，半尺寸才是碰撞要用的量）。"""
    size = box.geometry.size
    return (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])


def _world_to_local(body: RigidBody, point) -> tuple[float, float, float]:
    """世界坐标点 → 该物体局部坐标。

    盒子的一切计算都在局部坐标里做（盒子绕自身主轴是对角惯量、
    面对齐坐标轴），算完再转回世界。这比在世界坐标里做 OBB 判断简单得多。
    """
    delta = vec3.sub(point, body.position)
    return quat.quat_rotate_vector(quat.quat_conjugate(body.orientation), delta)


def _local_to_world(body: RigidBody, point) -> tuple[float, float, float]:
    """该物体局部坐标 → 世界坐标。"""
    return vec3.add(body.position, quat.quat_rotate_vector(body.orientation, point))


# ---------------------------------------------------------------------------
# 球 - 球
# ---------------------------------------------------------------------------

def collide_sphere_sphere(
    body_a: RigidBody, body_b: RigidBody
) -> ContactPoint | None:
    """两个球体的碰撞检测。

    判据：球心距小于半径之和。法向取**从 A 指向 B** 的单位向量。
    接触点取两个表面沿法向的**中点**，这样穿透时法向和位置都不会偏向任何一方。
    """
    radius_a = body_a.geometry.radius
    radius_b = body_b.geometry.radius

    delta = vec3.sub(body_b.position, body_a.position)
    distance = vec3.length(delta)
    radius_sum = radius_a + radius_b

    if distance >= radius_sum:
        return None

    if distance < 1e-9:
        # 两个球心完全重合：方向没有定义。取任意固定方向（+X），
        # 虽然任意，但至少是确定性的，不会每帧抖动。
        normal = (1.0, 0.0, 0.0)
        distance = 1e-9
    else:
        normal = vec3.scale(delta, 1.0 / distance)

    penetration = radius_sum - distance

    contact_position = vec3.madd(
        body_a.position, normal, radius_a - 0.5 * penetration
    )

    return ContactPoint(
        position=contact_position,
        normal=normal,
        penetration=penetration,
    )


# ---------------------------------------------------------------------------
# 球 - 地面
# ---------------------------------------------------------------------------

def collide_sphere_ground(
    body: RigidBody, ground: GroundPlane
) -> ContactPoint | None:
    """球体与水平地面的碰撞检测。

    地面被当作 body_a（``GROUND_BODY``），因此法向恒为 +Z（从地面指向球）。
    """
    radius = body.geometry.radius

    height = body.position[2] - ground.z
    if height >= radius:
        return None

    penetration = radius - height

    contact_z = ground.z + 0.5 * (height - radius)
    return ContactPoint(
        position=(body.position[0], body.position[1], contact_z),
        normal=(0.0, 0.0, 1.0),
        penetration=penetration,
    )


# ---------------------------------------------------------------------------
# 球 - 盒
# ---------------------------------------------------------------------------

def collide_sphere_box(sphere: RigidBody, box: RigidBody) -> ContactPoint | None:
    """球与有向包围盒（OBB）的碰撞检测。

    算法：把球心转到盒子局部坐标，**夹紧**到盒子范围内得到最近点，
    两者的差就是碰撞方向。盒子是凸体，最近点必然在这个夹紧结果上，
    因此这是精确解而不是近似。

    两种情形
    --------
    1. 球心在盒外：``|球心 - 最近点| < r`` 才算碰撞，
       法向是"最近点 → 球心"的方向，穿透 = ``r - 距离``。
    2. 球心在盒内（深度穿透）：最近点退化到球心自身，方向没有定义。
       此时取**距离最近的盒面**作为脱离方向，穿透 = ``r + 到该面的距离``。

    返回
    ----
    ContactPoint | None
        法向**从盒子指向球**（即 ``body_a=box, body_b=sphere`` 的约定）。
        不做碰撞时返回 None。
    """
    radius = sphere.geometry.radius
    half = _box_half_extents(box)

    # 球心在盒子局部坐标中的位置
    local = _world_to_local(box, sphere.position)

    # 夹紧到盒子范围内 = 盒面上离球心最近的点（局部坐标）
    closest = (
        max(-half[0], min(half[0], local[0])),
        max(-half[1], min(half[1], local[1])),
        max(-half[2], min(half[2], local[2])),
    )

    delta = vec3.sub(local, closest)
    distance_sq = vec3.length_squared(delta)

    if distance_sq > 1e-18:
        # --- 情形 1：球心在盒外 ---
        distance = math.sqrt(distance_sq)
        if distance >= radius:
            return None
        normal_local = vec3.scale(delta, 1.0 / distance)
        penetration = radius - distance
    else:
        # --- 情形 2：球心在盒内 ---
        # 各轴到对应面的距离，取最小者作为脱离方向
        gaps = [half[i] - abs(local[i]) for i in range(3)]
        axis = min(range(3), key=lambda i: gaps[i])
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        axis_normal = [0.0, 0.0, 0.0]
        axis_normal[axis] = sign
        normal_local = tuple(axis_normal)
        penetration = radius + gaps[axis]

    # 法向转回世界坐标（从盒指向球）
    normal = quat.quat_rotate_vector(box.orientation, normal_local)

    # 接触点：球面上沿法向"往回退 (r - 穿透/2)"的位置，
    # 也就是重叠区的中点——与球-球的取法保持一致。
    contact_position = vec3.madd(sphere.position, normal, -(radius - 0.5 * penetration))

    return ContactPoint(
        position=contact_position,
        normal=normal,
        penetration=penetration,
    )


# ---------------------------------------------------------------------------
# 盒 - 地面
# ---------------------------------------------------------------------------

def collide_box_ground(
    box: RigidBody, ground: GroundPlane
) -> list[ContactPoint]:
    """盒子与水平地面的碰撞检测，返回最多 4 个接触点。

    做法：把 8 个顶点变换到世界坐标，凡是低于地面的都产生一个接触。
    面贴面时恰好有 4 个顶点在地面以下，于是得到 4 点流形——这正是盒子
    能稳定平放、而不是像球一样绕单点翻转的原因。

    为什么必须多点
    --------------
    单个接触点只能约束 1 个方向的平动，盒子绕该点的旋转完全自由；
    求解器虽然会靠摩擦勉强撑住，但表现为持续抖动与缓慢倾斜。
    4 个点分布在一个面上，法向约束天然阻止了翻转。

    特征 ID 的约定
    --------------
    每个点用自己的**顶点序号** ``4*ix + 2*iy + iz`` 作 ``feature_id``，
    与 ``support.feature_world_tagged`` 对盒顶点的编号同源。

    这一点曾经写反过：早先四个点共用 ``feature_id=0``，理由是"这四个点法向相同，
    一个标量就够"。实测证明不成立——偏心负载下四角累积冲量实测为
    0.011 / 0.015 / 0.030 / 0.026（相差近 3 倍），共用一个键意味着后写入的覆盖
    先写入的，下一帧再把这个值施给四个角：净冲量偏大、分布全错。

    返回
    ----
    list[ContactPoint]
        法向恒为 +Z（地面指向盒子），与 ``body_a=GROUND_BODY`` 的约定一致。
        未接触时返回空列表。
    """
    half = _box_half_extents(box)
    points: list[ContactPoint] = []

    for index_x, sign_x in enumerate((-1.0, 1.0)):
        for index_y, sign_y in enumerate((-1.0, 1.0)):
            for index_z, sign_z in enumerate((-1.0, 1.0)):
                local = (sign_x * half[0], sign_y * half[1], sign_z * half[2])
                world = _local_to_world(box, local)

                if world[2] >= ground.z:
                    continue

                points.append(
                    ContactPoint(
                        # 接触点取"顶点与地面投影"的中点，与球-地面的取法一致
                        position=(world[0], world[1], 0.5 * (world[2] + ground.z)),
                        normal=(0.0, 0.0, 1.0),
                        penetration=ground.z - world[2],
                        feature_id=4 * index_x + 2 * index_y + index_z,
                    )
                )

    if len(points) > MAX_BOX_GROUND_POINTS:
        # 取最深的若干个：它们才是真正承担支撑的点
        points.sort(key=lambda p: p.penetration, reverse=True)
        del points[MAX_BOX_GROUND_POINTS:]

    return points


# ---------------------------------------------------------------------------
# 柱 / 胶囊 - 地面
# ---------------------------------------------------------------------------

def collide_cylinder_ground(
    body: RigidBody, ground: GroundPlane
) -> list[ContactPoint]:
    """圆柱与水平地面：解析求出两条端面圆周上的最低点，最多返回 4 个接触点。

    为什么不能像盒那样"顶点下落"
    ----------------------------
    圆柱没有顶点——它的最低点落在端面圆周上，而圆周上最低的那个角度是**可解析
    求出**的：

        ``z(θ) = z_c + s·(h/2)·n_z + r·(cosθ·e1_z + sinθ·e2_z)``

    对 ``θ`` 求极小得 ``θ* = atan2(-e2_z, -e1_z)``（``n`` 是轴、``e1/e2`` 是
    端面内的两个正交单位向量）。``θ*`` 那一点就是**精确**最低点，深度不带任何
    采样误差。

    再以 ``θ*`` 为起点每 90° 取一个点，是为了"平放"时的多点流形：立着的圆柱
    四个点深度相同，能撑住不翻；倾斜时其余三点深度更浅（甚至为负被裁掉），
    自然退化成 1–2 点接触。深度逐点按投影算，不搞"统一深度"。
    """
    geometry = body.geometry
    radius = geometry.radius
    half_height = 0.5 * geometry.height

    axis = support_module.world_axis(body)
    e1, e2 = vec3.orthogonal_basis(axis)

    # 圆周上 z 最低的角度
    if math.hypot(e1[2], e2[2]) > 1e-12:
        theta0 = math.atan2(-e2[2], -e1[2])
    else:
        theta0 = 0.0   # 圆柱立得笔直：整圈等高，角度任取

    points: list[ContactPoint] = []
    for cap_index, cap_sign in enumerate((-1.0, 1.0)):
        cap_center = vec3.madd(body.position, axis, cap_sign * half_height)
        for step in range(4):
            angle = theta0 + step * 0.5 * math.pi
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            world = (
                cap_center[0] + (e1[0] * cos_a + e2[0] * sin_a) * radius,
                cap_center[1] + (e1[1] * cos_a + e2[1] * sin_a) * radius,
                cap_center[2] + (e1[2] * cos_a + e2[2] * sin_a) * radius,
            )
            if world[2] >= ground.z:
                continue
            points.append(
                ContactPoint(
                    position=(world[0], world[1], 0.5 * (world[2] + ground.z)),
                    normal=(0.0, 0.0, 1.0),
                    penetration=ground.z - world[2],
                    # 身份 = 端面序号 × 4 + 圆周序号，与 feature_world_tagged 同源
                    feature_id=cap_index * 4 + step,
                )
            )

    if len(points) > MAX_BOX_GROUND_POINTS:
        points.sort(key=lambda item: item.penetration, reverse=True)
        del points[MAX_BOX_GROUND_POINTS:]
    return points


def collide_capsule_ground(
    body: RigidBody, ground: GroundPlane
) -> list[ContactPoint]:
    """胶囊与水平地面：两个端球的球底各一个接触点（横放时是 2 点了）。

    胶囊 = 线段 ⊕ 球，最低点必然出现在**线段的某个端点下方 r 处**
    （线段本身是直线，z 的最小值只在端点取到）。所以这里没有采样误差：
    端点球的最低点就是胶囊表面的最低点，是精确解。
    """
    geometry = body.geometry
    radius = geometry.radius
    half_height = 0.5 * geometry.height
    axis = support_module.world_axis(body)

    points: list[ContactPoint] = []
    for index, sign in enumerate((-1.0, 1.0)):
        endpoint = vec3.madd(body.position, axis, sign * half_height)
        lowest_z = endpoint[2] - radius
        if lowest_z >= ground.z:
            continue
        points.append(
            ContactPoint(
                position=(endpoint[0], endpoint[1], 0.5 * (lowest_z + ground.z)),
                normal=(0.0, 0.0, 1.0),
                penetration=ground.z - lowest_z,
                feature_id=index,
            )
        )
    return points


def collide_convex_ground(body, ground: GroundPlane) -> list[ContactPoint]:
    """通用凸体-地面：取沿 ``-Z`` 的**支撑特征**，裁掉没入地面的点。

    这一条公式覆盖了"没有专属实现"的所有凸体——球、圆锥、圆台、椭球、球冠，
    以及将来任何实现了 ``support()`` 的几何。

    为什么是"支撑特征"而不是"支撑点"
    --------------------------------
    只取 ``support(-Z)`` 会得到**一个**点。球可以用（严格凸，本来就只碰一点），
    但底面是平圆盘的东西不行：立着的圆锥若只在地面圆盘上取一个点，法向约束只
    挡住一个方向，物体会绕该点翻转——正是盒-地面当初必须给 4 个点的原因。
    ``feature_world`` 因此返回"沿该方向同样远的一整片"：圆盘给 4 个圆周点，
    严格凸体给 1 个点。

    深度逐点按投影算，不做"统一深度"：倾斜时其余点更浅甚至为负，会被裁掉，
    流形自然退化，不需要额外写退化逻辑。

    球走这条路与解析式 ``collide_sphere_ground`` **逐位相同**：接触点取
    "表面点与地面投影的中点"，与解析式的 ``ground.z + 0.5(h - r)`` 是同一个式子。
    """
    points: list[ContactPoint] = []
    for world, tag in support_module.feature_world_tagged(body, (0.0, 0.0, -1.0)):
        if world[2] >= ground.z:
            continue
        points.append(
            ContactPoint(
                position=(world[0], world[1], 0.5 * (world[2] + ground.z)),
                normal=(0.0, 0.0, 1.0),
                penetration=ground.z - world[2],
                feature_id=tag,
            )
        )

    if len(points) > MAX_BOX_GROUND_POINTS:
        points.sort(key=lambda item: item.penetration, reverse=True)
        del points[MAX_BOX_GROUND_POINTS:]
    return points


def _sphere_ground_points(body, ground: GroundPlane) -> list[ContactPoint]:
    """``collide_sphere_ground`` 的列表化包装，便于和分派表统一返回类型。"""
    single = collide_sphere_ground(body, ground)
    return [single] if single is not None else []


def _part_ground_points(view, ground: GroundPlane) -> list[ContactPoint]:
    """单个部件与地面的接触点。``view`` 可以是刚体，也可以是复合体的部件视图。

    未登记的几何（例如还没接入能力矩阵的新图元）直接走通用解；若它连支撑映射
    都没有，``feature_world`` 会带着类型名抛 ``TypeError``，而不是静默返回空。
    """
    kind = capabilities.kind_of(view.geometry)
    handler = _GROUND_DISPATCH.get(kind)
    if handler is not None:
        return handler(view, ground)
    return collide_convex_ground(view, ground)


def collide_compound_ground(
    body: RigidBody, ground: GroundPlane
) -> list[ContactPoint]:
    """复合体与地面：逐部件求交，特征 ID 按部件加偏移。

    点数上限
    --------
    部件一多，地面接触点会线性增长，而求解器是"迭代次数 × 接触点数"。
    这里按深度保留最深的 ``_MAX_COMPOUND_GROUND_POINTS`` 个：真正撑住物体的
    就是这些点，浅的那几个对姿态几乎没有贡献。排序键用
    ``(-深度, 特征 ID)``，保证结果逐帧确定，不会因为字典顺序抖动。
    """
    points: list[ContactPoint] = []
    for part_index, view in iter_parts(body):
        offset = part_index * _PART_FEATURE_STRIDE
        for point in _part_ground_points(view, ground):
            points.append(
                ContactPoint(
                    position=point.position,
                    normal=point.normal,
                    penetration=point.penetration,
                    feature_id=offset + point.feature_id,
                )
            )

    if len(points) > _MAX_COMPOUND_GROUND_POINTS:
        points.sort(key=lambda item: (-item.penetration, item.feature_id))
        del points[_MAX_COMPOUND_GROUND_POINTS:]
    return points


# ---------------------------------------------------------------------------
# 通用凸体（GJK/EPA）
# ---------------------------------------------------------------------------

def collide_gjk(body_a: RigidBody, body_b: RigidBody) -> list[ContactPoint]:
    """任意凸体对的通用窄相：GJK 求分离距离、EPA 求穿透，返回 1–4 点流形。

    覆盖柱、胶囊参与的 9 种组合，以及将来新增的所有几何体。球-球、盒-盒等
    已有解析解的组合**不走这里**——它们的闭式解更精确也更快。
    """
    return gjk_module.collide(body_a, body_b)


# ---------------------------------------------------------------------------
# 宽阶段 + 分派
# ---------------------------------------------------------------------------

def _local_polytope(geometry: ConvexHullGeometry) -> Polytope:
    """凸包在其**局部（主轴）坐标系**下的多面体。

    顶点在构造 ``ConvexHullGeometry`` 时已归一化（质心在原点、主轴对齐），
    所以这里不需要任何变换，直接复用即可。这也意味着这个函数很便宜，
    但每次碰撞都重建对象仍有开销——顶点表本身不复制，只包一层。
    """
    return Polytope(
        vertices=geometry.vertices,
        faces=geometry.faces,
        normals=geometry.face_normals,
        edges=geometry.edges,
        center=(0.0, 0.0, 0.0),
    )


def collide_sphere_hull(
    sphere: RigidBody, hull: RigidBody
) -> ContactPoint | None:
    """球-凸包：把球退化成"点 + 半径"，求点到多面体表面的最近点。

    返回的法向是**凸包指向球**，与 ``collide_sphere_box`` 的约定一致，
    分派时按 a/b 顺序决定是否取反。

    为什么用"最近点"而不是 SAT
    -------------------------
    球是光滑体，没有面也没有棱，SAT 的轴集合对它不成立（它有无穷多个支撑方向）。
    但球的接触完全由"球心到表面的最近距离"决定，所以直接用最近点即可，精确且更便宜。
    """
    geometry = hull.geometry
    poly = _local_polytope(geometry)

    # 球心转到凸包的局部坐标
    inverse = quat.quat_conjugate(hull.orientation)
    relative = vec3.sub(sphere.position, hull.position)
    local_center = quat.quat_rotate_vector(inverse, relative)

    closest_local, distance, normal_local = closest_point_on_polytope(
        poly, local_center
    )

    radius = sphere.geometry.radius
    penetration = radius - distance
    if penetration <= 0.0:
        return None

    normal_world = quat.quat_rotate_vector(hull.orientation, normal_local)
    closest_world = vec3.add(
        hull.position, quat.quat_rotate_vector(hull.orientation, closest_local)
    )
    # 接触点取两表面之间的中点，避免贴在某一侧导致抖动
    position = vec3.add(closest_world, vec3.scale(normal_world, 0.5 * penetration))

    return ContactPoint(
        position=position,
        normal=normal_world,
        penetration=penetration,
    )


def collide_hull_ground(
    body: RigidBody, ground: GroundPlane
) -> list[ContactPoint]:
    """凸包与水平地面：所有低于地面的顶点各产生一个接触，取最深的 4 个。

    与 ``collide_box_ground`` 完全同构——盒子只是"8 个顶点的凸包"。
    多点在这里同样关键：3 个以上不共线的接触点是阻止翻转的必要条件。
    """
    geometry = body.geometry
    points: list[ContactPoint] = []

    for local in geometry.vertices:
        world = _local_to_world(body, local)
        if world[2] >= ground.z:
            continue
        points.append(
            ContactPoint(
                position=(world[0], world[1], 0.5 * (world[2] + ground.z)),
                normal=(0.0, 0.0, 1.0),
                penetration=ground.z - world[2],
            )
        )

    if len(points) > MAX_BOX_GROUND_POINTS:
        points.sort(key=lambda p: p.penetration, reverse=True)
        del points[MAX_BOX_GROUND_POINTS:]

    return points


#: 地面接触分派表：几何种类 -> 处理函数（统一返回 ``list[ContactPoint]``）。
#:
#: 表里只放**有专属精确解**的几何：多面体族用顶点解、柱与胶囊用解析的圆周/端球
#: 最低点解，它们的 TOI 也都是精确或单调可证的。其余一律走
#: :func:`collide_convex_ground` 的支撑特征解——于是"加一种凸体"依然不需要写
#: 任何地面代码，这与物体对分派表的策略一致。
#:
#: 放在所有地面函数**之后**：本表在导入时求值，函数对象必须已经存在。
_GROUND_DISPATCH = {
    "sphere": _sphere_ground_points,
    "box": collide_box_ground,
    "hull": collide_hull_ground,
    "cylinder": collide_cylinder_ground,
    "capsule": collide_capsule_ground,
}


# ---------------------------------------------------------------------------
# 物体对分派表
# ---------------------------------------------------------------------------
#
# 以前这里是一串 isinstance 分支，于是"支持哪一对几何"写在两个地方：分派代码
# 里一份，``capabilities.NARROWPHASE_IMPL`` 里一份。两份必须同步，而没人会记得
# 同步——加一种几何时漏掉一处，症状是"矩阵说支持、实际静默不产生接触"，也就是
# 两个物体互相穿过。
#
# 现在分派表是唯一实现，能力矩阵是唯一声明，两者由 ``test_capabilities`` 强制
# 对齐：表里出现的组合，其函数名必须与矩阵声明的名字一致；矩阵里有而表里没有
# 的组合，必须落在 GJK/EPA 兜底上（矩阵也正是这么写的）。
#
# 表里只放**值得特判**的组合——闭式解或 SAT 能给出比 GJK 更准或更便宜的结果。
# 其余一律兜底，于是新增一种凸体只要实现 ``support()``，不用碰这个文件。


def _sphere_sphere_pair(body_a, body_b) -> list[ContactPoint]:
    point = collide_sphere_sphere(body_a, body_b)
    return [point] if point is not None else []


def _flipped(point: ContactPoint) -> ContactPoint:
    """把接触点法向取反，用于"天然返回 b→a"的函数。

    ``feature_id`` 一并带过去：暖启动的匹配键，丢了它会让缓存永远命不中。
    """
    return ContactPoint(
        position=point.position,
        normal=vec3.scale(point.normal, -1.0),
        penetration=point.penetration,
        feature_id=point.feature_id,
    )


def _box_sphere_pair(body_a, body_b) -> list[ContactPoint]:
    """a=盒、b=球。``collide_sphere_box`` 返回"盒→球"，正好是 a→b。"""
    point = collide_sphere_box(body_b, body_a)
    return [point] if point is not None else []


def _sphere_box_pair(body_a, body_b) -> list[ContactPoint]:
    """a=球、b=盒。返回值天然是"盒→球" = b→a，取反。"""
    point = collide_sphere_box(body_a, body_b)
    return [_flipped(point)] if point is not None else []


def _hull_sphere_pair(body_a, body_b) -> list[ContactPoint]:
    """a=凸包、b=球：``collide_sphere_hull`` 返回 hull→sphere，正是 a→b。"""
    point = collide_sphere_hull(body_b, body_a)
    return [point] if point is not None else []


def _sphere_hull_pair(body_a, body_b) -> list[ContactPoint]:
    """a=球、b=凸包：返回值是 hull→sphere = b→a，取反。"""
    point = collide_sphere_hull(body_a, body_b)
    return [_flipped(point)] if point is not None else []


#: 几何类型 -> 多面体构造器。用 ``type()`` 精确匹配而不是 ``isinstance``：
#: 几何类都是叶子类，精确匹配更快，也避免子类被悄悄当成父类处理。
_POLYTOPE_BUILDERS = {
    BoxGeometry: lambda geometry, body: polytope_from_box(
        geometry.size, body.position, body.orientation
    ),
    ConvexHullGeometry: lambda geometry, body: polytope_from_hull(
        geometry, body.position, body.orientation
    ),
}


def _polytope_of(body) -> Polytope:
    """把盒/凸包转成 SAT 用的多面体表示。"""
    builder = _POLYTOPE_BUILDERS.get(type(body.geometry))
    if builder is None:
        raise TypeError(
            f"{type(body.geometry).__name__} 不能转为多面体；"
            f"只有 {tuple(cls.__name__ for cls in _POLYTOPE_BUILDERS)} 走 SAT 分派"
        )
    return builder(body.geometry, body)


def _polytope_pair(body_a, body_b) -> list[ContactPoint]:
    """盒/凸包之间的广义 SAT + 流形裁剪，法向天然是 a→b。"""
    return collide_polytopes(_polytope_of(body_a), _polytope_of(body_b))


#: (几何 A, 几何 B) -> (能力矩阵里声明的函数名, 实际分派函数)。
#:
#: 键是**有序对**：非对称组合（球-盒）写两条，各自负责自己的法向方向。
#: 不在分派时做一次"谁是球"的换序判断——那等于把 isinstance 换个马甲。
_PAIR_DISPATCH = {
    ("sphere", "sphere"): ("collide_sphere_sphere", _sphere_sphere_pair),
    ("sphere", "box"): ("collide_sphere_box", _sphere_box_pair),
    ("box", "sphere"): ("collide_sphere_box", _box_sphere_pair),
    ("sphere", "hull"): ("collide_sphere_hull", _sphere_hull_pair),
    ("hull", "sphere"): ("collide_sphere_hull", _hull_sphere_pair),
    ("box", "box"): ("collide_box_box", collide_box_box),
    ("box", "hull"): ("collide_polytopes", _polytope_pair),
    ("hull", "box"): ("collide_polytopes", _polytope_pair),
    ("hull", "hull"): ("collide_polytopes", _polytope_pair),
}


def _collide_points(body_a, body_b) -> list[ContactPoint]:
    """查表分派；表里没有的组合一律走 GJK/EPA 兜底。

    参数是"像刚体的东西"：真实 ``RigidBody``，或复合体展开出的 ``PartView``。
    因此这里只允许读 ``geometry`` / ``position`` / ``orientation``。

    法向一律**从 body_a 指向 body_b**：表里每个非对称组合写了两条，各自处理
    取反，所以这里没有任何方向判断。

    兜底的意义
    ----------
    曲面体（柱、胶囊）没有有限 SAT 轴集合，圆锥、圆台、椭球、球冠同理；它们
    全部落在 ``collide_gjk`` 上。于是"加一种凸体"的代价从 O(已有几何数) 个
    窄相函数降到 0——只要它有 ``support()``。
    """
    kind_a = capabilities.kind_of(body_a.geometry)
    kind_b = capabilities.kind_of(body_b.geometry)
    if kind_a is None or kind_b is None:
        raise TypeError(
            f"几何类型未登记到能力矩阵，无法分派窄相："
            f"{type(body_a.geometry).__name__} / {type(body_b.geometry).__name__}"
        )

    entry = _PAIR_DISPATCH.get((kind_a, kind_b))
    if entry is not None:
        return entry[1](body_a, body_b)

    return collide_gjk(body_a, body_b)


#: 复合体特征 ID 的命名空间步长。各部件的 ID 都小于它（凸包最大约 3e5），
#: 于是 ``部件对编号 × 步长 + 局部 ID`` 在所有部件对之间都不会撞号。
_PART_FEATURE_STRIDE = 1_000_000


def collide_compound(index_a, body_a, index_b, body_b) -> list[ContactPoint]:
    """复合体参与的一对刚体：逐部件求交，把接触点并到一起。

    为什么要给特征 ID 加偏移
    ----------------------
    暖启动的键是 ``(body_a, body_b, feature_id)``。复合体的两个不同部件
    完全可能产生相同的局部 ID（比如两个盒子部件都给出 feature_id=1），
    不动偏移就会把冲量喂到另一个部件的接触点上——症状是静置的物体莫名抖动。

    为什么结果仍然只有一个流形
    ------------------------
    求解器按流形遍历，同一对刚体的接触点是否拆成两个流形在数学上等价；
    合成一个可以少一次材质合成，也避免"同一对物体出现两条记录"这种容易
    写错下游代码的形态。点数不限，求解器本来就是逐点迭代。
    """
    parts_a = iter_parts(body_a)
    parts_b = iter_parts(body_b)

    points: list[ContactPoint] = []
    for part_index_a, view_a in parts_a:
        for part_index_b, view_b in parts_b:
            local_points = _collide_points(view_a, view_b)
            if not local_points:
                continue
            offset = (part_index_a * len(parts_b) + part_index_b) * _PART_FEATURE_STRIDE
            for point in local_points:
                points.append(
                    ContactPoint(
                        position=point.position,
                        normal=point.normal,
                        penetration=point.penetration,
                        feature_id=offset + point.feature_id,
                    )
                )
    return points


def _collide_pair(
    index_a: int,
    body_a: RigidBody,
    index_b: int,
    body_b: RigidBody,
) -> ContactManifold | None:
    """检测一对刚体，返回带材质参数的流形；无接触返回 ``None``。

    复合体在这里展开成部件对，其余几何按类型分派（见 ``_collide_points``）。
    """
    if isinstance(body_a.geometry, CompoundGeometry) or isinstance(
        body_b.geometry, CompoundGeometry
    ):
        points = collide_compound(index_a, body_a, index_b, body_b)
    else:
        points = _collide_points(body_a, body_b)

    if not points:
        return None

    restitution, friction = merge_pair_materials(
        body_a.restitution,
        body_b.restitution,
        body_a.friction,
        body_b.friction,
    )
    return ContactManifold(
        body_a=index_a,
        body_b=index_b,
        points=points,
        friction=friction,
        restitution=restitution,
    )


def detect_contacts(
    bodies: list[RigidBody],
    ground: GroundPlane | None,
    candidate_pairs: list[tuple[int, int]] | None = None,
) -> list[ContactManifold]:
    """检测整个场景中的所有接触，返回接触流形列表。

    参数
    ----
    bodies          : 刚体列表
    ground          : 地面；传 None 表示没有地面（自由空间）
    candidate_pairs : 宽阶段给出的候选物体对；None 表示**暴力枚举所有对**

    ``candidate_pairs`` 的意义
    -------------------------
    传 None 时退化为 O(N²) 暴力枚举，这是宽阶段出现之前的行为，
    保留下来用于对照实验（``BruteForceBroadphase``）。

    传入候选对后，只有候选对会进入窄相。关键细节：**输出顺序必须与暴力枚举一致**。
    因此这里把候选对按第一个下标分组、组内升序，再按 ``i`` 升序遍历——
    于是"地面接触 + 物体对"的交错顺序与暴力路径完全相同，结果逐位一致。
    如果只是简单地"遍历候选对列表"，顺序会变，求解器的迭代次序随之改变，
    数值上出现微小差异，让"加速比"的对照失去意义。

    返回
    ----
    list[ContactManifold] : 每个有接触的物体对对应一个流形
    """
    manifolds: list[ContactManifold] = []
    count = len(bodies)

    # 按 i 分组，保持与暴力路径完全相同的遍历顺序
    grouped: dict[int, list[int]] | None = None
    if candidate_pairs is not None:
        grouped = {}
        for index_a, index_b in candidate_pairs:
            first, second = (
                (index_a, index_b) if index_a < index_b else (index_b, index_a)
            )
            grouped.setdefault(first, []).append(second)
        for bucket in grouped.values():
            bucket.sort()

    for i in range(count):
        body_i = bodies[i]

        # --- 与地面 ---
        if ground is not None and not body_i.is_static:
            if isinstance(body_i.geometry, CompoundGeometry):
                ground_points = collide_compound_ground(body_i, ground)
            else:
                ground_points = _part_ground_points(body_i, ground)

            if ground_points:
                manifolds.append(
                    ContactManifold(
                        body_a=GROUND_BODY,
                        body_b=i,
                        points=ground_points,
                        friction=ground.friction,
                        restitution=ground.restitution,
                    )
                )

        # --- 两两之间 ---
        if grouped is None:
            neighbors = range(i + 1, count)
        else:
            neighbors = grouped.get(i, ())

        for j in neighbors:
            body_j = bodies[j]
            # 两个静态物体永远不会产生有意义的接触
            if body_i.is_static and body_j.is_static:
                continue

            manifold = _collide_pair(i, body_i, j, body_j)
            if manifold is not None:
                manifolds.append(manifold)

    return manifolds
