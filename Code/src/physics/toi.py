# -*- coding: utf-8 -*-
"""预测接触时刻（time of impact, TOI）。

为什么需要它
------------
窄相检测只能看到"此刻已经穿透"的物体。但一个子步内可能发生**多次先后碰撞**：
A 先撞 B，B 被推动后再撞 C。如果只在一个时刻做一次检测、一次联立求解，
这三者会被当成"同时接触"，求解器于是给出"整体同时变形"的解——那是软材料的物理。

对硬物（钢球等），正确行为是**接触逐个传递**：A→B 完全解算完，B 才带着新速度去撞 C。
这正是牛顿摆只飞出一个球的原因。

所以在推进位置之前，先解析地算出"下一次接触会发生在什么时刻"，
把子步切分成若干段，每段只解算此刻真正发生的接触。

算法
----
在两段位置之间，物体近似做匀速直线运动（子步内速度只被冲量改变），
因此球-球接触时刻是一个**一元二次方程**的较小正根：

    |d₀ + w·t| = R        （d₀ 相对位置，w 相对速度，R 半径和）

展开得 ``(w·w)t² + 2(d₀·w)t + (d₀·d₀ - R²) = 0``。

球-地面同理，退化为一元一次：``t = gap / (-v_z)``。

只考虑"尚未接触且正在接近"的情形。已经重叠的物体不产生 TOI 事件——
它们属于**持续接触**，由正常求解流程处理。
"""

from __future__ import annotations

import math

from ..mathutils import vec3
from ..geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CompoundGeometry,
    ConvexHullGeometry,
    CylinderGeometry,
    SphereGeometry,
)
from . import gjk as _gjk
from .body import RigidBody, iter_parts
from .convex_sat import (
    Polytope,
    candidate_axes,
    closest_point_on_polytope,
    polytope_from_box,
    polytope_from_hull,
    separation_with_side,
    _separation,
)
from .narrowphase import GroundPlane
from .support import support_world

#: 侧面是曲面的几何：没有有限 SAT 轴集合，TOI 走"GJK 距离 + 保守推进"。
_CURVED = (CylinderGeometry, CapsuleGeometry)

#: 判定"相对速度可忽略"的阈值（m/s）。低于此值不认为会再次碰撞。
_MIN_CLOSING_SPEED_SQUARED = 1e-20

#: TOI 小于该值（秒）时认为"此刻已经接触"，不再作为未来的事件。
_MIN_TOI = 1e-12

#: 间隙小于该值（米）就算已经接触。凸包顶点是双精度，1 µm 远小于任何建模精度。
_TOUCH_EPS = 1e-6

#: 闭合速度下界（m/s）。低于它认为窗口内不会接触，直接判为"无事件"。
_MIN_CLOSING = 1e-9

#: 切分时刻**故意**多推进的穿透量（米）。
#:
#: 为什么不精确切在"间隙=0"
#: ------------------------
#: 凸多面体的接触流形靠面裁剪生成。两盒恰好相切（间隙 0）时，裁剪多边形的
#: 面积退化为零，窄相只能给出 **1 个** 接触点。单点约束解不掉整体的法向速度：
#: 实测一次求解只把 -7.22 m/s 降到 -4.95 m/s，剩下的速度在本子步继续推进，
#: 于是穿透 3.9 mm——比不做切分还差。
#:
#: 留一点穿透（200 µm，远小于 slop 2 mm）就能拿到完整的 4 点面流形，
#: 求解器一次把法向速度压到零。这个量必须小到不影响视觉与后续判据，
#: 又必须大到能撑起裁剪多边形。
_CONTACT_OVERLAP = 2e-4


def sphere_sphere_toi(
    body_a: RigidBody,
    body_b: RigidBody,
    max_time: float,
) -> float | None:
    """预测两个球体的接触时刻。

    参数
    ----
    body_a, body_b : 两个球刚体（静态物体的速度视为 0）
    max_time       : 只关心 ``(0, max_time]`` 区间内的事件

    返回
    ----
    float | None : 接触时刻（秒）；不会碰撞、或已在接触、或超出区间时返回 None
    """
    # 目前只有球-球有解析 TOI（本质是一元二次方程）。
    # 盒子需要考虑朝向与角速度，属于另一类问题（保守推进 / GJK 射线），
    # 在实现之前直接跳过——跳过只意味着"不做子步切分"，
    # 碰撞仍会在子步边界的常规检测中被处理，不会漏碰撞。
    if not isinstance(body_a.geometry, SphereGeometry) or not isinstance(
        body_b.geometry, SphereGeometry
    ):
        return None

    radius_sum = body_a.geometry.radius + body_b.geometry.radius

    dx = body_b.position[0] - body_a.position[0]
    dy = body_b.position[1] - body_a.position[1]
    dz = body_b.position[2] - body_a.position[2]

    wx = body_b.linear_velocity[0] - body_a.linear_velocity[0]
    wy = body_b.linear_velocity[1] - body_a.linear_velocity[1]
    wz = body_b.linear_velocity[2] - body_a.linear_velocity[2]

    # 二次项系数
    a = wx * wx + wy * wy + wz * wz
    if a < _MIN_CLOSING_SPEED_SQUARED:
        # 相对静止：不会在未来发生新的接触
        return None

    # 一次项与常数项
    b = 2.0 * (dx * wx + dy * wy + dz * wz)
    c = dx * dx + dy * dy + dz * dz - radius_sum * radius_sum

    if c <= 0.0:
        # 已经重叠：属于持续接触，交给正常求解流程，不作为未来的事件
        return None

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        # 轨迹与接触球面不相交，不会碰撞
        return None

    sqrt_disc = math.sqrt(discriminant)
    # 较小根 = 首次接触时刻；较大根 = 若穿透后继续运动则会分离的时刻
    t = (-b - sqrt_disc) / (2.0 * a)

    if t <= _MIN_TOI or t > max_time:
        # t <= 0 说明正在分离（c > 0 且 t < 0 意味着永远不会接近）
        return None
    return t


def sphere_ground_toi(
    body: RigidBody,
    ground: GroundPlane,
    max_time: float,
) -> float | None:
    """预测球体与水平地面的接触时刻。

    地面静止且无限大，因此退化为一元一次方程。

    参数
    ----
    body     : 球刚体
    ground   : 地面
    max_time : 只关心 ``(0, max_time]`` 区间内的事件

    返回
    ----
    float | None : 接触时刻（秒）；不适用时返回 None
    """
    # 只有球能用"球心高度减半径"求接触时刻；盒子的最低点是旋转后的顶点，
    # 需要另一套推导，实现前先跳过（影响仅是失去子步切分，不会漏碰撞）。
    if not isinstance(body.geometry, SphereGeometry):
        return None

    gap = body.position[2] - ground.z - body.geometry.radius

    if gap <= 0.0:
        # 已经接触或已穿透，不作为未来事件
        return None

    vz = body.linear_velocity[2]
    if vz >= 0.0:
        # 没在下降，不会接触地面
        return None

    t = gap / (-vz)
    if t <= _MIN_TOI or t > max_time:
        return None
    return t


def _overlapping(polytope_a: Polytope, polytope_b: Polytope) -> bool:
    """两个多面体**此刻**是否已经互相嵌入。

    为什么需要这个前置判断
    ----------------------
    按 TOI 切分之后，两个物体往往已经压进 ``_CONTACT_OVERLAP``（200 µm）。
    此时"约束轴"的间隙是负的、会被跳过，但**别的轴**仍可能是正的（斜面接触时
    很常见）。于是那条轴算出 ``t = (小间隙 + 200 µm) / 闭合速度`` —— 一个极小
    的正数，又被当成"未来事件"→ 再切一次 → 时间几乎没推进，却多做一次全量
    接触求解（还带着完整的 Baumgarte 修正）。

    实测后果：8 层棱柱堆叠的事件数暴涨、耗时 +160%，峰值穿透反而从 30 mm
    升到 36 mm（多余的求解在往塔里注入位移）。

    判据很直接：SAT 只要存在**一条**分离轴就是分离；一条都没有，就是已经嵌入。
    去重后轴只有个位数，这个检查很便宜。
    """
    for axis, _kind, _index_a, _index_b in candidate_axes(polytope_a, polytope_b, with_indices=False):
        if _separation(polytope_a, polytope_b, axis) > 0.0:
            return False
    return True


def _cached_polytope(body: RigidBody, cache: dict | None) -> Polytope | None:
    """取多面体；给定缓存时同一次 TOI 调用内不重复构造。

    为什么值得缓存：``earliest_toi`` 对每一对物体都要求世界坐标多面体。
    N 个物体就是 N(N-1)/2 对，而每个物体的多面体在其中会被**重建 N-1 次**
    ——8 个物体时同一个物体的顶点被变换了 7 遍。实测堆叠场景的耗时因此翻倍，
    而结果完全一样（同一时刻的姿态是固定的）。
    """
    if cache is None:
        return _polytope(body)
    key = id(body)
    if key not in cache:
        cache[key] = _polytope(body)
    return cache[key]


def _polytope(body: RigidBody) -> Polytope | None:
    """把一个刚体变成世界坐标多面体；不支持的类型返回 ``None``。

    只支持盒与凸包：柱与胶囊的侧面是曲面，SAT 轴集合不成立（面法向不够用），
    硬套会给出偏晚的 TOI——那比不做更糟，因为它会让子步切分"以为算准了"。
    """
    geometry = body.geometry
    if isinstance(geometry, ConvexHullGeometry):
        return polytope_from_hull(geometry, body.position, body.orientation)
    if isinstance(geometry, BoxGeometry):
        return polytope_from_box(geometry.size, body.position, body.orientation)
    return None


def _spin_tip_speed(body: RigidBody) -> float:
    """``|ω| × 包围半径``：物体上任何一点因自转而获得的速度上限。

    这是整个凸包 TOI 的关键界。没有任何解析式能直接给出"旋转的凸包何时碰到
    另一个旋转的凸包"——棱扫过的是一条曲纹面。但"任一点的速度不超过
    ``|ω|R``"是**严格**的，于是可以把它当成间隙的收缩速度上限，
    得到一个**保守**（永不偏晚）的接触时刻。
    """
    omega = body.angular_velocity
    spin = math.sqrt(omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2)
    if spin <= 0.0:
        return 0.0
    return spin * body.geometry.bounding_radius()


def hull_hull_toi(
    body_a: RigidBody,
    body_b: RigidBody,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """预测两个**盒子/凸包**的接触时刻（保守，永不偏晚）。

    算法：沿 SAT 候选轴逐条求"间隙何时收完"
    ---------------------------------------
    对任意单位轴 ``n``，两个多面体在该轴上的间隙是

        ``g_n = max(a_low - b_high, b_low - a_high)``

    它随时间收缩的速率不超过

        ``闭合速率 ≤ max(0, 侧 · (v_B - v_A)·n) + |ω_A|R_A + |ω_B|R_B``

    第一项是平移沿该轴的投影（侧符号说明是哪一侧的投影给出的间隙），
    后两项是自转——凸包上任意点因自转获得的速度不超过 ``|ω|R``。

    于是 ``t_n = g_n / 闭合速率`` 是"这条轴收完"的时刻。取所有轴的**最小**值
    就是 TOI 的**保守**估计：真实接触一定不早于任何一个单轴收完的时刻，
    所以 ``min_n t_n ≤ 真实 TOI``。

    为什么"保守"在这里是安全的
    -------------------------
    偏早只会让子步切得更细——推进到那一刻、检测、发现没接触、继续推进，
    代价是一两次多余的窄相查询。偏晚则会漏掉切分、让接触在同一时刻联立求解，
    那才是要避免的（牛顿摆会飞出两个球）。所以宁可偏早。

    对纯平移的情形（不旋转），沿正确轴的估计是**精确**的：面-面接近由面法向
    给出精确解，棱-棱接近由叉积轴给出精确解。

    返回值
    ------
    float | None : ``(0, max_time]`` 内的接触时刻；不适用/不会接触/已接触时 None
    """
    if body_a.is_static and body_b.is_static:
        return None

    polytope_a = _cached_polytope(body_a, cache)
    polytope_b = _cached_polytope(body_b, cache)
    if polytope_a is None or polytope_b is None:
        return None

    # 已经嵌在一起 ⇒ 属于持续接触，由求解器推出，不再报未来事件
    if _overlapping(polytope_a, polytope_b):
        return None

    velocity_x = body_b.linear_velocity[0] - body_a.linear_velocity[0]
    velocity_y = body_b.linear_velocity[1] - body_a.linear_velocity[1]
    velocity_z = body_b.linear_velocity[2] - body_a.linear_velocity[2]

    # --- 包围球剪枝 ---
    # 两球的闭合速率同样有上界，若 gap 在窗口内收不完，整对物体直接跳过。
    # 这一步很关键：两个 64 面凸包的候选轴有上千条，逐个算会拖慢整个仿真，
    # 而绝大多数物体对在任一时刻都离得很远。
    dx = body_b.position[0] - body_a.position[0]
    dy = body_b.position[1] - body_a.position[1]
    dz = body_b.position[2] - body_a.position[2]
    center_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    radius_sum = body_a.geometry.bounding_radius() + body_b.geometry.bounding_radius()
    sphere_gap = center_distance - radius_sum
    if sphere_gap > 0.0:
        speed = math.sqrt(
            velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2
        ) + _spin_tip_speed(body_a) + _spin_tip_speed(body_b)
        if speed <= _MIN_CLOSING or sphere_gap / speed > max_time:
            return None

    spin_bound = _spin_tip_speed(body_a) + _spin_tip_speed(body_b)
    best: float | None = None

    for axis, _kind, _index_a, _index_b in candidate_axes(polytope_a, polytope_b, with_indices=False):
        gap, side = separation_with_side(polytope_a, polytope_b, axis)

        if gap <= _TOUCH_EPS:
            # 这条轴已经收完（或重叠）：不是未来事件，交给常规求解流程
            continue

        closing = max(
            0.0, side * (velocity_x * axis[0]
                         + velocity_y * axis[1]
                         + velocity_z * axis[2])
        ) + spin_bound

        if closing <= _MIN_CLOSING:
            continue   # 沿这条轴没有接近趋势

        # 多推进 _CONTACT_OVERLAP，让裁剪多边形有面积（见常量说明）
        t = (gap + _CONTACT_OVERLAP) / closing
        if t <= _MIN_TOI or t > max_time:
            continue
        if best is None or t < best:
            best = t

    return best


def hull_ground_toi(
    body: RigidBody,
    ground: GroundPlane,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """预测盒子/凸包与水平地面的接触时刻（对纯平移+自转是**精确**的）。

    为什么地面这里能精确、而对物体只能保守
    --------------------------------------
    地面是一个平面，物体上的点撞地时**一定在顶点上**（凸多面体的极值必在顶点）。
    而顶点在恒定 ``v``、``ω`` 下的轨迹是一条**直线**：

        ``z_i(t) = z_i + (v_z + (ω × r_i)_z) · t``

    所以每个顶点给出一个精确的撞击时刻 ``t_i = (z_i - g) / (-v_{i,z})``，
    TOI 就是这些 ``t_i`` 的最小值。没有保守余量，也不需要迭代。

    这条也是"球-地面一元一次方程"的自然推广：球可以看成无穷多个顶点，
    其中最低的那个就等价于球心高度减半径。
    """
    polytope = _cached_polytope(body, cache)
    if polytope is None:
        return None

    vx, vy, vz = body.linear_velocity
    wx, wy, wz = body.angular_velocity
    cx, cy, cz = body.position

    # 只要有顶点已经贴地，这个物体就处在**持续接触**里，不产生未来事件。
    #
    # 这里必须整体返回 None，而不是"跳过这个顶点、继续算别的顶点"。
    # 后者会给出"另一侧顶点何时落地"的时刻——听起来合理，实则有害：
    # 贴地的物体由求解器接管，再报一个 0.3 s 后的事件会让子步在毫无接触的
    # 时刻被切开一次，白白扰动求解流程（实测：一个底面恰好贴地的盒子，
    # 报出 t=0.3001 s，而它其实一直躺着）。
    for vertex in polytope.vertices:
        if vertex[2] - ground.z <= _TOUCH_EPS:
            return None

    best: float | None = None
    for vertex in polytope.vertices:
        gap = vertex[2] - ground.z

        # 该顶点相对质心的位置（自转绕质心）
        rx = vertex[0] - cx
        ry = vertex[1] - cy
        rz = vertex[2] - cz

        # (ω × r) 的 z 分量
        vertex_vz = vz + (wx * ry - wy * rx)

        if vertex_vz >= -_MIN_CLOSING:
            continue   # 这个顶点不在下降

        # 与 hull_hull_toi 同样的理由：留 200 µm 让接触流形撑得开
        t = (gap + _CONTACT_OVERLAP) / (-vertex_vz)
        if t <= _MIN_TOI or t > max_time:
            continue
        if best is None or t < best:
            best = t

    return best


def sphere_hull_toi(
    sphere_body: RigidBody,
    polytope_body: RigidBody,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """预测球与盒子/凸包的接触时刻（保守）。

    球退化成"点 + 半径"：先求球心到多面体表面的最近距离，再沿该处的法向
    看它收得多快。凸包的自转同样用 ``|ω|R`` 作为上界。
    """
    polytope = _cached_polytope(polytope_body, cache)
    if polytope is None:
        return None

    point, distance, normal = closest_point_on_polytope(
        polytope, sphere_body.position
    )
    # distance 为负表示球心在多面体内部
    gap = distance - sphere_body.geometry.radius
    if gap <= _TOUCH_EPS:
        return None

    relative_x = sphere_body.linear_velocity[0] - polytope_body.linear_velocity[0]
    relative_y = sphere_body.linear_velocity[1] - polytope_body.linear_velocity[1]
    relative_z = sphere_body.linear_velocity[2] - polytope_body.linear_velocity[2]

    # 法向指向多面体外部（朝球），间隙收缩 = 球沿 -normal 靠近
    approach = -(
        relative_x * normal[0] + relative_y * normal[1] + relative_z * normal[2]
    )
    closing = max(0.0, approach) + _spin_tip_speed(polytope_body)
    if closing <= _MIN_CLOSING:
        return None

    t = gap / closing
    if t <= _MIN_TOI or t > max_time:
        return None
    return t


def curved_pair_toi(
    body_a: RigidBody,
    body_b: RigidBody,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """含柱/胶囊的物体对：GJK 距离 + **保守推进**（永不偏晚）。

    为什么曲面不能用 SAT 那套
    ------------------------
    SAT 的前提是"有限条候选轴"：多面体的棱与面能生成这几条轴，曲面生成不了
    （柱的侧面有无穷多条法向）。硬凑一个有限集会给出**偏晚**的 TOI，而偏晚
    等于漏切分——牛顿摆会一次飞出两个球。

    这里改用间隙收缩速率的严格上界：

        gap(t) ≥ gap - ( max(0, (v_B - v_A)·n) + |ω_A|R_A + |ω_B|R_B ) · t

    右边括号里是"任何一点的相对速度沿法向投影"的上界（自转对一个点的贡献
    不超过 ``|ω|R``）。于是 ``t = gap / closing`` 一定**不晚于**真实接触时刻，
    属于"保守"一档：偏早只会多重切一刀，代价是一两次多余的窄相查询。

    对称性
    ------
    ``distance`` 对称、``(v_B - v_A)·n`` 交换后符号同时翻转，所以
    ``curved_pair_toi(a, b)`` 与 ``curved_pair_toi(b, a)`` 逐位相等——
    能力表测试会检查这一点。
    """
    gap, point_a, point_b, _simplex = _gjk.distance(body_a, body_b)
    if gap <= _TOUCH_EPS:
        return None

    delta = vec3.sub(point_b, point_a)
    length = vec3.length(delta)
    if length <= _MIN_CLOSING:
        return None
    normal = (delta[0] / length, delta[1] / length, delta[2] / length)

    relative_x = body_b.linear_velocity[0] - body_a.linear_velocity[0]
    relative_y = body_b.linear_velocity[1] - body_a.linear_velocity[1]
    relative_z = body_b.linear_velocity[2] - body_a.linear_velocity[2]

    # 法向是 A→B，所以"接近"是相对速度在 -n 上的投影
    closing = max(
        0.0,
        -(
            relative_x * normal[0]
            + relative_y * normal[1]
            + relative_z * normal[2]
        ),
    ) + _spin_tip_speed(body_a) + _spin_tip_speed(body_b)
    if closing <= _MIN_CLOSING:
        return None

    # 多推进一点穿透，让窄相能报出"真实穿透"（见 _CONTACT_OVERLAP 的说明）
    t = (gap + _CONTACT_OVERLAP) / closing
    if t <= _MIN_TOI or t > max_time:
        return None
    return t


def curved_ground_toi(
    body: RigidBody,
    ground: GroundPlane,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """柱/胶囊与水平地面：支撑点最低高度 + 保守推进。

    最低点用 ``support(-Z)`` 直接取到（对柱是端面圆周上的精确最低点，对胶囊
    是端球底部），所以间隙是精确的；不确定的只有"它下降得多快"，用
    ``max(0, -v_z) + |ω|R`` 作为上界，因此整体是保守估计。
    """
    lowest = support_world(body, (0.0, 0.0, -1.0))
    gap = lowest[2] - ground.z
    if gap <= _TOUCH_EPS:
        return None

    closing = max(0.0, -body.linear_velocity[2]) + _spin_tip_speed(body)
    if closing <= _MIN_CLOSING:
        return None

    t = (gap + _CONTACT_OVERLAP) / closing
    if t <= _MIN_TOI or t > max_time:
        return None
    return t


def compound_pair_toi(
    body_a: RigidBody,
    body_b: RigidBody,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """复合体参与的一对刚体：逐部件对预测，取**最小**时刻。

    取最小是对的：任何一对部件先碰上，那次碰撞就是真实事件，必须在那一刻
    切开子步。取错了（比如取首个非空结果）会让后发生的部件碰撞被提前处理，
    表现为"还没接触就被弹开"。

    部件速度用 ``v + ω × r`` 换算（见 ``body.PartView``）。用整体质心速度会
    得出"旋转中的哑铃没有接近"——高速自转的复合体就会穿模。
    """
    earliest: float | None = None
    window = max_time

    for _index_a, view_a in iter_parts(body_a):
        for _index_b, view_b in iter_parts(body_b):
            t = _dispatch_toi(view_a, view_b, window, cache)
            if t is not None and (earliest is None or t < earliest):
                earliest = t
                window = t    # 窗口收窄：后续只关心更早的事件

    return earliest


def compound_ground_toi(
    body: RigidBody,
    ground: GroundPlane,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """复合体与地面：逐部件预测，取最早时刻。"""
    earliest: float | None = None
    window = max_time

    for _index, view in iter_parts(body):
        t = ground_toi(view, ground, window, cache)
        if t is not None and (earliest is None or t < earliest):
            earliest = t
            window = t

    return earliest


def may_collide(body_a: RigidBody, body_b: RigidBody, max_time: float) -> bool:
    """包围球层面的**精确**拒绝：这对物体在 ``max_time`` 内不可能相碰。

    为什么这层剪枝值得单独写
    ------------------------
    ``earliest_toi`` 是 O(N²) 的：N 个物体就是 N(N-1)/2 对，每一对都可能去算
    "球心到多面体最近点"（要遍历所有面、还要做多边形投影）。56 个物体的堆积
    场景里，单次子步有 1 534 对，其中绝大多数相隔很远、**根本不可能在这一子步里
    碰上**，却在做完整的最近点计算。剖析显示它是单步耗时的 98%（见
    ``docs/14`` 的性能一节）。

    判据是严格的：物体各自被包围球包住，包围球都不相交 ⇒ 物体必然不接触。
    而包围球**不旋转**，球心距离的变化速率不超过相对平动速度
    ``|Δv|``。于是"``R_a + R_b + |Δv|·t`` 仍然小于球心距"就意味着到 ``t`` 为
    止都不可能有事件。这里连 ``sqrt`` 都不做：用 L1 范数作 ``|Δv|`` 的上界，
    再比较平方。

    这一步不会改变任何结果，只是把"算不出来的拒绝"提前成"一眼看得出"。
    """
    pa = body_a.position
    pb = body_b.position
    dx = pb[0] - pa[0]
    dy = pb[1] - pa[1]
    dz = pb[2] - pa[2]
    distance_sq = dx * dx + dy * dy + dz * dz

    reach = body_a.geometry.bounding_radius() + body_b.geometry.bounding_radius()
    va = body_a.linear_velocity
    vb = body_b.linear_velocity
    # L1 ≥ L2，所以它是相对速度的严格上界，不需要开方
    relative_speed = (
        abs(vb[0] - va[0]) + abs(vb[1] - va[1]) + abs(vb[2] - va[2])
    )
    limit = reach + relative_speed * max_time
    return distance_sq <= limit * limit


def body_toi(
    body_a: RigidBody,
    body_b: RigidBody,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """两个刚体之间的接触时刻（按几何类型分派）。

    球-球走精确的一元二次方程；含球的其他组合走"最近点 + 法向"；其余走
    ``hull_hull_toi`` 的保守轴推进。三种路径的共同要求是**不偏晚**。

    进入分派前先做一次包围球拒绝（:func:`may_collide`）——它不改结果，只把
    绝大多数"隔着半个场景"的物体对挡在外面。
    """
    if not may_collide(body_a, body_b, max_time):
        return None
    return _dispatch_toi(body_a, body_b, max_time, cache)


def _dispatch_toi(
    body_a: RigidBody,
    body_b: RigidBody,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """按几何类型分派，**不做**包围球预判。

    单独拆出来是给 :func:`earliest_toi` 用的：那里已经把同一个判断内联进了内层
    循环（省掉一次函数调用），再走一遍 :func:`may_collide` 属于重复计算。
    """
    a_is_sphere = isinstance(body_a.geometry, SphereGeometry)
    b_is_sphere = isinstance(body_b.geometry, SphereGeometry)
    a_is_curved = isinstance(body_a.geometry, _CURVED)
    b_is_curved = isinstance(body_b.geometry, _CURVED)

    if isinstance(body_a.geometry, CompoundGeometry) or isinstance(
        body_b.geometry, CompoundGeometry
    ):
        return compound_pair_toi(body_a, body_b, max_time, cache)
    if a_is_curved or b_is_curved:
        return curved_pair_toi(body_a, body_b, max_time, cache)
    if a_is_sphere and b_is_sphere:
        return sphere_sphere_toi(body_a, body_b, max_time)
    if a_is_sphere:
        return sphere_hull_toi(body_a, body_b, max_time, cache)
    if b_is_sphere:
        return sphere_hull_toi(body_b, body_a, max_time, cache)
    return hull_hull_toi(body_a, body_b, max_time, cache)


def ground_toi(
    body: RigidBody,
    ground: GroundPlane,
    max_time: float,
    cache: dict | None = None,
) -> float | None:
    """刚体与地面的接触时刻（球走一元一次，柱/胶囊走支撑点，其余走顶点解）。"""
    if isinstance(body.geometry, CompoundGeometry):
        return compound_ground_toi(body, ground, max_time, cache)
    if isinstance(body.geometry, _CURVED):
        return curved_ground_toi(body, ground, max_time, cache)
    if isinstance(body.geometry, SphereGeometry):
        return sphere_ground_toi(body, ground, max_time)
    return hull_ground_toi(body, ground, max_time, cache)


def earliest_toi(
    bodies: list[RigidBody],
    ground: GroundPlane | None,
    max_time: float,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[float, str] | None:
    """找出整个场景中最早发生的接触事件。

    参数
    ----
    bodies   : 刚体列表
    ground   : 地面；None 表示自由空间
    max_time : 时间窗口（秒）
    pairs    : 可选候选对。给定后只检查这些对——由宽阶段把 AABB 按"本子步最大
               位移"膨胀后给出，是**超集**（不会漏事件），但把 O(N²) 降成了 O(邻居)。
               为 ``None`` 时退化为全枚举，供单元测试与小型场景使用。

    返回
    ----
    (时刻, 描述) 二元组；没有任何事件时返回 None。
    描述字符串仅用于日志与调试（例如 ``'ball_a' <-> 'ball_b'``）。

    为什么内层要摊平
    ----------------
    这是全流程里唯一还在 O(N²) 上跑循环的地方。循环体里每多一次属性访问或方法
    调用，就乘以 N²/2 次——56 个物体时是 1 534 次/子步，500 个物体时是 124 750 次。
    所以位置、速度、半径、静态标记在循环**之前**取成元组，内层只做算术。
    """
    best_time: float | None = None
    best_label = ""

    count = len(bodies)
    # 同一次调用内复用世界坐标多面体：每对物体都要用，重建 N-1 次是浪费
    cache: dict = {}

    # 内层循环的摊平数据（见上面"为什么内层要摊平"）
    flat = [
        (
            body.position[0], body.position[1], body.position[2],
            body.linear_velocity[0], body.linear_velocity[1], body.linear_velocity[2],
            body.bounding_radius,
            body.is_static,
        )
        for body in bodies
    ]

    if ground is not None:
        for body in bodies:
            if body.is_static:
                continue
            t = ground_toi(body, ground, max_time, cache)
            if t is not None and (best_time is None or t < best_time):
                best_time = t
                best_label = f"地面 <-> '{body.name}'"

    if pairs is None:
        candidate_iter = (
            (i, j) for i in range(count) for j in range(i + 1, count)
        )
    else:
        candidate_iter = pairs

    for i, j in candidate_iter:
        xi, yi, zi, vxi, vyi, vzi, ri, static_i = flat[i]
        xj, yj, zj, vxj, vyj, vzj, rj, static_j = flat[j]
        if static_i and static_j:
            continue

        # 已经找到更早的事件时，窗口收窄，剪枝更狠
        window = max_time if best_time is None else (
            best_time if best_time < max_time else max_time
        )

        # 包围球拒绝（同 may_collide，但内联以省掉 N² 次函数调用）
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi
        limit = ri + rj + (
            abs(vxj - vxi) + abs(vyj - vyi) + abs(vzj - vzi)
        ) * window
        if dx * dx + dy * dy + dz * dz > limit * limit:
            continue

        t = _dispatch_toi(bodies[i], bodies[j], window, cache)
        if t is not None and (best_time is None or t < best_time):
            best_time = t
            best_label = f"'{bodies[i].name}' <-> '{bodies[j].name}'"

    if best_time is None:
        return None
    return best_time, best_label
