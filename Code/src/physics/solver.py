# -*- coding: utf-8 -*-
"""顺序冲量（sequential impulse）接触求解器。

这是整个刚体引擎的**核心**，也是最难写对的部分。它不求解线性方程组，
而是反复"逐个接触点施加冲量并修正速度"，迭代若干次逼近满足所有约束的
速度场。优点是极其简单、天然支持多接触、容易加摩擦与限幅。

一次求解做的事情
----------------
对每个接触点，把它看成一条**速度非穿透约束**：

    目标：接触点的法向相对速度 ≥ 0（不允许继续靠近）

用累积冲量 ``P`` 迭代修正：

    vn = 接触点法向相对速度
    dj = (vn_target - vn) / K        # K 是该方向的"有效质量"倒数
    P_new = max(0, P + dj)           # 单侧约束：只能推、不能拉
    apply(P_new - P)

"只能推不能拉"正是 `max(0, ...)` 的含义。如果漏掉这个限幅，物体会被
"粘"在一起互相吸引——这是最常见的症状。

四个关键参数
------------
恢复系数 restitution
    碰撞后法向速度反向的比例。只在**接近速度足够大**时启用，
    否则静置物体会无限微弹（见 ``restitution_threshold``）。

摩擦 friction
    切向冲量上限为 ``mu * P_normal``（库仑锥的近似）。求解器里对两个
    正交切向分别限幅，而不是严格解摩擦锥——这是所有实时引擎的通行简化。

Baumgarte 系数
    穿透不会被速度约束直接修正（速度约束只管"别再深入"），所以需要
    额外给一个**分离速度**把物体推出来：``beta/h * (penetration - slop)``。
    太大会产生"弹跳感"，太小会持续下陷。

slop（穿透容差）
    允许保留一点穿透不去修正。没有它，静置物体会被反复推开又落下，
    表现为持续的高频抖动。

为什么必须先捕获初始法向速度
----------------------------
恢复系数要基于**碰撞前**的接近速度，而不是迭代过程中的瞬时速度。
如果每次迭代都用当前速度乘恢复系数，能量会被反复注入，物体会越弹越高。
"""

from __future__ import annotations

from ..mathutils import vec3
from .body import RigidBody
from .contact import ContactManifold


class _Constraint:
    """一个接触点的求解上下文（预处理结果 + 累积冲量）。

    把每个迭代都要用的量（力臂、有效质量、切向基）在这里算一次并缓存，
    避免在迭代循环里重复计算三角函数与矩阵旋转。
    """

    __slots__ = (
        "body_a",
        "body_b",
        "r_a",
        "r_b",
        "normal",
        "t1",
        "t2",
        "normal_mass",
        "t1_mass",
        "t2_mass",
        "target_normal_velocity",
        "friction",
        "accumulated_normal",
        "accumulated_t1",
        "accumulated_t2",
    )

    def __init__(self) -> None:
        """只做占位初始化，真正的填充在预处理阶段完成。"""
        self.body_a: RigidBody | None = None
        self.body_b: RigidBody | None = None
        self.r_a = vec3.zeros()
        self.r_b = vec3.zeros()
        self.normal = (0.0, 0.0, 1.0)
        self.t1 = (1.0, 0.0, 0.0)
        self.t2 = (0.0, 1.0, 0.0)
        self.normal_mass = 0.0
        self.t1_mass = 0.0
        self.t2_mass = 0.0
        self.target_normal_velocity = 0.0
        self.friction = 0.0
        self.accumulated_normal = 0.0
        self.accumulated_t1 = 0.0
        self.accumulated_t2 = 0.0


def _effective_mass(
    body_a: RigidBody,
    body_b: RigidBody,
    r_a,
    r_b,
    direction,
) -> float:
    """计算沿 ``direction`` 方向的约束"有效质量"的倒数 ``K``。

    物理含义：在这个方向上施加 1 单位的冲量，接触点的相对速度会变化 ``K``。

    ``K = 1/m_a + 1/m_b + n·((I_a⁻¹(r_a×n))×r_a) + n·((I_b⁻¹(r_b×n))×r_b)``

    前两项是平动贡献，后两项是转动贡献（力臂产生的力矩反过来影响接触点速度）。
    静态物体的 ``inv_mass`` 与逆惯量都是 0，因此自然只贡献 0。
    """
    k = body_a.inv_mass + body_b.inv_mass

    if not body_a.is_static:
        ra_cross_n = vec3.cross(r_a, direction)
        k += vec3.dot(
            vec3.cross(body_a.inverse_inertia_world(ra_cross_n), r_a), direction
        )

    if not body_b.is_static:
        rb_cross_n = vec3.cross(r_b, direction)
        k += vec3.dot(
            vec3.cross(body_b.inverse_inertia_world(rb_cross_n), r_b), direction
        )

    return k


def _relative_velocity(
    body_a: RigidBody, body_b: RigidBody, r_a, r_b
) -> tuple[float, float, float]:
    """返回 B 相对 A 在接触点处的速度（``v_b - v_a``）。

    注意用的是**接触点速度**而不是质心速度——否则旋转的物体撞到东西时
    响应会完全不对。
    """
    va = body_a.velocity_at(r_a)
    vb = body_b.velocity_at(r_b)
    return vec3.sub(vb, va)


def _apply_pair_impulse(
    body_a: RigidBody,
    body_b: RigidBody,
    r_a,
    r_b,
    impulse,
) -> None:
    """对一对物体施加大小相等、方向相反的冲量。

    约定 ``impulse`` 沿从 A 指向 B 的方向：A 受到 ``-impulse``，B 受到 ``+impulse``。
    """
    body_a.apply_impulse(vec3.scale(impulse, -1.0), r_a)
    body_b.apply_impulse(impulse, r_b)


class SequentialImpulseSolver:
    """顺序冲量求解器。

    典型用法（由 ``PhysicsWorld`` 调用）::

        solver = SequentialImpulseSolver(iterations=12)
        solver.solve(manifolds, get_body, h)

    其中 ``get_body(i)`` 负责把 ``GROUND_BODY(-1)`` 解析成静态替身。
    """

    __slots__ = ("iterations", "baumgarte", "slop", "restitution_threshold")

    def __init__(
        self,
        iterations: int = 12,
        baumgarte: float = 0.25,
        slop: float = 0.002,
        restitution_threshold: float = 1.0,
    ) -> None:
        """配置求解器。

        参数
        ----
        iterations            : 迭代次数。8–15 是常见区间；更多迭代让接触更"硬"
        baumgarte             : 位置修正系数（0–1），见模块文档
        slop                  : 穿透容差（米）
        restitution_threshold : 低于此接近速度（m/s）时忽略恢复系数
        """
        self.iterations = int(iterations)
        self.baumgarte = float(baumgarte)
        self.slop = float(slop)
        self.restitution_threshold = float(restitution_threshold)

    # -- 预处理 -----------------------------------------------------------

    def _build_constraints(self, manifolds, get_body, h: float) -> list[_Constraint]:
        """把所有接触点转换成内部约束结构。

        这一步做三件事：

        1. 计算接触点相对两个质心的力臂 ``r_a`` / ``r_b``
        2. 计算法向与两个切向的**有效质量**
        3. 捕获**碰撞前**的法向相对速度，据此确定本次求解的目标分离速度
           （恢复系数 + Baumgarte 修正）
        """
        constraints: list[_Constraint] = []

        for manifold in manifolds:
            body_a = get_body(manifold.body_a)
            body_b = get_body(manifold.body_b)

            for point in manifold.points:
                constraint = _Constraint()
                constraint.body_a = body_a
                constraint.body_b = body_b
                constraint.normal = point.normal
                constraint.friction = manifold.friction

                # 力臂：接触点 - 质心
                constraint.r_a = vec3.sub(point.position, body_a.position)
                constraint.r_b = vec3.sub(point.position, body_b.position)

                # 切向基：用于把摩擦拆成两个正交方向
                constraint.t1, constraint.t2 = vec3.orthogonal_basis(point.normal)

                # 三个方向的有效质量
                constraint.normal_mass = _effective_mass(
                    body_a, body_b, constraint.r_a, constraint.r_b, constraint.normal
                )
                constraint.t1_mass = _effective_mass(
                    body_a, body_b, constraint.r_a, constraint.r_b, constraint.t1
                )
                constraint.t2_mass = _effective_mass(
                    body_a, body_b, constraint.r_a, constraint.r_b, constraint.t2
                )

                # 有效质量退化为 0 说明两侧都是静态物体，直接跳过
                if constraint.normal_mass <= 1e-12:
                    continue

                # --- 捕获碰撞前的接近速度 ---
                v_rel = _relative_velocity(body_a, body_b, constraint.r_a, constraint.r_b)
                initial_normal_velocity = vec3.dot(v_rel, constraint.normal)

                # 恢复系数只在"撞得够快"时生效，否则静置物体会无限微弹
                if initial_normal_velocity < -self.restitution_threshold:
                    bounce = -manifold.restitution * initial_normal_velocity
                else:
                    bounce = 0.0

                # Baumgarte：把超出容差的穿透，按比例转成"分离速度"
                excess = max(0.0, point.penetration - self.slop)
                bias = self.baumgarte * excess / h

                constraint.target_normal_velocity = bounce + bias
                constraints.append(constraint)

        return constraints

    # -- 迭代求解 ---------------------------------------------------------

    def _solve_normal(self, constraint: _Constraint) -> None:
        """求解一次法向约束（单侧，只能推开）。"""
        body_a = constraint.body_a
        body_b = constraint.body_b

        v_rel = _relative_velocity(body_a, body_b, constraint.r_a, constraint.r_b)
        vn = vec3.dot(v_rel, constraint.normal)

        # 需要补上的速度差，除以有效质量得到冲量增量
        delta = (constraint.target_normal_velocity - vn) / constraint.normal_mass

        # 累积并限幅：接触冲量必须非负
        accumulated = constraint.accumulated_normal + delta
        if accumulated < 0.0:
            accumulated = 0.0
        applied = accumulated - constraint.accumulated_normal
        constraint.accumulated_normal = accumulated

        if applied != 0.0:
            _apply_pair_impulse(
                body_a, body_b, constraint.r_a, constraint.r_b,
                vec3.scale(constraint.normal, applied),
            )

    def _solve_friction(self, constraint: _Constraint) -> None:
        """求解一次摩擦约束（两个正交切向，受库仑锥限幅）。"""
        body_a = constraint.body_a
        body_b = constraint.body_b

        # 法向没有压力时不可能有摩擦
        max_friction = constraint.friction * constraint.accumulated_normal
        if max_friction <= 0.0:
            return

        v_rel = _relative_velocity(body_a, body_b, constraint.r_a, constraint.r_b)

        # --- 切向 1 ---
        vt1 = vec3.dot(v_rel, constraint.t1)
        delta1 = -vt1 / constraint.t1_mass
        new1 = constraint.accumulated_t1 + delta1
        new1 = max(-max_friction, min(max_friction, new1))
        applied1 = new1 - constraint.accumulated_t1
        constraint.accumulated_t1 = new1
        if applied1 != 0.0:
            _apply_pair_impulse(
                body_a, body_b, constraint.r_a, constraint.r_b,
                vec3.scale(constraint.t1, applied1),
            )

        # --- 切向 2 ---（重新取相对速度，因为切向 1 已改变了速度）
        v_rel = _relative_velocity(body_a, body_b, constraint.r_a, constraint.r_b)
        vt2 = vec3.dot(v_rel, constraint.t2)
        delta2 = -vt2 / constraint.t2_mass
        new2 = constraint.accumulated_t2 + delta2
        new2 = max(-max_friction, min(max_friction, new2))
        applied2 = new2 - constraint.accumulated_t2
        constraint.accumulated_t2 = new2
        if applied2 != 0.0:
            _apply_pair_impulse(
                body_a, body_b, constraint.r_a, constraint.r_b,
                vec3.scale(constraint.t2, applied2),
            )

    def solve(self, manifolds, get_body, h: float) -> int:
        """求解一批接触。

        参数
        ----
        manifolds : ``ContactManifold`` 列表
        get_body  : 索引 → 刚体 的解析函数（负责把 ``GROUND_BODY`` 映射到静态替身）
        h         : 当前子步的时长（秒）

        返回
        ----
        int : 实际参与求解的接触点数量，可用于日志与性能统计
        """
        constraints = self._build_constraints(manifolds, get_body, h)
        if not constraints:
            return 0

        for _ in range(self.iterations):
            for constraint in constraints:
                self._solve_normal(constraint)
                self._solve_friction(constraint)

        return len(constraints)
