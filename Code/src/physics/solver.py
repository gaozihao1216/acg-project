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

暖启动（warm starting）
-----------------------
静置的接触每帧的冲量几乎不变（就是支撑重力的那点冲量）。冷启动意味着每帧都从
``P = 0`` 开始，迭代只能一点点把冲量"爬"上去——接触越多、链越长，爬得越慢，
这正是"少迭代时堆叠发软、持续下陷"的直接原因。

暖启动把上一帧的累积冲量直接作为初值施加一遍，于是迭代只需要修正**变化量**。
代价是必须能跨帧认出"同一个接触点"，这就是 ``ContactPoint.feature_id`` 存在的
理由：按数组下标匹配，一旦流形点数变化就会把冲量喂错对象。

两个实现上的坑：

1. **只暖启动法向，不暖启动切向**。这一条是实测结论，不是省事：把切向冲量也
   缓存并复用之后，3×3 宽堆在 8 次迭代下的峰值穿透从 0.63 mm 涨到 2.17 mm，
   而只暖启动法向时是 0.35 mm（见 ``scripts/bench_warm_start.py``）。原因是
   四点流形上每个角点的切向冲量都会产生力矩，四角互相不对称，迭代次数有限时
   收敛不掉，反而累积成缓慢自旋与蠕移。法向冲量没有这个问题——它就是支撑重力的
   那一份，四角同向。
2. **同一子步内只能暖启动一次**。TOI 会把一个子步切成多段、每段各求解一次；
   若每段都暖启动，静置接触会在同一个子步内被反复施加支撑冲量（实测棱柱落地
   穿透 0.148 → 1.317 mm）。缓存里记了子步号，同一子步内的后续求解一律冷启动。
3. **法向冲量在两帧之间会微转**，所以缓存的是标量、施加时沿**当前**法向——
   不缓存冲量向量，也就不存在"把旧基下的分量投影到新基"的问题。
4. **Baumgarte 的偏置冲量也在缓存里**。看起来会"多推一次"，但迭代第一步就会把
   它扣掉（推出去之后 ``vn`` 超过目标，``delta`` 为负），所以不额外处理——
   前提是 ``iterations ≥ 1``。
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
        "key",
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
        self.key: tuple[int, int, int] | None = None
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

    __slots__ = (
        "iterations",
        "baumgarte",
        "slop",
        "restitution_threshold",
        "warm_start",
        "_impulse_cache",
        "_solves",
        "_epoch",
    )

    #: 缓存里保留多少"次求解"的历史。超过这个跨度的接触点视为已分离。
    _CACHE_KEEP = 3

    def __init__(
        self,
        iterations: int = 12,
        baumgarte: float = 0.25,
        slop: float = 0.002,
        restitution_threshold: float = 1.0,
        warm_start: bool = True,
    ) -> None:
        """配置求解器。

        参数
        ----
        iterations            : 迭代次数。8–15 是常见区间；更多迭代让接触更"硬"
        baumgarte             : 位置修正系数（0–1），见模块文档
        slop                  : 穿透容差（米）
        restitution_threshold : 低于此接近速度（m/s）时忽略恢复系数
        warm_start            : 是否跨帧复用累积冲量。关掉它用于 A/B 对照，
                                也是复现"没有暖启动时"行为的开关
        """
        self.iterations = int(iterations)
        self.baumgarte = float(baumgarte)
        self.slop = float(slop)
        self.restitution_threshold = float(restitution_threshold)
        self.warm_start = bool(warm_start)

        #: 键 = (body_a 索引, body_b 索引, feature_id)
        #: 值 = (法向累积冲量, 时间戳, 子步号)
        self._impulse_cache: dict[tuple[int, int, int], tuple[float, int, int]] = {}
        self._solves = 0
        self._epoch = 0

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
                constraint.key = (manifold.body_a, manifold.body_b, point.feature_id)
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
            # 但**残留的切向冲量必须撤销**。暖启动会先把上一步的切向冲量施加出去，
            # 若随后法向冲量被迭代压回 0（常见于"上一帧是撞击、这一帧只是贴着"），
            # 那个切向冲量就变成了无源之水：它不在库仑锥内，求解器也不会再碰它。
            # 实测症状是棱柱落地后横向漂移、两层堆叠持续缓慢自旋。
            if constraint.accumulated_t1 != 0.0 or constraint.accumulated_t2 != 0.0:
                total = vec3.add(
                    vec3.scale(constraint.t1, -constraint.accumulated_t1),
                    vec3.scale(constraint.t2, -constraint.accumulated_t2),
                )
                _apply_pair_impulse(
                    body_a, body_b, constraint.r_a, constraint.r_b, total
                )
                constraint.accumulated_t1 = 0.0
                constraint.accumulated_t2 = 0.0
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

    # -- 暖启动 -----------------------------------------------------------

    def reset_cache(self) -> None:
        """清空跨帧冲量缓存。

        场景重载、物体数量变化、或手动改写位姿之后必须调用，否则会把属于旧状态
        的冲量施加到新状态上——症状是"第一次接触时物体被莫名弹开"。
        """
        self._impulse_cache.clear()
        self._solves = 0
        self._epoch = 0

    def begin_substep(self) -> None:
        """进入新的子步。同一个子步内允许多次求解（TOI 切分），但**只能暖启动一次**。

        这一点很关键，也是本实现第一次跑就踩到的坑：TOI 会把一个子步切成若干段，
        每段各求解一次。若每段都暖启动，静置的接触在同一个子步内会被反复施加
        上一子步的支撑冲量——每施加一次就多推一下。实测症状是棱柱落地穿透从
        0.148 mm 反弹到 1.317 mm（物体被"顶"离地面，再落下来）。

        所以缓存里记了"这笔冲量来自哪个子步"：同一子步内的后续求解一律冷启动。
        """
        self._epoch += 1

    def _warm_start_impulses(self, constraints: list[_Constraint]) -> int:
        """把上次求解的法向累积冲量沿当前法向施加，返回命中的接触点数。

        只处理法向：切向冲量**故意不缓存**，原因见模块文档第 1 条（四点流形的
        切向冲量会累积成力矩，迭代次数有限时收敛不掉，反而更抖）。
        """
        if not self._impulse_cache:
            return 0

        hits = 0
        for constraint in constraints:
            entry = self._impulse_cache.get(constraint.key)
            if entry is None:
                continue
            impulse, _stamp, epoch = entry
            if epoch >= self._epoch:
                # 同一子步内已经用过这笔冲量（TOI 把子步切成了多段），
                # 再施加一次就是重复计费
                continue
            if impulse <= 0.0:
                continue

            constraint.accumulated_normal = impulse
            _apply_pair_impulse(
                constraint.body_a, constraint.body_b,
                constraint.r_a, constraint.r_b,
                vec3.scale(constraint.normal, impulse),
            )
            hits += 1

        return hits

    def _store_impulses(self, constraints: list[_Constraint]) -> None:
        """把本次求解的法向累积冲量写回缓存，并清理久未出现的键。"""
        self._solves += 1
        stamp = self._solves

        for constraint in constraints:
            self._impulse_cache[constraint.key] = (
                constraint.accumulated_normal, stamp, self._epoch
            )

        # 清理：连续若干次求解都没出现的接触点视为已分离。
        # 用"按插入序遍历 + 删除"而不是重建字典，保证顺序可复现。
        if stamp % 16 == 0 or len(self._impulse_cache) > 4096:
            cutoff = stamp - self._CACHE_KEEP
            stale = [
                key for key, (_impulse, seen, _epoch) in self._impulse_cache.items()
                if seen < cutoff
            ]
            for key in stale:
                del self._impulse_cache[key]

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

        if self.warm_start:
            self._warm_start_impulses(constraints)
        elif self._impulse_cache:
            # 运行中把暖启动关掉时必须丢弃旧冲量，否则 A/B 对照会被残留状态污染
            self.reset_cache()

        for _ in range(self.iterations):
            for constraint in constraints:
                self._solve_normal(constraint)
                self._solve_friction(constraint)

        if self.warm_start:
            self._store_impulses(constraints)

        return len(constraints)
