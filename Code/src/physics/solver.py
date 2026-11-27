# -*- coding: utf-8 -*-
"""顺序冲量速度求解与独立的 split-impulse 几何纠正。

速度层：法向累积冲量非负，恢复系数只在足够大的接近速度下启用；切向采用
二维有效质量谱界控制的投影梯度，把累积冲量投影到库仑摩擦圆盘。
几何层：每个完整子步结束后，独立求解虚拟速度以减少超出 slop 的穿透。
虚拟冲量不进入真实线速度、世界角动量或暖启动缓存。

ContactManager 保存两侧局部表面锚点、法向、时间步和冲量。仅连续子步允许
几何有效的接触暖启动，特征 ID 是提示而非身份保证；同一子步只施加一次。
当前只缓存法向冲量。切向暖启动仍关闭，避免在未独立校准前扩大行为变化。
历史策略实验的数据描述旧实现，不能作为当前参数下的性能数字。
"""

from __future__ import annotations

import math

from ..mathutils import vec3
from ..mathutils import quaternion as quat
from .body import RigidBody
from .contact import ContactManifold
from .contact_cache import ContactManager, ensure_anchors


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
        "t12_mass",
        "target_normal_velocity",
        "friction",
        "accumulated_normal",
        "accumulated_t1",
        "accumulated_t2",
        "anchor_a", "anchor_b", "cache_entry", "h",
        "bias_target", "accumulated_bias",
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
        self.t12_mass = 0.0
        self.target_normal_velocity = 0.0
        self.friction = 0.0
        self.accumulated_normal = 0.0
        self.accumulated_t1 = 0.0
        self.accumulated_t2 = 0.0
        self.anchor_a = self.anchor_b = (0.0, 0.0, 0.0)
        self.cache_entry = None
        self.h = 0.0
        self.bias_target = self.accumulated_bias = 0.0


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
        "contacts",
    )

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
        self.contacts = ContactManager(self.slop)

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
        self.contacts.slop = self.slop
        self.contacts.begin_query()

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
                constraint.h = h
                constraint.cache_entry = self.contacts.bind(point, constraint.key, body_a, body_b)
                constraint.anchor_a, constraint.anchor_b = point.local_anchor_a, point.local_anchor_b

                # 力臂：接触点 - 质心
                constraint.r_a = quat.quat_rotate_vector(body_a.orientation, constraint.anchor_a)
                constraint.r_b = quat.quat_rotate_vector(body_b.orientation, constraint.anchor_b)

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
                constraint.t12_mass = 0.0
                for body, r in ((body_a, constraint.r_a), (body_b, constraint.r_b)):
                    if not body.is_static:
                        response = vec3.cross(body.inverse_inertia_world(vec3.cross(r, constraint.t2)), r)
                        constraint.t12_mass += vec3.dot(constraint.t1, response)

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
                radii = [b.bounding_radius for b in (body_a, body_b) if b.geometry is not None]
                correction_limit = min(.05, .2 * min(radii, default=1))
                bias = min(self.baumgarte * excess, correction_limit) / h

                constraint.target_normal_velocity = bounce
                constraint.bias_target = bias
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

        vt1 = vec3.dot(v_rel, constraint.t1)
        vt2 = vec3.dot(v_rel, constraint.t2)
        k1, k2, k12 = constraint.t1_mass, constraint.t2_mass, constraint.t12_mass
        determinant = k1 * k2 - k12 * k12
        if determinant <= 1e-24:
            return
        # 二维投影梯度：用切向 K 的最大特征值控制步长。
        # 直接用 K^-1 再作欧氏圆盘裁剪不是 K 度量下的约束解，会使滑动方向偏转。
        largest = .5 * (k1 + k2 + math.hypot(k1 - k2, 2 * k12))
        new1 = constraint.accumulated_t1 - vt1 / largest
        new2 = constraint.accumulated_t2 - vt2 / largest
        length = math.hypot(new1, new2)
        if length > max_friction:
            scale = max_friction / length
            new1 *= scale
            new2 *= scale
        applied1 = new1 - constraint.accumulated_t1
        applied2 = new2 - constraint.accumulated_t2
        constraint.accumulated_t1 = new1
        constraint.accumulated_t2 = new2
        if applied1 != 0.0 or applied2 != 0.0:
            _apply_pair_impulse(
                body_a, body_b, constraint.r_a, constraint.r_b,
                vec3.add(vec3.scale(constraint.t1, applied1), vec3.scale(constraint.t2, applied2)),
            )

    # -- 暖启动 -----------------------------------------------------------

    def _position_constraints(self, manifolds, get_body, h):
        """Only prepare penetrating normal constraints; no friction or cache matching."""
        constraints = []
        for manifold in manifolds:
            active = [p for p in manifold.points if p.penetration > self.slop]
            if not active:
                continue
            a, b = get_body(manifold.body_a), get_body(manifold.body_b)
            radii = [body.bounding_radius for body in (a, b) if body.geometry is not None]
            limit = min(.05, .2 * min(radii, default=1))
            for point in active:
                ensure_anchors(point, a, b)
                c = _Constraint()
                c.body_a, c.body_b, c.normal = a, b, point.normal
                c.r_a = quat.quat_rotate_vector(a.orientation, point.local_anchor_a)
                c.r_b = quat.quat_rotate_vector(b.orientation, point.local_anchor_b)
                c.normal_mass = _effective_mass(a, b, c.r_a, c.r_b, c.normal)
                if c.normal_mass <= 1e-12:
                    continue
                c.bias_target = min(self.baumgarte * max(0., point.penetration - self.slop), limit) / h
                constraints.append(c)
        return constraints

    def solve_positions(self, manifolds, get_body, h):
        """Split impulse: virtual velocities correct overlap without changing p or L.

        Called once after the complete substep, never at every TOI probe. No
        restitution, friction or warm-start impulses participate in this solve.
        """
        if self.baumgarte <= 0 or not manifolds:
            return 0
        constraints = self._position_constraints(manifolds, get_body, h)
        if not constraints:
            return 0
        pseudo = {}
        for c in constraints:
            for b in (c.body_a, c.body_b):
                if not b.is_static:
                    pseudo.setdefault(id(b), [b, (0., 0., 0.), (0., 0., 0.)])

        def velocity(b, r):
            state = pseudo.get(id(b))
            return (0., 0., 0.) if state is None else vec3.add(state[1], vec3.cross(state[2], r))

        for _ in range(max(8, self.iterations)):
            for c in constraints:
                vn = vec3.dot(vec3.sub(velocity(c.body_b, c.r_b), velocity(c.body_a, c.r_a)), c.normal)
                new = max(0., c.accumulated_bias + (c.bias_target - vn) / c.normal_mass)
                delta = new - c.accumulated_bias
                c.accumulated_bias = new
                for b, r, sign in ((c.body_a, c.r_a, -1), (c.body_b, c.r_b, 1)):
                    if b.is_static:
                        continue
                    impulse = vec3.scale(c.normal, sign * delta)
                    state = pseudo[id(b)]
                    state[1] = vec3.madd(state[1], impulse, b.inv_mass)
                    state[2] = vec3.add(state[2], b.inverse_inertia_world(vec3.cross(r, impulse)))
        for b, v, omega in pseudo.values():
            b.position = vec3.madd(b.position, v, h)
            # L 固定而姿态改变仍可能抬高转动能。回溯角修正，避免几何修补注入动能。
            # 无可接受的角修正时只保留平移纠正，不修改真实动量来补偿误差。
            energy = .5 * vec3.dot(b.angular_momentum, b.angular_velocity)
            amount = h
            for _ in range(12):
                candidate = quat.quat_integrate(b.orientation, omega, amount)
                next_energy = .5 * vec3.dot(b.angular_momentum,
                                           b.inverse_inertia_world(b.angular_momentum, orientation=candidate))
                if next_energy <= energy + 1e-12 * max(1., energy):
                    b.orientation = candidate
                    break
                amount *= .5
        return len(constraints)

    @property
    def _impulse_cache(self):
        """兼容诊断脚本的只读快照；缓存本体由 ContactManager 管理。"""
        return {key: (e.normal_impulse, e.seen_epoch, e.warm_epoch)
                for key, e in self.contacts.entries.items()}

    def reset_cache(self) -> None:
        """清空跨帧冲量缓存。

        场景重载、物体数量变化、或手动改写位姿之后必须调用，否则会把属于旧状态
        的冲量施加到新状态上——症状是"第一次接触时物体被莫名弹开"。
        """
        self.contacts.clear()

    def begin_substep(self) -> None:
        """进入新的子步。同一个子步内允许多次求解（TOI 切分），但**只能暖启动一次**。

        这一点很关键，也是本实现第一次跑就踩到的坑：TOI 会把一个子步切成若干段，
        每段各求解一次。若每段都暖启动，静置的接触在同一个子步内会被反复施加
        上一子步的支撑冲量——每施加一次就多推一下。实测症状是棱柱落地穿透从
        0.148 mm 反弹到 1.317 mm（物体被"顶"离地面，再落下来）。

        所以缓存里记了"这笔冲量来自哪个子步"：同一子步内的后续求解一律冷启动。
        """
        self.contacts.begin_substep()

    def _warm_start_impulses(self, constraints: list[_Constraint]) -> int:
        """把上次求解的法向累积冲量沿当前法向施加，返回命中的接触点数。

        只处理法向：切向冲量**故意不缓存**，原因见模块文档第 1 条（四点流形的
        切向冲量会累积成力矩，迭代次数有限时收敛不掉，反而更抖）。
        """
        if not self.contacts.entries:
            return 0

        hits = 0
        for constraint in constraints:
            entry = constraint.cache_entry
            if entry is None:
                continue
            if entry.seen_epoch >= self.contacts.epoch or entry.warm_epoch == self.contacts.epoch:
                # 同一子步内已经用过这笔冲量（TOI 把子步切成了多段），
                # 再施加一次就是重复计费
                continue
            impulse = entry.normal_impulse * constraint.h / entry.h
            if impulse <= 0.0:
                continue

            constraint.accumulated_normal = impulse
            _apply_pair_impulse(
                constraint.body_a, constraint.body_b,
                constraint.r_a, constraint.r_b,
                vec3.scale(constraint.normal, impulse),
            )
            hits += 1
            entry.warm_epoch = self.contacts.epoch

        return hits

    def _store_impulses(self, constraints: list[_Constraint]) -> None:
        """把本次求解的法向累积冲量写回缓存，并清理久未出现的键。"""
        self.contacts.store(constraints, constraints[0].h)

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
        elif self.contacts.entries:
            # 运行中把暖启动关掉时必须丢弃旧冲量，否则 A/B 对照会被残留状态污染
            self.reset_cache()

        for _ in range(self.iterations):
            for constraint in constraints:
                self._solve_normal(constraint)
                self._solve_friction(constraint)

        if self.warm_start:
            self._store_impulses(constraints)

        return len(constraints)
