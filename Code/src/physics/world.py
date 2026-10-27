# -*- coding: utf-8 -*-
"""物理世界：把刚体、窄相检测、求解器、积分器串成一步仿真。

单步流程
--------
每个子步（``physics_dt / substeps``）依次执行：

1. **积分速度**：``v += g * h``（只对动态物体）
2. **碰撞检测**：找出所有接触，生成接触流形
3. **约束求解**：顺序冲量迭代修正速度，使接触满足非穿透 + 摩擦
4. **积分位置**：``x += v * h``，姿态用四元数积分再归一化

这个顺序（先速度后位置，即 semi-implicit / symplectic Euler）比显式欧拉
稳定得多：接触冲量在**位置更新之前**就已经反映到速度里，物体不会先穿进去
再被拉出来。代价是它仍然是一阶方法，但对本项目的尺度完全够用。

为什么把检测放在积分速度之后
----------------------------
接触点的"碰撞前接近速度"必须包含本步重力的贡献。若先检测再加重力，
快速下落时恢复系数会被低估，球会越弹越低——一个很隐蔽的 bug。
"""

from __future__ import annotations

import math

from ..mathutils import quaternion as quat
from ..mathutils import vec3
from ..scene.scene_spec import SceneSpec, SimulationSpec
from .body import RigidBody, make_ground_proxy
from .broadphase import BruteForceBroadphase, SpatialHashBroadphase
from .contact import GROUND_BODY
from .narrowphase import (
    GroundPlane,
    detect_contacts,
    is_supported,
    supported_pair,
)
from .solver import SequentialImpulseSolver
from .toi import earliest_toi
from .validate import find_initial_overlaps, format_violations

#: 小于该时长（秒）的剩余时间不再切分，直接推进。
_MIN_TIME = 1e-12

#: 越过预测接触时刻的微小提前量（秒）。
#: 窄相只有在**真实穿透**时才报告接触，所以必须略微越过接触时刻，
#: 让穿透量变成一个可被浮点数表示的正值。1e-8 秒在 20 m/s 下只有 0.2 微米。
_TOI_NUDGE = 1e-8


class PhysicsWorld:
    """一个独立的物理仿真世界。"""

    __slots__ = (
        "bodies",
        "ground",
        "simulation",
        "solver",
        "broadphase",
        "time",
        "step_count",
        "last_event_count",
        "last_candidate_count",
        "_ground_proxy",
        "_name_to_index",
    )

    def __init__(
        self,
        bodies: list[RigidBody],
        ground: GroundPlane | None,
        simulation: SimulationSpec,
    ) -> None:
        """直接构造物理世界。

        一般不建议手工调用，用 ``from_scene`` 更方便。

        参数
        ----
        bodies     : 刚体列表
        ground     : 地面；None 表示自由空间
        simulation : 仿真参数（步长、迭代次数等）
        """
        self.bodies = list(bodies)
        self.ground = ground
        self.simulation = simulation
        self.solver = SequentialImpulseSolver(
            iterations=simulation.iterations,
            baumgarte=simulation.baumgarte,
            slop=simulation.slop,
            restitution_threshold=simulation.restitution_threshold,
        )
        # 宽阶段：决定"哪些物体对需要送进窄相"
        if simulation.broadphase == "brute":
            self.broadphase = BruteForceBroadphase()
        else:
            cell = simulation.broadphase_cell_size
            self.broadphase = SpatialHashBroadphase(
                bodies=self.bodies if self.bodies else None,
                cell_size=cell if cell > 0.0 else None,
            )

        self.time = 0.0
        self.step_count = 0
        #: 最近一个子步内触发了多少次 TOI 事件（用于诊断与性能评估）
        self.last_event_count = 0
        #: 最近一次宽阶段产生的候选对数（用于诊断与加速比分析）
        self.last_candidate_count = 0
        self._ground_proxy = make_ground_proxy()
        self._name_to_index = {b.name: i for i, b in enumerate(self.bodies)}

    # -- 构建 -------------------------------------------------------------

    @classmethod
    def from_scene(cls, scene: SceneSpec) -> "PhysicsWorld":
        """从场景描述构建物理世界。

        参数
        ----
        scene : 已解析的场景（必须包含 ``simulation`` 段且 ``enabled=true``）

        异常
        -----
        ValueError : 未启用仿真，或场景中存在当前窄相尚不支持的几何体。
        """
        sim = scene.simulation
        if not sim.enabled:
            raise ValueError(
                f"场景 '{scene.name}' 没有启用物理仿真；"
                f"请在 JSON 中加入 \"simulation\": {{\"enabled\": true, ...}}"
            )

        bodies: list[RigidBody] = []
        for spec in scene.objects:
            if not is_supported(spec.geometry):
                raise ValueError(
                    f"物体 '{spec.name}' 的几何类型 {spec.geometry.type_name!r} "
                    f"暂未被窄相支持。当前支持球的全部组合，以及球-盒、盒-地面。"
                )
            bodies.append(RigidBody.from_object_spec(spec))

        # 物体对层面的校验：单个几何体都被支持，不代表任意**组合**都被支持。
        # 目前球-球、球-盒、盒-盒、以及各自与地面的组合都已实现，
        # 但保留这一层检查，以便将来加入"单独支持、组合未支持"的几何时
        # 能在加载阶段就拦下，而不是让物体在仿真里静默穿过彼此。
        #
        # 判据用**包围球是否相交**：不相交 ⇒ 必然不接触，可以放行；
        # 相交却无实现时才报错。这样"同一场景里放两个永远不会相遇的物体"仍然可运行。
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                body_i, body_j = bodies[i], bodies[j]
                if supported_pair(body_i.geometry, body_j.geometry):
                    continue
                delta = vec3.sub(body_j.position, body_i.position)
                reach = body_i.bounding_radius + body_j.bounding_radius
                if vec3.length_squared(delta) > reach * reach:
                    continue
                raise ValueError(
                    f"物体 '{body_i.name}'（{body_i.geometry.type_name}）与 "
                    f"'{body_j.name}'（{body_j.geometry.type_name}）之间"
                    f"的碰撞尚未实现，请暂时避免让它们互相接触。"
                )

        ground = GroundPlane(
            z=sim.ground_z,
            friction=sim.ground_friction,
            restitution=sim.ground_restitution,
        )
        world = cls(bodies, ground, sim)
        world.validate_initial_state()
        return world

    # -- 初始状态校验 -----------------------------------------------------

    def validate_initial_state(self) -> None:
        """检查仿真开始时是否存在物体互相嵌入。

        为什么必须拦下来
        ----------------
        初始就嵌入的接触会在第一个子步**同时生效**，求解器只能给出
        "整体同时变形"的解——那对应软材料的物理。对硬物，正确行为是
        接触按时间先后逐个传递。实测仅一微米的重叠就会让三球对撞的结果
        从 ``[0,0,3]`` 跳到 ``[-1,2,2]``。

        异常
        -----
        ValueError : 存在超过 ``overlap_tolerance`` 的初始穿透，
                     且场景没有显式声明 ``allow_initial_overlap``。
        """
        if self.simulation.allow_initial_overlap:
            return

        violations = find_initial_overlaps(
            self.bodies, self.ground, self.simulation.overlap_tolerance
        )
        if violations:
            raise ValueError(format_violations(violations))

    # -- 索引解析 ---------------------------------------------------------

    def get_body(self, index: int) -> RigidBody:
        """把接触流形里的物体索引解析成刚体。

        ``GROUND_BODY(-1)`` 返回静态替身，使求解器无需为地面写分支。
        """
        if index == GROUND_BODY:
            return self._ground_proxy
        return self.bodies[index]

    def body_by_name(self, name: str) -> RigidBody:
        """按名字取刚体，便于测试脚本定位对象。"""
        return self.bodies[self._name_to_index[name]]

    # -- 单步 -------------------------------------------------------------

    def _integrate_velocities(self, h: float) -> None:
        """给动态物体加重力并施加阻尼（显式欧拉的一步）。

        静态物体速度恒为 0，不受重力影响。

        阻尼用指数衰减 ``v *= exp(-k*h)`` 而不是 ``v -= k*v*h``：
        指数形式与步长解耦（改变 dt 不会改变衰减速率），且在 k 很大时
        也不会把速度减成负数。

        为什么需要阻尼
        --------------
        理想刚性球在水平面上**纯滚动**时，接触点速度恰好为零，摩擦冲量为零，
        于是球会以恒定速度永远滚下去。真实世界之所以会停，是滚动阻力起作用。
        这里用角速度衰减近似它：角速度掉了之后纯滚动被破坏、开始打滑，
        摩擦自然就会把平动也刹住——效果与真实滚动阻力一致。
        """
        gx, gy, gz = self.simulation.gravity
        linear_decay = math.exp(-self.simulation.linear_damping * h)
        angular_decay = math.exp(-self.simulation.angular_damping * h)

        for body in self.bodies:
            if body.is_static:
                continue
            body.linear_velocity = (
                (body.linear_velocity[0] + gx * h) * linear_decay,
                (body.linear_velocity[1] + gy * h) * linear_decay,
                (body.linear_velocity[2] + gz * h) * linear_decay,
            )
            body.angular_velocity = vec3.scale(body.angular_velocity, angular_decay)

    def _integrate_positions(self, h: float) -> None:
        """用求解后的速度更新位置与姿态。

        姿态使用四元数一阶积分后归一化：

            q ← normalize(q + 0.5 * h * (0, ω) ⊗ q)

        归一化是必须的，否则模长漂移会让旋转逐渐失真。
        """
        for body in self.bodies:
            if body.is_static:
                continue
            body.position = vec3.madd(body.position, body.linear_velocity, h)
            if body.angular_velocity != (0.0, 0.0, 0.0):
                body.orientation = quat.quat_integrate(
                    body.orientation, body.angular_velocity, h
                )

    def _solve_current_contacts(self, h: float) -> int:
        """检测并求解"此刻"存在的接触，返回参与求解的接触点数。

        参数
        ----
        h : 用于 Baumgarte 位置修正的子步时长（不是实际推进的时间）
        """
        pairs = self.broadphase.candidate_pairs(self.bodies)
        self.last_candidate_count = len(pairs)

        manifolds = detect_contacts(self.bodies, self.ground, pairs)
        if not manifolds:
            return 0
        return self.solver.solve(manifolds, self.get_body, h)

    def substep(self, h: float) -> int:
        """执行一个子步，返回参与求解的接触点数。

        默认走 TOI 分段流程（见 ``_solve_with_toi``）；
        若场景里关闭了 ``use_toi_splitting``，则退化为"一次检测 + 一次联立求解"。
        """
        self._integrate_velocities(h)

        if not self.simulation.use_toi_splitting:
            contact_count = self._solve_current_contacts(h)
            self._integrate_positions(h)
        else:
            contact_count = self._solve_with_toi(h)

        self.time += h
        return contact_count

    def _solve_with_toi(self, h: float) -> int:
        """把子步按预测接触时刻切成若干段，逐段求解。

        分两个阶段：

        **阶段 A —— 处理已存在的接触**
        子步开始时已经贴合的物体（静置、堆叠）不属于"未来事件"，
        必须先解算，否则它们会在这个子步里自由下陷。

        **阶段 B —— 按时间顺序处理未来碰撞**
        反复找出最早的接触时刻，把位置推进到那一刻（略微越过一点点，
        使窄相能检测到正穿透），然后**只解算此刻发生的接触**。

        这样"同一子步内的多次碰撞"就被拆成了有先后顺序的独立求解，
        A→B 解算完、B 带着新速度继续走，才会去撞 C。这正是硬物碰撞的物理次序，
        也是牛顿摆只飞出一个球的原因。

        为什么要限制事件数
        ------------------
        极端情况下（大量物体、密集布置）一个子步内可能有几十次事件，
        逐个切分会显著拖慢仿真。达到上限后剩余时间一次性推进，
        退化为传统做法——误差有界，且不会卡死。

        参数
        ----
        h : 子步时长

        返回
        ----
        int : 本子步内累计求解的接触点数
        """
        total = self._solve_current_contacts(h)

        remaining = h
        events = 0
        max_events = self.simulation.max_events_per_substep

        while remaining > _MIN_TIME and events < max_events:
            hit = earliest_toi(self.bodies, self.ground, remaining)
            if hit is None:
                # 剩余时间内不会再发生碰撞，直接走完
                break

            step_time = min(hit[0] + _TOI_NUDGE, remaining)
            self._integrate_positions(step_time)
            remaining -= step_time

            total += self._solve_current_contacts(h)
            events += 1

        if remaining > _MIN_TIME:
            self._integrate_positions(remaining)

        self.last_event_count = events
        return total

    def step(self) -> int:
        """执行一个完整仿真步（切分为 ``substeps`` 个子步）。

        子步越多，穿透越浅、高速碰撞越不容易漏检，代价是线性增长的计算量。

        返回
        ----
        int : 本步内累计求解的接触点数
        """
        substeps = self.simulation.substeps
        h = self.simulation.physics_dt / substeps

        total = 0
        for _ in range(substeps):
            total += self.substep(h)

        self.step_count += 1
        return total

    # -- 诊断 -------------------------------------------------------------

    def total_kinetic_energy(self) -> float:
        """所有动态物体的动能之和。"""
        return sum(b.kinetic_energy() for b in self.bodies)

    def total_potential_energy(self) -> float:
        """所有动态物体的重力势能之和（以地面高度为零点）。"""
        gz = self.simulation.gravity[2]
        base = self.ground.z if self.ground is not None else 0.0
        energy = 0.0
        for body in self.bodies:
            if body.is_static:
                continue
            # U = -m * g · (x - 地面)，gz 为负时等价于 m·|g|·h
            energy += -body.mass * gz * (body.position[2] - base)
        return energy

    def total_energy(self) -> float:
        """总机械能 = 动能 + 势能。用于验证"能量不得凭空增加"。"""
        return self.total_kinetic_energy() + self.total_potential_energy()

    def max_penetration(self) -> float:
        """当前所有接触中最深的穿透深度（米）。

        静置阶段这个值应当稳定在 ``slop`` 附近，若有持续增长说明
        Baumgarte 不足或迭代次数太少。
        """
        if self.ground is None:
            return 0.0
        manifolds = detect_contacts(self.bodies, self.ground)
        if not manifolds:
            return 0.0
        return max(m.deepest.penetration for m in manifolds)

    def max_speed(self) -> float:
        """所有动态物体中最大的质心速度，用于判定是否已静置。"""
        return max((b.speed() for b in self.bodies if not b.is_static), default=0.0)

    def check_finite(self) -> None:
        """校验所有状态量都是有限值。

        数值爆炸（NaN / inf）在视觉上表现为"物体瞬间消失"，不加这个检查
        很难定位是哪一步出的问题，所以测试里会主动调用。

        异常
        -----
        ValueError : 检测到非有限数值，并指出是哪个物体的哪个量。
        """
        for body in self.bodies:
            for field_name in ("position", "orientation", "linear_velocity", "angular_velocity"):
                value = getattr(body, field_name)
                for component in value:
                    if not math.isfinite(component):
                        raise ValueError(
                            f"刚体 '{body.name}' 的 {field_name} 出现非有限数值：{value}"
                        )

    # -- 位姿导出 ---------------------------------------------------------

    def snapshot(self) -> list[dict]:
        """导出当前所有刚体的位姿，供渲染脚本使用。"""
        return [body.to_pose() for body in self.bodies]

    def run(self, duration: float | None = None, fps: int | None = None) -> dict:
        """运行仿真并导出位姿序列。

        参数
        ----
        duration : 总时长（秒）；默认取 ``simulation.duration``
        fps      : 导出帧率；默认取 ``simulation.render_fps``

        返回
        ----
        dict : 形如 ``{"fps": 24, "frames": [{"frame": 0, "poses": [...]}, ...]}``
               的位姿序列，可直接 json.dump 给渲染脚本
        """
        sim = self.simulation
        total_duration = sim.duration if duration is None else float(duration)
        frame_rate = sim.render_fps if fps is None else int(fps)

        # 每个渲染帧之间要跑多少个物理步
        steps_per_frame = max(1, int(round(1.0 / (frame_rate * sim.physics_dt))))
        total_steps = int(round(total_duration / sim.physics_dt))

        frames: list[dict] = [
            {"frame": 0, "time": 0.0, "poses": self.snapshot()}
        ]

        for step_index in range(1, total_steps + 1):
            self.step()
            if step_index % steps_per_frame == 0:
                frame_index = step_index // steps_per_frame
                frames.append({
                    "frame": frame_index,
                    "time": step_index * sim.physics_dt,
                    "poses": self.snapshot(),
                })

        return {
            "fps": frame_rate,
            "duration": total_duration,
            "steps_per_frame": steps_per_frame,
            "physics_dt": sim.physics_dt,
            "frames": frames,
        }

    def __repr__(self) -> str:
        """调试友好的一行摘要。"""
        return (
            f"PhysicsWorld(bodies={len(self.bodies)}, "
            f"t={self.time:.3f}s, steps={self.step_count})"
        )
