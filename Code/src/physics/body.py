# -*- coding: utf-8 -*-
"""刚体状态与基本操作。

``RigidBody`` 保存的是**仿真状态**（位置、姿态、速度），而不是场景描述。
场景描述（``ObjectSpec``）是只读的输入，刚体是每帧可变的输出。

状态量与动量
------------
刚体用 6 个自由度描述：

- 位置 ``position``              : 质心在世界坐标中的位置
- 姿态 ``orientation``           : 单位四元数
- 线速度 ``linear_velocity``     : 质心速度（世界坐标）
- 角速度 ``angular_velocity``    : **世界坐标系**下的角速度

为什么惯量存"局部对角 + 逆"
----------------------------
四种参数化几何都高度对称，惯量张量在**局部坐标系**下是对角阵。
因此只需存 3 个数，世界坐标下的逆惯量作用通过
``R · diag(1/I) · Rᵀ · v`` 现算。这样每步只多两次旋转，却省掉
一次 3×3 矩阵求逆以及"惯量随时间漂移"的数值问题。

静态物体
--------
``is_static=True`` 时 ``inv_mass = 0``、逆惯量为零向量。
求解器里不需要为静态物体写任何分支——所有用到 ``inv_mass`` 的地方
乘 0 自然就"推不动"，方程形式完全统一。
"""

from __future__ import annotations

from ..geom.geometry import CompoundGeometry, GeometrySpec
from ..geom.inertia import compute_inertia
from ..mathutils import quaternion as quat
from ..mathutils import vec3
from ..scene.scene_spec import ObjectSpec


class RigidBody:
    """一个刚体的完整仿真状态。"""

    __slots__ = (
        "name",
        "geometry",
        "position",
        "orientation",
        "linear_velocity",
        "angular_velocity",
        "mass",
        "inv_mass",
        "inertia_local",
        "inv_inertia_local",
        "restitution",
        "friction",
        "is_static",
    )

    def __init__(
        self,
        name: str,
        geometry: GeometrySpec | None,
        position=vec3.zeros(),
        orientation=quat.quat_identity(),
        mass: float = 1.0,
        inertia_local=(1.0, 1.0, 1.0),
        restitution: float = 0.25,
        friction: float = 0.5,
        is_static: bool = False,
    ) -> None:
        """构造刚体。

        参数
        ----
        name             : 物体名（导出位姿序列时作为稳定 ID）
        geometry         : 几何体；静态替身可以为 None
        position         : 初始质心位置
        orientation      : 初始姿态四元数
        mass             : 质量（kg）；静态物体传任意正值即可，内部会置零
        inertia_local    : 局部坐标系下的转动惯量对角元
        restitution      : 恢复系数（单侧材质值，接触时再与对方合成）
        friction         : 摩擦系数（同上）
        is_static        : 是否静态（质量无穷大）
        """
        self.name = str(name)
        self.geometry = geometry
        self.position = (float(position[0]), float(position[1]), float(position[2]))
        self.orientation = quat.quat_normalize(orientation)
        self.is_static = bool(is_static)
        self.restitution = float(restitution)
        self.friction = float(friction)

        if self.is_static:
            # 质量无穷大 ⇒ 逆质量为 0，任何冲量都产生不了速度变化
            self.mass = float("inf")
            self.inv_mass = 0.0
            self.inertia_local = (float("inf"),) * 3
            self.inv_inertia_local = (0.0, 0.0, 0.0)
        else:
            if mass <= 0.0:
                raise ValueError(f"刚体 '{name}' 的质量必须为正数，收到：{mass}")
            self.mass = float(mass)
            self.inv_mass = 1.0 / float(mass)
            self.inertia_local = (
                float(inertia_local[0]),
                float(inertia_local[1]),
                float(inertia_local[2]),
            )
            # 惯量理论上恒为正；若算出 0 说明几何退化，此处显式拦下
            for i, v in enumerate(self.inertia_local):
                if v <= 0.0:
                    raise ValueError(
                        f"刚体 '{name}' 的第 {i} 轴转动惯量非正（{v}），几何可能退化"
                    )
            self.inv_inertia_local = tuple(1.0 / v for v in self.inertia_local)

        self.linear_velocity = (0.0, 0.0, 0.0)
        self.angular_velocity = (0.0, 0.0, 0.0)

    # -- 构造 -------------------------------------------------------------

    @classmethod
    def from_object_spec(cls, spec: ObjectSpec) -> "RigidBody":
        """根据场景物体描述创建刚体。

        质量与惯量都从 ``spec`` 里取：``mass`` 在解析阶段已按
        "体积 × 1000" 兜底过，惯量由 ``compute_inertia`` 按几何类型算出。
        """
        explicit = getattr(spec, "inertia", None)

        if explicit is not None:
            # 资产已经把质量分布算准了（可能是空心、可能含非均匀密度），
            # 从碰撞几何反推会丢掉这些信息：空心壳的碰撞代理是实心外包盒，
            # 惯量会被放大 1.25 倍，而且这个误差不会随精度提高而消失。
            #
            # 此时不叠加 ``inertia_frame()``：资产里的主惯量本就是相对
            # "质心在工作坐标系原点、轴与主轴重合" 那个坐标系给出的，
            # 而碰撞点也已经存在同一坐标系里，两者已经对齐。
            inertia = explicit
            frame = (0.0, 0.0, 0.0, 1.0)
        else:
            inertia = compute_inertia(spec.geometry, spec.mass)
            # 惯量主轴并进姿态：R_最终 = R_场景 ⊗ Q_主轴。
            #
            # 任意凸多面体的惯量张量不是对角阵，而引擎只存对角值。
            # 把几何体旋转到主轴系之后，``I_world = (R·Q)Λ(R·Q)ᵀ`` 依然成立，
            # 所以求解器一行都不用改。参数化图元的 Q 是单位四元数，结果不变。
            #
            # 顺序不能反：四元数乘法是"先施加右、再施加左"，
            # 写成本文这样才是先转主轴、再按场景姿态摆放。
            frame = spec.geometry.inertia_frame()

        orientation = quat.quat_multiply(spec.transform.orientation, frame)

        body = cls(
            name=spec.name,
            geometry=spec.geometry,
            position=spec.transform.position,
            orientation=orientation,
            mass=spec.mass,
            inertia_local=inertia,
            restitution=spec.physics.restitution,
            friction=spec.physics.friction,
            is_static=spec.physics.is_static,
        )
        body.linear_velocity = tuple(spec.physics.linear_velocity)
        body.angular_velocity = tuple(spec.physics.angular_velocity)
        return body

    # -- 几何辅助 ---------------------------------------------------------

    @property
    def bounding_radius(self) -> float:
        """包围球半径，供宽阶段快速剔除与地面筛选使用。"""
        if self.geometry is None:
            return 0.0
        return self.geometry.bounding_radius()

    def inverse_inertia_world(self, vector) -> tuple[float, float, float]:
        """对世界坐标向量施加世界坐标下的逆惯量矩阵。

        即计算 ``I_world⁻¹ · v``，其中 ``I_world = R · I_local · Rᵀ``。
        展开后等价于：把 ``v`` 转到局部坐标、逐分量除以惯量、再转回世界。

        求解器里每个接触点、每次迭代都要调用它，是本模块最热的一小段。
        """
        if self.is_static:
            return (0.0, 0.0, 0.0)

        inv_q = quat.quat_conjugate(self.orientation)
        local = quat.quat_rotate_vector(inv_q, vector)
        scaled = (
            local[0] * self.inv_inertia_local[0],
            local[1] * self.inv_inertia_local[1],
            local[2] * self.inv_inertia_local[2],
        )
        return quat.quat_rotate_vector(self.orientation, scaled)

    def velocity_at(self, r) -> tuple[float, float, float]:
        """返回刚体上"相对质心位移为 ``r``"的那一点的速度。

        公式：``v_point = v_cm + ω × r``。

        接触点通常不在质心上，力矩来自 ``r × impulse``，所以接触点的
        速度必须用这个函数算，直接用质心速度会漏掉旋转贡献——表现就是
        "物体被斜着撞到却不转"。
        """
        return vec3.add(self.linear_velocity, vec3.cross(self.angular_velocity, r))

    def apply_impulse(self, impulse, r) -> None:
        """在质心位移 ``r`` 处施加冲量 ``impulse``。

        效果：

        - 线速度增加 ``impulse / m``
        - 角速度增加 ``I⁻¹ (r × impulse)``

        参数
        ----
        impulse : 冲量向量（N·s）
        r       : 作用点相对质心的位移
        """
        if self.is_static:
            return
        self.linear_velocity = vec3.madd(self.linear_velocity, impulse, self.inv_mass)
        torque = vec3.cross(r, impulse)
        self.angular_velocity = vec3.add(
            self.angular_velocity, self.inverse_inertia_world(torque)
        )

    # -- 能量与诊断 -------------------------------------------------------

    def kinetic_energy(self) -> float:
        """返回平动动能与转动动能之和。

        用于"能量不得凭空增加"的数值验证——一个正确的求解器在
        无外力且恢复系数 ≤ 1 时，总机械能只应下降不应上升。
        """
        if self.is_static:
            return 0.0
        linear = 0.5 * self.mass * vec3.length_squared(self.linear_velocity)

        # 转动动能 = 1/2 ωᵀ I ω；在局部坐标下 I 是对角阵，逐分量相乘即可
        inv_q = quat.quat_conjugate(self.orientation)
        local_w = quat.quat_rotate_vector(inv_q, self.angular_velocity)
        angular = 0.5 * (
            self.inertia_local[0] * local_w[0] * local_w[0]
            + self.inertia_local[1] * local_w[1] * local_w[1]
            + self.inertia_local[2] * local_w[2] * local_w[2]
        )
        return linear + angular

    def speed(self) -> float:
        """质心速度模长，用于静置判定与日志。"""
        return vec3.length(self.linear_velocity)

    def to_pose(self) -> dict:
        """导出位姿，供渲染脚本使用。"""
        return {
            "name": self.name,
            "position": list(self.position),
            "orientation": list(self.orientation),
        }

    def __repr__(self) -> str:
        """调试友好的一行摘要。"""
        flag = "static" if self.is_static else f"m={self.mass:.3f}"
        return (
            f"RigidBody({self.name!r}, {flag}, "
            f"pos=({self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}))"
        )


class PartView:
    """复合几何里某一个部件在世界坐标下的位姿。

    碰撞/TOI 函数只读 ``geometry``、``position``、``orientation``（TOI 还要
    两个速度），所以一个部件可以直接伪装成"一个刚体"喂给现有分派逻辑，
    不必给每种几何再写一遍复合体版本。

    速度为什么要算 ``v + ω × r``
    ----------------------------
    部件的质心不在刚体质心上。刚体转动时部件在平移，TOI 若用整体质心速度，
    一个旋转中的哑铃会算不出任何接近——高速旋转的物体就会穿模。
    """

    __slots__ = (
        "geometry",
        "position",
        "orientation",
        "linear_velocity",
        "angular_velocity",
        "friction",
        "restitution",
        "is_static",
    )

    def __init__(self, body: "RigidBody", part) -> None:
        offset = quat.quat_rotate_vector(body.orientation, part.position)
        self.geometry = part.geometry
        self.position = vec3.add(body.position, offset)
        self.orientation = quat.quat_multiply(body.orientation, part.orientation)
        self.linear_velocity = vec3.add(
            body.linear_velocity, vec3.cross(body.angular_velocity, offset)
        )
        self.angular_velocity = body.angular_velocity
        self.friction = body.friction
        self.restitution = body.restitution
        self.is_static = body.is_static

    @property
    def bounding_radius(self) -> float:
        """包围球半径。

        必须与 ``RigidBody.bounding_radius`` 同名同义：宽阶段对复合体是**逐部件**
        算 AABB 的（``world_aabb`` 会拿 ``PartView`` 再递归一次），少了这个属性，
        "复合体 + TOI 切分"这条路径会在运行期抛 ``AttributeError``。
        """
        return self.geometry.bounding_radius()


def iter_parts(body) -> list:
    """列出参与碰撞的部件：``[(部件编号, 部件视图), ...]``。

    普通刚体就是"编号 0 的自己"——这样上层代码不需要为复合体写两条路径。
    对 ``RigidBody`` 直接返回它自身，零分配。
    """
    geometry = getattr(body, "geometry", None)
    if isinstance(geometry, CompoundGeometry):
        return [(index, PartView(body, part)) for index, part in enumerate(geometry.parts)]
    return [(0, body)]


def make_ground_proxy(name: str = "__ground__") -> RigidBody:
    """创建地面用的静态替身刚体。

    求解器对静态物体没有任何分支，地面就是以"逆质量为零、速度恒为零"
    的刚体身份参与计算。这样 ``GROUND_BODY(-1)`` 这个伪索引可以直接
    查表得到它，接触方程形式与普通物体完全一致。
    """
    return RigidBody(
        name=name,
        geometry=None,
        position=(0.0, 0.0, 0.0),
        mass=1.0,
        is_static=True,
    )
