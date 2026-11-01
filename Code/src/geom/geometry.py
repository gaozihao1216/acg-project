# -*- coding: utf-8 -*-
"""参数化几何体定义。

本模块用**闭式参数**描述几何体，而不是保存顶点网格。这样做的好处：

1. 体积、惯量、包围球半径都能解析求出，数值精确且不依赖网格分辨率；
2. 场景 JSON 可以短到一行，便于做参数扫描；
3. 碰撞检测可以走解析路径（球-球、球-盒），比网格碰撞快几个数量级。

几何体尺寸即**最终物理尺寸**（单位：米），不使用父节点缩放。
"""

from __future__ import annotations

import math
from typing import Sequence

from ..mathutils import quaternion as quat
from .convexhull import convex_hull
from .massprops import from_mesh as _from_mesh_mass_properties
from .massprops import matrix3_to_quaternion


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot3(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize3(a):
    n = math.sqrt(_dot3(a, a))
    if n <= 0.0:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


class GeometrySpec:
    """参数化几何体的抽象基类。

    子类必须实现 ``type_name``、``volume()``、``bounding_radius()``
    以及 ``to_dict()``，并注册到 ``from_dict`` 的分发表中。
    """

    #: 出现在场景 JSON ``geometry.type`` 字段中的类型名
    type_name: str = "abstract"

    def volume(self) -> float:
        """返回几何体体积（立方米）。

        体积在刚体动力学里用于按密度反推质量，也用于复合体的惯量配比。
        """
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """返回以质心为球心、能包住整个几何体的最小球半径。

        碰撞宽阶段（空间哈希 / 包围球剔除）用它做快速否定测试：
        两球心距大于半径和时可以安全跳过窄相检测。
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典，便于记录实验参数。"""
        raise NotImplementedError

    def inertia_frame(self) -> tuple[float, float, float, float]:
        """几何坐标系 → 惯量主轴系的旋转四元数。

        参数化图元（球、盒、柱、胶囊）的惯量张量在其自身坐标系下**已经是对角阵**，
        所以默认返回单位四元数，不需要任何额外变换。

        任意凸多面体则不是：它的张量有非对角项（惯量积），必须旋转到主轴系
        才能用"三个对角值"表示。``ConvexHullGeometry`` 会重写这个方法。

        ``RigidBody`` 用 ``spec_orientation ⊗ inertia_frame()`` 作为最终姿态，
        于是 ``I_world = (R·Q)Λ(R·Q)ᵀ``，引擎的对角惯量实现保持不变。
        """
        return (1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(data: dict) -> "GeometrySpec":
        """根据场景 JSON 的 ``geometry`` 字段构造具体几何体。

        参数
        ----
        data : 形如 ``{"type": "box", "size": [1, 1, 1]}`` 的字典

        异常
        ----
        ValueError : 类型未知，或必填参数缺失 / 非法。
        """
        if not isinstance(data, dict):
            raise ValueError(f"geometry 必须是对象，收到：{type(data).__name__}")

        type_name = data.get("type")
        if type_name is None:
            raise ValueError("geometry 缺少必填字段 'type'")

        builder = _GEOMETRY_REGISTRY.get(str(type_name).lower())
        if builder is None:
            supported = ", ".join(sorted(_GEOMETRY_REGISTRY))
            raise ValueError(
                f"不支持的几何类型 '{type_name}'；当前支持：{supported}"
            )
        return builder(data)


# ---------------------------------------------------------------------------
# 具体几何体实现
# ---------------------------------------------------------------------------

class BoxGeometry(GeometrySpec):
    """轴对齐长方体，尺寸为三个轴向的**全长**（不是半长）。

    ``size = [2, 1, 0.5]`` 表示 x 方向 2 米、y 方向 1 米、z 方向 0.5 米。
    """

    type_name = "box"

    __slots__ = ("size",)

    def __init__(self, size: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
        """构造长方体。

        参数
        ----
        size : 三个轴向的全长，必须全部为正数
        """
        sx, sy, sz = (float(size[0]), float(size[1]), float(size[2]))
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
            raise ValueError(f"box.size 必须全部为正数，收到：{[sx, sy, sz]}")
        self.size = (sx, sy, sz)

    def volume(self) -> float:
        """长方体体积 = 三边乘积。"""
        sx, sy, sz = self.size
        return sx * sy * sz

    def bounding_radius(self) -> float:
        """包围球半径 = 体对角线的一半。"""
        sx, sy, sz = self.size
        return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)

    def to_dict(self) -> dict:
        """导出为 ``{"type": "box", "size": [...]}``。"""
        return {"type": self.type_name, "size": list(self.size)}


class SphereGeometry(GeometrySpec):
    """球体，由半径唯一确定。"""

    type_name = "sphere"

    __slots__ = ("radius",)

    def __init__(self, radius: float = 0.5) -> None:
        """构造球体。

        参数
        ----
        radius : 半径，必须为正数
        """
        r = float(radius)
        if r <= 0.0:
            raise ValueError(f"sphere.radius 必须为正数，收到：{r}")
        self.radius = r

    def volume(self) -> float:
        """球体体积 = 4/3 π r³。"""
        return (4.0 / 3.0) * math.pi * self.radius ** 3

    def bounding_radius(self) -> float:
        """球体的包围球半径就是自身半径。"""
        return self.radius

    def to_dict(self) -> dict:
        """导出为 ``{"type": "sphere", "radius": r}``。"""
        return {"type": self.type_name, "radius": self.radius}


class CylinderGeometry(GeometrySpec):
    """实心圆柱，轴线沿**局部 Z 轴**，底面中心位于 -h/2、顶面位于 +h/2。"""

    type_name = "cylinder"

    __slots__ = ("radius", "height")

    def __init__(self, radius: float = 0.5, height: float = 1.0) -> None:
        """构造圆柱。

        参数
        ----
        radius : 截面半径，必须为正数
        height : 圆柱高度（全长），必须为正数
        """
        r, h = float(radius), float(height)
        if r <= 0.0:
            raise ValueError(f"cylinder.radius 必须为正数，收到：{r}")
        if h <= 0.0:
            raise ValueError(f"cylinder.height 必须为正数，收到：{h}")
        self.radius = r
        self.height = h

    def volume(self) -> float:
        """圆柱体积 = π r² h。"""
        return math.pi * self.radius ** 2 * self.height

    def bounding_radius(self) -> float:
        """包围球半径 = 半高与半径构成的斜边长度。"""
        r, h = self.radius, self.height
        return math.sqrt(r * r + (0.5 * h) ** 2)

    def to_dict(self) -> dict:
        """导出为 ``{"type": "cylinder", "radius": r, "height": h}``。"""
        return {"type": self.type_name, "radius": self.radius, "height": self.height}


class CapsuleGeometry(GeometrySpec):
    """胶囊体：中间是圆柱、两端各接一个半球，轴线沿**局部 Z 轴**。

    参数 ``height`` 指的是**中间圆柱段**的高度，不含两端半球。
    因此胶囊总高度为 ``height + 2 * radius``——这一点在写场景时容易搞错，
    所以名字上刻意与 ``CylinderGeometry.height`` 保持一致语义。
    """

    type_name = "capsule"

    __slots__ = ("radius", "height")

    def __init__(self, radius: float = 0.25, height: float = 1.0) -> None:
        """构造胶囊体。

        参数
        ----
        radius : 半球半径，同时也是圆柱半径
        height : 中间圆柱段高度（不含两端半球）
        """
        r, h = float(radius), float(height)
        if r <= 0.0:
            raise ValueError(f"capsule.radius 必须为正数，收到：{r}")
        if h < 0.0:
            raise ValueError(f"capsule.height 不能为负，收到：{h}")
        self.radius = r
        self.height = h

    def total_height(self) -> float:
        """胶囊的总高度 = 圆柱段 + 两端半球。"""
        return self.height + 2.0 * self.radius

    def volume(self) -> float:
        """体积 = 圆柱段体积 + 一个完整球体的体积（两个半球合成一球）。"""
        r, h = self.radius, self.height
        return math.pi * r * r * h + (4.0 / 3.0) * math.pi * r ** 3

    def bounding_radius(self) -> float:
        """包围球半径 = 半高 + 半径，球心仍在胶囊质心。"""
        return 0.5 * self.height + self.radius

    def to_dict(self) -> dict:
        """导出为 ``{"type": "capsule", "radius": r, "height": h}``。"""
        return {"type": self.type_name, "radius": self.radius, "height": self.height}


# ---------------------------------------------------------------------------
# 任意凸多面体
# ---------------------------------------------------------------------------

class ConvexHullGeometry(GeometrySpec):
    """任意凸多面体，由**点集凸包**生成。

    这是"不规则物体"进入碰撞系统的入口。输入只是一堆点，内部会
    求凸包、算质量属性、并归一化到主轴系。

    三件必须在构造时做掉的事
    ------------------------
    1. **质心平移到原点**。刚体的位置指的是质心。若网格原点不在质心，
       物体会绕着一个偏心的点旋转——表现为"甩尾"，不报错，极难查。
    2. **旋转到主轴系**。任意形状的惯量张量有非对角项，而引擎只存对角值。
       把主轴旋转 Q 并进姿态后 ``I_world = (R·Q)Λ(R·Q)ᵀ`` 依然成立，
       所以引擎一行都不用改。``inertia_frame()`` 给出这个 Q。
    3. **面法向与棱表**。SAT 需要它们，且必须与顶点同坐标系。

    代价上限
    --------
    SAT 的棱-棱轴数是 ``E_A·E_B``。64 面凸包约 96 条棱，两两相交是 9216 条轴，
    可以接受；512 面凸包是 59 万条轴，一次查询就能让仿真失去响应。
    所以 ``max_faces`` 默认 64，**在加载阶段就拒绝**——而不是等仿真卡死。

    参数
    ----
    points    : 任意点集（内部点会被凸包自动剔除）
    max_faces : 允许的最大面数，超过则报错。碰撞代理应当用粗网格。
    """

    type_name = "hull"

    __slots__ = (
        "source_points",
        "vertices",
        "faces",
        "face_normals",
        "face_offsets",
        "edges",
        "triangles",
        "_volume",
        "_radius",
        "_unit_moments",
        "_frame",
        "com_offset",
    )

    def __init__(self, points: Sequence[Sequence[float]], max_faces: int = 64) -> None:
        raw = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        if not raw:
            raise ValueError("ConvexHullGeometry 的点集不能为空")

        poly = convex_hull(raw)
        if len(poly.faces) > int(max_faces):
            raise ValueError(
                f"凸包有 {len(poly.faces)} 个面，超过上限 {max_faces}。"
                f"SAT 的棱-棱轴数是 E²，面数过多会让碰撞检测失去响应。"
                f"请先对网格做简化（碰撞代理通常 8–32 面即可），"
                f"或显式调高 max_faces 并接受相应开销。"
            )

        # 单位密度下的质量属性 → 完整张量 + 质心 + 体积
        props = _from_mesh_mass_properties(poly.vertices, poly.triangles, mass=1.0)
        moments, axes = props.principal()

        # axes 的**列**是主轴在几何坐标系下的方向，即 Q 满足 I = Q Λ Qᵀ。
        #
        # 这里有两个方向相反、但都必须用对的旋转，混用会让世界逆惯量错得离谱：
        #   · 顶点 → 主轴坐标:  v' = Qᵀ v        （用 Q 的共轭）
        #   · 姿态复合:         R_最终 = R_场景 · Q  （用 Q 本身）
        # 早先两处都用了 Qᵀ，结果引擎算出的逆惯量与完整张量差 7.24。
        self._frame = matrix3_to_quaternion(axes)
        to_principal = quat.quat_conjugate(self._frame)

        center = props.center_of_mass
        self.com_offset = center

        # 平移 + 旋转到主轴系
        vertices = []
        for v in poly.vertices:
            shifted = (v[0] - center[0], v[1] - center[1], v[2] - center[2])
            vertices.append(quat.quat_rotate_vector(to_principal, shifted))
        self.vertices = vertices

        # 面与法向在**新坐标系**下重算，保证自洽
        self.faces = [list(face) for face in poly.faces]
        self.face_normals = []
        self.face_offsets = []
        for face in self.faces:
            a = self.vertices[face[0]]
            b = self.vertices[face[1]]
            c = self.vertices[face[2]]
            normal = _normalize3(_cross3((b[0] - a[0], b[1] - a[1], b[2] - a[2]),
                                         (c[0] - a[0], c[1] - a[1], c[2] - a[2])))
            self.face_normals.append(normal)
            self.face_offsets.append(_dot3(normal, a))

        self.edges = list(poly.edges)
        self.triangles = list(poly.triangles)
        self._volume = props.volume
        self._unit_moments = moments
        self._radius = max(
            math.sqrt(_dot3(v, v)) for v in self.vertices
        )
        self.source_points = raw

    def volume(self) -> float:
        """凸包体积（散度定理算出，精确）。"""
        return self._volume

    def bounding_radius(self) -> float:
        """以质心为球心的包围球半径。"""
        return self._radius

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """给定质量下的**主惯量**（对角）。

        本对象存的是单位质量的主惯量，惯量与质量成正比，所以直接乘即可。
        """
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        return tuple(mass * value for value in self._unit_moments)

    def inertia_frame(self) -> tuple[float, float, float, float]:
        """几何坐标系 → 主轴系的旋转四元数。

        刚体的 ``orientation`` 必须是 ``spec_orientation ⊗ 本四元数``，
        否则世界逆惯量会算错——而症状只是"转起来有点怪"。
        """
        return self._frame

    def to_dict(self) -> dict:
        """导出为点集形式（凸包会在加载时重新构造）。"""
        return {
            "type": "hull",
            "points": [[v[0], v[1], v[2]] for v in self.source_points],
        }


# ---------------------------------------------------------------------------
# 组合体（解析路线的"拼装建模"）
# ---------------------------------------------------------------------------

class CompoundPart:
    """复合几何中的一个部件：几何 + 相对复合体质心的位姿。"""

    __slots__ = ("geometry", "position", "orientation")

    def __init__(
        self,
        geometry: GeometrySpec,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        orientation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        if not isinstance(geometry, GeometrySpec):
            raise TypeError(
                f"部件几何必须是 GeometrySpec，收到：{type(geometry).__name__}"
            )
        self.geometry = geometry
        self.position = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
        self.orientation = quat.quat_normalize(
            (
                float(orientation[0]),
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            )
        )

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典。"""
        return {
            "geometry": self.geometry.to_dict(),
            "position": list(self.position),
            "orientation": list(self.orientation),
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"CompoundPart({self.geometry.type_name} @ "
            f"{tuple(round(v, 4) for v in self.position)})"
        )


class CompoundGeometry(GeometrySpec):
    """由若干基本几何体拼成的刚体 —— "解析组合建模"的入口。

    为什么不是凸包
    -------------
    凸包只能表示**凸**的**多面体**。L 形支架、人形、带凹槽的夹具都不凸：
    取凸包会把凹槽填平，碰撞体积凭空变大，物体悬在半空；离散成网格又丢掉
    解析精度、且没有闭式惯量。复合体保留每个部件的原始解析形状，碰撞时逐
    部件求交、结果取并集，体积/惯量/包围球全部精确。

    质量属性怎么算
    --------------
    按**单位密度**处理：部件质量正比于自身体积，于是

    - 质心 ``c = Σ V_i p_i / Σ V_i``
    - 张量 ``I = Σ [ V_i·R_i Λ_i R_iᵀ + V_i(|d_i|²E - d_i d_iᵀ) ]``，
      ``d_i = p_i - c``，后一项是**平行轴定理**

    得到的是完整 3×3 张量（一般含惯量积），随后做主轴分解，把主轴旋转
    ``Q = inertia_frame()`` 并进姿态——和凸包完全同一套机制，求解器不用改。
    部件的位姿也一并平移到质心、旋到主轴系，于是碰撞代码拿到的就是最终坐标。

    质量分布的可调性
    ----------------
    现在只按体积分配。想做"底座是钢、头是木头"，需要给部件单独的密度，
    那是下一步（场景里加 ``density``），不影响这里的接口。
    """

    type_name = "compound"

    __slots__ = (
        "parts",
        "com_offset",
        "_volume",
        "_radius",
        "_unit_moments",
        "_frame",
    )

    def __init__(self, parts: Sequence[CompoundPart]) -> None:
        if not parts:
            raise ValueError("复合几何至少需要一个部件")
        for part in parts:
            if not isinstance(part, CompoundPart):
                raise TypeError(
                    f"复合几何的部件必须是 CompoundPart，收到：{type(part).__name__}"
                )

        # 延迟导入：inertia 反向依赖本模块，模块级导入会成环
        from .inertia import compute_inertia

        volumes = [float(part.geometry.volume()) for part in parts]
        total_volume = sum(volumes)
        if total_volume <= 0.0:
            raise ValueError(f"复合几何的总体积必须为正，收到：{total_volume}")

        # --- 1. 质心（按体积加权）---
        center = [0.0, 0.0, 0.0]
        for part, volume in zip(parts, volumes):
            for axis in range(3):
                center[axis] += part.position[axis] * volume
        center = (
            center[0] / total_volume,
            center[1] / total_volume,
            center[2] / total_volume,
        )

        # --- 2. 单位密度下的完整张量 ---
        tensor = [[0.0, 0.0, 0.0] for _ in range(3)]
        for part, volume in zip(parts, volumes):
            # 部件本身的主惯量（单位质量）与它的主轴
            moments = compute_inertia(part.geometry, 1.0)
            rotation = quat.quat_multiply(part.orientation, part.geometry.inertia_frame())
            axes = [
                quat.quat_rotate_vector(rotation, unit)
                for unit in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            ]
            # I = Σ λ_k u_k u_kᵀ ：用主轴外积展开旋转后的张量
            for index in range(3):
                weight = volume * moments[index]
                for row in range(3):
                    for column in range(3):
                        tensor[row][column] += weight * axes[index][row] * axes[index][column]

            # 平行轴定理
            offset = (
                part.position[0] - center[0],
                part.position[1] - center[1],
                part.position[2] - center[2],
            )
            squared = _dot3(offset, offset)
            for row in range(3):
                for column in range(3):
                    delta = squared if row == column else 0.0
                    tensor[row][column] += volume * (
                        delta - offset[row] * offset[column]
                    )

        # --- 3. 主轴分解 ---
        from .massprops import MassProperties

        properties = MassProperties(
            mass=total_volume,
            center_of_mass=(0.0, 0.0, 0.0),
            inertia=(tuple(tensor[0]), tuple(tensor[1]), tuple(tensor[2])),
            volume=total_volume,
            source="compound",
        )
        if properties.is_diagonal():
            # 张量本来就是对角阵：直接用原坐标轴的顺序，别为了"凑升序"去rotate。
            # 否则单部件的复合体会长出一个 90° 的主轴旋转——物理等价，但姿态
            # 无端被置换，渲染和"和原图元逐位一致"的对照都会被打乱。
            moments = properties.diagonal
            self._frame = (1.0, 0.0, 0.0, 0.0)
        else:
            moments, axes_matrix = properties.principal()
            self._frame = matrix3_to_quaternion(axes_matrix)

        self._unit_moments = tuple(moment / total_volume for moment in moments)
        self._volume = total_volume
        self.com_offset = center

        # --- 4. 部件位姿搬进"质心 + 主轴"坐标系 ---
        to_principal = quat.quat_conjugate(self._frame)
        local_parts = []
        radius = 0.0
        for part in parts:
            shifted = (
                part.position[0] - center[0],
                part.position[1] - center[1],
                part.position[2] - center[2],
            )
            position = quat.quat_rotate_vector(to_principal, shifted)
            orientation = quat.quat_multiply(to_principal, part.orientation)
            local_parts.append(CompoundPart(part.geometry, position, orientation))
            radius = max(
                radius,
                math.sqrt(_dot3(position, position)) + part.geometry.bounding_radius(),
            )
        self.parts = local_parts
        self._radius = radius

    # -- 几何量 -----------------------------------------------------------

    def volume(self) -> float:
        """总体积（各部件之和，重叠部分不扣减）。"""
        return self._volume

    def bounding_radius(self) -> float:
        """以质心为球心、包住所有部件的包围球半径。"""
        return self._radius

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """给定质量下的主惯量（对角）。"""
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        return tuple(mass * value for value in self._unit_moments)

    def inertia_frame(self) -> tuple[float, float, float, float]:
        """几何坐标系 → 惯量主轴系的旋转四元数（复合体一般不是单位旋转）。"""
        return self._frame

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典（部件相对**原始**质心给出）。"""
        parts = []
        for part in self.parts:
            parts.append(
                {
                    "geometry": part.geometry.to_dict(),
                    "position": list(part.position),
                    "orientation": list(part.orientation),
                }
            )
        return {"type": "compound", "parts": parts}

    @classmethod
    def from_dict(cls, data: dict) -> "CompoundGeometry":
        """从场景 JSON 的 ``{"type": "compound", "parts": [...]}`` 构造。"""
        raw_parts = data.get("parts")
        if not raw_parts:
            raise ValueError("compound 至少需要一个 parts 条目")

        parts = []
        for index, item in enumerate(raw_parts):
            if not isinstance(item, dict):
                raise ValueError(f"compound.parts[{index}] 必须是对象")
            geometry_data = item.get("geometry")
            if not isinstance(geometry_data, dict):
                raise ValueError(f"compound.parts[{index}] 缺少 geometry")
            if "orientation_euler_deg" in item:
                rx, ry, rz = item["orientation_euler_deg"]
                orientation = quat.quat_from_euler_degrees(rx, ry, rz)
            else:
                orientation = item.get("orientation", (1.0, 0.0, 0.0, 0.0))
            parts.append(
                CompoundPart(
                    GeometrySpec.from_dict(geometry_data),
                    item.get("position", (0.0, 0.0, 0.0)),
                    orientation,
                )
            )
        return cls(parts)


# ---------------------------------------------------------------------------
# 类型注册表与解析辅助
# ---------------------------------------------------------------------------

#: 类型名 -> 构造函数，供 ``GeometrySpec.from_dict`` 分发使用。
#: 后续若增加楔形/圆台等图元，只需在此登记即可，不需要改解析逻辑。
_GEOMETRY_REGISTRY = {
    "box": lambda d: BoxGeometry(d.get("size", (1.0, 1.0, 1.0))),
    "sphere": lambda d: SphereGeometry(d.get("radius", 0.5)),
    "cylinder": lambda d: CylinderGeometry(
        d.get("radius", 0.5), d.get("height", 1.0)
    ),
    "capsule": lambda d: CapsuleGeometry(d.get("radius", 0.25), d.get("height", 1.0)),
    # 任意凸多面体：输入点集，内部求凸包
    "hull": lambda d: ConvexHullGeometry(
        d["points"], d.get("max_faces", 64)
    ),
    # 组合体：若干基本几何体 + 各自的局部位姿
    "compound": lambda d: CompoundGeometry.from_dict(d),
}


def supported_geometry_types() -> list[str]:
    """返回当前支持的几何类型名列表，供报错信息与文档生成使用。"""
    return sorted(_GEOMETRY_REGISTRY)
