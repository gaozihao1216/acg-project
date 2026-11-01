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
from .revolution import solid_of_revolution as _solid_of_revolution


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


#: 方向向量长度小于它就算"没有方向"，支撑映射退回中心/轴端。
_SUPPORT_EPS = 1e-12

#: 包含测试的绝对容差（米）。边界上一层的采样点算不算材料无所谓。
_CONTAINS_TOL = 1e-9


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
    # 几何查询：支撑映射与包含测试
    # ------------------------------------------------------------------
    #
    # 这两个是**唯一**需要为新几何补的几何接口：
    #   · ``support``   —— GJK / EPA 的全部输入，有了它就有与所有已有几何的碰撞
    #   · ``contains``  —— 复合体的重叠惯量近似（在包围盒里撒点）要用它
    # 两者都只在**局部坐标**里工作，因此与姿态、位置无关，实现都只有几行。

    def support(self, direction):
        """局部坐标下沿 ``direction`` 的最远点（支撑映射）。

        长度为零的方向必须返回一个**确定**的点（不能随浮点抖动翻转），
        否则 GJK 在退化方向上的行为会不稳定。

        非凸几何（组合体）没有支撑映射——它们靠逐部件展开参与碰撞，
        对整体调用它是编程错误，所以这里直接抛 ``TypeError``。
        """
        raise TypeError(f"不支持支撑映射的几何类型：{type(self).__name__}")

    def contains(self, point) -> bool:
        """局部坐标点是否落在几何体**内部或表面**上。

        用绝对容差而不是严格不等式：调用方（重叠惯量的采样器）关心的是
        "这个采样点算不算材料"，边界上的一层薄壳对体积估计没有影响。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 尚未实现 contains()"
        )

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

    def support(self, direction):
        """盒的支撑点就是三个轴向按符号取的角点。

        方向分量恰为 0 时取正侧（不是"任一"），保证结果确定。
        """
        half = (0.5 * self.size[0], 0.5 * self.size[1], 0.5 * self.size[2])
        return (
            half[0] if direction[0] >= 0.0 else -half[0],
            half[1] if direction[1] >= 0.0 else -half[1],
            half[2] if direction[2] >= 0.0 else -half[2],
        )

    def contains(self, point) -> bool:
        """轴对齐盒：三个分量都在半长以内。"""
        half = (0.5 * self.size[0], 0.5 * self.size[1], 0.5 * self.size[2])
        return (
            abs(point[0]) <= half[0] + _CONTAINS_TOL
            and abs(point[1]) <= half[1] + _CONTAINS_TOL
            and abs(point[2]) <= half[2] + _CONTAINS_TOL
        )

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

    def support(self, direction):
        """球的支撑点 = 半径乘单位方向（各向同性，只有一个支撑点）。"""
        length = math.sqrt(_dot3(direction, direction))
        if length <= _SUPPORT_EPS:
            return (0.0, 0.0, 0.0)
        scale = self.radius / length
        return (direction[0] * scale, direction[1] * scale, direction[2] * scale)

    def contains(self, point) -> bool:
        """球：到球心的距离不超过半径。"""
        return _dot3(point, point) <= (self.radius + _CONTAINS_TOL) ** 2

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

    def support(self, direction):
        """柱的支撑点：先按方向 z 分量选端面，再沿径向走到圆周。"""
        half_height = 0.5 * self.height
        z = half_height if direction[2] >= 0.0 else -half_height
        radial = math.hypot(direction[0], direction[1])
        if radial <= _SUPPORT_EPS:
            return (0.0, 0.0, z)
        scale = self.radius / radial
        return (direction[0] * scale, direction[1] * scale, z)

    def contains(self, point) -> bool:
        """柱：径向距离不超过半径，且轴向在半高以内。"""
        if abs(point[2]) > 0.5 * self.height + _CONTAINS_TOL:
            return False
        radial_sq = point[0] * point[0] + point[1] * point[1]
        return radial_sq <= (self.radius + _CONTAINS_TOL) ** 2

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

    def support(self, direction):
        """胶囊 = 线段 ⊕ 球：方向压到某端点，再叠加球的支撑。"""
        half_height = 0.5 * self.height
        end_z = half_height if direction[2] >= 0.0 else -half_height
        length = math.sqrt(_dot3(direction, direction))
        if length <= _SUPPORT_EPS:
            return (0.0, 0.0, end_z)
        scale = self.radius / length
        return (
            direction[0] * scale,
            direction[1] * scale,
            end_z + direction[2] * scale,
        )

    def contains(self, point) -> bool:
        """胶囊：到中心线段（局部 Z 轴上的区间）的距离不超过半径。"""
        half_height = 0.5 * self.height
        clamped_z = max(-half_height, min(half_height, point[2]))
        dx = point[0]
        dy = point[1]
        dz = point[2] - clamped_z
        return dx * dx + dy * dy + dz * dz <= (self.radius + _CONTAINS_TOL) ** 2

    def to_dict(self) -> dict:
        """导出为 ``{"type": "capsule", "radius": r, "height": h}``。"""
        return {"type": self.type_name, "radius": self.radius, "height": self.height}


# ---------------------------------------------------------------------------
# 旋转体：圆锥 / 圆台 / 球冠
# ---------------------------------------------------------------------------
#
# 这三个都是"把半径曲线绕局部 Z 轴转一圈"的实心体，质量属性统一由
# ``revolution.solid_of_revolution`` 闭式积分给出，不采样、不近似。
#
# 它们有一个共同的、容易踩坑的约定：**质心在局部原点**。圆锥的质心不在底面，
# 球冠的质心也不在球心，但 ``RigidBody.inertia_local`` 是关于 ``position``
# （即几何原点）的惯量，而 ``ConvexHullGeometry`` / ``CompoundGeometry`` 都已经
# 把质心平移到原点。若不统一，物体会绕一个偏心的点旋转——不报错，表现为甩尾，
# 极难排查。所以这里把形状整体平移，让质心落在原点，并把平移量记在私有字段里
# 供 ``support`` / ``contains`` / ``bounding_radius`` 使用。


class ConeGeometry(GeometrySpec):
    """实心圆锥，轴线沿局部 Z 轴，**质心在原点**，顶点朝 +Z。

    圆锥质心距底面 ``h/4``、距顶点 ``3h/4``（这个比例经常被写反）。质心被挪到
    原点之后，顶点落在 ``z = +3h/4``，底面圆落在 ``z = -h/4``。

    参数
    ----
    radius : 底面半径
    height : 高（顶点到底面）
    """

    type_name = "cone"

    __slots__ = ("radius", "height", "_base_z", "_apex_z", "_unit_moments")

    def __init__(self, radius: float = 0.5, height: float = 1.0) -> None:
        r, h = float(radius), float(height)
        if r <= 0.0:
            raise ValueError(f"cone.radius 必须为正数，收到：{r}")
        if h <= 0.0:
            raise ValueError(f"cone.height 必须为正数，收到：{h}")
        self.radius = r
        self.height = h

        # 自然坐标：底面在 -h/2、顶点在 +h/2，半径线性收缩到 0
        # r(z) = a + b·z，于是 r² = a² + 2ab·z + b²z²
        a = 0.5 * r
        b = -r / h
        moments = _solid_of_revolution(
            (a * a, 2.0 * a * b, b * b),
            -0.5 * h,
            0.5 * h,
        )
        self._base_z = -0.5 * h - moments.centroid_z
        self._apex_z = 0.5 * h - moments.centroid_z
        self._unit_moments = (
            moments.ixx / moments.volume,
            moments.ixx / moments.volume,
            moments.izz / moments.volume,
        )

    def volume(self) -> float:
        """圆锥体积 = π r² h / 3。"""
        return math.pi * self.radius ** 2 * self.height / 3.0

    def bounding_radius(self) -> float:
        """包围球半径：底面圆周与顶点中离质心更远的那个。"""
        rim = math.sqrt(self.radius ** 2 + self._base_z ** 2)
        return max(rim, abs(self._apex_z))

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """给定质量的主惯量（单位质量值乘质量）。"""
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        return tuple(mass * value for value in self._unit_moments)

    def support(self, direction):
        """支撑点只可能落在顶点或底面圆周上——比较两处即可。"""
        apex = (0.0, 0.0, self._apex_z)
        best = apex
        best_score = direction[2] * self._apex_z

        radial = math.hypot(direction[0], direction[1])
        if radial > _SUPPORT_EPS:
            scale = self.radius / radial
            candidate = (direction[0] * scale, direction[1] * scale, self._base_z)
        else:
            # 方向与轴平行：底面圆上任一点等价，取圆心保证结果确定
            candidate = (0.0, 0.0, self._base_z)

        score = (
            candidate[0] * direction[0]
            + candidate[1] * direction[1]
            + candidate[2] * direction[2]
        )
        return candidate if score > best_score else best

    def contains(self, point) -> bool:
        """圆锥：轴向在底面与顶点之间，且径向不超过该高度的允许半径。"""
        z = point[2]
        if z < self._base_z - _CONTAINS_TOL or z > self._apex_z + _CONTAINS_TOL:
            return False
        span = self._apex_z - self._base_z
        ratio = (self._apex_z - z) / span
        allowed = self.radius * max(0.0, min(1.0, ratio))
        return point[0] ** 2 + point[1] ** 2 <= (allowed + _CONTAINS_TOL) ** 2

    def to_dict(self) -> dict:
        """导出为 ``{"type": "cone", "radius": r, "height": h}``。"""
        return {"type": self.type_name, "radius": self.radius, "height": self.height}


class FrustumGeometry(GeometrySpec):
    """实心圆台（圆锥截去顶端），轴线沿局部 Z 轴，**质心在原点**。

    圆锥是它的退化情形（``top_radius = 0``），但圆锥有独立类型：圆锥的闭式
    惯量是常识公式，而圆台只能靠积分；两者分开写便于互相校验（见
    ``tests/test_primitive_geometry.py`` 里的等价性测试）。

    参数
    ----
    bottom_radius : 底面（-Z 侧）半径，可以为 0（倒立的圆锥）
    top_radius    : 顶面（+Z 侧）半径，可以为 0（正立的圆锥）
    height        : 高，必须为正
    """

    type_name = "frustum"

    __slots__ = (
        "bottom_radius",
        "top_radius",
        "height",
        "_bottom_z",
        "_top_z",
        "_unit_moments",
    )

    def __init__(
        self,
        bottom_radius: float = 0.5,
        top_radius: float = 0.25,
        height: float = 1.0,
    ) -> None:
        r1, r2, h = float(bottom_radius), float(top_radius), float(height)
        if r1 < 0.0 or r2 < 0.0:
            raise ValueError(f"frustum 的半径不能为负，收到：{[r1, r2]}")
        if r1 <= 0.0 and r2 <= 0.0:
            raise ValueError("frustum 的两个半径不能同时为 0")
        if h <= 0.0:
            raise ValueError(f"frustum.height 必须为正数，收到：{h}")
        self.bottom_radius = r1
        self.top_radius = r2
        self.height = h

        # 自然坐标：底面 -h/2、顶面 +h/2，半径线性变化 r(z) = a + b·z
        a = 0.5 * (r1 + r2)
        b = (r2 - r1) / h
        moments = _solid_of_revolution(
            (a * a, 2.0 * a * b, b * b),
            -0.5 * h,
            0.5 * h,
        )
        self._bottom_z = -0.5 * h - moments.centroid_z
        self._top_z = 0.5 * h - moments.centroid_z
        self._unit_moments = (
            moments.ixx / moments.volume,
            moments.ixx / moments.volume,
            moments.izz / moments.volume,
        )

    def volume(self) -> float:
        """圆台体积 = π h (R₁² + R₁R₂ + R₂²) / 3。"""
        r1, r2, h = self.bottom_radius, self.top_radius, self.height
        return math.pi * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0

    def bounding_radius(self) -> float:
        """包围球半径：上、下两个圆周里离质心更远的那个。"""
        bottom = math.sqrt(self.bottom_radius ** 2 + self._bottom_z ** 2)
        top = math.sqrt(self.top_radius ** 2 + self._top_z ** 2)
        return max(bottom, top)

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """给定质量的主惯量（单位质量值乘质量）。"""
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        return tuple(mass * value for value in self._unit_moments)

    def support(self, direction):
        """支撑点在两个端面圆周上取——两者比较一次即可，不需要更细的分割。"""
        radial = math.hypot(direction[0], direction[1])
        best_score = None
        best = (0.0, 0.0, self._top_z)

        for radius, z in (
            (self.bottom_radius, self._bottom_z),
            (self.top_radius, self._top_z),
        ):
            if radial > _SUPPORT_EPS:
                scale = radius / radial
                candidate = (direction[0] * scale, direction[1] * scale, z)
            else:
                candidate = (0.0, 0.0, z)
            score = (
                candidate[0] * direction[0]
                + candidate[1] * direction[1]
                + candidate[2] * direction[2]
            )
            if best_score is None or score > best_score:
                best_score = score
                best = candidate
        return best

    def contains(self, point) -> bool:
        """圆台：轴向在两端面之间，径向不超过该高度的插值半径。"""
        z = point[2]
        if z < self._bottom_z - _CONTAINS_TOL or z > self._top_z + _CONTAINS_TOL:
            return False
        span = self._top_z - self._bottom_z
        ratio = (z - self._bottom_z) / span
        ratio = max(0.0, min(1.0, ratio))
        allowed = self.bottom_radius + (self.top_radius - self.bottom_radius) * ratio
        return point[0] ** 2 + point[1] ** 2 <= (allowed + _CONTAINS_TOL) ** 2

    def to_dict(self) -> dict:
        """导出为 ``{"type": "frustum", "bottom_radius": ..., ...}``。"""
        return {
            "type": self.type_name,
            "bottom_radius": self.bottom_radius,
            "top_radius": self.top_radius,
            "height": self.height,
        }


class EllipsoidGeometry(GeometrySpec):
    """三轴椭球，半轴沿局部坐标轴，**质心在原点**。

    球是它的特例，但球有独立的更快路径（支撑点只需一次归一化），
    所以椭球不替代球。

    参数
    ----
    radii : 三个半轴长度 (a, b, c)，必须全部为正
    """

    type_name = "ellipsoid"

    __slots__ = ("radii",)

    def __init__(self, radii: Sequence[float] = (0.5, 0.4, 0.3)) -> None:
        a, b, c = (float(radii[0]), float(radii[1]), float(radii[2]))
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            raise ValueError(f"ellipsoid.radii 必须全部为正数，收到：{[a, b, c]}")
        self.radii = (a, b, c)

    def volume(self) -> float:
        """椭球体积 = 4/3 π a b c。"""
        a, b, c = self.radii
        return (4.0 / 3.0) * math.pi * a * b * c

    def bounding_radius(self) -> float:
        """包围球半径 = 最长半轴。"""
        return max(self.radii)

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """椭球关于质心的主惯量。

        推导：把单位球经线性变换 ``diag(a, b, c)`` 拉伸成椭球，均匀球的主惯量
        ``2/5 m R²`` 在该变换下给出 ``Ixx = m(b² + c²)/5``，其余两轴轮换。
        """
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        a, b, c = self.radii
        return (
            mass * (b * b + c * c) / 5.0,
            mass * (a * a + c * c) / 5.0,
            mass * (a * a + b * b) / 5.0,
        )

    def support(self, direction):
        """椭球的支撑点：``(a²dx, b²dy, c²dz) / |(a dx, b dy, c dz)|``。

        这是"沿方向最大化 d·x 满足椭球方程"的拉格朗日解，精确。
        """
        a, b, c = self.radii
        nx = a * direction[0]
        ny = b * direction[1]
        nz = c * direction[2]
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm <= _SUPPORT_EPS:
            return (0.0, 0.0, 0.0)
        return (a * nx / norm, b * ny / norm, c * nz / norm)

    def contains(self, point) -> bool:
        """椭球：三个方向的归一化平方和不超过 1。"""
        a, b, c = self.radii
        return (
            (point[0] / a) ** 2 + (point[1] / b) ** 2 + (point[2] / c) ** 2
            <= 1.0 + _CONTAINS_TOL
        )

    def to_dict(self) -> dict:
        """导出为 ``{"type": "ellipsoid", "radii": [...]}``。"""
        return {"type": self.type_name, "radii": list(self.radii)}


class SphericalCapGeometry(GeometrySpec):
    """球冠：球面被一个平面切下的那块，**质心在原点**，开口朝 +Z。

    这是"胶囊不该是基本体"的直接结果：胶囊 = 圆柱 + 两个球冠，而球冠底面与
    圆柱端面恰好重合（不重叠、体积可加）。也就是说，有了圆柱 + 球冠，胶囊
    可以被组合体**精确**表达，而球冠本身是曲面体、拼不出来。

    参数
    ----
    radius : 所属球的半径
    height : 球冠的高（从底面平面到极点），``0 < height ≤ 2·radius``；
             ``height = radius`` 即半球，``= 2·radius`` 退化为整球
    """

    type_name = "spherical_cap"

    __slots__ = (
        "radius",
        "height",
        "_base_z",
        "_sphere_center_z",
        "_rim_radius",
        "_unit_moments",
    )

    def __init__(self, radius: float = 0.5, height: float = 0.25) -> None:
        r, h = float(radius), float(height)
        if r <= 0.0:
            raise ValueError(f"spherical_cap.radius 必须为正数，收到：{r}")
        if h <= 0.0:
            raise ValueError(f"spherical_cap.height 必须为正数，收到：{h}")
        if h > 2.0 * r + _CONTAINS_TOL:
            raise ValueError(
                f"spherical_cap.height 不能超过 2·radius，收到：h={h}, r={r}"
            )
        self.radius = r
        self.height = h

        # 自然坐标：球心在原点，球冠占据 z ∈ [r - h, r]
        moments = _solid_of_revolution((r * r, 0.0, -1.0), r - h, r)
        self._sphere_center_z = -moments.centroid_z
        self._base_z = (r - h) - moments.centroid_z
        self._rim_radius = math.sqrt(max(0.0, 2.0 * r * h - h * h))
        self._unit_moments = (
            moments.ixx / moments.volume,
            moments.ixx / moments.volume,
            moments.izz / moments.volume,
        )

    def volume(self) -> float:
        """球冠体积 = π h² (3r − h) / 3（h = 2r 时退化为整球）。"""
        r, h = self.radius, self.height
        return math.pi * h * h * (3.0 * r - h) / 3.0

    def bounding_radius(self) -> float:
        """包围球半径：极点与底面圆周里离质心更远的那个。"""
        pole = abs(self.radius + self._sphere_center_z)
        rim = math.sqrt(self._rim_radius ** 2 + self._base_z ** 2)
        return max(pole, rim)

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """给定质量的主惯量（单位质量值乘质量）。"""
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        return tuple(mass * value for value in self._unit_moments)

    def support(self, direction):
        """球冠的支撑点：先在整球上取支撑点，若不落在冠内就夹到底面圆周。

        球冠 = 球 ∩ 半空间，两个约束都是凸的，所以沿方向的最远点要么是球的
        支撑点（当它满足 ``z ≥ 底面`` 时），要么在底面圆盘的圆周上。
        """
        length = math.sqrt(_dot3(direction, direction))
        if length <= _SUPPORT_EPS:
            return (0.0, 0.0, 0.0)

        sphere_z = self._sphere_center_z + self.radius * direction[2] / length
        if sphere_z >= self._base_z - _CONTAINS_TOL:
            scale = self.radius / length
            return (
                direction[0] * scale,
                direction[1] * scale,
                sphere_z,
            )

        radial = math.hypot(direction[0], direction[1])
        if radial <= _SUPPORT_EPS:
            return (0.0, 0.0, self._base_z)
        scale = self._rim_radius / radial
        return (direction[0] * scale, direction[1] * scale, self._base_z)

    def contains(self, point) -> bool:
        """球冠：既在球内，又在底面平面之上。"""
        if point[2] < self._base_z - _CONTAINS_TOL:
            return False
        dx = point[0]
        dy = point[1]
        dz = point[2] - self._sphere_center_z
        return dx * dx + dy * dy + dz * dz <= (self.radius + _CONTAINS_TOL) ** 2

    def to_dict(self) -> dict:
        """导出为 ``{"type": "spherical_cap", "radius": r, "height": h}``。"""
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

    def support(self, direction):
        """凸包的支撑点必然是一个顶点，线性扫一遍即可。

        顶点数由 ``max_faces`` 间接限制（默认 64 面），量级在几十，
        不值得为它建加速结构。
        """
        vertices = self.vertices
        best = vertices[0]
        best_score = _dot3(best, direction)
        for vertex in vertices:
            score = _dot3(vertex, direction)
            if score > best_score:
                best_score = score
                best = vertex
        return best

    def contains(self, point) -> bool:
        """凸包：所有面法向的半空间都满足（凸性是这里能这样判定的前提）。"""
        for normal, offset in zip(self.face_normals, self.face_offsets):
            if _dot3(normal, point) > offset + _CONTAINS_TOL:
                return False
        return True

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


#: 组合体部件重叠性的两种声明。场景 JSON 里 **必须** 显式写其中之一。
#:
#: 为什么不让程序自动判断：判断"两个部件是否相交"本身就要算碰撞，而重叠惯量
#: 是**加载期**一次性计算的，自动判断等于把代价付两遍；更要紧的是，"是否允许
#: 重叠"是建模者的意图，不是几何的客观属性——同一个模型可以按两种意图使用。
OVERLAP_DISJOINT = "disjoint"
OVERLAP_ALLOWED = "overlap"
OVERLAP_MODES = (OVERLAP_DISJOINT, OVERLAP_ALLOWED)

#: 重叠模式的默认采样数。10 万点时体积的相对误差约 0.3%，
#: 构造耗时在几十毫秒量级——它只发生一次，与仿真帧率无关。
DEFAULT_OVERLAP_SAMPLES = 100_000


def _part_aabb(part: "CompoundPart") -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """部件在组合体坐标系下的轴对齐包围盒 ``(lo, hi)``。

    盒用"旋转后的半长在各轴上的投影和"算**精确**包围盒——这一步很重要：
    采样是在这个盒子里撒点，盒子越松，命中率越低，误差越大。其它几何退化为
    ``bounding_radius`` 给出的球包围盒（保守但足够紧）。
    """
    geometry = part.geometry
    if isinstance(geometry, BoxGeometry):
        half = (0.5 * geometry.size[0], 0.5 * geometry.size[1], 0.5 * geometry.size[2])
        axes = [
            quat.quat_rotate_vector(part.orientation, unit)
            for unit in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ]
        extent = tuple(
            abs(axes[0][axis]) * half[0]
            + abs(axes[1][axis]) * half[1]
            + abs(axes[2][axis]) * half[2]
            for axis in range(3)
        )
    else:
        radius = geometry.bounding_radius()
        extent = (radius, radius, radius)

    return (
        tuple(part.position[axis] - extent[axis] for axis in range(3)),
        tuple(part.position[axis] + extent[axis] for axis in range(3)),
    )


def _halton(index: int, base: int) -> float:
    """Halton 低差异序列的第 ``index`` 项（radical inverse）。

    用低差异序列而不是伪随机数有两个好处：**没有种子**（同一份输入每次得到
    逐位相同的结果），并且分层比纯随机更均匀，同样的 N 给出更小的误差。
    """
    result = 0.0
    fraction = 1.0 / base
    current = index
    while current > 0:
        result += fraction * (current % base)
        current //= base
        fraction /= base
    return result


def _disjoint_mass_properties(
    parts: Sequence["CompoundPart"],
) -> tuple[float, tuple[float, float, float], list[list[float]]]:
    """部件互不重叠：逐部件张量旋转 + 平行轴定理求和（精确）。

    按**单位密度**处理，于是部件"质量"就等于自身体积：

    - 质心 ``c = Σ V_i p_i / Σ V_i``
    - 张量 ``I = Σ [ V_i·R_i Λ_i R_iᵀ + V_i(|d_i|²E − d_i d_iᵀ) ]``，``d_i = p_i − c``

    返回 ``(总体积, 质心, 单位密度张量)``。
    """
    # 延迟导入：inertia 反向依赖本模块，模块级导入会成环
    from .inertia import compute_inertia

    volumes = [float(part.geometry.volume()) for part in parts]
    total_volume = sum(volumes)
    if total_volume <= 0.0:
        raise ValueError(f"复合几何的总体积必须为正，收到：{total_volume}")

    center = [0.0, 0.0, 0.0]
    for part, volume in zip(parts, volumes):
        for axis in range(3):
            center[axis] += part.position[axis] * volume
    centroid = (
        center[0] / total_volume,
        center[1] / total_volume,
        center[2] / total_volume,
    )

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
            part.position[0] - centroid[0],
            part.position[1] - centroid[1],
            part.position[2] - centroid[2],
        )
        squared = _dot3(offset, offset)
        for row in range(3):
            for column in range(3):
                delta = squared if row == column else 0.0
                tensor[row][column] += volume * (delta - offset[row] * offset[column])

    return total_volume, centroid, tensor


def _overlap_mass_properties(
    parts: Sequence["CompoundPart"],
    samples: int,
) -> tuple[float, tuple[float, float, float], list[list[float]]]:
    """部件可能重叠：在联合 AABB 里撒点，按**并集**估计体积、质心与二阶矩。

    数学上就是三条定义式，采样只是它们的蒙特卡洛估计：

    - ``V = V_box · (命中数 / 采样数)``
    - ``c = E[x]``（对命中点求均值）
    - ``I = V · (tr(M)E − M)``，``M = E[(x−c)(x−c)ᵀ]``

    关键点是"命中"判据用**并集**：点只要落在任一部件的内部就算材料，所以重叠
    区只计一次。这正是"各部件相加"做不到的地方。

    返回 ``(并集体积, 质心, 单位密度张量)``。
    """
    if samples <= 0:
        raise ValueError(f"overlap 模式的采样数必须为正，收到：{samples}")

    lower = [math.inf, math.inf, math.inf]
    upper = [-math.inf, -math.inf, -math.inf]
    for part in parts:
        part_lower, part_upper = _part_aabb(part)
        for axis in range(3):
            lower[axis] = min(lower[axis], part_lower[axis])
            upper[axis] = max(upper[axis], part_upper[axis])

    span = [upper[axis] - lower[axis] for axis in range(3)]
    box_volume = span[0] * span[1] * span[2]
    if box_volume <= 0.0:
        raise ValueError(f"组合体的包围盒体积必须为正，收到：{box_volume}")

    # 预先把部件的逆向姿态算好：内层循环里每个采样点都要用一次，
    # 放在这里可以把"求共轭"的次数从 O(N·部件数) 降到 O(部件数)。
    probes = [
        (
            part.geometry,
            part.position,
            quat.quat_conjugate(part.orientation),
        )
        for part in parts
    ]

    hits = 0
    sum_axis = [0.0, 0.0, 0.0]
    sum_product = [[0.0, 0.0, 0.0] for _ in range(3)]

    for index in range(1, samples + 1):
        point = (
            lower[0] + span[0] * _halton(index, 2),
            lower[1] + span[1] * _halton(index, 3),
            lower[2] + span[2] * _halton(index, 5),
        )

        for geometry, position, inverse in probes:
            local = quat.quat_rotate_vector(
                inverse,
                (
                    point[0] - position[0],
                    point[1] - position[1],
                    point[2] - position[2],
                ),
            )
            if geometry.contains(local):
                hits += 1
                for axis in range(3):
                    sum_axis[axis] += point[axis]
                for row in range(3):
                    for column in range(3):
                        sum_product[row][column] += point[row] * point[column]
                break

    if hits == 0:
        raise ValueError(
            f"重叠惯量采样未命中任何部件（{samples} 个采样点全部落在物体之外）。"
            f"包围盒 {box_volume:.6g} m³ 相对部件体积过大，请检查部件位姿是否合理。"
        )

    volume = box_volume * hits / samples
    centroid = (sum_axis[0] / hits, sum_axis[1] / hits, sum_axis[2] / hits)

    # 二阶中心矩 M = E[xxᵀ] − c cᵀ ，再由 M 换成惯量张量
    moment = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in range(3):
        for column in range(3):
            moment[row][column] = (
                sum_product[row][column] / hits - centroid[row] * centroid[column]
            )

    trace = moment[0][0] + moment[1][1] + moment[2][2]
    tensor = [
        [
            volume * ((trace if row == column else 0.0) - moment[row][column])
            for column in range(3)
        ]
        for row in range(3)
    ]
    return volume, centroid, tensor


class CompoundGeometry(GeometrySpec):
    """由若干基本几何体拼成的刚体 —— "解析组合建模"的入口。

    为什么不是凸包
    -------------
    凸包只能表示**凸**的**多面体**。L 形支架、人形、带凹槽的夹具都不凸：
    取凸包会把凹槽填平，碰撞体积凭空变大，物体悬在半空；离散成网格又丢掉
    解析精度、且没有闭式惯量。复合体保留每个部件的原始解析形状，碰撞时逐
    部件求交、结果取并集。

    两种质量属性模型：``overlap`` 必须由场景显式声明
    ------------------------------------------------
    **"部件互不重叠"是个假设，不是事实。** 哑铃、T 形支架这类由离散件拼起来的
    物体确实不重叠；但凡"表面负载"式建模——薄壳上贴一块配重、球冠拼胶囊、
    圆柱插进底座——部件就会互相插入。此时把各部件质量相加，等于把重叠区算了
    两遍，惯量偏大，物体会显得比实际"更沉"。

    所以 ``CompoundGeometry`` 要求调用方（以及场景 JSON）显式声明是哪种：

    ``overlap="disjoint"``
        **声明部件互不重叠**。走解析路径：逐部件张量旋转 + 平行轴定理求和，
        精确、无采样、构造开销可忽略。声明错了不会报错，只会悄悄偏大——所以
        这个值必须由建模的人负责。

    ``overlap="overlap"``
        **声明部件可能有重叠**。走**并集采样**：先取所有部件的 AABB，在其内用
        确定性低差异序列（Halton）撒点，落在任一部件内部的点才算材料。由命中的
        点估计体积、质心、二阶矩，再换成惯量张量。体积也因此变成**并集体积**
        （而不是各部件之和）。

        为什么可以采样而不心虚
        ----------------------
        1. 这是**加载期一次性的**计算：JSON 读进来算一次，写进刚体，之后每帧
           只是读取，不会重算。所以采样开销与仿真步数无关。
        2. 采样序列是**确定性**的（Halton 低差异序列，不是伪随机数），同一份
           JSON 每次得到逐位相同的结果，不需要种子。
        3. 误差随 ``1/√N`` 收敛，默认 ``samples=100000`` 时体积误差约 0.3%。
           需要更准就调大 ``samples``，但它是"用一次构造时间换精度"，与运行期
           性能无关。

    两种模式共同的后续步骤
    --------------------
    得到完整 3×3 张量（一般含惯量积）后做主轴分解，把主轴旋转
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
        "source_parts",
        "com_offset",
        "overlap",
        "samples",
        "_volume",
        "_radius",
        "_unit_moments",
        "_frame",
    )

    def __init__(
        self,
        parts: Sequence[CompoundPart],
        overlap: str = OVERLAP_DISJOINT,
        samples: int = DEFAULT_OVERLAP_SAMPLES,
    ) -> None:
        """构造组合体。

        参数
        ----
        parts   : 部件列表（几何 + 相对该组合体坐标原点的位姿）
        overlap : ``"disjoint"``（部件互不重叠，解析求和）或
                  ``"overlap"``（可能有重叠，采样并集）
        samples : 仅 ``overlap`` 模式使用，采样点数
        """
        if not parts:
            raise ValueError("复合几何至少需要一个部件")
        for part in parts:
            if not isinstance(part, CompoundPart):
                raise TypeError(
                    f"复合几何的部件必须是 CompoundPart，收到：{type(part).__name__}"
                )

        mode = str(overlap).lower()
        if mode not in OVERLAP_MODES:
            raise ValueError(
                f"compound.overlap 只能是 {OVERLAP_MODES} 之一，收到：{overlap!r}。"
                f"请显式声明部件是否可能互相重叠——这个假设无法从几何自动判断。"
            )
        self.overlap = mode
        self.samples = int(samples) if mode == OVERLAP_ALLOWED else 0

        # --- 1. 体积 / 质心 / 单位密度张量 ---
        if mode == OVERLAP_DISJOINT:
            total_volume, center, tensor = _disjoint_mass_properties(parts)
        else:
            total_volume, center, tensor = _overlap_mass_properties(parts, self.samples)

        # --- 2. 主轴分解 ---
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
        #: 调用方给的原始部件（质心与主轴坐标系之前）。``to_dict`` 导出这一份，
        #: 它才是"模型"的忠实描述；``parts`` 是碰撞用的归一化结果。
        self.source_parts = tuple(parts)

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
        """总体积。

        ``disjoint`` 模式下等于各部件之和（精确）；``overlap`` 模式下是**并集**
        体积（采样估计），此时它小于各部件之和——差值正是被重复计算的重叠区。
        """
        return self._volume

    def bounding_radius(self) -> float:
        """以质心为球心、包住所有部件的包围球半径。"""
        return self._radius

    def contains(self, point) -> bool:
        """组合体：任一部件的包含测试为真即可（并集，重叠无影响）。

        部件位姿在构造时已经搬到"质心 + 主轴"坐标系，所以这里只需把点转到
        部件自身的坐标系，不用再做整体变换。
        """
        for part in self.parts:
            shifted = (
                point[0] - part.position[0],
                point[1] - part.position[1],
                point[2] - part.position[2],
            )
            local = quat.quat_rotate_vector(
                quat.quat_conjugate(part.orientation), shifted
            )
            if part.geometry.contains(local):
                return True
        return False

    def principal_moments(self, mass: float) -> tuple[float, float, float]:
        """给定质量下的主惯量（对角）。"""
        if mass <= 0.0:
            raise ValueError(f"质量必须为正数，收到：{mass}")
        return tuple(mass * value for value in self._unit_moments)

    def inertia_frame(self) -> tuple[float, float, float, float]:
        """几何坐标系 → 惯量主轴系的旋转四元数（复合体一般不是单位旋转）。"""
        return self._frame

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典。

        导出的部件是**调用方给的那一份**（``source_parts``），不是内部归一化后的
        ``parts``。这一点对 ``overlap`` 模式是必须的：采样是在部件包围盒里按
        Halton 序列撒点，盒子随坐标系变化时采样点的相对位置也跟着变，导出归一化
        部件会让往返后的体积差 1e-5 量级。导出原始部件则往返**逐位一致**。

        ``overlap`` 也必须一起导出：它描述的是**建模意图**，不是可以由几何推出的
        属性。同一个部件表在两种模式下算出不同的惯量，丢了它就重建不出同一个物体。
        """
        parts = []
        for part in self.source_parts:
            parts.append(
                {
                    "geometry": part.geometry.to_dict(),
                    "position": list(part.position),
                    "orientation": list(part.orientation),
                }
            )
        data = {"type": "compound", "overlap": self.overlap, "parts": parts}
        if self.overlap == OVERLAP_ALLOWED:
            data["samples"] = self.samples
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CompoundGeometry":
        """从场景 JSON 构造。

        ``overlap`` 是**必填**字段：部件会不会互相插入，只有建模的人知道，
        程序不去猜（猜错的结果是惯量偏大而没有任何报错）。
        """
        overlap = data.get("overlap")
        if overlap is None:
            raise ValueError(
                "compound 缺少必填字段 'overlap'：请显式声明 "
                f"{OVERLAP_DISJOINT!r}（部件互不重叠，解析求和）或 "
                f"{OVERLAP_ALLOWED!r}（可能有重叠，采样并集）。"
            )

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
        return cls(
            parts,
            overlap=overlap,
            samples=data.get("samples", DEFAULT_OVERLAP_SAMPLES),
        )


# ---------------------------------------------------------------------------
# 类型注册表与解析辅助
# ---------------------------------------------------------------------------

#: 类型名 -> 构造函数，供 ``GeometrySpec.from_dict`` 分发使用。
#: 新增图元只需在此登记，不需要改解析逻辑。
_GEOMETRY_REGISTRY = {
    "box": lambda d: BoxGeometry(d.get("size", (1.0, 1.0, 1.0))),
    "sphere": lambda d: SphereGeometry(d.get("radius", 0.5)),
    "cylinder": lambda d: CylinderGeometry(
        d.get("radius", 0.5), d.get("height", 1.0)
    ),
    "capsule": lambda d: CapsuleGeometry(d.get("radius", 0.25), d.get("height", 1.0)),
    # 旋转体：圆锥、圆台、球冠。三者共用闭式积分（revolution.py）
    "cone": lambda d: ConeGeometry(d.get("radius", 0.5), d.get("height", 1.0)),
    "frustum": lambda d: FrustumGeometry(
        d.get("bottom_radius", 0.5),
        d.get("top_radius", 0.25),
        d.get("height", 1.0),
    ),
    "spherical_cap": lambda d: SphericalCapGeometry(
        d.get("radius", 0.5), d.get("height", 0.25)
    ),
    # 椭球：三个半轴
    "ellipsoid": lambda d: EllipsoidGeometry(d.get("radii", (0.5, 0.4, 0.3))),
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
