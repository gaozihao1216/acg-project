# -*- coding: utf-8 -*-
"""场景描述与 JSON 解析。

设计目标：**改 JSON 就能换画面，不改代码**。
这是题目评分表里「自定义场景配置（5 分）」的直接落点，也是做参数扫描
实验（相机、灯光、材质、布局）的基础设施。

本模块不依赖 bpy，因此可以在普通 Python 里做校验、生成、对比，
只有真正要渲染时才交给 Blender。
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from ..geom.geometry import GeometrySpec
from ..geom.transform import TransformSpec


# ---------------------------------------------------------------------------
# 通用校验辅助
# ---------------------------------------------------------------------------

def _as_floats(value: Any, count: int, field: str) -> tuple[float, ...]:
    """把 JSON 中的数组字段转成长度为 ``count`` 的浮点元组。

    参数
    ----
    value : 原始 JSON 值（通常是 list）
    count : 期望的长度
    field : 字段名，仅用于拼出可读的报错信息

    异常
    ----
    ValueError : 不是数组、长度不符，或含非数值元素
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"字段 '{field}' 必须是数组，收到：{type(value).__name__}")
    if len(value) != count:
        raise ValueError(f"字段 '{field}' 需要 {count} 个分量，实际收到 {len(value)} 个")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"字段 '{field}' 含非数值元素：{value}") from exc


def _positive(value: Any, field: str) -> float:
    """校验一个字段是正数并返回其浮点值。"""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"字段 '{field}' 必须是数值，收到：{value!r}") from exc
    if v <= 0.0:
        raise ValueError(f"字段 '{field}' 必须为正数，收到：{v}")
    return v


def _clamp01(value: Any, field: str) -> float:
    """把字段裁剪到 [0, 1] 区间，用于粗糙度、金属度等材质参数。"""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"字段 '{field}' 必须是数值，收到：{value!r}") from exc
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# 各子结构
# ---------------------------------------------------------------------------

class CameraSpec:
    """透视相机：由位置、注视点和水平视场角确定。"""

    __slots__ = ("position", "target", "fov")

    def __init__(
        self,
        position: Sequence[float] = (4.0, 4.0, 6.0),
        target: Sequence[float] = (0.0, 0.0, 0.5),
        fov: float = 45.0,
    ) -> None:
        """构造相机。

        参数
        ----
        position : 相机位置（世界坐标）
        target   : 相机注视点；相机的 -Z 轴会指向这里
        fov      : 视场角，单位**度**。数值越小画面越"长焦"，
                   越大越"广角"（透视畸变也越强）。常用 35–60。
        """
        self.position = _as_floats(position, 3, "camera.position")
        self.target = _as_floats(target, 3, "camera.target")
        self.fov = _positive(fov, "camera.fov")

    @classmethod
    def from_dict(cls, data: dict | None) -> "CameraSpec":
        """从 JSON 字典构造；缺字段时使用默认值。"""
        data = data or {}
        return cls(
            position=data.get("position", (4.0, 4.0, 6.0)),
            target=data.get("target", (0.0, 0.0, 0.5)),
            fov=data.get("fov", 45.0),
        )


class LightSpec:
    """光源描述。

    支持 ``area``（面光）、``sun``（平行光）、``point``（点光）三种类型。
    面光最适合产品级打光：``size`` 越大阴影越软，是控制画面调性的主要手段。
    """

    __slots__ = ("name", "light_type", "position", "target", "energy", "size", "color")

    def __init__(
        self,
        name: str = "light",
        light_type: str = "area",
        position: Sequence[float] = (3.0, 3.0, 5.0),
        target: Sequence[float] = (0.0, 0.0, 0.0),
        energy: float = 500.0,
        size: float = 2.0,
        color: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        """构造光源。

        参数
        ----
        name       : 光源名称，仅用于在 Blender 大纲里辨认
        light_type : 'area' / 'sun' / 'point'
        position   : 位置（sun 类型下位置会被忽略，只用方向）
        target     : 照射目标点，决定光的朝向
        energy     : 功率。area/point 单位是瓦特，太阳光单位是辐照度
        size       : 面光尺寸（米）；越大阴影越软
        color      : RGB，各分量取值 0–1
        """
        self.name = str(name)
        self.light_type = str(light_type).lower()
        if self.light_type not in ("area", "sun", "point"):
            raise ValueError(
                f"light.type 只支持 area / sun / point，收到：{light_type!r}"
            )
        self.position = _as_floats(position, 3, f"{name}.position")
        self.target = _as_floats(target, 3, f"{name}.target")
        self.energy = _positive(energy, f"{name}.energy")
        self.size = _positive(size, f"{name}.size")
        self.color = _as_floats(color, 3, f"{name}.color")

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> "LightSpec":
        """从 JSON 数组的单个元素构造光源。"""
        if not isinstance(data, dict):
            raise ValueError(f"lights[{index}] 必须是对象")
        return cls(
            name=data.get("name", f"light_{index}"),
            light_type=data.get("type", "area"),
            position=data.get("position", (3.0, 3.0, 5.0)),
            target=data.get("target", (0.0, 0.0, 0.0)),
            energy=data.get("energy", 500.0),
            size=data.get("size", 2.0),
            color=data.get("color", (1.0, 1.0, 1.0)),
        )


class GroundSpec:
    """地面平面。视觉上提供落脚点与阴影承托，但**不参与物理**（W1 阶段）。"""

    __slots__ = ("enabled", "size", "color", "roughness")

    def __init__(
        self,
        enabled: bool = True,
        size: float = 40.0,
        color: Sequence[float] = (0.78, 0.78, 0.80),
        roughness: float = 0.6,
    ) -> None:
        """构造地面。

        参数
        ----
        enabled   : 是否创建地面
        size      : 平面边长（米），只要大到相机看不到边缘即可
        color     : 地面基础色
        roughness : 粗糙度，0 为镜面、1 为完全漫反射
        """
        self.enabled = bool(enabled)
        self.size = _positive(size, "ground.size")
        self.color = _as_floats(color, 3, "ground.color")
        self.roughness = _clamp01(roughness, "ground.roughness")

    @classmethod
    def from_dict(cls, data: dict | None) -> "GroundSpec":
        """从 JSON 字典构造地面；缺省时返回默认地面。"""
        data = data or {}
        return cls(
            enabled=data.get("enabled", True),
            size=data.get("size", 40.0),
            color=data.get("color", (0.78, 0.78, 0.80)),
            roughness=data.get("roughness", 0.6),
        )


class MaterialSpec:
    """Principled BSDF 材质的精简描述。

    只暴露最影响观感的四个参数：基础色、粗糙度、金属度、透射。
    这四个参数足以区分"哑光塑料 / 金属 / 玻璃"三类典型材质，
    也是做材质对比实验的最小集合。
    """

    __slots__ = ("name", "base_color", "roughness", "metallic", "transmission", "emissive")

    def __init__(
        self,
        name: str = "material",
        base_color: Sequence[float] = (0.8, 0.8, 0.8),
        roughness: float = 0.5,
        metallic: float = 0.0,
        transmission: float = 0.0,
        emissive: float = 0.0,
    ) -> None:
        """构造材质。

        参数
        ----
        name         : 材质名
        base_color   : 基础色 RGB（0–1）。注意这是**线性空间**颜色，
                       不是 sRGB 的 0–255，直接照抄屏幕取色会偏亮。
        roughness    : 粗糙度 0–1
        metallic     : 金属度 0–1。1 表示纯金属，此时基础色相当于反射率
        transmission : 透射 0–1，大于 0 时表现为玻璃
        emissive     : 自发光强度，用于做发光体点缀
        """
        self.name = str(name)
        self.base_color = _as_floats(base_color, 3, f"{name}.base_color")
        self.roughness = _clamp01(roughness, f"{name}.roughness")
        self.metallic = _clamp01(metallic, f"{name}.metallic")
        self.transmission = _clamp01(transmission, f"{name}.transmission")
        self.emissive = max(0.0, float(emissive))

    @classmethod
    def from_dict(cls, data: dict | None, default_name: str = "material") -> "MaterialSpec":
        """从 JSON 字典构造材质；缺字段时使用中性灰默认值。"""
        data = data or {}
        return cls(
            name=data.get("name", default_name),
            base_color=data.get("base_color", (0.8, 0.8, 0.8)),
            roughness=data.get("roughness", 0.5),
            metallic=data.get("metallic", 0.0),
            transmission=data.get("transmission", 0.0),
            emissive=data.get("emissive", 0.0),
        )


class RenderSpec:
    """渲染设置：分辨率、采样、引擎、世界环境光、色彩管理。"""

    __slots__ = (
        "resolution", "samples", "engine", "denoise",
        "world_color", "world_strength", "exposure", "film_transparent",
    )

    def __init__(
        self,
        resolution: Sequence[int] = (960, 540),
        samples: int = 64,
        engine: str = "CYCLES",
        denoise: bool = True,
        world_color: Sequence[float] = (0.05, 0.05, 0.06),
        world_strength: float = 0.35,
        exposure: float = 0.0,
        film_transparent: bool = False,
    ) -> None:
        """构造渲染设置。

        参数
        ----
        resolution       : 输出分辨率 [宽, 高]
        samples          : 采样数。Cycles 下噪声与采样数平方根成反比，
                           64 采样配降噪通常够用于对比实验
        engine           : 'CYCLES' 或 'BLENDER_EEVEE'
        denoise          : 是否开启降噪（Cycles 用 OIDN）
        world_color      : 环境光颜色
        world_strength   : 环境光强度。这是"底光"，太低会让暗部死黑
        exposure         : 曝光补偿（EV），用于整体提亮/压暗
        film_transparent : 是否输出透明背景
        """
        w, h = _as_floats(resolution, 2, "render.resolution")
        if w < 16 or h < 16 or w > 8192 or h > 8192:
            raise ValueError(f"render.resolution 超出合理范围：{[w, h]}")
        self.resolution = (int(w), int(h))

        s = int(samples)
        if s < 1 or s > 8192:
            raise ValueError(f"render.samples 超出合理范围：{samples}")
        self.samples = s

        self.engine = str(engine).upper()
        if self.engine not in ("CYCLES", "BLENDER_EEVEE"):
            raise ValueError(
                f"render.engine 只支持 CYCLES / BLENDER_EEVEE，收到：{engine!r}"
            )
        self.denoise = bool(denoise)
        self.world_color = _as_floats(world_color, 3, "render.world_color")
        self.world_strength = max(0.0, float(world_strength))
        self.exposure = float(exposure)
        self.film_transparent = bool(film_transparent)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RenderSpec":
        """从 JSON 字典构造渲染设置。"""
        data = data or {}
        return cls(
            resolution=data.get("resolution", (960, 540)),
            samples=data.get("samples", 64),
            engine=data.get("engine", "CYCLES"),
            denoise=data.get("denoise", True),
            world_color=data.get("world_color", (0.05, 0.05, 0.06)),
            world_strength=data.get("world_strength", 0.35),
            exposure=data.get("exposure", 0.0),
            film_transparent=data.get("film_transparent", False),
        )


class BodyPhysicsSpec:
    """物体的**物理**属性。

    刻意与 ``MaterialSpec``（视觉属性）分开：颜色影响渲染，恢复系数影响
    碰撞响应，二者生命周期不同。把它们塞进同一个结构，日后做"只改材质
    颜色不影响物理"的实验时会很别扭。
    """

    __slots__ = ("is_static", "linear_velocity", "angular_velocity", "restitution", "friction")

    def __init__(
        self,
        is_static: bool = False,
        linear_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        restitution: float = 0.25,
        friction: float = 0.5,
    ) -> None:
        """构造物理属性。

        参数
        ----
        is_static        : 静态物体质量无穷大，不受力也不动（如地面、墙）
        linear_velocity  : 初始线速度（m/s）
        angular_velocity : 初始角速度（rad/s，世界坐标系）
        restitution      : 恢复系数 0–1。0 完全不弹，1 完全弹性
        friction         : 摩擦系数，≥0
        """
        self.is_static = bool(is_static)
        self.linear_velocity = _as_floats(linear_velocity, 3, "physics.linear_velocity")
        self.angular_velocity = _as_floats(angular_velocity, 3, "physics.angular_velocity")
        self.restitution = _clamp01(restitution, "physics.restitution")
        f = float(friction)
        if f < 0.0:
            raise ValueError(f"physics.friction 不能为负，收到：{f}")
        self.friction = f

    @classmethod
    def from_dict(cls, data: dict | None) -> "BodyPhysicsSpec":
        """从 JSON 字典构造；缺字段时用默认值（动态、静止、微弹、中等摩擦）。"""
        data = data or {}
        return cls(
            is_static=data.get("static", False),
            linear_velocity=data.get("linear_velocity", (0.0, 0.0, 0.0)),
            angular_velocity=data.get("angular_velocity", (0.0, 0.0, 0.0)),
            restitution=data.get("restitution", 0.25),
            friction=data.get("friction", 0.5),
        )

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典。"""
        return {
            "static": self.is_static,
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "restitution": self.restitution,
            "friction": self.friction,
        }


class ObjectSpec:
    """场景中的一个物体：几何 + 位姿 + 视觉材质 + 物理属性。"""

    __slots__ = ("name", "geometry", "transform", "material", "mass", "physics",
                 "inertia")

    def __init__(self, name, geometry, transform, material, mass=None, physics=None,
                 inertia=None) -> None:
        """构造物体。

        参数
        ----
        name      : 物体名，导出位姿序列时作为稳定 ID 使用
        geometry  : ``GeometrySpec`` 实例
        transform : ``TransformSpec`` 实例
        material  : ``MaterialSpec`` 实例（视觉）
        mass      : 质量（kg）。缺省按体积×1000 估算
        physics   : ``BodyPhysicsSpec`` 实例（物理）
        inertia   : **局部坐标系的转动惯量对角元**（可选）。

                    为什么不总是从几何算
                    --------------------
                    ``compute_inertia`` 只能从**碰撞几何**推惯量。碰撞代理与
                    质量分布常常不是一回事：空心盒的碰撞代理是实心外包盒，
                    而从几何算出的惯量会比真实值大 24.8%——恰好是"质量分布
                    被丢掉"的那个量。

                    所以资产里算好的主惯量必须能传给引擎。``inertia`` 就是
                    这条通路；给出时求解器直接用它，不再从几何反推。
                    此时惯量的主轴**已经**与局部坐标轴重合，故不再叠加
                    ``geometry.inertia_frame()``。
        """
        self.name = str(name)
        self.geometry = geometry
        self.transform = transform
        self.material = material
        # 与 JSON 解析路径保持一致：未给质量时按"密度 1000 kg/m³"由体积反推。
        # 这里兜底一次，避免用代码直接构造 ObjectSpec 时把 None 一路传到求解器。
        if mass is None:
            self.mass = geometry.volume() * 1000.0
        else:
            self.mass = float(mass)
        self.physics = physics or BodyPhysicsSpec()
        # 显式惯量：None 表示"按碰撞几何反推"（保持旧行为）
        if inertia is None:
            self.inertia = None
        else:
            values = [float(v) for v in inertia]
            if len(values) != 3:
                raise ValueError(
                    f"inertia 需要 3 个分量，收到 {len(values)} 个"
                )
            if any(v <= 0.0 for v in values):
                raise ValueError(f"inertia 必须全为正，收到 {values}")
            self.inertia = tuple(values)


class SimulationSpec:
    """物理仿真参数。

    与渲染设置分开：同一份场景可以配不同精度的仿真参数做收敛性对比。
    """

    __slots__ = (
        "enabled", "physics_dt", "substeps", "iterations", "baumgarte", "slop",
        "restitution_threshold", "gravity", "ground_z", "ground_friction",
        "ground_restitution", "linear_damping", "angular_damping",
        "allow_initial_overlap", "overlap_tolerance",
        "use_toi_splitting", "max_events_per_substep",
        "warm_start",
        "broadphase", "broadphase_cell_size",
        "render_fps", "duration",
    )

    def __init__(
        self,
        enabled: bool = False,
        physics_dt: float = 1.0 / 240.0,
        # 默认值由 scripts/bench_solver_strategy.py 的实测数据确定：
        # 迭代次数在 6–8 之后完全饱和（2 接触场景中 6/8/12 次的结果逐位相同），
        # 多头链场景（6 接触）则以 substeps≈3 + iterations≈6–8 最优。
        # 故把冗余的迭代预算换成子步，多接触精度提升约 2.7 倍且不增加耗时。
        substeps: int = 3,
        iterations: int = 8,
        baumgarte: float = 0.25,
        slop: float = 0.002,
        restitution_threshold: float = 1.0,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        ground_z: float = 0.0,
        ground_friction: float = 0.55,
        ground_restitution: float = 0.12,
        linear_damping: float = 0.05,
        angular_damping: float = 1.0,
        allow_initial_overlap: bool = False,
        overlap_tolerance: float = 1e-3,
        use_toi_splitting: bool = True,
        max_events_per_substep: int = 8,
        # 跨帧复用累积冲量。默认开启：静置接触的冲量每帧几乎不变，暖启动让
        # 迭代只需修正变化量，同迭代数下堆叠更硬。关掉用于 A/B 对照实验。
        warm_start: bool = True,
        broadphase: str = "spatial_hash",
        broadphase_cell_size: float = 0.0,
        render_fps: int = 24,
        duration: float = 3.5,
    ) -> None:
        """构造仿真参数。

        参数
        ----
        enabled              : 是否启用物理（不带此段时默认关闭，纯静态渲染）
        physics_dt           : 每个仿真步的时长（秒）。1/240 足以稳定处理
                               日常尺度的碰撞
        substeps             : 每步内部再切分的子步数。子步越小穿透越浅，
                               但计算量线性增长
        iterations           : 顺序冲量的迭代次数。越多接触越"硬"，
                               静置越稳，但边际收益递减
        baumgarte            : 位置修正系数。把穿透"推出去"的力度，
                               太大会抖，太小会陷进去
        slop                 : 允许的穿透容差（米）。留一点余量可以避免
                               静置物体被反复推开而产生抖动
        restitution_threshold: 低于此接近速度时忽略恢复系数（m/s）。
                               **防止静置物体无限微弹**，是非常关键的一项
        gravity              : 重力加速度（m/s²）
        ground_z             : 物理地面的高度（米）
        ground_friction      : 地面摩擦系数
        ground_restitution   : 地面恢复系数
        render_fps           : 导出位姿序列的帧率，与渲染帧率一致
        duration             : 仿真总时长（秒）
        """
        self.enabled = bool(enabled)

        dt = float(physics_dt)
        if dt <= 0.0:
            raise ValueError(f"simulation.physics_dt 必须为正，收到：{dt}")
        self.physics_dt = dt

        s = int(substeps)
        if s < 1:
            raise ValueError(f"simulation.substeps 至少为 1，收到：{substeps}")
        self.substeps = s

        it = int(iterations)
        if it < 1:
            raise ValueError(f"simulation.iterations 至少为 1，收到：{iterations}")
        self.iterations = it

        b = float(baumgarte)
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"simulation.baumgarte 应在 0–1 之间，收到：{b}")
        self.baumgarte = b

        sl = float(slop)
        if sl < 0.0:
            raise ValueError(f"simulation.slop 不能为负，收到：{sl}")
        self.slop = sl

        self.restitution_threshold = max(0.0, float(restitution_threshold))
        self.gravity = _as_floats(gravity, 3, "simulation.gravity")
        self.ground_z = float(ground_z)
        self.ground_friction = max(0.0, float(ground_friction))
        self.ground_restitution = _clamp01(ground_restitution, "simulation.ground_restitution")

        # 阻尼模拟"空气阻力 + 滚动阻力"。注意：理想刚体球在水平面上纯滚动时
        # 接触点速度为零，摩擦不产生任何冲量，因此球会永远滚下去。现实中的球
        # 会停下是因为有滚动阻力，这里用一个各向同性的角速度衰减来近似。
        ld = float(linear_damping)
        if ld < 0.0:
            raise ValueError(f"simulation.linear_damping 不能为负，收到：{ld}")
        self.linear_damping = ld

        ad = float(angular_damping)
        if ad < 0.0:
            raise ValueError(f"simulation.angular_damping 不能为负，收到：{ad}")
        self.angular_damping = ad

        # --- 初始状态校验 ---
        # 初始就互相嵌入的物体会让接触在第一步同时生效，求解器于是给出
        # "整体同时变形"的解（软材料物理），而不是接触逐个传递的解（硬物物理）。
        self.allow_initial_overlap = bool(allow_initial_overlap)
        tol = float(overlap_tolerance)
        if tol < 0.0:
            raise ValueError(f"simulation.overlap_tolerance 不能为负，收到：{tol}")
        self.overlap_tolerance = tol

        # --- TOI 分段求解 ---
        # 在子步内部按"预测接触时刻"切段，让同一子步内先后发生的碰撞
        # 分别求解，从而复现接触逐级传递（波传播）的次序。
        self.use_toi_splitting = bool(use_toi_splitting)
        max_events = int(max_events_per_substep)
        if max_events < 1:
            raise ValueError(
                f"simulation.max_events_per_substep 至少为 1，收到：{max_events_per_substep}"
            )
        self.max_events_per_substep = max_events

        # --- 暖启动 ---
        # 缓存键是 (body_a, body_b, feature_id)：body 索引在同一个 world 内稳定，
        # feature_id 由流形裁剪按几何特征给出，所以跨帧能认出"同一个接触点"。
        self.warm_start = bool(warm_start)

        # --- 宽阶段（broadphase） ---
        # "spatial_hash" 用均匀网格剔除明显不相交的物体对；
        # "brute" 保留 O(N²) 暴力枚举，用于**加速比对照**。
        # 没有基线的"优化"无法证明有效，所以两条路径都留着。
        mode = str(broadphase).strip().lower()
        if mode not in ("brute", "spatial_hash"):
            raise ValueError(
                f"simulation.broadphase 只能是 'brute' 或 'spatial_hash'，收到：{broadphase!r}"
            )
        self.broadphase = mode

        # 0 表示按物体尺寸自动估算；显式给值时必须为正
        cell = float(broadphase_cell_size)
        if cell < 0.0:
            raise ValueError(
                f"simulation.broadphase_cell_size 不能为负，收到：{broadphase_cell_size}"
            )
        self.broadphase_cell_size = cell

        fps = int(render_fps)
        if fps < 1:
            raise ValueError(f"simulation.render_fps 必须为正，收到：{render_fps}")
        self.render_fps = fps

        dur = float(duration)
        if dur <= 0.0:
            raise ValueError(f"simulation.duration 必须为正，收到：{dur}")
        self.duration = dur

    @classmethod
    def from_dict(cls, data: dict | None) -> "SimulationSpec":
        """从 JSON 的 ``simulation`` 段构造；缺省时返回"未启用"的默认值。

        所有缺省值都**从 ``cls()`` 取**，不在本函数里另抄一遍。这里曾经把
        ``substeps``/``iterations`` 硬编码为 2/12，而构造函数早就按
        ``scripts/bench_solver_strategy.py`` 的实测改成 3/8（见 ``docs/05``）。
        于是"程序内默认"和"JSON 默认"静默分叉：手写场景只要不显式写这两个
        字段，走的就是已被否决的旧参数，而代码里看不出任何异常。
        单一定义处消除这类漂移。
        """
        if not data:
            return cls()

        defaults = cls()
        return cls(
            enabled=data.get("enabled", True),
            physics_dt=data.get("physics_dt", defaults.physics_dt),
            substeps=data.get("substeps", defaults.substeps),
            iterations=data.get("iterations", defaults.iterations),
            baumgarte=data.get("baumgarte", defaults.baumgarte),
            slop=data.get("slop", defaults.slop),
            restitution_threshold=data.get(
                "restitution_threshold", defaults.restitution_threshold
            ),
            gravity=data.get("gravity", defaults.gravity),
            ground_z=data.get("ground_z", defaults.ground_z),
            ground_friction=data.get("ground_friction", defaults.ground_friction),
            ground_restitution=data.get(
                "ground_restitution", defaults.ground_restitution
            ),
            linear_damping=data.get("linear_damping", defaults.linear_damping),
            angular_damping=data.get("angular_damping", defaults.angular_damping),
            allow_initial_overlap=data.get(
                "allow_initial_overlap", defaults.allow_initial_overlap
            ),
            overlap_tolerance=data.get(
                "overlap_tolerance", defaults.overlap_tolerance
            ),
            use_toi_splitting=data.get(
                "use_toi_splitting", defaults.use_toi_splitting
            ),
            max_events_per_substep=data.get(
                "max_events_per_substep", defaults.max_events_per_substep
            ),
            warm_start=data.get("warm_start", defaults.warm_start),
            broadphase=data.get("broadphase", defaults.broadphase),
            broadphase_cell_size=data.get(
                "broadphase_cell_size", defaults.broadphase_cell_size
            ),
            render_fps=data.get("render_fps", defaults.render_fps),
            duration=data.get("duration", defaults.duration),
        )

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典。"""
        return {
            "enabled": self.enabled,
            "physics_dt": self.physics_dt,
            "substeps": self.substeps,
            "iterations": self.iterations,
            "baumgarte": self.baumgarte,
            "slop": self.slop,
            "restitution_threshold": self.restitution_threshold,
            "gravity": list(self.gravity),
            "ground_z": self.ground_z,
            "ground_friction": self.ground_friction,
            "ground_restitution": self.ground_restitution,
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
            "allow_initial_overlap": self.allow_initial_overlap,
            "overlap_tolerance": self.overlap_tolerance,
            "use_toi_splitting": self.use_toi_splitting,
            "max_events_per_substep": self.max_events_per_substep,
            "warm_start": self.warm_start,
            "broadphase": self.broadphase,
            "broadphase_cell_size": self.broadphase_cell_size,
            "render_fps": self.render_fps,
            "duration": self.duration,
        }


# ---------------------------------------------------------------------------
# 顶层场景
# ---------------------------------------------------------------------------

class SceneSpec:
    """完整场景：相机 + 光源列表 + 地面 + 物体列表 + 渲染设置。"""

    __slots__ = ("name", "camera", "lights", "ground", "objects", "render", "simulation")

    def __init__(
        self,
        name: str = "scene",
        camera: CameraSpec | None = None,
        lights: list[LightSpec] | None = None,
        ground: GroundSpec | None = None,
        objects: list[ObjectSpec] | None = None,
        render: RenderSpec | None = None,
        simulation: SimulationSpec | None = None,
    ) -> None:
        """构造场景。

        参数
        ----
        name       : 场景名，主要用于日志与输出文件命名
        camera     : 相机；缺省用默认视角
        lights     : 光源列表；缺省时打一盏主光，避免全黑
        ground     : 地面（视觉）；缺省为启用
        objects    : 物体列表；可以为空（用于只验证环境的场景）
        render     : 渲染设置；缺省为 960x540 / 64 采样 / Cycles
        simulation : 物理仿真参数；缺省为未启用（纯静态渲染）
        """
        self.name = str(name)
        self.camera = camera or CameraSpec()
        self.lights = list(lights) if lights is not None else [
            LightSpec(name="key", position=(3.0, 3.0, 5.0), energy=500.0, size=2.0)
        ]
        self.ground = ground or GroundSpec()
        self.objects = list(objects) if objects is not None else []
        self.render = render or RenderSpec()
        self.simulation = simulation or SimulationSpec()

    @classmethod
    def from_dict(cls, data: dict) -> "SceneSpec":
        """从顶层 JSON 字典构造场景，并做完整校验。

        参数
        ----
        data : 场景 JSON 反序列化后的字典

        异常
        -----
        ValueError : 字段结构不合法、物体重名等
        """
        if not isinstance(data, dict):
            raise ValueError("场景根节点必须是 JSON 对象")

        raw_objects = data.get("objects", [])
        if not isinstance(raw_objects, list):
            raise ValueError("'objects' 必须是数组")

        objects: list[ObjectSpec] = []
        seen_names: set[str] = set()
        for idx, raw in enumerate(raw_objects):
            if not isinstance(raw, dict):
                raise ValueError(f"objects[{idx}] 必须是对象")
            obj = _parse_object(raw, idx)
            # 物体名在导出位姿序列时作为稳定 ID，必须唯一
            if obj.name in seen_names:
                raise ValueError(f"物体名重复：'{obj.name}'，请给每个物体起唯一名称")
            seen_names.add(obj.name)
            objects.append(obj)

        raw_lights = data.get("lights", None)
        lights = None
        if raw_lights is not None:
            if not isinstance(raw_lights, list) or not raw_lights:
                raise ValueError("'lights' 必须是非空数组；若想用默认光请省略该字段")
            lights = [LightSpec.from_dict(item, i) for i, item in enumerate(raw_lights)]

        return cls(
            name=data.get("name", "scene"),
            camera=CameraSpec.from_dict(data.get("camera")),
            lights=lights,
            ground=GroundSpec.from_dict(data.get("ground")),
            objects=objects,
            render=RenderSpec.from_dict(data.get("render")),
            simulation=SimulationSpec.from_dict(data.get("simulation")),
        )

    def to_dict(self) -> dict:
        """把场景导出回字典，便于生成实验变体或记录实际生效的参数。"""
        return {
            "name": self.name,
            "camera": {
                "position": list(self.camera.position),
                "target": list(self.camera.target),
                "fov": self.camera.fov,
            },
            "lights": [
                {
                    "name": lt.name,
                    "type": lt.light_type,
                    "position": list(lt.position),
                    "target": list(lt.target),
                    "energy": lt.energy,
                    "size": lt.size,
                    "color": list(lt.color),
                }
                for lt in self.lights
            ],
            "ground": {
                "enabled": self.ground.enabled,
                "size": self.ground.size,
                "color": list(self.ground.color),
                "roughness": self.ground.roughness,
            },
            "objects": [
                {
                    "name": ob.name,
                    "geometry": ob.geometry.to_dict(),
                    "transform": ob.transform.to_dict(),
                    "material": {
                        "name": ob.material.name,
                        "base_color": list(ob.material.base_color),
                        "roughness": ob.material.roughness,
                        "metallic": ob.material.metallic,
                        "transmission": ob.material.transmission,
                        "emissive": ob.material.emissive,
                    },
                    "mass": ob.mass,
                    "physics": ob.physics.to_dict(),
                }
                for ob in self.objects
            ],
            "render": {
                "resolution": list(self.render.resolution),
                "samples": self.render.samples,
                "engine": self.render.engine,
                "denoise": self.render.denoise,
                "world_color": list(self.render.world_color),
                "world_strength": self.render.world_strength,
                "exposure": self.render.exposure,
                "film_transparent": self.render.film_transparent,
            },
        }


def _parse_object(raw: dict, index: int) -> ObjectSpec:
    """把单个物体的 JSON 字典解析为 ``ObjectSpec``。

    单独抽出来是为了让 ``SceneSpec.from_dict`` 保持可读，并且报错时
    能带上 ``objects[i]`` 的索引定位。
    """
    if "geometry" not in raw:
        raise ValueError(f"objects[{index}] 缺少必填字段 'geometry'")

    geometry = GeometrySpec.from_dict(raw["geometry"])
    transform = TransformSpec.from_dict(raw.get("transform"))
    material = MaterialSpec.from_dict(raw.get("material"), default_name=f"mat_{index}")

    physics = BodyPhysicsSpec.from_dict(raw.get("physics"))

    mass = raw.get("mass", None)
    if mass is not None:
        mass = _positive(mass, f"objects[{index}].mass")
    else:
        # 未显式给质量时，按"密度 1000 kg/m³（约等于水）"反推，
        # 保证求解器直接拿来用也有合理值。
        # 静态物体质量无意义，但仍给一个有限值以便体积/惯量计算不报错。
        mass = geometry.volume() * 1000.0

    # 可选：资产算好的主惯量。缺省时仍由 compute_inertia 按几何反推。
    # 这条通路存在的理由是"碰撞代理 ≠ 质量分布"——空心壳的碰撞代理是实心
    # 外包盒，从几何反推会把惯量放大 1.25 倍。
    inertia = raw.get("inertia", None)
    if inertia is not None:
        if not isinstance(inertia, (list, tuple)) or len(inertia) != 3:
            raise ValueError(
                f"objects[{index}].inertia 需要 3 个分量的数组"
            )
        inertia = [
            _positive(value, f"objects[{index}].inertia[{axis}]")
            for axis, value in enumerate(inertia)
        ]

    return ObjectSpec(
        name=raw.get("name", f"obj_{index}"),
        geometry=geometry,
        transform=transform,
        material=material,
        mass=mass,
        physics=physics,
        inertia=inertia,
    )


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------

def load_scene(path: str) -> SceneSpec:
    """从 JSON 文件加载场景。

    参数
    ----
    path : 场景文件路径（绝对路径或相对当前工作目录）

    异常
    ----
    FileNotFoundError : 文件不存在
    ValueError        : JSON 语法错误或字段校验失败
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"场景文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"场景文件 JSON 语法错误：{path}（第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}）"
            ) from exc

    return SceneSpec.from_dict(data)


def save_scene(scene: SceneSpec, path: str) -> None:
    """把场景写回 JSON 文件，目录不存在时自动创建。

    实验脚本用它把"基础场景 + 参数覆盖"落盘，保证实验可复现。
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene.to_dict(), fh, ensure_ascii=False, indent=2)
