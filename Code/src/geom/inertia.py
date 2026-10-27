# -*- coding: utf-8 -*-
"""刚体转动惯量计算。

本模块为各参数化几何体给出**关于质心、沿局部主轴**的转动惯量对角元
``(Ixx, Iyy, Izz)``。因为 box / sphere / cylinder / capsule 都具有
足够高的对称性，三个主轴恰好与几何体的局部坐标轴对齐，所以惯量张量是
对角矩阵，不需要保存完整的 3x3 矩阵。

为什么惯量重要
--------------
刚体角加速度由 ``ω̇ = I⁻¹ (τ - ω × (I ω))`` 决定。若惯量给错（例如把
长方体当成质点、或用错轴），表现就是物体旋转起来"太轻"或"太沉"，
而且碰撞响应会明显不对。所以这里每个公式都从定义出发写清楚。

单位：质量 kg，长度 m，惯量 kg·m²。
"""

from __future__ import annotations

from .geometry import (
    BoxGeometry,
    CapsuleGeometry,
    ConvexHullGeometry,
    CylinderGeometry,
    GeometrySpec,
    SphereGeometry,
)


def box_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """长方体关于质心的转动惯量对角元。

    推导：对边长 a、b、c 的均匀长方体，
    ``Ixx = ∫(y²+z²)dm = m(b²+c²)/12``，其余两轴同理轮换。

    参数
    ----
    mass : 质量，必须为正
    size : 三个轴向全长 (a, b, c)
    """
    a, b, c = size
    return (
        mass * (b * b + c * c) / 12.0,
        mass * (a * a + c * c) / 12.0,
        mass * (a * a + b * b) / 12.0,
    )


def sphere_inertia(mass: float, radius: float) -> tuple[float, float, float]:
    """实心球关于质心的转动惯量。

    球体各向同性，三个主轴惯量相同：``I = 2/5 m r²``。

    注意这只是"实心球"的结果；薄球壳是 ``2/3 m r²``，两者差得很多，
    不要混用。
    """
    i = 0.4 * mass * radius * radius
    return (i, i, i)


def cylinder_inertia(mass: float, radius: float, height: float) -> tuple[float, float, float]:
    """实心圆柱关于质心的转动惯量，轴线沿局部 Z 轴。

    推导：
    - 绕自身对称轴 ``Izz = 1/2 m r²``
    - 绕垂直于轴的直径 ``Ixx = Iyy = 1/12 m (3r² + h²)``

    返回顺序为 (Ixx, Iyy, Izz)，与几何体的局部坐标轴对应。
    """
    izz = 0.5 * mass * radius * radius
    ixx = mass * (3.0 * radius * radius + height * height) / 12.0
    return (ixx, ixx, izz)


def capsule_inertia(mass: float, radius: float, height: float) -> tuple[float, float, float]:
    """胶囊体关于质心的转动惯量，轴线沿局部 Z 轴。

    做法：把胶囊拆成"一个圆柱 + 两个半球"，分别算惯量后用**平行轴定理**
    平移到胶囊质心再相加。

    半球的两条公式（质量 m_h、半径 r，绕半球自身质心）：
    - 绕对称轴        ``Izz = 2/5 m_h r²``
    - 绕垂直直径      ``Ixx = 83/320 m_h r²``
    - 半球质心距底面   ``3r/8``

    参数
    ----
    mass   : 胶囊总质量
    radius : 半球半径
    height : 中间圆柱段高度（不含两端半球）
    """
    r, h = radius, height

    # 1) 按体积比例把总质量分配到"圆柱段"和"两个半球（合起来算一份）"
    vol_cyl = 3.141592653589793 * r * r * h
    vol_hemis = (4.0 / 3.0) * 3.141592653589793 * r ** 3
    vol_total = vol_cyl + vol_hemis
    mass_cyl = mass * vol_cyl / vol_total
    mass_hemis = mass * vol_hemis / vol_total   # 两个半球的总质量
    mass_hemi = 0.5 * mass_hemis                # 单个半球质量

    # 2) 圆柱段：质心恰在胶囊中心，无需平移
    izz = 0.5 * mass_cyl * r * r
    ixx = mass_cyl * (3.0 * r * r + h * h) / 12.0

    # 3) 两个半球：先算绕各自质心的惯量，再用平行轴定理平移到胶囊中心
    hemi_izz_local = (2.0 / 5.0) * mass_hemi * r * r
    hemi_ixx_local = (83.0 / 320.0) * mass_hemi * r * r

    # 半球质心相对胶囊中心的轴向距离
    offset = 0.5 * h + 3.0 * r / 8.0

    izz += 2.0 * hemi_izz_local                      # 沿轴平移不改变绕轴惯量
    ixx += 2.0 * (hemi_ixx_local + mass_hemi * offset * offset)

    return (ixx, ixx, izz)


def compute_inertia(geometry: GeometrySpec, mass: float) -> tuple[float, float, float]:
    """按几何体类型分派到对应的惯量公式。

    这是求解器唯一需要调用的入口——求解器不关心具体几何形状，
    只拿到三个对角惯量用于计算逆惯量。

    参数
    ----
    geometry : 任意 ``GeometrySpec`` 子类实例
    mass     : 刚体质量，必须为正

    异常
    ----
    ValueError  : 质量为非正数
    TypeError   : 几何类型尚未支持惯量计算
    """
    if mass <= 0.0:
        raise ValueError(f"质量必须为正数，收到：{mass}")

    if isinstance(geometry, BoxGeometry):
        return box_inertia(mass, geometry.size)
    if isinstance(geometry, SphereGeometry):
        return sphere_inertia(mass, geometry.radius)
    if isinstance(geometry, CylinderGeometry):
        return cylinder_inertia(mass, geometry.radius, geometry.height)
    if isinstance(geometry, CapsuleGeometry):
        return capsule_inertia(mass, geometry.radius, geometry.height)
    if isinstance(geometry, ConvexHullGeometry):
        # 凸包的张量一般不是对角的，这里返回的是**主惯量**——
        # 它只有在物体姿态里并入了 ``inertia_frame()`` 之后才是正确的对角表示。
        # 所以 RigidBody 必须同时使用 geometry.inertia_frame()，两者是一对。
        return geometry.principal_moments(mass)

    raise TypeError(
        f"几何类型 {type(geometry).__name__} 尚未实现惯量计算，"
        f"请在 inertia.py 中补充对应公式"
    )
