# -*- coding: utf-8 -*-
"""刚体变换（位置 + 姿态 + 缩放）模块。

本模块同样是纯数学，不依赖 bpy / numpy，可被渲染脚本与求解器共用。

约定
----
- ``position``    : 世界坐标系下的平移量 (x, y, z)
- ``orientation`` : 姿态四元数 (w, x, y, z)
- ``scale``       : 各轴缩放 (sx, sy, sz)，仅用于**视觉网格**，
                    物理尺寸由 ``GeometrySpec`` 单独表达。

为什么要把几何尺寸和缩放分开
----------------------------
如果允许几何体通过 ``scale`` 表达真实尺寸，那么质量、体积、惯量的计算
就会隐式依赖父节点的缩放链，一旦嵌套层级变化数值就会错。
所以本项目的规定是：``GeometrySpec`` 里写的尺寸**就是最终物理尺寸**，
``scale`` 只影响外观（例如把球压扁成椭球做装饰），物理计算一律忽略它。
"""

from __future__ import annotations

from typing import Sequence

from ..mathutils import quaternion as quat

Vec3 = Sequence[float]
Quat = Sequence[float]


class TransformSpec:
    """描述一个刚体（或相机、灯光）在世界坐标系中的位姿。

    这是一个轻量的数据容器，只负责"解释 JSON 字段"和"做矩阵运算"，
    不持有任何仿真状态。
    """

    __slots__ = ("position", "orientation", "scale")

    def __init__(
        self,
        position: Vec3 = (0.0, 0.0, 0.0),
        orientation: Quat = quat.quat_identity(),
        scale: Vec3 = (1.0, 1.0, 1.0),
    ) -> None:
        """构造一个变换。

        参数
        ----
        position    : 平移，默认原点
        orientation : 姿态四元数，默认无旋转
        scale       : 视觉缩放，默认不缩放

        三个参数都有默认值，所以**单位变换就是 ``TransformSpec()``**，本模块不再
        单独提供单位变换构造函数：多一个函数只会多一处"默认值到底是谁"的分歧。
        """
        self.position = (float(position[0]), float(position[1]), float(position[2]))
        self.orientation = quat.quat_normalize(orientation)
        self.scale = (float(scale[0]), float(scale[1]), float(scale[2]))

    # -- 构造 / 序列化 ----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict | None) -> "TransformSpec":
        """从场景 JSON 的 ``transform`` 字段构造变换。

        字段缺省时使用默认值（原点、无旋转、无缩放），因此场景文件里
        可以只写关心的那部分。支持两种书写姿态的方式：

        - ``orientation``            : 直接给四元数 ``[w, x, y, z]``
        - ``orientation_euler_deg``  : 给 XYZ 欧拉角 ``[rx, ry, rz]``（角度制）

        后者对人工编写更友好，内部会转成四元数。
        """
        if not data:
            return cls()

        if "orientation_euler_deg" in data:
            rx, ry, rz = data["orientation_euler_deg"]
            orientation = quat.quat_from_euler_degrees(rx, ry, rz)
        else:
            orientation = data.get("orientation", quat.quat_identity())

        return cls(
            position=data.get("position", (0.0, 0.0, 0.0)),
            orientation=orientation,
            scale=data.get("scale", (1.0, 1.0, 1.0)),
        )

    def to_dict(self) -> dict:
        """导出为可写回 JSON 的字典，便于记录实验所用的确切参数。"""
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "scale": list(self.scale),
        }

    # -- 矩阵与坐标变换 ---------------------------------------------------

    def to_matrix(self) -> tuple[tuple[float, float, float, float], ...]:
        """返回 4x4 齐次变换矩阵（行优先嵌套元组）。

        矩阵结构为::

            [ R * S   t ]
            [   0     1 ]

        其中 ``R`` 是姿态旋转矩阵，``S`` 是缩放矩阵，``t`` 是平移向量。
        复合变换（父子层级）时可以直接做矩阵乘法。
        """
        r = quat.quat_to_matrix(self.orientation)
        sx, sy, sz = self.scale
        px, py, pz = self.position

        return (
            (r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, px),
            (r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, py),
            (r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, pz),
            (0.0, 0.0, 0.0, 1.0),
        )

    def transform_point(self, point: Vec3) -> tuple[float, float, float]:
        """把局部坐标系下的点变换到世界坐标系。

        求解器导出接触点、计算力臂时需要这个函数。
        """
        m = self.to_matrix()
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
        )

    def transform_direction(self, direction: Vec3) -> tuple[float, float, float]:
        """把局部坐标系下的**方向**变换到世界坐标系（忽略平移）。

        与 ``transform_point`` 的区别：方向向量只受旋转影响，不受平移影响。
        计算接触法向、力臂方向时用这个。
        """
        return quat.quat_rotate_vector(self.orientation, direction)

    def compose(self, other: "TransformSpec") -> "TransformSpec":
        """复合变换：返回"先应用 ``other``，再应用 ``self``"的结果。

        用于实现父子层级。注意缩放按各轴直接相乘，这只在旋转对齐时严格成立，
        本项目不依赖嵌套缩放，因此足够。
        """
        # 位置：用 self 的旋转作用在 other 的平移上，再叠加 self 的平移
        rotated = quat.quat_rotate_vector(self.orientation, other.position)
        new_pos = (
            self.position[0] + rotated[0] * self.scale[0],
            self.position[1] + rotated[1] * self.scale[1],
            self.position[2] + rotated[2] * self.scale[2],
        )
        new_orientation = quat.quat_normalize(
            quat.quat_multiply(self.orientation, other.orientation)
        )
        new_scale = (
            self.scale[0] * other.scale[0],
            self.scale[1] * other.scale[1],
            self.scale[2] * other.scale[2],
        )
        return TransformSpec(new_pos, new_orientation, new_scale)
