# -*- coding: utf-8 -*-
"""接触点与接触流形（contact manifold）数据结构。

为什么要单独定义"流形"而不是简单的"接触对"
--------------------------------------------
球与球之间永远是**一个**接触点，但盒子不一样：

- 面贴面 → 4 个接触点
- 棱碰棱 → 2 个接触点
- 顶点碰面 → 1 个接触点

如果盒子的接触只保留一个点，物体会绕着那个点无限旋转、堆叠会持续抖动。
所以从第一天起就把接口定成"一个接触对 → 多个接触点"，后面加盒碰撞时
求解器一行都不用改。

法向约定
--------
``normal`` 是**单位向量，方向从 body_a 指向 body_b**。

这个约定必须全局一致，否则冲量的符号会反，表现为"碰撞时物体互相吸引"
——一种很难从现象上直接看出来、但排查起来非常痛苦的 bug。

``body_a == GROUND_BODY`` 表示静态地面，求解器会把它当作质量无穷大处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 静态地面的伪 body 索引。求解器遇到它时使用"质量无穷大、速度为零"的替身。
GROUND_BODY: int = -1


@dataclass
class ContactPoint:
    """单个接触点。

    属性
    ----
    position    : 接触点在世界坐标系中的位置（通常取两表面之间的中点）
    normal      : 单位法向，从 body_a 指向 body_b
    penetration : 穿透深度（米），**恒为非负**。0 表示恰好相切
    feature_id  : 接触特征的稳定编号，用于跨帧匹配（暖启动的挂钩）

    ``feature_id`` 的契约
    --------------------
    **作用域**：只在"同一个物体对"内有意义。求解器的缓存键是
    ``(body_a, body_b, feature_id)``，不同物体对之间允许重复。

    **两条硬性要求**：

    1. **唯一**。同一物体对、同一次求解里的各个接触点，``feature_id`` 必须互不
       相同。重复的后果不是"不够精细"，而是错的：暖启动缓存按这个键存一个标量，
       后写的覆盖先写的，下一帧再把这个值施给所有共用该键的点。实测一个偏心
       负载的盒子压在地面上，四角累积法向冲量是 0.011 / 0.015 / 0.030 / 0.026
       （相差近 3 倍），共用键等于把净冲量放大、把四角的分布抹平。
    2. **由几何特征决定，而不是列表下标**。流形会按深度排序再截断，倾斜时点的
       个数也会变，下标于是每帧漂移，冲量就会喂到别的角上。现有编码：
       ``sat`` 用"参考面 + 入射顶点"，``convex_sat`` 用"参考面 + 标签"，
       ``gjk`` 用 ``tag_a * support.TAG_LIMIT + tag_b``，地面接触用
       ``support.feature_world_tagged`` 的标签（盒是顶点序号、回转体是
       "端面序号 × 4 + 圆周序号"）。

    **"特征驱动"不等于"跨帧持久"**。上面的编码保证"序号不随列表位置漂移"，而且
    盒顶点、凸包顶点、母线端点是稳定的几何特征，回转体端面槽位（圆周起点已固定
    在体坐标系）则是稳定的**体坐标采样身份**——圆周上并没有四个天然顶点，这四点
    仍是人为选取的离散槽位。整体强度仍是：

        批内唯一 = 硬要求（暖启动的键不能撞）
        跨帧稳定 = 局部特征级稳定，但遇到"模式切换"就失效

    端面接触一旦偏出 3° 容差就变成侧面或单点接触（标签整体换段），流形点数也会
    随姿态变化，P0-1 重建接触区域后采样点还会移动。所以真正的持久匹配仍需缓存
    两侧**局部锚点**，并在暖启动前校验法向夹角、锚点距离与间隙；不满足就丢弃，
    而不是凭一个整数 ID 强行复用。这一步尚未做。

    单点接触（球-球、球-面、严格凸体只有一个支撑点）固定为 0：只有一个点，
    既不会冲突，也没有稳定性可言。

    组合体在部件对的局部 ID 之上加 ``narrowphase._PART_FEATURE_STRIDE`` 的偏移，
    使不同部件对之间不撞号。
    """

    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    penetration: float
    feature_id: int = 0

    def __post_init__(self) -> None:
        """做最基本的合法性检查，把错误挡在产生它的地方。"""
        if self.penetration < 0.0:
            raise ValueError(f"穿透深度必须非负，收到：{self.penetration}")
        nx, ny, nz = self.normal
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if abs(length - 1.0) > 1e-6:
            raise ValueError(f"接触法向必须是单位向量，当前长度 {length:.6f}")


@dataclass
class ContactManifold:
    """一对物体之间的全部接触点。

    属性
    ----
    body_a      : 物体 A 的索引（或 ``GROUND_BODY``）
    body_b      : 物体 B 的索引
    points      : 接触点列表；球-球/球-面固定为 1 个，盒-盒最多 4 个
    friction    : 这一对物体的等效摩擦系数
    restitution : 这一对物体的等效恢复系数
    """

    body_a: int
    body_b: int
    points: list[ContactPoint] = field(default_factory=list)
    friction: float = 0.5
    restitution: float = 0.25

    def __post_init__(self) -> None:
        """校验索引与点数，避免构造出无意义的流形。"""
        if self.body_a == self.body_b:
            raise ValueError(f"接触流形的两侧不能是同一个物体：{self.body_a}")
        if not self.points:
            raise ValueError("接触流形至少要有一个接触点")

    @property
    def deepest(self) -> ContactPoint:
        """返回穿透最深的那个接触点。

        调试与日志打印时常用；求解器本身会处理全部接触点。
        """
        return max(self.points, key=lambda p: p.penetration)

    def __len__(self) -> int:
        """返回接触点数量，便于直接 ``len(manifold)``。"""
        return len(self.points)


def merge_pair_materials(
    restitution_a: float,
    restitution_b: float,
    friction_a: float,
    friction_b: float,
) -> tuple[float, float]:
    """把两侧材质的物理参数合成为接触对参数。

    规则（工程上常用的简化做法）：

    - **恢复系数取较大值**：只要有一侧是弹的，碰撞就应当弹。取平均会让
      "钢球撞泥巴"变得不弹，直觉上不合适。
    - **摩擦系数取几何平均**：``sqrt(mu_a * mu_b)``。这样只要一侧接近无摩擦，
      整体摩擦就低；两侧相同则等于自身。比取最小值更平滑。

    参数
    ----
    restitution_a, restitution_b : 两侧恢复系数
    friction_a, friction_b       : 两侧摩擦系数

    返回
    ----
    (restitution, friction) 二元组
    """
    restitution = max(float(restitution_a), float(restitution_b))
    friction = (max(0.0, float(friction_a)) * max(0.0, float(friction_b))) ** 0.5
    return restitution, friction
