# -*- coding: utf-8 -*-
"""初始状态校验：禁止物体在仿真开始前就互相嵌入。

为什么这件事必须拦下来
----------------------
刚体冲量法无法表达"多体同时碰撞"对材料刚度的依赖（这是刚性建模的固有缺陷）。
在离散仿真里，**唯一能让求解器落在"波传播"这一档（也就是钢球等硬物的真实行为）
的办法，是保证接触是一个接一个出现的**。

如果场景一开始就有物体互相嵌入，那么这些接触在第一步就**同时有效**，
求解器只能给出"整体同时变形"的解——那是软材料的物理，不是硬物的。

实测（见 ``tests/test_multicontact.py`` 第 8 项）：三球对撞，仅把初始间距从
恰好相切改成一微米的重叠，结果就从 ``[0, 0, 3]``（只飞一个球，正确）
跳到 ``[-1, 2, 2]``（两个球一起飞）。**一微米的差别，结果完全不同。**

所以校验不是洁癖，它直接决定仿真落在哪个物理区间。而且这类错误
（"我把两个球放在同一个位置了"）在手工写 JSON 时非常容易犯，
越早报错越省时间。
"""

from __future__ import annotations

from dataclasses import dataclass

from .body import RigidBody
from .narrowphase import GroundPlane, detect_contacts

#: 默认允许的初始重叠（米）。手工编写场景时的坐标四舍五入通常远小于这个量级，
#: 而真正的摆放错误（物体叠在一起）会是厘米级以上，因此 1 毫米足以区分两者。
DEFAULT_OVERLAP_TOLERANCE = 1e-3


@dataclass
class OverlapViolation:
    """一处在仿真开始前就已经发生的穿透。"""

    label: str
    """人类可读的物体描述，如 ``'ball_a' <-> 'ball_b'``"""

    penetration: float
    """穿透深度（米）"""

    def __str__(self) -> str:
        """格式化为一行错误信息。"""
        return f"{self.label}：初始穿透 {self.penetration * 1000:.3f} mm"


def find_initial_overlaps(
    bodies: list[RigidBody],
    ground: GroundPlane | None = None,
    tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
) -> list[OverlapViolation]:
    """找出初始状态中超出容差的互相嵌入。

    直接复用窄相检测，因此判据与求解器完全一致——校验通过的场景，
    求解器看到的就是同一个接触集合，不会出现"校验说没问题但求解时穿透"的情况。

    参数
    ----
    bodies    : 刚体列表
    ground    : 地面；None 表示自由空间
    tolerance : 允许的重叠深度（米），默认 1 mm

    返回
    ----
    list[OverlapViolation] : 按穿透深度降序排列，便于一眼看到最严重的问题
    """
    violations: list[OverlapViolation] = []

    for manifold in detect_contacts(bodies, ground):
        depth = manifold.deepest.penetration
        if depth <= tolerance:
            continue

        if manifold.body_a < 0:
            # body_a < 0 表示地面（GROUND_BODY）
            label = f"地面 <-> '{bodies[manifold.body_b].name}'"
        else:
            label = (
                f"'{bodies[manifold.body_a].name}' <-> "
                f"'{bodies[manifold.body_b].name}'"
            )
        violations.append(OverlapViolation(label=label, penetration=depth))

    violations.sort(key=lambda v: v.penetration, reverse=True)
    return violations


def format_violations(violations: list[OverlapViolation], limit: int = 8) -> str:
    """把违规列表格式化成可直接打印的多行文本。

    参数
    ----
    violations : ``find_initial_overlaps`` 的返回值
    limit      : 最多列出多少条，避免物体很多时刷屏
    """
    if not violations:
        return "（无初始重叠）"

    lines = [f"发现 {len(violations)} 处初始穿透（仿真开始前物体就已经互相嵌入）："]
    for violation in violations[:limit]:
        lines.append(f"  - {violation}")
    if len(violations) > limit:
        lines.append(f"  ... 另有 {len(violations) - limit} 处未列出")

    lines.append(
        "请把它们分开到互不接触的位置再开始仿真。原因：初始就嵌入的接触会在"
        "第一步同时生效，使求解器给出“整体同时变形”的解，而那是软材料的物理；"
        "对于硬物（钢球等），正确行为是接触按时间先后逐个传递。"
    )
    return "\n".join(lines)
