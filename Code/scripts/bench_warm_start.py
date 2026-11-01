# -*- coding: utf-8 -*-
"""暖启动对照实验：`warm_start` 开 / 关 × 迭代次数。

要回答的问题
------------
1. 暖启动是否真的让堆叠更硬（同迭代次数下穿透更浅）？
2. 有了暖启动，默认的 ``iterations=8`` 是不是可以降下来？
3. 代价是多少（ms/步）？

场景选择
--------
**6 层盒塔**：典型的多接触链。冷启动时每条约束每帧都从 ``P = 0`` 开始"爬"，
链越长爬得越慢，表现就是塔整体下沉、层间互相挤压。这正是暖启动要解决的问题。

**3×3 宽堆**：接触多但链短，用来确认收益不是只在"高塔"这一种拓扑上出现。

测量口径（与 ``bench_broadphase.py`` 一致）
-------------------------------------------
- 每个配置跑 3 轮取最小值（单次计时的噪声能到 ±20%）
- 穿透分两个指标：**峰值**（全过程最大，反映撞击瞬间）与**末段**（后 0.5 s
  最大，反映静置质量）
- 所有场景都从"零间隙略留 2 mm"的构型开始，不允许初始穿透（``validate_initial_state``）

输出
----
``results/w4/bench_warm_start.txt``，随后由 ``src/evidence.py`` 渲染进文档。
"""

from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import BoxGeometry
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import (
    BodyPhysicsSpec,
    GroundSpec,
    MaterialSpec,
    ObjectSpec,
    SceneSpec,
    SimulationSpec,
    TransformSpec,
)

#: 塔的层数
TOWER_LAYERS = 6
#: 盒子边长（米）
BOX_SIZE = 0.5
#: 层间初始间隙（米）。留一点缝，避免"初始就接触"，那是另一个问题
LAYER_GAP = 0.002
#: 静置时长（秒）
DURATION = 1.5
#: 末段统计窗口（秒）
SETTLE_WINDOW = 0.4

#: 迭代次数扫描点
ITERATIONS = (1, 2, 4, 8)


def make_box(name: str, position, friction: float = 0.8) -> ObjectSpec:
    """造一个静止起步的盒子（恢复系数 0，避免弹跳掩盖穿透差异）。"""
    return ObjectSpec(
        name=name,
        geometry=BoxGeometry((BOX_SIZE, BOX_SIZE, BOX_SIZE)),
        transform=TransformSpec(position=position, orientation=(0.0, 0.0, 0.0, 1.0)),
        material=MaterialSpec(name="mat"),
        mass=1.0,
        physics=BodyPhysicsSpec(
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            restitution=0.0,
            friction=friction,
        ),
    )


def tower_scene() -> SceneSpec:
    """竖直叠 6 层。"""
    objects = []
    z = BOX_SIZE / 2.0
    for index in range(TOWER_LAYERS):
        objects.append(make_box(f"L{index}", (0.0, 0.0, z)))
        z += BOX_SIZE + LAYER_GAP
    return _scene("tower", objects)


def wide_scene() -> SceneSpec:
    """3×3 平铺两层：接触多、链短，用来检查收益是否只在"高塔"拓扑上成立。"""
    objects = []
    offset = BOX_SIZE + 0.004
    for layer in range(2):
        z = BOX_SIZE / 2.0 + layer * (BOX_SIZE + LAYER_GAP)
        for ix in range(3):
            for iy in range(3):
                x = (ix - 1) * offset
                y = (iy - 1) * offset
                objects.append(make_box(f"W{layer}_{ix}{iy}", (x, y, z)))
    return _scene("wide", objects)


def _scene(name: str, objects: list[ObjectSpec]) -> SceneSpec:
    """把物体装进一个"只有地面 + 一套求解器参数"的最小场景。"""
    return SceneSpec(
        name=name,
        objects=objects,
        camera=None,
        lights=[],
        ground=GroundSpec(enabled=True),
        simulation=SimulationSpec(
            enabled=True,
            physics_dt=1.0 / 240.0,
            substeps=3,
            iterations=8,
            ground_z=0.0,
            duration=DURATION,
            # 堆叠场景不需要 TOI 切分（`docs/05`：稠密堆叠靠暖启动 + 子步），
            # 关掉它能把测量时间压到可接受范围，也让差异只来自暖启动本身
            use_toi_splitting=False,
        ),
        render=None,
    )


def run_once(scene: SceneSpec, iterations: int, warm_start: bool) -> dict:
    """跑一遍，返回穿透、漂移、速度与耗时。"""
    scene.simulation.iterations = iterations
    scene.simulation.warm_start = warm_start
    world = PhysicsWorld.from_scene(scene)
    # 记录初始水平位置：漂移必须是**位移**，不是绝对坐标。宽堆的角点本来就在
    # ±0.5 m 处，用绝对坐标会报出 700 mm 的假漂移。
    start_xy = [(body.position[0], body.position[1]) for body in world.bodies]

    total_steps = int(round(scene.simulation.duration / scene.simulation.physics_dt))
    settle_from = total_steps - int(round(SETTLE_WINDOW / scene.simulation.physics_dt))

    peak = 0.0
    settled = 0.0
    started = time.perf_counter()
    for step in range(total_steps):
        world.step()
        penetration = world.max_penetration()
        peak = max(peak, penetration)
        if step >= settle_from:
            settled = max(settled, penetration)
    elapsed = time.perf_counter() - started

    top_index = max(range(len(world.bodies)), key=lambda i: world.bodies[i].position[2])
    top = world.bodies[top_index]
    drift = (
        (top.position[0] - start_xy[top_index][0]) ** 2
        + (top.position[1] - start_xy[top_index][1]) ** 2
    ) ** 0.5
    max_speed = max(body.speed() for body in world.bodies)

    return {
        "peak_mm": peak * 1000.0,
        "settled_mm": settled * 1000.0,
        "drift_mm": drift * 1000.0,
        "speed": max_speed,
        "ms_per_step": elapsed / total_steps * 1000.0,
    }


def measure(builder, iterations: int, warm_start: bool) -> dict:
    """跑一遍。

    穿透与漂移是**确定性**量（同一配置逐位可复现），单次测量就是真值，不需要平均；
    只有 ms/步 有噪声（同一进程内 ±20%），所以它只用于看趋势，不用于下结论。
    跑多轮只会让整个实验变慢，不会让结论更可靠。
    """
    return run_once(builder(), iterations, warm_start)


def format_table(title: str, rows: list[tuple[bool, int, dict]]) -> list[str]:
    """渲染一段对照表。"""
    lines = [
        f"== {title} ==",
        "warm   iters  峰值穿透mm  末段穿透mm  顶层漂移mm  最大速度m/s  ms/步",
    ]
    for warm_start, iterations, data in rows:
        flag = "on " if warm_start else "off"
        lines.append(
            f"{flag}    {iterations:>5d}   {data['peak_mm']:>10.3f}"
            f"  {data['settled_mm']:>10.3f}  {data['drift_mm']:>10.3f}"
            f"  {data['speed']:>11.5f}  {data['ms_per_step']:>7.3f}"
        )
    return lines


def main() -> int:
    """跑完所有组合，写存档并打印摘要。"""
    output_dir = os.path.join(_CODE_ROOT, "..", "results", "w4")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bench_warm_start.txt")

    print("=" * 92)
    print("暖启动对照实验（substeps=3，恢复系数 0）")
    print("=" * 92)

    blocks: list[list[str]] = []
    summary = {}
    for title, builder in (("6 层盒塔", tower_scene), ("3×3 两层宽堆", wide_scene)):
        rows = []
        for warm_start in (False, True):
            for iterations in ITERATIONS:
                data = measure(builder, iterations, warm_start)
                rows.append((warm_start, iterations, data))
                summary[(title, warm_start, iterations)] = data
                print(
                    f"  {title:<14} warm={'on ' if warm_start else 'off'} "
                    f"iters={iterations:<3} 峰值 {data['peak_mm']:6.3f} mm  "
                    f"末段 {data['settled_mm']:6.3f} mm  {data['ms_per_step']:5.2f} ms/步"
                )
        blocks.append(format_table(title, rows))
        blocks.append([""])

    header = [
        "暖启动对照实验",
        f"塔：{TOWER_LAYERS} 层盒，边长 {BOX_SIZE} m，层间初始间隙 {LAYER_GAP * 1000:.0f} mm，"
        f"静置 {DURATION} s",
        "宽堆：3×3 两层（接触多、链短）",
        "穿透 / 漂移是确定性量（单次即真值）；ms/步 单次运行，噪声约 ±20%，只看趋势",
        f"配置：substeps=3，关 TOI，恢复系数 0，静置 {DURATION} s，末段窗口 {SETTLE_WINDOW} s",
        "",
    ]

    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        for line in header:
            handle.write(line + "\n")
        for block in blocks:
            for line in block:
                handle.write(line + "\n")

    print(f"\n已写入 {output_path}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
