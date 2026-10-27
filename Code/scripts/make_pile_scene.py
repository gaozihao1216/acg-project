# -*- coding: utf-8 -*-
"""生成"容器堆积"演示场景：一个静态盒容器 + 大量球与盒子落入堆叠。

为什么用生成而不是手写 JSON
--------------------------
手工摆 40 多个物体既繁琐又容易违反"禁止初始互相嵌入"的约束。
用脚本按确定性伪随机（固定种子）生成，可以保证：

- 每次生成结果完全相同（可复现）
- 生成时立即检查物体间距，不会出现初始重叠

这个场景同时覆盖了三种碰撞组合：球-球、球-盒（容器壁）、盒-盒（盒子互相压在一起），
而且物体数量足以让宽阶段的收益可见。

用法::

    python Code/scripts/make_pile_scene.py [--out Code/scenes/demo_pile.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

#: 容器内半边长（墙内表面到中心的距离）。
#:
#: 这个值要和物体数量匹配：容器太大，几十个物体会摊成薄薄一层，
#: 既不像"堆积"也没什么碰撞可言；太小又会在下落过程中卡死。
#: 实测 0.85 m 半径配约 50 个物体能得到 2–3 层、有真实互相支撑的堆。
INNER_HALF = 0.85

#: 墙厚与墙高。
#: 墙不能太高——相机需要俯视才能看见容器内部，墙太高会把内容全挡住。
WALL_THICKNESS = 0.14
WALL_HEIGHT = 1.10

#: 球与盒的颜色池（RGB 0–1）
PALETTE = [
    [0.16, 0.40, 0.78],   # 蓝
    [0.78, 0.28, 0.20],   # 砖红
    [0.20, 0.55, 0.42],   # 青绿
    [0.82, 0.62, 0.22],   # 琥珀
    [0.50, 0.32, 0.70],   # 紫
    [0.72, 0.72, 0.74],   # 浅灰
    [0.86, 0.45, 0.20],   # 橙
]


def make_material(name: str, color) -> dict:
    """构造材质段。"""
    return {
        "name": name,
        "base_color": color,
        "roughness": 0.38,
        "metallic": 0.0,
    }


def build_scene(sphere_count: int, box_count: int, seed: int) -> dict:
    """生成场景描述字典。"""
    rng = random.Random(seed)

    objects = []

    # --- 容器：四个静态墙 ---
    walls = [
        ("wall_x_neg", (-INNER_HALF - WALL_THICKNESS / 2.0, 0.0, WALL_HEIGHT / 2.0),
         (WALL_THICKNESS, 2.0 * INNER_HALF + 2.0 * WALL_THICKNESS, WALL_HEIGHT)),
        ("wall_x_pos", (INNER_HALF + WALL_THICKNESS / 2.0, 0.0, WALL_HEIGHT / 2.0),
         (WALL_THICKNESS, 2.0 * INNER_HALF + 2.0 * WALL_THICKNESS, WALL_HEIGHT)),
        ("wall_y_neg", (0.0, -INNER_HALF - WALL_THICKNESS / 2.0, WALL_HEIGHT / 2.0),
         (2.0 * INNER_HALF, WALL_THICKNESS, WALL_HEIGHT)),
        ("wall_y_pos", (0.0, INNER_HALF + WALL_THICKNESS / 2.0, WALL_HEIGHT / 2.0),
         (2.0 * INNER_HALF, WALL_THICKNESS, WALL_HEIGHT)),
    ]
    for name, position, size in walls:
        objects.append(
            {
                "name": name,
                "geometry": {"type": "box", "size": list(size)},
                "transform": {"position": list(position)},
                "material": make_material(
                    f"mat_{name}", [0.42, 0.42, 0.45]
                ),
                "mass": 500.0,
                "physics": {"static": True, "restitution": 0.10, "friction": 0.75},
            }
        )

    # --- 已放置物体的记录，用于间距检查 ---
    placed: list[tuple[float, float, float, float]] = []   # (x, y, z, 半径)

    def far_enough(x, y, z, radius) -> bool:
        """与所有已放置物体保持至少 1 cm 的间隙。"""
        for px, py, pz, pr in placed:
            distance = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
            if distance < radius + pr + 0.01:
                return False
        return True

    def spawn(kind: str, index: int) -> dict | None:
        """尝试放置一个物体，失败返回 None。"""
        for _ in range(200):
            if kind == "sphere":
                radius = rng.uniform(0.11, 0.19)
                size = None
                extent = radius
            else:
                side = rng.uniform(0.26, 0.38)
                size = [side, side, side]
                radius = side * math.sqrt(3.0) / 2.0   # 外接球
                extent = side / 2.0

            margin = INNER_HALF - extent - 0.02
            x = rng.uniform(-margin, margin)
            y = rng.uniform(-margin, margin)
            z = rng.uniform(2.30, 4.20)

            if not far_enough(x, y, z, radius):
                continue
            placed.append((x, y, z, radius))

            color = PALETTE[index % len(PALETTE)]
            name = f"{kind}_{index:02d}"
            if kind == "sphere":
                geometry = {"type": "sphere", "radius": radius}
            else:
                geometry = {"type": "box", "size": size}

            entry = {
                "name": name,
                "geometry": geometry,
                "transform": {"position": [x, y, z]},
                "material": make_material(f"mat_{name}", color),
                "mass": 20.0,
                "physics": {"restitution": 0.16, "friction": 0.62},
            }
            if kind == "box":
                entry["transform"]["orientation_euler_deg"] = [
                    rng.uniform(-25.0, 25.0) for _ in range(3)
                ]
            return entry
        return None

    # 盒子少一些，球多一些——球滚动更活跃，画面更好看
    for index in range(box_count):
        entry = spawn("box", index)
        if entry is not None:
            objects.append(entry)
    for index in range(sphere_count):
        entry = spawn("sphere", index)
        if entry is not None:
            objects.append(entry)

    dynamic_count = len(objects) - len(walls)
    print(f"生成 {dynamic_count} 个动态物体（{box_count} 盒 + {sphere_count} 球）+ {len(walls)} 面静态墙")

    return {
        "name": "demo_pile",
        "camera": {
            # 俯视约 44°：必须高于墙顶（1.1 m）才能看见容器内部。
            # 角度再小一点近处的墙就会挡住整堆物体。
            "position": [2.00, -2.62, 3.55],
            "target": [0.0, 0.0, 0.40],
            "fov": 46.0,
        },
        "lights": [
            {
                "name": "key",
                "type": "area",
                "position": [2.6, -2.4, 5.2],
                "target": [0.0, 0.0, 0.9],
                "energy": 520.0,
                "size": 3.4,
                "color": [1.0, 0.96, 0.90],
            },
            {
                "name": "fill",
                "type": "area",
                "position": [-3.6, -2.8, 2.8],
                "target": [0.0, 0.0, 0.8],
                "energy": 160.0,
                "size": 4.6,
                "color": [0.86, 0.92, 1.0],
            },
            {
                "name": "rim",
                "type": "area",
                "position": [0.4, 4.2, 3.6],
                "target": [0.0, 0.0, 1.0],
                "energy": 240.0,
                "size": 3.4,
                "color": [1.0, 1.0, 1.0],
            },
        ],
        "ground": {
            "enabled": True,
            "size": 40.0,
            "color": [0.60, 0.60, 0.63],
            "roughness": 0.58,
        },
        "objects": objects,
        "simulation": {
            "enabled": True,
            "physics_dt": 1.0 / 240.0,
            "substeps": 3,
            "iterations": 8,
            "baumgarte": 0.25,
            "slop": 0.002,
            "restitution_threshold": 0.6,
            "gravity": [0.0, 0.0, -9.81],
            "ground_z": 0.0,
            "ground_friction": 0.72,
            "ground_restitution": 0.05,
            "linear_damping": 0.05,
            "angular_damping": 1.0,
            "use_toi_splitting": True,
            "broadphase": "spatial_hash",
            "render_fps": 24,
            "duration": 2.4,
        },
        "render": {
            "resolution": [960, 540],
            "samples": 40,
            "engine": "CYCLES",
            "denoise": True,
            "world_color": [0.055, 0.058, 0.07],
            "world_strength": 0.32,
            "exposure": 0.0,
            "film_transparent": False,
        },
    }


def main() -> int:
    """解析参数、生成并写出场景。"""
    parser = argparse.ArgumentParser(description="生成容器堆积演示场景")
    parser.add_argument(
        "--out", default=os.path.join(_CODE_ROOT, "scenes", "demo_pile.json")
    )
    parser.add_argument("--spheres", type=int, default=40)
    parser.add_argument("--boxes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()

    scene = build_scene(args.spheres, args.boxes, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"已写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
