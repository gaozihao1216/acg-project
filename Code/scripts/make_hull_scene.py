# -*- coding: utf-8 -*-
"""生成凸包（任意凸多面体）演示场景。

这个场景要证明的东西
--------------------
前面的盒-盒演示只能说明"有 6 个面、12 条棱的东西能相撞"。它无法区分
"SAT 真的推广到了任意凸多面体"和"代码里其实只认盒子"。

所以这里用三种**面数不同、且不轴对齐**的凸体：

- 正四面体（4 面 6 棱）
- 正八面体（8 面 12 棱）
- 不对称三棱柱（5 面 9 棱）

它们同时落地、互相碰撞、并以各自的姿态静置。如果窄相只对盒子有效，
这些物体会直接穿过彼此或穿过地面。

同时放一个球，用来验证球-凸包这条独立的分支。

运行::

    python Code/scripts/make_hull_scene.py
"""

from __future__ import annotations

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.meshutil import prism_mesh

SCENE_PATH = os.path.join(_CODE_ROOT, "scenes", "demo_hull.json")


def tetrahedron(scale: float = 0.5):
    """正四面体顶点（棱长 = scale）。"""
    a = scale / math.sqrt(2.0)
    return [
        (a, a, a),
        (a, -a, -a),
        (-a, a, -a),
        (-a, -a, a),
    ]


def octahedron(radius: float = 0.42):
    """正八面体顶点。"""
    r = radius
    return [(r, 0.0, 0.0), (-r, 0.0, 0.0), (0.0, r, 0.0),
            (0.0, -r, 0.0), (0.0, 0.0, r), (0.0, 0.0, -r)]


def asymmetric_prism():
    """不对称三棱柱顶点：惯量张量有非对角项，必须用完整张量。"""
    vertices, _ = prism_mesh(
        [(0.0, 0.0), (0.9, 0.0), (0.3, 1.0)], height=0.55
    )
    return vertices


def body(name, points, position, euler_deg, color, mass):
    """构造一个 hull 物体描述。"""
    return {
        "name": name,
        "geometry": {
            "type": "hull",
            "points": [[p[0], p[1], p[2]] for p in points],
        },
        "transform": {
            "position": list(position),
            "euler_degrees": list(euler_deg),
        },
        "mass": mass,
        "material": {"base_color": color, "roughness": 0.45, "metallic": 0.05},
        "physics": {"restitution": 0.15, "friction": 0.75},
    }


def sphere(name, radius, position, color, mass):
    return {
        "name": name,
        "geometry": {"type": "sphere", "radius": radius},
        "transform": {"position": list(position)},
        "mass": mass,
        "material": {"base_color": color, "roughness": 0.35, "metallic": 0.0},
        "physics": {"restitution": 0.30, "friction": 0.6},
    }


def build() -> dict:
    objects = [
        # 三个凸体在同一区域上方依次落下，必然互相碰撞。
        # 垂直间距必须留足：初始嵌入会被 validate_initial_state 拒绝
        # （这正是那道校验存在的意义——同时生效的接触会得到"软材料"的解）。
        body(
            "tetra", tetrahedron(0.62), (-0.30, 0.05, 1.30),
            (18.0, 12.0, 25.0), (0.20, 0.55, 0.85), 3.2,
        ),
        body(
            "octa", octahedron(0.42), (0.28, -0.06, 2.20),
            (30.0, -20.0, 10.0), (0.85, 0.42, 0.22), 3.6,
        ),
        body(
            "prism", asymmetric_prism(), (0.02, 0.10, 3.10),
            (-15.0, 35.0, -8.0), (0.35, 0.72, 0.40), 3.0,
        ),
        # 球从侧上方砸下来，验证球-凸包分支
        sphere("ball", 0.22, (-1.18, 0.34, 1.10), (0.92, 0.84, 0.30), 2.2),
    ]

    return {
        "name": "demo_hull",
        "camera": {
            # 物体从 z≈3.1 落到 z≈0.25，竖直跨度 3 米，机位必须拉得足够远
            "position": [3.05, -4.55, 2.70],
            "target": [-0.15, 0.0, 0.85],
            "fov": 42.0,
        },
        "ground": {
            "enabled": True,
            "z": 0.0,
            "friction": 0.8,
            "restitution": 0.05,
            "color": [0.22, 0.23, 0.25],
        },
        "objects": objects,
        "simulation": {
            "enabled": True,
            "physics_dt": 0.004166666666666667,
            "substeps": 3,
            "iterations": 8,
            "baumgarte": 0.25,
            "slop": 0.002,
            "restitution_threshold": 0.6,
            "gravity": [0.0, 0.0, -9.81],
            "ground_z": 0.0,
            "ground_friction": 0.8,
            "ground_restitution": 0.05,
            "linear_damping": 0.05,
            "angular_damping": 1.0,
            "use_toi_splitting": True,
            "max_events_per_substep": 8,
            "broadphase": "spatial_hash",
            "broadphase_cell_size": 0.0,
            "render_fps": 24,
            "duration": 1.9,
        },
    }


def main() -> int:
    scene = build()
    with open(SCENE_PATH, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"已写出 {SCENE_PATH}")
    print(f"物体数：{len(scene['objects'])}")
    for spec in scene["objects"]:
        geometry = spec["geometry"]
        if geometry["type"] == "hull":
            print(f"  {spec['name']:>6}: hull，输入点 {len(geometry['points'])} 个")
        else:
            print(f"  {spec['name']:>6}: {geometry['type']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
