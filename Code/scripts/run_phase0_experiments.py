# -*- coding: utf-8 -*-
"""Phase 0 参数对比实验。

思路：以 ``scenes/demo_basic.json`` 为基础场景，每项实验只覆盖少量参数
（相机、灯光、材质、布局），生成一份独立场景 JSON，再一次性交给 Blender
批量渲染。

这样做同时验证了两件事：

1. **渲染管线可用**——能稳定出图；
2. **场景参数化可用**——不改一行代码，只改 JSON 就能换画面。

用法（用**系统 Python** 运行，不是 Blender 的 Python）::

    python Code/scripts/run_phase0_experiments.py

可选参数::

    --blender PATH   指定 blender.exe
    --out-dir DIR    输出目录（默认 results/phase0）
    --only NAME      只跑某一项实验
    --list           只列出实验清单，不渲染
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(_HERE)                 # .../project/Code
PROJECT_ROOT = os.path.dirname(CODE_ROOT)          # .../project

# 让 src 可被导入（用于 apply_patches 与场景校验）
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from src.scene.patch import apply_patches          # noqa: E402
from src.scene.scene_spec import SceneSpec         # noqa: E402

BASE_SCENE = os.path.join(CODE_ROOT, "scenes", "demo_basic.json")
GENERATED_DIR = os.path.join(CODE_ROOT, "scenes", "generated")
RENDER_SCRIPT = os.path.join(CODE_ROOT, "render", "render_static.py")

#: Blender 默认安装位置（Windows）。找不到时可用 --blender 覆盖。
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"


# ---------------------------------------------------------------------------
# 实验定义
# ---------------------------------------------------------------------------
# 每项实验 = 名字 + 说明 + 参数覆盖（点号路径 -> 新值）。
# 覆盖路径的写法见 src/scene/patch.py。

def _sphere_row() -> list[dict]:
    """构造一排金属球，用于展示「粗糙度」对高光的影响。

    这是材质对比最经典的呈现方式：几何完全相同，只改 roughness，
    高光从锐利的镜面反射逐步变成柔和的哑光。
    """
    roughness_values = [0.03, 0.22, 0.52, 0.88]
    x_positions = [-1.95, -0.65, 0.65, 1.95]
    colors = [
        [0.85, 0.73, 0.45],
        [0.72, 0.75, 0.80],
        [0.80, 0.55, 0.42],
        [0.45, 0.60, 0.72],
    ]
    objects = []
    for idx, (rough, x, color) in enumerate(zip(roughness_values, x_positions, colors)):
        objects.append({
            "name": f"roughness_{rough:.2f}".replace(".", "_"),
            "geometry": {"type": "sphere", "radius": 0.45},
            "transform": {"position": [x, 0.0, 0.45]},
            "material": {
                "name": f"metal_rough_{idx}",
                "base_color": color,
                "roughness": rough,
                "metallic": 1.0,
            },
        })
    return objects


def _stack_layout() -> list[dict]:
    """构造堆叠布局，作为 W2 刚体堆叠的构图预演。

    三块逐渐变小的盒子垂直叠放，周围散落几个球。
    这段布局在 W2 会直接变成"落体堆叠"测试的初始状态。
    """
    objects = [
        {
            "name": "stack_1",
            "geometry": {"type": "box", "size": [1.20, 1.20, 1.20]},
            "transform": {"position": [0.0, 0.0, 0.60]},
            "material": {"name": "mat_stack_1", "base_color": [0.70, 0.16, 0.05],
                         "roughness": 0.42, "metallic": 0.0},
        },
        {
            "name": "stack_2",
            "geometry": {"type": "box", "size": [0.80, 0.80, 0.80]},
            "transform": {"position": [0.0, 0.0, 1.60],
                          "orientation_euler_deg": [0.0, 0.0, 12.0]},
            "material": {"name": "mat_stack_2", "base_color": [0.74, 0.62, 0.42],
                         "roughness": 0.55, "metallic": 0.0},
        },
        {
            "name": "stack_3",
            "geometry": {"type": "box", "size": [0.60, 0.60, 0.60]},
            "transform": {"position": [0.02, 0.02, 2.30],
                          "orientation_euler_deg": [0.0, 0.0, -16.0]},
            "material": {"name": "mat_stack_3", "base_color": [0.06, 0.20, 0.52],
                         "roughness": 0.35, "metallic": 0.0},
        },
        {
            "name": "stack_ball_1",
            "geometry": {"type": "sphere", "radius": 0.34},
            "transform": {"position": [1.45, -0.75, 0.34]},
            "material": {"name": "mat_stack_ball_1", "base_color": [0.80, 0.80, 0.83],
                         "roughness": 0.12, "metallic": 1.0},
        },
        {
            "name": "stack_ball_2",
            "geometry": {"type": "sphere", "radius": 0.26},
            "transform": {"position": [-1.60, -1.05, 0.26]},
            "material": {"name": "mat_stack_ball_2", "base_color": [0.86, 0.36, 0.06],
                         "roughness": 0.30, "metallic": 0.0},
        },
        {
            "name": "stack_cyl",
            "geometry": {"type": "cylinder", "radius": 0.30, "height": 0.80},
            "transform": {"position": [-2.05, 0.35, 0.40]},
            "material": {"name": "mat_stack_cyl", "base_color": [0.10, 0.42, 0.30],
                         "roughness": 0.50, "metallic": 0.0},
        },
    ]
    return objects


EXPERIMENTS: list[dict] = [
    {
        "name": "01_baseline",
        "desc": "基线：默认相机 + 三点布光 + 五种基础几何",
        "patches": {},
    },
    {
        "name": "02_camera_low",
        "desc": "低机位仰视：强化物体的体量感，背景压缩",
        "patches": {
            "camera.position": [4.7, -3.9, 0.78],
            "camera.target": [0.0, 0.0, 0.95],
            "camera.fov": 46.0,
        },
    },
    {
        "name": "03_camera_closeup",
        "desc": "长焦特写：小视场角压缩透视，突出金属球细节",
        "patches": {
            "camera.position": [3.70, -3.60, 1.85],
            "camera.target": [0.45, -0.05, 0.80],
            "camera.fov": 38.0,
        },
    },
    {
        "name": "04_light_soft",
        "desc": "大面积柔光：阴影边缘柔化，对比度降低",
        "patches": {
            "lights.0.size": 8.0,
            "lights.0.energy": 430.0,
            "lights.1.size": 6.0,
            "lights.1.energy": 150.0,
            "render.world_strength": 0.70,
        },
    },
    {
        "name": "05_light_hard",
        "desc": "小面积硬光：阴影锐利、明暗对比强",
        "patches": {
            "lights.0.size": 0.35,
            "lights.0.energy": 480.0,
            "lights.1.energy": 20.0,
            "lights.2.energy": 80.0,
            "render.world_strength": 0.05,
        },
    },
    {
        "name": "06_material_roughness",
        "desc": "材质对比：几何完全相同，仅粗糙度 0.03/0.22/0.52/0.88",
        "patches": {
            "objects": _sphere_row(),
            "camera.position": [0.0, -7.4, 2.40],
            "camera.target": [0.0, 0.0, 0.55],
            "camera.fov": 40.0,
        },
    },
    {
        "name": "07_layout_stack",
        "desc": "堆叠布局：三块盒子垂直叠放 + 散落球体，为 W2 刚体测试预演构图",
        "patches": {
            "objects": _stack_layout(),
            "camera.position": [5.4, -6.2, 3.5],
            "camera.target": [0.0, 0.0, 1.25],
            "camera.fov": 43.0,
        },
    },
    {
        "name": "08_hero",
        "desc": "主视觉：高分辨率 + 高采样 + 轻微曝光补偿",
        "patches": {
            "render.resolution": [1600, 900],
            "render.samples": 192,
            "render.exposure": 0.15,
            "camera.position": [4.3, -4.7, 2.55],
            "camera.fov": 39.0,
        },
    },
]


# ---------------------------------------------------------------------------
# 场景生成
# ---------------------------------------------------------------------------

def _load_base() -> dict:
    """读取基础场景 JSON 的原始字典。"""
    with open(BASE_SCENE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_scenes(experiments: list[dict]) -> str:
    """为每项实验生成场景 JSON，并写出批量渲染清单。

    参数
    ----
    experiments : 实验定义列表

    返回
    ----
    str : 批量清单文件路径
    """
    base = _load_base()
    os.makedirs(GENERATED_DIR, exist_ok=True)

    jobs = []
    for exp in experiments:
        patched = apply_patches(base, exp["patches"])
        patched["name"] = exp["name"]

        # 生成后立刻按正式解析器校验一遍：宁可在这里报错，
        # 也不要等到 Blender 跑到一半才失败。
        SceneSpec.from_dict(patched)

        scene_path = os.path.join(GENERATED_DIR, f"{exp['name']}.json")
        with open(scene_path, "w", encoding="utf-8") as fh:
            json.dump(patched, fh, ensure_ascii=False, indent=2)

        jobs.append({
            "scene": scene_path,
            "out": os.path.join(OUT_DIR, f"{exp['name']}.png"),
        })

    manifest_path = os.path.join(GENERATED_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"jobs": jobs}, fh, ensure_ascii=False, indent=2)

    return manifest_path


# ---------------------------------------------------------------------------
# Blender 调用
# ---------------------------------------------------------------------------

def find_blender(explicit: str | None) -> str:
    """定位 blender.exe。

    优先用命令行显式指定的路径，其次用项目默认路径，
    最后尝试从 PATH 中查找。

    异常
    ----
    FileNotFoundError : 三条路径都没找到
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"指定的 Blender 不存在：{explicit}")
        return explicit

    if os.path.isfile(DEFAULT_BLENDER):
        return DEFAULT_BLENDER

    found = shutil.which("blender")
    if found:
        return found

    raise FileNotFoundError(
        "找不到 blender.exe。请用 --blender 指定路径，例如：\n"
        r'  --blender "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"'
    )


def run_blender(blender: str, manifest_path: str) -> None:
    """调用 Blender 批量渲染清单中的全部场景。

    ``--factory-startup`` 跳过用户自定义配置与插件，保证结果可复现。
    这里不捕获子进程输出，让渲染进度直接打印到终端。
    """
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python", RENDER_SCRIPT,
        "--",
        "--manifest", manifest_path,
    ]
    print("执行：", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender 渲染失败，退出码 {completed.returncode}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "phase0")


def main() -> int:
    """解析参数 → 生成场景 → 调用 Blender → 打印结果清单。"""
    global OUT_DIR

    parser = argparse.ArgumentParser(description="Phase 0 参数对比实验")
    parser.add_argument("--blender", default=None, help="blender.exe 路径")
    parser.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    parser.add_argument("--only", default=None, help="只运行指定名称的实验")
    parser.add_argument("--list", action="store_true", help="只列出实验，不渲染")
    args = parser.parse_args()

    OUT_DIR = os.path.abspath(args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in EXPERIMENTS if e["name"] == args.only]
        if not experiments:
            names = ", ".join(e["name"] for e in EXPERIMENTS)
            print(f"错误：没有名为 '{args.only}' 的实验。可选：{names}", file=sys.stderr)
            return 2

    if args.list:
        print(f"{'实验名':<26}{'说明'}")
        print("-" * 90)
        for exp in experiments:
            print(f"{exp['name']:<26}{exp['desc']}")
        return 0

    manifest_path = generate_scenes(experiments)
    print(f"已生成 {len(experiments)} 个场景 -> {GENERATED_DIR}")
    print(f"渲染清单 -> {manifest_path}")
    print(f"输出目录 -> {OUT_DIR}\n")

    start = time.perf_counter()
    run_blender(find_blender(args.blender), manifest_path)
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 90)
    print(f"完成，总耗时 {elapsed:.1f}s")
    for exp in experiments:
        path = os.path.join(OUT_DIR, f"{exp['name']}.png")
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        flag = "OK " if size > 0 else "缺失"
        print(f"  [{flag}] {exp['name']:<26}{exp['desc']}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
