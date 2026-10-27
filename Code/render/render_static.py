# -*- coding: utf-8 -*-
"""静态场景渲染入口（在 Blender 内部运行）。

用法
----
单个场景::

    blender --background --factory-startup --python render/render_static.py -- \
        --scene scenes/demo_basic.json --out ../results/phase0/01_basic.png

批量场景（一次启动 Blender 渲染多张，省去重复启动开销）::

    blender --background --factory-startup --python render/render_static.py -- \
        --manifest scenes/generated/manifest.json

注意
----
- ``--`` 之后的参数才是本脚本的参数，之前的都是 Blender 自己的参数。
- 本脚本必须用 Blender 的 Python 运行（依赖 ``bpy``），不能用系统 python。
- 批量模式下一个进程内连续渲染多个场景，每个场景前都会重置 Blender 场景，
  因此材质名、物体名不会互相污染。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# 让 src 与 render 两个包可被导入
# ---------------------------------------------------------------------------
# Blender 用 --python 执行脚本时，sys.path 里没有本项目目录，
# 因此手动把 Code/ 加进去（render/ 的上一级）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from render.blender_builder import build_scene, render_to_file  # noqa: E402
from src.scene.scene_spec import load_scene                      # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析 ``--`` 之后的命令行参数。

    参数
    ----
    argv : 已剥离 Blender 自身参数后的参数列表

    返回
    ----
    argparse.Namespace : 含 scene / out / manifest 三个可选字段
    """
    parser = argparse.ArgumentParser(
        prog="render_static",
        description="按场景 JSON 渲染单帧静态图（Phase 0）",
    )
    parser.add_argument("--scene", help="单个场景 JSON 路径")
    parser.add_argument("--out", help="单场景模式下的输出 PNG 路径")
    parser.add_argument("--manifest", help="批量任务清单 JSON 路径")
    return parser.parse_args(argv)


def _load_manifest(path: str) -> list[dict]:
    """读取批量任务清单。

    支持两种格式::

        [ {"scene": "...", "out": "..."}, ... ]
        { "jobs": [ {"scene": "...", "out": "..."}, ... ] }

    参数
    ----
    path : 清单文件路径

    异常
    ----
    ValueError : 格式不合法或任务项缺字段
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        jobs = data.get("jobs")
        if jobs is None:
            raise ValueError(f"清单 {path} 是对象但缺少 'jobs' 数组")
    elif isinstance(data, list):
        jobs = data
    else:
        raise ValueError(f"清单 {path} 必须是数组或含 'jobs' 的对象")

    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"清单第 {idx} 项必须是对象")
        for key in ("scene", "out"):
            if key not in job:
                raise ValueError(f"清单第 {idx} 项缺少字段 '{key}'")

    return jobs


def _run_job(scene_path: str, out_path: str) -> float:
    """执行单个渲染任务，返回耗时（秒）。

    参数
    ----
    scene_path : 场景 JSON 路径
    out_path   : 输出 PNG 路径
    """
    start = time.perf_counter()

    scene_spec = load_scene(scene_path)
    build_scene(scene_spec)
    render_to_file(out_path)

    elapsed = time.perf_counter() - start
    print(
        f"[render] {scene_spec.name:<28} -> {out_path}  "
        f"({scene_spec.render.resolution[0]}x{scene_spec.render.resolution[1]}, "
        f"{scene_spec.render.samples} spp, {elapsed:.1f}s)",
        flush=True,
    )
    return elapsed


def main() -> int:
    """脚本主流程：解析参数 → 逐任务渲染 → 打印汇总。

    返回
    ----
    int : 进程退出码，0 表示成功
    """
    raw_argv = sys.argv
    if "--" in raw_argv:
        argv = raw_argv[raw_argv.index("--") + 1:]
    else:
        argv = []

    args = _parse_args(argv)

    # 组装待办任务列表：单场景模式或批量模式
    jobs: list[tuple[str, str]] = []
    if args.manifest:
        for job in _load_manifest(args.manifest):
            jobs.append((job["scene"], job["out"]))
    elif args.scene and args.out:
        jobs.append((args.scene, args.out))
    else:
        print("错误：需要 --manifest，或者同时提供 --scene 与 --out", file=sys.stderr)
        return 2

    total_start = time.perf_counter()
    total = 0.0
    for scene_path, out_path in jobs:
        try:
            total += _run_job(scene_path, out_path)
        except Exception as exc:  # noqa: BLE001
            # 批量模式下单个任务失败不应中断其余任务，
            # 但必须把错误打印清楚并让最终退出码非零。
            print(f"[error] 渲染失败 {scene_path}: {exc}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            return 1

    wall = time.perf_counter() - total_start
    print(
        f"[render] 完成 {len(jobs)} 张，累计渲染 {total:.1f}s，"
        f"进程总耗时 {wall:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
