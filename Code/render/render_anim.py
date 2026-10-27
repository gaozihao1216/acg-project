# -*- coding: utf-8 -*-
"""把物理仿真导出的位姿序列渲染成视频（在 Blender 内部运行）。

用法::

    blender --background --factory-startup --python render/render_anim.py -- \
        --scene scenes/sim_spheres.json \
        --poses ../results/w2/poses_sim_spheres.json \
        --out   ../results/w2/sim_spheres.mp4

只看某几帧（快速检查构图，不渲染整段）::

    ... --out x.mp4 --still-frames 0,20,40 --stills-dir ../results/w2/stills

工作方式
--------
1. 用 ``blender_builder`` 按场景 JSON 搭出几何、材质、灯光、相机
2. 把位姿序列写进每个物体的关键帧（逐帧插关键帧，线性插值）
3. 用 Blender 内置的 FFmpeg 编码视频

为什么用关键帧而不是 frame_change 回调
--------------------------------------
回调依赖渲染循环的触发时机，在某些批处理模式下不保证按预期顺序执行；
关键帧由动画系统求值，只要 ``scene.frame_set`` 或动画渲染走到某一帧，
姿态一定是确定的。对于"每帧都有关键帧"这种数据，两者开销都很小，
但关键帧的可复现性更好。

为什么逐帧插关键帧反而更简单
----------------------------
仿真已经把每一帧的姿态都算好了，不需要任何插值建模。逐帧落关键帧相当于
把结果"播放"出来，逻辑上不会引入误差。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from render.blender_builder import build_scene          # noqa: E402
from src.scene.scene_spec import load_scene             # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析 ``--`` 之后的参数。"""
    parser = argparse.ArgumentParser(
        prog="render_anim",
        description="按位姿序列渲染动画（Phase W2）",
    )
    parser.add_argument("--scene", required=True, help="场景 JSON 路径（提供相机/灯光/材质）")
    parser.add_argument("--poses", required=True, help="位姿序列 JSON 路径")
    parser.add_argument("--out", required=True, help="视频输出路径（.mp4）")
    parser.add_argument("--still-frames", default=None,
                        help="只渲染这些帧号（逗号分隔），输出 PNG 而不是视频")
    parser.add_argument("--stills-dir", default=None, help="静帧输出目录")
    return parser.parse_args(argv)


def _load_poses(path: str) -> dict:
    """读取位姿序列 JSON。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _apply_pose_map(pose_map: dict, frame: int) -> None:
    """把一帧的所有姿态写入物体并插入关键帧。

    参数
    ----
    pose_map : ``{物体名: {"position": [...], "orientation": [...]}}``
    frame    : Blender 帧号（从 1 开始）
    """
    for name, pose in pose_map.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            # 位姿序列里出现了场景中没有的物体，跳过而不是中断整段渲染
            print(f"[warn] 位姿序列中的 '{name}' 在场景里不存在，已跳过", flush=True)
            continue

        obj.location = pose["position"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = pose["orientation"]

        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)


def _force_linear_interpolation() -> None:
    """把所有关键帧插值改为线性。

    逐帧都有关键帧时，线性插值与贝塞尔插值在整数帧上的取值完全相同，
    但线性可以避免曲线过冲（四元数分量过冲会导致姿态抖动）。
    """
    for obj in bpy.data.objects:
        action = getattr(getattr(obj, "animation_data", None), "action", None)
        if action is None:
            continue
        for curve in action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"


def _configure_video_output(scene, output_path: str) -> str:
    """配置 FFmpeg 视频输出，返回实际使用的 filepath（不含扩展名）。

    Blender 在不同版本里对"是否自动追加扩展名"处理略有差异，
    因此这里统一传入不含扩展名的路径、让 Blender 自己补 ``.mp4``，
    渲染完成后再校验文件是否真的存在。
    """
    base, _ = os.path.splitext(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)

    # 先删掉上一次的产物：否则 Blender 会写成 ``xxx0001-0049.mp4`` 这种带帧范围的
    # 名字，而 ``_verify_video`` 会把残留的旧文件误认为本次结果（这个坑踩过一次）。
    stale = base + ".mp4"
    if os.path.isfile(stale):
        os.remove(stale)

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = 12
    scene.render.use_file_extension = True
    scene.render.filepath = base
    return base


def _configure_still_output(scene, path: str) -> None:
    """配置单帧 PNG 输出。"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = path


def _verify_video(base_path: str) -> str:
    """确认视频文件确实生成，返回最终路径。

    Blender 在 FFmpeg 输出时会给文件名追加帧范围（如 ``0001-0097``），
    得到 ``xxx0001-0097.mp4`` 这种名字。这里检测到这种情况后**重命名为
    期望的干净名字**，保证外部脚本可以按固定路径引用视频。
    """
    expected = base_path + ".mp4"
    if os.path.isfile(expected):
        return expected

    directory = os.path.dirname(os.path.abspath(base_path)) or "."
    prefix = os.path.basename(base_path)
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith(prefix) and name.endswith(".mp4")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"渲染结束但未找到 mp4 文件（期望 {expected}）。"
            f"请检查 Blender 的 FFmpeg 输出是否被跳过。"
        )

    produced = max(candidates, key=os.path.getmtime)
    if os.path.abspath(produced) != os.path.abspath(expected):
        os.replace(produced, expected)
        return expected
    return produced


def main() -> int:
    """主流程：搭场景 → 写关键帧 → 渲染。"""
    raw_argv = sys.argv
    argv = raw_argv[raw_argv.index("--") + 1:] if "--" in raw_argv else []
    args = _parse_args(argv)

    scene_spec = load_scene(args.scene)
    sequence = _load_poses(args.poses)
    frames = sequence["frames"]
    fps = int(sequence.get("fps", scene_spec.simulation.render_fps))

    print(f"[anim] 场景 '{scene_spec.name}'，{len(frames)} 帧 @ {fps} fps", flush=True)

    # 1) 搭出完整场景（几何、材质、灯光、相机、世界）
    build_scene(scene_spec)

    blender_scene = bpy.context.scene
    blender_scene.render.fps = fps
    blender_scene.frame_start = 1
    blender_scene.frame_end = len(frames)

    # 2) 逐帧写入关键帧（数据帧 i 对应 Blender 帧 i+1）
    for index, frame_data in enumerate(frames):
        pose_map = {p["name"]: p for p in frame_data["poses"]}
        _apply_pose_map(pose_map, index + 1)

    _force_linear_interpolation()

    # 3) 渲染
    if args.still_frames:
        still_dir = args.stills_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "stills"
        )
        indices = [int(s) for s in args.still_frames.split(",") if s.strip()]
        for index in indices:
            blender_frame = index + 1
            if blender_frame < 1 or blender_frame > len(frames):
                print(f"[warn] 帧 {index} 超出范围 [0, {len(frames)-1}]，跳过", flush=True)
                continue
            blender_scene.frame_set(blender_frame)
            out_path = os.path.join(still_dir, f"frame_{index:04d}.png")
            _configure_still_output(blender_scene, out_path)
            bpy.ops.render.render(write_still=True)
            print(f"[anim] 静帧 {index} -> {out_path}", flush=True)
        return 0

    base = _configure_video_output(blender_scene, args.out)
    bpy.ops.render.render(animation=True)

    final = _verify_video(base)
    size_mb = os.path.getsize(final) / (1024.0 * 1024.0)
    print(f"[anim] 视频 -> {final}（{size_mb:.2f} MB）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
