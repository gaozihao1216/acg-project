# -*- coding: utf-8 -*-
"""运行物理仿真并导出位姿序列。

用法（用**系统 Python** 运行）::

    # ① 场景 JSON（自包含快照，可当基准题）
    python Code/scripts/run_sim.py --scene Code/scenes/sim_spheres.json \
        --out results/w2/poses_sim_spheres.json

    # ② 场景资产（只存引用，物体一改就跟着变）
    python Code/scripts/run_sim.py --scene-asset results/w3/stack/output/stack.scene.json \
        --out results/w3/stack/output/stack.poses.json

两条入口的差别：场景资产是**创作期**格式，物品数据永远指向唯一真值；
场景 JSON 是**运行期**快照，自包含。前者不必先生成中间文件——这里直接把
引用解析成实体再喂给引擎。

可选参数::

    --duration SEC   覆盖场景里的仿真时长
    --fps N          覆盖导出帧率
    --render         仿真结束后自动调用 Blender 渲染视频
    --stills 0,20,40 只渲染这几帧为 PNG（用于快速看构图，不渲染整段视频）

导出的位姿序列格式::

    {
      "fps": 24,
      "duration": 3.5,
      "frames": [
        {"frame": 0, "time": 0.0, "poses": [{"name":..., "position":[3], "orientation":[4]}, ...]},
        ...
      ]
    }

每一步仿真与渲染严格解耦：仿真只写 JSON，渲染只读 JSON。这样改渲染参数
不需要重跑物理，改物理参数也不需要重新搭 Blender 场景。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(CODE_ROOT)

if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from src.physics.world import PhysicsWorld                      # noqa: E402
from src.scene.scene_spec import load_scene                     # noqa: E402

RENDER_SCRIPT = os.path.join(CODE_ROOT, "render", "render_anim.py")
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"


def load_spec(scene_path: str | None, scene_asset_path: str | None):
    """从场景 JSON 或**场景资产**得到 ``SceneSpec``。

    两条入口并存是有原因的：场景资产是**创作期**的格式（只存引用，物体一改就
    跟着变），场景 JSON 是**运行期**的快照（自包含，可当基准题）。资产方式
    不必先生成一个中间文件，直接选出引用的实体。
    """
    if scene_asset_path:
        from src.geom.scene_asset import (
            load_scene_asset,
            scene_asset_to_scene_dict,
        )
        from src.scene.scene_spec import SceneSpec

        asset = load_scene_asset(scene_asset_path)
        payload = scene_asset_to_scene_dict(asset)
        print(f"场景资产：{scene_asset_path}")
        print(f"  引用 {len(asset['resolved'])} 个物体资产，已解析为实体")
        return SceneSpec.from_dict(payload)

    return load_scene(scene_path)


def run_simulation(
    scene_path: str | None,
    duration: float | None,
    fps: int | None,
    *,
    scene_asset_path: str | None = None,
) -> dict:
    """加载场景、跑仿真、返回位姿序列。

    参数
    ----
    scene_path      : 场景 JSON 路径（与 ``scene_asset_path`` 二选一）
    duration        : 覆盖总时长；None 表示用场景里的设置
    fps             : 覆盖导出帧率；None 表示用场景里的设置
    scene_asset_path: 场景资产路径（``scene_asset/1``）

    返回
    ----
    dict : 位姿序列
    """
    scene = load_spec(scene_path, scene_asset_path)
    world = PhysicsWorld.from_scene(scene)

    print(f"场景：{scene.name}")
    print(f"刚体：{len(world.bodies)} 个，地面 z={world.simulation.ground_z}")
    print(
        f"仿真：dt={world.simulation.physics_dt:.6f}s，"
        f"子步={world.simulation.substeps}，迭代={world.simulation.iterations}"
    )

    start = time.perf_counter()
    sequence = world.run(duration=duration, fps=fps)
    elapsed = time.perf_counter() - start

    world.check_finite()

    print(
        f"完成：{len(sequence['frames'])} 帧 / {world.step_count} 个物理步，"
        f"耗时 {elapsed:.2f}s"
    )
    print(f"终态最大速度 {world.max_speed():.4f} m/s，"
          f"最大穿透 {world.max_penetration()*1000:.2f} mm")
    return sequence


def write_expanded_scene(scene_asset_path: str, out_path: str) -> str:
    """把场景资产展开成场景 JSON，供 Blender 渲染层使用。

    ``render_anim.py`` 需要一份自包含的场景（相机/灯光/材质/几何全在里面），
    而场景资产是引用式的。所以渲染前必须展开一次。

    这份文件是**派生物**，不是真值：物体资产一改它就过期。
    所以文件名带 ``.expanded``，且每次渲染前重新生成，绝不复用。
    """
    from src.geom.scene_asset import (
        load_scene_asset,
        scene_asset_to_scene_dict,
    )

    payload = scene_asset_to_scene_dict(load_scene_asset(scene_asset_path))
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"展开后的场景（给 Blender）-> {out_path}")
    return out_path


def save_sequence(sequence: dict, output_path: str) -> None:
    """把位姿序列写成 JSON。"""
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(sequence, fh, ensure_ascii=False, indent=1)
    size_kb = os.path.getsize(output_path) / 1024.0
    print(f"位姿序列 -> {output_path}（{size_kb:.1f} KB）")


def find_blender(explicit: str | None) -> str:
    """定位 blender.exe。"""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"指定的 Blender 不存在：{explicit}")
        return explicit
    if os.path.isfile(DEFAULT_BLENDER):
        return DEFAULT_BLENDER
    raise FileNotFoundError("找不到 blender.exe，请用 --blender 指定路径")


def render_video(
    blender: str,
    scene_path: str,
    poses_path: str,
    out_path: str,
    still_frames: list[int] | None,
) -> None:
    """调用 Blender 把位姿序列渲染成视频（或指定帧的静帧）。"""
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python", RENDER_SCRIPT,
        "--",
        "--scene", os.path.abspath(scene_path),
        "--poses", os.path.abspath(poses_path),
        "--out", os.path.abspath(out_path),
    ]
    if still_frames:
        cmd += ["--still-frames", ",".join(str(f) for f in still_frames)]

    print("执行：", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender 渲染失败，退出码 {completed.returncode}")


def main() -> int:
    """解析参数 → 跑仿真 → 保存 → 可选渲染。"""
    parser = argparse.ArgumentParser(description="运行刚体仿真并导出位姿序列")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", help="场景 JSON 路径（自包含快照）")
    group.add_argument(
        "--scene-asset",
        dest="scene_asset",
        help="场景资产路径（scene_asset/1，只存引用；不需先生成中间文件）",
    )
    parser.add_argument("--out", required=True, help="位姿序列输出路径")
    parser.add_argument("--duration", type=float, default=None, help="覆盖仿真时长")
    parser.add_argument("--fps", type=int, default=None, help="覆盖导出帧率")
    parser.add_argument("--render", action="store_true", help="仿真后自动渲染视频")
    parser.add_argument("--video", default=None, help="视频输出路径（配合 --render）")
    parser.add_argument("--stills", default=None, help="只渲染这些帧，如 0,20,40")
    parser.add_argument("--blender", default=None, help="blender.exe 路径")
    args = parser.parse_args()

    sequence = run_simulation(
        args.scene, args.duration, args.fps, scene_asset_path=args.scene_asset
    )
    save_sequence(sequence, args.out)

    if args.render:
        still_frames = None
        if args.stills:
            still_frames = [int(s) for s in args.stills.split(",") if s.strip()]

        # Blender 侧需要自包含的场景；资产是引用式的，先展开一次。
        # 每次都重新生成（绝不复用），避免拿旧快照渲染新物体。
        render_scene = args.scene
        if args.scene_asset:
            stem = os.path.splitext(os.path.abspath(args.out))[0]
            render_scene = write_expanded_scene(
                args.scene_asset, f"{stem}.expanded.scene.json"
            )

        # 默认视频放在位姿文件旁边，而不是硬编码的 results/w2：
        # ① 案例目录自包含，产物不该散到别的周次目录里
        # ② Blender 的静帧目录也跟随视频（<video_dir>/stills），
        #    所以视频路径决定了静帧落在哪，硬编码会让静帧跑到别的案例里
        #    （实测：没给 --video 时，stack 的静帧出现在 results/w2/stills）
        video = args.video or os.path.join(
            os.path.dirname(os.path.abspath(args.out)),
            f"{os.path.splitext(os.path.basename(args.out))[0]}.mp4",
        )
        render_video(
            find_blender(args.blender), render_scene, args.out, video, still_frames
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
