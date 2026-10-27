# -*- coding: utf-8 -*-
"""把若干已存在的物体案例装配成一个场景资产（只存引用）。

用法
----
::

    python Code/scripts/assemble_scene.py --root results/w3 --name stack \
        --place shell=0 --place prism=1.15

    # 顺带导出一份引擎能直接加载的场景 JSON
    python Code/scripts/assemble_scene.py --root results/w3 --name stack \
        --place shell=0 --place prism=1.15 --emit-engine results/w3/stack/output/stack.scene.json

产物
----
::

    results/w3/stack/
    ├── input/    request.json   这次的装配配方（可重放）
    └── output/   stack.scene.json     场景资产：名称 + 引用 + 环境配置

**场景资产里没有物体级数值。** 质量、惯量、凸包一律沿着 ``asset`` 引用去读
``<name>/output/<name>.asset.json``——那是唯一真值。原因见
``src/geom/scene_asset.py`` 的模块文档：复制会产生三个具体问题
（占空间、会分叉、没法定位错误）。

摆位约定
--------
``--place name=z`` 把物体放在 ``(0, 0, z)``；``--place name=x,y,z`` 给完整坐标。
``z`` 是**底面离地高度**（``0`` = 贴地），不是质心高度。

为什么强调这一点：如果按质心高度理解，一个 0.6 m 高的物体给 ``z=0``
就会有一半埋在土里，而引擎的初始穿透校验会直接拒绝它。实测报错是
``地面 <-> 'shell'：初始穿透 480.000 mm``——这正是那道校验存在的意义。
用"底面离地高度"就没有这个歧义：``0`` 永远等于"放上去"。

转换在写场景时一次完成（加上各物体的 ``ground_z_offset``），
所以场景资产里存的仍是**质心位置**，与引擎的语义一致，没有隐藏约定。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.asset import case_paths, load_asset
from src.geom.scene_asset import (
    build_scene_asset,
    format_scene_checks,
    load_scene_asset,
    save_scene_asset,
    scene_asset_to_objects,
    scene_asset_to_scene_dict,
)

_LINE = "=" * 78
_SUB = "-" * 78


def _parse_place(text: str) -> tuple[str, list[float]]:
    """解析 ``name=z`` 或 ``name=x,y,z``。"""
    if "=" not in text:
        raise ValueError(f"--place 需要 name=z 或 name=x,y,z，收到 {text!r}")
    name, _, value = text.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"--place 缺少物体名：{text!r}")

    parts = [chunk for chunk in value.split(",") if chunk.strip()]
    try:
        numbers = [float(chunk) for chunk in parts]
    except ValueError as exc:
        raise ValueError(f"--place 坐标不是数字：{text!r}（{exc}）") from exc

    if len(numbers) == 1:
        return name, [0.0, 0.0, numbers[0]]
    if len(numbers) == 3:
        return name, numbers
    raise ValueError(f"--place 坐标要么 1 个数要么 3 个数，收到 {value!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把物体案例装配成场景资产（只存引用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", required=True, help="案例根目录（与 analyze_object 的 --out-dir 相同）")
    parser.add_argument("--name", required=True, help="场景名")
    parser.add_argument(
        "--place",
        action="append",
        default=[],
        metavar="NAME=X,Y,Z",
        help="摆放：name=z 或 name=x,y,z。可重复；不写则落在原点",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="仿真时长（秒）")
    parser.add_argument("--emit-engine", metavar="FILE", help="额外导出引擎可直接加载的场景 JSON")
    parser.add_argument("--no-verify", action="store_true", help="跳过装配后的加载自检")
    args = parser.parse_args(argv)

    if not args.place:
        parser.error("至少给一个 --place")

    placements: list[tuple[str, list[float]]] = []
    seen = set()
    for text in args.place:
        name, position = _parse_place(text)
        if name in seen:
            raise ValueError(f"物体 {name!r} 重复出现")
        seen.add(name)
        placements.append((name, position))

    print(_LINE)
    print(f"装配场景：{args.name}")
    print(_LINE)

    objects = []
    for name, placement in placements:
        layout = case_paths(args.root, name)
        if not os.path.isfile(layout["asset_json"]):
            raise FileNotFoundError(
                f"找不到 {name} 的物体资产：{layout['asset_json']}\n"
                f"  先运行：python Code/scripts/analyze_object.py "
                f"<网格> --mass <M> --out-dir {args.root}"
            )

        # 用户给的是"底面离地高度"，场景里要存"质心位置"。
        # 差值就是该物体自己的 ground_z_offset——读资产拿，不靠猜。
        asset = load_asset(layout["asset_json"])
        ground_offset = float(asset["engine"]["ground_z_offset"])
        position = [placement[0], placement[1], placement[2] + ground_offset]

        objects.append({
            "name": name,
            "asset": layout["asset_json"],
            "transform": {"position": position},
        })
        if abs(placement[2]) > 1e-9 and abs(ground_offset) > 1e-9:
            print(
                f"  {name}: 底面离地 {placement[2]:.3f} m "
                f"⇒ 质心 {position[2]:.3f} m（含贴地高 {ground_offset:.3f}）"
            )

    scene_layout = case_paths(args.root, args.name)
    os.makedirs(scene_layout["output"], exist_ok=True)
    os.makedirs(scene_layout["input"], exist_ok=True)

    scene = build_scene_asset(
        name=args.name,
        objects=objects,
        path_base=scene_layout["output"],
    )
    # 只覆盖时长，其余求解器参数保留 build_scene_asset 的完整默认值。
    # （之前把 simulation 传成 {"duration": ...}，等于用一份残缺配置替换了
    #   整套参数，物理子步、阻尼、TOI 全都被悄悄清空。）
    scene["simulation"]["duration"] = args.duration

    scene_path = os.path.join(scene_layout["output"], f"{args.name}.scene.json")
    save_scene_asset(scene_path, scene)

    # --- 装配配方：与物体案例的 input/request.json 同一个约定 ---
    #
    # command 必须完整：漏了 --emit-engine，重放就不会产出引擎快照；
    # 漏了 --duration，改过时长就重放不回同一份场景。这两种漏记都会让
    # reproduce.py 报"产物不符"，而原因在记录侧不在重放侧。
    replay = ["assemble_scene.py", "--root", args.root, "--name", args.name]
    replay += [
        f"--place={name}={','.join(repr(v) for v in position)}"
        for name, position in placements
    ]
    if args.duration != 2.0:
        replay += ["--duration", repr(args.duration)]
    if args.emit_engine:
        replay += ["--emit-engine", args.emit_engine]

    request = {
        "schema": "scene_request/1",
        "name": args.name,
        "command": replay,
        "objects": [
            {"name": name, "position": position} for name, position in placements
        ],
        "duration": args.duration,
        "note": "场景资产只存引用；物体级数值得去 <name>/output/<name>.asset.json 读。",
    }
    with open(scene_layout["request"], "w", encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print()
    print("场景自检")
    print(_SUB)
    for line in format_scene_checks(scene):
        print(f"  {line}")

    print()
    print("引用清单")
    print(_SUB)
    for entry in scene["objects"]:
        print(f"  {entry['name']:<10} → {entry['asset']}")
        print(f"  {'':<10}   position={entry['transform']['position']}")

    if args.emit_engine:
        _emit_engine(scene_path, args.emit_engine, verify=not args.no_verify)

    print()
    print(f"案例目录：{scene_layout['case']}")
    print(_SUB)
    print(f"  {args.name}/")
    print(f"    input/")
    request_size = os.path.getsize(scene_layout["request"]) / 1024
    print(f"      {request_size:>8.1f} KB  request.json（装配配方）")
    print(f"    output/")
    scene_size = os.path.getsize(scene_path) / 1024
    print(f"      {scene_size:>8.1f} KB  {args.name}.scene.json   (场景资产，只存引用)")
    if args.emit_engine and os.path.isfile(args.emit_engine):
        engine_size = os.path.getsize(args.emit_engine) / 1024
        print(
            f"      {engine_size:>8.1f} KB  {os.path.basename(args.emit_engine)}"
            "   (展开后的快照，不是真值)"
        )

    print()
    print(_LINE)
    return 0


def _emit_engine(scene_path: str, out_path: str, *, verify: bool) -> None:
    """把场景资产摊平成引擎能加载的场景 JSON，并就地验证一次。

    摊平只发生在**导出**这一步：场景资产本身始终只存引用。
    这样"唯一真值"和"运行时快照"各司其职，前者不会因为后者而退化。
    """
    print()
    print("导出引擎场景")
    print(_SUB)

    scene = load_scene_asset(scene_path)
    # 展开逻辑只有一处（scene_asset_to_scene_dict），run_sim.py 用同一个，
    # 所以"手工导出"和"直接喂仿真"不可能展开出两种结果。
    payload = scene_asset_to_scene_dict(scene)
    objects = payload["objects"]

    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"  {len(objects)} 个物体 → {out_path}")
    print(f"  大小 {os.path.getsize(out_path) / 1024:.1f} KB"
          "（这一份是把引用展开后的快照，不要当作真值）")

    if not verify:
        return

    from src.physics.world import PhysicsWorld
    from src.scene.scene_spec import load_scene

    world = PhysicsWorld.from_scene(load_scene(out_path))
    expected = {
        item["name"]: item["asset"]["principal"]["moments"]
        for item in scene["resolved"]
    }
    print()
    print("加载自检")
    print(_SUB)
    print(f"  物体数          {len(world.bodies)}")
    worst = 0.0
    for body in world.bodies:
        moments = expected.get(body.name)
        if moments:
            for i in range(3):
                reference = moments[i]
                if reference:
                    worst = max(
                        worst, abs(body.inertia_local[i] / reference - 1.0)
                    )
    print(f"  主惯量 vs 物体资产  最大相对差 = {worst:.2e}")

    for _ in range(int(240 * scene["simulation"]["duration"])):
        world.step()
    print(f"  仿真 {scene['simulation']['duration']:.1f} s 后最大速度 = "
          f"{max(body.speed() for body in world.bodies):.5f} m/s")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
