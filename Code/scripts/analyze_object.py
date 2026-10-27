# -*- coding: utf-8 -*-
"""一条命令：从网格/点云文件算出质量属性，并产出自包含的案例目录。

你放进去一个不规则物体的信息（碰撞几何、质量），这里把它变成引擎真正需要的
数字，外加一份可审计、可带走、可进版本库的案例目录。

用法
----
::

    # 从网格算（最常用）
    python Code/scripts/analyze_object.py bunny.obj --mass 2.5

    # 用密度代替质量
    python Code/scripts/analyze_object.py part.stl --density 1200

    # 只有点云（任意分布，自动判断该不该加截断修正）
    python Code/scripts/analyze_object.py --points cloud.csv --mass 3.0

    # 出资产三件套（案例目录）
    python Code/scripts/analyze_object.py bunny.obj --mass 2.5 --out-dir results/out

输出：案例目录
--------------
``--out-dir ROOT`` 会建立**每个物体一个目录**、内部再分输入与输出：

::

    ROOT/
    └── bunny/
        ├── input/     bunny.obj（源文件副本：案例可独立带走）
        └── output/    bunny.asset.json / .blend / .preview.png

按物体分而不是按文件类型分，是因为"删掉一个物体"应当只删一个目录；
而 ``input`` / ``output`` 两层把"进来的"和"算出去的"隔开。

三件套的分工（见 ``docs/13_输出资产体系.md``）：

1. ``.asset.json``：**数据**。质量、质心、完整惯量张量、主惯量、主轴、
   凸包、引擎交接三件套、8 条自检、源文件 sha256。机器读、可 diff。
2. ``.blend``：**资产**。渲染网格 + 独立碰撞线框 + 质心/主轴标记，统一在 body 系。
3. ``.preview.png``：**证据**。QA 板，一眼核对数字与形状是否对得上。

点云只描述质量分布，不描述表面
------------------------------
点云没有表面，也就没有法向与穿透深度，所以**不能用来做碰撞**。
碰撞几何必须另外给网格。两者的分工是明确的：点云 → 惯量；网格 → 碰撞代理。

为什么报告里要同时给"完整张量"和"主惯量"
--------------------------------------
引擎内部只用对角值，但**完整张量是物理事实**，主惯量是它在主轴系下的表示。
两者都打印出来，读者才能判断"非对角项到底有多大"——如果一个物体的惯量积
只有对角元的 1e-3，那用对角近似是可接受的；如果是 30%，就绝对不能近似。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_CODE_ROOT)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.asset import (
    asset_to_object_spec,
    build_asset,
    build_request,
    case_paths,
    format_checks,
    save_asset,
    save_request,
    stage_input,
)
from src.geom.geometry import ConvexHullGeometry
from src.geom.massprops import MassProperties, from_mesh, from_points
from src.geom.meshio import load_mesh, load_points

_LINE = "=" * 78
_SUB = "-" * 78

#: 与 ``run_sim.py`` 保持同一个默认路径（两份脚本各自独立，避免互相导入）
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------

def _fmt(value: float, digits: int = 6) -> str:
    """紧凑的数字格式：大数/极小数走科学计数法，其余用 ``%g``。"""
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1e-12 or magnitude >= 1e7:
        return f"{value:.{digits + 3}e}"
    return f"{value:.{digits}g}"


def _print_tensor(tensor) -> None:
    for row in tensor:
        print("      [" + "  ".join(f"{value:>15.9g}" for value in row) + "]")


def _products(tensor) -> tuple[float, float, float]:
    """惯量积 ``(Ixy, Ixz, Iyz)``。

    注意惯量张量的非对角元在物理上带负号定义（``Ixy = -∫xy dm``），
    这里报的是**张量的元**本身，与引擎直接用的一致。
    """
    return tensor[0][1], tensor[0][2], tensor[1][2]


def _specificity(tensor) -> float:
    """惯量积相对对角元的比例，用来判断"能不能只留对角"。"""
    scale = max(abs(tensor[i][i]) for i in range(3))
    if scale <= 0.0:
        return 0.0
    return max(abs(value) for value in _products(tensor)) / scale


def report_mass_properties(props: MassProperties, mass: float) -> None:
    """质量属性报告。

    ``质心`` 与 ``惯量张量`` 都在**输入坐标系**里——这样它们才能和 CAD、
    和教科书公式直接对照。转换到 body 系（质心原点 + 主轴）是资产里的
    ``frames`` 负责的事，不在这里混进来。
    """
    print("质量属性")
    print(_SUB)

    if props.volume is not None:
        print(f"  体积            {_fmt(props.volume)} m³")
    print(f"  质量            {_fmt(mass)} kg")
    if props.volume:
        print(f"  密度            {_fmt(mass / props.volume)} kg/m³")
    if props.weighted:
        print("  密度分布        非均匀（逐点质量）")

    center = props.center_of_mass
    print(
        f"  质心            ({_fmt(center[0])}, {_fmt(center[1])}, {_fmt(center[2])})"
        "  ← 相对输入坐标系原点"
    )

    print("  惯量张量（绕质心，输入坐标系）")
    print()
    _print_tensor(props.inertia)
    print()

    ixy, ixz, iyz = _products(props.inertia)
    print(f"  惯量积          Ixy={_fmt(ixy)}  Ixz={_fmt(ixz)}  Iyz={_fmt(iyz)}")

    specificity = _specificity(props.inertia)
    verdict = (
        "可用对角近似"
        if specificity < 1e-3
        else "**非对角项显著，必须用完整张量**"
    )
    print(f"  非对角占比      {specificity * 100:.3f}%  → {verdict}")

    moments, _ = props.principal()
    quaternion = props.principal_quaternion()
    print(
        "  主惯量          ("
        + ", ".join(_fmt(value) for value in moments)
        + ")"
    )
    print(
        "  主轴四元数      ("
        + ", ".join(_fmt(value) for value in quaternion)
        + ")"
    )
    print()
    print("  引擎用法：inertia_local = 主惯量，orientation = R_场景 ⊗ 主轴四元数")
    print("            （RigidBody.from_object_spec 会自动做这个复合）")
    if props.correction_applied not in ("none", None):
        print(f"  点云修正        {props.correction_applied!r}")


def report_mesh_health(mesh) -> None:
    """网格健康：闭合、缠绕、欧拉特征——散度定理的前提。"""
    print("输入网格")
    print(_SUB)
    print(f"  顶点 / 面       {len(mesh.vertices)} / {len(mesh.faces)}")
    print(f"  闭合            {mesh.is_closed}")
    print(f"  缠绕一致        {mesh.is_consistently_oriented}")
    print(f"  欧拉特征        χ = {mesh.euler_characteristic}（闭合凸体应为 2）")
    if mesh.duplicate_faces:
        print(f"  ⚠ 重复面        {len(mesh.duplicate_faces)} 个")


def build_hull(vertices, max_faces: int):
    """求凸包；失败时返回 ``(None, 原因)`` 而不是抛异常。

    碰撞代理失败不该让整份质量属性报告作废——惯量仍然是有用的。
    """
    try:
        return ConvexHullGeometry(vertices, max_faces=max_faces), None
    except ValueError as exc:
        return None, str(exc)


def report_hull(
    geometry: ConvexHullGeometry | None,
    mesh_volume: float | None,
    error: str | None,
) -> None:
    """碰撞代理报告：面/棱数与 SAT 轴数决定能不能实时跑。"""
    print()
    print("碰撞代理（凸包）")
    print(_SUB)

    if geometry is None:
        print(f"  未生成：{error}")
        print("  提示：碰撞代理应当用粗网格；网格过密时会超过面数上限。")
        return

    faces = len(geometry.faces)
    edges = len(geometry.edges)
    # 广义 SAT 的候选轴：两个体的面法向各一份 + 棱叉积 |E_A|·|E_B|
    sat_axes = 2 * faces + edges ** 2

    print(f"  面 / 棱         {faces} / {edges}")
    volume = geometry.volume()
    if mesh_volume:
        ratio = volume / mesh_volume
        print(
            f"  凸包体积        {_fmt(volume)} m³（占本体 {ratio * 100:.1f}%）"
            + ("" if abs(ratio - 1.0) < 1e-6 else "  ← 有凹陷，外近似")
        )
    print(f"  SAT 候选轴      ≤ {faces} + {faces} + {edges}² = {sat_axes}")
    if sat_axes <= 256:
        print("  规模合适，可实时")
    else:
        print("  ⚠ 轴数偏多，碰撞检测会成为瓶颈；建议降面")

    lowest = min(v[2] for v in geometry.vertices)
    print(
        f"  最低点（相对质心） {_fmt(lowest)} m"
        f"  → 放在地面上时 position.z = {_fmt(-lowest)}"
    )


# ---------------------------------------------------------------------------
# 外部工具
# ---------------------------------------------------------------------------

def find_blender(explicit: str | None) -> str:
    """定位 blender.exe。"""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"指定的 Blender 不存在：{explicit}")
        return explicit
    if os.path.isfile(DEFAULT_BLENDER):
        return DEFAULT_BLENDER
    raise FileNotFoundError(
        "找不到 blender.exe。用 --blender 指定路径，或加 --no-render 只出 JSON。"
    )


def render_asset_files(
    asset_path: str,
    out_dir: str,
    name: str,
    *,
    blender: str | None,
    resolution: str,
    samples: int,
    glb: bool = False,
) -> tuple[str, str, str | None]:
    """调用 Blender 产出 ``.blend`` 与 QA 预览图（可选再出一份 GLB）。"""
    executable = find_blender(blender)
    blend_path = os.path.join(out_dir, f"{name}.blend")
    png_path = os.path.join(out_dir, f"{name}.preview.png")
    glb_path = os.path.join(out_dir, f"{name}.glb") if glb else None

    command = [
        executable,
        "--background",
        "--factory-startup",
        "--python",
        os.path.join(_CODE_ROOT, "render", "render_asset.py"),
        "--",
        "--asset", os.path.abspath(asset_path),
        "--blend", os.path.abspath(blend_path),
        "--png", os.path.abspath(png_path),
        "--resolution", resolution,
        "--samples", str(samples),
    ]
    if glb_path:
        command += ["--glb", os.path.abspath(glb_path)]

    completed = subprocess.run(command, cwd=_PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender 生成资产失败，退出码 {completed.returncode}")
    return blend_path, png_path, glb_path


def report_outputs(root: str, name: str, paths: list[tuple[str, str]]) -> None:
    """打印案例目录树与产物清单（含字节数，方便确认不是空文件）。"""
    layout = case_paths(root, name)
    print()
    print(f"案例目录：{layout['case']}")
    print(_SUB)
    print(f"  {name}/")
    if os.path.isdir(layout["input"]):
        entries = sorted(os.listdir(layout["input"]))
        if entries:
            print("    input/")
            for entry in entries:
                path = os.path.join(layout["input"], entry)
                size = os.path.getsize(path)
                label = "（输入卡：参数与随机种子）" if entry == "request.json" else ""
                print(f"      {size / 1024:>8.1f} KB  {entry}{label}")
    print("    output/")
    for label, path in paths:
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"      {size / 1024:>8.1f} KB  {os.path.basename(path)}   ({label})")


# ---------------------------------------------------------------------------
# 场景导出
# ---------------------------------------------------------------------------

def emit_scene(path: str, asset: dict) -> None:
    """从**资产**派生一个可直接加载的场景 JSON。

    物体的几何、质量、初始高度全部取自 :func:`asset_to_object_spec`，
    所以资产改了、场景自然跟着变，不会出现两份数字打架。

    为什么留 2 cm 余量：引擎有一道初始穿透校验，物体一开始就嵌进地面会被
    直接拒绝（这是它存在的意义）。贴地高度加一点余量，仿真开始后会自然落下。
    """
    obj = asset_to_object_spec(asset)
    obj["transform"]["position"][2] += 0.02
    # 材质与物理参数取自资产，不在这里另写一份——否则资产的 render 段
    # 就成了摆设，而且调用方一多就会各自写死不同的颜色。
    obj["physics"] = {"restitution": 0.15, "friction": 0.75}
    # 这两个字段是给人核对用的，引擎不认；写进场景会被当成未知键
    obj.pop("_inertia_local_expected", None)
    obj.pop("_orientation_offset_expected", None)

    scene = {
        "name": asset["name"],
        "camera": {
            "position": [2.6, -3.6, 2.2],
            "target": [0.0, 0.0, 0.5],
            "fov": 45.0,
        },
        "ground": {
            "enabled": True,
            "z": 0.0,
            "friction": 0.8,
            "restitution": 0.05,
            "color": [0.22, 0.23, 0.25],
        },
        "objects": [obj],
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
            "ground_friction": 0.8,
            "ground_restitution": 0.05,
            "linear_damping": 0.05,
            "angular_damping": 1.0,
            "use_toi_splitting": True,
            "max_events_per_substep": 8,
            "broadphase": "spatial_hash",
            "broadphase_cell_size": 0.0,
            "render_fps": 24,
            "duration": 1.5,
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def emit_properties(path: str, props: MassProperties, mass: float) -> None:
    """写出纯数值的 JSON（旧格式，保留兼容；新东西一律走资产）。"""
    moments, _ = props.principal()
    payload = {
        "mass": mass,
        "volume": props.volume,
        "density": (mass / props.volume) if props.volume else None,
        "center_of_mass": list(props.center_of_mass),
        "inertia": [list(row) for row in props.inertia],
        "principal_moments": list(moments),
        "principal_quaternion": list(props.principal_quaternion()),
        "source": props.source,
        "correction_applied": props.correction_applied,
        "weighted": props.weighted,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从网格/点云算质量属性，并产出案例目录（JSON + .blend + 预览图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="?", help="网格文件（.obj / .stl）")
    parser.add_argument("--points", metavar="FILE", help="文本点云（每行 x y z）")
    parser.add_argument("--mass", type=float, help="总质量（kg）")
    parser.add_argument("--density", type=float, help="材料密度（kg/m³），与 --mass 二选一")
    parser.add_argument("--name", help="资产名（默认取文件名）")
    parser.add_argument("--max-faces", type=int, default=64, help="凸包最大面数（默认 64）")
    parser.add_argument(
        "--out-dir",
        metavar="ROOT",
        help="案例根目录。会创建 ROOT/<name>/input（源文件副本）与 "
             "ROOT/<name>/output（资产三件套）",
    )
    parser.add_argument(
        "--no-copy-input",
        action="store_true",
        help="不把源文件复制进 ROOT/<name>/input（案例将不自包含）",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="只出资产 JSON，不调用 Blender（没有 Blender 或只想取数时用）",
    )
    parser.add_argument("--blender", default=None, help="blender.exe 路径")
    parser.add_argument("--resolution", default="1000x750", help="预览图分辨率（默认 1000x750）")
    parser.add_argument("--samples", type=int, default=48, help="预览图采样数（默认 48）")
    parser.add_argument(
        "--glb",
        action="store_true",
        help="额外导出 .glb（可在 3D 查看器里自由旋转；不替代预览图）",
    )
    parser.add_argument("--emit", metavar="FILE", help="写出可直接加载的场景 JSON")
    parser.add_argument("--json", metavar="FILE", help="写出纯数值 JSON（旧格式）")

    args = parser.parse_args(argv)

    if not args.input and not args.points:
        parser.error("必须给出网格文件，或者用 --points 指定点云")
    if args.input and args.points:
        parser.error("input 与 --points 只能给一个")
    if (args.mass is None) == (args.density is None):
        parser.error("--mass 与 --density 必须且只能给一个")

    name = args.name or os.path.splitext(
        os.path.basename(args.input or args.points)
    )[0]

    try:
        if args.points:
            return _analyze_points(args, name)
        return _analyze_mesh(args, name)
    except (ValueError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


def _resolve_mass(mass, density, volume) -> float:
    if mass is not None:
        return float(mass)
    if volume is None or volume <= 0.0:
        raise ValueError("体积无效，无法用密度反推质量")
    return float(density) * volume


def _output_dir(args, name: str) -> str | None:
    """资产 JSON 将写入的目录；没有 ``--out-dir`` 时返回 ``None``。

    返回 ``None`` 意味着路径保留绝对形式——那是"只导出场景、不建案例目录"
    的临时用法，不需要可搬移性。
    """
    if not args.out_dir:
        return None
    return case_paths(args.out_dir, name)["output"]


def _replay_command(args) -> list[str]:
    """从解析后的参数重建一条**可原样重放**的命令。

    为什么不直接用 ``sys.argv``：测试里会以 ``main([...])`` 的形式调用，
    那时 ``sys.argv`` 是 pytest 自己的参数。从解析结果重建更可控，
    也顺便把默认值省掉，让输入卡只记录**真正影响结果**的选项。
    """
    command = ["analyze_object.py"]
    if args.input:
        command.append(args.input)
    if args.points:
        command += ["--points", args.points]
    if args.mass is not None:
        command += ["--mass", repr(args.mass)]
    if args.density is not None:
        command += ["--density", repr(args.density)]
    if args.name:
        command += ["--name", args.name]
    if args.max_faces != 64:
        command += ["--max-faces", str(args.max_faces)]
    if args.out_dir:
        command += ["--out-dir", args.out_dir]
    return command


def _write_request(args, name: str, asset: dict, points_spec=None) -> str | None:
    """写出 ``input/request.json``（输入卡）。

    它记的是**产物反推不回来的输入自由度**：质量还是密度、
    随机点云的种子、凸包面数上限、要求的是哪种修正式。
    ``output/`` 说"算出了什么"，``input/`` 说"我要什么"。
    """
    if not args.out_dir:
        return None
    layout = case_paths(args.out_dir, name)
    request = build_request(
        name=name,
        command=_replay_command(args),
        mass=args.mass,
        density=args.density,
        source=(asset.get("source") or {}).get("path"),
        points=points_spec,
        max_faces=args.max_faces,
        correction="auto",
    )
    save_request(layout["request"], request)
    return layout["request"]


def _stage_source(args, name: str, source_path: str) -> str:
    """把源文件收进 ``ROOT/<name>/input``，返回应当记入资产的路径。

    为什么要改记副本路径：案例目录要能**独立带走**。``source.path`` 写成
    用户桌面上的原文件，换台机器就指不到了；写成 ``input/prism.obj``，
    它和资产在同一个文件夹里，``sha256`` 还能校验。
    """
    if not args.out_dir or args.no_copy_input:
        return source_path
    layout = case_paths(args.out_dir, name)
    try:
        return stage_input(source_path, layout["input"])
    except OSError as exc:
        print(f"  提示：源文件未能复制进 input/（{exc}），案例将不自包含")
        return source_path


def _analyze_mesh(args, name: str) -> int:
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到文件：{args.input}")

    mesh = load_mesh(args.input)

    print(_LINE)
    print(f"物体分析：{args.input}")
    print(_LINE)
    print()
    report_mesh_health(mesh)
    print()

    volume = mesh.signed_volume()
    mass = _resolve_mass(args.mass, args.density, abs(volume))

    try:
        props = from_mesh(mesh.vertices, mesh.faces, mass=mass)
    except ValueError as exc:
        print(f"  质量属性计算失败：{exc}", file=sys.stderr)
        print("  提示：网格可能未闭合或缠绕不一致，散度定理会给出无意义的数字。")
        return 1

    report_mass_properties(props, mass)
    print()

    geometry, error = build_hull(mesh.vertices, args.max_faces)
    report_hull(geometry, props.volume, error)

    if args.json:
        emit_properties(args.json, props, mass)
        print()
        print(f"数值已写入 {args.json}")

    # --- 资产（输出的唯一真值） ---
    asset = None
    if args.out_dir or args.emit:
        asset = build_asset(
            name=name,
            mesh=mesh,
            mass=mass,
            source_path=_stage_source(args, name, args.input),
            max_faces=args.max_faces,
            path_base=_output_dir(args, name),
        )

    if args.out_dir:
        if _publish_asset(args, name, asset) != 0:
            return 1

    if args.emit:
        if asset is None or not (asset.get("collision") or {}).get("points"):
            print()
            print("未写出场景：没有碰撞代理，见上面的提示。", file=sys.stderr)
            return 1
        emit_scene(args.emit, asset)
        print()
        print(f"场景已写入 {args.emit}（由资产派生，position 已设为质心贴地）")

    print()
    print(_LINE)
    return 0


def _publish_asset(args, name: str, asset: dict, points_spec=None) -> int:
    """写出 ``.asset.json``，并按需调用 Blender 生成 ``.blend`` 与预览图。

    目录约定见 :func:`src.geom.asset.case_paths`。
    """
    layout = case_paths(args.out_dir, name)
    os.makedirs(layout["output"], exist_ok=True)
    save_asset(layout["asset_json"], asset)
    _write_request(args, name, asset, points_spec)

    print()
    print("资产自检")
    print(_SUB)
    for line in format_checks(asset):
        print(f"  {line}")

    paths: list[tuple[str, str]] = [("资产 JSON", layout["asset_json"])]

    can_render = bool((asset.get("collision") or {}).get("points")) or bool(
        ((asset.get("render") or {}).get("mesh") or {}).get("path")
    )

    if args.no_render:
        print()
        print("  （--no-render：未生成 .blend 与预览图）")
    elif not can_render:
        print()
        print("  （没有可渲染的网格：跳过 .blend 与预览图）")
    else:
        print()
        print("正在调用 Blender 生成 .blend 与预览图…")
        try:
            blend_path, png_path, glb_path = render_asset_files(
                layout["asset_json"],
                layout["output"],
                name,
                blender=args.blender,
                resolution=args.resolution,
                samples=args.samples,
                glb=args.glb,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  失败：{exc}", file=sys.stderr)
            report_outputs(args.out_dir, name, paths)
            return 1
        paths.extend([(".blend 资产", blend_path), ("预览图", png_path)])
        if glb_path:
            # 可选产物：不进默认流程。它是给人转着看的，不是求解器的输入。
            paths.append(("GLB 可自由旋转", glb_path))

    report_outputs(args.out_dir, name, paths)
    return 0


def _load_or_generate_points(args):
    """读取点云，或按 ``random:N:SEED:SX,SY,SZ`` 现场生成。

    为什么要支持"现场生成"
    ----------------------
    随机点云的**种子**是产物里反推不出来的信息。把点云存成 CSV 也能复现，
    但 30 万行文本既不适合进版本库，也看不出来"它是怎么来的"。
    写成 ``random:8000:7:1,1,1`` 这一行，参数本身就是数据，
    输入卡记下来就能原样重放。

    返回 ``(points, spec)``；从文件读取时 ``spec`` 为 ``None``。
    """
    text = str(args.points)
    if not text.lower().startswith("random:"):
        return load_points(text), None

    parts = text.split(":")
    if len(parts) != 4:
        raise ValueError(
            "随机点云的写法是 random:<点数>:<种子>:<sx>,<sy>,<sz>，"
            f"收到 {text!r}"
        )
    try:
        count = int(parts[1])
        seed = int(parts[2])
        bounds = tuple(float(value) for value in parts[3].split(","))
    except ValueError as exc:
        raise ValueError(f"随机点云参数解析失败：{exc}") from exc
    if len(bounds) != 3 or any(value <= 0.0 for value in bounds):
        raise ValueError(f"包围盒尺寸必须三个正数，收到 {parts[3]!r}")

    import random as _random

    from src.geom.meshutil import random_points_in_box

    rng = _random.Random(seed)
    points = random_points_in_box(bounds, count, rng)
    spec = {
        "mode": "random",
        "count": count,
        "seed": seed,
        "bounds": list(bounds),
        "generator": "uniform_in_box",
    }
    return points, spec


def _analyze_points(args, name: str) -> int:
    # ``random:...`` 是现场生成的配方，不是文件，不能拿去检查存在性
    if not str(args.points).lower().startswith("random:") and not os.path.exists(
        args.points
    ):
        raise FileNotFoundError(f"找不到文件：{args.points}")

    points, points_spec = _load_or_generate_points(args)

    print(_LINE)
    print(f"点云分析：{args.points}")
    print(_LINE)
    print()
    print("输入")
    print(_SUB)
    print(f"  点数            {len(points)}")
    if points_spec:
        print(f"  生成方式        random（seed={points_spec['seed']}，"
              f"盒={points_spec['bounds']}）")
        print("                  ⇒ 种子已记入 input/request.json，可原样重放")

    if args.mass is None:
        raise ValueError(
            "点云无法给出体积，必须用 --mass 指定总质量（点云只描述质量分布形状）"
        )
    mass = float(args.mass)

    props = from_points(points, mass)
    print(f"  修正模式        {props.correction_applied!r}")
    if props.correction_applied == "none":
        print("                  （判定为随机采样：估计量本身无偏，不应加修正）")
    else:
        print("                  （判定为规则采样：已加 ΔI = (1/6)Σmᵢhᵢ²）")
    if props.volume is None:
        print("  体积            未知（点云不定义体积，惯量按等质量点计算）")
    print()

    report_mass_properties(props, mass)

    if args.json:
        emit_properties(args.json, props, mass)
        print()
        print(f"数值已写入 {args.json}")

    if args.out_dir:
        # 随机生成的点云没有源文件：``source`` 必须是 None，不能把
        # ``random:4000:7:1,1,1`` 这种配方当路径写进去（那会变成一个
        # 永远解析不到的伪路径）。种子由输入卡的 inputs.points 记录。
        staged = None if points_spec else _stage_source(args, name, args.points)
        asset = build_asset(
            name=name,
            points=points,
            mass=mass,
            source_path=staged,
            path_base=_output_dir(args, name),
        )
        _publish_asset(args, name, asset, points_spec)

        print()
        print("  点云没有表面 ⇒ 不产生碰撞代理，也画不出预览图。")
        print("  想要 .blend 与预览图，请另外提供网格。")

    if args.emit:
        print()
        print("未写出场景：点云没有碰撞代理。", file=sys.stderr)
        return 1

    print()
    print(_LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
