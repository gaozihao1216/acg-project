# -*- coding: utf-8 -*-
"""物体资产：把一个不规则物体整理成一份自洽的 JSON。

为什么需要它
------------
前面每一项能力都是单独的模块：网格解析、质量属性、凸包、场景装配。但一个物体
要从"文件"走到"能渲染、能碰撞、能仿真"，中间必须有一份**约定好的数据形态**，
否则每个消费者都会自己再解释一遍坐标、单位、坐标系——这正是出错的地方。

本模块定义 ``object_asset/1`` 这份约定，并提供：

- :func:`build_asset` 从网格或点云生成资产字典
- :func:`save_asset` / :func:`load_asset` 读写
- :func:`asset_to_object_spec` 把资产转回场景物体（闭合"资产 → 引擎"）

两个坐标系必须显式
------------------
这是最容易出错的地方，所以它在 schema 里是一等公民：

====================  ==========================================  ============
坐标系                 定义                                         用途
====================  ==========================================  ============
``source``            输入文件自己的坐标（OBJ/STL 原样）            对照 CAD、看质心
``body``              原点在质心、坐标轴与惯量主轴对齐              碰撞、渲染、引擎
====================  ==========================================  ============

**质量属性报在 ``source`` 系**（这样质心和非对角项才有物理意义，也能和 CAD 对照）；
**碰撞几何报在 ``body`` 系**（引擎要的就是这个）。两者用 ``frames`` 里的变换互转。

变换里有两个四元数，它们是**共轭**关系，混用会让世界逆惯量错得离谱：

- ``frames.source_to_body.rotation``：把点从 source 转到 body，等于 Qᵀ
- ``engine.orientation_offset``：并进物体姿态用，等于 Q 本身

``checks`` 里有一条专门验证这两个互为转置。

为什么没有时间戳
----------------
资产要能进版本库、要能 diff。时间戳会让同样的输入产生不同的文件，
于是每次都"有变化"。所以除了 ``source.sha256``，不放任何随生成时刻变化的东西。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Sequence

from ..mathutils import quaternion as quat
from .geometry import ConvexHullGeometry
from .massprops import MassProperties, from_mesh, from_points, matrix3_to_quaternion
from .meshio import Mesh

#: schema 标识。消费者应当检查它以拒绝不认识的版本。
SCHEMA = "object_asset/1"


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------

def build_asset(
    *,
    name: str,
    mesh: Mesh | None = None,
    points: Sequence[Sequence[float]] | None = None,
    weights: Sequence[float] | None = None,
    mass: float,
    source_path: str | None = None,
    density: float | None = None,
    max_faces: int = 64,
    correction: str = "auto",
    path_base: str | None = None,
) -> dict[str, Any]:
    """生成一份 ``object_asset/1`` 字典。

    参数
    ----
    name        : 资产名（一般取文件名，用作场景里的物体名）
    mesh        : 网格（与 ``points`` 二选一）。给了网格才能做碰撞代理。
    points      : 点云（与 ``mesh`` 二选一）。点云没有表面，**不产生碰撞代理**。
    weights     : 点云的逐点相对质量（可表达非均匀密度）
    mass        : 总质量
    source_path : 原始文件路径（仅记录，用于溯源）
    density     : 材料密度（与 ``mass`` 二选一，这里由调用方折算好传 ``mass``）
    max_faces   : 凸包面数上限（SAT 的棱-棱轴数是 E²）
    correction  : 点云的截断修正模式
    path_base   : 资产 JSON 将写入的目录。给出时所有路径都写成相对它的形式，
                  于是案例目录可以整体搬走。``None`` 时保留绝对路径。

    返回
    ----
    dict：可直接 ``json.dump`` 的资产。**不含网格顶点**——几何在
    ``input/`` 的源文件和 ``output/*.blend`` 里，这里只存数值与引用。
    """
    if (mesh is None) == (points is None):
        raise ValueError("mesh 与 points 必须且只能给一个")

    if mesh is not None:
        props = from_mesh(mesh.vertices, mesh.faces, mass=mass)
        origin_kind = "mesh"
    else:
        props = from_points(points, mass, weights=weights, correction=correction)
        origin_kind = "points"

    if props.volume is None or props.volume <= 0.0:
        # 点云无法给出体积；用密度反推（若给了密度），否则留空
        volume = None
    else:
        volume = float(props.volume)

    moments, axes = props.principal()
    principal_quaternion = matrix3_to_quaternion(axes)
    # 点 → body 的旋转是 Qᵀ；姿态复合用的是 Q。两者共轭，见模块文档。
    to_body_rotation = quat.quat_conjugate(principal_quaternion)
    center = props.center_of_mass

    asset: dict[str, Any] = {
        "schema": SCHEMA,
        "name": name,
        "units": {"length": "m", "mass": "kg", "angle": "rad"},
        "paths_relative_to": "." if path_base else None,
        "source": _describe_source(source_path, mesh, props, origin_kind, path_base),
        "frames": {
            "source": {
                "origin": "输入文件的坐标原点",
                "axes": "输入文件自己的坐标轴",
            },
            "body": {
                "origin": "质心",
                "axes": "惯量主轴（按主惯量升序 λ1 ≤ λ2 ≤ λ3）",
            },
            "source_to_body": {
                "formula": "p_body = R(rotation) @ (p_source + translation)",
                "translation": [_clean(-center[0]), _clean(-center[1]), _clean(-center[2])],
                "rotation_wxyz": [_clean(v) for v in to_body_rotation],
            },
            "body_to_source": {
                "formula": "p_source = R(rotation) @ p_body + translation",
                "translation": [_clean(center[0]), _clean(center[1]), _clean(center[2])],
                "rotation_wxyz": [_clean(v) for v in principal_quaternion],
            },
        },
        "mass_properties": {
            "frame": "source",
            "mass": _clean(props.mass),
            "volume": _clean(volume) if volume is not None else None,
            "density": _clean(props.mass / volume) if volume else None,
            "surface_area": _clean(mesh.surface_area()) if mesh is not None else None,
            "center_of_mass": [_clean(v) for v in center],
            # 对称化：浮点求和会让 I[r][c] 与 I[c][r] 差 1e-17，
            # 写进 JSON 后看着像两行不一致。物理上它必须对称，取平均即可。
            "inertia_com": [
                [
                    _clean(0.5 * (props.inertia[r][c] + props.inertia[c][r]))
                    for c in range(3)
                ]
                for r in range(3)
            ],
            "inertia_products": [_clean(v) for v in props.products()],
            "non_diagonal_ratio": _clean(_specificity(props)),
            "correction": {
                "mode": props.correction_applied,
                "weighted": bool(props.weighted),
            },
        },
        "principal": {
            "moments": [_clean(v) for v in moments],
            "axes_columns": [
                [_clean(axes[r][c]) for c in range(3)] for r in range(3)
            ],
            "quaternion_wxyz": [_clean(v) for v in principal_quaternion],
        },
        "collision": None,
        "render": _describe_render(name, mesh, source_path, path_base),
        "engine": None,
        "checks": [],
    }

    # --- 碰撞代理 ---
    hull = None
    if mesh is not None:
        try:
            hull = ConvexHullGeometry(mesh.vertices, max_faces=max_faces)
        except ValueError as exc:
            asset["collision"] = {
                "representation": None,
                "error": str(exc),
                "hint": "碰撞代理应当用粗网格；或者显式调高 max_faces 接受 E² 开销。",
            }
        else:
            asset["collision"] = {
                "representation": "convex_hull",
                "frame": "body",
                "points": [
                    [_clean(v[0]), _clean(v[1]), _clean(v[2])] for v in hull.vertices
                ],
                "faces": [list(face) for face in hull.faces],
                "edges": [list(edge) for edge in hull.edges],
                "face_count": len(hull.faces),
                "edge_count": len(hull.edges),
                "volume": _clean(hull.volume()),
                "hull_fill_ratio": _clean(
                    hull.volume() / volume if volume else None
                ),
                "sat_candidate_axes": (
                    2 * len(hull.faces) + len(hull.edges) ** 2
                ),
            }

    # --- 引擎交接 ---
    ground_offset = 0.0
    if hull is not None:
        ground_offset = -min(v[2] for v in hull.vertices)
    elif mesh is not None:
        ground_offset = -(
            min(v[2] for v in mesh.vertices) - center[2]
        )

    asset["engine"] = {
        "inertia_local": [_clean(v) for v in moments],
        "orientation_offset_wxyz": [_clean(v) for v in principal_quaternion],
        "orientation_formula": "orientation = quat_multiply(spec_orientation, offset)",
        "ground_z_offset": _clean(ground_offset),
        "notes": (
            "引擎内部只存对角惯量；把主轴旋转并进姿态后 "
            "I_world = (R·Q)·diag(inertia_local)·(R·Q)ᵀ 依然成立。"
        ),
    }

    asset["checks"] = _run_checks(
        asset, props, moments, axes, principal_quaternion, to_body_rotation,
        hull, mesh, volume,
    )
    return asset


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def save_asset(path: str, asset: dict[str, Any]) -> None:
    """写出资产 JSON（UTF-8、缩进 2、结尾换行）。"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(asset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def case_paths(root: str, name: str) -> dict[str, str]:
    """一个物体的案例目录约定。

    为什么按"物体"分目录、而不是按"文件类型"分
    ------------------------------------------
    按类型分（所有 json 一堆、所有 png 一堆）时，想看某个物体做了什么，
    要在三个目录之间来回找；而且删掉一个物体要记得在三个地方各删一次。

    按物体分则每个案例**自包含**，代价是同一类产物散落在各处——所以下面再分
    ``input`` / ``output`` 两层，把"进来的"和"算出去的"隔开：

    ::

        root/
        └── <name>/
            ├── input/   原始网格/点云（工具会拷进来，案例可独立带走）
            └── output/  <name>.asset.json / .blend / .preview.png

    ``input`` 是**输入**而不是中间产物：有了它，一年后还能确认这份资产
    是从哪个文件、哪个版本算出来的（``source.sha256`` 与它对应）。
    """
    base = os.path.join(root, name)
    input_dir = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    return {
        "case": base,
        "input": input_dir,
        "output": output_dir,
        "request": os.path.join(input_dir, "request.json"),
        "asset_json": os.path.join(output_dir, f"{name}.asset.json"),
        "blend": os.path.join(output_dir, f"{name}.blend"),
        "preview": os.path.join(output_dir, f"{name}.preview.png"),
    }


def stage_input(source_path: str, input_dir: str) -> str:
    """把源文件复制进案例的 ``input/``，返回副本路径。

    已经在该目录里就原样返回（避免自我复制）。复制而不是移动：
    原文件可能被别的东西引用，搬走会打断它们。

    用 ``copy2`` 保留修改时间——``source.sha256`` 只保证内容一致，
    时间戳是排查"这份资产是不是过期了"的另一个线索。
    """
    import shutil

    source = os.path.abspath(source_path)
    target_dir = os.path.abspath(input_dir)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.dirname(source) == target_dir:
        return source

    target = os.path.join(target_dir, os.path.basename(source))
    if os.path.exists(target) and os.path.samefile(source, target):
        return target
    shutil.copy2(source, target)
    return target


def build_request(
    *,
    name: str,
    command: Sequence[str],
    mass: float | None = None,
    density: float | None = None,
    source: str | None = None,
    points: dict[str, Any] | None = None,
    max_faces: int = 64,
    correction: str = "auto",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成 ``input/request.json``：这次运行的**输入卡**。

    为什么需要它
    ------------
    ``output/`` 里的东西都是**算出来的**——从资产能反推出惯量、凸包、轴。
    但有几样东西反推不回来：

    =================== ==================================================
    信息                 为什么反推不回来
    =================== ==================================================
    ``--mass`` 还是       同一个网格给 3 kg 和给密度 1200，算出的惯量不同，
    ``--density``        但产物里只有一个"质量"数字
    **随机点云的种子**     点云文件能存下来，但"它是怎么生成的"存不下来
    凸包面数上限           同一点集换个 ``max_faces`` 会得到不同代理
    修正式的选择           资产里记了**结果**（``correction.mode``），
                        但记不了"我要求的是 auto 还是强制 nn"
    =================== ==================================================

    所以输入卡记的是**决定这次结果的自由度**，不是结果的副本。
    它和 ``output/`` 的分工：``input/`` 说"我要什么"，``output/`` 说"算出了什么"。

    ``command`` 存完整命令行（不含可执行文件路径），于是这个案例可以被**原样重放**：

    ::

        python Code/scripts/analyze_object.py <input/request.json 里的参数>

    ``extra`` 用于放调用方的补充字段（如点云的随机参数）。
    """
    request: dict[str, Any] = {
        "schema": "case_request/1",
        "name": name,
        "command": [str(part) for part in command],
        "inputs": {
            "source": source,
            "points": points,
        },
        "mass_spec": {
            "mass": _clean(mass) if mass is not None else None,
            "density": _clean(density) if density is not None else None,
            "rule": (
                "mass 直接给定"
                if mass is not None
                else "density × 网格体积"
            ),
        },
        "geometry_spec": {
            "max_faces": int(max_faces),
        },
        "correction_spec": {
            "requested": correction,
            "note": (
                "点云专用。资产里的 correction.mode 是**最终决定**——"
                "auto 可能判定为不应修正。"
            ),
        },
    }
    if extra:
        request["extra"] = extra
    return request


def save_request(path: str, request: dict[str, Any]) -> None:
    """写出输入卡 ``input/request.json``。"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(request, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_asset(path: str) -> dict[str, Any]:
    """读入资产并校验 schema。"""
    with open(path, "r", encoding="utf-8") as handle:
        asset = json.load(handle)
    schema = asset.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"不认识的资产版本 {schema!r}；本代码支持 {SCHEMA!r}"
        )
    return asset


def asset_to_object_spec(asset: dict[str, Any]) -> dict[str, Any]:
    """把资产转成场景 JSON 里的一个物体。

    这样 ``*.asset.json`` 就是唯一真值：场景可以从资产派生，而不是各写一份。

    位置取"贴地"而不是原点：``position.z = -body 系最低点``，
    于是物体一加载就静止在地面上（引擎的初始穿透校验会拒绝嵌入的场景）。
    """
    if not asset.get("collision") or not asset["collision"].get("points"):
        raise ValueError(
            f"资产 {asset.get('name')!r} 没有碰撞代理，无法生成场景物体。"
            f"点云输入不产生碰撞几何——请另外提供网格。"
        )

    collision = asset["collision"]
    engine = asset["engine"]
    moments = asset["principal"]["moments"]
    offset = engine["orientation_offset_wxyz"]

    return {
        "name": asset["name"],
        "geometry": {
            "type": "hull",
            "points": [list(p) for p in collision["points"]],
        },
        "transform": {
            "position": [0.0, 0.0, _clean(engine["ground_z_offset"])],
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
        "mass": asset["mass_properties"]["mass"],
        # 资产里带了渲染材质，场景就该用它，而不是让每个调用方各写一套颜色。
        # 之前 emit_scene 固定写死一个蓝色，等于把资产的 render 段当摆设。
        "material": _material_from_asset(asset),
        # 主惯量必须真的传给引擎，否则它就只是 JSON 里的一句断言。
        #
        # 这条通路是必要的：引擎原来无条件按**碰撞几何**反推惯量，
        # 而碰撞代理与质量分布常常不是一回事。实测空心盒（壁厚 0.12）
        # 从外包实心盒反推得到 1.25 倍的真实惯量，误差 24.8%——
        # 正是"质量分布被丢掉"的那个量，而且提高精度也不会消失。
        "inertia": [_clean(value) for value in moments],
        # 下面两个仅供人核对，引擎不读
        "_inertia_local_expected": list(moments),
        "_orientation_offset_expected": list(offset),
    }


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------

def _material_from_asset(asset: dict[str, Any]) -> dict[str, Any]:
    """把资产里的 ``render.materials[0]`` 转成场景物体要的材质字典。

    缺省值是有意给一个中性灰：如果资产没写材质，让物体看起来"没有颜色"
    比随便挑一个颜色更诚实。
    """
    materials = (asset.get("render") or {}).get("materials") or []
    if not materials:
        return {"base_color": [0.7, 0.7, 0.7], "roughness": 0.5, "metallic": 0.0}

    first = materials[0]
    # 资产的材质是 **RGBA**（Blender 用），场景规格要 **RGB**（三个分量）。
    # 直接把 4 个分量传下去会被场景解析器当场拒绝——这个严格性是好事，
    # 它把"两套材质格式不一样"这件事变成了一条明确的错误而不是渲染出怪色。
    raw = list(first.get("base_color") or [0.7, 0.7, 0.7, 1.0])
    color = [float(value) for value in raw[:3]]
    while len(color) < 3:
        color.append(0.7)
    return {
        "base_color": color,
        "roughness": float(first.get("roughness", 0.5)),
        "metallic": float(first.get("metallic", 0.0)),
    }


def _clean(value) -> float | None:
    """去掉 -0.0 与浮点噪声，让 JSON 可读可 diff。"""
    if value is None:
        return None
    number = float(value)
    if number == 0.0:
        return 0.0
    # 把 1e-17 这类噪声压成 0，否则每行都有科学计数法
    if abs(number) < 1e-15:
        return 0.0
    return number


def _specificity(props: MassProperties) -> float | None:
    scale = max(abs(props.inertia[i][i]) for i in range(3))
    if scale <= 0.0:
        return None
    return max(abs(v) for v in props.products()) / scale


def _relpath(path, base) -> str | None:
    """把绝对路径写成**相对于 JSON 所在目录**的 posix 路径。

    为什么用相对路径：案例目录整体搬走之后引用仍然成立。写成绝对路径，
    换台机器就指不到了，而 ``source.sha256`` 也就没有校验对象。
    基准目录由 ``paths_relative_to`` 字段显式声明，不留歧义。
    """
    if path is None:
        return None
    if base is None:
        return str(path).replace("\\", "/")
    try:
        relative = os.path.relpath(os.path.abspath(path), os.path.abspath(base))
    except ValueError:
        # Windows 上跨盘符无法求相对路径，只能退回绝对路径
        return os.path.abspath(path).replace("\\", "/")
    return relative.replace("\\", "/")


def _describe_source(source_path, mesh, props, origin_kind, base) -> dict[str, Any]:
    info: dict[str, Any] = {
        "origin": origin_kind,
        "path": _relpath(source_path, base),
        "sha256": _sha256(source_path) if source_path else None,
    }
    if mesh is not None:
        info.update(
            {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_closed": bool(mesh.is_closed),
                "is_consistently_oriented": bool(mesh.is_consistently_oriented),
                "euler_characteristic": int(mesh.euler_characteristic),
                "signed_volume": _clean(mesh.signed_volume()),
            }
        )
    else:
        info["weighted"] = bool(props.weighted)
    return info


def _describe_render(name, mesh, source_path, base) -> dict[str, Any]:
    """渲染层需要的信息：只要**引用**，不要几何。

    为什么不内联顶点
    ----------------
    几何已经在两处了：``input/`` 里的源文件（原始、可追溯）和
    ``output/<name>.blend``（可直接用）。JSON 再存一份顶点表就是**第三份重复**：
    一个 5 万面的模型会让 JSON 从 10 KB 涨到几 MB，而它提供不了任何
    前两者没有的信息。

    所以这里只放"去哪里找"以及"它长什么样"的统计量（顶点/面数、坐标系），
    后者用于**校验**引用的文件没被换掉——配合 ``source.sha256`` 足以自证。
    """
    block: dict[str, Any] = {
        "object_name": name,
        "mesh": {
            "path": _relpath(source_path, base),
            "frame": "source",
        },
        "materials": [
            {"slot": "body", "base_color": [0.62, 0.68, 0.78, 1.0],
             "roughness": 0.45, "metallic": 0.0},
        ],
        "ground": {"enabled": True, "size": 8.0},
    }
    if mesh is not None:
        block["mesh"]["vertices"] = len(mesh.vertices)
        block["mesh"]["faces"] = len(mesh.faces)
    else:
        block["mesh"] = {"path": None, "frame": None}
        block["note"] = "点云输入没有表面，渲染需要另外提供网格。"
    return block


def _sha256(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# 公开别名：场景级资产（scene_asset.py）复用这两个，避免两处实现漂移。
# 相对路径的解析基准必须与指纹算法完全一致，否则"引用能读到但指纹对不上"
# 这类问题会以最难查的形式出现。
relpath = _relpath
sha256_file = _sha256


def _check(
    name: str, ok: bool, detail: str, severity: str = "error"
) -> dict[str, Any]:
    """一条自检结果。

    ``severity`` 区分两类结论，混在一起会让报告失真：

    - ``"error"``：不通过就意味着**某个数字不能用**（体积为负、张量不可重构）
    - ``"info"`` ：这是**形状的性质**而不是缺陷（有凹陷所以凸包偏大、
      非对角项大所以不能用对角近似）。``ok == False`` 只是提醒消费者别做某种简化。
    """
    return {"name": name, "ok": bool(ok), "detail": detail, "severity": severity}


def _run_checks(
    asset, props, moments, axes, principal_quaternion, to_body_rotation,
    hull, mesh, volume,
) -> list[dict[str, Any]]:
    """把"能验证的都验证一遍"，结果写进资产。

    消费者不必相信生成者：每条都能自己复核。任何一条 ``ok == False``
    都意味着某个数字不能用，而不是"精度略差"。
    """
    checks: list[dict[str, Any]] = []

    # 1. 主轴正交且右手系
    orthonormal_error = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(axes[k][i] * axes[k][j] for k in range(3))
            target = 1.0 if i == j else 0.0
            orthonormal_error = max(orthonormal_error, abs(dot - target))
    determinant = (
        axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1])
        - axes[0][1] * (axes[1][0] * axes[2][2] - axes[1][2] * axes[2][0])
        + axes[0][2] * (axes[1][0] * axes[2][1] - axes[1][1] * axes[2][0])
    )
    checks.append(_check(
        "principal_axes_orthonormal",
        orthonormal_error < 1e-12 and abs(determinant - 1.0) < 1e-12,
        f"正交误差={orthonormal_error:.2e}，det={determinant:+.12f}（须为 +1）",
    ))

    # 2. 张量可重构：Q·diag(Λ)·Qᵀ 必须回到原张量
    reconstruction_error = 0.0
    for r in range(3):
        for c in range(3):
            value = sum(
                axes[r][k] * moments[k] * axes[c][k] for k in range(3)
            )
            reconstruction_error = max(
                reconstruction_error, abs(value - props.inertia[r][c])
            )
    scale = max(abs(props.inertia[i][i]) for i in range(3)) or 1.0
    checks.append(_check(
        "tensor_reconstruction",
        reconstruction_error / scale < 1e-12,
        f"max|QΛQᵀ − I| / 最大对角元 = {reconstruction_error / scale:.2e}",
    ))

    # 3. 两个四元数互为共轭（这条专门防"主轴方向混用"那类 bug）
    offset = principal_quaternion
    expected_conjugate = (offset[0], -offset[1], -offset[2], -offset[3])
    conjugate_error = max(
        abs(to_body_rotation[i] - expected_conjugate[i]) for i in range(4)
    )
    checks.append(_check(
        "frame_quaternions_conjugate",
        conjugate_error < 1e-15,
        f"source_to_body 与 orientation_offset 应互为共轭，差异={conjugate_error:.2e}",
    ))

    # 4. 质量与体积
    if volume:
        checks.append(_check(
            "volume_positive", volume > 0.0, f"体积={volume:.9f} m³",
        ))
    else:
        checks.append(_check(
            "volume_positive", False,
            "点云无法给出体积：密度与贴地高度都无从计算，仅惯量可用",
        ))

    # 5. 网格可闭合（否则散度定理给出无意义数字）
    if mesh is not None:
        checks.append(_check(
            "mesh_closed",
            bool(mesh.is_closed) and bool(mesh.is_consistently_oriented),
            f"闭合={mesh.is_closed}，缠绕一致={mesh.is_consistently_oriented}，"
            f"χ={mesh.euler_characteristic}",
        ))

    # 6. 凸包覆盖度：< 1 说明有凹陷，碰撞代理是外近似
    if hull is not None and volume:
        fill = hull.volume() / volume
        checks.append(_check(
            "collision_is_exact",
            1.0 - 1e-9 <= fill <= 1.0 + 1e-9,
            f"凸包/本体 = {fill:.6f}"
            + ("" if abs(fill - 1.0) < 1e-9 else "  ← 有凹陷，碰撞代理偏大"),
            severity="info",
        ))

    # 7. 对角近似是否可接受
    specificity = _specificity(props)
    if specificity is not None:
        checks.append(_check(
            "diagonal_approximation_safe",
            specificity < 1e-3,
            f"非对角占比={specificity * 100:.3f}%"
            + ("（可用对角近似）" if specificity < 1e-3
               else "（不能用对角近似；主轴旋转已并进姿态，引擎自动处理）"),
            severity="info",
        ))

    # 8. 贴地偏移必须非负
    ground = asset["engine"]["ground_z_offset"]
    checks.append(_check(
        "ground_offset_nonnegative",
        ground >= -1e-9,
        f"ground_z_offset={ground:+.6f} m（把物体放到地面上所需的抬升）",
    ))

    return checks


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def format_checks(asset: dict[str, Any]) -> list[str]:
    """把 checks 渲染成若干行。

    错误与提示分开打印——混在一起会让人以为"有凹陷"是一个故障，
    而它只是一个形状事实。
    """
    checks = asset.get("checks") or []
    errors = [c for c in checks if not c["ok"] and c.get("severity") == "error"]
    notices = [c for c in checks if not c["ok"] and c.get("severity") != "error"]

    if errors:
        lines = [f"自检：{len(errors)} 项不通过（数据不可用）"]
        for item in errors:
            lines.append(f"  ✗ {item['name']}: {item['detail']}")
    else:
        lines = [f"自检：{len(checks)} 项全部通过"]

    for item in notices:
        lines.append(f"  · {item['name']}: {item['detail']}")
    return lines
