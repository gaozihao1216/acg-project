# -*- coding: utf-8 -*-
"""场景级资产 ``scene_asset/1``：只存名字与引用。

设计原则：场景只说它自己才有答案的事
------------------------------------
一个物体自己的数据（质量、惯量、主轴、凸包、材质）**已经**在
``object_asset/1`` 里了。场景如果把它们复制一份，会产生三个具体问题：

1. **占空间**：凸包点集是最肥的部分，20 个物体能到几 MB，而这份数据一个字
   都是抄来的
2. **会分叉**：物体资产改了，场景里的副本不会跟着变。于是同一份数据有两个
   版本，而且没人知道哪个对
3. **没法定位错误**：数字对不上时，不知道该信场景还是信物品

所以场景只回答这四件事：

- **谁在场**（名称）
- **在哪里、什么朝向**（``transform``）
- **引用哪份物品资产**（``asset`` + ``asset_sha256``）
- **环境怎么配**（地面、相机、光照、求解器）

物体级的任何数值都不出现在这里。要拿惯量？沿着 ``asset`` 去读那一份。

引用而不是复制，代价是"场景不能单独拷走"——所以 ``asset`` 写成**相对路径**，
并且 ``case_dir`` 的结构保证了 ``../../<name>/output/<name>.asset.json``
这样的引用在 ``results/`` 整体搬家后仍然成立。
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from .asset import asset_to_object_spec, load_asset, relpath, sha256_file

#: 场景资产 schema 标识
SCENE_SCHEMA = "scene_asset/1"


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------

def build_scene_asset(
    *,
    name: str,
    objects: Sequence[dict[str, Any]],
    path_base: str | None = None,
    ground: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    lights: Sequence[dict[str, Any]] | None = None,
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成 ``scene_asset/1`` 字典。

    参数
    ----
    name      : 场景名
    objects   : 每项 ``{"name", "asset", "transform"?, "material"?, "physics"?}``。
                ``asset`` 是物品资产的路径，会被写成相对 ``path_base`` 的形式。
    path_base : 场景 JSON 将写入的目录。给出时路径相对化，整个 ``results/``
                可以整体搬走。
    ground / camera / lights / simulation
              : 场景级配置。缺省给一套能直接跑的值。

    返回
    ----
    dict，可直接 ``json.dump``。**不含任何物体级数值。**
    """
    prepared: list[dict[str, Any]] = []
    for item in objects:
        if "name" not in item:
            raise ValueError("每个物体都必须有 name")
        asset_path = item.get("asset")
        if not asset_path:
            raise ValueError(f"物体 {item['name']!r} 缺少 asset 引用")

        entry: dict[str, Any] = {
            "name": item["name"],
            "asset": relpath(asset_path, path_base),
            # 校验引用没被偷偷换掉：物体资产一旦改动，场景这里就对不上
            "asset_sha256": sha256_file(asset_path),
        }

        transform = dict(item.get("transform") or {})
        transform.setdefault("position", [0.0, 0.0, 0.0])
        transform.setdefault("orientation", [0.0, 0.0, 0.0, 1.0])
        entry["transform"] = transform

        # 只允许**场景级覆盖**，不允许把物品级数据搬进来
        if "material" in item:
            entry["material"] = item["material"]
        if "physics" in item:
            entry["physics"] = item["physics"]
        prepared.append(entry)

    scene: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "name": name,
        "units": {"length": "m", "mass": "kg", "angle": "rad"},
        "paths_relative_to": "." if path_base else None,
        "objects": prepared,
        "ground": ground if ground is not None else {
            "enabled": True, "z": 0.0, "friction": 0.8,
            "restitution": 0.05, "color": [0.22, 0.23, 0.25],
        },
        "camera": camera if camera is not None else {
            "position": [3.4, -4.6, 2.8], "target": [0.0, 0.0, 0.7], "fov": 42.0,
        },
        "lights": list(lights) if lights is not None else [
            {"name": "key", "type": "area", "position": [3.0, -3.5, 4.2],
             "target": [0.0, 0.0, 0.5], "energy": 520.0, "size": 3.0},
            {"name": "fill", "type": "area", "position": [-3.6, -1.2, 2.0],
             "target": [0.0, 0.0, 0.5], "energy": 180.0, "size": 3.4,
             "color": [0.82, 0.88, 1.0]},
        ],
        "simulation": simulation if simulation is not None else {
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
            "render_fps": 24,
            "duration": 2.0,
        },
        "checks": [],
    }
    scene["checks"] = check_scene(scene, path_base)
    return scene


def check_scene(scene: dict[str, Any], base: str | None) -> list[dict[str, Any]]:
    """场景自检：引用是否解析得到、指纹是否一致、有没有偷偷复制数据。

    三条都是**写盘时就能查**的，不必等到加载阶段才发现某个引用断了。
    第三条（``no_inlined_object_data``）是设计约束的守门人：一旦有人图方便
    把 ``mass_properties`` 塞进场景，它会立刻报出来。
    """
    checks: list[dict[str, Any]] = []
    base_dir = base or os.getcwd()

    missing: list[str] = []
    mismatched: list[str] = []
    duplicated: list[str] = []

    for entry in scene.get("objects") or []:
        reference = entry.get("asset")
        resolved = (
            reference if os.path.isabs(str(reference))
            else os.path.join(base_dir, str(reference))
        )
        if not os.path.isfile(resolved):
            missing.append(f"{entry.get('name')}→{reference}")
            continue
        actual = sha256_file(resolved)
        if entry.get("asset_sha256") and actual != entry["asset_sha256"]:
            mismatched.append(entry.get("name"))
        # 场景条目里不该出现物体级字段
        for forbidden in ("mass_properties", "principal", "collision", "geometry"):
            if forbidden in entry:
                duplicated.append(f"{entry.get('name')}:{forbidden}")

    checks.append({
        "name": "references_resolved",
        "ok": not missing,
        "severity": "error",
        "detail": (
            f"{len(scene.get('objects') or [])} 个引用全部可解析"
            if not missing
            else f"解析不到：{', '.join(missing)}"
        ),
    })
    checks.append({
        "name": "asset_fingerprints_match",
        "ok": not mismatched,
        "severity": "error",
        "detail": (
            "物体指纹与场景记录一致"
            if not mismatched
            else f"资产已被改动：{', '.join(mismatched)}（场景应重新生成）"
        ),
    })
    checks.append({
        "name": "no_inlined_object_data",
        "ok": not duplicated,
        "severity": "error",
        "detail": (
            "场景只存引用，未复制物体级数据"
            if not duplicated
            else f"发现内联的物体级数据：{', '.join(duplicated)}"
        ),
    })
    return checks


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def save_scene_asset(path: str, scene: dict[str, Any]) -> None:
    """写出场景资产。与物体资产同样的确定性策略：不放时间戳。"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_scene_asset(path: str) -> dict[str, Any]:
    """读入场景资产并**解析引用**。

    解析在这里做，而不是丢给每个消费者各做一遍——否则"相对路径怎么解析"
    这个细节会被实现出好几套，而且每套都可能错在不同地方。
    """
    with open(path, "r", encoding="utf-8") as handle:
        scene = json.load(handle)
    if scene.get("schema") != SCENE_SCHEMA:
        raise ValueError(
            f"不认识的场景资产版本 {scene.get('schema')!r}；"
            f"本代码支持 {SCENE_SCHEMA!r}"
        )

    base = os.path.dirname(os.path.abspath(path))
    resolved: list[dict[str, Any]] = []
    for entry in scene.get("objects") or []:
        reference = entry.get("asset")
        if not reference:
            raise ValueError(f"物体 {entry.get('name')!r} 没有 asset 引用")

        asset_path = (
            reference if os.path.isabs(str(reference))
            else os.path.join(base, str(reference))
        )
        if not os.path.isfile(asset_path):
            raise FileNotFoundError(
                f"物体 {entry['name']!r} 引用的资产不存在：{reference}\n"
                f"  （相对于 {base} 解析）"
            )

        resolved.append({
            "name": entry["name"],
            "asset_path": asset_path,
            "asset": load_asset(asset_path),
            "transform": entry.get("transform") or {},
            "material": entry.get("material"),
            "physics": entry.get("physics"),
        })

    scene["resolved"] = resolved
    scene["base_dir"] = base
    return scene


def scene_asset_to_objects(scene: dict[str, Any], *, clearance: float = 0.02):
    """把已解析的场景资产摊平成可直接喂给引擎的物体列表。

    几何与质量来自**各物体自己的资产**；场景只提供位置与朝向。
    物体资产里的 ``ground_z_offset`` 用来把物体摆到地面上，再加一点余量——
    引擎的初始穿透校验会拒绝一开始就嵌进地面的场景。
    """
    specs: list[dict[str, Any]] = []
    for item in scene.get("resolved") or []:
        spec = asset_to_object_spec(item["asset"])
        spec["name"] = item["name"]

        transform = item["transform"]
        position = list(transform.get("position") or [0.0, 0.0, 0.0])
        if "position_z" in item:  # 兼容旧写法
            position[2] = item["position_z"]
        spec["transform"]["position"] = [
            float(position[0]), float(position[1]), float(position[2]) + clearance,
        ]
        spec["transform"]["orientation"] = list(
            transform.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        )

        if item.get("material"):
            spec["material"] = item["material"]
        if item.get("physics"):
            spec["physics"] = item["physics"]
        spec.pop("_inertia_local_expected", None)
        spec.pop("_orientation_offset_expected", None)
        specs.append(spec)
    return specs


def scene_asset_to_scene_dict(scene: dict[str, Any], *, clearance: float = 0.02):
    """把已解析的场景资产摊平成 ``SceneSpec.from_dict`` / ``load_scene`` 能吃的字典。

    这是"引用 → 具体"的唯一展开点。``run_sim.py`` 和 ``assemble_scene.py``
    都调它，避免两处各写一遍展开逻辑——那种重复迟早会在某个字段上分叉。
    """
    return {
        "name": scene.get("name", "scene"),
        "camera": scene.get("camera"),
        "ground": scene.get("ground"),
        "lights": scene.get("lights"),
        "objects": scene_asset_to_objects(scene, clearance=clearance),
        "simulation": scene.get("simulation"),
    }


def format_scene_checks(scene: dict[str, Any]) -> list[str]:
    """把场景自检渲染成若干行。"""
    checks = scene.get("checks") or []
    errors = [c for c in checks if not c["ok"] and c.get("severity") == "error"]
    if errors:
        lines = [f"场景自检：{len(errors)} 项不通过"]
        for item in errors:
            lines.append(f"  ✗ {item['name']}: {item['detail']}")
        return lines
    return [f"场景自检：{len(checks)} 项全部通过"]
