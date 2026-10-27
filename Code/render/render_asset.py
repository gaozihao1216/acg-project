# -*- coding: utf-8 -*-
"""把一个 ``object_asset/1`` 变成两件东西：``.blend`` 与 QA 预览图。

为什么要有这一步
----------------
JSON 是数据，但不是资产：它没法被美术打开、检查、微调。而一张图不是资产，
只是一个截图。所以这里一次产出两个，各管一半：

- ``.blend``：渲染网格 + **碰撞代理（独立集合，线框）** + 质心/主轴标记，
  全部在**同一坐标系**里，所以"数字对不对"用眼睛就能核对。
- ``.png``：QA 板。它不追求好看，追求**一眼看出网格、凸包、主轴三者是否吻合**。

统一到 body 系
--------------
资产里渲染网格报在 ``source`` 系、碰撞点报在 ``body`` 系。如果直接摆进同一个
场景，两者会错开一个"质心平移 + 主轴旋转"。所以这里**统一换到 body 系**：

    p_body = R(source_to_body.rotation) @ (p_source + source_to_body.translation)

于是质心落在世界原点、主轴与世界坐标轴重合，轴标就成了 X/Y/Z 三条——
这也是引擎内部看到的姿态。

地面放在 ``z = -ground_z_offset``，物体正好贴地，而 body 原点仍留在世界原点。

用法::

    blender --background --factory-startup --python Code/render/render_asset.py -- \
        --asset results/out/prism.asset.json \
        --blend results/out/prism.blend \
        --png   results/out/prism.preview.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

import bpy
from mathutils import Quaternion, Vector

from src.scene.scene_spec import CameraSpec, GroundSpec, LightSpec, MaterialSpec, RenderSpec
from render.blender_builder import (
    configure_render,
    configure_world,
    create_camera,
    create_ground,
    create_light,
    create_material,
    render_to_file,
    reset_scene,
)

#: 轴的颜色：X 红 / Y 绿 / Z 蓝。用纯色而非渐变，避免和物体材质混淆。
_AXIS_COLORS = {
    "X": (0.92, 0.22, 0.20),
    "Y": (0.30, 0.78, 0.32),
    "Z": (0.25, 0.50, 0.95),
}

_LINE = "=" * 74


# ---------------------------------------------------------------------------
# 坐标变换
# ---------------------------------------------------------------------------

def _to_body(point, translation, rotation) -> Vector:
    """``p_body = R(rotation) @ (p_source + translation)``。"""
    shifted = Vector(point) + Vector(translation)
    return rotation.to_matrix() @ shifted


def _resolve_mesh_path(asset, asset_dir: str) -> str | None:
    """把 ``render.mesh.path`` 解析成绝对路径。

    JSON 里的路径**相对于 JSON 所在目录**（``paths_relative_to`` 声明了这一点），
    所以这里不能直接 ``open``，必须先拼上 ``asset_dir``。否则从别的工作目录
    调用 Blender 就会找不到文件——而这属于"换台机器才炸"的那类 bug。
    """
    mesh = (asset.get("render") or {}).get("mesh") or {}
    reference = mesh.get("path")
    if not reference:
        return None
    if os.path.isabs(reference):
        return reference
    return os.path.normpath(os.path.join(asset_dir, reference))


def _load_render_mesh(asset, asset_dir: str):
    """读出渲染网格的顶点与面（``source`` 系），失败返回 ``None``。

    几何不在 JSON 里——它要么在 ``input/`` 的源文件里，要么在 ``.blend`` 里。
    这里读源文件：``.blend`` 正是这个脚本要**产出**的东西，不能反过来依赖它。
    """
    path = _resolve_mesh_path(asset, asset_dir)
    if not path or not os.path.isfile(path):
        return None
    try:
        from src.geom.meshio import load_mesh

        mesh = load_mesh(path)
        return list(mesh.vertices), [list(face) for face in mesh.faces]
    except Exception as exc:  # noqa: BLE001
        print(f"[asset] 读取渲染网格失败（{path}）：{exc}")
        return None


def _body_vertices(asset, asset_dir: str) -> list[Vector]:
    """把渲染网格的顶点换到 body 系；没有网格时退回碰撞点。"""
    transform = asset["frames"]["source_to_body"]
    translation = transform["translation"]
    rotation = Quaternion(transform["rotation_wxyz"])

    loaded = _load_render_mesh(asset, asset_dir)
    if loaded is not None:
        vertices, _ = loaded
        return [_to_body(v, translation, rotation) for v in vertices]

    collision = asset.get("collision") or {}
    if collision.get("points"):
        # 碰撞点本身已在 body 系，不需要变换
        return [Vector(p) for p in collision["points"]]
    return []


# ---------------------------------------------------------------------------
# 物体构造
# ---------------------------------------------------------------------------

def _set_alpha(material, alpha: float) -> None:
    """给 Principled BSDF 设 Alpha（Cycles 用它做透明）。

    还要设 ``blend_method``：glTF 导出器靠它决定 ``alphaMode``。
    只改 Alpha 端口的话，渲染是透明的，但导出的 GLB 里材质仍是 OPAQUE，
    在浏览器/3D 查看器里就看不出内部——而"能看见内部"正是这张资产的意义。
    """
    try:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None and "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        material.blend_method = "BLEND"
    except Exception as exc:  # noqa: BLE001
        print(f"[asset] 设置透明度失败（不影响其余部分）：{exc}")


def _collect_asset_objects(name: str) -> list:
    """挑出属于这个资产的对象，供 GLB 导出。

    按名字筛而不是让 ``build_visual`` 返回列表：那样要改它的签名和所有返回值，
    而这里只多一个消费者。名字是内部约定，集中在函数里，改动也只在一处。

    **不导标签**：``label_X/Y/Z`` 是朝向相机的广告牌文字，在可自由旋转的
    查看器里立刻变成"贴在屏幕上"的怪东西。GLB 是给人转着看的，标签属于
    固定视角的 QA 图。
    """
    import bpy

    prefixes = ("axis_",)
    exact = {f"{name}_render", f"{name}_collision", "COM"}
    picked = []
    for obj in bpy.context.scene.objects:
        if obj.type not in {"MESH"}:
            continue
        if obj.name in exact or obj.name.startswith(prefixes):
            picked.append(obj)
    return picked


def export_glb(path: str, objects: list) -> str:
    """把资产相关对象导出成 GLB（可选的第四件产物）。

    为什么它**不替代** ``.preview.png``
    ----------------------------------
    GLB 能转着看，但导不出那张图的语义：质心点、三条主轴、凸包线框的
    线宽都是为固定视角摆出来的辅助几何，换视角就不再是"标注"而只是几何。
    所以 GLB 是补充（给外部工具、给自由观察），QA 证据仍以 PNG 为准。

    ``export_yup=True`` 是 glTF 规范要求（Y 轴向上），Blender 的 Z 会映射到
    glTF 的 Y——这正是我们想要的：资产是 Z-up 的 body 系，导出去在查看器里
    仍然是"立着"的。
    """
    import bpy

    if not objects:
        raise RuntimeError("没有可导出的对象（渲染网格和碰撞代理都缺失）")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=os.path.abspath(path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True,
    )
    print(f"[asset] GLB -> {path}（{len(objects)} 个对象）")
    return path


def _blender_mesh(name: str, vertices, faces, material=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(f) for f in faces])
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj


def _tube(name: str, start: Vector, end: Vector, radius: float, material):
    """两点之间的一段圆柱。

    为什么不用"只有棱的网格"：Cycles 里**没有面的边不参与渲染**，
    线框会完全看不见。所以凸包边用细圆柱画——可见、有厚度、不用改渲染设置。
    """
    direction = end - start
    length = direction.length
    if length <= 1e-9:
        return None
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length, vertices=12, location=(0.0, 0.0, 0.0)
    )
    obj = bpy.context.active_object
    obj.name = name
    # 默认圆柱沿 +Z；转到 direction 方向
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
    obj.location = (start + end) * 0.5
    obj.data.materials.append(material)
    return obj


def _cone(name: str, tip: Vector, direction: Vector, length: float, radius: float,
          material):
    bpy.ops.mesh.primitive_cone_add(
        radius1=radius, depth=length, vertices=16, location=(0.0, 0.0, 0.0)
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
    obj.location = tip - direction * (length * 0.5)
    obj.data.materials.append(material)
    return obj


def _label(name: str, text: str, location: Vector, camera, material, size: float):
    """朝向相机的文字标签。失败也不影响资产生成。"""
    try:
        bpy.ops.object.text_add(location=tuple(location))
        obj = bpy.context.active_object
        obj.name = name
        obj.data.body = text
        obj.data.size = size
        obj.data.align_x = "CENTER"
        obj.data.align_y = "CENTER"
        obj.data.materials.append(material)
        constraint = obj.constraints.new(type="TRACK_TO")
        constraint.target = camera
        constraint.track_axis = "TRACK_Z"
        constraint.up_axis = "UP_Y"
        return obj
    except Exception as exc:  # noqa: BLE001
        print(f"[asset] 文字标签 {name!r} 失败（不影响其余部分）：{exc}")
        return None


def build_visual(asset, asset_dir: str, *, show_hull: bool, show_axes: bool):
    """按 body 系搭出"网格 + 凸包线框 + 质心 + 主轴"。"""
    name = asset["name"]
    vertices = _body_vertices(asset, asset_dir)
    if not vertices:
        raise ValueError("资产里既没有可读的渲染网格，也没有碰撞点，无法出图")

    radius = max(v.length for v in vertices) or 1.0

    # --- 渲染网格 ---
    body_material = create_material(MaterialSpec(
        name="body", base_color=(0.34, 0.45, 0.66), roughness=0.50, metallic=0.0,
    ))
    # 半透明：主轴与质心标记在物体内部，不透明的话会被完全挡住，
    # 而"它们是否穿过质心"恰恰是这张 QA 图要核对的东西。
    # Cycles 直接认 Principled 的 Alpha 端口。
    _set_alpha(body_material, 0.40)
    loaded = _load_render_mesh(asset, asset_dir)
    if loaded is not None:
        source_vertices, source_faces = loaded
        transform = asset["frames"]["source_to_body"]
        translation = transform["translation"]
        rotation = Quaternion(transform["rotation_wxyz"])
        body = _blender_mesh(
            f"{name}_render",
            [_to_body(v, translation, rotation) for v in source_vertices],
            source_faces,
            body_material,
        )
        for polygon in body.data.polygons:
            polygon.use_smooth = False
        # 注意：f-string 的替换域里不能换行（Python 3.10 及以前），
        # 所以文件名先取出来再插值。
        mesh_file = os.path.basename(str(_resolve_mesh_path(asset, asset_dir)))
        print(
            f"[asset] 渲染网格：{len(source_vertices)} 顶点 / "
            f"{len(source_faces)} 面（读自 {mesh_file}）"
        )
    else:
        body = None
        print("[asset] 读不到渲染网格，只画碰撞代理")

    # --- 碰撞代理（线框） ---
    collision = asset.get("collision") or {}
    hull_objects = []
    if show_hull and collision.get("points"):
        hull_material = create_material(MaterialSpec(
            name="hull_wire", base_color=(1.0, 0.62, 0.10), roughness=0.5,
            emissive=0.7,
        ))
        points = [Vector(p) for p in collision["points"]]
        tube_radius = radius * 0.008
        for index, edge in enumerate(collision.get("edges") or []):
            piece = _tube(
                f"hull_edge_{index}", points[edge[0]], points[edge[1]],
                tube_radius, hull_material,
            )
            if piece is not None:
                hull_objects.append(piece)
        # 顶点小球，方便数面数
        for index, point in enumerate(points):
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=tube_radius * 2.2, segments=12, ring_count=8,
                location=tuple(point),
            )
            sphere = bpy.context.active_object
            sphere.name = f"hull_vertex_{index}"
            sphere.data.materials.append(hull_material)
            hull_objects.append(sphere)
        if hull_objects:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in hull_objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = hull_objects[0]
            bpy.ops.object.join()
            joined = bpy.context.active_object
            joined.name = f"{name}_collision"
            print(f"[asset] 碰撞代理：{len(points)} 顶点 / "
                  f"{len(collision.get('edges') or [])} 棱")

    # --- 质心 ---
    com_material = create_material(MaterialSpec(
        name="com", base_color=(1.0, 1.0, 1.0), roughness=0.3, emissive=1.2,
    ))
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius * 0.034, segments=16, ring_count=12, location=(0.0, 0.0, 0.0)
    )
    com = bpy.context.active_object
    com.name = "COM"
    com.data.materials.append(com_material)

    # --- 主轴 ---
    axis_objects = []
    if show_axes:
        for label, direction in (
            ("X", Vector((1.0, 0.0, 0.0))),
            ("Y", Vector((0.0, 1.0, 0.0))),
            ("Z", Vector((0.0, 0.0, 1.0))),
        ):
            material = create_material(MaterialSpec(
                name=f"axis_{label}", base_color=_AXIS_COLORS[label],
                roughness=0.45, emissive=0.45,
            ))
            # 必须显著长出物体轮廓，否则轴会被半透明网格糊住、看不见指向
            shaft_end = direction * (radius * 1.10)
            axis_objects.append(
                _tube(f"axis_{label}_shaft", Vector((0, 0, 0)), shaft_end,
                      radius * 0.009, material)
            )
            axis_objects.append(
                _cone(f"axis_{label}_head", direction * (radius * 1.28), direction,
                      radius * 0.13, radius * 0.030, material)
            )
        # 反向的短轴，表示主轴是"过质心的直线"而不是射线
        for label, direction in (
            ("X", Vector((-1.0, 0.0, 0.0))),
            ("Y", Vector((0.0, -1.0, 0.0))),
            ("Z", Vector((0.0, 0.0, -1.0))),
        ):
            material = create_material(MaterialSpec(
                name=f"axis_{label}_neg", base_color=_AXIS_COLORS[label[-1]],
                roughness=0.6, emissive=0.25,
            ))
            axis_objects.append(
                _tube(f"axis_{label}_neg_shaft", Vector((0, 0, 0)),
                      direction * (radius * 0.55), radius * 0.006, material)
            )

    return body, radius


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _parse_args(argv):
    parser = argparse.ArgumentParser(description="资产 → .blend + QA 预览图")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--blend", help=".blend 输出路径")
    parser.add_argument("--png", help="预览图输出路径")
    parser.add_argument(
        "--glb",
        help="可选：导出 GLB（可自由旋转观察；不能替代 PNG 作为 QA 证据）",
    )
    parser.add_argument("--resolution", default="1000x750")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--no-hull", action="store_true", help="不画碰撞代理")
    parser.add_argument("--no-axes", action="store_true", help="不画主轴")
    parser.add_argument("--no-labels", action="store_true", help="不画轴标签")
    return parser.parse_args(argv)


def main() -> int:
    raw = sys.argv
    argv = raw[raw.index("--") + 1:] if "--" in raw else []
    args = _parse_args(argv)

    with open(args.asset, "r", encoding="utf-8") as handle:
        asset = json.load(handle)
    if asset.get("schema") != "object_asset/1":
        print(f"[asset] 不认识的资产版本 {asset.get('schema')!r}", file=sys.stderr)
        return 1

    width, height = (int(v) for v in args.resolution.lower().split("x"))
    name = asset["name"]

    print(_LINE)
    print(f"资产可视化：{name}")
    print(_LINE)

    reset_scene()
    # 资产里的路径相对于它自己的目录，所以必须先把目录传下去
    asset_dir = os.path.dirname(os.path.abspath(args.asset))
    body, radius = build_visual(
        asset, asset_dir, show_hull=not args.no_hull, show_axes=not args.no_axes
    )

    # --- 相机：按包围半径自动取景 ---
    distance = radius * 3.9
    camera = create_camera(CameraSpec(
        position=(distance * 0.92, -distance * 1.15, distance * 0.72),
        target=(0.0, 0.0, 0.0),
        fov=38.0,
    ))

    if not args.no_labels:
        label_material = create_material(MaterialSpec(
            name="label", base_color=(1.0, 1.0, 1.0), roughness=0.9, emissive=1.6,
        ))
        for text, direction in (
            ("X", Vector((1.0, 0.0, 0.0))),
            ("Y", Vector((0.0, 1.0, 0.0))),
            ("Z", Vector((0.0, 0.0, 1.0))),
        ):
            _label(
                f"label_{text}", text, direction * (radius * 1.46),
                camera, label_material, radius * 0.10,
            )

    # --- 光 ---
    create_light(LightSpec(
        name="key", light_type="area", position=(radius * 2.2, -radius * 2.0, radius * 2.6),
        target=(0.0, 0.0, 0.0), energy=105.0 * max(radius, 0.3) ** 2, size=radius * 2.2,
    ))
    create_light(LightSpec(
        name="fill", light_type="area", position=(-radius * 2.6, -radius * 0.6, radius * 1.2),
        target=(0.0, 0.0, 0.0), energy=42.0 * max(radius, 0.3) ** 2, size=radius * 2.6,
        color=(0.82, 0.88, 1.0),
    ))

    # --- 地面：贴在物体最低点，body 原点仍在世界原点 ---
    # 地面要够大，让它的边缘落在画面外——否则那条对角线会横穿物体，
    # 看起来像是几何错误。物体贴地（地面在 -ground_offset），body 原点仍在世界原点。
    ground_offset = float(asset["engine"]["ground_z_offset"])
    ground = create_ground(GroundSpec(enabled=True, size=radius * 9.0))
    if ground is not None:
        ground.location.z = -ground_offset

    render_spec = RenderSpec(resolution=(width, height), samples=args.samples)
    configure_world(render_spec)
    configure_render(render_spec)

    # --- 把资产摘要存进 .blend，让文件自带出处 ---
    scene = bpy.context.scene
    summary = {
        "schema": asset["schema"],
        "name": name,
        "mass": asset["mass_properties"]["mass"],
        "volume": asset["mass_properties"]["volume"],
        "center_of_mass_source": asset["mass_properties"]["center_of_mass"],
        "principal_moments": asset["principal"]["moments"],
        "orientation_offset_wxyz": asset["engine"]["orientation_offset_wxyz"],
        "ground_z_offset": ground_offset,
        "coordinate_frame": "body (质心在原点、轴与惯量主轴重合)",
    }
    for key, value in summary.items():
        try:
            scene[key] = value
        except (TypeError, ValueError):
            scene[key] = str(value)

    if args.blend:
        os.makedirs(os.path.dirname(os.path.abspath(args.blend)), exist_ok=True)
        # 关掉自动备份：Blender 覆盖保存时会生成 .blend1，
        # 在 output/ 里就是一份无意义的 800 KB 垃圾（而且每次重跑多一份）。
        try:
            bpy.context.preferences.filepaths.save_version = 0
        except AttributeError:
            pass
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))
        print(f"[asset] .blend -> {args.blend}")

    if args.png:
        os.makedirs(os.path.dirname(os.path.abspath(args.png)), exist_ok=True)
        render_to_file(os.path.abspath(args.png))
        print(f"[asset] 预览图 -> {args.png}")

    if args.glb:
        # 放在最后导出：它只读场景，不影响渲染；反过来渲染会改相机/灯光，
        # 但那些本来也不导出（export_selection）。
        export_glb(args.glb, _collect_asset_objects(asset["name"]))

    print(_LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
