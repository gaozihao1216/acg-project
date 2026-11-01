# -*- coding: utf-8 -*-
"""把 :class:`SceneSpec` 翻译成 Blender 场景并渲染。

本模块**只能在 Blender 的 Python 环境里导入**（因为依赖 ``bpy``）。
它不包含任何物理逻辑，职责边界很清晰：

    SceneSpec  ──►  bpy 场景（几何/材质/灯光/相机）  ──►  PNG

为什么先做这一层
----------------
渲染是题目三的**红色核心要求**（必须产出精良视频），而 bpy 这条链路上
的坑非常零碎：坐标系朝向、材质节点名、面法线方向、相机视场角、
Cycles 采样与降噪、色彩管理。把这些问题在"还没有物理"的阶段一次性
解决掉，后面写求解器时就不会被渲染问题干扰。

版本适配说明
------------
目标环境是 Blender 3.6，但 Principled BSDF 的输入端口名在 4.x 有变化
（例如 ``Transmission`` → ``Transmission Weight``）。因此所有材质参数
都通过 ``_set_bsdf_input`` 按"候选名列表"设置，缺失的端口会被静默跳过，
保证脚本跨版本不会直接崩。
"""

from __future__ import annotations

import math
import os

import bpy
from mathutils import Vector

from src.geom.geometry import (
    BoxGeometry,
    CapsuleGeometry,
    CompoundGeometry,
    ConvexHullGeometry,
    CylinderGeometry,
    GeometrySpec,
    SphereGeometry,
)
from src.scene.scene_spec import (
    CameraSpec,
    GroundSpec,
    LightSpec,
    MaterialSpec,
    ObjectSpec,
    RenderSpec,
    SceneSpec,
)

#: 色彩管理使用的视图变换。'Filmic' 高光过渡柔和、不易死白；
#: 若想让画面更"硬"、颜色更饱和，可改成 'Standard'。
VIEW_TRANSFORM = "Filmic"

#: 球形与圆柱的细分段数。段数太低会出现明显棱边；
#: 64 段在 1080p 下已看不出多边形。
SPHERE_SEGMENTS = 64
SPHERE_RINGS = 32
CYLINDER_VERTICES = 64


# ---------------------------------------------------------------------------
# 场景初始化
# ---------------------------------------------------------------------------

def reset_scene() -> None:
    """清空当前 Blender 场景，恢复出厂设置。

    使用 ``use_empty=True`` 可以连默认的立方体、灯光、相机一起清掉，
    避免出现"画面里莫名多一个方块"的情况。

    批量渲染时每个场景前都要调用，否则上一次的材质与物体名会残留，
    导致 ``bpy.data.materials.new`` 生成 ``mat.001`` 这类名字。
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


# ---------------------------------------------------------------------------
# 材质
# ---------------------------------------------------------------------------

def _set_bsdf_input(bsdf, candidates: list[str], value) -> bool:
    """按候选名列表给 Principled BSDF 的某个输入端口赋值。

    Blender 4.x 重命名了若干端口（如 ``Transmission`` →
    ``Transmission Weight``），此函数按顺序尝试，命中即返回。

    参数
    ----
    bsdf       : Principled BSDF 节点
    candidates : 端口名候选列表，按优先级排列
    value      : 要写入的值

    返回
    ----
    bool : 是否成功写入（端口全都不存在时返回 False）
    """
    for name in candidates:
        port = bsdf.inputs.get(name)
        if port is not None:
            port.default_value = value
            return True
    return False


def create_material(spec: MaterialSpec) -> bpy.types.Material:
    """根据 :class:`MaterialSpec` 创建 Principled BSDF 材质。

    参数
    ----
    spec : 材质描述（基础色、粗糙度、金属度、透射、自发光）

    返回
    ----
    bpy.types.Material : 可直接挂到物体上的材质
    """
    mat = bpy.data.materials.new(name=spec.name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        # 极端情况下节点树结构不同，此时返回一个默认材质即可，
        # 不让整个渲染流程因为材质失败而中断。
        return mat

    r, g, b = spec.base_color
    _set_bsdf_input(bsdf, ["Base Color"], (r, g, b, 1.0))
    _set_bsdf_input(bsdf, ["Roughness"], spec.roughness)
    _set_bsdf_input(bsdf, ["Metallic"], spec.metallic)
    _set_bsdf_input(bsdf, ["Transmission", "Transmission Weight"], spec.transmission)

    # 透射材质需要提高折射率才像玻璃；纯金属不需要
    if spec.transmission > 0.0:
        _set_bsdf_input(bsdf, ["IOR"], 1.45)

    if spec.emissive > 0.0:
        _set_bsdf_input(bsdf, ["Emission Color", "Emission"], (r, g, b, 1.0))
        _set_bsdf_input(bsdf, ["Emission Strength"], spec.emissive)

    return mat


# ---------------------------------------------------------------------------
# 几何体
# ---------------------------------------------------------------------------

def _shade_smooth(obj: bpy.types.Object) -> None:
    """对曲面物体启用平滑着色，消除多边形棱边。

    长方体不调用此函数——平滑后棱角会被"磨圆"，反而不像刚体。
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()


def _create_box(geom: BoxGeometry) -> bpy.types.Object:
    """创建长方体。

    ``primitive_cube_add(size=1)`` 生成边长为 1、中心在原点的立方体，
    因此直接把 ``scale`` 设成三边全长即可，不必修改顶点数据。
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.scale = geom.size
    return obj


def _create_sphere(geom: SphereGeometry) -> bpy.types.Object:
    """创建 UV 球。"""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=geom.radius,
        segments=SPHERE_SEGMENTS,
        ring_count=SPHERE_RINGS,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    _shade_smooth(obj)
    return obj


def _create_cylinder(geom: CylinderGeometry) -> bpy.types.Object:
    """创建圆柱。Blender 的圆柱轴线沿局部 Z 轴，与项目约定一致。"""
    bpy.ops.mesh.primitive_cylinder_add(
        radius=geom.radius,
        depth=geom.height,
        vertices=CYLINDER_VERTICES,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    _shade_smooth(obj)
    return obj


def _create_capsule(geom: CapsuleGeometry) -> bpy.types.Object:
    """创建胶囊体。

    做法：中间一段圆柱 + 两端各一个球，最后 ``join`` 成一个物体。
    Blender 没有内置胶囊图元，这是最省事且几何完全正确的拼法
    （两端的球各有一半露在外面，正好对应两个半球）。
    """
    parts: list[bpy.types.Object] = []
    half_h = 0.5 * geom.height

    if geom.height > 1e-9:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=geom.radius,
            depth=geom.height,
            vertices=CYLINDER_VERTICES,
            location=(0.0, 0.0, 0.0),
        )
        parts.append(bpy.context.active_object)

    for sign in (1.0, -1.0):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=geom.radius,
            segments=SPHERE_SEGMENTS,
            ring_count=SPHERE_RINGS,
            location=(0.0, 0.0, sign * half_h),
        )
        parts.append(bpy.context.active_object)

    # 合并：先全选这几个部件，以第一个为活动物体执行 join
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object
    _shade_smooth(obj)
    return obj


def _create_hull(geom: ConvexHullGeometry) -> bpy.types.Object:
    """由凸包的顶点表与多边形面表直接建网格。

    这里**必须**使用 ``geom.vertices`` / ``geom.faces``，而不是场景里的原始点集：
    凸包顶点已经在构造时被平移（质心到原点）并旋转（对齐主轴），
    而物体的姿态里同样并入了这个主轴旋转。两者是配套的——
    如果这里用原始点集，渲染出来的形状会和碰撞形状错开。

    多边形面用 Blender 的 ``from_pydata`` 直接支持，不需要三角化，
    这样法向与平面着色都更干净（共面的三角形会被 Blender 重新计算成同一法向，
    但保留多边形能避免内部棱边影响平滑着色的判断）。
    """
    vertices = [tuple(v) for v in geom.vertices]
    faces = [tuple(face) for face in geom.faces]

    mesh = bpy.data.meshes.new("hull")
    mesh.from_pydata(vertices, [], faces)
    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new("hull", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _create_compound(geom: CompoundGeometry) -> bpy.types.Object:
    """组合体：逐部件建网格 → 按部件位姿摆好 → join 成一个物体。

    必须用 ``geom.parts`` 而不是场景里的原始部件位姿：构造复合几何时，
    整体已经被平移到质心、旋转到惯量主轴系，部件的局部位姿也一并换了坐标系。
    用原始位姿渲染，形状会和碰撞形状整体错开（同一个坑见 ``_create_hull``）。
    """
    pieces: list[bpy.types.Object] = []
    for part in geom.parts:
        child = _create_mesh(part.geometry)
        child.rotation_mode = "QUATERNION"
        child.rotation_quaternion = part.orientation
        child.location = part.position
        pieces.append(child)

    bpy.ops.object.select_all(action="DESELECT")
    for piece in pieces:
        piece.select_set(True)
    bpy.context.view_layer.objects.active = pieces[0]
    if len(pieces) > 1:
        bpy.ops.object.join()

    obj = bpy.context.active_object
    _shade_smooth(obj)
    return obj


def _create_mesh(geom: GeometrySpec) -> bpy.types.Object:
    """按几何类型分派到对应的创建函数。

    异常
    ----
    TypeError : 几何类型尚未支持渲染
    """
    if isinstance(geom, BoxGeometry):
        return _create_box(geom)
    if isinstance(geom, SphereGeometry):
        return _create_sphere(geom)
    if isinstance(geom, CylinderGeometry):
        return _create_cylinder(geom)
    if isinstance(geom, CapsuleGeometry):
        return _create_capsule(geom)
    if isinstance(geom, ConvexHullGeometry):
        return _create_hull(geom)
    if isinstance(geom, CompoundGeometry):
        return _create_compound(geom)
    raise TypeError(f"几何类型 {type(geom).__name__} 尚未实现 Blender 渲染")


def create_object(spec: ObjectSpec) -> bpy.types.Object:
    """创建场景物体：几何 + 位姿 + 材质。

    参数
    ----
    spec : 物体描述

    返回
    ----
    bpy.types.Object : 已设置好变换与材质的 Blender 物体
    """
    obj = _create_mesh(spec.geometry)
    obj.name = spec.name

    # 位姿：Blender 的四元数分量顺序也是 (w, x, y, z)，可直接赋值
    tr = spec.transform
    obj.location = Vector(tr.position)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = tr.orientation
    obj.scale = (
        obj.scale[0] * tr.scale[0],
        obj.scale[1] * tr.scale[1],
        obj.scale[2] * tr.scale[2],
    )

    obj.data.materials.append(create_material(spec.material))
    return obj


# ---------------------------------------------------------------------------
# 地面 / 相机 / 灯光 / 世界
# ---------------------------------------------------------------------------

def create_ground(spec: GroundSpec) -> bpy.types.Object | None:
    """创建地面平面。

    地面只提供视觉参考与阴影承托，**不参与物理**（W1 阶段尚无物理）。
    W2 起会另建一块带碰撞的静态平面，二者可以共用同一个外观。
    """
    if not spec.enabled:
        return None

    bpy.ops.mesh.primitive_plane_add(size=spec.size, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = "Ground"

    mat = create_material(
        MaterialSpec(
            name="GroundMaterial",
            base_color=spec.color,
            roughness=spec.roughness,
            metallic=0.0,
        )
    )
    obj.data.materials.append(mat)
    return obj


def create_camera(spec: CameraSpec) -> bpy.types.Object:
    """创建透视相机并设为场景当前相机。

    视场角约定：``spec.fov`` 是**水平视场角**（度）。Blender 的
    ``camera.angle`` 在横屏（宽 > 高）时作用在水平方向，因此直接赋值即可；
    若把分辨率改成竖屏，它会自动改为作用在竖直方向。
    """
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.angle = math.radians(spec.fov)

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.location = Vector(spec.position)
    direction = Vector(spec.target) - Vector(spec.position)
    if direction.length < 1e-9:
        raise ValueError("相机位置与注视点重合，无法确定朝向")

    # 让相机的 -Z 轴指向目标，+Y 作为上方向
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    bpy.context.scene.camera = cam_obj
    return cam_obj


def create_light(spec: LightSpec) -> bpy.types.Object:
    """创建光源。

    三种类型的参数语义不同，这里统一映射：

    - ``area``  : ``energy`` 是瓦特，``size`` 是正方形发光面的边长（米）。
                  面积越大阴影越软，是控制画面调性的主要手段。
    - ``sun``   : ``energy`` 是辐照度（W/m²），``size`` 被解释为
                  **角直径（度）**，用于控制阴影的软硬。
    - ``point`` : ``energy`` 是瓦特，``size`` 用作光源半径（越大越软）。
    """
    light_data = bpy.data.lights.new(name=spec.name, type=spec.light_type.upper())

    if spec.light_type == "sun":
        light_data.energy = spec.energy
        # 太阳光的软硬由角直径决定，把它限制在合理范围内
        light_data.angle = math.radians(max(0.01, min(spec.size, 30.0)))
    elif spec.light_type == "point":
        light_data.energy = spec.energy
        light_data.shadow_soft_size = spec.size
    else:  # area
        light_data.energy = spec.energy
        light_data.shape = "SQUARE"
        light_data.size = spec.size

    light_data.color = spec.color

    light_obj = bpy.data.objects.new(spec.name, light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    light_obj.location = Vector(spec.position)
    direction = Vector(spec.target) - Vector(spec.position)
    if direction.length < 1e-9:
        raise ValueError(f"光源 '{spec.name}' 位置与照射目标重合")
    light_obj.rotation_mode = "QUATERNION"
    light_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    return light_obj


def configure_world(render: RenderSpec) -> None:
    """设置环境光（世界背景）。

    环境光决定了画面的"底光"。设得太低会让暗部死黑、金属没有反射内容；
    设为中性深灰（默认 0.05 强度 0.35）通常能得到干净的棚拍观感。
    """
    world = bpy.data.worlds.new(name="World")
    bpy.context.scene.world = world
    world.use_nodes = True

    background = world.node_tree.nodes.get("Background")
    if background is None:
        return

    r, g, b = render.world_color
    background.inputs["Color"].default_value = (r, g, b, 1.0)
    background.inputs["Strength"].default_value = render.world_strength


# ---------------------------------------------------------------------------
# 渲染设置
# ---------------------------------------------------------------------------

def configure_render(render: RenderSpec) -> None:
    """应用分辨率、采样、引擎、曝光等渲染参数。

    参数
    ----
    render : 渲染设置。注意 ``engine`` 为 'BLENDER_EEVEE' 时，
             采样数的含义是 TAA 采样，与 Cycles 的路径采样不同。
    """
    scene = bpy.context.scene

    scene.render.engine = render.engine
    scene.render.resolution_x = render.resolution[0]
    scene.render.resolution_y = render.resolution[1]
    scene.render.resolution_percentage = 100

    # 输出格式固定为 PNG，便于后续 ffmpeg 合成与逐张对比
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = render.film_transparent

    # 尽量吃满 CPU 线程；批量渲染时这一项影响很大
    scene.render.threads_mode = "AUTO"

    if render.engine == "CYCLES":
        scene.cycles.samples = render.samples
        scene.cycles.preview_samples = min(16, render.samples)
        scene.cycles.use_denoising = render.denoise
        scene.cycles.device = "CPU"
        # 限制反弹次数：本项目的场景以基础几何为主，多次反弹收获很小
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 3
        scene.cycles.transmission_bounces = 6
    else:
        scene.eevee.taa_render_samples = render.samples

    # 色彩管理与曝光
    scene.view_settings.view_transform = VIEW_TRANSFORM
    scene.view_settings.look = "None"
    scene.view_settings.exposure = render.exposure


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def build_scene(scene_spec: SceneSpec) -> None:
    """按 :class:`SceneSpec` 完整搭建一个 Blender 场景。

    调用顺序有讲究：先清空 → 再配置渲染与世界 → 最后建物体。
    因为灯光与相机的默认参数会受到场景单位与色彩管理的影响。

    参数
    ----
    scene_spec : 已解析并校验过的场景描述
    """
    reset_scene()

    configure_render(scene_spec.render)
    configure_world(scene_spec.render)

    create_ground(scene_spec.ground)
    create_camera(scene_spec.camera)
    for light in scene_spec.lights:
        create_light(light)
    for obj_spec in scene_spec.objects:
        create_object(obj_spec)


def render_to_file(output_path: str) -> None:
    """把当前场景渲染并写入指定 PNG 文件。

    参数
    ----
    output_path : 输出文件路径；父目录不存在时会自动创建
    """
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    bpy.context.scene.render.filepath = os.path.abspath(output_path)
    bpy.ops.render.render(write_still=True)
