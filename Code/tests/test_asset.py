# -*- coding: utf-8 -*-
"""资产规格测试：``object_asset/1`` 的数值必须自洽。

这组测试针对的是一类特别隐蔽的错误
------------------------------------
资产里同时存在**两个坐标系**和**两个四元数**。任何一处混用，出来的都是
"能读、能跑、数字全错"的文件：

- 质量属性报在 ``source`` 系（质心、非对角项才有意义）
- 碰撞点报在 ``body`` 系（质心在原点、轴与主轴重合）
- ``source_to_body.rotation`` = Qᵀ，``engine.orientation_offset`` = Q

所以每条都要有**独立可算**的期望值，而不是"能生成就算过"：

1. ``Q·diag(Λ)·Qᵀ`` 必须还原原张量 —— 主轴与主惯量的定义本身
2. 两个四元数必须互为共轭 —— 专防上面那个混用
3. 凸包点的体积（在 body 系算）必须等于资产里报的体积 —— 证明坐标没被重复变换
4. 资产派生的场景，物体最低点必须恰好落在 z=0 —— 证明贴地高度是对的
5. 同样的输入必须产出**逐字节相同**的 JSON —— 资产要能进版本库

运行::

    python Code/tests/test_asset.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _CODE_ROOT)
sys.path.insert(0, os.path.join(_CODE_ROOT, "scripts"))

from src.geom.asset import (
    SCHEMA,
    asset_to_object_spec,
    build_asset,
    load_asset,
    save_asset,
)
from src.geom.massprops import from_mesh
from src.geom.meshio import Mesh, load_mesh, save_obj
from src.geom.meshutil import box_mesh, prism_mesh
from src.mathutils import quaternion as quat

TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.4, 1.1)]
HEIGHT = 0.6
MASS = 3.0
PRISM_OBJ = os.path.join(_CODE_ROOT, "assets", "prism.obj")


def _prism_asset():
    mesh = load_mesh(PRISM_OBJ)
    return build_asset(name="prism", mesh=mesh, mass=MASS, source_path=PRISM_OBJ)


def _det(matrix) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


# ---------------------------------------------------------------------------
# 组一：数值自洽
# ---------------------------------------------------------------------------

def test_tensor_reconstruction() -> tuple[bool, str]:
    """``Q·diag(Λ)·Qᵀ`` 必须还原资产里的完整张量。

    这是"主轴/主惯量"这组数字的**定义**，不是近似。若不成立，说明
    ``principal`` 与 ``inertia_com`` 来自不同的计算，消费者只能信一个。
    """
    asset = _prism_asset()
    tensor = asset["mass_properties"]["inertia_com"]
    moments = asset["principal"]["moments"]
    axes = asset["principal"]["axes_columns"]

    worst = 0.0
    for row in range(3):
        for col in range(3):
            value = sum(axes[row][k] * moments[k] * axes[col][k] for k in range(3))
            worst = max(worst, abs(value - tensor[row][col]))

    scale = max(abs(tensor[i][i]) for i in range(3))
    ok = worst / scale < 1e-12
    return ok, f"max|QΛQᵀ − I| / 最大对角元 = {worst / scale:.2e}"


def test_quaternions_are_conjugate() -> tuple[bool, str]:
    """``source_to_body.rotation`` 与 ``orientation_offset`` 必须互为共轭。

    这两个是**相反方向**的同一个旋转：一个把点转进 body 系（Qᵀ），
    一个并进物体姿态（Q）。我在凸包那一轮就栽在这里——顶点位置全对，
    世界逆惯量错到 7.24，因为两处用了同一个。
    """
    asset = _prism_asset()
    to_body = asset["frames"]["source_to_body"]["rotation_wxyz"]
    offset = asset["engine"]["orientation_offset_wxyz"]

    expected = (offset[0], -offset[1], -offset[2], -offset[3])
    worst = max(abs(to_body[i] - expected[i]) for i in range(4))

    # 再验一次：两者合成的旋转应为单位阵
    composed = quat.quat_multiply(offset, to_body)
    identity_error = max(
        abs(abs(composed[0]) - 1.0),
        abs(composed[1]), abs(composed[2]), abs(composed[3]),
    )

    ok = worst < 1e-15 and identity_error < 1e-12
    return ok, (
        f"与共轭的最大差={worst:.2e}，两四元数合成后与单位元的差={identity_error:.2e}"
    )


def test_axes_orthonormal() -> tuple[bool, str]:
    """主轴必须正交且 ``det = +1``。

    特征向量的**符号是任意的**，Jacobi 迭代约一半概率给出左手系（det = −1）。
    那对"重建张量"毫无影响，但它是**反射不是旋转**，转四元数就崩。
    """
    asset = _prism_asset()
    axes = asset["principal"]["axes_columns"]

    worst = 0.0
    for i in range(3):
        for j in range(3):
            dot = sum(axes[k][i] * axes[k][j] for k in range(3))
            worst = max(worst, abs(dot - (1.0 if i == j else 0.0)))
    determinant = _det(axes)

    ok = worst < 1e-12 and abs(determinant - 1.0) < 1e-12
    return ok, f"正交误差={worst:.2e}，det={determinant:+.12f}（须为 +1）"


def test_collision_points_in_body_frame() -> tuple[bool, str]:
    """用资产里的凸包点（body 系）重算体积，必须等于资产报的体积。

    这条专门抓"坐标被变换了两次"或"漏了一次"：如果碰撞点不在 body 系，
    散度定理算出来的体积会明显偏离本体体积。
    """
    asset = _prism_asset()
    points = asset["collision"]["points"]
    faces = asset["collision"]["faces"]

    # 凸包的侧面是**四边形**，只取前三个顶点会漏掉一半三角形
    # （实测漏掉 0.11 = 体积的 1/3）。必须按扇形三角化。
    tensor_volume = 0.0
    for face in faces:
        a = points[face[0]]
        for k in range(1, len(face) - 1):
            b = points[face[k]]
            c = points[face[k + 1]]
            tensor_volume += (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            ) / 6.0

    reported = asset["collision"]["volume"]
    error = abs(abs(tensor_volume) / reported - 1.0)
    body_volume = asset["mass_properties"]["volume"]

    ok = error < 1e-12 and abs(reported / body_volume - 1.0) < 1e-12
    return ok, (
        f"由凸包点重算体积={abs(tensor_volume):.12f}，资产报={reported:.12f}"
        f"（相对差={error:.2e}）；本体体积={body_volume:.12f}"
        f"（凸包/本体={reported / body_volume:.6f}）"
    )


def test_ground_offset_exact() -> tuple[bool, str]:
    """``ground_z_offset`` 必须等于 body 系最低点的相反数。

    引擎用它把物体放到地面上。差一点点就会被初始穿透校验拒绝，
    差多了物体就从空中掉下来。
    """
    asset = _prism_asset()
    lowest = min(p[2] for p in asset["collision"]["points"])
    ground = asset["engine"]["ground_z_offset"]
    error = abs(lowest + ground)

    ok = error < 1e-12
    return ok, (
        f"body 最低点 z={lowest:+.12f}，ground_z_offset={ground:+.12f}，"
        f"两者之和={lowest + ground:.2e}（应为 0）"
    )


# ---------------------------------------------------------------------------
# 组二：能被消费者正确使用
# ---------------------------------------------------------------------------

def test_object_spec_roundtrip_to_engine() -> tuple[bool, str]:
    """资产 → 场景 → 物理世界：主惯量必须与资产里的一致，且落地后静止。

    这是整个输出体系的验收：只验证"JSON 写出来了"是不够的，
    必须让引擎真的读进去、跑起来、结果与资产对得上。
    """
    from analyze_object import emit_scene
    from src.physics.world import PhysicsWorld
    from src.scene.scene_spec import load_scene

    asset = _prism_asset()
    expected_moments = asset["principal"]["moments"]

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        scene_path = os.path.join(directory, "scene.json")
        emit_scene(scene_path, asset)
        world = PhysicsWorld.from_scene(load_scene(scene_path))
        body = world.bodies[0]

        inertia_error = max(
            abs(body.inertia_local[i] / expected_moments[i] - 1.0) for i in range(3)
        )

        for _ in range(240 * 3):
            world.step()

        lowest = min(
            quat.quat_rotate_vector(body.orientation, v)[2]
            for v in body.geometry.vertices
        ) + body.position[2]

    ok = inertia_error < 1e-9 and body.speed() < 0.01 and abs(lowest) < 0.002
    return ok, (
        f"主惯量与资产的最大相对差={inertia_error:.2e}；"
        f"3 秒后 |v|={body.speed():.5f}，最低点 z={lowest * 1000:+.3f} mm"
    )


def test_spec_matches_body_frame_vertices() -> tuple[bool, str]:
    """``asset_to_object_spec`` 的碰撞点必须**未经二次变换**。

    凸包几何在构造时已经归一化到 body 系，资产又原样存下来。
    如果这里再套一次 ``source_to_body``，点会被平移旋转第二次，
    物体形状就毁了——而且仍然"能加载"。
    """
    asset = _prism_asset()
    spec = asset_to_object_spec(asset)

    stored = asset["collision"]["points"]
    produced = spec["geometry"]["points"]
    count_ok = len(stored) == len(produced)
    worst = 0.0
    if count_ok:
        worst = max(
            abs(stored[i][k] - produced[i][k])
            for i in range(len(stored))
            for k in range(3)
        )

    ok = count_ok and worst < 1e-15
    return ok, (
        f"点数 {len(stored)}→{len(produced)}，逐坐标最大差={worst:.2e}"
        f"（应为 0：body 系点不需要再变换）"
    )


def test_point_cloud_has_no_collision() -> tuple[bool, str]:
    """点云资产必须**明确没有碰撞代理**，且转场景物体时报错而不是猜。

    点云没有表面，就没有法向与穿透深度。给它编一个碰撞代理是最坏的做法——
    仿真会跑，但接触是假的。
    """
    from src.geom.meshutil import random_points_in_box
    import random

    rng = random.Random(3)
    points = random_points_in_box((1.0, 1.0, 1.0), 500, rng)
    asset = build_asset(name="cloud", points=points, mass=2.0)

    collision_none = asset["collision"] is None
    volume_check = next(
        c for c in asset["checks"] if c["name"] == "volume_positive"
    )
    rejected = False
    try:
        asset_to_object_spec(asset)
    except ValueError:
        rejected = True

    ok = collision_none and (not volume_check["ok"]) and rejected
    return ok, (
        f"collision=None={collision_none}，体积检查未通过={not volume_check['ok']}，"
        f"转场景物体被拒绝={rejected}"
    )


# ---------------------------------------------------------------------------
# 组三：文件层面
# ---------------------------------------------------------------------------

def test_json_deterministic() -> tuple[bool, str]:
    """同样的输入必须产出**逐字节相同**的 JSON。

    资产要进版本库。时间戳会让每次生成都"有变化"，于是 diff 全是噪声，
    真正改了什么反而看不见。
    """
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        first = os.path.join(directory, "a.json")
        second = os.path.join(directory, "b.json")
        save_asset(first, _prism_asset())
        save_asset(second, _prism_asset())
        with open(first, "rb") as handle:
            bytes_a = handle.read()
        with open(second, "rb") as handle:
            bytes_b = handle.read()

    ok = bytes_a == bytes_b and b"20" not in bytes_a[:200]
    return ok, (
        f"两次生成字节数 {len(bytes_a)} / {len(bytes_b)}，"
        f"完全相同={bytes_a == bytes_b}（无时间戳）"
    )


def test_json_roundtrip() -> tuple[bool, str]:
    """写盘再读回，所有数字必须逐位相同（JSON 是双精度往返安全的）。"""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "prism.asset.json")
        original = _prism_asset()
        save_asset(path, original)
        restored = load_asset(path)

    worst = 0.0
    for key in ("moments",):
        for a, b in zip(original["principal"][key], restored["principal"][key]):
            worst = max(worst, abs(a - b))
    for a, b in zip(
        original["engine"]["orientation_offset_wxyz"],
        restored["engine"]["orientation_offset_wxyz"],
    ):
        worst = max(worst, abs(a - b))

    ok = worst == 0.0 and restored["schema"] == SCHEMA
    return ok, f"读回后最大差={worst:.2e}（应为 0），schema={restored['schema']!r}"


def test_empty_error_on_bad_schema() -> tuple[bool, str]:
    """不认识的版本必须被拒绝，而不是"尽力解析"。"""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "object_asset/99", "name": "x"}, handle)
        rejected = False
        try:
            load_asset(path)
        except ValueError:
            rejected = True

    return rejected, f"object_asset/99 被拒绝={rejected}"


def test_case_layout() -> tuple[bool, str]:
    """案例目录约定：``root/<name>/input`` 与 ``root/<name>/output``。

    按物体分目录而不是按文件类型分，是因为"删掉一个物体"应当只删一个目录；
    再分 input/output 两层，是为了把"进来的"和"算出去的"隔开。
    """
    from src.geom.asset import case_paths

    layout = case_paths("root", "bunny")
    expected = {
        "input": os.path.join("root", "bunny", "input"),
        "output": os.path.join("root", "bunny", "output"),
        "asset_json": os.path.join("root", "bunny", "output", "bunny.asset.json"),
        "blend": os.path.join("root", "bunny", "output", "bunny.blend"),
        "preview": os.path.join("root", "bunny", "output", "bunny.preview.png"),
    }
    wrong = [key for key, value in expected.items() if layout[key] != value]

    ok = not wrong
    return ok, (
        f"input={layout['input']}，output={layout['output']}；"
        + ("全部符合约定" if ok else f"不符合：{wrong}")
    )


def test_stage_input_selfcontained() -> tuple[bool, str]:
    """源文件必须被复制进 ``input/``，且重复调用不报错、不产生副本。

    为什么复制而不是移动：原文件可能被别的东西引用，搬走会打断它们。
    为什么要复制：案例目录要能独立带走——``source.path`` 写绝对路径的话，
    换台机器就指不到了，而 ``sha256`` 也就失去了校验对象。
    """
    import tempfile

    from src.geom.asset import stage_input

    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "part.obj")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

        input_dir = os.path.join(directory, "case", "input")
        first = stage_input(source, input_dir)
        second = stage_input(source, input_dir)

        copied = os.path.isfile(first)
        same_name = os.path.basename(first) == "part.obj"
        same_dir = os.path.dirname(os.path.abspath(first)) == os.path.abspath(input_dir)
        idempotent = first == second
        with open(first, "r", encoding="utf-8") as handle:
            content_ok = handle.read() == "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        no_duplicate = len(os.listdir(input_dir)) == 1

    ok = copied and same_name and same_dir and idempotent and content_ok and no_duplicate
    return ok, (
        f"复制到 input/={copied}，同名={same_name}，重复调用返回同一路径={idempotent}，"
        f"内容一致={content_ok}，目录内只有 1 个文件={no_duplicate}"
    )


def test_mass_distribution_reaches_engine() -> tuple[bool, str]:
    """空心物体的**真实惯量**必须传到引擎，而不是被碰撞几何覆盖。

    这是整个"双表示"设计的意义所在，也是一个真实缺陷的回归测试：

    曾经 ``RigidBody.from_object_spec`` 无条件调用 ``compute_inertia(geometry)``
    ——只从**碰撞几何**反推惯量。对实心凸体没问题，但空心盒的碰撞代理是
    实心外包盒，从它反推得到真实惯量的 **1.25 倍**（实测相对差 24.8%）。
    资产里明明算准了，却在最后一步被丢掉，而且提高精度也不会消失。

    修法是把主惯量作为显式字段交给引擎。这条测试要求**逐位一致**。
    """
    from analyze_object import emit_scene
    from src.physics.world import PhysicsWorld
    from src.scene.scene_spec import load_scene
    from src.geom.massprops import from_mesh
    from src.geom.meshutil import shell_mesh

    vertices, faces = shell_mesh((1.0, 1.0, 1.0), (0.76, 0.76, 0.76))
    mesh = Mesh(vertices=vertices, faces=faces, source="shell").diagnose()
    asset = build_asset(name="shell", mesh=mesh, mass=2.4)

    # 先确认这个形状确实"碰撞代理 ≠ 质量分布"，否则这条测试测不到东西
    solid = from_mesh(vertices, faces, mass=2.4)
    fill = asset["collision"]["hull_fill_ratio"]

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        scene_path = os.path.join(directory, "scene.json")
        emit_scene(scene_path, asset)
        world = PhysicsWorld.from_scene(load_scene(scene_path))
        body = world.bodies[0]

    expected = asset["principal"]["moments"]
    worst = max(abs(body.inertia_local[i] / expected[i] - 1.0) for i in range(3))

    ok = worst < 1e-12 and fill > 1.5
    return ok, (
        f"凸包/本体 = {fill:.3f}（确属空心，从几何反推会偏大）；"
        f"引擎惯量与资产主惯量的最大相对差 = {worst:.2e}"
    )


def test_request_records_irreproducible_inputs() -> tuple[bool, str]:
    """``input/request.json`` 必须记下**产物反推不回来**的输入。

    典型的是随机点云的种子：点云文件能存，但"它是怎么生成的"存不下来；
    而同一份产物既可能来自 3 kg 也可能来自密度 1200，产物里只有一个质量数。
    """
    from src.geom.asset import build_request

    request = build_request(
        name="cloud",
        command=["analyze_object.py", "--points", "random:8000:7:1,1,1", "--mass", "3.0"],
        mass=3.0,
        points={"mode": "random", "count": 8000, "seed": 7, "bounds": [1.0, 1.0, 1.0]},
        max_faces=64,
    )

    has_seed = request["inputs"]["points"]["seed"] == 7
    mass_not_density = (
        request["mass_spec"]["mass"] == 3.0
        and request["mass_spec"]["density"] is None
    )
    replayable = "random:8000:7:1,1,1" in request["command"]
    has_limit = request["geometry_spec"]["max_faces"] == 64

    ok = has_seed and mass_not_density and replayable and has_limit
    return ok, (
        f"种子={request['inputs']['points']['seed']}，质量/密度区分={mass_not_density}，"
        f"命令可重放={replayable}，凸包上限={has_limit}"
    )


def test_scene_asset_stores_references_only() -> tuple[bool, str]:
    """场景资产只能存引用，不能把物体级数据抄进来。

    复制会带来三个具体问题：占空间、**会分叉**（物体改了场景不跟着变）、
    出错时不知道该信哪一份。所以这里逐项检查场景里没有物体级字段，
    并验证解析后能拿到**与物体资产逐位相同**的惯量。
    """
    import tempfile

    from src.geom.asset import case_paths
    from src.geom.scene_asset import (
        build_scene_asset,
        load_scene_asset,
        save_scene_asset,
        scene_asset_to_objects,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = os.path.join(directory, "cases")
        assets = {}
        for label, source in (("prism", "prism.obj"), ("shell", "shell.obj")):
            layout = case_paths(root, label)
            os.makedirs(layout["output"], exist_ok=True)
            mesh = load_mesh(os.path.join(_CODE_ROOT, "assets", source))
            asset = build_asset(
                name=label,
                mesh=mesh,
                mass=2.0,
                source_path=os.path.join(_CODE_ROOT, "assets", source),
                path_base=layout["output"],
            )
            save_asset(layout["asset_json"], asset)
            assets[label] = asset

        scene_layout = case_paths(root, "stack")
        os.makedirs(scene_layout["output"], exist_ok=True)
        scene = build_scene_asset(
            name="stack",
            objects=[
                {"name": "shell",
                 "asset": case_paths(root, "shell")["asset_json"],
                 "transform": {"position": [0.0, 0.0, 0.5]}},
                {"name": "prism",
                 "asset": case_paths(root, "prism")["asset_json"],
                 "transform": {"position": [0.0, 0.0, 1.35]}},
            ],
            path_base=scene_layout["output"],
        )
        scene_path = os.path.join(scene_layout["output"], "stack.scene.json")
        save_scene_asset(scene_path, scene)

        # ① 场景条目里不许出现物体级字段
        forbidden = ("mass_properties", "principal", "collision", "geometry", "mass")
        leaked = [
            f"{entry['name']}:{key}"
            for entry in scene["objects"]
            for key in forbidden
            if key in entry
        ]

        # ② 引用必须是相对路径（否则整个 results/ 搬走就断）
        relative = all(
            not os.path.isabs(entry["asset"]) for entry in scene["objects"]
        )

        # ③ 解析后惯量与物体资产逐位一致
        loaded = load_scene_asset(scene_path)
        specs = scene_asset_to_objects(loaded)
        worst = 0.0
        for spec, item in zip(specs, loaded["resolved"]):
            expected = item["asset"]["principal"]["moments"]
            for axis in range(3):
                worst = max(
                    worst, abs(spec["inertia"][axis] / expected[axis] - 1.0)
                )

        size = os.path.getsize(scene_path)

    ok = not leaked and relative and worst < 1e-15 and size < 4096
    return ok, (
        f"场景 JSON 仅 {size} 字节（不含任何物体几何）；"
        f"泄漏的物体级字段={leaked or '无'}，引用为相对路径={relative}，"
        f"解析后惯量最大相对差={worst:.2e}"
    )


def test_material_channel_count() -> tuple[bool, str]:
    """资产的材质是 **RGBA**，场景规格要 **RGB**——转换必须显式做。

    这不是理论问题：直接把 4 个分量传下去，``MaterialSpec`` 会以
    ``字段 'mat_0.base_color' 需要 3 个分量，实际收到 4 个`` 抛错。
    两套格式（Blender 用 alpha，物理场景不用）本来就不一样，
    所以这里要求**转换后能通过场景解析**，而不只是"字段存在"。
    """
    from src.geom.asset import asset_to_object_spec
    from src.scene.scene_spec import SceneSpec

    asset = _prism_asset()
    raw = asset["render"]["materials"][0]["base_color"]
    spec = asset_to_object_spec(asset)
    color = spec["material"]["base_color"]

    payload = {
        "name": "prism",
        "objects": [{
            "name": "prism",
            "geometry": spec["geometry"],
            "transform": spec["transform"],
            "material": spec["material"],
            "mass": spec["mass"],
            "inertia": spec["inertia"],
        }],
    }
    parsed = SceneSpec.from_dict(payload)

    rgb_ok = len(color) == 3 and all(abs(a - b) < 1e-12 for a, b in zip(color, raw[:3]))
    ok = rgb_ok and len(parsed.objects) == 1
    return ok, (
        f"资产 RGBA={len(raw)} 分量 → 场景 RGB={len(color)} 分量，"
        f"前三分量一致={rgb_ok}；场景解析通过={len(parsed.objects) == 1}"
    )


CHECKS = [
    ("1. QΛQᵀ 还原完整张量", test_tensor_reconstruction),
    ("2. 两个四元数互为共轭", test_quaternions_are_conjugate),
    ("3. 主轴正交且 det=+1", test_axes_orthonormal),
    ("4. 碰撞点在 body 系（体积自洽）", test_collision_points_in_body_frame),
    ("5. 贴地高度精确", test_ground_offset_exact),
    ("6. 资产 → 引擎：主惯量一致且静置", test_object_spec_roundtrip_to_engine),
    ("7. 碰撞点未经二次变换", test_spec_matches_body_frame_vertices),
    ("8. 点云资产没有碰撞代理", test_point_cloud_has_no_collision),
    ("9. JSON 逐字节确定", test_json_deterministic),
    ("10. JSON 读写无损", test_json_roundtrip),
    ("11. 未知版本被拒绝", test_empty_error_on_bad_schema),
    ("12. 案例目录 input/output 约定", test_case_layout),
    ("13. 源文件复制进 input 且自包含", test_stage_input_selfcontained),
    ("14. 质量分布能传到引擎（空心盒）", test_mass_distribution_reaches_engine),
    ("15. 输入卡记录不可复现的输入", test_request_records_irreproducible_inputs),
    ("16. 场景资产只存引用", test_scene_asset_stores_references_only),
    ("17. 材质分量数 RGBA→RGB", test_material_channel_count),
]


def main() -> int:
    print("=" * 100)
    print("物体资产：object_asset/1 的数值自洽性")
    print("=" * 100)

    passed = 0
    for title, func in CHECKS:
        try:
            ok, message = func()
        except Exception as exc:  # noqa: BLE001
            ok, message = False, f"抛出异常：{type(exc).__name__}: {exc}"

        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {message}")
        if ok:
            passed += 1

    print("-" * 100)
    print(f"结果：{passed}/{len(CHECKS)} 通过")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
