# -*- coding: utf-8 -*-
"""输入管道测试：OBJ / STL / 点云 / 分析命令。

为什么这组测试必须存在
----------------------
解析器是"算错了也不报错"的典型：

- OBJ 的**负索引**漏了，模型会读成完全错误的形状，体积却还是个数
- STL 的顶点不去重，顶点数是三角形数的 3 倍，**闭合性诊断直接失效**
- 二进制 STL 若按"开头是不是 solid"判断，头部写了 solid 的文件会读不出顶点

这些错误的共同点是：程序照常跑完，输出一个看似合理的数字。所以每一项都要
配一个**独立可算的期望值**（体积用解析式，惯量用 M/6 等）。

运行::

    python Code/tests/test_object_input.py
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.massprops import from_mesh, from_points
from src.geom.meshio import (
    Mesh,
    load_mesh,
    load_points,
    save_obj,
    write_stl_ascii,
    write_stl_binary,
)
from src.geom.meshutil import box_mesh, lattice_points_in_box, prism_mesh, random_points_in_box

TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.4, 1.1)]
PRISM_HEIGHT = 0.6


def _prism() -> Mesh:
    vertices, faces = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    return Mesh(vertices=vertices, faces=faces, source="prism").diagnose()


def _temp_dir():
    return tempfile.TemporaryDirectory()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# 组一：OBJ
# ---------------------------------------------------------------------------

def test_obj_roundtrip() -> tuple[bool, str]:
    """OBJ 写→读之后，体积与惯量必须逐位一致。"""
    mesh = _prism()
    with _temp_dir() as directory:
        path = os.path.join(directory, "prism.obj")
        save_obj(path, mesh)
        loaded = load_mesh(path)

    reference = from_mesh(mesh.vertices, mesh.faces, mass=3.0)
    recovered = from_mesh(loaded.vertices, loaded.faces, mass=3.0)

    volume_error = abs(recovered.volume / reference.volume - 1.0)
    inertia_error = max(
        abs(recovered.inertia[i][j] - reference.inertia[i][j])
        / max(abs(reference.inertia[i][j]), 1e-30)
        for i in range(3)
        for j in range(3)
    )

    ok = volume_error < 1e-12 and inertia_error < 1e-12
    return ok, (
        f"体积相对差={volume_error:.2e}，惯量张量最大相对差={inertia_error:.2e}，"
        f"闭合={loaded.is_closed}"
    )


def test_obj_face_formats() -> tuple[bool, str]:
    """``f a``、``f a/b/c``、``f a//c``、**负索引**、多边形面都要正确解析。

    负索引是最容易漏的一条：``f -4 -3 -2`` 指的是"当前已读入的 4 个顶点的
    倒数第 4、3、2 个"，即 ``(0, 1, 2)``。漏了它会读成完全不同的形状。
    """
    text = """\
# 一个四面体，用四种面写法混合
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
vt 0 0
vn 0 0 1
f -4 -3 -2
f -4 -1//3 -3
f 1/1/1 3/1/1 4/1/1
f 2 4 3
"""
    with _temp_dir() as directory:
        path = os.path.join(directory, "tetra.obj")
        _write(path, text)
        mesh = load_mesh(path)

    expected = [
        (0, 1, 2),
        (0, 3, 1),
        (0, 2, 3),
        (1, 3, 2),
    ]
    ok = mesh.faces == expected
    detail = (
        f"读到 {len(mesh.vertices)} 顶点 / {len(mesh.faces)} 面；"
        f"面 = {mesh.faces}；期望 = {expected}"
    )
    return ok, detail


def test_obj_polygon_triangulated() -> tuple[bool, str]:
    """四边形面必须按扇形三角化成 2 个三角形（不是丢弃、也不是乱连）。"""
    text = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
    with _temp_dir() as directory:
        path = os.path.join(directory, "quad.obj")
        _write(path, text)
        mesh = load_mesh(path)

    ok = mesh.faces == [(0, 1, 2), (0, 2, 3)]
    return ok, f"四边形 → {mesh.faces}（期望 [(0,1,2), (0,2,3)]）"


# ---------------------------------------------------------------------------
# 组二：STL
# ---------------------------------------------------------------------------

def test_stl_ascii() -> tuple[bool, str]:
    """ASCII STL 与 OBJ 必须给出相同的体积与惯量。"""
    mesh = _prism()
    with _temp_dir() as directory:
        obj_path = os.path.join(directory, "a.obj")
        stl_path = os.path.join(directory, "a.stl")
        save_obj(obj_path, mesh)
        write_stl_ascii(stl_path, mesh)
        from_obj = load_mesh(obj_path)
        from_stl = load_mesh(stl_path)

    a = from_mesh(from_obj.vertices, from_obj.faces, mass=3.0)
    b = from_mesh(from_stl.vertices, from_stl.faces, mass=3.0)
    error = abs(a.inertia[0][0] - b.inertia[0][0]) / a.inertia[0][0]

    ok = error < 1e-12 and from_stl.is_closed
    return ok, (
        f"OBJ vs ASCII STL：体积 {a.volume:.12f} / {b.volume:.12f}，"
        f"Ixx 相对差={error:.2e}，STL 闭合={from_stl.is_closed}"
    )


def test_stl_binary_with_solid_header() -> tuple[bool, str]:
    """**二进制** STL 的头部写着 "solid" 时也必须识别为二进制。

    这是 STL 最经典的坑：不少导出器在二进制文件的前 80 字节里写
    ``solid xxx``，于是"看开头是不是 solid"的判据会把它当 ASCII 解析，
    读不到任何 ``vertex`` 行，最终报"没有有效顶点"。
    唯一可靠的判据是 ``文件大小 == 84 + 50 × 三角形数``。
    """
    mesh = _prism()
    with _temp_dir() as directory:
        path = os.path.join(directory, "tricky.stl")
        write_stl_binary(path, mesh)
        # 覆盖前 80 字节，写入 "solid" 开头的假 ASCII 头
        with open(path, "r+b") as handle:
            header = b"solid exported-by-some-tool".ljust(80, b"\0")
            handle.seek(0)
            handle.write(header)
        loaded = load_mesh(path)

    ok = len(loaded.faces) == len(mesh.faces) and loaded.is_closed
    return ok, (
        f"头部为 {header[:18]!r}… 的二进制 STL 识别成功："
        f"{len(loaded.vertices)} 顶点 / {len(loaded.faces)} 面，闭合={loaded.is_closed}"
    )


def test_stl_vertex_dedup() -> tuple[bool, str]:
    """STL 的"每面独立 3 顶点"必须合并成共享顶点表。

    不去重的话顶点数是 3F 而不是 V：闭合性诊断会因为"每条边都是孤立的"
    而失败。
    """
    mesh = _prism()
    with _temp_dir() as directory:
        path = os.path.join(directory, "b.stl")
        write_stl_binary(path, mesh)
        loaded = load_mesh(path)

    raw_count = len(mesh.faces) * 3
    ok = (
        len(loaded.vertices) == len(mesh.vertices)
        and len(loaded.vertices) < raw_count
        and loaded.is_closed
        and loaded.is_consistently_oriented
    )
    return ok, (
        f"原始 {raw_count} 条顶点记录 → 去重后 {len(loaded.vertices)} 个顶点"
        f"（OBJ 为 {len(mesh.vertices)}），闭合={loaded.is_closed}，"
        f"缠绕一致={loaded.is_consistently_oriented}"
    )


def test_open_mesh_detected() -> tuple[bool, str]:
    """缺一个面的网格必须被诊断为"不闭合"。

    散度定理对开网格给出的体积是"几个有符号四面体之和"，是个无意义的数，
    而且**不会报错**。所以闭合性诊断必须在算质量属性之前就把话说清。
    """
    mesh = _prism()
    open_mesh = Mesh(
        vertices=mesh.vertices, faces=mesh.faces[:-1], source="open"
    ).diagnose()

    closed_mesh = _prism()

    ok = (not open_mesh.is_closed) and closed_mesh.is_closed
    return ok, (
        f"去掉 1 个面后闭合={open_mesh.is_closed}"
        f"（χ={open_mesh.euler_characteristic}）；完整网格闭合={closed_mesh.is_closed}"
        f"（χ={closed_mesh.euler_characteristic}）"
    )


# ---------------------------------------------------------------------------
# 组四：点云输入
# ---------------------------------------------------------------------------

def test_lattice_auto_corrected() -> tuple[bool, str]:
    """规则点阵：``auto`` 必须判定为规则采样并修正到精确值。

    单位立方体、质量 8 时解析值 ``Ixx = m(b²+c²)/12 = 8·2/12 = 4/3``。
    """
    points = lattice_points_in_box((1.0, 1.0, 1.0), 10)
    props = from_points(points, 8.0)

    expected = 8.0 * (1.0 + 1.0) / 12.0
    error = abs(props.inertia[0][0] / expected - 1.0)

    ok = props.correction_applied == "nn" and error < 1e-12
    return ok, (
        f"{len(points)} 点：判定={props.correction_applied!r}，"
        f"Ixx={props.inertia[0][0]:.12f}（解析 {expected:.12f}），误差={error:.2e}"
    )


def test_random_auto_not_corrected() -> tuple[bool, str]:
    """随机点云：``auto`` 必须**拒绝**修正（估计量本身无偏）。"""
    rng = random = None
    import random as _random

    rng = _random.Random(11)
    points = random_points_in_box((1.0, 1.0, 1.0), 4000, rng)
    props = from_points(points, 8.0)

    ok = props.correction_applied == "none"
    return ok, (
        f"{len(points)} 点：判定={props.correction_applied!r}"
        f"（随机采样无偏，加修正只会引入正偏差）"
    )


def test_points_file_parsing() -> tuple[bool, str]:
    """点云文件必须容忍表头、注释、逗号/空格/制表符混用。"""
    text = """\
# 一个立方点阵
x,y,z
0,0,0
1 0 0
0\t1\t0
0,0,1
"""
    with _temp_dir() as directory:
        path = os.path.join(directory, "cloud.csv")
        _write(path, text)
        points = load_points(path)

    ok = len(points) == 4 and points[1] == (1.0, 0.0, 0.0)
    return ok, f"读到 {len(points)} 点（跳过表头与注释），第二条={points[1]}"


# ---------------------------------------------------------------------------
# 组五：命令端到端
# ---------------------------------------------------------------------------

def test_command_emits_loadable_scene() -> tuple[bool, str]:
    """``analyze_object.py --emit`` 生成的场景必须能被引擎直接加载并静置。

    这是整条管道的验收测试：文件 → 质量属性 → 凸包 → 场景 JSON → 物理世界。
    只检查"命令跑完没报错"是不够的——必须让生成的场景真的跑起来、
    落到地面上、并且主惯量与命令打印的一致。
    """
    from src.mathutils import quaternion as quat
    from src.physics.world import PhysicsWorld
    from src.scene.scene_spec import load_scene

    mesh = _prism()
    script = os.path.join(_CODE_ROOT, "scripts", "analyze_object.py")

    with _temp_dir() as directory:
        obj_path = os.path.join(directory, "prism.obj")
        scene_path = os.path.join(directory, "scene.json")
        save_obj(obj_path, mesh)

        result = subprocess.run(
            [sys.executable, script, obj_path, "--mass", "3.0", "--emit", scene_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return False, f"命令失败：{result.stderr[-400:]}"

        world = PhysicsWorld.from_scene(load_scene(scene_path))
        body = world.bodies[0]

        for _ in range(240 * 3):
            world.step()

        reference = from_mesh(mesh.vertices, mesh.faces, mass=3.0)
        expected = reference.principal()[0]
        inertia_error = max(
            abs(body.inertia_local[i] - expected[i]) / expected[i]
            for i in range(3)
        )

        lowest = min(
            quat.quat_rotate_vector(body.orientation, v)[2]
            for v in body.geometry.vertices
        ) + body.position[2]

    settled = body.speed() < 0.01
    resting = abs(lowest) < 0.002
    ok = settled and resting and inertia_error < 1e-9

    return ok, (
        f"物体 {body.name}（凸包 {len(body.geometry.faces)} 面），"
        f"主惯量与网格值最大相对差={inertia_error:.2e}；"
        f"3 秒后 |v|={body.speed():.5f}，最低点 z={lowest * 1000:+.3f} mm"
    )


CHECKS = [
    ("1. OBJ 往返：体积/惯量逐位一致", test_obj_roundtrip),
    ("2. OBJ 四种面写法 + 负索引", test_obj_face_formats),
    ("3. OBJ 四边形面三角化", test_obj_polygon_triangulated),
    ("4. ASCII STL 与 OBJ 一致", test_stl_ascii),
    ("5. 二进制 STL 头部写 solid 也能识别", test_stl_binary_with_solid_header),
    ("6. STL 顶点去重与闭合性", test_stl_vertex_dedup),
    ("7. 开网格被诊断为不闭合", test_open_mesh_detected),
    ("8. 规则排列的 auto 判定与精确修正", test_lattice_auto_corrected),
    ("9. 随机点云 auto='none'", test_random_auto_not_corrected),
    ("10. 点云文件格式容错", test_points_file_parsing),
    ("11. 命令生成的场景可加载并静置", test_command_emits_loadable_scene),
]


def main() -> int:
    print("=" * 100)
    print("输入管道：OBJ / STL / 点云 / 分析命令")
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
