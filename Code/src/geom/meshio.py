# -*- coding: utf-8 -*-
"""网格与点云的文件读写：OBJ / STL / 文本点云。

为什么自己写而不引入 trimesh / open3d
------------------------------------
这两个库都带着几十 MB 的依赖树，而本项目只需要"把顶点和三角面读进来"。
更重要的是**判据**：解析器写错了（比如漏掉负索引、把四边形面的对角线
按错误顺序连起来），算出来的体积和惯量会**静默地**差一点，没有任何异常。
自己写的解析器只有 300 行，可以逐条对着格式规范测，比调库更容易保证正确。

支持的格式
----------
- **OBJ**：``v`` / ``f``。面支持 ``f a``、``f a/b``、``f a//c``、``f a/b/c``、
  **负索引**（相对当前顶点数倒数），多边形面按**扇形三角化**。
- **STL**：ASCII 与二进制自动识别（按文件大小与头部计数交叉验证）。
  二进制 STL 每个三角形独立存 3 个顶点，读入时按坐标**去重**，
  否则顶点数会是三角形数的 3 倍，边表也会炸掉。
- **点云**：``.csv`` / ``.txt`` / ``.xyz``，每行 ``x,y,z``（逗号/空格/制表符），
  自动跳过注释行与表头。

一个必须说清的约定
------------------
读进来的网格必须**闭合且缠绕一致朝外**，否则散度定理给出的体积会出现正负抵消。
``load_mesh`` 会顺手做一次检查并把结论放在返回值里，而不是让它悄悄出错。
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Mesh:
    """三角网格：顶点表 + 三角面表 + 来源信息。"""

    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    #: 来源描述，用于报告（文件名与格式）
    source: str = ""

    #: 诊断信息，由 ``load_mesh`` 填写
    is_closed: bool = False
    is_consistently_oriented: bool = False
    euler_characteristic: int = 0
    duplicate_faces: int = 0

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """包围盒的 (最小角, 最大角)。"""
        if not self.vertices:
            raise ValueError("空网格没有包围盒")
        low = tuple(min(v[k] for v in self.vertices) for k in range(3))
        high = tuple(max(v[k] for v in self.vertices) for k in range(3))
        return low, high

    def signed_volume(self) -> float:
        """有符号体积。闭合且缠绕朝外时为正，等于真实体积。"""
        total = 0.0
        for a, b, c in self.faces:
            va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
            total += (
                va[0] * (vb[1] * vc[2] - vb[2] * vc[1])
                - va[1] * (vb[0] * vc[2] - vb[2] * vc[0])
                + va[2] * (vb[0] * vc[1] - vb[1] * vc[0])
            ) / 6.0
        return total

    def surface_area(self) -> float:
        total = 0.0
        for a, b, c in self.faces:
            va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
            ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
            wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
            cx, cy, cz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
            total += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
        return total

    def diagnose(self) -> "Mesh":
        """填充闭合性与缠绕一致性诊断。

        判据
        ----
        - **闭合**：每条无向边恰好被 2 个三角形共用。
        - **缠绕一致**：每条**有向**边恰好出现一次（相邻三角形成对反向）。
          若某个有向边出现两次，说明两个三角形同向共用该边，缠绕不一致。
        - **欧拉示性数** ``χ = V − E + F``：闭合可定向曲面应为 ``2 − 2g``
          （球面 2，环面 0），是一个独立于上面两条的交叉验证。
        """
        undirected: dict[tuple[int, int], int] = {}
        directed: dict[tuple[int, int], int] = {}

        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                undirected[key] = undirected.get(key, 0) + 1
                directed[(u, v)] = directed.get((u, v), 0) + 1

        self.is_closed = all(count == 2 for count in undirected.values())
        self.is_consistently_oriented = all(
            count == 1 for count in directed.values()
        )
        self.duplicate_faces = sum(
            1 for count in undirected.values() if count > 2
        )
        self.euler_characteristic = (
            len(self.vertices) - len(undirected) + len(self.faces)
        )
        return self


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------

def load_obj(path: str) -> Mesh:
    """读取 OBJ 文件，返回三角化后的网格。

    只关心 ``v`` 与 ``f`` 两种行，其余（``vt`` / ``vn`` / ``o`` / ``g`` /
    ``s`` / ``mtllib`` / ``usemtl``）全部跳过——它们不影响几何。

    面索引有三种坑，都必须处理：

    1. ``f 1 2 3`` 是 **1 基**，要减 1
    2. ``f 1/2/3``、``f 1//3``、``f 1/2`` 只取第一个字段
    3. ``f -1 -2 -3`` 是**相对当前顶点数的负索引**，要加到 ``len(vertices)`` 上

    第 3 条最容易漏：漏了它导出的模型会读成完全错误的形状，而且不报错。
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] == "#":
                continue

            parts = stripped.split()
            keyword = parts[0]

            if keyword == "v":
                if len(parts) < 4:
                    raise ValueError(f"{path}: 顶点行字段不足：{stripped!r}")
                vertices.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            elif keyword == "f":
                indices = []
                for token in parts[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    index = int(raw)
                    # 负索引相对"当前已读入的顶点数"
                    indices.append(index - 1 if index > 0 else len(vertices) + index)

                if len(indices) < 3:
                    continue

                # 多边形面按扇形三角化。要求面是凸的——OBJ 的 n-gon 通常是，
                # 非凸面在导出阶段就该被三角化，这里不做耳切。
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))

    if not vertices or not faces:
        raise ValueError(f"{path}: 没有读到有效的顶点/面")

    count = len(vertices)
    for face in faces:
        for index in face:
            if index < 0 or index >= count:
                raise ValueError(
                    f"{path}: 面索引 {index} 越界（共 {count} 个顶点）"
                )

    return Mesh(
        vertices=vertices,
        faces=faces,
        source=f"{os.path.basename(path)}（OBJ）",
    ).diagnose()


def save_obj(path: str, mesh: Mesh) -> None:
    """写出 OBJ（1 基索引）。主要用于测试往返一致性。"""
    with open(path, "w", encoding="utf-8") as handle:
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for a, b, c in mesh.faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------

def _is_binary_stl(path: str) -> bool:
    """按文件大小与头部计数交叉验证判断是否二进制 STL。

    不能只看开头是不是 ``solid``：**二进制 STL 的 80 字节头里经常也写着
    "solid"**（不少导出器这么干）。唯一可靠的判据是

        file_size == 84 + 50 × triangle_count

    二进制 STL 的结构是固定的 80 字节头 + 4 字节计数 + 每个三角形 50 字节。
    """
    size = os.path.getsize(path)
    if size < 84:
        return False
    with open(path, "rb") as handle:
        header = handle.read(84)
    claimed = struct.unpack("<I", header[80:84])[0]
    return size == 84 + claimed * 50


def load_stl(path: str) -> Mesh:
    """读取 STL（自动识别 ASCII / 二进制），顶点按坐标去重。"""
    if _is_binary_stl(path):
        return _load_stl_binary(path)
    return _load_stl_ascii(path)


def _deduplicate(
    raw_vertices: Sequence[Sequence[float]], digits: int = 9
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """把 STL 的"每面独立 3 顶点"合并成共享顶点表。

    不去重的话，顶点数是三角形数的 3 倍：边表会变成 3F 条而不是约 1.5F 条，
    闭合性诊断也会失效（每个三角形都是孤立的三条边）。
    """
    vertices: list[tuple[float, float, float]] = []
    lookup: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []

    for i in range(0, len(raw_vertices), 3):
        triangle = []
        for offset in range(3):
            point = raw_vertices[i + offset]
            key = (round(point[0], digits), round(point[1], digits), round(point[2], digits))
            index = lookup.get(key)
            if index is None:
                index = len(vertices)
                lookup[key] = index
                vertices.append((float(point[0]), float(point[1]), float(point[2])))
            triangle.append(index)
        faces.append((triangle[0], triangle[1], triangle[2]))

    return vertices, faces


def _load_stl_binary(path: str) -> Mesh:
    with open(path, "rb") as handle:
        data = handle.read()

    count = struct.unpack("<I", data[80:84])[0]
    raw: list[tuple[float, float, float]] = []
    expected = 84 + count * 50

    for i in range(count):
        base = 84 + i * 50
        if base + 50 > len(data):
            raise ValueError(
                f"{path}: 二进制 STL 截断（声明 {count} 个三角形，"
                f"文件只有 {len(data)} 字节，应为 {expected}）"
            )
        # 跳过前 12 字节法向（STL 的法向经常是错的或全零，用它反而出错）
        values = struct.unpack("<9f", data[base + 12 : base + 48])
        raw.append((values[0], values[1], values[2]))
        raw.append((values[3], values[4], values[5]))
        raw.append((values[6], values[7], values[8]))

    vertices, faces = _deduplicate(raw)
    return Mesh(
        vertices=vertices,
        faces=faces,
        source=f"{os.path.basename(path)}（二进制 STL，{count} 个三角形）",
    ).diagnose()


def _load_stl_ascii(path: str) -> Mesh:
    raw: list[tuple[float, float, float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("vertex"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                raise ValueError(f"{path}: vertex 行字段不足：{stripped!r}")
            raw.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if len(raw) % 3 != 0:
        raise ValueError(
            f"{path}: 顶点数 {len(raw)} 不是 3 的倍数，ASCII STL 可能损坏"
        )

    vertices, faces = _deduplicate(raw)
    return Mesh(
        vertices=vertices,
        faces=faces,
        source=f"{os.path.basename(path)}（ASCII STL，{len(faces)} 个三角形）",
    ).diagnose()


# ---------------------------------------------------------------------------
# 点云
# ---------------------------------------------------------------------------

def load_points(path: str) -> list[tuple[float, float, float]]:
    """读取文本点云：每行 ``x,y,z``，分隔符可为逗号/空格/制表符。

    自动跳过空行、``#`` 注释行、以及与数据行格式不符的行（表头）。
    容忍表头是必要的——从 Excel / CSV 导出的点云第一行几乎总是 ``x,y,z``。
    """
    points: list[tuple[float, float, float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] == "#":
                continue

            fields = stripped.replace(",", " ").replace("\t", " ").split()
            if len(fields) < 3:
                continue

            try:
                point = (float(fields[0]), float(fields[1]), float(fields[2]))
            except ValueError:
                continue   # 表头或非数字行

            points.append(point)

    if len(points) < 4:
        raise ValueError(f"{path}: 有效点只有 {len(points)} 个，至少需要 4 个")
    return points


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

#: 扩展名 -> 读取函数
_MESH_READERS = {
    ".obj": load_obj,
    ".stl": load_stl,
}


def load_mesh(path: str) -> Mesh:
    """按扩展名读取网格。

    异常
    ----
    ValueError : 扩展名不支持，或解析失败
    """
    extension = os.path.splitext(path)[1].lower()
    reader = _MESH_READERS.get(extension)
    if reader is None:
        supported = ", ".join(sorted(_MESH_READERS))
        raise ValueError(
            f"不支持的网格格式 {extension!r}；当前支持：{supported}"
        )
    return reader(path)


def write_stl_ascii(path: str, mesh: Mesh) -> None:
    """写出 ASCII STL（法向按右手定则现算，不写零向量）。"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("solid mesh\n")
        for a, b, c in mesh.faces:
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
            wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
            nx, ny, nz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            handle.write(
                f"  facet normal {nx / length:.6g} {ny / length:.6g} {nz / length:.6g}\n"
                "    outer loop\n"
            )
            for vertex in (va, vb, vc):
                handle.write(
                    f"      vertex {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n"
                )
            handle.write("    endloop\n  endfacet\n")
        handle.write("endsolid mesh\n")


def write_stl_binary(path: str, mesh: Mesh) -> None:
    """写出二进制 STL。"""
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(mesh.faces)))
        for a, b, c in mesh.faces:
            va, vb, vc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
            ux, uy, uz = vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]
            wx, wy, wz = vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]
            nx, ny, nz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
            length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            handle.write(
                struct.pack(
                    "<3f", nx / length, ny / length, nz / length
                )
            )
            for vertex in (va, vb, vc):
                handle.write(struct.pack("<3f", *vertex))
            handle.write(struct.pack("<H", 0))
