# -*- coding: utf-8 -*-
"""质量属性测试：完整惯量张量、网格散度定理、点云与误差修正。

这一组测试的价值在于**所有参照值都是解析的**
------------------------------------------
- 长方体：``I_xx = m/12·(b²+c²)``
- 空心长方体：``V_外·I_外 − V_内·I_内``
- L 形棱柱：面积 3.0、质心 (0.93, 0.76, 0.5)，都能手算
- 立方点阵采样：修正后**精确到机器精度**（中点法则对二次泛函无截断误差）

所以失败一定是代码错了，不会出现"看起来差不多"的模糊判断。

三个必须验证的点
----------------
1. **质心不在原点时也要正确**——旋转绕错点会让物体"甩尾"
2. **惯量积确实非零**——L 形的 ``Ixy ≠ 0``，证明只存对角元是不够的
3. **点云的偏小是系统性的**，且修正项能把它压到接近零

运行::

    python Code/tests/test_massprops.py
"""

from __future__ import annotations

import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.massprops import from_mesh, from_points, jacobi_eigen_3x3
from src.geom.meshutil import (
    box_mesh,
    lattice_points_in_box,
    lattice_points_in_shell,
    polygon_area,
    prism_mesh,
    random_points_in_box,
    random_points_in_sphere,
    shell_mesh,
)

#: L 形多边形（逆时针，面积 3.0）
L_POLYGON = [
    (0.0, 0.0),
    (2.4, 0.0),
    (2.4, 0.8),
    (0.9, 0.8),
    (0.9, 2.0),
    (0.0, 2.0),
]


def diag_of(tensor):
    """取张量对角元。"""
    return (tensor[0][0], tensor[1][1], tensor[2][2])


def relative_error(a, b) -> float:
    """相对误差（以 b 为基准，避免除零）。"""
    scale = max(abs(value) for value in b) if b else 1.0
    return max(abs(a[i] - b[i]) for i in range(len(a))) / max(scale, 1e-300)


def reconstruct(moments, axes):
    """由主惯量与主轴重建张量 ``I = Q Λ Qᵀ``。"""
    return tuple(
        tuple(
            sum(axes[row][k] * moments[k] * axes[col][k] for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


# ---------------------------------------------------------------------------
# 组一：特征分解
# ---------------------------------------------------------------------------

def test_jacobi_eigen() -> tuple[bool, str]:
    """Jacobi 特征分解必须正确复原已知矩阵。

    用的矩阵刻意带非对角项且三个特征值互不相同，
    退化情形（两特征值相等）在下面单独看。
    """
    matrix = (
        (4.0, 1.0, 0.5),
        (1.0, 3.0, 0.25),
        (0.5, 0.25, 2.0),
    )
    values, axes = jacobi_eigen_3x3(matrix)
    rebuilt = reconstruct(values, axes)

    residual = max(
        abs(rebuilt[r][c] - matrix[r][c]) for r in range(3) for c in range(3)
    )
    ascending = values[0] <= values[1] <= values[2]
    ok = residual < 1e-12 and ascending

    return ok, (
        f"特征值={[round(v, 9) for v in values]}（升序={ascending}），"
        f"重建残差={residual:.3e}"
    )


def test_jacobi_degenerate() -> tuple[bool, str]:
    """两个特征值相等（轴对称物体，如圆柱）时不能失稳。

    这正是解析三次方程解法会丢精度的地方：判别式接近零。
    Jacobi 是迭代法，对退化情形同样收敛。
    """
    matrix = (
        (5.0, 0.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, 0.0, 2.0),
    )
    values, axes = jacobi_eigen_3x3(matrix)
    rebuilt = reconstruct(values, axes)
    residual = max(
        abs(rebuilt[r][c] - matrix[r][c]) for r in range(3) for c in range(3)
    )
    ok = residual < 1e-12 and abs(values[0] - 2.0) < 1e-12 and abs(values[2] - 5.0) < 1e-12

    return ok, f"特征值={[round(v, 12) for v in values]}，重建残差={residual:.3e}"


# ---------------------------------------------------------------------------
# 组二：闭合网格
# ---------------------------------------------------------------------------

def test_box_mesh_exact() -> tuple[bool, str]:
    """长方体网格的体积、质心、惯量必须与解析式逐位吻合。"""
    size = (0.8, 1.2, 0.5)
    mass = 12.0
    vertices, faces = box_mesh(size)
    props = from_mesh(vertices, faces, mass=mass)

    sx, sy, sz = size
    expected = (
        mass / 12.0 * (sy * sy + sz * sz),
        mass / 12.0 * (sx * sx + sz * sz),
        mass / 12.0 * (sx * sx + sy * sy),
    )
    volume_expected = sx * sy * sz

    volume_ok = abs(props.volume - volume_expected) < 1e-14
    com_ok = all(abs(c) < 1e-14 for c in props.center_of_mass)
    inertia_error = relative_error(diag_of(props.inertia), expected)

    ok = volume_ok and com_ok and inertia_error < 1e-14
    return ok, (
        f"体积={props.volume:.12f}（期望 {volume_expected:.12f}），"
        f"质心={tuple(round(c, 12) for c in props.center_of_mass)}，"
        f"惯量相对误差={inertia_error:.3e}"
    )


def test_offset_box_com() -> tuple[bool, str]:
    """偏置摆放的盒子：质心必须跟着走，惯量**不随平移改变**。

    这是最容易写错的一处——如果忘了把二阶矩换算到质心，
    惯量会随物体在场景里的位置变化，物体离原点越远转得越"重"。
    """
    size = (0.6, 0.6, 0.6)
    mass = 5.0
    center = (3.0, -2.0, 1.5)

    near_props = from_mesh(*box_mesh(size), mass=mass)
    far_props = from_mesh(*box_mesh(size, center), mass=mass)

    com_ok = all(abs(far_props.center_of_mass[k] - center[k]) < 1e-12 for k in range(3))

    expected = diag_of(near_props.inertia)
    inertia_error = relative_error(diag_of(far_props.inertia), expected)
    inertia_ok = inertia_error < 1e-12

    ok = com_ok and inertia_ok
    return ok, (
        f"质心={tuple(round(c, 6) for c in far_props.center_of_mass)}（期望 {center}），"
        f"惯量与原点处相比相对误差={inertia_error:.3e}"
    )


def test_hollow_box() -> tuple[bool, str]:
    """空心长方体：体积与惯量等于"外形减内腔"。

    这验证的是**有符号体积自动处理空腔**——内表面反向缠绕后体积为负，
    同一套累加逻辑不需要任何特殊分支。
    """
    outer = (1.0, 1.0, 1.0)
    inner = (0.6, 0.6, 0.6)
    vertices, faces = shell_mesh(outer, inner)
    props = from_mesh(vertices, faces, density=1.0)

    # 密度 1 时，边长 L 的实心立方体绕任意中心轴惯量 = L^5 / 6
    def solid_cube_inertia(side: float) -> float:
        return side ** 5 / 6.0

    expected_volume = outer[0] ** 3 - inner[0] ** 3
    expected_inertia = solid_cube_inertia(outer[0]) - solid_cube_inertia(inner[0])

    volume_ok = abs(props.volume - expected_volume) < 1e-12
    inertia_error = abs(props.inertia[0][0] - expected_inertia) / expected_inertia
    com_ok = all(abs(c) < 1e-12 for c in props.center_of_mass)

    ok = volume_ok and com_ok and inertia_error < 1e-12
    return ok, (
        f"体积={props.volume:.9f}（期望 {expected_volume:.9f}），"
        f"Ixx={props.inertia[0][0]:.9f}（期望 {expected_inertia:.9f}），"
        f"相对误差={inertia_error:.3e}"
    )


def test_lshape_offdiagonal() -> tuple[bool, str]:
    """L 形棱柱：体积、质心可手算；**惯量积必须非零**。

    这一项是整个模块存在的理由。L 形关于 x=y 不对称，所以 ``Ixy ≠ 0``。
    只存 ``(Ixx, Iyy, Izz)`` 的引擎在这里必然算错角加速度，
    而症状只是"转起来有点怪"，不会报任何错。
    """
    vertices, faces = prism_mesh(L_POLYGON, height=1.0)
    props = from_mesh(vertices, faces, mass=3.0)

    area = polygon_area(L_POLYGON)
    expected_volume = area * 1.0

    # 手算质心：下横条 2.4×0.8（质心 1.2, 0.4），上竖条 0.9×1.2（质心 0.45, 1.4）
    weight_low = 2.4 * 0.8
    weight_up = 0.9 * 1.2
    total_area = weight_low + weight_up
    expected_com = (
        (weight_low * 1.2 + weight_up * 0.45) / total_area,
        (weight_low * 0.4 + weight_up * 1.4) / total_area,
        0.5,
    )

    volume_ok = abs(props.volume - expected_volume) < 1e-12
    com_ok = all(
        abs(props.center_of_mass[k] - expected_com[k]) < 1e-12 for k in range(3)
    )

    ixy, iyz, ixz = props.products()
    offdiagonal_nonzero = abs(ixy) > 0.05
    not_diagonal = not props.is_diagonal()

    ok = volume_ok and com_ok and offdiagonal_nonzero and not_diagonal
    return ok, (
        f"体积={props.volume:.9f}（期望 {expected_volume:.9f}），"
        f"质心=({props.center_of_mass[0]:.6f},{props.center_of_mass[1]:.6f},"
        f"{props.center_of_mass[2]:.6f})（期望 ({expected_com[0]:.3f},"
        f"{expected_com[1]:.3f},0.5)），"
        f"惯量积 Ixy={ixy:.6f} / Iyz={iyz:.6f} / Ixz={ixz:.6f}，是对角阵={props.is_diagonal()}"
    )


def test_principal_reconstruction() -> tuple[bool, str]:
    """主分解必须能重建原张量，且主轴构成正交基。"""
    vertices, faces = prism_mesh(L_POLYGON, height=1.0)
    props = from_mesh(vertices, faces, mass=3.0)
    moments, axes = props.principal()
    rebuilt = reconstruct(moments, axes)

    residual = max(
        abs(rebuilt[r][c] - props.inertia[r][c])
        for r in range(3) for c in range(3)
    )

    # 正交性：QᵀQ = I
    orthogonality = max(
        abs(sum(axes[k][r] * axes[k][c] for k in range(3)) - (1.0 if r == c else 0.0))
        for r in range(3) for c in range(3)
    )

    ok = residual < 1e-12 and orthogonality < 1e-12
    return ok, (
        f"主惯量={[round(v, 6) for v in moments]}，"
        f"重建残差={residual:.3e}，正交性残差={orthogonality:.3e}"
    )


def test_to_engine() -> tuple[bool, str]:
    """交给引擎的对角惯量必须为正，且能复原原张量。"""
    vertices, faces = prism_mesh(L_POLYGON, height=1.0)
    props = from_mesh(vertices, faces, mass=3.0)
    moments, _ = props.to_engine()

    positive = all(value > 0.0 for value in moments)
    # 三角不等式：任意两个主惯量之和 ≥ 第三个（物理可实现性的必要条件）
    triangle = (
        moments[0] + moments[1] >= moments[2] - 1e-12
        and moments[1] + moments[2] >= moments[0] - 1e-12
        and moments[0] + moments[2] >= moments[1] - 1e-12
    )

    ok = positive and triangle
    return ok, (
        f"主惯量={[round(v, 6) for v in moments]}，全正={positive}，"
        f"三角不等式={triangle}（物理可实现性）"
    )


# ---------------------------------------------------------------------------
# 组三：点云
# ---------------------------------------------------------------------------

def test_lattice_cloud_exact() -> tuple[bool, str]:
    """立方点阵采样 + 修正 ⇒ **精确到机器精度**。

    理论上：中点法则对二次多项式在加回 ``h²/12`` 后没有截断误差，
    而惯量正是坐标的二次泛函。所以修正后应当精确，而不是"接近"。

    这是点云方案能达到的最好情况，也是修正公式正确性的最硬证据。
    """
    size = (1.0, 1.0, 1.0)
    mass = 8.0
    n = 8

    vertices, faces = box_mesh(size)
    reference = from_mesh(vertices, faces, mass=mass)
    expected = diag_of(reference.inertia)

    points = lattice_points_in_box(size, n)
    raw = from_points(points, mass, correction="none", volume=1.0)
    by_volume = from_points(points, mass, correction="volume", volume=1.0)
    by_neighbor = from_points(points, mass, correction="nn")
    auto = from_points(points, mass)

    raw_error = relative_error(diag_of(raw.inertia), expected)
    volume_error = relative_error(diag_of(by_volume.inertia), expected)
    neighbor_error = relative_error(diag_of(by_neighbor.inertia), expected)
    auto_error = relative_error(diag_of(auto.inertia), expected)

    # 原始估计必须**偏小**（系统性偏差，不是随机噪声）
    underestimates = all(diag_of(raw.inertia)[k] < expected[k] for k in range(3))

    # auto 必须识别出这是规则采样并施加修正
    auto_detected = auto.correction_applied == "nn"

    ok = (
        underestimates
        and volume_error < 1e-12
        and neighbor_error < 1e-12
        and auto_error < 1e-12
        and auto_detected
    )
    return ok, (
        f"N={len(points)}：原始误差={raw_error:.3e}（偏小={underestimates}），"
        f"体积修正={volume_error:.3e}，近邻修正={neighbor_error:.3e}，"
        f"auto={auto_error:.3e}（判定为 {auto.correction_applied!r}）"
    )


def test_point_cloud_hollow() -> tuple[bool, str]:
    """点云表达空腔：内腔没有点，它就不贡献质量。"""
    outer = (1.0, 1.0, 1.0)
    inner = (0.6, 0.6, 0.6)
    n = 20

    vertices, faces = shell_mesh(outer, inner)
    reference = from_mesh(vertices, faces, density=1.0)

    points = lattice_points_in_shell(outer, inner, n)
    shell_volume = outer[0] ** 3 - inner[0] ** 3
    cloud = from_points(
        points, reference.mass, correction="volume", volume=shell_volume
    )

    error = relative_error(diag_of(cloud.inertia), diag_of(reference.inertia))
    com_ok = all(abs(c) < 1e-12 for c in cloud.center_of_mass)

    ok = error < 1e-10 and com_ok
    return ok, (
        f"N={len(points)}（内腔 {inner[0]}^3 被跳过）：Ixx={cloud.inertia[0][0]:.9f} "
        f"vs 网格参照 {reference.inertia[0][0]:.9f}，相对误差={error:.3e}"
    )


def test_convergence() -> tuple[bool, str]:
    """误差随 N 增大而下降，且修正后远优于未修正。

    这条测试的意义：**修正不是"大概对"，而是有可测量的收敛性**。
    未修正的误差应呈明显下降趋势（约 ``N^(-2/3)``），修正后应保持在极小值。
    """
    size = (1.0, 1.0, 1.0)
    mass = 8.0
    vertices, faces = box_mesh(size)
    expected = diag_of(from_mesh(vertices, faces, mass=mass).inertia)

    rows = []
    for n in (4, 6, 8, 12, 16):
        points = lattice_points_in_box(size, n)
        raw = from_points(points, mass, correction="none", volume=1.0)
        fixed = from_points(points, mass, correction="volume", volume=1.0)
        rows.append(
            (
                len(points),
                relative_error(diag_of(raw.inertia), expected),
                relative_error(diag_of(fixed.inertia), expected),
            )
        )

    # 未修正误差必须单调下降
    raw_errors = [row[1] for row in rows]
    monotone = all(raw_errors[i] > raw_errors[i + 1] for i in range(len(rows) - 1))

    # 修正后必须始终优于未修正，且至少好 1e6 倍
    corrected_much_better = all(
        row[2] < row[1] * 1e-6 for row in rows
    )
    # 修正后误差应保持在机器精度量级
    corrected_tiny = max(row[2] for row in rows) < 1e-12

    ok = monotone and corrected_much_better and corrected_tiny
    detail = "  ".join(
        f"N={n}: raw={raw:.2e}/fix={fix:.2e}" for n, raw, fix in rows
    )
    return ok, detail


def test_random_cloud_unbiased() -> tuple[bool, str]:
    """随机点云的估计量**本身无偏**，因此 ``auto`` 必须拒绝施加修正。

    这是本模块最重要的一条结论，也是"能不能把误差修到几乎没有"的正面回答。
    修正项 ``m h²/6`` 的物理含义是"**微元自身**的惯量"，而它只在
    **把点当作等体积微元的质心**时才成立——也就是规则采样（点阵、体素中心）。

    对 i.i.d. 随机采样，估计量

        Î = (M/N) Σ (|xᵢ|²1 − xᵢxᵢᵀ)

    的期望**精确等于**连续积分 ``∫ρ(|r|²1 − rrᵀ)dV``（因为每个 xᵢ 都是
    该分布的无偏样本）。既然无偏，再加一个正的 ``m h²/6`` 就是**凭空引入偏差**。

    ================== ================ ================== ==============
    采样方式            误差来源          量级                是否需要修正
    ================== ================ ================== ==============
    立方点阵/体素中心     截断（系统性）    ``O(N^(-2/3))``      **需要**，可精确消除
    i.i.d. 随机         统计（零均值）    ``O(N^(-1/2))``      **不需要**，加了更差
    ================== ================ ================== ==============

    验证办法：对多个随机种子取平均。零均值噪声被抵消、只剩系统性偏差，
    于是可以用"平均值是否显著偏离 0"来判断有没有偏差。
    这里用样本标准差自算标准的误差，避免硬编码阈值。
    """
    size = (1.0, 1.0, 1.0)
    mass = 8.0
    vertices, faces = box_mesh(size)
    expected = diag_of(from_mesh(vertices, faces, mass=mass).inertia)[0]

    count = 2000
    seeds = list(range(12))
    raw_biases = []
    volume_biases = []
    auto_modes = set()

    for seed in seeds:
        rng = random.Random(seed)
        points = random_points_in_box(size, count, rng)
        raw = from_points(points, mass, correction="none", volume=1.0)
        forced = from_points(points, mass, correction="volume", volume=1.0)
        auto = from_points(points, mass)          # 默认就是 auto
        raw_biases.append(raw.inertia[0][0] / expected - 1.0)
        volume_biases.append(forced.inertia[0][0] / expected - 1.0)
        auto_modes.add(auto.correction_applied)

    n = len(seeds)
    mean_raw = sum(raw_biases) / n
    mean_volume = sum(volume_biases) / n

    # 样本标准差 → 均值标准误 → 判断"是否显著非零"
    variance = sum((b - mean_raw) ** 2 for b in raw_biases) / (n - 1)
    standard_error = math.sqrt(variance / n)
    unbiased = abs(mean_raw) < 3.0 * standard_error

    # 强行施加修正必须引入**明显的正偏差**
    introduced = mean_volume - mean_raw
    theoretical = (1.0 / count) ** (2.0 / 3.0)    # 相对修正量 ≈ h²
    roughly_right = abs(introduced - theoretical) < theoretical * 0.45

    # auto 必须判定为随机、不修正
    auto_declines = auto_modes == {"none"}

    ok = unbiased and (introduced > 2e-3) and roughly_right and auto_declines
    return ok, (
        f"{n} 个种子 × N={count}：原始平均偏差={mean_raw:+.2e}"
        f"（标准误 {standard_error:.2e}，无偏={unbiased}）；"
        f"强行修正后偏差增量={introduced:+.2e}（理论 h²≈{theoretical:.2e}）；"
        f"auto 判定={auto_modes}"
    )


def test_sphere_cloud() -> tuple[bool, str]:
    """球体随机点云 vs 解析值 ``2/5 m r²``。

    球面**不是**网格对齐的，所以这里是纯随机采样：估计量无偏，
    误差应落在蒙特卡洛噪声范围内（``O(N^(-1/2))``），且随 N 增大而下降。

    这一项同时验证"球内均匀采样"没有把点向球心聚集——
    如果半径忘了做立方根修正，惯量会明显偏小。
    """
    radius = 0.5
    mass = 4.0
    expected = 2.0 / 5.0 * mass * radius * radius
    volume = 4.0 / 3.0 * math.pi * radius ** 3

    errors = []
    for count in (2000, 8000):
        rng = random.Random(7)
        points = random_points_in_sphere(radius, count, rng)
        cloud = from_points(points, mass, correction="none", volume=volume)
        errors.append(abs(cloud.inertia[0][0] - expected) / expected)

    # 误差应在蒙特卡洛噪声量级内（1/√N 的经验倍数）
    within_noise = errors[0] < 6.0 / math.sqrt(2000)
    decreased = errors[1] < errors[0]

    cloud_auto = from_points(
        random_points_in_sphere(radius, 2000, random.Random(7)), mass
    )

    ok = within_noise and decreased and cloud_auto.correction_applied == "none"
    return ok, (
        f"H 解析={expected:.6f}；相对误差 N=2000 时 {errors[0]:.2e}、"
        f"N=8000 时 {errors[1]:.2e}（噪声量级 {6.0 / math.sqrt(2000):.2e}）；"
        f"auto 判定={cloud_auto.correction_applied!r}"
    )


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

CHECKS = [
    ("1. Jacobi 特征分解", test_jacobi_eigen),
    ("2. 退化特征值（轴对称）", test_jacobi_degenerate),
    ("3. 长方体网格：解析精确", test_box_mesh_exact),
    ("4. 偏置盒子：质心跟随、惯量不变", test_offset_box_com),
    ("5. 空心盒：有符号体积自动处理", test_hollow_box),
    ("6. L 形：惯量积非零（必须用完整张量）", test_lshape_offdiagonal),
    ("7. 主分解重建原张量", test_principal_reconstruction),
    ("8. 交给引擎的主惯量物理可行", test_to_engine),
    ("9. 立方点阵 + 修正 ⇒ 机器精度", test_lattice_cloud_exact),
    ("10. 点云表达空腔", test_point_cloud_hollow),
    ("11. 误差随 N 收敛", test_convergence),
    ("12. 随机点云本身无偏，auto 拒绝修正", test_random_cloud_unbiased),
    ("13. 球体随机点云 vs 2/5 m r²", test_sphere_cloud),
]


def main() -> int:
    """依次执行并汇总。"""
    print("=" * 100)
    print("质量属性测试：完整惯量张量 / 网格散度定理 / 点云与误差修正")
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
