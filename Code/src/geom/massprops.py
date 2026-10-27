# -*- coding: utf-8 -*-
"""质量属性：从网格或等质量点云求体积、质心、完整 3×3 转动惯量张量。

为什么需要这个模块
------------------
参数化图元（球、盒、柱、胶囊）的质心就在原点、惯量张量是对角阵，
一条闭式公式就够了（见 ``inertia.py``）。但任意形状不行：

1. **质心不在原点**——旋转会绕着一个错误点发生，物体"甩尾"
2. **惯量张量有非对角项**（惯量积）——只存 ``(Ixx, Iyy, Izz)`` 会算错角加速度

两者都不会报错，只是让物体的运动"看着别扭"，极难排查。所以必须支持完整张量。

两种输入，两种用途
------------------
- **闭合网格**（``from_mesh``）：走散度定理，**解析精确**。
  同时是点云方案的验证基准。
- **等质量点云**（``from_points``）：用点的疏密表达密度分布，可表达空腔与
  梯度密度。代价是有 ``O(h²)`` 的**系统性偏小**，见第 3 节。

网格与点云的分工
----------------
**网格负责碰撞（它提供表面与法向），点云只负责惯量。**
点云没有"内外"概念，生成不了接触法向与穿透深度，不能参与碰撞检测。

1. 完整惯量张量怎么接进引擎
---------------------------
引擎现在只存对角惯量 ``(Ixx, Iyy, Izz)``，靠 ``R·diag(1/I)·Rᵀ`` 作用。
任意形状的张量不是对角的，但**对称正定矩阵一定可以正交对角化**：

    I = Q Λ Qᵀ,   Λ 对角、Q 是旋转

于是

    I_world = R I Rᵀ = R Q Λ Qᵀ Rᵀ = (RQ) Λ (RQ)ᵀ

也就是说：**把主轴旋转 Q 并进物体的朝向，引擎的对角实现就仍然成立**，
一行都不用改。做法是加载时把顶点旋转到主轴系、把质心平移到原点，
并记住这个固定的 Q 与质心偏移。

2. 闭合网格的质量属性（散度定理）
--------------------------------
把闭合三角网格从任意参考点剖分成**有符号四面体**，逐个累加二阶矩：

    V_tet = det(a, b, c) / 6                    （带符号，自动处理缠绕方向与空腔）
    M1  += V_tet · (a + b + c) / 4              （一阶矩，用于质心）
    S   += (V_tet/20) · [Σ vᵢvᵢᵀ + (Σ vᵢ)(Σ vᵢ)ᵀ]

其中求和遍历四面体的 4 个顶点 ``{0, a, b, c}``。这是单纯形上一阶/二阶矩的
标准结果（``(n+2)(n+1)`` 的系数，n=3 时即 20）。

最后换算到质心并对角线取迹：

    S_c = S − V·c cᵀ
    I   = tr(S_c)·1 − S_c

**空腔自动处理**：内表面若缠绕方向正确，其有符号体积为负，贡献自动被减掉。
所以空心物体不需要特殊代码。

3. 点云的偏差与修正——分两种采样，结论相反
------------------------------------------
等质量点云把连续积分换成了求和：

    I = ∫ ρ(|r|²1 − rrᵀ)dV  ≈  Σ mᵢ(|xᵢ|²1 − xᵢxᵢᵀ)

**误差的性质完全取决于点是怎么来的**，这一点极容易搞混：

============================ ==================== =================================
采样方式                       误差性质              是否需要修正
============================ ==================== =================================
立方点阵 / 体素中心 / 分层       截断误差（系统性，恒偏小） **需要**，可精确消除
i.i.d. 随机采样（泊松）          统计噪声（零均值，无偏）   **不需要**，加了更差
============================ ==================== =================================

**为什么规则采样偏小**：把点当作"等体积微元的质心"时，真实积分里每个微元
对自己质心还有贡献，而求和式漏掉了它。记每点代表的体积为 Vᵢ、等效边长
hᵢ = Vᵢ^(1/3)，则立方微元对自身质心的惯量是 ``mᵢhᵢ²/6``，于是

    ΔI = (1/6) Σ mᵢ hᵢ² · 1        （只加对角线，各向同性，不改变惯量积）

对立方点阵 ``hᵢ = h``，ΔI = Mh²/6。**可以证明修正后精确到机器精度**：
中点法则对二次多项式在加回 ``h²/12`` 项后没有截断误差，而惯量恰是坐标的二次泛函。

**为什么随机采样不需要**：估计量

    Î = (M/N) Σ (|xᵢ|²1 − xᵢxᵢᵀ)

的期望**精确等于**连续积分，因为每个 xᵢ 都是该分布的无偏样本。既然无偏，
再加一个正的 ``m h²/6`` 就是**凭空引入偏差**。

实测（``scripts/bench_massprops.py``，立方体，250 个随机种子）：

- 规则采样：未修正误差 ``O(N^(-2/3))``（拟合 **−0.6667**），修正后 **1e-16**
- 随机采样：偏差/标准误全在 ±3 内（无偏），误差 ``O(N^(-1/2))``，修正不降反升

所以修正必须**按采样类型决定**，这就是 ``correction="auto"`` 的作用：
用近邻间距的变异系数判断（点阵 **0.00**，随机点 **0.37**，阈值 0.1）。

实用结论：**要精度就用点阵/体素采样，不要随机撒点。** 同样的 N，
规则采样误差小一个数量级，而且可以精确修正。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: 判定体积退化（几乎为零）的阈值，立方米
_VOLUME_EPS = 1e-12

#: 判定"点是规则采样"的近邻间距变异系数阈值。
#:
#: 立方点阵的所有近邻距离完全相同 ⇒ 变异系数为 0；
#: i.i.d. 随机采样（泊松过程）的近邻距离有显著涨落 ⇒ 变异系数约 0.2–0.6。
#: 取 0.1 作为分界，两侧都留有充足余量。
#:
#: 这个判据决定了 ``correction="auto"`` 是否施加修正。**必须有这一层判断**：
#: 修正项只在"点代表等体积微元的质心"时才成立，随机点云加修正会引入正偏差。
_REGULAR_VARIATION_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# 3×3 对称矩阵工具
# ---------------------------------------------------------------------------

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def jacobi_eigen_3x3(matrix: Matrix3) -> tuple[list[float], Matrix3]:
    """用 Jacobi 旋转法求 3×3 对称矩阵的特征值与特征向量。

    为什么不用解析公式：三次方程的三角解法在近退化情形（两个特征值接近）
    会丢精度，而惯量张量恰恰经常出现这种情况（对称物体的两个惯量相等）。
    Jacobi 是迭代法，对退化情形同样收敛，而且只涉及 2×2 旋转，数值稳定。

    返回
    ----
    (eigenvalues, eigenvectors)
        ``eigenvalues`` 按升序排列；``eigenvectors`` 的**第 i 列**
        是第 i 个特征值对应的单位特征向量。
    """
    # 复制一份，避免修改入参
    a = [list(row) for row in matrix]

    # 特征向量矩阵，初始为单位阵，随旋转累积
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(64):
        # 找到最大的非对角元
        off = 0.0
        p, q = 0, 1
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(a[i][j]) > off:
                    off = abs(a[i][j])
                    p, q = i, j

        if off < 1e-300:
            break
        # 判据：非对角元相对量级足够小即可停止
        scale = max(abs(a[0][0]), abs(a[1][1]), abs(a[2][2]), 1e-300)
        if off <= 1e-15 * scale:
            break

        # 计算 2×2 旋转角，使 a[p][q] 归零
        theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
        if theta >= 0.0:
            t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
        else:
            t = -1.0 / (-theta + math.sqrt(1.0 + theta * theta))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        # 施加相似变换 A ← Jᵀ A J
        for k in range(3):
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(3):
            apk = a[p][k]
            aqk = a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk

        for k in range(3):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    values = [a[0][0], a[1][1], a[2][2]]
    columns = [(v[0][i], v[1][i], v[2][i]) for i in range(3)]

    # 按特征值升序排列，同时重排特征向量
    order = sorted(range(3), key=lambda i: values[i])
    sorted_values = [values[i] for i in order]
    sorted_columns = [columns[i] for i in order]

    columns = list(sorted_columns)

    # 强制右手系：det(Q) = +1。
    #
    # 特征向量的**符号是任意的**——(v) 与 (−v) 都是同一特征值的特征向量。
    # Jacobi 迭代得到的基底因此有约一半的概率是左手系（det = −1）。
    # 那对"重建张量"毫无影响（col·colᵀ 不变号），但**不是旋转而是反射**，
    # 一旦被转成四元数就会得到完全错误的结果。
    #
    # 实测症状：凸包的顶点位置正确（顶点用了共轭，反射被抵消），
    # 但世界逆惯量错到 2–13，看起来像惯量算错了，其实是旋转表示错了。
    # 翻转任意一列即可修正，且不改变张量。
    determinant = (
        columns[0][0] * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
        - columns[0][1] * (columns[1][0] * columns[2][2] - columns[1][2] * columns[2][0])
        + columns[0][2] * (columns[1][0] * columns[2][1] - columns[1][1] * columns[2][0])
    )
    if determinant < 0.0:
        columns[2] = (-columns[2][0], -columns[2][1], -columns[2][2])

    eigenvectors: Matrix3 = (
        (columns[0][0], columns[1][0], columns[2][0]),
        (columns[0][1], columns[1][1], columns[2][1]),
        (columns[0][2], columns[1][2], columns[2][2]),
    )
    return sorted_values, eigenvectors


def matrix3_to_quaternion(rotation: Matrix3) -> tuple[float, float, float, float]:
    """把正交矩阵（列为主轴）转成四元数 ``(w, x, y, z)``。

    引擎的四元数约定是 ``(w, x, y, z)``——这一点在写死的常量最容易搞错，
    所以这里显式用 Shepperd 方法取绝对值最大的分量作为除数，保证数值稳定。
    """
    m = rotation
    trace = m[0][0] + m[1][1] + m[2][2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s

    length = math.sqrt(w * w + x * x + y * y + z * z)
    if length <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / length, x / length, y / length, z / length)


# ---------------------------------------------------------------------------
# 质量属性容器
# ---------------------------------------------------------------------------

@dataclass
class MassProperties:
    """一个刚体的质量属性。

    属性
    ----
    mass           : 质量（kg）
    center_of_mass : 质心在**输入坐标系**中的位置
    inertia        : 关于质心的完整 3×3 惯量张量（输入坐标系）
    volume         : 体积（m³）；点云输入时为 None（未提供时无法得知）
    source         : 来源标记，用于日志（``"mesh"`` / ``"points"``）

    ``inertia`` 是**关于质心**的，不是关于输入原点的。这是最容易搞错的地方：
    关于原点的张量会随原点选择变化，没有物理意义。
    """

    mass: float
    center_of_mass: tuple[float, float, float]
    inertia: Matrix3
    volume: float | None = None
    source: str = "mesh"
    #: 实际施加的修正模式（``"none"`` / ``"nn"`` / ``"volume"``）。
    #: ``correction="auto"`` 会在这里给出**最终决定**，便于核对自动判据是否合理。
    correction_applied: str = "none"
    #: 是否使用了**非均匀**点权（即密度有分布，而不是恒定）。
    #: 等质量点云恒为 ``False`` —— 那意味着密度均匀，无法表达梯度。
    weighted: bool = False

    # -- 对角与主轴 -------------------------------------------------------

    @property
    def diagonal(self) -> tuple[float, float, float]:
        """张量的对角元 ``(Ixx, Iyy, Izz)``。

        只有当惯量积为零时它才等于主惯量。对称形状（球、轴对齐盒）满足，
        任意形状不满足——**不能直接拿它当主惯量用**。
        """
        return (self.inertia[0][0], self.inertia[1][1], self.inertia[2][2])

    def products(self) -> tuple[float, float, float]:
        """惯量积 ``(Ixy, Iyz, Ixz)``，即张量的三个非对角元。

        恒为零说明张量已经是对角阵；非零则说明必须做主轴分解。
        """
        return (self.inertia[0][1], self.inertia[1][2], self.inertia[0][2])

    def principal(
        self,
    ) -> tuple[tuple[float, float, float], Matrix3]:
        """求主惯量与主轴。

        返回
        ----
        (moments, axes)
            ``moments`` 升序；``axes`` 的第 i 列是第 i 个主惯量对应的主轴单位向量。

        对任意刚体，``I = axes · diag(moments) · axesᵀ``。
        """
        return jacobi_eigen_3x3(self.inertia)

    def principal_quaternion(self) -> tuple[float, float, float, float]:
        """从输入坐标系转到主轴系的旋转四元数。"""
        _, axes = self.principal()
        return matrix3_to_quaternion(axes)

    def to_engine(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """转成引擎需要的``(对角惯量, 主轴旋转)``。

        引擎只存对角惯量。把物体几何旋转到主轴系之后，
        ``I_world = (R·Q)·diag(Λ)·(R·Q)ᵀ`` 依然成立，所以对角实现无需改动。
        """
        moments, axes = self.principal()
        return (
            (moments[0], moments[1], moments[2]),
            matrix3_to_quaternion(axes),
        )

    def is_diagonal(self, tolerance: float = 1e-9) -> bool:
        """判断张量是否已是对角阵（相对最大对角元）。"""
        scale = max(abs(self.inertia[0][0]), abs(self.inertia[1][1]),
                    abs(self.inertia[2][2]), 1e-300)
        return all(abs(value) <= tolerance * scale for value in self.products())


# ---------------------------------------------------------------------------
# 闭合网格 → 质量属性
# ---------------------------------------------------------------------------

def from_mesh(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    *,
    mass: float | None = None,
    density: float | None = None,
    source: str = "mesh",
) -> MassProperties:
    """从闭合三角网格求质量属性（散度定理，解析精确）。

    参数
    ----
    vertices : 顶点列表，每个是 ``(x, y, z)``
    faces    : 三角面列表，每个是三个顶点下标（**缠绕方向须一致且朝外**）
    mass     : 总质量；与 ``density`` 二选一
    density  : 密度 kg/m³；与 ``mass`` 二选一

    异常
    -----
    ValueError : 缺少质量参数、下标越界、或网格体积退化（可能未闭合/缠绕不一致）

    说明
    ----
    ``mass`` 与 ``density`` 给一个即可：

    - 给 ``mass``：密度由 ``mass / volume`` 反推
    - 给 ``density``：质量由 ``density × volume`` 算

    两者都给会以 ``mass`` 为准并忽略 ``density``（避免冲突时行为含糊）。
    """
    if mass is None and density is None:
        raise ValueError("from_mesh 需要 mass 或 density 之一")
    if len(faces) == 0:
        raise ValueError("from_mesh 的 faces 为空")

    count = len(vertices)
    points = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]

    volume = 0.0
    first_moment = [0.0, 0.0, 0.0]
    # 二阶矩矩阵 S = ∫ x xᵀ dV（关于输入原点）
    second = [[0.0] * 3 for _ in range(3)]

    for face in faces:
        if len(face) != 3:
            raise ValueError(f"from_mesh 只支持三角形面，收到 {len(face)} 个顶点")
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        if not (0 <= i0 < count and 0 <= i1 < count and 0 <= i2 < count):
            raise ValueError(f"from_mesh 的 face 下标越界：{face}")

        a = points[i0]
        b = points[i1]
        c = points[i2]

        # 有符号体积：det(a, b, c) / 6
        cross_x = b[1] * c[2] - b[2] * c[1]
        cross_y = b[2] * c[0] - b[0] * c[2]
        cross_z = b[0] * c[1] - b[1] * c[0]
        signed_volume = (a[0] * cross_x + a[1] * cross_y + a[2] * cross_z) / 6.0
        volume += signed_volume

        # 一阶矩：四面体 (0,a,b,c) 的质心是 (a+b+c)/4
        for k in range(3):
            first_moment[k] += signed_volume * (a[k] + b[k] + c[k]) / 4.0

        # 二阶矩：S_tet = V/20 · [Σ vᵢvᵢᵀ + (Σ vᵢ)(Σ vᵢ)ᵀ]，v₀ = 0
        total = (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])
        coefficient = signed_volume / 20.0
        for row in range(3):
            for col in range(3):
                diagonal_sum = a[row] * a[col] + b[row] * b[col] + c[row] * c[col]
                second[row][col] += coefficient * (
                    diagonal_sum + total[row] * total[col]
                )

    if abs(volume) <= _VOLUME_EPS:
        raise ValueError(
            f"网格有符号体积退化（{volume:.3e}）：可能未闭合，"
            f"或三角面缠绕方向不一致"
        )
    if volume < 0.0:
        # 整体朝内缠绕：翻转符号即可，不必报错
        volume = -volume
        first_moment = [-value for value in first_moment]
        second = [[-value for value in row] for row in second]

    center = (
        first_moment[0] / volume,
        first_moment[1] / volume,
        first_moment[2] / volume,
    )

    if mass is None:
        total_mass = float(density) * volume
    else:
        total_mass = float(mass)
    if total_mass <= 0.0:
        raise ValueError(f"质量必须为正数，收到：{total_mass}")

    # 平移到质心：S_c = S − V·c cᵀ
    shifted = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for col in range(3):
            shifted[row][col] = second[row][col] - volume * center[row] * center[col]

    # 惯量 I = tr(S_c)·1 − S_c，再按密度换算成质量单位
    trace = shifted[0][0] + shifted[1][1] + shifted[2][2]
    scale = total_mass / volume
    inertia: Matrix3 = tuple(
        tuple(
            scale * ((trace if row == col else 0.0) - shifted[row][col])
            for col in range(3)
        )
        for row in range(3)
    )

    return MassProperties(
        mass=total_mass,
        center_of_mass=center,
        inertia=inertia,
        volume=volume,
        source=source,
    )


# ---------------------------------------------------------------------------
# 等质量点云 → 质量属性
# ---------------------------------------------------------------------------

def _mean_neighbor_distance(points, count: int, cell_size: float) -> list[float]:
    """估计每个点到最近 ``count`` 个邻居的平均距离（局部采样间距）。

    为什么需要它：修正项 ``ΔI = (1/6)Σmᵢhᵢ²`` 里的 ``hᵢ`` 是**每点代表的
    微小体积的等效边长**。等质量点云里这个量不直接存在，只能用局部点距估计。

    实现用均匀网格分桶把复杂度从 O(N²) 降到约 O(N)。格子边长取得比典型间距大
    一些，保证 3×3×3 邻域内通常已经够 ``count`` 个邻居。
    """
    total = len(points)
    if total <= count:
        return [cell_size] * total

    buckets: dict[tuple[int, int, int], list[int]] = {}
    inverse = 1.0 / cell_size
    for index, point in enumerate(points):
        key = (
            int(math.floor(point[0] * inverse)),
            int(math.floor(point[1] * inverse)),
            int(math.floor(point[2] * inverse)),
        )
        buckets.setdefault(key, []).append(index)

    def neighbors_within(px, py, pz, radius_cells):
        """收集邻域内的候选下标。"""
        base = (
            int(math.floor(px * inverse)),
            int(math.floor(py * inverse)),
            int(math.floor(pz * inverse)),
        )
        found = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    bucket = buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if bucket:
                        found.extend(bucket)
        return found

    result = []
    for index, point in enumerate(points):
        # 逐级扩大搜索半径，直到候选数足够。
        # 不能只看固定半径就取"最近的 k 个"：如果格子比真实间距小，
        # 真正的近邻可能落在搜索范围之外，于是"最近的 k 个"里会混进远处的点，
        # 把间距估计得偏大（实测立方点阵上偏大 8%，直接导致修正过量）。
        radius = 1
        candidates = neighbors_within(point[0], point[1], point[2], radius)
        while len(candidates) <= count and radius < 5:
            radius += 1
            candidates = neighbors_within(point[0], point[1], point[2], radius)

        distances = []
        for other in candidates:
            if other == index:
                continue
            q = points[other]
            dx = point[0] - q[0]
            dy = point[1] - q[1]
            dz = point[2] - q[2]
            distances.append(math.sqrt(dx * dx + dy * dy + dz * dz))

        if len(distances) < count:
            # 极端稀疏（点数接近 k）时的兜底
            result.append(cell_size)
            continue

        distances.sort()
        selected = distances[:count]
        result.append(sum(selected) / count)

    return result


def _spacing_statistics(points, count: int) -> tuple[list[float], float]:
    """返回每点的近邻平均间距，以及这些间距的**变异系数**。

    变异系数（标准差 / 均值）用来区分规则采样与随机采样：

    - 立方点阵：所有近邻距离严格相同 → 系数 **0**
    - 泊松随机点：近邻距离服从近似指数分布 → 系数 **0.2–0.6**

    这个统计量决定了是否需要施加"微元自身惯量"修正，见 ``from_points``。
    """
    total = len(points)
    spans = [max(p[k] for p in points) - min(p[k] for p in points) for k in range(3)]

    # 用 span/(n-1)：立方点阵的点位于格子中心，相邻间距恰为 span/(n-1)。
    axis_count = max(1.0, total ** (1.0 / 3.0) - 1.0)
    cell_size = max(max(spans) / axis_count, 1e-9)

    spacing = _mean_neighbor_distance(points, count, cell_size)

    mean = sum(spacing) / len(spacing)
    if mean <= 0.0:
        return spacing, 0.0
    variance = sum((value - mean) ** 2 for value in spacing) / len(spacing)
    return spacing, math.sqrt(variance) / mean


def from_points(
    points: Sequence[Sequence[float]],
    mass: float,
    *,
    correction: str = "auto",
    neighbors: int = 1,
    volume: float | None = None,
    weights: Sequence[float] | None = None,
    source: str = "points",
) -> MassProperties:
    """从点云求质心与惯量张量。

    参数
    ----
    points     : 点列表
    mass       : 总质量
    weights    : **每点的相对质量**（可选）。给出时按比例归一化到 ``mass``，
                 于是可以表达**非均匀密度**。``None`` 表示等质量点
                 （``mᵢ = mass / N``），此时密度恒定。
    correction : 修正方式
                 ``"none"``   − 不加修正（用于展示原始偏差）
                 ``"nn"``     − 用局部近邻间距估计 hᵢ
                 ``"volume"`` − 用已知总体积估计 h = (V/N)^(1/3)（需传 ``volume``）
    neighbors  : ``"nn"`` 模式下用于估计间距的近邻个数
    volume     : ``"volume"`` 模式所需的总体积

    ``"nn"`` 是已知有缺陷的，待替换
    -------------------------------
    它把**最近邻距离**直接当成 hᵢ，这实际上假设了点呈规则排布：
    同样密度下，立方点阵的 ``r₁ = h``，而三维泊松过程 ``E[r₁] = 0.5539h``，
    换算常数差 **1.8 倍**。所以对非规则点云它有系统性偏差。

    正确的做法是估计**局部体积密度**而不是邻居间距，规格见
    ``docs/12_点云截断修正的算法路径.md``。在实现之前，``"nn"`` 只应
    用于接近规则排列的点集（``"auto"`` 会据此判断）。

    ``"volume"`` 不是缺陷
    --------------------
    它假设每点代表等体积 V/N，这等价于"点按密度成比例撒布、质量相等"。
    那种情况下它**精确**，而且不含任何排布假设，是本文修正的一个特例。

    等质量还是加权：这是一个必须想清楚的选择
    ----------------------------------------
    **等质量点 + 规则点阵 ⟹ 密度恒定。** 每格体积都是 h³、每点质量都是 M/N，
    密度 = m/h³ 是常数。所以"梯度密度"在这个组合下**根本没有自由度**，
    不是精度不够，而是模型里没这个东西。

    表示密度分布有两条路，互斥：

    =============== ================== ===================== ============
    方案             位置               权重                   密度梯度
    =============== ================== ===================== ============
    等质量 + 非均匀采样 点向稠密处聚集    等质量                 可表达（精度靠 N）
    **加权 + 规则网格** 规则格中心         每格质量 mᵢ            **可表达且精确可修正**
    =============== ================== ===================== ============

    本函数两种都支持。加权路径（``weights``）通常更好：它是"体素密度场"，
    每个格子是一个体积为 h³ 的微元，密度可以在格内连续取值。

    空心/空腔**在等质量方案里也能表达**——点只放在实体所在处，
    空腔位置没有格子中心，自然没有质量。所以"能不能做空心"和
    "能不能做密度梯度"是两个不同的问题，别混在一起。

    修正的理论依据
    -------------
    估计式漏掉了每个微元自身的惯量。立方微元（边长 h、质量 m）对自身质心
    任意轴的惯量是 ``m h²/6``，所以

        ΔI = (1/6) Σ mᵢ hᵢ² · 1

    这个修正**只加对角线**、各向同性，不改变惯量积。

    注意它对权重分布**不敏感**：规则点阵上 ``Σ mᵢ = mass``，
    所以 ``ΔI = (h²/6)·Σmᵢ = mass·h²/6``，与质量怎么分布无关。
    换句话说，加上权重不破坏"精确可修正"这个性质。
    """
    total = len(points)
    if total == 0:
        raise ValueError("点云不能为空")
    if mass <= 0.0:
        raise ValueError(f"质量必须为正数，收到：{mass}")

    coords = [(float(p[0]), float(p[1]), float(p[2])) for p in points]

    # --- 每点质量 ---
    if weights is None:
        masses = None
        particle_mass = mass / total
    else:
        if len(weights) != total:
            raise ValueError(
                f"weights 长度 {len(weights)} 与点数 {total} 不一致"
            )
        cleaned = [float(w) for w in weights]
        if any(w < 0.0 for w in cleaned):
            raise ValueError("weights 不能为负")
        total_weight = sum(cleaned)
        if total_weight <= 0.0:
            raise ValueError("weights 之和必须为正")
        # 归一化到给定的总质量：用户给"密度形状"，总质量由 mass 定
        scale = mass / total_weight
        masses = [w * scale for w in cleaned]
        particle_mass = 0.0   # 加权模式下不使用

    # 质心
    if masses is None:
        center = (
            sum(p[0] for p in coords) / total,
            sum(p[1] for p in coords) / total,
            sum(p[2] for p in coords) / total,
        )
    else:
        center = (
            sum(masses[i] * coords[i][0] for i in range(total)) / mass,
            sum(masses[i] * coords[i][1] for i in range(total)) / mass,
            sum(masses[i] * coords[i][2] for i in range(total)) / mass,
        )

    # 二阶矩（关于质心）：S = Σ mᵢ rᵢrᵢᵀ
    second = [[0.0] * 3 for _ in range(3)]
    for index, p in enumerate(coords):
        weight = particle_mass if masses is None else masses[index]
        rx = p[0] - center[0]
        ry = p[1] - center[1]
        rz = p[2] - center[2]
        offset = (rx, ry, rz)
        for row in range(3):
            for col in range(3):
                second[row][col] += weight * offset[row] * offset[col]

    trace = second[0][0] + second[1][1] + second[2][2]
    inertia = [
        [
            (trace if row == col else 0.0) - second[row][col]
            for col in range(3)
        ]
        for row in range(3)
    ]

    # --- 修正项 ---
    mode = correction.lower()
    if mode not in ("none", "volume", "nn", "auto"):
        raise ValueError(
            f"未知的 correction 模式 {correction!r}；"
            f"可选 'auto' / 'none' / 'nn' / 'volume'"
        )

    spacing = None
    variation = 0.0
    if mode in ("nn", "auto"):
        spacing, variation = _spacing_statistics(coords, int(neighbors))

    if mode == "auto":
        # 只有**规则采样**（点阵、分层、体素中心）才需要修正。
        # i.i.d. 随机采样的估计量本身无偏，加修正反而引入正偏差。
        mode = "nn" if variation < _REGULAR_VARIATION_THRESHOLD else "none"

    if mode == "none":
        pass
    elif mode == "volume":
        if volume is None or volume <= 0.0:
            raise ValueError("correction='volume' 需要提供正的 volume")
        # 每点代表 V/N 的体积 ⇒ 等效边长 h = (V/N)^(1/3)。
        # 修正量 = (h²/6)·Σmᵢ，而 Σmᵢ = mass，所以与权重分布无关。
        edge = (float(volume) / total) ** (1.0 / 3.0)
        delta = mass * edge * edge / 6.0
        for axis in range(3):
            inertia[axis][axis] += delta
    else:  # nn
        # 这里必须逐点用 mᵢ：加权点云的质量不均匀，
        # 若仍用 mass/N 会把修正量摊错。
        if masses is None:
            delta = sum(particle_mass * s * s / 6.0 for s in spacing)
        else:
            delta = sum(masses[i] * spacing[i] ** 2 / 6.0 for i in range(total))
        for axis in range(3):
            inertia[axis][axis] += delta

    tensor: Matrix3 = tuple(tuple(row) for row in inertia)
    return MassProperties(
        mass=float(mass),
        center_of_mass=center,
        inertia=tensor,
        volume=float(volume) if volume is not None else None,
        source=source,
        correction_applied=mode,
        weighted=masses is not None,
    )
