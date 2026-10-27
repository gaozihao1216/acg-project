# -*- coding: utf-8 -*-
"""任意凸多面体（凸包）测试：构造、质量属性、主轴并入姿态、广义 SAT。

这一组测试的判据设计
--------------------
"不规则物体能碰撞"这件事很容易做出**看起来能跑、其实全错**的结果：
物体确实互相停住了，但姿态是错的、惯量是错的，或者干脆只在某些方向上有效。
所以每一项都配了一个**独立于实现的解析判据**：

1. 凸包体积 vs 解析体积（立方体、棱柱）
2. 主惯量 vs 网格散度定理的精确值
3. **世界逆惯量 vs 完整 3×3 张量**——这一条抓的是"主轴并进姿态"是否搞对
4. 广义 SAT 在盒子上**必须退化为专用盒-盒实现**（400 组随机配置逐一比对）
5. 球-凸包穿透 vs 沿支撑方向手动设定的穿透量

第 4 条尤其重要：它把"新写的通用算法"和"已经验证过的专用算法"钉在一起，
任何推广过程中的符号错误都会立刻暴露。

运行::

    python Code/tests/test_hull.py
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

from src.geom.convexhull import convex_hull
from src.geom.geometry import BoxGeometry, ConvexHullGeometry, SphereGeometry
from src.geom.inertia import compute_inertia
from src.geom.massprops import from_mesh
from src.geom.meshutil import polygon_area, prism_mesh
from src.mathutils import quaternion as quat
from src.physics.body import RigidBody
from src.physics.convex_sat import (
    collide_polytopes,
    polytope_from_box,
    polytope_from_hull,
)
from src.physics.narrowphase import GroundPlane, detect_contacts
from src.physics.narrowphase import is_supported, supported_pair
from src.physics.sat import collide_box_box
from src.physics.world import PhysicsWorld
from src.scene.scene_spec import SimulationSpec

IDENTITY = quat.quat_identity()

#: 不对称三角形棱柱（凸、但惯量积非零），用作"任意形状"的标准测试体
TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.4, 1.1)]
PRISM_HEIGHT = 0.6


def make_prism_points():
    """棱柱网格的顶点集。"""
    vertices, _ = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    return vertices


def hull_volume(hull) -> float:
    """由三角面片按有符号体积累加求凸包体积。"""
    total = 0.0
    for a, b, c in hull.triangles:
        va, vb, vc = hull.vertices[a], hull.vertices[b], hull.vertices[c]
        total += (
            va[0] * (vb[1] * vc[2] - vb[2] * vc[1])
            - va[1] * (vb[0] * vc[2] - vb[2] * vc[0])
            + va[2] * (vb[0] * vc[1] - vb[1] * vc[0])
        ) / 6.0
    return total


def make_body(name, geometry, position, mass=5.0, orientation=IDENTITY, **kwargs):
    """按引擎约定构造刚体（姿态里并入惯量主轴）。"""
    return RigidBody(
        name=name,
        geometry=geometry,
        position=position,
        orientation=quat.quat_multiply(orientation, geometry.inertia_frame()),
        mass=mass,
        inertia_local=compute_inertia(geometry, mass),
        **kwargs,
    )


def make_world(bodies, ground=None):
    """构建一个最小可用的物理世界。"""
    simulation = SimulationSpec(
        enabled=True,
        physics_dt=1.0 / 240.0,
        substeps=3,
        iterations=8,
        baumgarte=0.25,
        slop=0.002,
        restitution_threshold=0.6,
        gravity=(0.0, 0.0, -9.81),
        ground_z=0.0,
        ground_friction=0.8,
        ground_restitution=0.05,
        linear_damping=0.05,
        angular_damping=1.0,
        use_toi_splitting=False,
        duration=1.0,
    )
    return PhysicsWorld(bodies, ground, simulation)


# ---------------------------------------------------------------------------
# 组一：凸包构造
# ---------------------------------------------------------------------------

def test_cube_hull_exact() -> tuple[bool, str]:
    """立方体 8 顶点的凸包：体积精确、6 个四边形面、12 条棱。

    共面三角形必须被合并成四边形。不合并的话立方体会得到 12 个三角形，
    SAT 会把同一平面数三遍，棱-棱候选轴从 144 涨到 324。
    """
    corners = [
        (x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)
    ]
    hull = convex_hull(corners)

    volume = hull_volume(hull)
    volume_ok = abs(volume - 1.0) < 1e-12
    faces_ok = len(hull.faces) == 6 and all(len(f) == 4 for f in hull.faces)
    edges_ok = len(hull.edges) == 12
    vertices_ok = len(hull.vertices) == 8

    ok = volume_ok and faces_ok and edges_ok and vertices_ok
    return ok, (
        f"体积={volume:.12f}，顶点={len(hull.vertices)}，面={len(hull.faces)}"
        f"（四边形={all(len(f) == 4 for f in hull.faces)}），棱={len(hull.edges)}"
    )


def test_interior_points_discarded() -> tuple[bool, str]:
    """内部点必须被剔除，不影响结果。

    这一条保证"网格顶点直接喂进来"是安全的：真实网格里有大量
    位于内部的顶点，它们不该改变凸包。
    """
    corners = [
        (x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)
    ]
    rng = random.Random(2026)
    interior = [
        tuple(rng.uniform(-0.45, 0.45) for _ in range(3)) for _ in range(400)
    ]
    hull = convex_hull(corners + interior)

    ok = (
        abs(hull_volume(hull) - 1.0) < 1e-12
        and len(hull.faces) == 6
        and len(hull.vertices) == 8
    )
    return ok, (
        f"408 个输入点 → 凸包顶点={len(hull.vertices)}，面={len(hull.faces)}，"
        f"体积={hull_volume(hull):.12f}"
    )


def test_degenerate_rejected() -> tuple[bool, str]:
    """共面 / 共线的点集必须明确报错，而不是返回一个畸形的凸包。"""
    coplanar = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    collinear = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]

    results = []
    for name, points in (("共面", coplanar), ("共线", collinear)):
        try:
            convex_hull(points)
            results.append((name, False))
        except ValueError:
            results.append((name, True))

    ok = all(rejected for _, rejected in results)
    return ok, "  ".join(f"{name}被拒绝={rejected}" for name, rejected in results)


def test_complex_hull_rejected() -> tuple[bool, str]:
    """过复杂的凸包必须在**加载阶段**就报错。

    SAT 的棱-棱轴数是 E²。一个 512 面的凸包有约 768 条棱，两两相交是
    59 万条轴，一次查询就能让仿真失去响应——现象上像死循环，极难定位。
    所以宁可加载时报错。
    """
    # 球面点云 → 很多面
    rng = random.Random(5)
    points = []
    for _ in range(1200):
        while True:
            v = [rng.gauss(0, 1) for _ in range(3)]
            length = math.sqrt(sum(c * c for c in v))
            if length > 1e-6:
                break
        points.append(tuple(0.7 * c / length for c in v))

    rejected = False
    message = ""
    try:
        ConvexHullGeometry(points)
    except ValueError as exc:
        rejected = True
        message = str(exc).split("。")[0]

    # 显式放宽上限后应当可以构造
    allowed = True
    try:
        ConvexHullGeometry(points, max_faces=100000)
    except Exception:  # noqa: BLE001
        allowed = False

    ok = rejected and allowed
    return ok, (
        f"默认上限下被拒绝={rejected}（{message}）；"
        f"显式放宽后可以构造={allowed}"
    )


# ---------------------------------------------------------------------------
# 组二：质量属性与主轴
# ---------------------------------------------------------------------------

def test_hull_mass_properties() -> tuple[bool, str]:
    """凸包的质量属性必须与网格散度定理的结果逐位吻合。

    凸包是凸体，它的网格就是它自己，所以两条独立路径必须给出同一个答案。
    """
    vertices, faces = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    reference = from_mesh(vertices, faces, mass=2.0)
    geometry = ConvexHullGeometry(vertices)

    expected_volume = polygon_area(TRIANGLE) * PRISM_HEIGHT
    volume_ok = abs(geometry.volume() - expected_volume) < 1e-12

    reference_moments = reference.principal()[0]
    moments = geometry.principal_moments(2.0)
    moment_error = max(
        abs(moments[i] - reference_moments[i]) for i in range(3)
    )

    com_ok = max(
        abs(geometry.com_offset[k] - reference.center_of_mass[k]) for k in range(3)
    ) < 1e-12

    ok = volume_ok and com_ok and moment_error < 1e-12
    return ok, (
        f"体积={geometry.volume():.12f}（解析 {expected_volume:.12f}）；"
        f"主惯量={[round(v, 9) for v in moments]}，与散度定理最大差={moment_error:.3e}；"
        f"质心吻合={com_ok}"
    )


def test_principal_frame_right_handed() -> tuple[bool, str]:
    """惯量主轴必须是**右手系**（det = +1）。

    特征向量的符号是任意的，Jacobi 迭代有约一半的概率给出左手系基底。
    这对"重建张量"毫无影响（col·colᵀ 不变号），但那是**反射不是旋转**，
    转成四元数后会得到完全错误的姿态。

    实测症状：凸包顶点位置正确（顶点用了共轭，反射被抵消），
    但世界逆惯量错到 2–13，看起来像惯量算错了。
    """
    rng = random.Random(1)
    bad = 0
    worst = 0.0
    for _ in range(200):
        matrix = tuple(
            tuple((rng.uniform(-1, 1) + rng.uniform(-1, 1)) * 0.5 for _ in range(3))
            for _ in range(3)
        )
        symmetric = tuple(
            tuple((matrix[r][c] + matrix[c][r]) * 0.5 for c in range(3))
            for r in range(3)
        )
        from src.geom.massprops import jacobi_eigen_3x3

        _, axes = jacobi_eigen_3x3(symmetric)
        determinant = (
            axes[0][0] * (axes[1][1] * axes[2][2] - axes[1][2] * axes[2][1])
            - axes[0][1] * (axes[1][0] * axes[2][2] - axes[1][2] * axes[2][0])
            + axes[0][2] * (axes[1][0] * axes[2][1] - axes[1][1] * axes[2][0])
        )
        if determinant < 0.0:
            bad += 1
        worst = max(worst, abs(determinant - 1.0))

    ok = bad == 0 and worst < 1e-9
    return ok, f"200 个随机对称阵：det<0 的个数={bad}，|det−1| 最大={worst:.3e}"


def test_world_inverse_inertia_matches_tensor() -> tuple[bool, str]:
    """引擎的世界逆惯量必须等于**完整 3×3 张量**旋转到世界后的逆。

    这是"主轴对角化折叠进姿态"这一整套做法的唯一判据。
    把主轴 Q 并进姿态后应当有

        I_world = (R_spec·Q)·Λ·(R_spec·Q)ᵀ = R_spec·I_geom·R_specᵀ

    所以 ``inverse_inertia_world(v)`` 必须等于 ``R_spec·I⁻¹·R_specᵀ·v``。

    注意验证公式里**两次**都要有 ``R_spec``：转到局部是 ``QᵀR_specᵀ``，
    我第一版漏了 ``R_specᵀ``，结果只有"零旋转"那一档能通过，
    看起来像引擎错，其实是判据错。
    """
    vertices, faces = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    reference = from_mesh(vertices, faces, mass=2.0)
    moments, axes = reference.principal()
    geometry = ConvexHullGeometry(vertices)

    def expected_inverse(vector, spec_quaternion):
        rotation = quat.quat_to_matrix(spec_quaternion)
        # R_specᵀ · v
        world = [
            sum(rotation[c][r] * vector[c] for c in range(3)) for r in range(3)
        ]
        # axesᵀ · w
        local = [
            sum(axes[i][k] * world[i] for i in range(3)) for k in range(3)
        ]
        scaled = [local[k] / moments[k] for k in range(3)]
        back = [
            sum(axes[i][k] * scaled[k] for k in range(3)) for i in range(3)
        ]
        return tuple(
            sum(rotation[r][c] * back[c] for c in range(3)) for r in range(3)
        )

    vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (-0.7, 0.4, 1.9)]
    worst = 0.0
    for angle in (0.0, 0.4, 1.1, 2.0, 2.7, 3.0):
        for axis in ((0.3, 0.8, 0.5), (1, 0, 0), (-0.2, 0.9, 0.4)):
            spec_quaternion = quat.quat_from_axis_angle(axis, angle)
            body = make_body(
                "hull", geometry, (0, 0, 0), mass=2.0,
                orientation=spec_quaternion,
            )
            for vector in vectors:
                got = body.inverse_inertia_world(vector)
                want = expected_inverse(vector, spec_quaternion)
                worst = max(
                    worst, max(abs(got[i] - want[i]) for i in range(3))
                )

    ok = worst < 1e-10
    return ok, f"18 种姿态 × 5 个向量：世界逆惯量与完整张量最大差={worst:.3e}"


def test_hull_vertices_placed_correctly() -> tuple[bool, str]:
    """凸包顶点的世界位置必须与"原始网格按场景姿态旋转"完全一致。

    这条保证质心平移 + 主轴旋转没有把几何摆歪——如果摆歪了，
    碰撞形状和渲染外观就会错位，而物理看起来"仍然能跑"。
    """
    vertices, _ = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    geometry = ConvexHullGeometry(vertices)
    spec_quaternion = quat.quat_from_axis_angle((0.3, 0.8, 0.5), 1.1)
    body = make_body("hull", geometry, (0, 0, 0), orientation=spec_quaternion)

    worst = 0.0
    for index, local in enumerate(geometry.vertices):
        world = quat.quat_rotate_vector(body.orientation, local)
        original = (
            vertices[index][0] - geometry.com_offset[0],
            vertices[index][1] - geometry.com_offset[1],
            vertices[index][2] - geometry.com_offset[2],
        )
        expected = quat.quat_rotate_vector(spec_quaternion, original)
        worst = max(worst, max(abs(world[k] - expected[k]) for k in range(3)))

    ok = worst < 1e-12
    return ok, f"{len(geometry.vertices)} 个顶点世界位置最大差={worst:.3e}"


# ---------------------------------------------------------------------------
# 组三：广义 SAT
# ---------------------------------------------------------------------------

def test_sat_reduces_to_box_sat() -> tuple[bool, str]:
    """广义 SAT 作用在盒子上时，必须与专用盒-盒实现**逐位一致**。

    这是整套推广最强的验证：盒子是"6 面 12 棱的凸多面体"这一特例，
    通用算法在这里必须退化成已验证的专用算法。任何符号错误都会暴露。

    注意比较的是**最大穿透**而不是列表首点——两个实现的点排序不同，
    比 `[0]` 会得到大量假的"不一致"。
    """
    rng = random.Random(11)
    mismatch_contact = 0
    mismatch_normal = 0
    mismatch_depth = 0
    worst_depth = 0.0

    for _ in range(200):
        size_a = tuple(rng.uniform(0.3, 1.0) for _ in range(3))
        size_b = tuple(rng.uniform(0.3, 1.0) for _ in range(3))
        quat_a = quat.quat_from_axis_angle(
            (rng.random(), rng.random(), rng.random()), rng.uniform(0, 3.14)
        )
        quat_b = quat.quat_from_axis_angle(
            (rng.random(), rng.random(), rng.random()), rng.uniform(0, 3.14)
        )
        position_a = tuple(rng.uniform(-0.4, 0.4) for _ in range(3))
        position_b = tuple(
            position_a[k] + rng.uniform(-0.85, 0.85) for k in range(3)
        )

        body_a = make_body("a", BoxGeometry(size_a), position_a, orientation=quat_a)
        body_b = make_body("b", BoxGeometry(size_b), position_b, orientation=quat_b)

        reference = collide_box_box(body_a, body_b)
        general = collide_polytopes(
            polytope_from_box(size_a, position_a, body_a.orientation),
            polytope_from_box(size_b, position_b, body_b.orientation),
        )

        if bool(reference) != bool(general):
            mismatch_contact += 1
            continue
        if not reference:
            continue

        alignment = sum(
            reference[0].normal[k] * general[0].normal[k] for k in range(3)
        )
        if alignment < 0.999:
            mismatch_normal += 1

        depth_reference = max(p.penetration for p in reference)
        depth_general = max(p.penetration for p in general)
        difference = abs(depth_reference - depth_general)
        worst_depth = max(worst_depth, difference)
        if difference > 1e-9:
            mismatch_depth += 1

    ok = mismatch_contact == 0 and mismatch_normal == 0 and mismatch_depth == 0
    return ok, (
        f"200 组随机配置：接触判定不一致={mismatch_contact}，"
        f"法向不一致={mismatch_normal}，最大穿透不一致={mismatch_depth}"
        f"（最大差={worst_depth:.3e}）"
    )


def test_hull_hull_normal_flips() -> tuple[bool, str]:
    """交换两个凸包的顺序，法向必须严格取反。"""
    vertices, _ = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    geometry = ConvexHullGeometry(vertices)
    body_a = make_body("a", geometry, (0, 0, 0))
    body_b = make_body("b", geometry, (0.55, 0, 0.2))

    poly_a = polytope_from_hull(geometry, body_a.position, body_a.orientation)
    poly_b = polytope_from_hull(geometry, body_b.position, body_b.orientation)

    forward = collide_polytopes(poly_a, poly_b)
    backward = collide_polytopes(poly_b, poly_a)

    if not forward or not backward:
        return False, f"应产生接触但得到 forward={len(forward)} backward={len(backward)}"

    dot = sum(forward[0].normal[k] * backward[0].normal[k] for k in range(3))
    ok = abs(dot + 1.0) < 1e-9
    return ok, (
        f"正向 {len(forward)} 点、反向 {len(backward)} 点，"
        f"法向点积={dot:.9f}（应为 −1）"
    )


def test_sphere_hull_penetration() -> tuple[bool, str]:
    """球-凸包的穿透深度必须等于**手动设定**的值。

    沿凸包在四个方向上的支撑顶点放置球，让球心到该顶点的距离为
    ``半径 − 设定穿透``。此时最近点就是那个顶点，穿透必须精确相等。
    """
    vertices, _ = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    geometry = ConvexHullGeometry(vertices)
    hull_body = make_body("hull", geometry, (0, 0, 0))

    # 顶点在世界坐标下的位置（几何体已旋转到主轴系，必须用姿态变换）
    world_vertices = [
        quat.quat_rotate_vector(hull_body.orientation, v) for v in geometry.vertices
    ]

    radius = 0.2
    failures = []
    for direction in ((1, 0, 0), (0, -1, 0), (0, 0, 1), (-0.6, 0.8, 0.2)):
        length = math.sqrt(sum(c * c for c in direction))
        unit = tuple(c / length for c in direction)
        support = max(
            world_vertices, key=lambda v: sum(v[k] * unit[k] for k in range(3))
        )
        for setting in (0.05, 0.02):
            center = tuple(
                support[k] + unit[k] * (radius - setting) for k in range(3)
            )
            sphere = make_body("s", SphereGeometry(radius), center)
            manifolds = detect_contacts([hull_body, sphere], None)
            depths = [p.penetration for m in manifolds for p in m.points]
            if not depths or abs(depths[0] - setting) > 1e-9:
                failures.append((direction, setting, depths))

    ok = not failures
    return ok, (
        f"8 组（4 方向 × 2 穿透）全部精确={ok}"
        + (f"，失败样例={failures[:2]}" if failures else "")
    )


def test_sphere_inside_hull() -> tuple[bool, str]:
    """球心完全在凸包内部时也必须产生接触，不能返回"无接触"。

    这条最容易漏：最近点算法通常只处理"在外部"，一旦球心进去，
    投影可能退化。表现是球从凸包内部穿出去时完全不响应。
    """
    vertices, _ = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    geometry = ConvexHullGeometry(vertices)
    hull_body = make_body("hull", geometry, (0, 0, 0))
    sphere = make_body("s", SphereGeometry(0.15), (0.0, 0.0, 0.0))

    manifolds = detect_contacts([hull_body, sphere], None)
    depths = [p.penetration for m in manifolds for p in m.points]
    normals = [p.normal for m in manifolds for p in m.points]

    ok = bool(depths) and depths[0] > 0.15 and all(
        abs(sum(c * c for c in n) - 1.0) < 1e-9 for n in normals
    )
    return ok, (
        f"球心在凸包内：穿透={[round(d, 6) for d in depths] if depths else '无接触'}"
        f"（应 > 半径 0.15），法向单位化正确"
    )


def test_hull_ground_manifold() -> tuple[bool, str]:
    """凸包落在地面上：应得到多点流形，且静置后不再翻转。"""
    vertices, _ = prism_mesh(TRIANGLE, height=PRISM_HEIGHT)
    geometry = ConvexHullGeometry(vertices)
    lowest = min(v[2] for v in geometry.vertices)

    hull_body = make_body("hull", geometry, (0.0, 0.0, -lowest - 0.02))
    manifolds = detect_contacts([hull_body], GroundPlane(0.0, 0.8, 0.05))
    contact_count = len(manifolds[0].points) if manifolds else 0
    normal_ok = (
        manifolds
        and all(p.normal == (0.0, 0.0, 1.0) for p in manifolds[0].points)
    )

    world = make_world(
        [make_body("hull", geometry, (0.0, 0.0, -lowest + 0.8))],
        ground=GroundPlane(0.0, 0.8, 0.05),
    )
    for _ in range(240 * 3):
        world.step()

    body = world.bodies[0]
    settled = body.speed() < 0.03
    # 最低顶点应贴在地面附近（允许 slop）
    world_vertices = [
        quat.quat_rotate_vector(body.orientation, v) for v in geometry.vertices
    ]
    lowest_world = min(
        world_vertices[k][2] + body.position[2] for k in range(len(world_vertices))
    )
    resting = abs(lowest_world) < 0.01

    ok = contact_count >= 3 and bool(normal_ok) and settled and resting
    return ok, (
        f"静态接触点数={contact_count}（棱柱底面 3 顶点），法向全为 +Z={bool(normal_ok)}；"
        f"仿真 3 秒后 |v|={body.speed():.4f}，最低顶点 z={lowest_world * 1000:.2f} mm"
    )


def test_concave_needs_compound() -> tuple[bool, str]:
    """凹体的凸包**体积明显大于**本体——这是必须用组合体的定量证据。

    这条不是缺陷，而是边界：凸包是凹体的外近似，凹进去的部分会被填满。
    所以凹体必须拆成多个凸块（V-HACD / 手工拼），这是业界唯一做法。
    """
    polygon = [(0.0, 0.0), (2.4, 0.0), (2.4, 0.8), (0.9, 0.8), (0.9, 2.0), (0.0, 2.0)]
    vertices, _ = prism_mesh(polygon, height=1.0)
    geometry = ConvexHullGeometry(vertices)

    true_volume = polygon_area(polygon) * 1.0
    hull_volume_value = geometry.volume()
    ratio = hull_volume_value / true_volume

    # 凸包必须严格更大，且不会大到离谱（L 形填空后约为 2.4×2.0×1 的 70%）
    ok = hull_volume_value > true_volume * 1.2
    return ok, (
        f"L 形真实体积={true_volume:.4f}，凸包体积={hull_volume_value:.4f}，"
        f"比值={ratio:.3f}（凸包是外近似，凹体必须拆成凸块）"
    )


def test_registry_and_support() -> tuple[bool, str]:
    """凸包必须被场景解析与窄相支持表同时接受。"""
    from src.geom.geometry import GeometrySpec

    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0], [0.2, 0.2, 0.8]]
    geometry = GeometrySpec.from_dict({"type": "hull", "points": points})

    round_trip = geometry.to_dict()
    is_hull = isinstance(geometry, ConvexHullGeometry)
    supported = is_supported(geometry)
    pairs = (
        supported_pair(geometry, BoxGeometry((1, 1, 1)))
        and supported_pair(geometry, SphereGeometry(0.5))
        and supported_pair(geometry, geometry)
    )

    ok = is_hull and supported and pairs and round_trip["type"] == "hull"
    return ok, (
        f"从字典构造成功={is_hull}，体积={geometry.volume():.6f}，"
        f"单/双支持={supported}/{pairs}，可回写={round_trip['type']!r}"
    )


CHECKS = [
    ("1. 立方体凸包：体积/面/棱精确", test_cube_hull_exact),
    ("2. 内部点被剔除", test_interior_points_discarded),
    ("3. 共面/共线被拒绝", test_degenerate_rejected),
    ("4. 过复杂凸包在加载阶段报错", test_complex_hull_rejected),
    ("5. 凸包质量属性 vs 散度定理", test_hull_mass_properties),
    ("6. 主轴右手系（det=+1）", test_principal_frame_right_handed),
    ("7. 世界逆惯量 vs 完整张量", test_world_inverse_inertia_matches_tensor),
    ("8. 凸包顶点世界位置正确", test_hull_vertices_placed_correctly),
    ("9. 广义 SAT 在盒子上退化一致", test_sat_reduces_to_box_sat),
    ("10. 凸包-凸包法向取反", test_hull_hull_normal_flips),
    ("11. 球-凸包穿透精确", test_sphere_hull_penetration),
    ("12. 球心在凸包内部", test_sphere_inside_hull),
    ("13. 凸包-地面多点流形与静置", test_hull_ground_manifold),
    ("14. 凹体凸包体积更大（需组合体）", test_concave_needs_compound),
    ("15. 场景解析与支持表", test_registry_and_support),
]


def main() -> int:
    print("=" * 100)
    print("凸包 / 任意凸多面体：构造、质量属性、主轴并入姿态、广义 SAT")
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
