# -*- coding: utf-8 -*-
"""地面接触的通用解与分派表。

第 5 步把地面接触从"每种几何一个 isinstance 分支"改成"分派表 + 支撑特征通用解"。
本文件负责证明两件事：

1. **通用解对已有几何与专属实现等价**。球走通用解必须与解析式逐位相同；
   立着的圆柱走通用解必须给出同样多的点、同样的深度（圆周点方位可以不同，
   因为 45° 起点是刻意的，见 ``support._face_ring``）。只要这条成立，就说明
   "统一"没有偷偷改变物理行为。
2. **新图元有地面接触**。圆锥、圆台、椭球、球冠此前完全没有地面实现；它们现在
   靠同一个通用解工作，且"端面圆盘正对地面"时给出 4 点而不是 1 点——单点会让
   物体绕接触点翻转。
"""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_HERE)
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from src.geom.geometry import (
    ConeGeometry,
    CylinderGeometry,
    EllipsoidGeometry,
    FrustumGeometry,
    SphereGeometry,
    SphericalCapGeometry,
)
from src.physics import capabilities
from src.physics.body import RigidBody
from src.physics.narrowphase import (
    GroundPlane,
    _GROUND_DISPATCH,
    collide_cylinder_ground,
    collide_sphere_ground,
    _part_ground_points,
    collide_convex_ground,
)

GROUND = GroundPlane(z=0.0)

#: 绕 X 轴翻 180°：把"底面朝下"的几何翻成"底面朝上"。
FLIPPED = (0.0, 1.0, 0.0, 0.0)


def make_body(geometry, z: float, orientation=(1.0, 0.0, 0.0, 0.0)) -> RigidBody:
    return RigidBody(
        name="probe",
        geometry=geometry,
        mass=1.0,
        position=(0.0, 0.0, z),
        orientation=orientation,
    )


def _depths(points) -> list[float]:
    return sorted((point.penetration for point in points), reverse=True)


# ---------------------------------------------------------------------------
# 1. 通用解与已有专属实现等价
# ---------------------------------------------------------------------------

def test_sphere_generic_matches_analytic() -> tuple[bool, str]:
    """球走通用解必须与 ``collide_sphere_ground`` 逐位相同。

    球是唯一一个能这么比的情形：解析式是闭式的，通用解走支撑特征，两条完全
    不同的代码路径给出同一个点和同一个深度，说明"表面点与地面投影的中点"这个
    通用取法没有引入偏差。
    """
    body = make_body(SphereGeometry(0.3), 0.2)
    generic = collide_convex_ground(body, GROUND)
    exact = collide_sphere_ground(body, GROUND)

    if len(generic) != 1 or exact is None:
        return False, f"点数异常：通用 {len(generic)}，解析 {'1' if exact else 'None'}"

    point = generic[0]
    same = (
        abs(point.position[0] - exact.position[0]) < 1e-15
        and abs(point.position[1] - exact.position[1]) < 1e-15
        and abs(point.position[2] - exact.position[2]) < 1e-15
        and abs(point.penetration - exact.penetration) < 1e-15
    )
    return same, (
        f"通用 z={point.position[2]:.15f} 深度={point.penetration:.15f}；"
        f"解析 z={exact.position[2]:.15f} 深度={exact.penetration:.15f}"
    )


def test_standing_cylinder_generic_matches_analytic() -> tuple[bool, str]:
    """立着的圆柱：通用解与解析解必须给出同样多的点和同样的深度集合。

    方位允许不同——通用解取 45°/135°/225°/315°，解析解以精确最低点为起点。
    两者都是 4 个等深点，物理上等价；方位固定的好处是避免接触点与坐标轴对齐
    导致摩擦方向退化。
    """
    body = make_body(CylinderGeometry(0.3, 1.0), 0.4)
    generic = collide_convex_ground(body, GROUND)
    exact = collide_cylinder_ground(body, GROUND)

    generic_depths = _depths(generic)
    exact_depths = _depths(exact)
    same = len(generic_depths) == len(exact_depths) and all(
        abs(a - b) < 1e-12 for a, b in zip(generic_depths, exact_depths)
    )
    return same, (
        f"通用 {len(generic)} 点 深度={[round(d, 6) for d in generic_depths]}；"
        f"解析 {len(exact)} 点 深度={[round(d, 6) for d in exact_depths]}"
    )


# ---------------------------------------------------------------------------
# 2. 新图元的地面接触
# ---------------------------------------------------------------------------

def test_cone_ground() -> tuple[bool, str]:
    """圆锥：底面朝下是 4 点面接触，顶点朝下是 1 点。"""
    upright = ConeGeometry(0.3, 1.2)          # 底面 z = -0.3，顶点 z = +0.9
    body = make_body(upright, 0.2)            # 底面落在 -0.1，穿透 0.1
    flat_points = collide_convex_ground(body, GROUND)

    apex_down = make_body(upright, 0.85, FLIPPED)   # 顶点落在 -0.05，穿透 0.05
    point_points = collide_convex_ground(apex_down, GROUND)

    flat_depth = max((p.penetration for p in flat_points), default=0.0)
    ok = (
        len(flat_points) == 4
        and abs(flat_depth - 0.1) < 1e-6
        and len(point_points) == 1
        and abs(point_points[0].penetration - 0.05) < 1e-6
    )
    return ok, (
        f"底面朝下 {len(flat_points)} 点 深度={[round(p.penetration, 6) for p in flat_points]}；"
        f"顶点朝下 {len(point_points)} 点 深度={[round(p.penetration, 6) for p in point_points]}"
    )


def test_frustum_ground_both_faces() -> tuple[bool, str]:
    """圆台的两个端面都是圆盘：哪个朝下都应该得到 4 点。"""
    geometry = FrustumGeometry(0.4, 0.15, 0.9)
    results = []
    # 让朝向地面的那个端面下沉 5 cm：
    #   未翻转 → 世界最低 = position.z + _bottom_z
    #   翻转 180° → 局部 +Z 指向世界 -Z，最低变成原顶面 = position.z - _top_z
    for label, orientation, z in (
        ("底面朝下", (1.0, 0.0, 0.0, 0.0), -0.05 - geometry._bottom_z),
        ("顶面朝下", FLIPPED, geometry._top_z - 0.05),
    ):
        points = collide_convex_ground(make_body(geometry, z, orientation), GROUND)
        results.append((label, len(points), _depths(points)))

    ok = all(count == 4 for _label, count, _depths_ in results)
    return ok, "；".join(
        f"{label} {count} 点 深度={[round(d, 6) for d in depths]}"
        for label, count, depths in results
    )


def test_ellipsoid_and_cap_ground() -> tuple[bool, str]:
    """椭球严格凸 → 1 点；球冠底面朝下 → 4 点，倒扣 → 1 点。"""
    ellipsoid = EllipsoidGeometry((0.5, 0.4, 0.3))
    ellipsoid_points = collide_convex_ground(make_body(ellipsoid, 0.25), GROUND)

    cap = SphericalCapGeometry(0.5, 0.5)      # 半球：底面 z = -0.1875，极点 z = +0.3125
    base_points = collide_convex_ground(make_body(cap, 0.15), GROUND)
    pole_points = collide_convex_ground(make_body(cap, 0.28, FLIPPED), GROUND)

    ok = (
        len(ellipsoid_points) == 1
        and abs(ellipsoid_points[0].penetration - 0.05) < 1e-9
        and len(base_points) == 4
        and len(pole_points) == 1
    )
    return ok, (
        f"椭球 {len(ellipsoid_points)} 点（深度 {ellipsoid_points[0].penetration:.6f}）；"
        f"球冠底面朝下 {len(base_points)} 点，倒扣 {len(pole_points)} 点"
    )


# ---------------------------------------------------------------------------
# 3. 分派表
# ---------------------------------------------------------------------------

def test_ground_dispatch_covers_capability_table() -> tuple[bool, str]:
    """分派表必须覆盖能力矩阵里的**每一个**地面种类。

    漏一个的症状是：矩阵说"支持地面"、加载阶段放行，实际却落到通用解——如果
    通用解也支持，那还好；如果连支撑映射都没有，就是运行期 ``TypeError``。
    这里要求两者的键集合完全相等，让"悄悄多/少一条"无处可藏。
    """
    # 复合体不在表里：它在 detect_contacts 里就被逐部件展开，压根不会走到这张表
    declared = set(capabilities.GROUND_IMPL) - {"compound"}
    dispatched = set(_GROUND_DISPATCH)

    # 表里有、矩阵没有 → 表在声称一个矩阵不认的能力，错
    extra = dispatched - declared
    # 矩阵有、表里没有 → 允许，但矩阵必须声明它由通用解承担
    generic = declared - dispatched
    wrongly_generic = [
        kind
        for kind in generic
        if capabilities.GROUND_IMPL[kind][0] != "collide_convex_ground"
    ]

    ok = not extra and not wrongly_generic
    return ok, (
        f"矩阵 {len(declared)} 种（不含复合体），专属分派 {len(dispatched)} 种，"
        f"通用解承担 {sorted(generic)}；"
        f"多余={sorted(extra) or '无'}，声明不符={sorted(wrongly_generic) or '无'}"
    )


def test_parametric_kinds_use_generic_ground() -> tuple[bool, str]:
    """四个参数化图元已在能力矩阵里，但地面**没有**专属函数，由通用解承担。

    这条钉住的是设计意图：它们能落地，但不是因为给每种几何补了一个地面函数，
    而是因为支撑特征解。若哪天有人往 ``_GROUND_DISPATCH`` 里给圆锥加了一条，
    这里会失败——那时应该先想清楚"为什么通用解不够"，而不是顺手加分支。
    """
    cone = ConeGeometry(0.3, 1.2)
    body = make_body(cone, 0.2)
    points = _part_ground_points(body, GROUND)
    kind = capabilities.kind_of(cone)
    specialized = sorted(kind for kind in ("cone", "frustum", "ellipsoid", "spherical_cap") if kind in _GROUND_DISPATCH)
    ok = kind == "cone" and not specialized and len(points) == 4
    return ok, (
        f"能力矩阵种类={kind}，地面专属分支={specialized or '无'}，"
        f"通用解给出 {len(points)} 点"
    )


# ---------------------------------------------------------------------------
# 4. 地面 TOI 的保守性
# ---------------------------------------------------------------------------

def test_cone_ground_toi_is_usable() -> tuple[bool, str]:
    """圆锥的地面 TOI：必须给出有限时刻，且切分点几乎正好贴地。

    圆锥没有精确 TOI 分支，走的是"支撑点最低高度 + 自转上界"的保守推进。
    保守的含义是**不会算晚**：切分时刻的穿透应当只在 ``_CONTACT_OVERLAP``
    （200 µm）量级，不会出现"切进去很深才发现"。
    """
    from src.physics import toi as toi_module

    cone = ConeGeometry(0.3, 1.2)
    body = make_body(cone, 1.0)               # 底面在 0.7，离地 0.7
    body.linear_velocity = (0.0, 0.0, -3.5)   # 自由下落

    t = toi_module.ground_toi(body, GROUND, 10.0)
    if t is None:
        return False, "圆锥下落却没有任何 TOI 事件"

    lowest = body.position[2] - 3.5 * t + cone._base_z
    penetration = GROUND.z - lowest
    ok = 0.0 <= penetration <= 1e-3 and t > 0.0
    return ok, (
        f"t={t:.6f}s，此时底面高度 {lowest:.6f}，穿透 {penetration * 1000:.3f} mm"
        f"（应 ≤ 1 mm）"
    )


# ---------------------------------------------------------------------------
# 4. 端到端：新图元（经组合体包装）落地并静止
# ---------------------------------------------------------------------------

def test_cone_compound_falls_and_rests() -> tuple[bool, str]:
    """两个圆锥组成的复合体自由落体：落地、静止、不穿地。

    为什么要绕组合体这一层：四个新图元还没进能力矩阵（第 6 步才接），而组合体
    已经在矩阵里，所以"圆锥 + 组合体"是**现在**唯一能让新图元跑完整仿真流程的
    路径。它同时覆盖三件事：圆锥的闭式惯量、圆锥的支撑特征地面解、以及复合体
    在 TOI 切分下的宽阶段路径。

    后者曾经是坏的：``world_aabb(复合体)`` 会展开部件再递归一次，而 ``PartView``
    少了 ``bounding_radius``，一旦有 TOI 事件就抛 ``AttributeError``。复合体的
    既有测试都没触发到这一步，是这条用例把它照出来的。
    """
    from src.physics.world import PhysicsWorld
    from src.scene.scene_spec import SceneSpec

    scene_data = {
        "name": "cone_pair",
        "objects": [
            {
                "name": "twin_cones",
                "geometry": {
                    "type": "compound",
                    "overlap": "disjoint",
                    "parts": [
                        {
                            "geometry": {"type": "cone", "radius": 0.3, "height": 0.8},
                            "position": [-0.3, 0.0, 0.0],
                        },
                        {
                            "geometry": {"type": "cone", "radius": 0.3, "height": 0.8},
                            "position": [0.3, 0.0, 0.0],
                        },
                    ],
                },
                "transform": {"position": [0.0, 0.0, 0.85]},
                "mass": 1.5,
            }
        ],
        "simulation": {
            "enabled": True,
            "physics_dt": 1.0 / 240.0,
            "substeps": 3,
            "iterations": 8,
            "duration": 1.5,
            "use_toi_splitting": True,
        },
    }

    scene = SceneSpec.from_dict(scene_data)
    world = PhysicsWorld.from_scene(scene)
    body = world.bodies[0]
    steps = int(round(1.5 / scene.simulation.physics_dt))

    peak = 0.0
    for _ in range(steps):
        world.step()
        peak = max(peak, world.max_penetration())

    # 圆锥底面在局部 z = -h/4 = -0.2，所以支撑姿态下刚体中心停在 z ≈ 0.2
    settled_z = 0.2
    ok = (
        body.speed() < 1e-3
        and peak < 1e-3
        and abs(body.position[2] - settled_z) < 0.02
    )
    return ok, (
        f"末速度={body.speed():.6f} m/s，末位置 z={body.position[2]:.4f}"
        f"（应 ≈{settled_z}），峰值穿透={peak * 1000:.3f} mm（应 < 1 mm）"
    )


def test_standalone_parametric_scene_loads() -> tuple[bool, str]:
    """单独的圆锥/圆台/椭球/球冠现在能**直接**写进场景并跑起来。

    这是第 6 步的验收点：把四种图元接进能力矩阵之后，加载阶段的闸门（``is_supported``）
    才会放行它们。在此之前它们只能藏在组合体里，因为组合体本身是已支持的类型。

    四种几何都从同一高度落下：末速度应接近零、峰值穿透 < 1 mm。
    """
    from src.physics.world import PhysicsWorld
    from src.scene.scene_spec import SceneSpec

    results = []
    for label, geometry in (
        ("圆锥", {"type": "cone", "radius": 0.25, "height": 0.6}),
        ("圆台", {"type": "frustum", "bottom_radius": 0.28, "top_radius": 0.1, "height": 0.5}),
        ("椭球", {"type": "ellipsoid", "radii": [0.3, 0.24, 0.2]}),
        ("球冠", {"type": "spherical_cap", "radius": 0.3, "height": 0.3}),
    ):
        scene = SceneSpec.from_dict(
            {
                "name": f"standalone_{label}",
                "objects": [
                    {
                        "name": "probe",
                        "geometry": geometry,
                        "transform": {"position": [0.0, 0.0, 0.8]},
                        "mass": 1.0,
                    }
                ],
                "simulation": {
                    "enabled": True,
                    "physics_dt": 1.0 / 240.0,
                    "substeps": 3,
                    "iterations": 8,
                    "duration": 1.2,
                },
            }
        )
        world = PhysicsWorld.from_scene(scene)
        body = world.bodies[0]
        peak = 0.0
        for _ in range(int(round(1.2 / scene.simulation.physics_dt))):
            world.step()
            peak = max(peak, world.max_penetration())
        results.append((label, body.speed(), body.position[2], peak))

    failures = [
        f"{label} 末速 {speed:.4f} / z={z:.3f} / 穿透 {peak * 1000:.2f}mm"
        for label, speed, z, peak in results
        if speed > 1e-3 or peak > 1e-3 or z <= 0.0
    ]
    return not failures, (
        "四种图元都能独立加载并静止"
        if not failures
        else "；".join(failures)
    )


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------

CASES = [
    ("1. 球的通用解 == 解析解", test_sphere_generic_matches_analytic),
    ("2. 立着的圆柱：通用解 == 解析解（深度）", test_standing_cylinder_generic_matches_analytic),
    ("3. 圆锥：面接触 4 点 / 顶点接触 1 点", test_cone_ground),
    ("4. 圆台：两个端面都能 4 点", test_frustum_ground_both_faces),
    ("5. 椭球 1 点、球冠面接触 4 点", test_ellipsoid_and_cap_ground),
    ("6. 地面分派表覆盖能力矩阵", test_ground_dispatch_covers_capability_table),
    ("7. 参数化图元由通用解承担地面", test_parametric_kinds_use_generic_ground),
    ("8. 圆锥的地面 TOI 保守可用", test_cone_ground_toi_is_usable),
    ("9. 新图元端到端：落地并静止", test_cone_compound_falls_and_rests),
    ("10. 新图元可直接写进场景", test_standalone_parametric_scene_loads),
]


def main() -> int:
    print("=" * 88)
    print("地面接触：分派表 + 支撑特征通用解")
    print("=" * 88)

    failures = 0
    for title, case in CASES:
        try:
            ok, detail = case()
        except Exception as exc:  # 单个用例崩溃不应中断整轮
            ok, detail = False, f"异常：{type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {title}")
        print(f"       {detail}")
        if not ok:
            failures += 1

    print("-" * 88)
    print(f"结果：{len(CASES) - failures}/{len(CASES)} 通过")
    print("=" * 88)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
