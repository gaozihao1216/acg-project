# -*- coding: utf-8 -*-
"""刚体物理：接触、宽相/窄相、顺序冲量求解器、物理世界。

碰撞覆盖范围**不在这里手写**：能力矩阵（``capabilities.py``）是唯一来源，
``narrowphase.is_supported`` / ``supported_pair`` 由它派生。当前球形、盒、
凸包、柱、胶囊、圆锥、圆台、椭球、球冠、组合体的物体对全部可用（解析路径
或 GJK/EPA），另有 10 种几何的专门地面路径。
"""

from .body import RigidBody, make_ground_proxy
from .broadphase import (
    AABB,
    BruteForceBroadphase,
    SpatialHashBroadphase,
    world_aabb,
)
from .contact import GROUND_BODY, ContactManifold, ContactPoint, merge_pair_materials
from .narrowphase import GroundPlane, detect_contacts, is_supported, supported_pair
from .sat import collide_box_box
from .solver import SequentialImpulseSolver
from .support import (
    TAG_LIMIT,
    SupportPatch,
    feature_world_tagged,
    support_patch_world,
    support_world,
)
from .toi import earliest_toi, sphere_ground_toi, sphere_sphere_toi
from .validate import OverlapViolation, find_initial_overlaps, format_violations
from .world import PhysicsWorld

__all__ = [
    "RigidBody",
    "make_ground_proxy",
    "ContactPoint",
    "ContactManifold",
    "GROUND_BODY",
    "merge_pair_materials",
    "GroundPlane",
    "detect_contacts",
    "is_supported",
    "supported_pair",
    "collide_box_box",
    "SequentialImpulseSolver",
    "PhysicsWorld",
    "earliest_toi",
    "sphere_sphere_toi",
    "sphere_ground_toi",
    "support_world",
    "feature_world_tagged",
    "support_patch_world",
    "SupportPatch",
    "TAG_LIMIT",
    "find_initial_overlaps",
    "format_violations",
    "OverlapViolation",
    "AABB",
    "world_aabb",
    "BruteForceBroadphase",
    "SpatialHashBroadphase",
]
