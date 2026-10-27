# -*- coding: utf-8 -*-
"""刚体物理：接触、宽相/窄相、顺序冲量求解器、物理世界。

支持的碰撞组合：球-球、球-盒、盒-盒，以及球/盒与地面。
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
    "find_initial_overlaps",
    "format_violations",
    "OverlapViolation",
    "AABB",
    "world_aabb",
    "BruteForceBroadphase",
    "SpatialHashBroadphase",
]
