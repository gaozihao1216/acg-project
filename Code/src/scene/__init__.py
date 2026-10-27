# -*- coding: utf-8 -*-
"""场景描述：JSON 解析、参数覆盖、落盘。"""

from .scene_spec import (
    CameraSpec,
    GroundSpec,
    LightSpec,
    MaterialSpec,
    ObjectSpec,
    RenderSpec,
    SceneSpec,
    load_scene,
    save_scene,
)
from .patch import apply_patches

__all__ = [
    "CameraSpec",
    "LightSpec",
    "GroundSpec",
    "MaterialSpec",
    "RenderSpec",
    "ObjectSpec",
    "SceneSpec",
    "load_scene",
    "save_scene",
    "apply_patches",
]
