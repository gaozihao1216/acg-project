# -*- coding: utf-8 -*-
"""场景描述：JSON 解析与参数覆盖。

写出场景由调用方自己完成（脚本写自己的 JSON，资产走 `geom.scene_asset`），
所以这里只保留 `load_scene()` 与 `SceneSpec.to_dict()`。
"""

from .scene_spec import (
    CameraSpec,
    GroundSpec,
    LightSpec,
    MaterialSpec,
    ObjectSpec,
    RenderSpec,
    SceneSpec,
    load_scene,
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
    "apply_patches",
]
