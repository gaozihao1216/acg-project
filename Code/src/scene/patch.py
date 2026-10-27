# -*- coding: utf-8 -*-
"""场景参数覆盖工具。

用途：做参数对比实验时，我们不想为每一组参数都手写一份完整的场景 JSON，
而是"从基础场景出发，只改想观察的那一个参数"。

例如::

    apply_patches(base, {
        "camera.position": [6.0, 1.2, 5.0],
        "lights.0.energy": 1500.0,
        "objects.1.material.roughness": 0.05,
    })

路径用点号分隔，数字段表示列表下标。这样实验定义可以写成纯数据，
既便于批量生成，也便于在报告里列出"每个实验到底改了什么"。
"""

from __future__ import annotations

import copy
from typing import Any


def apply_patches(base: dict, patches: dict[str, Any]) -> dict:
    """返回套用覆盖后的**深拷贝**，不修改原始字典。

    参数
    ----
    base    : 基础场景字典
    patches : 路径 -> 新值 的映射

    异常
    ----
    KeyError   : 路径中的某个键不存在（通常说明实验定义写错了，应当早报错）
    IndexError : 列表下标越界
    ValueError : 路径为空，或试图用数字下标访问非列表
    """
    result = copy.deepcopy(base)
    for path, value in patches.items():
        _set_path(result, path, copy.deepcopy(value))
    return result


def _set_path(target: Any, path: str, value: Any) -> None:
    """沿点号路径定位到父节点，然后写入新值。

    参数
    ----
    target : 根对象（字典或列表）
    path   : 点号分隔的路径，如 ``objects.0.transform.position``
    value  : 要写入的值
    """
    if not path:
        raise ValueError("覆盖路径不能为空")

    parts = path.split(".")
    node = target

    # 沿路径走到最后一个键的父节点
    for depth, part in enumerate(parts[:-1]):
        node = _step(node, part, parts[: depth + 1])

    last = parts[-1]
    if isinstance(node, dict):
        if last not in node:
            raise KeyError(f"覆盖路径 '{path}' 中的键 '{last}' 不存在")
        node[last] = value
    elif isinstance(node, list):
        index = _as_index(last, path)
        if index >= len(node):
            raise IndexError(
                f"覆盖路径 '{path}' 的下标 {index} 越界（长度 {len(node)}）"
            )
        node[index] = value
    else:
        raise KeyError(f"覆盖路径 '{path}' 无法写入：父节点既不是字典也不是列表")


def _step(node: Any, part: str, prefix: list[str]) -> Any:
    """走一层路径，返回下一级节点。"""
    joined = ".".join(prefix)
    if isinstance(node, dict):
        if part not in node:
            raise KeyError(f"覆盖路径 '{joined}' 中的键 '{part}' 不存在")
        return node[part]
    if isinstance(node, list):
        index = _as_index(part, joined)
        if index >= len(node):
            raise IndexError(f"覆盖路径 '{joined}' 的下标 {index} 越界（长度 {len(node)}）")
        return node[index]
    raise KeyError(f"覆盖路径 '{joined}' 无法继续：当前节点不是容器")


def _as_index(part: str, path: str) -> int:
    """把路径片段解析为列表下标，失败时给出可读错误。"""
    try:
        return int(part)
    except ValueError as exc:
        raise ValueError(
            f"覆盖路径 '{path}' 处需要列表下标，但收到 '{part}'"
        ) from exc
