# -*- coding: utf-8 -*-
"""Blender 渲染层。

本包内的模块依赖 ``bpy``，只能在 Blender 的 Python 环境里导入。
若在普通 Python 中 ``import render``，会因为找不到 ``bpy`` 而失败——
这是有意为之：渲染代码不应被物理逻辑意外引入。
"""
