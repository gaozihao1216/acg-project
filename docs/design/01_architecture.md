# 架构：分层、数据流与改动入口

这一篇是"代码长什么样"的**总图**，不重复各专题的推导与数字。要改行为，先在这里
找到该动哪一层，再去对应的专题篇看细节与坑。

## 1. 四层结构

```
mathutils ──► geom / scene ──► physics ──► evidence / render
  (纯数值)      (纯数据+几何)     (仿真)      (证据与呈现)
```

依赖只允许往右。每一层都可以单独测试：

| 层 | 目录 | 职责 | 不做什么 |
|---|---|---|---|
| `mathutils` | `Code/src/mathutils/` | 四元数、向量、二维凸多边形（切向基 / 凸包 / 求交） | 不知道"刚体""几何体"存在 |
| `geom` | `Code/src/geom/` | 参数化几何、变换、转动惯量、凸包、网格 IO、资产 | 不做碰撞判定 |
| `scene` | `Code/src/scene/` | 场景 JSON 解析、字段校验、参数补丁 | 不含物理常量 |
| `physics` | `Code/src/physics/` | 刚体、宽相、窄相（SAT / GJK / EPA）、TOI、求解器、世界 | 不读文件、不碰 `bpy` |
| `evidence` | `Code/src/evidence.py` | 文档数字的唯一来源：基准存档 / 产物 → 表格 | 不做计算之外的格式化 |
| `render` | `Code/render/` | SceneSpec → bpy 场景、静态帧、动画、资产 QA | 依赖 `bpy`，只在 Blender 进程里导入 |

`render` 不在 `src/` 下，是唯一允许依赖外部引擎的层；`src/` 整体保持"纯 Python，
`python Code/tests/run_all.py` 不需要 Blender"。

## 2. 一次仿真步的数据流

```
场景 JSON ──parse──► SceneSpec ──► RigidBody[]（几何 + 位姿 + 质量属性）
                                        │
                          ┌─────────────┴─────────────┐
                          │  每个子步（substep）      │
                          └─────────────┬─────────────┘
                                        ▼
        Broadphase（空间哈希 → 候选对） ──► Narrowphase
                                        │   解析路径：球/盒/地面（精确公式）
                                        │   SAT 路径：盒-盒（15 轴）
                                        │   GJK/EPA：任意凸体 → 支撑 patch 求交
                                        ▼
                        TOI（earliest_toi → 切分到首次接触）
                                        ▼
                    SequentialImpulse（法向 + 摩擦 + 暖启动）
                                        ▼
                           积分（半隐式欧拉，四元数归一化）
```

三处容易搞错的边界：

1. **宽相只负责"不可能相碰"的排除**，判据用包围球（不旋转、保守）；AABB 只用于
   空间哈希的桶。详见 `docs/experiments/02_broadphase.md`。
2. **窄相有三条并行路径**，谁负责哪一种由 `physics/capabilities.py` 决定，不是
   写在 `if` 里。加几何体时要同时改能力表，否则"能解析但没有路径"。
3. **TOI 与流形是两件事**：TOI 决定"切到哪一刻"，流形决定"这一刻有哪些接触点"。
   混在一起会出现"切得准但流形错"的假象，P0-1 就是这么来的。

## 3. 五个"唯一真值"

改这些之前先确认没有第二份拷贝：

| 真值 | 位置 | 谁在读 |
|---|---|---|
| 碰撞能力边界（哪种几何 × 哪种几何有窄相 / 有 TOI） | `physics/capabilities.py` 的 `NARROWPHASE_IMPL` / `TOI_IMPL` / `GROUND_IMPL` | `narrowphase.is_supported`、`docs/user-guide/02_场景JSON.md` 的能力表（生成块） |
| 支撑映射（沿方向的最远点） | 各 `GeometrySpec.support_local`；`physics/support.py` 只做世界变换与特征采样 | GJK / EPA / 流形 / 地面窄相 |
| 接触点身份（暖启动键） | `physics/support.py` 的 `_TAG_*` 与 `patch.match_tags` | `gjk._build_manifold`、`contact.py` 的缓存 |
| 基准数字 | `results/` 下的存档文件 | `evidence.py` → 文档生成块 |
| 场景字段 | `scene/scene_spec.py` 的解析与默认值 | `docs/user-guide/02_场景JSON.md` |

文档里的能力表与基准表都是**生成块**（HTML 注释标记之间），手改无效，
`sync_docs.py` 会覆写，`test_capabilities` / `test_evidence` 会逐字比对。

## 4. 改动的入口对照

| 想做的事 | 先读 | 再动 |
|---|---|---|
| 跑起来 / 加一个实验脚本 | `docs/user-guide/01_运行指南.md` | `Code/scripts/` |
| 加场景字段 / 改默认值 | `docs/user-guide/02_场景JSON.md` | `scene/scene_spec.py`、`docs/…` 生成块 |
| 从 OBJ/STL/点云做资产 | `docs/user-guide/03_资产管道.md` | `geom/mesh_io.py`、`geom/massprops.py`、`geom/asset.py` |
| 加一种几何体 | 本篇 §3 + `docs/design/02_collision_pipeline.md` | `geom/geometry.py`、`physics/capabilities.py`、`physics/support.py` |
| 改窄相判定 / TOI | `docs/design/02_collision_pipeline.md` | `physics/narrowphase.py`、`physics/sat.py`、`physics/gjk.py`、`physics/toi.py` |
| 改接触点数量 / 位置 / 身份 | `docs/design/03_contact_manifold.md` | `physics/patch.py`、`physics/gjk.py`、`physics/support.py` |
| 改求解器参数 / 收敛性 | `docs/design/04_solver.md` | `physics/solver.py`、`physics/world.py` |
| 改惯量 / 质量属性 | `docs/design/05_mass_properties.md` | `geom/inertia.py`、`geom/massprops.py` |
| 写进文档的数字 | `docs/00_INDEX.md` §3 | `results/`、`Code/scripts/sync_docs.py` |

## 5. 与"阶段日志"的关系

`docs/experiments/` 下的三篇是**当时的实验记录**（Phase 0 渲染对比、宽阶段加速比、
暖启动 A/B），数字来自 `results/`，可以复现；结论只在那组参数下成立。
`docs/reviews/` 是 2026-11-01 的外部审阅基线，带"当前进度状态表"，不代表当前代码。

四类文档的读法与冲突裁决见 `docs/00_INDEX.md`。
