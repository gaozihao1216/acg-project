# 文档索引与冲突裁决

13 篇按写作时间编号的文档保留原样（它们是开发日志，后文推翻前文是常态），
但读者需要一条**读法**：哪些是当前规范、哪些只是当时的实验记录、遇到冲突听谁的。
这一页就是这条读法。**近期不移动文件**，下面的目录结构是目标形态，等真正迁移时
再按它分。

## 1. 现在的读法

| 想干什么 | 先看 | 再查 |
|---|---|---|
| 跑起来 | `docs/01_环境与运行指南.md` | `Code/README.md` 的"快速开始" |
| 写 / 改场景 | `docs/02_场景JSON规范.md` | 能力表生成块（见 §3） |
| 改碰撞、加几何 | `docs/07` §4（接触身份）、`docs/10`（凸体 / TOI） | `Code/src/physics/capabilities.py` |
| 做资产、导出 | `docs/11_物体输入管道.md`、`docs/13_输出资产体系.md` | `Code/src/geom/asset.py` |
| 看"为什么这么定" | `docs/03`（Phase 0）、`docs/05`（求解器策略）、`docs/08`（宽阶段）、`docs/09`（质量属性） | `results/` 里的基准存档 |
| 看别人怎么挑毛病 | `ACG_Architecture_Review_and_Migration.md`（2026-11-01 基线） | 文首的"当前进度状态表" |

## 2. 文档的四种性质

| 性质 | 篇目 | 读的时候要记住 |
|---|---|---|
| **当前规范** | `01` 运行指南、`02` 场景 JSON、`11` 物体输入、`13` 资产体系；`07` §4（接触身份）、`10`（凸体与 TOI） | 可以直接照做。改行为要同步改它们 |
| **实验记录** | `03` Phase 0 渲染对比、`05` 求解器策略、`08` 宽阶段、`09` 质量属性、`12` 点云截断推导 | 数字来自 `results/`，可复现；结论只在当时参数下成立 |
| **历史设计** | `04` W2 球体求解器、`06` 盒球碰撞 | 是写作当时的方案，**可能已被 `07`/`10` 推翻**。TOI 与接触流形不看这两篇 |
| **历史审阅** | `ACG_Architecture_Review_and_Migration.md` | 2026-11-01 的基线，问题清单带状态表；不代表当前代码 |

目标形态（迁移后再落地）：

```
docs/
├── 00_INDEX.md                  ← 本页
├── user-guide/                  01 运行指南、02 场景 JSON、13 资产管道
├── design/                      architecture、collision_pipeline、
│                                contact_manifold（07 §4）、solver
├── experiments/                 phase0_rendering、broadphase、warm_start
└── reviews/                     architecture_review_2026-11-01.md
```

## 3. 冲突时以谁为准

按这个顺序，**上面覆盖下面**：

1. **源码里的表**：`physics/capabilities.py` 的 `NARROWPHASE_IMPL` / `TOI_IMPL` /
   `GROUND_IMPL`，以及 `physics/support.py` 里的采样与标签规则。
2. **`results/` 里的存档**：基准数字、产物字节数。
3. **生成块**：文档里成对的 HTML 注释标记（`capabilities` 与各类 `evidence:*`）之间的
   表格由 `Code/scripts/sync_docs.py` 从 1、2 生成。手改这些块无效，
   `test_capabilities` / `test_evidence` 会逐字比对报红。
4. **散文描述**：其余正文。它可能只是没跟上。

几条具体的裁决（都被上面的规则支撑）：

- **TOI 精度**：以 `docs/10` §6 + `physics/toi.py` 的 docstring 为准。多面体落地是
  **顶点模型纯平移精确、有自转的瞬时速度一阶近似、没有保守上界**；并且**返回的切分
  时刻比首次接触早 `200 µm / |v_z|`**（穿透余量，纯平移也有），别把两者混为一谈。
  仍写着"顶点精确解 / 含自转精确"的旧文——`docs/04:35`、`docs/05:164`、`docs/06:148`
  ——属历史设计，不再维护，冲突时以 `docs/10` §6 为准。
- **接触身份**：以 `docs/07` §4 为准。`feature_id` 批内唯一是硬要求，跨帧稳定只到
  "局部特征级"，模式切换会失效。
- **能力边界**：只看 `docs/02` 的生成块。它回答"哪种几何 × 哪种几何有窄相 / 有 TOI"。

一条要单独说的教训：**生成只能保证"文档 = 真值"，不保证"真值本身正确"**。
`GROUND_IMPL` 里曾把盒/凸包落地标成 `精确（顶点解，含自转）`，`sync_docs.py`
忠实地把这个错误刷进 `docs/02`，`test_capabilities` 也一路绿灯。所以改表的人要为
**表的语义**负责，测试只能守住一致性。

## 4. 数值入口

- 一键测试：`python Code/tests/run_all.py`（26 个模块、242 项检查，全部通过；
  P0-1 的 4 项已于 2026-11-15 转正）。`run_all.py` 会先核对清单与
  `tests/test_*.py` 是否一一对应，漏登记直接失败。
- 文档同步：`python Code/scripts/sync_docs.py`（改能力矩阵 / 重跑基准后必须执行）。

## 5. 维护规则

1. 新增或改名文档，**必须**更新本页的分类表。
2. 改 `capabilities.py` 或重跑基准，**必须**跑 `sync_docs.py`，再跑
   `test_capabilities` 与 `test_evidence`。
3. 写"精确 / 保守 / 一阶"这类定性词时，先确认 `capabilities.py` 里有对应的档位，
   别在散文里另造一套说法。
