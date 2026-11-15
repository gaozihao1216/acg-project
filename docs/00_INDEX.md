# 文档索引与冲突裁决

文档按**用途**分四类，各有各的读法：

- `user-guide/` —— 当前规范，照做即可；
- `design/` —— 当前设计，解释"为什么这么做"以及改动的入口；
- `experiments/` —— 当时的实验记录，数字可复现，结论只在那组参数下成立；
- `reviews/` —— 外部审阅基线，带进度状态表，**不代表当前代码**。

原来的 13 篇按写作时间编号的文档已经按这个结构重新编排（合并、抽章），
旧编号到新位置的对照见 §6。

```
docs/
├── 00_INDEX.md                      ← 本页
├── user-guide/
│   ├── 01_运行指南.md
│   ├── 02_场景JSON.md
│   └── 03_资产管道.md                （原 11 + 13）
├── design/
│   ├── 01_architecture.md            分层、数据流、唯一真值、改动入口
│   ├── 02_collision_pipeline.md      窄相演进与 TOI（原 06 + 10）
│   ├── 03_contact_manifold.md        接触流形与身份（原 07）
│   ├── 04_solver.md                  顺序冲量实现与策略（原 04 + 05）
│   └── 05_mass_properties.md         质量属性与点云（原 09 + 12）
├── experiments/
│   ├── 01_phase0_rendering.md        8 组渲染对比（原 03）
│   ├── 02_broadphase.md              宽阶段加速比（原 08）
│   └── 03_warm_start.md              暖启动 A/B（原 05 §8）
└── reviews/
    └── architecture_review_2026-11-01.md   （原仓库根目录的 ACG 审阅）
```

## 1. 我想干什么

| 想干什么 | 先看 | 再查 |
|---|---|---|
| 跑起来 | `user-guide/01_运行指南.md` | `Code/README.md` 的"快速开始" |
| 写 / 改场景 | `user-guide/02_场景JSON.md` | 能力表生成块（见 §4） |
| 做资产、导出 | `user-guide/03_资产管道.md` | `Code/src/geom/asset.py` |
| 理解代码怎么分层 | `design/01_architecture.md` | 各专题篇 |
| 改碰撞、加几何 | `design/02_collision_pipeline.md`、`design/03_contact_manifold.md` | `Code/src/physics/capabilities.py` |
| 改求解器、调收敛 | `design/04_solver.md` | `physics/solver.py`、`results/w4/` |
| 改惯量 / 点云 | `design/05_mass_properties.md` | `geom/massprops.py` |
| 看"为什么这么定" | `experiments/` 三篇 | `results/` 里的基准存档 |
| 看别人怎么挑毛病 | `reviews/architecture_review_2026-11-01.md` | 文首的"当前进度状态表" |

## 2. 每篇一句话

| 篇目 | 一句话 |
|---|---|
| `user-guide/01_运行指南.md` | 目录结构、环境、三种运行方式、加实验的步骤、耗时参考 |
| `user-guide/02_场景JSON.md` | 场景文件全部字段与默认值，含 `physics` / `simulation` 段 |
| `user-guide/03_资产管道.md` | 输入：OBJ/STL/点云 → 惯量与碰撞代理；输出：三个资产文件各自管什么 |
| `design/01_architecture.md` | 四层结构、一步仿真的数据流、五个"唯一真值"、改动入口对照 |
| `design/02_collision_pipeline.md` | 盒-球解析解 → 凸包构造 → 广义 SAT → GJK/EPA → TOI 覆盖任意凸体 |
| `design/03_contact_manifold.md` | 盒-盒流形裁剪、接触点身份（暖启动键）、P0-1 的支撑 patch 求交 |
| `design/04_solver.md` | 顺序冲量求解器的实现细节与"子步优先于迭代"的策略校准 |
| `design/05_mass_properties.md` | 解析惯量、网格散度定理、点云惯量与截断修正 |
| `experiments/01_phase0_rendering.md` | Phase 0 的 8 组渲染参数对比与踩坑 |
| `experiments/02_broadphase.md` | 空间哈希 vs 暴力、加速比与那次不公平的比较 |
| `experiments/03_warm_start.md` | 暖启动开/关的穿透、收敛、耗时对照 |

## 3. 冲突时以谁为准

按这个顺序，**上面覆盖下面**：

1. **源码里的表**：`physics/capabilities.py` 的 `NARROWPHASE_IMPL` / `TOI_IMPL` /
   `GROUND_IMPL`，以及 `physics/support.py` 里的采样与标签规则。
2. **`results/` 里的存档**：基准数字、产物字节数。
3. **生成块**：文档里成对的 HTML 注释标记（`capabilities` 与各类 `evidence:*`）之间的
   表格由 `Code/scripts/sync_docs.py` 从 1、2 生成。手改这些块无效，
   `test_capabilities` / `test_evidence` 会逐字比对报红。
4. **散文描述**：其余正文。它可能只是没跟上。

几条具体的裁决：

- **TOI 精度**：以 `design/02_collision_pipeline.md` 的 TOI 一节 + `physics/toi.py`
  的 docstring 为准。多面体落地是**顶点模型纯平移精确、有自转的瞬时速度一阶近似、
  没有保守上界**；并且**返回的切分时刻比首次接触早 `200 µm / |v_z|`**（穿透余量，
  纯平移也有），别把两者混为一谈。凡写着"顶点精确解 / 含自转精确"的段落都是旧文
  （`design/04_solver.md` 与 `design/02_collision_pipeline.md` 的第一部分是写作当时的
  记录），冲突时以上面这条为准。
- **接触身份**：以 `design/03_contact_manifold.md` §4 为准。`feature_id` 批内唯一是
  硬要求，跨帧稳定只到"局部特征级"，模式切换会失效。
- **能力边界**：只看 `user-guide/02_场景JSON.md` 的生成块。它回答"哪种几何 × 哪种
  几何有窄相 / 有 TOI"。

一条要单独说的教训：**生成只能保证"文档 = 真值"，不保证"真值本身正确"**。
`GROUND_IMPL` 里曾把盒/凸包落地标成 `精确（顶点解，含自转）`，`sync_docs.py`
忠实地把这个错误刷进能力表，`test_capabilities` 也一路绿灯。所以改表的人要为
**表的语义**负责，测试只能守住一致性。

## 4. 数值入口

- 一键测试：`python Code/tests/run_all.py`（26 个模块、242 项检查，全部通过；
  P0-1 的 4 项已于 2026-11-15 转正）。`run_all.py` 会先核对清单与
  `tests/test_*.py` 是否一一对应，漏登记直接失败。
- 文档同步：`python Code/scripts/sync_docs.py`（改能力矩阵 / 重跑基准后必须执行）。
- 复现基准：`Code/scripts/reproduce.py`（产物逐字节复现见 `results/w3/`）。

## 5. 维护规则

1. 新增或改名文档，**必须**更新本页的目录树与 §2 的一句话表。
2. 改 `capabilities.py` 或重跑基准，**必须**跑 `sync_docs.py`，再跑
   `test_capabilities` 与 `test_evidence`。
3. 写"精确 / 保守 / 一阶"这类定性词时，先确认 `capabilities.py` 里有对应的档位，
   别在散文里另造一套说法。
4. 合并文档时**保持小节编号连续**，并在被合并部分前加"第 N 部分 · 原标题（原第 N 篇）"，
   否则跨文档的 `§k` 引用会指错位置。

## 6. 旧编号对照（合并前的 `docs/NN`）

| 旧 | 现在在哪 | 小节号变化 |
|---|---|---|
| `docs/00` | 本页 | — |
| `docs/01` | `user-guide/01_运行指南.md` | 不变 |
| `docs/02` | `user-guide/02_场景JSON.md` | 不变 |
| `docs/03` | `experiments/01_phase0_rendering.md` | 不变 |
| `docs/04` | `design/04_solver.md` 第 1 部分 | 不变 |
| `docs/05` | `design/04_solver.md` 第 2 部分 | §1–7 → §12–18，§9 → §19 |
| `docs/05` §8 | `experiments/03_warm_start.md` | 独立成篇 |
| `docs/06` | `design/02_collision_pipeline.md` 第 1 部分 | 不变 |
| `docs/07` | `design/03_contact_manifold.md` | 不变 |
| `docs/08` | `experiments/02_broadphase.md` | 不变 |
| `docs/09` | `design/05_mass_properties.md` 第 1 部分 | 不变 |
| `docs/10` | `design/02_collision_pipeline.md` 第 2 部分 | §k → §(k+9) |
| `docs/11` | `user-guide/03_资产管道.md` 第 1 部分 | 不变 |
| `docs/12` | `design/05_mass_properties.md` 第 2 部分 | §k → §(k+9) |
| `docs/13` | `user-guide/03_资产管道.md` 第 2 部分 | §k → §(k+8) |
| 根目录 `ACG_Architecture_Review_and_Migration.md` | `reviews/architecture_review_2026-11-01.md` | 不变 |
