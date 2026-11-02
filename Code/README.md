# Code —— ACG 2026 题目三：物理仿真

当前进度：**W1 静态渲染 ✅ / W2 刚体求解器（球+盒）✅ / W3 任意凸体 + 资产管道 ✅ /
W4 暖启动 + 文档守卫 ✅**

自写刚体物理，未使用任何现成物理引擎（无 MuJoCo / Bullet / PhysX / Unity / UE）。

详细文档见 `../docs/`：

| 文档 | 内容 |
|---|---|
| `01_环境与运行指南.md` | 目录结构、环境要求、三种运行方式、常见问题、耗时参考 |
| `02_场景JSON规范.md` | 场景文件全部字段的说明与约定（含 `physics` / `simulation` 段） |
| `03_Phase0_实验报告.md` | 8 组渲染参数对比实验的结论与踩坑记录 |
| `04_W2_球体刚体求解器.md` | 求解器原理、6 项数值验证、视频产出、踩坑记录 |

---

## 快速开始

在 `project/` 目录下执行（不是 `Code/` 里）。

### 1. 数值验证（最快，先跑这个）

```powershell
python Code\tests\run_all.py
```

二十二个模块共 207 项检查，不依赖 Blender，约 40 秒：

| 模块 | 检查内容 |
|---|---|
| `test_rigid_sphere.py` | 积分器、落地静置、高速不穿透、能量不增加、弹性速度交换、多球收敛 |
| `test_offcenter_collision.py` | 偏心碰撞与闭式解比对、切向速度守恒、无摩擦不自旋、动量守恒、偏心量扫描 |
| `test_multicontact.py` | 三体/四体同时接触、迭代收敛曲线、顺序无关性、多迭代 vs 小步长、初始重叠的影响 |
| `test_toi_split.py` | 球 TOI 与解析解一致、单子步内链式碰撞、穿透深度对比、事件上限、静置零事件、初始嵌入被拒绝 |
| `test_toi_convex.py` | **盒/凸包落地顶点精确解**、**纯自转由顶点速度决定**、**盒-盒纯平移与解析解一致**、不产生伪事件、**仿真穿透 3–70 倍下降**、球走精确解不吃余量 |
| `test_box_collision.py` | 球-盒窄相（面/棱/盒内）、盒-地面 4 点流形、**偏心产生力矩、对心不产生**、惯量张量、盒子静置不翻转 |
| `test_box_box.py` | 盒-盒 SAT：面贴面 4 点、棱-棱 1 点、顶点-面、法向取反、**特征 ID 跨帧稳定**、随机抽样、两层堆叠 |
| `test_broadphase.py` | AABB 按朝向计算、**哈希结果 == 所有 AABB 相交对**、格子边长鲁棒性、大物体兜底通道、地面接触不受影响 |
| `test_massprops.py` | 完整惯量张量、网格散度定理（实心/空心/L 形）、**惯量积非零**、点阵修正到机器精度、随机采样无偏、误差随 N 收敛 |
| `test_primitive_geometry.py` | 圆锥质心偏移与体积、**圆锥主惯量对照 `Izz=3mr²/10`、`Ixx=m(3r²/20+3h²/80)`**、圆台退化回圆柱/圆锥、椭球退化回球、球冠 `h=R` 为半球、`h=2R` 为整球、**支撑映射不劣于密集采样**、包含测试、JSON 往返、组合体不提供整体支撑映射 |
| `test_hull.py` | 凸包精确构造、共面合并、退化/超限拒绝、**世界逆惯量 vs 完整张量**、**广义 SAT 在盒子上退化一致**、球-凸包穿透精确、凸包-地面流形、凹体需组合体 |
| `test_compound.py` | 哑铃惯量与平行轴定理一致、不等体积部件的质心、单部件组合体等于该图元、**场景 JSON 支持组合体**、哑铃横放静置、T 形立着不倒、**两部件同时接触都报出**、地面特征 ID 不撞号、宽阶段 AABB 是部件并集、自转复合体靠 ω×r 报出 TOI |
| `test_inertia_modes.py` | 单球/单盒采样惯量对上解析值、**部件不重叠时 `disjoint` 与 `overlap` 一致**、重叠时体积等于并集、采样确定性、JSON 缺 `overlap` 报错、非法值被拒、往返保留模式 |
| `test_curved.py` | **柱/胶囊落地静置（含横放不翻滚）**、**球-柱深度与法向对照解析最近点解**、**立着的柱压在盒面上得到 4 点流形**、高速不穿透、柱-柱挤压深度、**曲面 TOI 与解析时刻差 0.1 ms**、**盒-盒上 GJK/EPA 与 SAT 零偏差**、距离双向对称 |
| `test_ground_generic.py` | **球/立着圆柱的通用解与解析解逐位一致**、圆锥面接触 4 点 / 顶点 1 点、圆台两个端面都 4 点、椭球 1 点、球冠面接触 4 点、**地面分派表不越出能力矩阵**（矩阵多出的必须声明由通用解承担）、四种参数化图元由通用解承担地面、**圆锥地面 TOI 保守**（切分点穿透 0.2 mm）、组合体端到端落地静止、**四种图元可直接写进场景并静置** |
| `test_contact_identity.py` | **各几何地面接触 ID 唯一**、地面 ID 与几何特征对应（盒 `{0,2,4,6}`、柱 `{0,1,2,3}`）、**倾斜裁掉浅点后剩余 ID 不被重新编号**、GJK 流形 ID 唯一、绕自身轴自转时按位置配对全同、**零方向也返回带标签的点**、所有标签 `< TAG_LIMIT`、**圆盘标签在体坐标系里稳定**（6 个姿态下同一 tag 的体坐标偏差 1e-16，世界位置确实随姿态移动）、偏心负载下缓存键数 = 接触点数、四角冲量分别存储 |
| `test_diagnostics.py` | **自由空间两球重叠 0.05 m 量得出来**、自由空间不接触才是 0、有地面时地面穿透仍被报出、**地面与物-物同时存在时取二者最大值**（两组数据互为对照，防止"只取地面"或"只取物体对"）、**多点流形取最深点**（倾斜盒四角两档深度 15.9 / 3.9 mm，指标必须等于最深那档） |
| `test_object_input.py` | OBJ 负索引/多面写法/三角化、**二进制 STL 头部写 solid 也能识别**、STL 顶点去重、开网格诊断、点云 `auto` 修正判定、点云文件格式容错、**命令生成的场景可加载并静置** |
| `test_asset.py` | **QΛQᵀ 还原张量**、**两个四元数互为共轭**、主轴 det=+1、碰撞点在 body 系（体积自洽）、贴地高度精确、**资产 → 引擎主惯量一致且静置**、碰撞点未经二次变换、点云无碰撞代理、**JSON 逐字节确定**、读写无损、未知版本被拒绝、案例目录 input/output 约定、源文件复制进 input 且自包含、**空心盒的真实惯量能传到引擎**、输入卡记录不可复现的输入、**场景资产只存引用**、材质分量数 RGBA→RGB |
| `test_capabilities.py` | 能力矩阵自洽、**表里的函数名真实存在**、`is_supported`/`supported_pair`/`body_toi`/`ground_toi` 逐格与表一致（含双向）、未支持几何在加载阶段被拒绝、**文档生成块与表逐字一致** |
| `test_evidence.py` | 8 个渲染器可用、**存档解析行数不塌陷**、生成块标记成对、**文档里的基准数字与产物字节数与真值逐字一致** |
| `test_warm_start.py` | 默认开启且可序列化、**低迭代时穿透减半、残余速度降两个量级**、与 TOI 共存不重复施加、**跨帧缓存不破坏确定性**、静置不注入能量 |

> **文档里的"能力"和"数字"都不是手写的**：
>
> - 能力边界（哪种几何组合有窄相、哪种有 TOI）只有一处定义：`src/physics/capabilities.py`，
>   `narrowphase.is_supported` / `supported_pair` 由它派生；
> - 基准数字来自 `results/w2/bench_*.txt`，产物字节数来自现场扫描 `results/w3/*/output/`，
>   由 `src/evidence.py` 渲染成表。
>
> 改了真值后跑 `python Code\scripts\sync_docs.py` 重写文档块。忘了同步，或者反过来
> 手改文档里的数字，`test_capabilities.py` / `test_evidence.py` 都会红——这就是这两个
> 测试存在的意义，它们守的不是代码，是**文档不许撒谎**。

### 2. 从网格/点云文件算质量属性

```powershell
# 网格 → 体积/质心/完整惯量张量/主惯量 + 碰撞代理 + 可直接跑的场景
python Code\scripts\analyze_object.py Code\assets\prism.obj --mass 3.0 --emit Code\scenes\demo_prism.json

# 只有点云（自动判断该不该加截断修正）
python Code\scripts\analyze_object.py --points cloud.csv --mass 8.0
```

### 3. 输出资产：每个物体一个案例目录

```powershell
python Code\scripts\analyze_object.py Code\assets\prism.obj --mass 3.0 --out-dir results\w3
```

```
results\w3\prism\
├── input\    prism.obj           源文件副本（案例可独立带走）
│             request.json        输入卡：质量/密度、随机种子、凸包上限
└── output\   prism.asset.json    数据（只存数值与引用，不含顶点）
              prism.blend         资产
              prism.preview.png   证据
```

按物体分目录（而不是按文件类型分），删掉一个物体只删一个目录；内部分
`input` / `output`，把"进来的"和"算出去的"隔开。

- `input\` 是**证据链的起点**：`source.path` 指向 `input\` 里的副本，
  `sha256` 就在旁边（路径全部相对 JSON 所在目录，整个 `results\` 可以搬走）
- `request.json` 记**产物反推不回来的输入**：给了 `--mass` 还是 `--density`、
  随机点云的种子、凸包面数上限、要求的修正式。`output\` 说"算出了什么"，
  `input\` 说"我要什么"

三者角色不同，不可互换：

| 文件 | 消费者 | 内容 |
|---|---|---|
| `.asset.json` | 引擎、脚本、版本库 | 质量/质心/完整张量/主惯量/主轴/凸包/引擎交接 + 8 条自检。**不存网格顶点**——几何在 `input\` 与 `.blend` 里 |
| `.blend` | 美术、渲染、人工检查 | 渲染网格 + **碰撞代理（独立、线框）** + 质心与主轴标记，全在 body 系 |
| `.preview.png` | 人（一眼） | 主轴箭头/质心点/凸包线框叠加，核对数字与形状是否对得上 |

没有 Blender 时加 `--no-render`，只出 JSON；加 `--no-copy-input` 则不复制源文件。

想让物体能**自由旋转**地看，加 `--glb` 再出一份 `.glb`（第四件产物，可选）：

```powershell
python Code\scripts\analyze_object.py Code\assets\prism.obj --mass 3.0 --out-dir results\w3 --glb
```

GLB 含渲染网格 + 碰撞线框 + 质心 + 主轴（共 12 个节点），**不含**相机/灯光/地面，
也**不含**轴标签——那是朝向相机的广告牌，自由旋转时只会变成贴在屏幕上的怪东西。
它不进默认流程：`output/` 里每个文件都该有明确消费者，而 GLB 的消费者是人和
外部工具，不是求解器。

支持 `.obj` / `.stl`（ASCII 与二进制自动识别）/ 文本点云（含 `random:N:SEED:SX,SY,SZ`
现场生成，种子进输入卡）。报告会同时给出完整张量与主惯量、非对角占比、
凸包面棱数与 SAT 轴数。详见 `docs/11_物体输入管道.md`、
`docs/13_输出资产体系.md`；点云修正项的现状见 `docs/12_点云截断修正的算法路径.md`。

### 4. 装配场景（只存引用）

```powershell
python Code\scripts\assemble_scene.py --root results\w3 --name stack `
    --place shell=0 --place prism=1.05 `
    --emit-engine results\w3\stack\output\stack.engine.json
```

`--place z` 是**底面离地高度**（`0` = 贴地），写进场景时自动换成质心位置。

```
results\w3\stack\
├── input\    request.json   装配配方（可重放）
└── output\   stack.scene.json     场景资产：名称 + 引用 + 环境，约 2.4 KB
              stack.engine.json    展开后的快照（给引擎，不是真值）
```

**场景资产里没有物体级数值。** 只有名称、引用、`asset_sha256`、位置朝向，
以及地面/相机/光照/求解器配置。物体多重、惯量多少，一律沿着 `asset`
去读 `../../<name>/output/<name>.asset.json`——那是唯一真值。

复制而不是引用会有三个具体问题：占空间（20 个物体能到几 MB）、**会分叉**
（物体资产改了场景不跟着变）、出错时不知道该信哪份。`--emit-engine` 的输出
是"展开后的快照"，明确标注不可当真值。

### 5. 从场景资产直接跑仿真 + 出视频

```powershell
python Code\scripts\run_sim.py --scene-asset results\w3\stack\output\stack.scene.json `
    --out results\w3\stack\output\stack.poses.json `
    --render --video results\w3\stack\output\stack.mp4
```

`--scene-asset` 直接解析引用，**不产生中间文件**。只有渲染时才展开一次
（`*.expanded.scene.json`），因为 Blender 层需要自包含的场景；那份展开文件
每次都重新生成，绝不复用。

### 6. 校验"可复现"不是空话

```powershell
python Code\scripts\reproduce.py --root results\w3
```

扫描每个案例的 `input/request.json`，把记录的命令**原样重放**到临时目录，
再与 `output/` 里的产物**逐字节比对**；场景案例还会重跑一遍仿真，验证求解器
的确定性。`results/w3` 当前 **4/4 全部逐字节一致**。

它同时审查资产设计：输入卡漏记任何一个能改变结果的自由度（`max_faces`、
修正式、质量来源、`--emit-engine`），重放出的产物就会不同，立刻暴露。

### 7. 跑仿真 + 出视频

```powershell
python Code\scripts\run_sim.py --scene Code\scenes\sim_spheres.json `
    --out results\w2\poses_sim_spheres.json --render --video results\w2\sim_spheres.mp4
```

`--stills 0,20,40` 可只渲染指定帧为 PNG，用于快速检查构图。

### 8. 静态渲染实验（W1）

```powershell
python Code\scripts\run_phase0_experiments.py
```

### 9. 渲染单个静态场景

```powershell
& "C:\Program Files\Blender Foundation\Blender 3.6\blender.exe" `
  --background --factory-startup `
  --python "Code\render\render_static.py" -- `
  --scene "Code\scenes\demo_basic.json" `
  --out   "results\phase0\demo.png"
```

> `render/*.py` **必须**用 Blender 自带 Python 运行（需要 `bpy`）；
> `scripts/` 与 `tests/` 用系统 Python 运行。

---

## 目录

```
Code/
├── src/                     纯 Python 库，不依赖 bpy
│   ├── mathutils/           四元数、三维向量
│   ├── geom/                参数化几何、变换、转动惯量
│   │   ├── geometry.py      球/盒/柱/胶囊/锥/台/椭球/球冠 + 凸包 + 组合体
│   │   ├── inertia.py       图元的闭式惯量；凸包返回主惯量
│   │   ├── massprops.py     质量属性：完整 3×3 张量、网格散度定理、点云
│   │   ├── convexhull.py    增量法凸包 + 共面三角形合并（立方体 → 6 个四边形面）
│   │   ├── meshio.py        OBJ / STL（ASCII+二进制）/ 文本点云 的读写
│   │   ├── asset.py         输出资产 object_asset/1：schema、生成、自检、输入卡
│   │   ├── scene_asset.py   场景资产 scene_asset/1：只存引用的装配与解析
│   │   └── meshutil.py      测试用网格与点云生成（盒、棱柱、空心、点阵）
│   ├── scene/               场景 JSON 解析、参数覆盖
│   ├── evidence.py          文档数字的唯一来源（基准存档 / 产物目录 → 表格）
│   └── physics/             刚体物理
│       ├── capabilities.py  **能力矩阵**：窄相 / TOI 的覆盖范围，只在这里定义
│       ├── contact.py       接触点与接触流形（多点，带跨帧稳定特征 ID）
│       ├── body.py          刚体状态、逆惯量作用、姿态并入惯量主轴
│       ├── broadphase.py    宽阶段：AABB + 空间哈希（保留暴力基线做加速比）
│       ├── narrowphase.py   窄相分派：球/盒/凸包 任意两两组合 + 各自-地面
│       ├── sat.py           盒-盒 SAT + 参考面/入射面裁剪（最多 4 点）
│       ├── convex_sat.py    广义 SAT：任意凸多面体（面法向 + 棱叉积）+ 球-凸包最近点
│       ├── toi.py           预测接触时刻（球走精确解；盒/凸包走顶点精确解
│       │                    + SAT 轴保守推进，见 docs/10 第 6 节）
│       ├── validate.py      初始互相嵌入检测
│       ├── solver.py        顺序冲量求解器
│       └── world.py         单步流程、TOI 分段、积分、位姿导出
├── render/                  Blender 层，依赖 bpy
│   ├── render_asset.py      资产 → .blend + QA 预览图（网格/凸包/主轴/质心）
│   ├── blender_builder.py   SceneSpec → bpy 场景
│   ├── render_static.py     静态单帧渲染
│   └── render_anim.py       位姿序列 → 关键帧 → 视频
├── scenes/
│   ├── demo_basic.json      静态演示（W1）
│   ├── sim_spheres.json     球体沉降演示（W2）
│   ├── demo_offcenter.json  对心 vs 偏心碰撞对照（W2）
│   ├── demo_boxes.json      倾斜盒落下 + 球偏心撞盒（W2）
│   ├── demo_box_stack.json  盒子金字塔 + 顶部落盒（W2）
│   ├── demo_pile.json       容器堆积 52 体（W2，由脚本生成）
│   ├── demo_hull.json       三种凸体 + 球落下互撞（W3，由脚本生成）
│   ├── demo_prism.json      由 analyze_object.py 生成的单物体场景（W3）
│   ├── demo_curved.json     柱 + 胶囊落上平台（W5，GJK/EPA 演示）
│   └── generated/           实验生成的场景（自动产生）
├── assets/                  测试用美术资源（不规则棱柱的 OBJ/STL、点阵点云）
├── scripts/
│   ├── run_phase0_experiments.py
│   ├── run_sim.py
│   ├── analyze_object.py          网格/点云 → 质量属性 + 凸包 + 案例目录
│   ├── assemble_scene.py          物体案例 → 场景资产（只存引用）
│   ├── reproduce.py               重放所有案例并逐字节校验产物
│   ├── make_pile_scene.py         生成容器堆积场景（确定性、防初始嵌入）
│   ├── make_hull_scene.py         生成凸包演示场景（四面体/八面体/棱柱/球）
│   ├── bench_solver_strategy.py   求解器策略决策实验（迭代 vs 子步）
│   ├── bench_broadphase.py        宽阶段加速比基准（空间哈希 vs 暴力）
│   ├── bench_massprops.py         质量属性误差研究（N 的影响、修正公式极限）
│   ├── bench_warm_start.py        暖启动 A/B 对照（开/关 × 迭代次数）
│   └── sync_docs.py               把能力表 / 基准数字 / 产物大小同步进文档
└── tests/
    ├── run_all.py                  一键运行全部测试
    ├── test_rigid_sphere.py        积分器 / 静置 / 能量 / 弹性碰撞
    ├── test_offcenter_collision.py 偏心碰撞专项
    ├── test_multicontact.py        多体同时接触专项
    ├── test_toi_split.py           球 TOI 分段与初始校验
    ├── test_toi_convex.py          盒/凸包 TOI：顶点精确解、SAT 轴保守推进
    ├── test_box_collision.py       盒-球 / 盒-地面 与转动动力学
    ├── test_box_box.py             盒-盒 SAT 与接触流形
    ├── test_broadphase.py          空间哈希宽阶段
    ├── test_massprops.py           质量属性与点云惯量
    ├── test_hull.py                凸包构造 / 主轴并入姿态 / 广义 SAT
    ├── test_object_input.py        OBJ/STL/点云解析、分析命令
    ├── test_asset.py               输出资产规格与自检
    ├── test_capabilities.py        能力矩阵：表 / 代码行为 / 文档 三方一致
    ├── test_evidence.py            文档数字与真值（基准存档 / 产物）一致
    └── test_warm_start.py          暖启动：收益、与 TOI 共存、确定性、能量
    ```

**分层原则**：`src/` 里一行 `import bpy` 都不许有。物理求解器要能在普通
Python 里跑测试，不能拖着一个 Blender 环境。

---

## 当前能力与限制

**已实现**

- 半隐式欧拉积分 + 四元数姿态更新
- 球-球 / 球-地面 / 球-盒 / **盒-盒（SAT）** / **盒-地面（4 点流形）** 解析碰撞
- **曲面几何（柱 / 胶囊）接入通用窄相**：支撑映射 + GJK/EPA。每种几何只需写一个
  `support(direction)`（球 1 行、胶囊 1 行、柱 3 行），五种几何的 15 种物体对、
  5 种地面组合全部可用；TOI 用"GJK 距离 + 保守推进"，与解析解差 0.1 ms。
  球-球/球-盒/盒-盒仍走原来的解析解与 SAT（与通用路径逐位对照：法向 0°、
  深度 0.0000 mm）
- **锥 / 台 / 椭球 / 球冠四种参数化图元**：闭式惯量（质心不在原点，构造时平移到原点），
  窄相复用同一套支撑映射 + GJK/EPA，地面接触由"支撑特征通用解"承担——端面圆盘给 4 点
  面接触、尖端/极点给 1 点。它们已经进能力矩阵，可以**直接**写进场景 JSON
  （`docs/02` §5.1），不必再借组合体包装；验收见 `tests/test_ground_generic.py` 第 10 项
- 顺序冲量求解：法向冲量、库仑摩擦（双切向）、Baumgarte 位置修正
- **暖启动**：跨帧复用上一子步的法向累积冲量，键为 (body_a, body_b, `feature_id`)。
  只暖启动法向（切向会累积成力矩）、每个子步只施加一次（TOI 切段时不能重复计费）。
  低迭代时穿透降一半、残余速度降两个量级；对照实验见 `results/w4/bench_warm_start.txt`
- **接触点身份契约**：`feature_id` 在同一物体对内唯一、并由**几何特征**决定——
  盒用顶点序号（`4*ix+2*iy+iz`）、回转体用"端面序号 × 4 + 圆周序号"、GJK 流形用
  `tag_a*512 + tag_b`，标签统一由 `support.feature_world_tagged()` 产出。
  此前地面多点共用 `0`、GJK 流形用列表下标：偏心负载下四角累积冲量实测
  0.011 / 0.015 / 0.030 / 0.026（差近 3 倍），共用键会让它们互相覆盖。见
  `docs/07` §4.1、`tests/test_contact_identity.py`。
  强度要说清：**批内唯一是硬要求**。跨帧稳定已达到"局部特征级"——盒/凸包顶点、
  母线端点、回转体端面槽位都稳定（圆周起点已固定在体坐标系，见 `docs/07` §4.2），
  但端面/侧面/单点模式切换时会失效。完整匹配要等两侧局部锚点 + 有效性校验
- 完整转动动力学：惯量张量、世界坐标逆惯量、力矩冲量、转动动能
- 静态物体（逆质量为零，求解器无分支）
- **接触流形裁剪**：参考面/入射面 + 多边形裁剪，最多 4 点，带**跨帧稳定的特征 ID**
- **空间哈希宽阶段**：O(N²) → 约 O(N)。400 体散开时检测快 49×、整步快 36×；
  密集堆叠时 7.8×。保留暴力基线用于加速比对照
- **TOI 分段求解**：按预测接触时刻切分子步，接触按时间先后逐个传递；
  链式碰撞穿透从 150 mm 降到 0.000 mm，代价 +16% 时间
- **TOI 覆盖任意凸体**：盒/凸包落地走**顶点精确解**（凸体的极值必在顶点，
  轨迹是直线），盒-盒走 SAT 轴保守推进（旋转用 $|\omega|R$ 作严格上界）。
  棱柱落地穿透 10.533 → **0.148 mm**，盒砸盒 5.439 → 0.856 mm。
  切分点故意多留 200 µm——相切时裁剪多边形退化成**单点流形**，反而让求解器
  只解掉 1/3 的法向速度（实测比不切分还差）。顺带把候选轴按方向去重
  （六棱柱 324 个叉积只产出 8 条轴），8 体场景整体快了 1.65×
- **初始状态校验**：拒绝仿真开始前就互相嵌入的场景（可显式放行）
- **质量属性**：闭合网格走散度定理（解析精确）、等质量点云走求和 + `ΔI = (1/6)Σmᵢhᵢ²`
  截断修正。点阵采样修正后达 **1e-16**；完整 3×3 张量经主轴对角化折叠进姿态，
  引擎的对角实现无需改动
- **任意凸多面体碰撞**：点集 → 凸包（增量法 + 共面合并）→ 广义 SAT。
  球/盒/凸包任意两两组合；凸包质量属性自动归一化到主轴系
- **物体输入管道**：`analyze_object.py` 一条命令从 OBJ/STL/点云算出体积、质心、
  完整惯量张量、主惯量、凸包代理，并可生成能直接加载的场景。
  OBJ 支持负索引与多边形面；STL 支持 ASCII/二进制（含"头部写 solid 的二进制"）
- **输出资产 `object_asset/1`**：一条命令出三件套——`.asset.json`（数据，含
  两个坐标系与互转、引擎交接三件套、8 条自检、sha256 溯源）、
  `.blend`（渲染网格 + 独立碰撞线框 + 质心/主轴标记，统一在 body 系）、
  `.preview.png`（QA 板，一眼核对数字与形状）。**逐字节确定**，可进版本库。
  JSON **不存网格顶点**（几何在 `input/` 与 `.blend` 里），路径全部相对化
- **场景资产 `scene_asset/1`**：只存名称、引用、指纹与环境配置，**不复制
  任何物体级数值**；20 个物体的场景仍是 2–3 KB。展开成引擎快照是单独一步，
  快照明确标注"不是真值"
- **质量分布能落地**：`ObjectSpec.inertia` 让资产算好的主惯量真的进入求解器。
  之前引擎无条件按碰撞几何反推，空心盒会得到 1.25 倍的真实惯量（24.8% 误差）
- **场景资产直连仿真**：`run_sim.py --scene-asset` 解析引用后直接喂引擎，
  不必先生成中间快照；只有渲染前才展开一次（并明确标注是派生物）
- **可复现是断言而不是声明**：`reproduce.py` 重放每条 `request.json` 并逐字节
  比对产物，顺带验证求解器确定性（同一场景两次跑出相同的位姿序列）
- 位姿序列导出 → Blender 关键帧 → FFmpeg 视频

> **规则网格（体素化）已整体移除。** 每格体积相同、每点质量相同 ⇒ 密度必然恒定，
> 表达不了密度分布；它那个"修正项精确"只是这个恒等式的副产品，不值得为它维护
> 一条独立管道。密度分布改由**逐点质量**（`from_points(weights=...)`）表达，
> 与网格无关。

**尚未实现**

- 凸分解（V-HACD 类）：组合体已经能手工拼出凹体（`CompoundGeometry`，见 `docs/02`），
  但还不能从单一网格**自动**分解成凸块
- 凸包简化（网格超过 64 面时直接报错，尚未自动降面）
- **点云截断修正的密度估计**：现在用最近邻距离直接当 $h_i$，对非规则排布有
  系统性偏差（同样密度下点阵与泊松的 $r_1$ 差 1.8 倍）。规格见 `docs/12`
- **资产批处理**：一次一个文件；多物体场景需手工拼
- 资产材质继承：固定一套灰蓝，还不能从源网格继承
- PLY / GLTF；多物体批量处理
- 流形块求解（同一流形的点当成一组解 LCP；暖启动已能缓过来一部分）
- 曲面上"面贴面"的精细裁剪：柱/胶囊与平面成片接触时，接触点是按支撑特征
  采样出来的（端面 4 点、侧面 2 点），不是把两个多边形真正裁剪求交；
  深度逐点按投影算，不会"整片吸住"，但边缘点的横向位置有采样精度
- 可变形固体
- `earliest_toi` 仍是 O(N²)：宽阶段只加速了窄相，没有进 TOI，物体多时它是新瓶颈
- TOI 对多面体**故意提前 200 µm**（两盒恰好相切时裁剪多边形退化成单点，
  单点流形解不掉法向速度）。所以 TOI 返回的时刻不是接触时刻：`0.375000` 的
  解析解会读到 `0.375100`，这不是误差，读数时别按解析解逐位比

---

## 下一步

按"计分收益 / 成本"排，不是按技术吸引力排：

1. **求解器热路径**：接触点跨子步缓存（不再每子步重跑宽相 + 窄相）、堆叠场景关 TOI、
   睡眠机制——目标是 500 体 ≤ 50 ms/步，这是"算法加速"一项的证据门槛
2. **流形块求解**：4 点当作 2×2 块解 LCP。暖启动已经补上了低迭代时的下限
   （见 `results/w4/bench_warm_start.txt`），块求解是"高迭代时更准"的另一半
3. **场景批量参数扫描 + 对照表**：`patch.py` 已支持点号路径覆盖，缺一个
   `bench_scene_sweep.py`（恢复系数 / 摩擦 / 迭代 / 子步 / 格子边长）
4. **主视觉出片**：1080p、相机运动、多段剪辑。渲染是核心要求，也是口头报告的
   视觉展示素材

**暂不排期**：可变形固体（XPBD）、凸分解 / 组合体、凸包简化。前两者是分值项，
但单人黑色部分上限 15 分、当前 5+5+5 已顶格，投入不产生增量；若确实要做，
先按公告要求联系助教确认是否额外计分。
