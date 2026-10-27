# 场景 JSON 规范

> 对应实现：`Code/src/scene/scene_spec.py`、`Code/src/geom/geometry.py`
> 最后更新：2026-10-23

场景文件是纯 JSON，UTF-8 编码。**改这个文件就能换画面，不需要改代码**——
这是评分表里「自定义场景配置」那一项的落点，也是做参数对比实验的基础。

---

## 1. 完整骨架

```json
{
  "name": "场景名（仅用于日志）",

  "camera": {
    "position": [x, y, z],
    "target":   [x, y, z],
    "fov":      45.0
  },

  "lights": [
    { "name": "...", "type": "area|sun|point",
      "position": [x, y, z], "target": [x, y, z],
      "energy": 340.0, "size": 2.6, "color": [1, 1, 1] }
  ],

  "ground": {
    "enabled": true, "size": 40.0,
    "color": [0.62, 0.62, 0.64], "roughness": 0.55
  },

  "objects": [
    { "name": "...",
      "geometry":  { "type": "box", "size": [1, 1, 1] },
      "transform": { "position": [0, 0, 0.5] },
      "material":  { "base_color": [0.72, 0.055, 0.03],
                     "roughness": 0.4, "metallic": 0.0 },
      "mass": 100.0 }
  ],

  "render": {
    "resolution": [960, 540], "samples": 64, "engine": "CYCLES",
    "denoise": true, "world_color": [0.055, 0.058, 0.07],
    "world_strength": 0.3, "exposure": 0.0, "film_transparent": false
  }
}
```

除 `objects[].geometry` 外，**所有字段都有默认值**，可以只写关心的部分。

---

## 2. `camera`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `position` | `[3]` | `[4,4,6]` | 相机位置（米） |
| `target` | `[3]` | `[0,0,0.5]` | 注视点；相机 -Z 轴指向这里 |
| `fov` | `number` | `45` | **水平**视场角（度）。小=长焦压缩，大=广角畸变 |

> 视场角作用在长边。分辨率改成竖屏时，`fov` 会自动变成垂直视场角。

**常用取值**：`35–45` 接近人眼；`28–32` 长焦、适合特写；`50+` 广角、空间感强但有畸变。

---

## 3. `lights[]`

最多建议 3–4 盏（主光 / 补光 / 轮廓光）。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | string | `light_i` | 光源名，便于在 Blender 里辨认 |
| `type` | string | `area` | `area` / `sun` / `point` |
| `position` | `[3]` | `[3,3,5]` | 位置。`sun` 只用方向，位置无意义 |
| `target` | `[3]` | `[0,0,0]` | 照射目标，决定朝向 |
| `energy` | number | `500` | 见下方说明 |
| `size` | number | `2.0` | 见下方说明 |
| `color` | `[3]` | `[1,1,1]` | RGB，0–1 |

**`energy` / `size` 在不同 type 下语义不同**

| type | `energy` 单位 | `size` 含义 | 用途 |
|---|---|---|---|
| `area` | 瓦特 | 正方形发光面边长（米） | 产品级布光首选；`size` 越大阴影越软 |
| `sun` | 辐照度 W/m² | 角直径（度），限制在 0.01–30 | 模拟日光，方向平行 |
| `point` | 瓦特 | 光源半径（越大越软） | 装饰性小光源 |

**曝光经验值**（5 米距离、普通浅色地面）

- 主光 `area`：**250–450 W / size 2–3** 比较稳
- 补光：主光的 1/4 左右，颜色略偏冷
- 轮廓光：主光的 1/2 左右，从背后打
- 超过 900 W 基本一定过曝（地面被打成纯白、颜色被洗掉）

---

## 4. `ground`

地面只提供视觉参考与阴影承托，**当前不参与物理**。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | `true` | 是否创建 |
| `size` | number | `40` | 平面边长（米） |
| `color` | `[3]` | `[0.78,0.78,0.80]` | 基础色 |
| `roughness` | number | `0.6` | 0=镜面，1=完全漫反射 |

> 地面不要用接近 1.0 的白：会吃掉所有层次。**0.5–0.7 之间**的浅灰更耐看。

---

## 5. `objects[]`

### 5.1 `geometry`（必填）

| `type` | 参数 | 说明 |
|---|---|---|
| `box` | `size: [sx, sy, sz]` | 三轴**全长**（不是半长） |
| `sphere` | `radius` | 半径 |
| `cylinder` | `radius`, `height` | 轴线沿局部 **Z**，底面在 -h/2（渲染支持，碰撞未接入） |
| `capsule` | `radius`, `height` | `height` 只算**中间圆柱段**，总高 = height + 2r（同上） |
| `hull` | `points: [[x,y,z], ...]`，可选 `max_faces`（默认 64） | **任意凸多面体**：由点集求凸包 |

尺寸单位是**米**，且**就是最终物理尺寸**。质量、惯量都直接用它算，
不读取 `transform.scale`。

#### `hull` 的两条约定

```json
{"type": "hull", "points": [[0,0,0], [1,0,0], [0.4,1.1,0], [0.3,0.4,0.6]]}
```

1. **点的顺序无所谓**，内部点会被自动剔除，无需先转成面表或三角网格。
2. **`transform.position` 指的是质心**，不是原始点集的坐标原点。构造时会把顶点
   平移到质心、并旋转到惯量主轴系（姿态里自动并入这个旋转，用户不需要处理）。
   所以把一个 `hull` 放在 `[0,0,2]`，它悬在空中的位置由**质心**决定。

`max_faces` 是规模保护：广义 SAT 的棱-棱轴数是 `E²`，超过 64 面（约 96 条棱 =
9216 条轴）会在**加载阶段**报错。碰撞代理应当用粗网格，通常 8–32 面足够；凹体
必须先用凸包近似，或用多个凸块拼（组合体尚未实现）。


### 5.2 `transform`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `position` | `[3]` | `[0,0,0]` | 世界坐标位置（指几何中心） |
| `orientation` | `[4]` | `[1,0,0,0]` | 四元数 `[w,x,y,z]` |
| `orientation_euler_deg` | `[3]` | — | XYZ 欧拉角（**度**），写起来更直观，优先于 `orientation` |
| `scale` | `[3]` | `[1,1,1]` | **仅视觉**缩放，物理计算忽略 |

> 手写场景建议用 `orientation_euler_deg`。例如"绕 Y 轴转 90° 让胶囊躺下"：
> `"orientation_euler_deg": [0, 90, 0]`。

### 5.3 `material`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | string | `mat_i` | 材质名 |
| `base_color` | `[3]` | `[0.8,0.8,0.8]` | **线性空间** RGB，0–1 |
| `roughness` | number | `0.5` | 0=镜面，1=哑光 |
| `metallic` | number | `0` | 0=电介质，1=金属 |
| `transmission` | number | `0` | >0 表现为玻璃 |
| `emissive` | number | `0` | 自发光强度 |

**`base_color` 是线性空间，不是屏幕 sRGB**
直接把取色器里的 0–255 除以 255 会明显偏亮、偏灰。
想要饱和的红，写 `[0.72, 0.055, 0.03]` 而不是 `[0.8, 0.3, 0.3]`。

**三组典型搭配**

| 效果 | 参数 |
|---|---|
| 哑光塑料 | `metallic=0`, `roughness=0.5–0.7` |
| 抛光金属 | `metallic=1`, `roughness=0.05–0.2` |
| 拉丝/磨砂金属 | `metallic=1`, `roughness=0.4–0.6` |
| 玻璃 | `transmission=1`, `roughness=0.05` |

### 5.4 `mass`

可选。不给时按 `体积 × 1000 kg/m³` 估算（约等于水的密度），保证 W2 直接可用。
Phase 0 不参与渲染，只是先把这个字段解析好，避免以后改 JSON 结构。

---

## 6. `render`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `resolution` | `[2]` | `[960,540]` | 输出分辨率 |
| `samples` | int | `64` | Cycles 路径采样数（EEVEE 下是 TAA 采样） |
| `engine` | string | `CYCLES` | `CYCLES` / `BLENDER_EEVEE` |
| `denoise` | bool | `true` | Cycles 用 OIDN 降噪 |
| `world_color` | `[3]` | `[0.055,0.058,0.07]` | 环境光颜色 |
| `world_strength` | number | `0.3` | 环境光强度（"底光"） |
| `exposure` | number | `0` | 曝光补偿（EV） |
| `film_transparent` | bool | `false` | 透明背景 |

**采样与噪声的关系**：噪声大致与 `1/sqrt(samples)` 成正比。
64 → 256 只把噪声减半，代价是 4 倍时间。**靠降噪比堆采样划算得多**。

---

## 7. 物理仿真段（`simulation` / `physics`）

静态渲染可以完全忽略这一段。要跑物理，必须在顶层加 `"simulation": {"enabled": true, ...}`。

### 7.1 `simulation` 字段

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | `false` | 关闭时 `run_sim.py` 会直接报错 |
| `physics_dt` | number | `1/240` | 每个物理步的时长（秒） |
| `substeps` | int | `2` | 每步内部再切分的子步数 |
| `iterations` | int | `12` | 顺序冲量迭代次数 |
| `baumgarte` | number | `0.25` | 位置修正强度 0–1 |
| `slop` | number | `0.002` | 允许的穿透容差（米） |
| `restitution_threshold` | number | `1.0` | 低于此接近速度忽略恢复系数（m/s） |
| `gravity` | `[3]` | `[0,0,-9.81]` | 重力加速度 |
| `ground_z` | number | `0` | 物理地面高度（与视觉地面无关） |
| `ground_friction` | number | `0.55` | 地面摩擦系数 |
| `ground_restitution` | number | `0.12` | 地面恢复系数 |
| `linear_damping` | number | `0.05` | 线速度指数衰减率（1/s） |
| `angular_damping` | number | `1.0` | 角速度指数衰减率（1/s），模拟滚动阻力 |
| `allow_initial_overlap` | bool | `false` | 允许初始互相嵌入（默认**拒绝**，见 7.3） |
| `overlap_tolerance` | number | `0.001` | 初始穿透的容许上限（米） |
| `use_toi_splitting` | bool | `true` | 按预测接触时刻切分子步（见 7.4） |
| `max_events_per_substep` | int | `8` | 每个子步最多切分多少次 |
| `broadphase` | string | `"spatial_hash"` | `"spatial_hash"` 或 `"brute"`（后者仅供加速比对照） |
| `broadphase_cell_size` | number | `0` | 空间哈希格子边长（米）；`0` 表示按物体尺寸自动估算 |
| `render_fps` | int | `24` | 位姿导出帧率 |
| `duration` | number | `3.5` | 仿真总时长（秒） |

### 7.2 `objects[].physics` 字段

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `static` | bool | `false` | 质量无穷大，不受力也不动 |
| `linear_velocity` | `[3]` | `[0,0,0]` | 初始线速度（m/s） |
| `angular_velocity` | `[3]` | `[0,0,0]` | 初始角速度（rad/s，世界坐标） |
| `restitution` | number | `0.25` | 恢复系数 0–1（接触时取两侧较大值） |
| `friction` | number | `0.5` | 摩擦系数（接触时取两侧几何平均） |

质量不写时按"体积 × 1000 kg/m³"反推。

### 7.3 初始互相嵌入会被拒绝

仿真开始前就互相嵌入的物体，会在第一个子步**同时**产生接触，求解器只能给出
"整体同时变形"的解——那是软材料的物理。对硬物，正确行为是接触按时间先后逐个传递。

实测（`tests/test_multicontact.py` 第 8 项）：三球对撞，仅把初始间距从"恰好相切"
改成一微米重叠，结果就从 `[0,0,3]`（只飞一个球，正确）跳到 `[-1,2,2]`。
**一微米的差别，结果完全不同。**

所以 `PhysicsWorld.from_scene()` 会主动检查，发现超过 `overlap_tolerance`
的穿透就抛异常，并列出所有违规的物体对。摆位置时记得留缝。

要临时放行（例如从存档恢复一个已经贴合的堆叠），显式写 `"allow_initial_overlap": true`。

### 7.4 TOI 分段求解

窄相只看得到"此刻已穿透"的物体。如果一个子步内先后发生多次碰撞（A 撞 B、
B 再撞 C），一次性检测会把它们当成"同时接触"。开启 `use_toi_splitting` 后，
求解器会在推进位置之前解析地算出**下一次接触的时刻**（球-球是一元二次方程，
球-地面是一元一次），把子步切开，每段只解算此刻真正发生的接触。

效果（`tests/test_toi_split.py`）：

| 场景 | 不分段 | 分段 |
|---|---:|---:|
| 单子步内链式碰撞的终态 | 错误 | `[0, 0, 60]` 正确 |
| 最大穿透深度 | 150 mm | **0.000 mm** |
| `sim_spheres` 480 步耗时 | 0.230 s | 0.266 s（+16%） |

静置物体不会触发事件（实测平均 0.000 次/步），所以没有持续的额外开销。

---

## 8. 约定与坑

1. **坐标系**：与 Blender 一致，**Z 轴向上**。`position` 指几何中心，
   所以一个 `size=[1,1,1]` 的盒子要落地，`position[2]` 应该写 `0.5`。
2. **旋转顺序**：`orientation_euler_deg` 是 **XYZ**（先 X，再 Y，最后 Z）。
3. **胶囊高度**：`height` 不含两端半球。半径 0.2、`height` 0.62 的胶囊，
   总高是 `0.62 + 0.4 = 1.02`，躺下时中心离地 `0.2`。
4. **物体名必须唯一**：W2 导出位姿序列时用它作稳定 ID，重名会在加载阶段直接报错。
5. **颜色别照抄屏幕值**：见 5.3。
6. **地面别用纯白**：见第 4 节。
