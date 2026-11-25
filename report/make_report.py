# -*- coding: utf-8 -*-
"""生成两页 SIGGRAPH 风格中期报告（A4 双栏）。

运行：``python report/make_report.py``
输出：``report/高子豪_中期报告.pdf``
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "高子豪_中期报告.pdf")

# 图 1 由 make_diagram.py 生成；缺了就现画
DIAGRAM = os.path.join(ROOT, "report", "_pipeline.png")
if not os.path.exists(DIAGRAM):
    import subprocess
    import sys as _sys

    subprocess.check_call([_sys.executable, os.path.join(ROOT, "report", "make_diagram.py")])

# ---------------------------------------------------------------- 字体
PDF_FONTS = [
    ("CN", "C:/Windows/Fonts/msyh.ttc", 0),
    ("CN-Bold", "C:/Windows/Fonts/msyhbd.ttc", 0),
    ("CN-Mono", "C:/Windows/Fonts/consola.ttf", 0),
]
for name, path, index in PDF_FONTS:
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
pdfmetrics.registerFontFamily("CN", normal="CN", bold="CN-Bold", italic="CN", boldItalic="CN-Bold")

PAGE_W, PAGE_H = A4
MARGIN = 15 * mm
GAP = 6 * mm
COL_W = (PAGE_W - 2 * MARGIN - GAP) / 2

# ---------------------------------------------------------------- 样式
TITLE = ParagraphStyle(
    "title", fontName="CN-Bold", fontSize=15.5, leading=19, alignment=TA_CENTER,
    spaceAfter=2,
)
SUBTITLE = ParagraphStyle(
    "subtitle", fontName="CN", fontSize=8.6, leading=11, alignment=TA_CENTER,
    textColor=colors.HexColor("#444444"), spaceAfter=4,
)
AUTHOR = ParagraphStyle(
    "author", fontName="CN", fontSize=9, leading=12, alignment=TA_CENTER,
)
ABSTRACT = ParagraphStyle(
    "abstract", fontName="CN", fontSize=8.3, leading=11.6, alignment=TA_JUSTIFY,
    wordWrap="CJK", leftIndent=6, rightIndent=6,
)
H1 = ParagraphStyle(
    "h1", fontName="CN-Bold", fontSize=9.6, leading=12, spaceBefore=6, spaceAfter=2.5,
    keepWithNext=1,
)
H2 = ParagraphStyle(
    "h2", fontName="CN-Bold", fontSize=8.6, leading=11, spaceBefore=4, spaceAfter=1.5,
)
BODY = ParagraphStyle(
    "body", fontName="CN", fontSize=8.4, leading=11.8, alignment=TA_JUSTIFY,
    wordWrap="CJK", spaceAfter=2.2,
)
BULLET = ParagraphStyle(
    "bullet", parent=BODY, leftIndent=8, bulletIndent=1, spaceAfter=1.4,
)
CAPTION = ParagraphStyle(
    "caption", fontName="CN", fontSize=7.3, leading=9.2, alignment=TA_JUSTIFY,
    wordWrap="CJK", spaceBefore=2, spaceAfter=4,
    textColor=colors.HexColor("#333333"),
)
CELL = ParagraphStyle("cell", fontName="CN", fontSize=7.2, leading=9.0, wordWrap="CJK")
CELL_B = ParagraphStyle("cellb", fontName="CN-Bold", fontSize=7.2, leading=9.0, wordWrap="CJK")
CELL_H = ParagraphStyle(
    "cellh", fontName="CN-Bold", fontSize=7.2, leading=9.0, wordWrap="CJK",
    textColor=colors.white,
)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullet(text):
    """一条要点整体不被分栏截断。"""
    return KeepTogether([Paragraph(text, BULLET, bulletText="·")])


def grid(data, widths, header=True, align="LEFT", padding=1.6):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F3B45")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.HexColor("#2F3B45")),
        ]
    table.setStyle(TableStyle(style))
    return table


def figure(path, width, caption):
    from PIL import Image as PILImage

    probe = PILImage.open(os.path.join(ROOT, path))
    height = width * probe.height / float(probe.width)
    image = Image(os.path.join(ROOT, path), width=width, height=height)
    return [image, P(caption, CAPTION)]


def figure_row(entries, caption):
    """两张图并排（各自带 (a)/(b) 标号），共用一条图注。"""
    from PIL import Image as PILImage

    half = COL_W * 0.5 - 3
    cells = []
    for path, label in entries:
        probe = PILImage.open(os.path.join(ROOT, path))
        height = half * probe.height / float(probe.width)
        cells.append([
            Image(os.path.join(ROOT, path), width=half, height=height),
            P("<b>(%s)</b>" % label, CAPTION),
        ])
    row = Table([cells], colWidths=[COL_W * 0.5, COL_W * 0.5])
    row.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    caption_style = ParagraphStyle("cap2", parent=CAPTION, spaceBefore=1)
    return [KeepTogether([row, P(caption, caption_style)])]


# ---------------------------------------------------------------- 正文
story = []

story.append(P("物理仿真：从零实现的刚体仿真与 Blender 渲染管线", TITLE))
story.append(P("中期报告 · Midterm Report · 题目三 物理仿真", SUBTITLE))
story.append(P("高子豪　清华大学　高等计算机图形学（2026 秋）　单人完成", AUTHOR))
story.append(Spacer(1, 3 * mm))
story.append(P(
    "<b>摘要</b>　本项目在不调用任何现成物理引擎的前提下，自写刚体仿真内核并接入 Blender"
    " 出片：窄相覆盖球/盒/地面解析解、盒-盒 15 轴 SAT 与任意凸体的 GJK/EPA，接触流形由"
    "两侧支撑集在切平面上求交得到；宽相用空间哈希，另有按顶点外推的 TOI 与顺序冲量求解器"
    "（含暖启动）。截至中期，核心三项与「15 分稳妥路径」的全部技术项已完成，进度较计划提前"
    "约两周；26 个模块、244 项自动化检查全绿，其中能力矩阵与文档数字均由真值生成并逐字比对。",
    ABSTRACT,
))
story.append(Spacer(1, 2.5 * mm))
story.append(NextPageTemplate("body"))

# ---- 1 目标与组件
story.append(P("1　项目目标与计划的技术组件", H1))
story.append(P(
    "目标是交付一个<b>可实际运行、可出片的刚体仿真器</b>：核心要求（一种类型、简单几何、"
    "精良出片，10 分）全部覆盖，稳妥路径（刚体×刚体 5 + 算法加速 5 + 场景参数化 5）拿满"
    "15 分；若 W7 顺利，再冲刺可变形固体（跨类型交互 +6，物理交互上限 13 分）。下表是计划"
    "的技术组件与当前状态。"
))
story.append(Spacer(1, 1.2 * mm))
component_rows = [
    [Paragraph("层 / 组件", CELL_H), Paragraph("内容与状态", CELL_H)],
    [Paragraph("数学 · 几何 · 场景", CELL_B), Paragraph("四元数 / 向量 / 二维凸多边形；参数化几何、解析惯量、网格散度定理、点云惯量；JSON → SceneSpec 与参数补丁——已完成", CELL)],
    [Paragraph("窄相", CELL_B), Paragraph("球/盒/地面解析解；盒-盒 15 轴 SAT；任意凸体 GJK + EPA；支撑 patch 切平面求交流形——已完成", CELL)],
    [Paragraph("宽相 · TOI", CELL_B), Paragraph("按朝向的 AABB + 空间哈希；TOI 三档（顶点外推 / SAT 保守推进 / 最近点），带 200 µm 余量——已完成", CELL)],
    [Paragraph("求解器", CELL_B), Paragraph("顺序冲量、恢复系数、双切向摩擦、Baumgarte 偏置、暖启动——已完成", CELL)],
    [Paragraph("资产管道", CELL_B), Paragraph("OBJ/STL/点云 → 惯量 + 碰撞代理 + 可跑场景 + .blend/QA 预览——已完成", CELL)],
    [Paragraph("渲染", CELL_B), Paragraph("SceneSpec → bpy（Cycles/EEVEE）、静态帧、位姿序列动画、资产 QA——已完成", CELL)],
    [Paragraph("证据链", CELL_B), Paragraph("基准存档 → 文档生成块 → 逐字比对（能力矩阵、基准表）——已完成", CELL)],
]
story.append(KeepTogether(grid(component_rows, [COL_W * 0.30, COL_W * 0.70])))
story.append(Spacer(1, 1.2 * mm))
story.append(P(
    "技术路线上的关键取舍是<b>不引入任何第三方物理库</b>：碰撞、接触求解、摩擦与时间步进"
    "全部自写，Blender 只承担渲染与编码。代价是每步都要自己验证；收益是所有结果可解释、"
    "可复现，「算法加速」也有真实基线。"
))
story.append(P(
    "与考核项的对应：核心三项由 W2 起自动满足；同类型交互 5 + 算法加速 5 + 场景参数化 5 "
    "在 W4 结束前到位，稳妥路径 15 分已锁定；跨类型交互（+6）作为冲刺项，不影响下限。"
    "单人 12 周预算 60–120 h，当前累计约 70 h（估计）。"
))
story.append(P(
    "边界：不做流体、布料/薄壳、GPU 并行加速与渲染质量加分——它们的工程量与「自写 + 可核验」"
    "这条约束冲突，或在 12 周预算内挤掉出片时间；可变形固体作为唯一冲刺项保留，且不影响下限。"
))
story.append(Spacer(1, 1.8 * mm))
story.append(Image(
    os.path.join(ROOT, "report", "_pipeline.png"),
    width=COL_W, height=COL_W * 256 / 600.0,
))
story.append(P(
    "图 1　一步仿真的数据流：每子步重复「宽相 → 窄相 → 流形 → 求解」，积分后交给渲染层；"
    "接触点身份 <span face='CN-Mono'>feature_id</span> 把跨帧的冲量连起来做暖启动。",
    CAPTION,
))

# ---- 2 时间表
schedule_rows = [
    [Paragraph("周次", CELL_H), Paragraph("日期", CELL_H), Paragraph("阶段目标与交付", CELL_H)],
    [Paragraph("W1", CELL_B), Paragraph("10/22–10/28", CELL), Paragraph("纯静态渲染管线：一条命令出 PNG，8 组参数对比图　<b>已完成（10/27）</b>", CELL)],
    [Paragraph("W2", CELL_B), Paragraph("10/29–11/04", CELL), Paragraph("积分器 + 球刚体 + 冲量求解 + 位姿驱动动画；6 项数值验证　<b>已完成</b>", CELL)],
    [Paragraph("W3", CELL_B), Paragraph("11/05–11/11", CELL), Paragraph("盒-盒 SAT + 同类型交互 + 接触流形与身份　<b>已完成（10/27）</b>", CELL)],
    [Paragraph("W4", CELL_B), Paragraph("11/12–11/18", CELL), Paragraph("空间哈希宽相 + 场景参数化 + 加速比表　<b>已完成</b>；提前项：暖启动、文档守卫", CELL)],
    [Paragraph("W5", CELL_B), Paragraph("11/19–11/25", CELL), Paragraph("主视觉出片：500 体场景、参数扫描对照表　<b>进行中</b>", CELL)],
    [Paragraph("W6", CELL_B), Paragraph("11/26–12/02", CELL), Paragraph("撰写并提交 2 页中期报告（12/02）　<b>本周</b>", CELL)],
    [Paragraph("W7", CELL_B), Paragraph("12/03–12/09", CELL), Paragraph("冲刺项：可变形固体（形状匹配），视 W5 余量决定", CELL)],
    [Paragraph("W8", CELL_B), Paragraph("12/10–12/16", CELL), Paragraph("双向耦合 + 加速收尾 + 最终出片（硬检查点：可答辩成片）", CELL)],
    [Paragraph("W9–W10", CELL_B), Paragraph("12/17–12/30", CELL), Paragraph("口头报告（5+3 分钟，含 live demo）与课堂焦点展示", CELL)],
    [Paragraph("W11–W12", CELL_B), Paragraph("12/31–01/13", CELL), Paragraph("≥3 页书面报告、源码与复现脚本整理（01/13 提交）", CELL)],
]
H1_KEEP = ParagraphStyle("h1keep", parent=H1, keepWithNext=1)
story.append(P("2　时间表", H1_KEEP))
story.append(grid(schedule_rows, [COL_W * 0.13, COL_W * 0.21, COL_W * 0.66], padding=1.0))

# ---- 3 已完成
story.append(P("3　已完成的技术组件与方法", H1))

story.append(P("3.1　渲染管线与场景参数化（W1）", H2))
story.append(P(
    "场景是纯 JSON：改文件即可换画面，不需要改代码。渲染层把 SceneSpec 翻译成 bpy 场景，"
    "静态单帧、位姿序列动画（→ mp4）、资产 QA 预览三种出口共用同一份构建代码。W1 用 8 组"
    "参数对比（相机、灯光软硬、材质粗糙度、构图）确认了「工业级渲染」这条核心要求，"
    "后续所有物理实验都复用这条管线出片。"
))

story.append(P("3.2　球体刚体与顺序冲量求解器（W2）", H2))
story.append(P(
    "求解器采用顺序冲量：逐接触点施加法向冲量（含恢复系数，且恢复系数按<b>碰撞前</b>接近"
    "速度计算，避免能量注入），摩擦在两个正交切向分别限幅——不严格解摩擦锥，这是实时引擎"
    "的通行简化；位置误差用 Baumgarte 偏置项修正。W2 用 6 项数值验证（自由落体、斜面、"
    "双球对撞、恢复系数、摩擦静止角、堆叠沉降）锁定了符号与量纲。"
))
story.append(P(
    "W4 追加的策略实验回答了「钱花在哪」：以凹槽静置为例，同样 12 次遍历，<b>12 次迭代 / 1 子步"
    "下沉 0.741 mm，而 12 子步 / 1 迭代只下沉 0.016 mm</b>（约 46 倍），代价是 0.43 ms → 1.42 ms/步。"
    "结论是子步优先于迭代，默认配置因此定为 substeps=3、iterations=8。"
))

story.append(P("3.3　盒-盒 SAT 与接触流形（W3，含本轮修复）", H2))
story.append(P(
    "盒-盒用 15 轴 SAT（3+3 面法向、9 条棱叉积），得到法向与深度后在切平面上裁剪出接触"
    "多边形，稀疏到 4 点。这里有两个只有测试能抓到的坑：稀疏点若取多边形极值顶点，"
    "微扰会让位置跳一个顶点间距（实测 1.7 cm），改成从重心沿四个方向射线求边界插值点后"
    "降到 1.3e-5 m；接触身份若按「最近槽位」分配，轴对齐面接触会整组轮转一位，改成两侧"
    "各自按极角排序后逐个对齐才稳定。身份 <span face='CN-Mono'>feature_id</span> 同时是暖启动的键。"
))
story.append(P(
    "本轮最大的修正是把「最近采样点配对」换成<b>支撑集切平面求交</b>：两侧各取高密度支撑"
    "patch（圆盘最多 64 边），投影到法向切平面求二维凸集交集，再按每侧<b>真实表面</b>逐点"
    "算间隙并先裁掉分离点、后稀疏。修完后偏置柱-盒场景的四点从全部落在投影外变为落在"
    "搭接带内，倾斜端面/母线的深度也从「一个平均值」变成逐点不同。"
))

story.append(P("3.4　宽相与算法加速（W4）", H2))
story.append(P(
    "宽相用按朝向计算的 AABB 建空间哈希，只对同桶及相邻桶的物体做窄相。200 个散布球的"
    "实测：窄相候选对 <b>19 900 → 0</b>，检测快 <b>23.9×</b>，含窄相与求解的端到端快 <b>19.5×</b>"
    "（16.998 ms → 0.871 ms/步）；N=50/100/200 时端到端加速分别为 4.86×/10.2×/19.5×。"
))
story.append(P(
    "求解器侧的加速是暖启动：跨帧缓存累积冲量，键为 (body_a, body_b, feature_id)，只暖启动"
    "法向、每子步施加一次。6 层盒塔静置 1.5 s，1 次迭代时顶层漂移由 <b>285 mm 降到 8.5 mm</b>，"
    "8 次迭代时由 2.76 mm 降到 0.87 mm——它补的是低迭代时的下限，不能用来抵掉迭代次数。"
))

story.append(P("3.5　任意凸体、TOI 与资产管道（W3–W4 追加）", H2))
story.append(P(
    "任意凸体走 GJK + EPA，几何由点云增量凸包构造，主轴并入姿态后复用同一套 SAT（换轴源"
    "不换算法）；球-凸包用最近点而不是 SAT。TOI 分三档：凸包落地按顶点外推（顶点模型纯平移"
    "精确、有自转是一阶近似且无保守上界）、盒-盒走 SAT 轴保守推进、球-盒走最近点；返回时刻"
    "统一再提前 200 µm 作为穿透余量。资产管道一条命令把 OBJ/STL/点云变成惯量 + 碰撞代理 + "
    "可跑场景，点云额外给出截断修正。"
))

# ---- 4 结果
story.append(P("4　可视化结果与量化指标", H1))
story.append(P(
    "下图 (a) 是 W1 的主视觉渲染（Cycles），(b) 是资产管道产出的棱柱资产 QA 预览——两者走"
    "同一条 SceneSpec → bpy 管线，差别只在输入是手写 JSON 还是 OBJ/点云。视频类产物包括球堆"
    "落地、盒塔堆叠、偏置盒-盒三组演示，均由位姿序列渲染成 mp4。"
))
metric_rows = [
    [Paragraph("指标", CELL_H), Paragraph("数值", CELL_H), Paragraph("条件", CELL_H)],
    [Paragraph("窄相候选对", CELL_B), Paragraph("19 900 → 0", CELL), Paragraph("200 球，spread 布局", CELL)],
    [Paragraph("端到端加速", CELL_B), Paragraph("19.5×", CELL), Paragraph("16.998 → 0.871 ms/步", CELL)],
    [Paragraph("同遍历次数下沉", CELL_B), Paragraph("0.741 → 0.016 mm", CELL), Paragraph("12 迭代/1 子步 vs 12 子步/1 迭代", CELL)],
    [Paragraph("暖启动顶层漂移", CELL_B), Paragraph("285 → 8.5 mm", CELL), Paragraph("6 层盒塔，1 次迭代", CELL)],
    [Paragraph("单次柱-盒 collide", CELL_B), Paragraph("0.95 ms", CELL), Paragraph("其中 GJK/EPA 0.2 ms", CELL)],
    [Paragraph("自动化检查", CELL_B), Paragraph("244 项 / 26 模块", CELL), Paragraph("约 45 s，无需 Blender", CELL)],
]
story.append(KeepTogether(grid(metric_rows, [COL_W * 0.30, COL_W * 0.30, COL_W * 0.40])))
story.append(Spacer(1, 1.5 * mm))
story.extend(figure_row(
    [("results/phase0/08_hero.png", "a"), ("results/w3/prism/output/prism.preview.png", "b")],
    "图 2　(a) Phase 0 主视觉：参数化场景 + Cycles 渲染，仅改 JSON 即可换构图与光照；"
    "(b) 资产管道产物：点云 → 凸包 + 惯量 → 场景 → .blend 与 QA 预览。",
))
story.append(P(
    "工程侧还做了一条「文档 = 真值」的守卫：能力矩阵与基准数字由源码与 "
    "<span face='CN-Mono'>results/</span> 生成后写入文档，测试逐字比对。它不能保证真值本身"
    "正确（曾经把盒落地标成「含自转精确」也一路绿灯），但能杜绝「文档说的和代码做的不是"
    "一回事」这类过期说法。"
))

# ---- 5 剩余计划
story.append(KeepTogether([
    P("5　剩余技术组件的计划", H1),
    bullet("<b>W5（进行中）</b>：500 体场景落地，产出宽相与求解器在大规模下的耗时表；"
           "补充场景参数扫描对照表（同一场景扫 substeps/iterations/摩擦系数）。"),
]))
story.append(bullet("<b>W6</b>：完成本报告并冻结中期接口，W7 起不再改公共 API。"))
story.append(bullet("<b>W7（冲刺，条件触发）</b>：可变形固体（形状匹配 + 位置约束），与刚体做双向"
                    "耦合；若 W5 余量不足则按稳妥路径放弃，不影响 15 分。"))
story.append(bullet("<b>W8</b>：成片硬检查点——主视觉视频 + 全部演示片段定型，之后只做包装。"))
story.append(bullet("<b>技术债</b>：<span face='CN-Mono'>SupportPatch</span> 契约收口（两点仍可能标"
                    " polygon）；空交集回退的 A/B 对称性（现只保证位置不偏）；"
                    "<span face='CN-Mono'>earliest_toi</span> 仍是 O(N²)。"))

# ---- 6 工具
story.append(P("6　外部工具", H1))
story.append(P(
    "<b>Blender 3.6 + bpy</b>：唯一的外部运行时依赖，负责场景构建、Cycles/EEVEE 渲染与 mp4 编码；"
    "<b>Python 3.10</b>：物理、几何、测试全部只用标准库（无 numpy），保证「自写」可核验；"
    "<b>Git / GitHub</b>：按阶段打提交，报告中的每个数字都能对应到一次运行；<b>VS Code</b>：编辑与调试。"
))
story.append(P(
    "<b>未使用</b>：MuJoCo、Bullet、PhysX、Unity、UE 等任何物理引擎或求解器。核验方式很直接——"
    "依赖清单里没有它们，代码里搜不到 <span face='CN-Mono'>mj_step</span> / "
    "<span face='CN-Mono'>pybullet</span> 之类的调用；早期旧项目中的自研数学模块（四元数、变换）"
    "经重写后移植，动力学与碰撞全部新写。W5–W8 不打算引入新的外部依赖：出片继续用同一套"
    " Blender 管线，测试与基准仍只依赖 Python 标准库。"
))

# ---- 7 结论
story.append(P("7　中期结论与复现方式", H1))
story.append(P(
    "计划中的 W1–W4 与两项提前项（暖启动、文档守卫）已全部完成，核心三项与稳妥路径的 15 分"
    "技术项均已到位，未使用任何现成物理引擎。当前主要风险不在功能而在出片：500 体场景的"
    "渲染耗时、以及密集堆叠在低迭代下的稳定性，还需 W5 的两周缓冲验证。因此 W7 的可变形"
    "固体只在 W5 有余量时触发，否则按稳妥路径放弃。"
))
story.append(P(
    "复现：<span face='CN-Mono'>python Code/tests/run_all.py</span>（244 项，约 45 s，不需要"
    " Blender）；<span face='CN-Mono'>python Code/scripts/run_sim.py &lt;scene.json&gt;</span>"
    "（Blender 3.6，出 mp4）；<span face='CN-Mono'>python Code/scripts/sync_docs.py</span>"
    "（把基准数字同步进文档）。上面每个数字都能由这三条命令重新得到，这也是答辩时的证据链。"
))

# ---------------------------------------------------------------- 文档
HEADER_FRAME_H = 76 * mm


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(MARGIN, 9 * mm, "高子豪 · 中期报告 · 物理仿真")
    canvas.drawRightString(PAGE_W - MARGIN, 9 * mm, "第 %d 页" % doc.page)
    canvas.setStrokeColor(colors.HexColor("#DDDDDD"))
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(
    OUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=16 * mm,
    title="物理仿真：从零实现的刚体仿真与 Blender 渲染管线（中期报告）",
    author="高子豪",
)

first = PageTemplate(
    id="first",
    frames=[
        Frame(MARGIN, PAGE_H - MARGIN - HEADER_FRAME_H, PAGE_W - 2 * MARGIN, HEADER_FRAME_H,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="head"),
        Frame(MARGIN, 16 * mm, COL_W,
              PAGE_H - 2 * MARGIN - HEADER_FRAME_H - GAP,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="c1"),
        Frame(MARGIN + COL_W + GAP, 16 * mm, COL_W,
              PAGE_H - 2 * MARGIN - HEADER_FRAME_H - GAP,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="c2"),
    ],
    onPage=on_page,
)

body = PageTemplate(
    id="body",
    frames=[
        Frame(MARGIN, 16 * mm, COL_W, PAGE_H - MARGIN - 16 * mm - 4 * mm,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="b1"),
        Frame(MARGIN + COL_W + GAP, 16 * mm, COL_W, PAGE_H - MARGIN - 16 * mm - 4 * mm,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="b2"),
    ],
    onPage=on_page,
)

doc.addPageTemplates([first, body])
doc.build(story)
print("written:", OUT)
