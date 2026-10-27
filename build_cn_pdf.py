# -*- coding: utf-8 -*-
"""Generate the Chinese version of ACG_Project_Announcement.pdf"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, Table, TableStyle)

# ---------------- fonts ----------------
def _reg(name, candidates):
    for path, idx in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx)
                                        if idx is not None else TTFont(name, path))
                return True
            except Exception:
                pass
    return False

if _reg('CN', [("C:/Windows/Fonts/simsun.ttc", 0),
               ("C:/Windows/Fonts/msyh.ttc", 0)]):
    _CN = 'CN'
else:
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    _CN = 'STSong-Light'

_CNB = 'CNB' if _reg('CNB', [("C:/Windows/Fonts/simhei.ttf", None),
                             ("C:/Windows/Fonts/msyhbd.ttc", 0)]) else _CN
_CNI = 'CNI' if _reg('CNI', [("C:/Windows/Fonts/simkai.ttf", None),
                             ("C:/Windows/Fonts/simsun.ttc", 0)]) else _CN

try:
    pdfmetrics.registerFontFamily('CN', normal=_CN, bold=_CNB,
                                  italic=_CNI, boldItalic=_CNB)
except Exception:
    pass

# Latin / math face: correct proportional metrics + full ASCII math coverage.
# (The CJK face is half-width for ALL ASCII and lacks U+2212 -> blank 1em box.)
MATH = 'Times-Roman'

RED = colors.HexColor('#A02020')
GREY = colors.HexColor('#666666')
REDHEX = '#A02020'

# ---------------- styles ----------------
S = {}
S['title'] = ParagraphStyle('title', fontName=_CNB, fontSize=17, leading=24,
                            alignment=TA_CENTER, spaceAfter=3)
S['subtitle'] = ParagraphStyle('subtitle', fontName=_CN, fontSize=12, leading=17,
                               alignment=TA_CENTER, spaceAfter=8, textColor=GREY)
S['note'] = ParagraphStyle('note', fontName=_CNI, fontSize=8.4, leading=13.5,
                           alignment=TA_LEFT, wordWrap='CJK',
                           textColor=colors.HexColor('#555555'), spaceAfter=9)
S['legend'] = ParagraphStyle('legend', fontName=_CN, fontSize=9.0, leading=14,
                             alignment=TA_LEFT, wordWrap='CJK', spaceAfter=4)
S['h1'] = ParagraphStyle('h1', fontName=_CNB, fontSize=12.5, leading=18,
                         alignment=TA_LEFT, wordWrap='CJK',
                         spaceBefore=12, spaceAfter=5, textColor=RED)
S['h2'] = ParagraphStyle('h2', fontName=_CNB, fontSize=10.6, leading=16,
                         alignment=TA_LEFT, wordWrap='CJK',
                         spaceBefore=8, spaceAfter=3)
S['body'] = ParagraphStyle('body', fontName=_CN, fontSize=9.8, leading=15.6,
                           alignment=TA_LEFT, wordWrap='CJK', spaceAfter=4)
S['b1'] = ParagraphStyle('b1', parent=S['body'], leftIndent=13, bulletIndent=2,
                         spaceAfter=2.5)
S['b2'] = ParagraphStyle('b2', parent=S['body'], leftIndent=26, bulletIndent=15,
                         spaceAfter=2)
S['b3'] = ParagraphStyle('b3', parent=S['body'], leftIndent=39, bulletIndent=29,
                         spaceAfter=2)
S['formula'] = ParagraphStyle('formula', fontName=MATH, fontSize=11.5, leading=17,
                              alignment=TA_CENTER, spaceBefore=6, spaceAfter=7)
S['cell'] = ParagraphStyle('cell', fontName=_CN, fontSize=8.6, leading=12.5,
                           alignment=TA_LEFT, wordWrap='CJK')
S['cellb'] = ParagraphStyle('cellb', fontName=_CNB, fontSize=8.6, leading=12.5,
                            alignment=TA_CENTER, wordWrap='CJK')
S['cellc'] = ParagraphStyle('cellc', fontName=_CN, fontSize=8.6, leading=12.5,
                            alignment=TA_CENTER, wordWrap='CJK')
S['cellm'] = ParagraphStyle('cellm', fontName=MATH, fontSize=9.6, leading=12.5,
                            alignment=TA_CENTER)

# ---------------- helpers ----------------
CORE_TAG = '【核心】'

def CORE(t):
    """Any item flagged with 【核心】 is a core (red) requirement: render it red."""
    if t.startswith(CORE_TAG):
        return ('<font color="%s"><b>%s</b>%s</font>'
                % (REDHEX, CORE_TAG, t[len(CORE_TAG):]))
    return t

SIGMA = '<font name="%s">Σ</font>' % _CN

def SYM(s):
    """Inline math symbol (italic Latin) so that 'O' is not read as '0'."""
    return '<font name="Times-Italic">%s</font>' % s

F = []
def H1(t): F.append(Paragraph(t, S['h1']))
def H2(t): F.append(Paragraph(t, S['h2']))
def BD(t): F.append(Paragraph(CORE(t), S['body']))
def NT(t): F.append(Paragraph(t, S['note']))
def FM(t): F.append(Paragraph(t, S['formula']))
def SP(h=4): F.append(Spacer(1, h))
def L1(t): F.append(Paragraph(CORE(t), S['b1'], bulletText='•'))
def L2(t): F.append(Paragraph(CORE(t), S['b2'], bulletText='–'))
def L3(t): F.append(Paragraph(CORE(t), S['b3'], bulletText='·'))

def mk_table(data, widths, header=True, align_center_cols=(), math_cols=()):
    rows = []
    for r_i, row in enumerate(data):
        rr = []
        for c_i, c in enumerate(row):
            if header and r_i == 0:
                stl = S['cellb']
            elif c_i in math_cols:
                stl = S['cellm']
            elif c_i in align_center_cols:
                stl = S['cellc']
            else:
                stl = S['cell']
            rr.append(Paragraph(str(c), stl))
        rows.append(rr)
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#B0B0B0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8E8E8')),
    ]))
    return t

# ================= content =================
F.append(Paragraph('高等计算机图形学 2026', S['title']))
F.append(Paragraph('课程项目说明（中文版）', S['subtitle']))
NT('译注：英文原版以<font color="%s">红色文字</font>表示该题的<b>核心要求</b>（必须完整实现，'
   '否则项目将被视为未完成），黑色文字为可选加分项。本中文版沿用同一标注方式：'
   '<font color="%s">红色条目即核心要求</font>，并附 <b>【核心】</b> 标签以便检索。'
   '本文档为课程公告的中文译本，评分与要求以英文原版为准。' % (REDHEX, REDHEX))
SP(2)
F.append(Paragraph(
    '<b>图例</b>　<font color="%s"><b>红色</b></font> ＝ 核心要求（必须完成）　｜　'
    '黑色 ＝ 可选加分项（括号内为分值）' % REDHEX, S['legend']))

F.append(mk_table([
    ['考核环节', '截止时间', '团队 / 个人', '分值', '符号'],
    ['中期书面报告', '2026 年 12 月 2 日', '团队', '5 分', 'M'],
    ['期末口头报告', '第 15 周', '个人', '25 分', 'O'],
    ['课堂焦点展示', '2026 年 12 月 30 日', '团队', '附加分', 'B'],
    ['期末书面报告', '2027 年 1 月 13 日', '个人', '10 分', 'W'],
    ['源代码提交', '2027 年 1 月 13 日', '团队', '系数', 'k'],
], [4.6*cm, 3.5*cm, 2.4*cm, 2.0*cm, 1.6*cm], align_center_cols=(1, 2, 3), math_cols=(4,)))
SP(7)

BD('项目总分为 40 分，另加附加分。')
FM('S = M + k (O + W) + B')
BD('其中 %s 表示期末个人口头报告成绩，评分同时依据所展示的结果与学生的技术讲解；'
   '%s 表示期末书面报告在扣除各项分数之后的成绩。' % (SYM('O'), SYM('W')))
BD('代码核验系数 %s 的取值范围为 0 到 1，初始值为 1。若源代码核验发现所提交的实现与所声称的功能'
   '或个人贡献之间存在实质性出入，%s 可能被调低。' % (SYM('k'), SYM('k')))

H1('1 中期书面报告')
BD('请提交一份两页的报告，使用 SIGGRAPH 模板总结你的项目进展。报告中需包含以下内容：')
L1('你选择的项目题目')
L1('你的项目目标，以及你计划实现的技术模块')
L1('整个项目的详细时间安排')
L1('你已经完成的技术模块、所使用的方法，以及目前已有的可视化结果')
L1('你对其余技术模块的实现计划')
L1('你已经使用或计划使用的外部工具')

H1('2 期末个人口头报告')
BD('每位学生都需要在第 15 周课余时间向助教进行约 5 分钟的个人展示，展示结束后有 3 分钟的助教提问环节。')
BD('口头报告将从两个方面进行评分：')
L1('<b>视觉展示</b>：准备一份简短的幻灯片或其他展示材料，呈现清晰的可视化结果与现场演示（live demo），'
   '以体现项目的质量以及你个人的具体贡献。')
L1('<b>技术讨论</b>：讲解你所使用的方法与实现过程，并回答助教针对你的工作及其技术细节提出的问题。'
   '你的得分将反映你讲解的清晰度、正确性，以及你对实现本身的理解程度。')
BD('根据期末个人口头报告的成绩，助教将挑选约 20 个团队在第 16 周的课堂上进行展示。')

H1('3 课堂焦点展示')
BD('由助教挑选的约 20 个团队将在第 16 周（2026 年 12 月 30 日）的课堂上进行焦点展示，'
   '通过可视化结果与现场演示来展示他们的项目。')
BD('所有选课学生都有机会为自己最喜欢的项目投票。最受欢迎的项目最多可获得 3 个附加分；'
   '所有进行课堂焦点展示的团队均可获得 2 个附加分。')

H1('4 期末个人书面报告')
BD('每位学生都必须提交一份报告，使用 SIGGRAPH 模板描述<font color="%s">自己的贡献</font>。'
   '报告长度不得少于三页，并且必须包含：引言、方法、结果、讨论、个人贡献声明以及参考文献。'
   '每位学生必须独立撰写自己的报告；<b>即使在同一团队内部，也不允许共享书面内容</b>。' % REDHEX)
BD('此外，你必须提交项目的源代码（推荐提供 GitHub 链接）。无论你是独立完成还是团队合作，'
   '都建议进行规范的 commit 管理，以便助教评估你的工作量。在期末书面报告中，'
   '请明确指出代码的哪些部分是由你贡献的。')

H1('5 项目选题')
BD('在以下各节中，<font color="%s">红色文字代表该题的核心要求</font>。'
   '你必须完整实现这些模块，否则项目将被视为未完成。你可以选择不去做某些可选模块（黑色部分），'
   '但完全不做整个项目是不允许的。' % REDHEX)
BD('口头报告的评分有两种可能的情况：')
L1('<b>独立完成</b>：你必须完成红色部分，由此自动获得 10 分。此后，你可以任意选择若干个黑色部分，'
   '最多再获得 15 分。')
L1('<b>两人组队</b>：你的团队必须完成红色部分，由此每人自动获得 15 分。'
   '之后，每人可以各选<font color="%s">不同的</font>黑色部分组合，最多再获得 10 分'
   '（团队合计 20 分）。这意味着红色部分是合作完成的，但每一个黑色部分只能由一名团队成员得分。' % REDHEX)
BD('期末书面报告并不针对项目的各个模块分别计分，而是按下式计算：')
FM('W = 10 - min (10, %s d<sub>i</sub>)' % SIGMA)
BD('其中 <font name="Times-Italic">d<sub>i</sub></font> 表示第 i 个模块因讲解不完整、'
   '或报告与实现不符而产生的扣分。')
BD('总体来说，我们鼓励大家组队完成项目。由于部分技术点的难度较高，我们认为独立完成 15 分'
   '与组队完成 20 分在难度上大致相当，因此总工作量是相近的。不过，如果你选择组队，'
   '你们更有可能获得更高的完成度与展示质量，从而在焦点展示中拿到更好的分数乃至附加分。')
BD('如果你希望在项目中实现任何高级技术，请在口头报告之前联系助教，讨论其可行性与评分标准。')

H1('5.1 题目一：图像渲染')
BD('本题要求你实现一个渲染器，可以是基于硬件的（GPU），也可以是基于软件的（CPU）。'
   '目标是生成一张静态图像。注意：<font color="%s">必须使用路径追踪（path tracing）渲染器</font>，'
   '基于光栅化的渲染器不予接受。你需要实现的功能如下：' % REDHEX)
L1('【核心】<b>基础</b>：实现路径追踪算法，能够正确处理漫反射与镜面反射材质。')
L1('【核心】<b>场景搭建</b>：构建一个具有美感的自定义场景，几何体可以是你自己创建的，'
   '也可以从网上获取（须注明来源）。（整洁度与美观度 1 分）')
L1('【核心】<b>加速结构</b>：实现诸如 BVH（层次包围盒）的加速结构。基于硬件的渲染器不作此要求，'
   '因为其加速结构已内置。（使用表面积启发式 SAH 或其他高级算法 2 分）')
L1('【核心】<b>材质</b>：创建一种（非平凡的）自定义材质。可选：')
L2('透射材质')
L2('Principled BSDF（2 分）')
L2('多层材质（2 分）')
L2('毛发、皮肤等材质的渲染（2 分）')
L1('【核心】<b>纹理</b>：创建你自己的（非平凡）纹理，并实现正确的纹理映射。可选：')
L2('颜色纹理')
L2('法线贴图、高度贴图、属性贴图，或任意程序化纹理映射（每项 1 分，最多 2 分）')
L2('实现自适应 mipmap 算法（2 分）')
L1('<b>重要性采样</b>：为路径追踪使用更高级的采样算法。（带俄罗斯轮盘赌的重要性采样、'
   '多重重要性采样 2 分）')
L1('<b>体绘制</b>：可选：')
L2('次表面散射（2 分）')
L2('均匀介质体渲染（1 分）')
L2('非均匀介质体渲染（1 分）')
L2('通道独立的次表面散射（1 分）')
L2('体发射（1 分）')
L2('体积 Alpha 阴影（2 分）')
L1('<b>特殊视觉效果</b>：可选：')
L2('【核心】运动模糊、景深')
L2('【核心】Alpha 阴影')
L2('卡通风格渲染（2 分）')
L2('色散（2 分）')
L1('<b>光照</b>：可选：')
L2('【核心】点光源与面光源')
L2('使用 HDR 的环境光照，例如天空盒（2 分）')
L1('【核心】<b>抗锯齿</b>：实现一种抗锯齿算法。')
L1('<b>基于仿真的内容生成</b>：最多 2 分。')

H1('5.2 题目二：交互式游戏')
BD('本题要求你设计一个交互式游戏。游戏的类型与内容不限，但<font color="%s">不得使用'
   'Unity、UE5 之类高度完善、功能完整的游戏引擎</font>。一个不错的起点是使用 Three.js。'
   '（如果你选用的引擎导致代码量明显少于 Three.js，你就应当担心工作量是否达到本项目的要求。）'
   '此外，严禁使用任何涉及可视化编程的引擎。' % REDHEX)
BD('下面列出游戏需要实现的基础功能与可选功能：')
L1('<b>美术资源</b>：')
L2('【核心】场景布置：合理的物体几何、纹理与材质')
L2('环境光照（2 分）')
L2('音画同步（2 分）')
L1('<b>动画</b>：')
L2('运动学：')
L3('【核心】包含动画，无论是简单的关键帧动画还是更复杂的动画')
L3('关节体：例如动物或植物，包含绑定（rigging）与蒙皮（skinning）（3 分）')
L2('动力学（考虑物体间的交互）：')
L3('流体仿真：例如水或风（3 分）')
L3('柔软 / 可变形物体（3 分）')
L3('碰撞处理（3 分）')
L1('<b>交互控制</b>：需保证在控制器输入下动作自然、直观、流畅：')
L2('【核心】主游戏角色的控制')
L2('【核心】第三人称或第一人称视角下的镜头运动控制（如果开发的是角色固定在屏幕中央的 2D 游戏，则不要求）')
L2('在碰撞或移动时实现恰当的镜头晃动，以增强沉浸感（3 分）')
L1('<b>UI 设计</b>：')
L2('【核心】合理的游戏开始与结束界面')
L2('额外的辅助界面（3 分）')
L2('界面友好、布局美观（2 分）')
L1('<b>高级渲染</b>：')
L2('运动模糊、景深等特殊视觉效果（3 分）')
L2('使用 PBR 材质的路径追踪渲染（3 分）')
L1('<b>其他高级游戏特性</b>：')
L2('多人模式（3 分）')
L2('支持多种控制器，例如不仅支持鼠标键盘，还支持实体手柄（3 分）')
L2('可自定义角色外观（3 分）')
L2('游戏难度随进程递增（3 分）')
L2('存档与读档系统，允许玩家保存并继续进度（3 分）')
L2('在线多人系统（4 分）')
L1('<b>完成度</b>：')
L2('【核心】一个功能完整、可实际游玩的游戏')
L2('【核心】用户手册或操作说明')
L2('便于安装，或可直接在线访问（3 分）')

H1('5.3 题目三：物理仿真')
BD('本题要求你仿真物体之间的物理交互，并使用工业级渲染器将结果渲染出来。'
   '<font color="%s">你必须自己实现核心的仿真算法</font>。通用的数值库与编程框架，'
   '例如 CUDA、NVIDIA Warp 和 Taichi，是允许使用的；但成品物理引擎与现成的仿真求解器不允许使用。'
   '具体功能如下：' % REDHEX)
L1('【核心】<b>基本类型</b>：至少选择一种。可选：')
L2('刚体')
L2('可变形固体')
L2('流体')
L2('布料 / 薄壳')
L1('<b>物理交互</b>（最多 13 分）：')
L2('实现同种或不同种物体之间的物理交互，包括关节连接（articulation）、碰撞、自碰撞与耦合（coupling）。')
L2('某一种交互要拿到满分，必须实现有意义的<b>双向</b>交互。')
SP(3)
F.append(mk_table([
    ['', '刚体', '可变形固体', '流体', '布料 / 薄壳'],
    ['刚体', '最多 5 分', '最多 6 分', '最多 7 分', '最多 6 分'],
    ['可变形固体', '最多 6 分', '最多 5 分', '最多 7 分', '最多 6 分'],
    ['流体', '最多 7 分', '最多 7 分', '最多 5 分', '最多 8 分'],
    ['布料 / 薄壳', '最多 6 分', '最多 6 分', '最多 8 分', '最多 5 分'],
], [3.0*cm, 3.0*cm, 3.2*cm, 2.6*cm, 3.2*cm], align_center_cols=(1, 2, 3, 4)))
SP(6)
L1('<b>几何</b>：')
L2('【核心】简单几何体')
L2('复杂（基于网格或基于方程的）几何体（最多 3 分）')
L1('<b>加速</b>：')
L2('算法加速（最多 5 分）。例如：')
L3('用于碰撞检测的加速结构')
L3('高级数值求解器')
L2('并行加速（最多 3 分）：')
L3('CPU 多线程、GPU 加速，或两者结合。')
L3('得分取决于你的实现投入与实测加速比。<b>仅仅启用一个 GPU 后端</b>（例如 ti.init(arch=ti.gpu)）'
   '本身并不能得分。')
L3('GPU 加速推荐使用 CUDA 与 NVIDIA Warp。')
L1('<b>控制</b>：')
L2('自定义场景配置（最多 5 分）')
L2('交互式场景。可选：')
L3('仿真过程中预定义的操作（最多 3 分）')
L3('实时交互（最多 3 分）')
L1('【核心】<b>渲染</b>：使用工业级渲染器输出一段精良的视频。')
L2('尽管这不是一个渲染项目，视觉效果同样重要。若结果仅以原始仿真图元（例如未着色的粒子或线段）'
   '呈现，渲染部分的得分将十分有限。')
L2('离线渲染与实时渲染均可接受。')
L2('渲染是必须完成的。渲染质量更高可获得额外加分（最多 6 分）。')

H1('5.4 题目四：ProjectX')
BD('如果上述项目题目都不符合你的兴趣，你可以提出一个新的项目想法。为此，'
   '你必须制作一份评分表，列出你项目计划涵盖的所有技术点，形式与前面列出的其他项目类似。'
   '然后，你需要在 <b>10 月 18 日之前</b>预约助教并取得助教的同意。'
   '只有在获得助教批准的情况下，你才可以去做自己的项目想法；否则，你仍然需要从上述题目列表中选择一个'
   '作为你的最终项目。')

# ================= build =================
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(_CN, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2*cm, A4[1] - 1.25*cm, '高等计算机图形学 2026 · 课程项目说明（中文版）')
    canvas.setStrokeColor(colors.HexColor('#CCCCCC'))
    canvas.setLineWidth(0.4)
    canvas.line(2*cm, A4[1] - 1.45*cm, A4[0] - 2*cm, A4[1] - 1.45*cm)
    canvas.line(2*cm, 1.45*cm, A4[0] - 2*cm, 1.45*cm)
    canvas.drawRightString(A4[0] - 2*cm, 1.0*cm, '第 %d 页' % doc.page)
    canvas.restoreState()

doc = BaseDocTemplate('ACG_Project_Announcement_中文版.pdf', pagesize=A4,
                      leftMargin=2*cm, rightMargin=2*cm,
                      topMargin=1.9*cm, bottomMargin=1.9*cm,
                      title='高等计算机图形学 2026 课程项目说明（中文版）',
                      author='ACG 2026')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='n')
doc.addPageTemplates([PageTemplate(id='all', frames=[frame], onPage=on_page)])
doc.build(F)

n_core = sum(1 for f in F if hasattr(f, 'text') and CORE_TAG in getattr(f, 'text', ''))
print('OK -> ACG_Project_Announcement_中文版.pdf  (core items:', n_core, ')')
