# -*- coding: utf-8 -*-
"""画一张管线示意（报告图 1）：顶部一条主链 + 底部渲染出口。"""

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "_pipeline.png")

S = 3                    # 超采样
W, H = 300, 128
FONT = "C:/Windows/Fonts/msyh.ttc"

image = Image.new("RGB", (W * S, H * S), "white")
draw = ImageDraw.Draw(image)
INK = (30, 36, 44)
LINE = (70, 78, 88)
MUTED = (110, 118, 128)


def font(size):
    return ImageFont.truetype(FONT, int(size * S), index=0)


def box(x, y, w, h, title, sub, fill):
    draw.rounded_rectangle(
        [x * S, y * S, (x + w) * S, (y + h) * S],
        radius=3.5 * S, fill=fill, outline=LINE, width=1 * S,
    )
    draw.text(((x + w / 2) * S, (y + h / 2 - 5.5) * S), title,
              font=font(7.4), fill=INK, anchor="mm")
    draw.text(((x + w / 2) * S, (y + h / 2 + 4.5) * S), sub,
              font=font(5.2), fill=MUTED, anchor="mm")


def arrow(x1, y1, x2, y2, width=1.6):
    draw.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=LINE, width=int(width * S))
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / length, dy / length
    tip = 3.2
    draw.polygon(
        [
            (x2 * S, y2 * S),
            ((x2 - ux * tip + uy * 1.7) * S, (y2 - uy * tip - ux * 1.7) * S),
            ((x2 - ux * tip - uy * 1.7) * S, (y2 - uy * tip + ux * 1.7) * S),
        ],
        fill=LINE,
    )


TOP_Y, TOP_H = 12, 32
BOXES = [
    ("场景", "JSON", 8),
    ("宽相", "空间哈希", 60),
    ("窄相", "SAT · GJK", 112),
    ("流形", "patch 求交", 164),
    ("求解", "顺序冲量", 216),
    ("积分", "半隐式欧拉", 268),
]
WIDTH = 24 if False else 0
for title, sub, x in BOXES:
    box(x, TOP_Y, 30, TOP_H, title, sub, (233, 240, 248) if x < 112 else (255, 243, 224))

for index in range(len(BOXES) - 1):
    x1 = BOXES[index][2] + 30
    x2 = BOXES[index + 1][2]
    arrow(x1 + 1.5, TOP_Y + TOP_H / 2, x2 - 1.5, TOP_Y + TOP_H / 2)

# 积分 → 渲染
arrow(283, TOP_Y + TOP_H, 283, 66)
box(268, 68, 30, 26, "渲染", "bpy / Cycles", (233, 245, 233))

draw.text((8 * S, 70 * S), "每子步重复", font=font(4.6), fill=MUTED)
draw.text((8 * S, 77 * S), "宽相 → 窄相 → 流形 → 求解", font=font(4.6), fill=MUTED)
draw.text((8 * S, 84 * S), "每个子步结束后积分一次", font=font(4.6), fill=MUTED)

draw.text((40 * S, 100 * S), "接触点身份 feature_id 串起跨帧暖启动", font=font(4.6), fill=MUTED)

image.resize((W * 2, H * 2), Image.LANCZOS).save(OUT)
print("written:", OUT)
