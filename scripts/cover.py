#!/usr/bin/env python3
"""cover: 视频封面。图片走和成片一样的 PC-98 处理（640×360 锁 16 色 + Bayer → 放大 3 倍 + 扫描线），标题用片中的像素大字和顶栏。

  cover.py            出 封面/封面_A_主视觉.png、封面_B_落日伸手.png（1920×1080）
"""
import numpy as np
from PIL import Image

import mv
import part1_lib as L
import pc98ify as P

W, H = L.W, L.H
OUT = L.ROOT / '封面'
FIX = lambda k: tuple(int(c) for c in mv.FIX[k])

VARIANTS = {
    'A_主视觉': dict(src='封面/主视觉_原图.png', title_x=18, title_y=30, sub_y=128),
    'B_落日伸手': dict(src='素材/CG/p3_1b_落日伸手_b.png', title_x=18, title_y=30, sub_y=128),
}


def quantize(img):
    q = mv.Quantized(lambda t: img, 0, 1)
    return q.at(0)


def topbar(img):
    img = img.copy()
    img[:18] = FIX('ink')
    img = L.draw_text(img, (6, 0), '▶ CLAUDE_POP.EXE', FIX('paper'))
    img = L.draw_text(img, (180, 0), 'FULL VER.  2:36', FIX('orange'))
    return L.draw_text(img, (W - 92, 0), '♪ LIVE', FIX('yel'))


def title(img, x, y, sub_y):
    img = mv.slam2(img, "I'M UPPING MY", 1.0, 2.0, x, style='block', anim='none', scale=2, y=y)
    img = mv.slam2(img, 'P(DOOM)', 1.0, 2.0, x, style='block', anim='none', scale=4, y=y + 30, color='yel')
    img = L.draw_text(img, (x + 3, sub_y + 1), 'a Claude Pop MV', FIX('ink'))
    return L.draw_text(img, (x + 2, sub_y), 'a Claude Pop MV', FIX('paper'))


def main():
    OUT.mkdir(exist_ok=True)
    for name, v in VARIANTS.items():
        img = quantize(L.load_rgb(v['src']))
        img = title(topbar(img), v['title_x'], v['title_y'], v['sub_y'])
        big = P.upscale(img)                                              # 1920×1080 + 扫描线
        Image.fromarray(big).save(OUT / f'封面_{name}.png')                 # 只出 16:9（4:3 往哪边裁都会切掉标题或人物）
        print('wrote', name)


if __name__ == '__main__':
    main()
