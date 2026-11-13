#!/usr/bin/env python3
"""sprite_demo: 精灵表 → 按拍换姿势的循环舞（「一张图出几个姿势」做 GIF 式有限动画的实验）。

  sprite_demo.py --sheet ../素材/精灵/gato_跳舞四姿势.png --out ../成片/demo_精灵表_Gato.mp4 [--gif ../成片/demo_精灵表_Gato.gif]

做法（参考片尾奏嘉宾的拆解：配角两三张姿势按拍来回换）：
  1. 精灵表一次生成（同一次生成里比例、画风最一致），透明底或纯品红底
  2. 按连通块切开，保留每张在表里的 y（同一条脚底线，跳起的那张自然高一点），x 按中心对齐
  3. 按 132 BPM 拍网格换姿势：前两小节一拍一换，后两小节半拍一换；换的瞬间挤压回弹 2 帧
  4. 背景用代码画（这里借第三部分 Gato 的蒸汽波落日），整帧锁 16 色 + Bayer
"""
import argparse
import subprocess

import cv2
import numpy as np
from PIL import Image

import codeshots_e as E
import mv
import part1_lib as L
import pc98ify as P

W, H, FPS = L.W, L.H, L.FPS
B = L.BEAT132


def cut_sheet(path, n=4, height=240):
    im = np.array(Image.open(path).convert('RGBA'))
    if (im[..., 3] < 250).mean() < .05:                                # 没有透明通道时按品红底抠
        rgb = im[..., :3].astype(int)
        im[..., 3] = np.where((rgb[..., 0] > 200) & (rgb[..., 2] > 200) & (rgb[..., 1] < 90), 0, 255)
    a = (im[..., 3] > 127).astype(np.uint8)
    blob = cv2.dilate(a, np.ones((25, 25), np.uint8))                 # 胡须、尾巴尖和身体连成一块
    k, lab, st, _ = cv2.connectedComponentsWithStats(blob)
    big = sorted(range(1, k), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:n]
    big.sort(key=lambda i: st[i, cv2.CC_STAT_LEFT])
    base = max(st[i, 1] + st[i, 3] for i in big)                       # 共同脚底线
    top = min(st[i, 1] for i in big)
    s = height / (base - top)
    out = []
    for i in big:
        x, y, w, h = st[i, :4]
        crop = im[y:y + h, x:x + w].copy()
        crop[..., 3] = np.where((lab[y:y + h, x:x + w] == i) & (crop[..., 3] > 127), 255, 0)
        crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        out.append((crop, int((base - (y + h)) * s)))                   # (贴图, 离脚底线的高度)
    return out


def choreo(t, t0):
    """(姿势序号, 这个姿势开始后过了多久)。前两小节一拍一换，后两小节半拍一换；跳起（3）放在每小节最后。"""
    beats = (t - t0) / B
    if beats < 8:
        seq, step = [0, 1, 0, 2, 0, 1, 0, 3], 1
    else:
        seq, step = [0, 1, 2, 1, 0, 2, 1, 3], .5
        beats -= 8
    k = int(np.floor(beats / step))
    return seq[k % len(seq)], (beats - k * step) * B


def frame(t, t0, sprites, bg_t):
    img = E.vapor(bg_t)
    pose, dt = choreo(t, t0)
    spr, lift = sprites[pose]
    sy, sx = (.9, 1.08) if dt < 2 / FPS else (1.0, 1.0)                 # 换姿势的一瞬间：挤扁 2 帧再弹回
    h, w = spr.shape[:2]
    spr = cv2.resize(spr, (int(w * sx), int(h * sy)), interpolation=cv2.INTER_NEAREST)
    h, w = spr.shape[:2]
    feet = 290 - lift - (10 if pose == 3 else 0)                        # 蒸汽波地平线下一点当舞台
    return L.paste_rgba(img, spr, W // 2 - w // 2, feet - h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gif')
    ap.add_argument('--t0', type=float, default=145.72, help='配的原曲从哪开始（尾奏，132 BPM）')
    ap.add_argument('--bars', type=int, default=4)
    a = ap.parse_args()
    sprites = cut_sheet(a.sheet)
    t0, t1 = a.t0, a.t0 + a.bars * 4 * B
    bg_t = lambda t: 90.9 + (t - t0) * .2                               # 落日慢慢沉
    rgb_at = lambda t: frame(t, t0, sprites, bg_t(t))
    q = mv.Quantized(rgb_at, t0, t1)
    n = int((t1 - t0) * FPS)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * P.S}x{H * P.S}',
                           '-r', str(FPS), '-i', '-', '-ss', f'{t0:.3f}', '-t', f'{n / FPS:.3f}', '-i', L.SONG,
                           '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', '-b:a', '192k', a.out], stdin=subprocess.PIPE)
    frames = []
    for i in range(n):
        t = t0 + i / FPS
        f = q.at(t)
        if (t - t0) / B >= 8: f = L.draw_text(f, (8, 4), 'HALF-BEAT', tuple(int(c) for c in mv.FIX['yel']))
        else: f = L.draw_text(f, (8, 4), 'ONE POSE / BEAT', tuple(int(c) for c in mv.FIX['yel']))
        frames.append(f)
        ff.stdin.write(P.upscale(f).tobytes())
    ff.stdin.close(); ff.wait()
    print('wrote', a.out)
    if a.gif:                                                           # GIF：后两小节（半拍一换）循环一遍，原尺寸 640×360
        k0 = int(8 * B * FPS)
        ims = [Image.fromarray(f) for f in frames[k0:]]
        ims[0].save(a.gif, save_all=True, append_images=ims[1:], duration=int(1000 / FPS), loop=0)
        print('wrote', a.gif)


if __name__ == '__main__':
    main()
