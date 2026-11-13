#!/usr/bin/env python3
"""clawd_extras.py — 把 ClaudeAnimationBase 渲出来的 Clawd 群演循环(纯绿底 PNG)抠成 RGBA 素材包。

输入:素材/ClaudeAnimationBase/out/xk/<style>/f0000.png ...(由 xk_extras.js 的 LOOPS 渲染)
输出:素材/群演/clawd_<style>.npz  (rgba: N×h×w×4 uint8,已裁到全循环的包围盒;bpm、beats=8)
      素材/群演/preview.png       (深色底上的抽帧预览)

  clawd_extras.py            处理 out/xk 下所有风格
"""
import glob
import os

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), '..', '素材')
SRC = os.path.join(ROOT, 'ClaudeAnimationBase', 'out', 'xk')
DST = os.path.join(ROOT, '群演')
BPM = 88  # 素材元数据;循环实际拍速以渲染时 ClaudeAnimationBase/src/config.js 的 bpm 为准


def key(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    bg = (hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)
    a = (~bg).astype(np.uint8) * 255
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a, (3, 3), 0)
    # 去绿边:绿通道不许超过红蓝的最大值
    rgb = img.copy()
    mx = np.maximum(rgb[..., 0], rgb[..., 2])
    rgb[..., 1] = np.minimum(rgb[..., 1], mx)
    return np.dstack([rgb, a])


def main():
    os.makedirs(DST, exist_ok=True)
    tiles = []
    for d in sorted(glob.glob(os.path.join(SRC, '*'))):
        style = os.path.basename(d)
        frames = [key(np.array(Image.open(f).convert('RGB'))) for f in sorted(glob.glob(os.path.join(d, 'f*.png')))]
        al = np.max(np.stack([f[..., 3] for f in frames]), 0) > 20
        ys, xs = np.where(al)
        y0, y1, x0, x1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
        rgba = np.stack([f[y0:y1, x0:x1] for f in frames])
        np.savez_compressed(os.path.join(DST, f'clawd_{style}.npz'), rgba=rgba, bpm=BPM, beats=8)
        print(f'[{style}] {len(frames)} frames  {x1 - x0}×{y1 - y0}')
        for k in (0, len(frames) // 3, 2 * len(frames) // 3):
            t = rgba[k].astype(np.float32)
            bg = np.full(t.shape[:2] + (3,), (40, 20, 60), np.float32)
            a = t[..., 3:] / 255
            tiles.append(cv2.resize((bg * (1 - a) + t[..., :3] * a).astype(np.uint8), (240, 200)))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    rows = [np.hstack([r, np.zeros((200, 1440 - r.shape[1], 3), np.uint8)]) for r in rows]
    Image.fromarray(np.vstack(rows)).save(os.path.join(DST, 'preview.png'))
    print('wrote', os.path.join(DST, 'preview.png'))


if __name__ == '__main__':
    main()
