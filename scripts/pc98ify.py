#!/usr/bin/env python3
"""pc98ify: PC-98 化的基础函数（live98.py、mv.py 都在用）：平涂、锁定调色板、Bayer 4x4 抖动量化、放大加扫描线、HUD 字体。

设计要点：
- 调色板每个镜头算一次后锁死（k-means 抽样全片），逐帧不重算 → 不闪。
- 用有序抖动（Bayer）不用误差扩散 → 画面动时网点不"沸腾"。
- HUD 在量化之后用调色板里的原色直接画，关抗锯齿 → 画面层严格只有 16 色（扫描线是放大后的显示效果）。
- 画面层按"一拍二"（12fps）更新，HUD 层 24fps，像日式有限动画。
- 底片本身应是 PC-98 画风（平涂、硬阴影、自带墨线），这里不另外描线。
"""
import os
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, S, FPS = 640, 360, 3, 24
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '素材', '字体', 'DotGothic16-Regular.ttf')

# 固定色（HUD 用），其余由 k-means 从画面里挑
FIXED = {'ink': (12, 11, 26), 'paper': (246, 241, 228), 'orange': (217, 119, 87),
         'blue': (34, 52, 150), 'yel': (255, 210, 63)}

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16


def load_frames(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def cel(img):
    """压成平涂色块：均值漂移把渐变收成赛璐璐一样的色面。"""
    return cv2.pyrMeanShiftFiltering(img, 5, 16)


def build_palette(cels, k):
    rng = np.random.default_rng(0)
    px = np.concatenate([c.reshape(-1, 3)[rng.choice(W * H, 6000, replace=False)] for c in cels[::3]])
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # 极暗像素交给固定的 ink，不让 k-means 在黑色上浪费名额，省下来给皮肤和头发的色阶
    lab = lab[lab[:, 0] > 40]
    _, _, centers = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .5), 4, cv2.KMEANS_PP_CENTERS)
    rgb = cv2.cvtColor(centers.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    pal = np.array(list(FIXED.values()) + [tuple(c) for c in rgb], dtype=np.uint8)
    return pal


def build_lut(pal):
    g = (np.arange(32) * 8 + 4).astype(np.float32)
    grid = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    wts = np.array([.30, .59, .11]) * 3
    d = (((grid[:, None, :] - pal[None].astype(np.float32)) ** 2) * wts).sum(-1)
    return d.argmin(1).astype(np.uint8).reshape(32, 32, 32)


def quantize(img, lut, spread=36.0):
    thr = np.tile(BAYER - .5, (H // 4 + 1, W // 4 + 1))[:H, :W, None]
    v = np.clip(img.astype(np.float32) + thr * spread, 0, 255).astype(np.uint8) >> 3
    return lut[v[..., 0], v[..., 1], v[..., 2]]


def bayer_mask(k):
    """Bayer 溶解遮罩：k∈[0,1]，用于闪白/转场，比透明度混合更像 PC-98。"""
    return np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W] < k


# ---------------- HUD ----------------
class Hud:
    def __init__(self, pal, shot):
        """shot：镜头配置 dict（title、meter、hits、alarm、lines、slams），见 live98.SHOT。"""
        self.pal = pal
        self.shot = shot
        self.ci = {n: i for i, n in enumerate(FIXED)}
        self.f16 = ImageFont.truetype(FONT_PATH, 16)
        # 未唱到的字用的暗色：调色板里离灰蓝最近的
        self.ci['dim'] = int(np.argmin(((pal.astype(int) - (110, 120, 170)) ** 2).sum(1)))

    def text(self, d, xy, s, c, font=None):
        d.text(xy, s, fill=int(self.ci[c]), font=font or self.f16)

    def big(self, im, s, x, y, scale, fg, shadow='ink', outline='orange'):
        """16px 字放大成方块大字（最近邻），带描边和投影。"""
        l, t, r, b = self.f16.getbbox(s)
        w, h = r - l + 2, b - t + 2
        m = Image.new('1', (w, h), 0)
        ImageDraw.Draw(m).text((1 - l, 1 - t), s, fill=1, font=self.f16)
        m = m.resize((w * scale, h * scale), Image.NEAREST)
        mask = np.array(m)
        a = np.array(im)
        def stamp(dx, dy, c):
            ys, xs = np.nonzero(mask)
            ys, xs = ys + y + dy, xs + x + dx
            ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            a[ys[ok], xs[ok]] = self.ci[c]
        stamp(scale, scale, shadow)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)): stamp(dx, dy, outline)
        stamp(0, 0, fg)
        return Image.fromarray(a, 'P')

    def draw(self, idx, t, song_t):
        im = Image.fromarray(idx, 'P')
        im.putpalette(self.pal.flatten().tolist())
        d = ImageDraw.Draw(im)
        d.fontmode = '1'
        C, sh = self.ci, self.shot
        # 顶栏
        d.rectangle((0, 0, W, 17), fill=C['ink'])
        self.text(d, (6, 0), '▶ CLAUDE_POP.EXE', 'paper')
        self.text(d, (180, 0), sh['title'], 'orange')
        mm, ss = divmod(song_t, 60)
        self.text(d, (W - 92, 0), f'{int(mm):02d}:{ss:05.2f}', 'yel')
        # P(DOOM) 仪表：关键帧插值，重拍后半秒内抖一抖，alarm 期间 99.9 闪烁
        ts, vs = zip(*sh['meter'])
        v = float(np.interp(t, ts, vs))
        for a in sh['hits']:
            if 0 <= t - a < .55: v += 1.5 * np.sin(t * 40)
        alarm = sh['alarm'] and sh['alarm'][0] <= t < sh['alarm'][1]
        if alarm: v = 99.9 if int(t * 12) % 2 == 0 else 61
        bx, by0, by1 = 610, 46, 246
        self.text(d, (W - 66, 22), 'P(DOOM)', 'paper')
        d.rectangle((bx - 2, by0 - 2, bx + 14, by1 + 2), outline=C['paper'], fill=C['ink'])
        top = by1 - int((by1 - by0) * min(v, 100) / 100)
        col = 'orange' if v < 90 else ('yel' if int(t * 12) % 2 else 'orange')
        d.rectangle((bx + 1, top, bx + 11, by1 - 1), fill=C[col])
        for yy in range(by0, by1, 20): d.line((bx + 12, yy, bx + 14, yy), fill=C['paper'])
        self.text(d, (W - 62, by1 + 4), f'{v:4.1f}%', 'yel' if v > 90 else 'paper')
        if alarm and int(t * 8) % 2 == 0:
            d.rectangle((W - 118, by1 + 26, W - 8, by1 + 44), fill=C['yel'])
            self.text(d, (W - 112, by1 + 27), '!! WARNING !!', 'ink')
        # 对白框（盖住 Seedance 水印）
        x0, y0, x1, y1 = 14, 290, W - 14, 352
        d.rectangle((x0, y0, x1, y1), fill=C['blue'])
        d.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=C['paper'])
        d.rectangle((x0 + 4, y0 + 4, x1 - 4, y1 - 4), outline=C['paper'])
        d.rectangle((x0 + 10, y0 - 10, x0 + 82, y0 + 6), fill=C['ink'], outline=C['orange'])
        self.text(d, (x0 + 14, y0 - 10), 'クロード', 'orange')
        for a, b, words, jp in sh['lines']:
            if not (a <= t < b): continue
            x = x0 + 14
            for i, (w, ws) in enumerate(words):
                nxt = words[i + 1][1] if i + 1 < len(words) else ws + .5
                c = 'dim' if t < ws else ('yel' if t < nxt else 'paper')
                self.text(d, (x, y0 + 12), w, c)
                x += int(self.f16.getlength(w + ' '))
            # 日文副行：唱完第一个词后逐字打出
            n = int(max(0, t - words[0][1]) * 14)
            self.text(d, (x0 + 14, y0 + 32), jp[:n], 'orange')
            if n >= len(jp) and int(t * 4) % 2 == 0:
                self.text(d, (x1 - 24, y1 - 22), '▼', 'yel')
        # 冲击大字
        for s, a, b, side in sh['slams']:
            if not (a <= t < b): continue
            k = (t - a) * FPS
            scale = 6 if k < 2 else (5 if k < 4 else 4)
            w = int(self.f16.getlength(s)) * scale
            x = {'left': 20, 'right': W - 78 - w, 'center': (W - 56 - w) // 2}[side]
            y = {'left': 60, 'right': 100, 'center': 22}[side] + (int(3 * np.sin(k * 2.2)) if k < 8 else 0)  # 居中字贴顶栏，别压人物的头
            im = self.big(im, s, x, y, scale, 'yel' if int(k / 3) % 2 == 0 else 'paper')
        return np.array(im)


def upscale(f, scan=True):
    """最近邻放大 S 倍；scan=True 时每个原像素的最后一行压暗，模拟 CRT 扫描线。"""
    up = f.repeat(S, 0).repeat(S, 1)
    if scan:
        up[S - 1::S] = (up[S - 1::S].astype(np.uint16) * 7 // 10).astype(np.uint8)
    return up
