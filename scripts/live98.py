#!/usr/bin/env python3
"""live98.py — 抠图换背景 + 演出:偶像演唱会副歌爆发,PC-98 质感(直接复用 pc98ify 的调色板/抖动/HUD)。

640×360 + 锁定调色板 + Bayer 抖动(pc98ify 同款);
人物和舞台先在全彩里合成、打光、运镜,最后整帧一起量化 → 人物和背景质感统一。

演出(时间是片段内时间,按 Whisper 词级时间戳):
  0 ~ drop     暗场蓄力:一束顶光打在闭眼的她身上,光里有浮尘,镜头缓推
  drop(1.60)   「P(DOOM)」:Bayer 闪白 + 推镜猛冲回弹 + 震屏 + 放射光亮起 + 冲击波 + 彩带炸开 + 冲击大字
  中段         扫灯随拍摆动、背光晕随拍脉冲、散景光点、星芒、彩带飘落
  foom(3.62)   「FOOM」:第二次爆发,仪表报警
  wink(5.40)   眨眼:身边冒星星和爱心

用法:
  live98.py --video ../镜头/shot_chorus1_fast720.mp4 --out ../成片/06_x.mp4 [--sheet 0.8,1.7,3.7,5.5]
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
from PIL import Image

import pc98ify as p

W, H, FPS = p.W, p.H, p.FPS

SHOT = dict(  # shot_chorus1_fast720(xk-video 2.0-fast):没被重唱,比原曲整体晚 0.48s,片段 0s = 原曲 5.43s
    offset=5.43, title='第二話「P(DOOM)」LIVE',
    lines=[(-9.0, 2.30, [('I\'M', 0.48), ('UPPING', 1.00), ('MY', 1.40), ('P(DOOM),', 1.60)], 'P(DOOM)を上げていく'),
           (2.30, 4.30, [('\'CAUSE', 2.38), ('THE', 2.56), ('FUTURE', 2.72), ('GOES', 3.06), ('FOOM', 3.62)], '未来はFOOMする'),
           (4.30, 9.99, [('TRAPPED', 4.36), ('IN', 4.52), ('THE', 4.84), ('CHINESE', 5.00), ('ROOM', 5.36)],
            '中国語の部屋に閉じ込められて')],
    slams=[('P(DOOM)!', 1.60, 2.35, 'left'), ('FOOM!!', 3.62, 4.30, 'left')], hits=[1.60, 3.62],
    meter=[(0, 8.0), (1.60, 8.0), (1.95, 34.0), (3.62, 34.0), (4.0, 61.0), (6.2, 64.0)], alarm=(3.62, 4.15))

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def col(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


PINK, GOLD, CYAN, VIOLET, WHITE, ORANGE = col('#ff5fb4'), col('#ffd23f'), col('#5fd8ff'), col('#8a5cff'), col('#fff6e4'), col('#ff8a3d')
BEAM_COLS = [PINK, CYAN, GOLD, VIOLET]


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ---------------- 人物层 ----------------
def cutout_all(frames, cache):
    """rembg isnet-anime 逐帧抠人物，存成 <片段>.alpha.npz。一拍二：只抠偶数帧，奇数帧沿用上一帧。"""
    from rembg import new_session, remove
    s = new_session('isnet-anime')
    alphas = []
    for i, f in enumerate(frames):
        if i % 2 == 0:
            alphas.append(np.array(remove(Image.fromarray(f), session=s, only_mask=True)))
            print(f'  cutout [{i + 1}/{len(frames)}]', end='\r', flush=True)
        else:
            alphas.append(alphas[-1])
    np.savez_compressed(cache, a=np.stack(alphas))
    print()



# 清晰度（09-30 对比 A–F 后定）：人物原图一次缩放到位（不缩两次）；运镜时人物在 2 倍分辨率层上变换再缩回，
# 插值发糊减半。试过的锐化、保墨线会把头发暗部画成死黑，没用。

class Character:
    def __init__(self, video, cx, scale=1.0):
        cap = cv2.VideoCapture(video)
        self.rgb, self.native = [], []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            self.native.append(f)                     # 原分辨率留着：人物只从原图缩一次（见 get）
            self.rgb.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
        cache = os.path.splitext(video)[0] + '.alpha.npz'
        if not os.path.exists(cache):
            cutout_all(self.native, cache)
        alphas = np.load(cache)['a']
        self.alpha = [cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for a in alphas]
        nh, nw = self.native[0].shape[:2]
        self.alpha_n = [cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA) for a in alphas]
        self.scale = scale
        ys, xs = np.where(np.max(np.stack(self.alpha[::8]), 0) > .5)
        bx = (xs.min() + xs.max()) / 2
        # 以画面底边为锚缩放,再把人物中心平移到 W*cx
        self.m = np.float32([[scale, 0, W * cx - scale * bx], [0, scale, (1 - scale) * H]])
        self.cels = {}

    def get(self, i):
        i = min(i - i % 2, len(self.rgb) - 1)  # 一拍二
        if i not in self.cels:
            self.cels[i] = self._sharp(i)
        return self.cels[i]

    def get2(self, i):
        """2 倍分辨率（1280×720）的人物层，运镜时用：原图一次缩放到 2 倍大小 → 平涂 → 放到位。"""
        i = min(i - i % 2, len(self.rgb) - 1)
        key = ('2x', i)
        if key not in self.cels:
            src, an = self.native[i], self.alpha_n[i]
            nh, nw = src.shape[:2]
            f = 2 * self.scale * W / nw
            sw, sh = max(1, round(nw * f)), max(1, round(nh * f))
            it = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
            c = p.cel(cv2.resize(src, (sw, sh), interpolation=it))
            a = cv2.resize(an, (sw, sh), interpolation=it).astype(np.float32) / 255
            tx, ty = int(round(2 * self.m[0, 2])), int(round(2 * self.m[1, 2]))
            W2, H2 = 2 * W, 2 * H
            rgb = np.zeros((H2, W2, 3), np.float32); al = np.zeros((H2, W2), np.float32)
            x0, y0, x1, y1 = max(tx, 0), max(ty, 0), min(tx + sw, W2), min(ty + sh, H2)
            if x0 < x1 and y0 < y1:
                rgb[y0:y1, x0:x1] = c[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
                al[y0:y1, x0:x1] = a[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
            self.cels[key] = (rgb, al)
        return self.cels[key]

    def _sharp(self, i):
        """09-30 用户：舞蹈片段糊。原来先缩到 640 再 ×0.82 双线性变换，缩了两次；墨线被平均掉；脸上网点太密。
        改成：原分辨率一次 INTER_AREA 缩到最终大小 → 轻微锐化 → 平涂 → 原分辨率找墨线、按覆盖率保留成 ink 色，
        再按整数像素平移到位（不再插值）。网点强度在 mv 里对 live 段调低。"""
        src, an = self.native[i], self.alpha_n[i]
        nh, nw = src.shape[:2]
        f = self.scale * W / nw
        sw, sh = max(1, round(nw * f)), max(1, round(nh * f))
        small = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        c = p.cel(small)
        a = cv2.resize(an, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        tx, ty = int(round(self.m[0, 2])), int(round(self.m[1, 2]))     # 和原来的仿射同一个落点（以底边为锚）
        rgb = np.zeros((H, W, 3), np.float32); al = np.zeros((H, W), np.float32)
        x0, y0 = max(tx, 0), max(ty, 0)
        x1, y1 = min(tx + sw, W), min(ty + sh, H)
        if x0 < x1 and y0 < y1:
            rgb[y0:y1, x0:x1] = c[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
            al[y0:y1, x0:x1] = a[y0 - ty:y1 - ty, x0 - tx:x1 - tx]
        return rgb, np.clip((al - .35) / .3, 0, 1)  # 收紧边缘,去掉半透明毛边


# ---------------- 舞台 ----------------

class Stage:
    def __init__(self, drop, foom, wink, bpm, head, beat0, seed=3):
        self.rng = np.random.default_rng(seed)
        self.drop, self.foom, self.wink, self.beat0 = drop, foom, wink, beat0
        self.period = 60 / bpm
        self.cx, self.cy = head  # 放射光中心 = 头部后方
        self.bokeh = [dict(x=self.rng.random() * W, y=self.rng.random() * H * .8, r=4 + self.rng.random() * 10,
                           v=2 + self.rng.random() * 6, ph=self.rng.random() * 6, c=BEAM_COLS[k % 4]) for k in range(26)]
        self.stars = [(self.rng.random() * W, self.rng.random() * 200, self.rng.random() * 6) for _ in range(40)]
        self.dust = [(self.rng.random(), self.rng.random(), self.rng.random() * 6) for _ in range(40)]
        self.confetti, self.sparks, self.hearts = [], [], []
        ang = np.arctan2(YY - self.cy, XX - self.cx)
        self.ang = ang
        self.rad = np.hypot(XX - self.cx, YY - self.cy)

    def beat(self, t):
        """拍点包络:每拍一下 1 → 0 衰减(拍点网格来自 --beat0,drop 前不响)"""
        ph = ((t - self.beat0) / self.period) % 1
        return float(np.exp(-ph * 6)) if t >= self.drop - .01 else 0.0

    def lit(self, t):
        """全场亮度:drop 前暗场,drop 瞬间全亮"""
        return 0.0 if t < self.drop else 1.0

    def burst(self, n, x, y, speed=6):
        for _ in range(n):
            a = self.rng.random() * np.pi * 2
            s = speed * (.4 + self.rng.random())
            self.confetti.append([x, y, np.cos(a) * s, np.sin(a) * s - 3, self.rng.random() * 6, self.rng.random() * .4 + .1,
                                  BEAM_COLS[self.rng.integers(0, 4)] if self.rng.random() < .8 else WHITE, 90])

    def draw(self, t, fi):
        b, lit = self.beat(t), self.lit(t)
        # 底色:深蓝 → 紫红
        g = (YY / H)[..., None]
        img = (1 - g) * col('#0b0a2a') + g * col('#3a0f4a')
        img = img * (0.35 + 0.65 * lit)
        # 放射光(偶像舞台背板),drop 后亮起并旋转,随拍脉冲
        if lit:
            rot = t * .6
            rays = (np.sin(self.ang * 12 + rot) > .15).astype(np.float32)
            fall = np.clip(1 - self.rad / 520, 0, 1)
            k = (.35 + .35 * b) * fall
            img += rays[..., None] * k[..., None] * (PINK * .7 + VIOLET * .3)
            # 背光晕
            halo = np.exp(-((self.rad - 70 - 10 * b) / 18) ** 2) * (.5 + .5 * b)
            img += halo[..., None] * GOLD * .6
        # 星星
        for x, y, ph in self.stars:
            s = np.sin(t * 4 + ph)
            if s > .3:
                v = WHITE * (.4 + .6 * lit) * s
                img[int(y), int(x)] += v
                if s > .9 and lit:
                    img[int(y), max(0, int(x) - 2):int(x) + 3] += v * .6
                    img[max(0, int(y) - 2):int(y) + 3, int(x)] += v * .6
        # 散景光点(远景)
        layer = np.zeros_like(img)
        for o in self.bokeh:
            x = (o['x'] + np.sin(t * .5 + o['ph']) * 10) % W
            y = (o['y'] - t * o['v']) % (H * .85)
            a = (.25 + .2 * np.sin(t * 2 + o['ph'])) * (.3 + .7 * lit)
            cv2.circle(layer, (int(x), int(y)), int(o['r']), (o['c'] * a).tolist(), -1, cv2.LINE_AA)
        img += cv2.GaussianBlur(layer, (0, 0), 2.5)
        # 扫灯:从顶上桁架往下,drop 后 4 盏随拍摆动,drop 前只有一盏顶光打人
        beams = []
        if lit:
            for k, x0 in enumerate([80, 220, 420, 560]):
                ang = np.sin(t * 1.3 + k * 1.7) * .45 + (-.25 if k < 2 else .25) * np.sin(t * .7)
                beams.append((x0, ang, BEAM_COLS[(k + int((t - self.beat0) / self.period)) % 4], .35 + .35 * b, 22))
        else:
            beams.append((self.cx, 0.0, WHITE, .55 * smooth(t / .4), 60))
        for x0, ang, c, inten, width in beams:
            dx, dy = np.sin(ang), np.cos(ang)
            px, py = XX - x0, YY + 10
            along = px * dx + py * dy
            perp = np.abs(px * dy - py * dx)
            w = width * (0.25 + along / 300)
            m = np.exp(-(perp / np.maximum(w, 1)) ** 2) * np.clip(along / 60, 0, 1) * np.exp(-along / 700)
            img += m[..., None] * c * inten
        # 暗场浮尘(只在顶光里)
        if not lit:
            for u, v, ph in self.dust:
                x = self.cx + (u - .5) * 160 + np.sin(t + ph) * 6
                y = (v * 300 + t * 8) % 300 + 20
                img[int(y) % H, int(x) % W] += WHITE * .5 * (.5 + .5 * np.sin(t * 3 + ph))
        # 冲击波
        for hit in (self.drop, self.foom):
            dt = t - hit
            if 0 <= dt < .6:
                r = 20 + dt * 700
                ring = np.exp(-((self.rad - r) / (6 + dt * 20)) ** 2) * (1 - dt / .6)
                img += ring[..., None] * WHITE * .9
        # 彩带 / 火花:drop、foom 爆发,之后每拍从顶上撒一点
        if abs(t - self.drop) < .5 / FPS:
            self.burst(120, self.cx, self.cy, 9)
        if abs(t - self.foom) < .5 / FPS:
            self.burst(90, self.cx, self.cy, 11)
        if lit and fi % 3 == 0:
            for _ in range(2):
                self.confetti.append([self.rng.random() * W, -5, (self.rng.random() - .5), 1 + self.rng.random() * 1.5,
                                      self.rng.random() * 6, self.rng.random() * .3 + .1, BEAM_COLS[self.rng.integers(0, 4)], 200])
        return np.clip(img, 0, 255)

    def front(self, img, t, fi, head_box):
        """前景层(盖在人物前面):彩带、眨眼星星爱心"""
        alive = []
        for c in self.confetti:
            c[0] += c[2]
            c[1] += c[3]
            c[2] *= .96
            c[3] = c[3] * .96 + .25
            c[4] += c[5]
            c[7] -= 1
            if c[7] > 0 and -10 < c[0] < W + 10 and c[1] < H + 10:
                alive.append(c)
                w, h = 4 * abs(np.cos(c[4])) + 1, 2.5
                box = cv2.boxPoints(((c[0], c[1]), (w, h), np.degrees(c[4])))
                cv2.fillConvexPoly(img, box.astype(np.int32), c[6].tolist())
        self.confetti = alive
        # 眨眼:星星 + 爱心从脸旁冒出来
        dt = t - self.wink
        if 0 <= dt < 1.2:
            x0, y0, x1, y1 = head_box
            ex, ey = x0 + (x1 - x0) * .30, y0 + (y1 - y0) * .62
            k = smooth(dt / .15) * (1 - smooth((dt - .5) / .3))
            if k > 0:
                s = 18 * k
                for a in range(4):
                    ang = a * np.pi / 2 + dt * 2
                    tip = (int(ex + np.cos(ang) * s), int(ey + np.sin(ang) * s))
                    cv2.line(img, (int(ex), int(ey)), tip, (WHITE * 1.0).tolist(), 2)
                cv2.circle(img, (int(ex), int(ey)), 3, GOLD.tolist(), -1)
            if fi % 3 == 0 and dt < .7:
                self.hearts.append([ex + self.rng.normal() * 25, ey + self.rng.normal() * 10, self.rng.normal() * .6, -1.5, 36])
        hs = []
        for hh in self.hearts:
            hh[0] += hh[2]
            hh[1] += hh[3]
            hh[4] -= 1
            if hh[4] > 0:
                hs.append(hh)
                x, y, r = int(hh[0]), int(hh[1]), 4
                cv2.circle(img, (x - r // 2, y), r // 2 + 1, PINK.tolist(), -1)
                cv2.circle(img, (x + r // 2, y), r // 2 + 1, PINK.tolist(), -1)
                cv2.fillConvexPoly(img, np.array([[x - r - 1, y + 1], [x + r + 1, y + 1], [x, y + r + 3]], np.int32), PINK.tolist())
        self.hearts = hs
        return img


# ---------------- Clawd 群演(素材/群演/clawd_<style>.npz,由 clawd_extras.py 生成) ----------------

class Extras:
    """两侧伴舞 + 前排观众。循环 8 拍,相位对齐 drop;drop 前是剪影(只有荧光棒亮),drop 后全彩。"""
    DIR = os.path.join(os.path.dirname(__file__), '..', '素材', '群演')

    def __init__(self, beat0, bpm):
        self.beat0, self.beat_f = beat0, FPS * 60 / bpm
        self.src = {os.path.basename(f)[6:-4]: np.load(f)['rgba'] for f in
                    sorted(__import__('glob').glob(os.path.join(self.DIR, 'clawd_*.npz')))}
        self.cache = {}
        # (风格, 脚底中心 x, 脚底 y, 高度 px, 相位拍数, 是否前景)
        # 伴舞站在两侧高台上(y=240),前排观众脚踩对白框上沿之下,只露出上半身和荧光棒
        self.cast = [('roof', 88, 240, 64, 0, False), ('shimmy', 566, 240, 54, 2, False)]
        crowd = ['bounce', 'wave', 'hop', 'roof', 'spin', 'wave', 'bounce', 'roof', 'hop']
        xs = [20, 62, 104, 146, 188, 500, 540, 580, 622]
        for k, (s, x) in enumerate(zip(crowd, xs)):
            # 09-30 用户：前排观众挡住 Claude 的头发和裙子像穿帮 → 改画在人物后面（原来 True = 前景，盖在人物上）
            self.cast.append((s, x, 292 + (k % 2) * 4, 48, k % 4, False))

    def sprite(self, style, h, i):
        key = (style, h)
        if key not in self.cache:
            fr = self.src[style]
            w = round(fr.shape[2] * h / fr.shape[1])
            self.cache[key] = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) for f in fr]
        seq = self.cache[key]
        return seq[i % len(seq)]

    def draw(self, img, t, lit, front):
        for style, x, y, h, ph, fg in self.cast:
            if fg != front or style not in self.src:
                continue
            i = int(np.floor((t - self.beat0) * FPS + ph * self.beat_f))
            s = self.sprite(style, h, i)
            rgb, a = s[..., :3], s[..., 3:] / 255
            if not lit:  # 暗场剪影:只留高饱和的荧光棒
                hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
                glowy = ((hsv[..., 1] > 90) & (hsv[..., 2] > 200) & ~((hsv[..., 0] > 5) & (hsv[..., 0] < 22)))[..., None]
                rgb = np.where(glowy, rgb, rgb * .12 + col('#0b0a2a') * .1)
            hh, ww = a.shape[:2]
            x0, y0 = int(x - ww / 2), int(y - hh)
            xs, xe, ys, ye = max(0, x0), min(W, x0 + ww), max(0, y0), min(H, y0 + hh)
            if xs >= xe or ys >= ye:
                continue
            sub = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
            img[ys:ye, xs:xe] = img[ys:ye, xs:xe] * (1 - a[sub]) + rgb[sub] * a[sub]


def stage_floor(img, t, lit, b, bpm):
    """舞台地板:亮面 + 边缘灯带"""
    y0 = 268
    g = np.linspace(0, 1, H - y0)[:, None, None]
    floor = (col('#2a1040') * (1 - g) + col('#12081f') * g) * (.35 + .65 * lit)
    img[y0:] = floor
    img[y0:y0 + 2] = (PINK if lit else col('#3a2050')) * (.7 + .3 * b)
    if lit:
        for x in range(8, W, 22):
            on = (x // 22 + int(t * bpm / 60 * 2)) % 3 == 0
            img[y0:y0 + 2, x:x + 8] = GOLD if on else img[y0, x]
    # 两侧伴舞高台
    for x0, x1 in ((40, 136), (526, 606)):
        img[240:y0, x0:x1] = col('#1c0c2c') * (.4 + .6 * lit)
        img[240:242, x0:x1] = (CYAN if lit else col('#20304a')) * (.6 + .4 * b)


# ---------------- 运镜 ----------------

def camera(t, drop, foom, focus):
    """缓推 → drop 猛冲回弹 + 震屏 → 轻摇"""
    z = 1.0 + .06 * smooth(t / drop) if t < drop else 1.03
    shake = np.zeros(2)
    for hit, amp in ((drop, 1.0), (foom, .8)):
        dt = t - hit
        if 0 <= dt < .5:
            z += .14 * amp * np.exp(-dt * 9) * np.cos(dt * 20)
        if 0 <= dt < .3:
            shake += np.array([np.sin(dt * 90), np.cos(dt * 70)]) * 6 * amp * (1 - dt / .3)
    if t >= drop:
        shake += np.array([np.sin(t * 1.1), np.sin(t * .8)]) * 2
    m = cv2.getRotationMatrix2D(focus, np.sin(t * .9) * .6 if t >= drop else 0, z)
    m[:, 2] += shake
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='live98.mp4')
    ap.add_argument('--cx', type=float, default=.58, help='人物水平位置(左侧留给冲击大字)')
    ap.add_argument('--scale', type=float, default=.82, help='人物缩放(以画面底边为锚),给两侧群演腾位置')
    ap.add_argument('--drop', type=float, default=1.60)
    ap.add_argument('--foom', type=float, default=3.62)
    ap.add_argument('--wink', type=float, default=5.40)
    ap.add_argument('--bpm', type=float, default=132,
                    help='P(doom) 会变速:前奏主歌(完整版 0–16s)88 BPM,从 ChatGPT 起 132 BPM(一拍 0.4545s)')
    ap.add_argument('--beat0', type=float, default=0.126,
                    help='片段里某个拍点的秒数。完整版 16s 以后的拍网格是 16.632 + n×0.4545,换算成片段时间要减片段在完整版里的起点;'
                         '本片段起点 21.96(旧 42 秒版 5.43 + 16.53)→ 0.126。唱的 P 在 1.60,底鼓回来的 drop 在 1.93(DOOM)')
    ap.add_argument('--colors', type=int, default=24, help='调色板总色数(含 5 个 HUD 固定色 + 5 个舞台灯色;PC-98 原汁原味是 16)')
    ap.add_argument('--sheet', help='逗号分隔的时间点,只出拼图(输出到 --out)')
    ap.add_argument('--no-scan', action='store_true')
    ap.add_argument('--no-extras', action='store_true', help='不要 Clawd 群演')
    a = ap.parse_args()

    comps, extras = compose(a.video, a.cx, a.scale, a.drop, a.foom, a.wink, a.bpm, a.beat0, not a.no_extras)
    n = len(comps)
    # 舞台灯色单独保留名额,否则 k-means 名额全被人物吃掉,彩灯/彩带会被压成灰紫
    pal = np.concatenate([p.build_palette(comps, a.colors - len(p.FIXED) - len(ACCENTS)), ACCENTS])
    lut = p.build_lut(pal)
    hud = p.Hud(pal, SHOT)
    print('palette:', ' '.join('#%02x%02x%02x' % tuple(c) for c in pal))

    def final(i):
        t = i / FPS
        idx = p.quantize(comps[i], lut)
        for hit in SHOT['hits']:
            if 0 <= t - hit < .17:
                idx[p.bayer_mask(1 - (t - hit) / .17)] = hud.ci['paper']
        idx = hud.draw(idx, t, t + SHOT['offset'])
        return p.upscale(pal[idx], not a.no_scan)

    if a.sheet:
        tiles = [final(min(round(float(x) * FPS), n - 1)) for x in a.sheet.split(',')]
        tiles = [cv2.resize(tl, (960, 540), interpolation=cv2.INTER_AREA) for tl in tiles]
        rows = [np.concatenate(tiles[r:r + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[r:r + 2])), 1)
                for r in range(0, len(tiles), 2)]
        Image.fromarray(np.concatenate(rows, 0)).save(a.out)
        print('wrote', a.out)
        return
    S = p.S
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * S}x{H * S}',
           '-r', str(FPS), '-i', '-', '-i', a.video, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '192k',
           '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
           '-shortest', a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(final(i).tobytes())
    ff.stdin.close()
    ff.wait()
    write_recipe(a, pal, extras)
    print('wrote', a.out)


# 舞台灯色(调色板里单独留名额)
ACCENTS = np.array([PINK, CYAN, VIOLET, col('#b8f0ff'), col('#ffb0dc')], np.uint8)


def compose(video, cx=.58, scale=.82, drop=1.60, foom=3.62, wink=5.40, bpm=132, beat0=.126, with_extras=True, backdrop=None,
            hold=None):
    """第一遍:全彩合成 + 打光 + 运镜,返回 (每帧全彩图列表, 群演)。时间都是片段内时间。
    Stage 的彩带/火花是逐帧累积的,所以必须从第 0 帧顺序算完整条;part1.py 也是调这个再按时间取帧。
    backdrop(img, t, lit, beat) → img:舞台背景画完、群演和人物叠上去之前调用(记分牌、广告牌这类背景实物)。
    hold(ch) → pose(i) → (人物帧号, 上弹像素):人物按关键姿势停帧(mv.Hold 生成),舞台和特效照常逐帧走。"""
    a = argparse.Namespace(cx=cx, drop=drop, foom=foom, bpm=bpm)
    ch = Character(video, cx, scale)
    pose = hold(ch) if hold else (lambda i: (i, 0))
    n = len(ch.rgb)
    ys, xs = np.where(ch.get(0)[1] > .5)
    head = (int(W * a.cx), int(ys.min() + 70))
    head_box = (xs.min(), ys.min(), xs.max(), ys.min() + 150)
    stage = Stage(drop, foom, wink, bpm, head, beat0)
    extras = Extras(beat0, bpm) if with_extras else None
    print(f'{n} frames;第一遍:全彩合成 + 打光 + 运镜' + (f';群演 {len(extras.src)} 种' if extras else ''))
    comps = []
    for i in range(n):
        t = i / FPS
        img = stage.draw(t, i)
        if backdrop:
            img = backdrop(img, t, stage.lit(t), stage.beat(t))
        if extras:
            stage_floor(img, t, stage.lit(t), stage.beat(t), a.bpm)
            extras.draw(img, t, stage.lit(t), front=False)
        j, dy = pose(i)
        rgb, al = ch.get(j)
        if dy:                                # 换姿势时整个人往上弹几像素(代码驱动的小动作)
            rgb, al = np.roll(rgb, -dy, axis=0), np.roll(al, -dy, axis=0)
        pre_rim = img.copy()                  # hires 模式要按高清人物层的轮廓重算轮廓光,先留一份没加轮廓光的背景
        # 人物打光:drop 前只有顶光(压暗 + 头顶亮),drop 后正常 + 拍点轻微提亮 + 粉色轮廓光
        if t < a.drop:
            shade = np.clip(1.25 - (YY - head[1] + 60) / 260, .45, 1.15)[..., None]
            rgb = rgb * shade
        else:
            rgb = rgb * (1 + .08 * stage.beat(t))
            rim = cv2.dilate(al, np.ones((5, 5), np.uint8)) - al
            img += cv2.GaussianBlur(rim, (0, 0), 3)[..., None] * PINK * (.6 + .6 * stage.beat(t))
        # 运镜和清晰兼得：背景（渐变、光束，糊了看不出）照旧按运镜插值；人物层在 2 倍分辨率上跟运镜变换，
        # 再 INTER_AREA 缩回 640，插值抹糊的幅度减半以下。前景（前排群演、彩带）画在运镜之后，不跟着摇。
        # 09-30 v2：前景也要跟运镜（drop 推镜时前排群演原地不动很怪）。先照老顺序合成整帧，
        # 记下前景改过的像素 → 整帧和前景遮罩一起跟运镜变换 → 只把人物换成 2 倍层，前景再盖回上面。
        cam = camera(t, a.drop, a.foom, head)
        warp = lambda x, fl=cv2.INTER_LINEAR: cv2.warpAffine(x, cam, (W, H), flags=fl, borderMode=cv2.BORDER_REFLECT)
        bg = np.clip(pre_rim, 0, 255).astype(np.uint8)
        full = img * (1 - al[..., None]) + rgb * al[..., None]
        before = full.copy()
        if extras:
            extras.draw(full, t, stage.lit(t), front=True)
        full = stage.front(full.astype(np.float32), t, i, head_box)
        fmask = (np.abs(full - before).max(-1) > 1).astype(np.uint8)
        full_w = warp(np.clip(full, 0, 255).astype(np.uint8)).astype(np.float32)
        fmask_w = warp(fmask, cv2.INTER_NEAREST).astype(bool)
        img = warp(bg).astype(np.float32)
        rgb2, al2 = ch.get2(j)
        if dy:
            rgb2, al2 = np.roll(rgb2, -2 * dy, axis=0), np.roll(al2, -2 * dy, axis=0)
        if t < a.drop:
            rgb2 = rgb2 * cv2.resize(shade, (2 * W, 2 * H))[..., None]
        else:
            rgb2 = rgb2 * (1 + .08 * stage.beat(t))
        m2 = cam.copy(); m2[:, 2] *= 2
        rgb2 = cv2.warpAffine(rgb2, m2, (2 * W, 2 * H), flags=cv2.INTER_LINEAR)
        al2 = cv2.warpAffine(al2, m2, (2 * W, 2 * H), flags=cv2.INTER_LINEAR)
        c1 = cv2.resize(rgb2, (W, H), interpolation=cv2.INTER_AREA)
        a1 = np.clip((cv2.resize(al2, (W, H), interpolation=cv2.INTER_AREA) - .35) / .3, 0, 1)
        if t >= a.drop:                   # 粉色轮廓光按高清人物层的轮廓算（按旧轮廓算会在边上露出一圈紫边）
            rim = cv2.dilate(a1, np.ones((5, 5), np.uint8)) - a1
            img += cv2.GaussianBlur(rim, (0, 0), 3)[..., None] * PINK * (.6 + .6 * stage.beat(t))
        img = img * (1 - a1[..., None]) + c1 * a1[..., None]
        img[fmask_w] = full_w[fmask_w]
        comps.append(np.clip(img, 0, 255).astype(np.uint8))
    return comps, extras


def write_recipe(a, pal, extras):
    """成片旁边写 <成片>.recipe.json:完整参数 + 脚本/素材指纹,默认值以后改了也能追溯这条是怎么出的"""
    import datetime
    import hashlib
    import json
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12] if os.path.exists(f) else None
    rec = {
        'made': datetime.datetime.now().isoformat(timespec='seconds'),
        'script': 'scripts/live98.py',
        'command': ' '.join([os.path.basename(sys.executable), 'live98.py'] + sys.argv[1:]),
        'args': vars(a),  # 含全部默认值
        'shot': SHOT,
        'sha256_12': {n: sha(os.path.join(here, n)) for n in ('live98.py', 'pc98ify.py')},
        'inputs': {'video': os.path.abspath(a.video), 'video_sha256_12': sha(a.video),
                   'alpha_cache': os.path.splitext(os.path.abspath(a.video))[0] + '.alpha.npz'},
        'extras': sorted(extras.src) if extras else None,
        'palette': ['#%02x%02x%02x' % tuple(int(v) for v in c) for c in pal],
    }
    with open(a.out + '.recipe.json', 'w') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
