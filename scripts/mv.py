#!/usr/bin/env python3
"""mv: 按歌曲时间轴出片的公共引擎（part1 / part2 / 以后整首歌共用）。每部分只写一个配置模块，然后 mv.run(cfg)。

每帧只由歌曲时间 t 决定：frame(t) = 该时刻所在段落的画面（锁定调色板 + Bayer 抖动）→ 段落间转场 → 故障 → 重拍 → HUD。
段落有这些来源（TIMELINE 第 4 项）：
  ('clip', 片段名)             Seedance 片段：cel 平涂 → 本段锁定 16 色调色板
  ('code', 模块, 镜头名)        纯代码镜头（接口见 part1_lib.py）
  ('live', 片段名)             live98 舞台（抠图换背景 + 群演），启动时整条预合成，参数在 cfg.LIVE[片段名]
  ('clipfx', 片段名, fx)       片段再加一层特效：fx(rgb_at, t, a, b) → 全彩帧（例：part1 的瞳孔推进）
  ('livefx', 片段名, fx)       live 舞台再叠一层前景（人物前面的东西，例：part3 的回形针堆、点赞雨）

配置模块里要有的名字（没有的按空处理）：
  START, END           本部分的歌曲时间范围
  CLIPS                {片段名: (路径, clip0)}  clip0 = 片段第 0 秒对应的歌曲时间
  LIVE                 {片段名: live98.compose 的参数（drop/foom/wink 和 backdrop 都按歌曲时间写，这里换算成片段时间）}
  TIMELINE             [(开始, 结束, 段名, 来源)]
  WARP / CAMS / DISSOLVE / ACCENTS   按段名：变速 / 数码推拉 / 开头溶解秒数 / 追加调色板颜色
  GLITCH, HITS         [(a, b)] CRT 故障区间；[t] 闪白震屏
  HOLD                 {片段名: spec}  人物按关键姿势停帧（第三阶段），见 Hold 类
  TILT                 [(a, b, 角度)]  整帧歪斜（HUD 一起歪）；(a, b, 角度0, 角度1) 在区间里平滑转
  OVERLAY              overlay(img, t) → img，HUD 之后、TILT 之前画（演职员表）
  PALETTE_SPAN         {段名: (a, b)}  调色板按这个区间抽样（默认就是段落本身；接缝段写成和上一部分一样）
  WARP 对 live 片段也生效（按片段名）：歌曲时间 → 片段时间
  CHAPTERS, LINES, SLAMS, HUD_OFF, TOPBAR_OFF   HUD：顶栏章节、对白框逐词、冲击大字、不画对白框 / 顶栏的区间
                       SLAMS 老写法 (字, a, b, 位置) 不变；第五项给 dict 就走 slam2（多种字效，见 slam2）
"""
import argparse
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

import part1_lib as L
import pc98ify as P

W, H, FPS = L.W, L.H, L.FPS

FIX = {k: tuple(int(x) for x in v) for k, v in P.FIXED.items()}
FIX['dim'] = (110, 120, 170)


def cfg_get(cfg, name, default):
    return getattr(cfg, name, default)


# ---------------- 段落来源 ----------------
LIVE_SPREAD = 36.0   # 舞蹈（live）段的网点强度。09-30 试过调到 20，用户看对比觉得丢 PC-98 质感，保持和其他段一样


class Quantized:
    """全彩帧来源 → 本段锁定调色板（抽样 k-means）→ 按时间出 RGB。"""
    K = 16

    def __init__(self, rgb_at, t0, t1, accents=None, spread=36.0):
        self.rgb_at, self.t0, self.t1, self.spread = rgb_at, t0, t1, spread
        samples = [P.cel(rgb_at(t)) for t in np.linspace(t0, t1 - 1e-3, 14)]
        k = self.K - len(P.FIXED) - (0 if accents is None else len(accents))
        pal = P.build_palette(samples, k)
        self.pal = pal if accents is None else np.concatenate([pal, accents])
        self.lut = P.build_lut(self.pal)

    def at(self, t):
        return self.pal[P.quantize(P.cel(self.rgb_at(t)), self.lut, self.spread)]


class Hold:
    """人物按关键姿势停帧（第三阶段：古早动画感）。每 every 拍一个姿势，姿势 = 该拍点附近动作最静的一帧
    （跳到最高点、手伸到底这类转折点，原画师会画的 key pose），停到下一个姿势再硬切。
    spec（配置 HOLD[片段名]）：
      every   几拍换一个姿势（默认 1；132 BPM 一拍 ≈ 11 帧）
      phase   拍网格偏移（拍，默认 0）
      search  在拍点前后多大范围里挑姿势（占一个姿势时长的比例，默认 .45）
      keys    [(歌曲时间, 片段时间)] 手写姿势，从该歌曲时间起显示片段该时刻的帧；有 keys 时拍网格不管用
      bounce  换姿势时整体上弹几像素（默认 3，0 关）
    每帧仍只由 t 决定：motion 是预先算好的每帧运动量。"""

    def __init__(self, spec, song2clip, motion):
        self.s = dict(dict(every=1, phase=0, search=.45, keys=None, bounce=3), **spec)
        self.song2clip, self.motion = song2clip, motion
        self.per = L.BEAT132 * self.s['every']

    def window(self, t):
        """(姿势开始的歌曲时间, 挑帧中心的片段时间)"""
        if self.s['keys']:
            ks = [k for k in self.s['keys'] if k[0] <= t] or self.s['keys'][:1]
            return ks[-1][0], ks[-1][1]
        b0 = L.B132_0 + self.s['phase'] * L.BEAT132
        k = np.floor((t - b0) / self.per)
        tc = b0 + k * self.per
        return tc, self.song2clip(tc)

    def index(self, t):
        tc, c = self.window(t)
        n = len(self.motion)
        if self.s['keys']:
            return int(np.clip(round(c * FPS), 0, n - 1)), t - tc
        r = self.s['search'] * self.per
        lo, hi = int(np.clip((c - r) * FPS, 0, n - 1)), int(np.clip((c + r) * FPS, 0, n - 1))
        return lo + int(np.argmin(self.motion[lo:hi + 1])), t - tc

    def bump(self, dt):
        """换姿势后的上弹：0 → 最高 → 落地，约 0.15 秒。"""
        return int(round(self.s['bounce'] * np.sin(np.pi * min(dt / .15, 1)))) if dt < .15 else 0


def motion_of(frames, alphas=None):
    """每帧运动量（到下一帧的平均灰度差，缩到 96 宽；有 alpha 时只看人物）。"""
    def small(k):
        g = cv2.cvtColor(np.asarray(frames[k], np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        if alphas is not None: g = g * alphas[k]
        return cv2.resize(g, (96, 54), interpolation=cv2.INTER_AREA)
    sm = [small(k) for k in range(len(frames))]
    m = [float(np.abs(sm[k + 1] - sm[k]).mean()) for k in range(len(sm) - 1)]
    return np.array(m + m[-1:])


class ClipSource:
    def __init__(self, path, clip0, warp=None, hold=None):
        self.clip0 = clip0
        self.warp = warp or (lambda t: t - clip0)
        self.frames = P.load_frames(str(path))
        self.hold = Hold(hold, self.warp, motion_of(self.frames)) if hold else None

    def rgb(self, t):
        """歌曲时间 → 片段帧（一拍二；超出片段范围钳到首尾帧）。有 HOLD 时按关键姿势停帧 + 换姿势时上弹。"""
        if self.hold:
            i, dt = self.hold.index(t)
            dy = self.hold.bump(dt)
            f = self.frames[i]
            return np.concatenate([f[dy:], np.repeat(f[-1:], dy, 0)]) if dy else f
        i = int(np.clip(round(self.warp(t) * FPS), 0, len(self.frames) - 1))
        return self.frames[i - i % 2]


class LiveSource:
    """live98 舞台：整条预合成后按时间取帧。kw 里的 drop/foom 是歌曲时间，这里换算成片段内时间。"""
    def __init__(self, path, clip0, kw, hold=None, warp=None):
        import live98
        self.clip0 = c = clip0
        self.warp = warp                                  # 歌曲时间 → 片段时间（part4 谢幕按四声 oh 卡动作）；没给就原速
        kw = dict(kw)
        if hold:                                          # 人物停帧，舞台照常逐帧走
            def mk(ch, c=c, spec=hold):
                h = Hold(spec, lambda t: t - c, motion_of(ch.rgb, ch.alpha))
                def pose(i):
                    j, dt = h.index(i / FPS + c)
                    return j, h.bump(dt)
                return pose
            kw['hold'] = mk
        self.accents = kw.pop('accents', live98.ACCENTS)      # 换了背景（不是偶像舞台）时可以不要舞台灯色
        for k in ('drop', 'foom', 'wink'):
            if k in kw and kw[k] < 900: kw[k] -= c       # 999 表示关掉，不换算（第二部分写的 99 换算后也超出片段，照样等于关掉）
        kw.setdefault('beat0', (L.B132_0 - c) % L.BEAT132)
        if kw.get('backdrop'):                            # backdrop 在配置里按歌曲时间写
            kw['backdrop'] = lambda img, t, lit, beat, f=kw['backdrop']: f(img, t + c, lit, beat)
        self.comps, _ = live98.compose(str(path), **kw)

    def rgb(self, t):
        ct = self.warp(t) if self.warp else t - self.clip0
        i = int(np.clip(round(ct * FPS), 0, len(self.comps) - 1))
        return self.comps[i]


def load_code(mod, name):
    """子镜头还没写好 / 运行出错时用占位画面，不挡整条出片（出错会打印一次）。"""
    def placeholder(t):
        img = np.full((H, W, 3), 40, np.uint8)
        return L.draw_text(img, (40, 140), f'{name} TODO', (255, 210, 63), scale=3)
    try:
        fn = {n: f for n, a, b, f in __import__(mod).SCENES}[name]
    except Exception as e:
        print(f'  [占位] {mod}.{name}: {e!r}')
        return placeholder
    warned = []
    def safe(t):
        try:
            return fn(t)
        except Exception as e:
            if not warned: print(f'  [占位] {mod}.{name} 运行出错: {e!r}'); warned.append(1)
            return placeholder(t)
    return safe


class Timeline:
    def __init__(self, cfg, lo, hi):
        self.cfg = cfg
        clips_cfg, warp = cfg_get(cfg, 'CLIPS', {}), cfg_get(cfg, 'WARP', {})
        accents, cams = cfg_get(cfg, 'ACCENTS', {}), cfg_get(cfg, 'CAMS', {})
        self.dissolve = cfg_get(cfg, 'DISSOLVE', {})
        self.segs = []
        clips, lives = {}, {}

        holds = cfg_get(cfg, 'HOLD', {})

        def clip(n):
            if n not in clips: clips[n] = ClipSource(*clips_cfg[n], warp.get(n), holds.get(n))
            return clips[n]

        def live(n):          # 同一条 live 片段可以拆成几段各自锁调色板，只合成一次
            if n not in lives: lives[n] = LiveSource(*clips_cfg[n], cfg_get(cfg, 'LIVE', {}).get(n, {}), holds.get(n), warp.get(n))
            return lives[n]

        for a, b, name, src in cfg.TIMELINE:
            if b <= lo - .5 or a >= hi + .5:
                self.segs.append((a, b, name, None)); continue
            print(f'准备 {name} ({a:.2f}–{b:.2f})')
            span = cfg_get(cfg, 'PALETTE_SPAN', {}).get(name, (a, b))    # 调色板抽样区间（接缝段要和上一部分同一个调色板）
            kind = src[0]
            if kind == 'clip':
                c = clip(src[1])
                q = Quantized(c.rgb, *span, accents.get(name))
            elif kind == 'live':
                c = live(src[1])
                acc = c.accents if name not in accents else np.concatenate([c.accents, accents[name]])
                q = Quantized(c.rgb, *span, acc, LIVE_SPREAD)
            elif kind == 'code':
                fn = load_code(src[1], src[2])
                if name in warp: fn = (lambda t, f=fn, w=warp[name]: f(w(t)))
                q = Quantized(fn, *span, accents.get(name))
            elif kind == 'livefx':                     # live 舞台再叠一层前景特效（人物前面的东西：回形针堆、点赞雨）
                c, fx = live(src[1]), src[2]
                acc = c.accents if name not in accents else np.concatenate([c.accents, accents[name]])
                q = Quantized(lambda t, c=c, fx=fx, a=a, b=b: fx(c.rgb, t, a, b), *span, acc, LIVE_SPREAD)
            elif kind == 'clipfx':
                c, fx = clip(src[1]), src[2]
                q = Quantized(lambda t, c=c, fx=fx, a=a, b=b: fx(c.rgb, t, a, b), *span, accents.get(name))
            else:
                raise ValueError(f'未知来源 {src!r}')
            if name in cams:
                q.rgb_at = camera(q.rgb_at, cams[name])
            self.segs.append((a, b, name, q))

    def seg_index(self, t):
        for k, (a, b, *_ ) in enumerate(self.segs):
            if a <= t < b: return k
        return len(self.segs) - 1

    def frame(self, t):
        k = self.seg_index(t)
        a, b, name, q = self.segs[k]
        img = q.at(t)
        d = self.dissolve.get(name, 0)
        if d and k > 0 and t < a + d and self.segs[k - 1][3] is not None:
            prev = self.segs[k - 1][3].at(t)      # 上一段越界时各来源自己钳到尾帧/按时间继续画
            m = L.bayer_mask((t - a) / d)
            img = np.where(m[..., None], img, prev)
        return img


def camera(rgb_at, cam):
    """给全彩来源套数码推拉：以 (cx,cy) 为中心放大 z 倍，并把中心逐渐移到画面中间。"""
    def f(t):
        img = rgb_at(t)
        cx, cy, z = cam(t)
        if z <= 1.001: return img
        k = (z - 1) / 3                      # 推得越近，越往画面中心对齐
        m = cv2.getRotationMatrix2D((cx, cy), 0, z)
        m[:, 2] += (W / 2 - cx) * min(k, 1), (H / 2 - cy) * min(k, 1)
        return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return f


def push(t, a, b, cx, cy, z0, z1):
    """CAMS 常用写法：a–b 之间从 z0 平滑推到 z1。"""
    k = L.smooth((t - a) / (b - a))
    return cx, cy, z0 + (z1 - z0) * k


def glitch(img, t, spans):
    """CRT 信号故障：横向行错位 + 色偏 + 雪花带，强度中间最大。纯函数（按帧号取随机）。"""
    for a, b in spans:
        if not (a <= t < b): continue
        k = np.sin(np.pi * (t - a) / (b - a))
        fi = int(t * FPS)
        out = img.copy()
        for band in range(10):                                   # 行错位
            y0 = int(L.hash01(fi, band) * H); h = 4 + int(L.hash01(fi, band, 1) * 26)
            out[y0:y0 + h] = np.roll(out[y0:y0 + h], int((L.hash01(fi, band, 2) - .5) * 120 * k), axis=1)
        out[..., 0] = np.roll(out[..., 0], int(8 * k), axis=1)      # 红通道右移、蓝通道左移
        out[..., 2] = np.roll(out[..., 2], -int(8 * k), axis=1)
        rng = np.random.default_rng(fi)
        noise = rng.random((H // 2, W // 2)) < .55 * k               # 2×2 像素雪花
        noise = noise.repeat(2, 0).repeat(2, 1)
        yb = int(L.hash01(fi, 99) * H)
        noise[:max(0, yb - 60)] &= rng.random((max(0, yb - 60), W)) < .35  # 一条滚动的强雪花带，其余稀疏
        out[noise] = np.where(rng.random((int(noise.sum()), 1)) < .5, 235, 20)
        return out
    return img


# ---------------- 重拍：震屏 + 闪白 ----------------
def hits(img, t, times):
    for h in times:
        dt = t - h
        if 0 <= dt < .25:
            img = np.roll(img, int(4 * np.sin(dt * 90) * (1 - dt / .25)), axis=1)
        if 0 <= dt < .17:
            img = img.copy(); img[L.bayer_mask(1 - dt / .17)] = FIX['paper']
    return img


# ---------------- HUD ----------------
F16 = L.font(16)


def hud(img, t, cfg):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.fontmode = '1'
    off = any(a <= t < b for a, b in cfg_get(cfg, 'HUD_OFF', []))
    if not any(a <= t < b for a, b in cfg_get(cfg, 'TOPBAR_OFF', [])):
        d.rectangle((0, 0, W, 17), fill=FIX['ink'])
        d.text((6, 0), '▶ CLAUDE_POP.EXE', fill=FIX['paper'], font=F16)
        titles = [s for a, s in cfg_get(cfg, 'CHAPTERS', []) if t >= a]
        if titles: d.text((180, 0), titles[-1], fill=FIX['orange'], font=F16)
        mm, ss = divmod(t, 60)
        d.text((W - 92, 0), f'{int(mm):02d}:{ss:05.2f}', fill=FIX['yel'], font=F16)
    if not off:
        x0, y0, x1, y1 = 14, 290, W - 14, 352
        d.rectangle((x0, y0, x1, y1), fill=FIX['blue'])
        d.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=FIX['paper'])
        d.rectangle((x0 + 4, y0 + 4, x1 - 4, y1 - 4), outline=FIX['paper'])
        d.rectangle((x0 + 10, y0 - 14, x0 + 82, y0 + 6), fill=FIX['ink'], outline=FIX['orange'])
        d.text((x0 + 14, y0 - 16), 'クロード', fill=FIX['orange'], font=F16)   # 字形上沿有 4px 空白，原来 y0-10 画会掉出名牌下沿（09-30 修，名牌加高 4px、字上下居中）
        for a, b, words, jp in cfg_get(cfg, 'LINES', []):
            if not (a <= t < b): continue
            x = x0 + 14
            for i, (w, ws) in enumerate(words):
                nxt = words[i + 1][1] if i + 1 < len(words) else ws + .5
                c = FIX['dim'] if t < ws else (FIX['yel'] if t < nxt else FIX['paper'])
                d.text((x, y0 + 12), w, fill=c, font=F16)
                x += int(F16.getlength(w + ' '))
            n = int(max(0, t - words[0][1]) * 14)   # 日文副行：唱完第一个词后逐字打出
            d.text((x0 + 14, y0 + 32), jp[:n], fill=FIX['orange'], font=F16)
            if n >= len(jp) and int(t * 4) % 2 == 0:
                d.text((x1 - 24, y1 - 22), '▼', fill=FIX['yel'], font=F16)
    out = np.array(im)
    for s, a, b, side, *opt in cfg_get(cfg, 'SLAMS', []):
        if not (a <= t < b): continue
        if not opt and side in ('left', 'center', 'vertical'): out = slam(out, s, t - a, side)   # 第一、二部分的老写法，保持原样
        else: out = slam2(out, s, t - a, b - a, side, **(opt[0] if opt else {}))
    return out


def slam(img, s, lt, side):
    """冲击大字：前几帧大一号再落定，黄白交替闪。vertical = 竖排日文（放左侧）。"""
    k = lt * FPS
    scale = 6 if k < 2 else (5 if k < 4 else 4)
    if side == 'vertical': scale = 4    # 竖排四个字，固定 4 倍（放大弹入会压到对白框），靠黄白闪烁出冲击
    fg = FIX['yel'] if int(k / 3) % 2 == 0 else FIX['paper']
    chars = list(s) if side == 'vertical' else [s]
    for j, ch in enumerate(chars):
        w = int(F16.getlength(ch)) * scale
        if side == 'vertical':
            x, y = 24, 22 + j * 16 * scale
        else:
            x = {'left': 20, 'center': (W - w) // 2}[side]
            y = 60 + (int(3 * np.sin(k * 2.2)) if k < 8 else 0)
        img = L.draw_text(img, (x + scale, y + scale), ch, FIX['ink'], scale=scale)          # 投影
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            img = L.draw_text(img, (x + dx, y + dy), ch, FIX['orange'], scale=scale)         # 描边
        img = L.draw_text(img, (x, y), ch, fg, scale=scale)
    return img


# ---------------- 第三阶段的大字：多种字效 + 入场 + 位置 ----------------
FIX['pink'], FIX['cyan'], FIX['red'] = (255, 70, 170), (70, 225, 255), (230, 40, 50)


def text_mask(s, scale):
    l, t_, r, b = F16.getbbox(s)
    m = Image.new('1', (r - l + 2, b - t_ + 2), 0)
    ImageDraw.Draw(m).text((1 - l, 1 - t_), s, fill=1, font=F16)
    if scale > 1: m = m.resize((m.width * scale, m.height * scale), Image.NEAREST)
    return np.array(m)


def stamp(img, mask, x, y, color):
    h, w = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1: return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    img[y0:y1, x0:x1][m] = color


def grow(mask, r, pad):
    m = np.pad(mask, pad).astype(np.uint8)
    return cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool) if r else m.astype(bool)


def slam2(img, s, lt, dur, where='left', style='block', anim='pop', scale=None, y=None, color=None, flicker=None):
    """冲击大字（第三阶段）。每段落换一种字效：
      style  block 描边立体字 / glitch 故障字 / neon 霓虹管 / outline 空心细字（淡入淡出）/ vertical 竖排
      anim   pop 放大弹入 / sweep 从左横扫 / drop 从上砸下 / shake 一直抖 / none
      where  left / center / right，或 x 像素；y 默认 60（'mid' 居中，'low' 靠下）
      scale  默认按宽度自动（最大 6）；color 主色（FIX 的键）；flicker 霓虹里接触不良的字母序号
    只用 FIX 里的颜色画（pink/cyan/red 用到时记得在该段 ACCENTS 里加上，保证锁在调色板里）。"""
    img = img.copy()
    k = lt * FPS
    fi = int((lt + dur) * 1000) + int(k)        # 故障随机数种子：每帧不同、但只由时间决定
    if style == 'vertical':
        chars = list(s)
        sc = scale or 4
        for j, ch in enumerate(chars):
            if k < j * 2: continue                                   # 逐字落下
            m = text_mask(ch, sc)
            x = where if isinstance(where, int) else (24 if where != 'right' else W - 24 - m.shape[1])
            yy = (y if isinstance(y, int) else 24) + j * 16 * sc
            g = grow(m, 2, 2)
            stamp(img, grow(m, 0, 2), x + sc - 2, yy + sc - 2, FIX['ink'])
            stamp(img, g, x - 2, yy - 2, FIX[color or 'orange'])
            stamp(img, m, x, yy, FIX['yel'] if int(k / 3) % 2 == 0 else FIX['paper'])
        return img
    w16 = F16.getlength(s) + 2
    sc = scale or int(np.clip((W - 48) // w16, 2, 6))
    if anim == 'pop': sc = sc + (2 if k < 2 else 1 if k < 4 else 0)
    m = text_mask(s, sc)
    h, w = m.shape
    x = {'left': 20, 'center': (W - w) // 2, 'right': W - 20 - w}.get(where, where if isinstance(where, int) else 20)
    yy = {'mid': (H - h) // 2 - 30, 'low': 206 - h, None: 60}.get(y, y) if not isinstance(y, int) else y
    if anim == 'sweep' and k < 5: x = int(-w + (x + w) * L.ease_out_back(k / 5))
    if anim == 'drop' and k < 5: yy = int(-h + (yy + h) * L.ease_out_back(k / 5, 2.5))
    if anim == 'shake' or (anim == 'pop' and k < 8): yy += int(3 * np.sin(k * 2.2))
    if anim == 'shake': x += int(3 * np.sin(k * 3.1))
    fg = FIX[color] if color else (FIX['yel'] if int(k / 3) % 2 == 0 else FIX['paper'])

    if style == 'block':                             # 描边立体字：斜下挤出几层厚度
        g = grow(m, 2, 2)
        for d in range(max(2, sc + 1), 0, -1):
            stamp(img, g, x - 2 + d, yy - 2 + d, FIX['ink'] if d > 2 else FIX['red'])
        stamp(img, g, x - 2, yy - 2, FIX['ink'])
        stamp(img, grow(m, 1, 2), x - 2, yy - 2, FIX['orange'])
        stamp(img, m, x, yy, fg)
    elif style == 'glitch':                          # 故障字：红青错位，横向切条抖动
        g = grow(m, 2, 2)
        on = L.hash01(fi, 7) < .75 or k < 4
        off = int((L.hash01(fi, 1) - .5) * 14) if on else 0
        stamp(img, g, x - 2, yy - 2, FIX['ink'])
        stamp(img, m, x - 4 + off, yy, FIX['pink'])
        stamp(img, m, x + 4 - off, yy, FIX['cyan'])
        stamp(img, m, x, yy, color and FIX[color] or FIX['paper'])
        x0, x1 = max(x - 20, 0), min(x + w + 20, W)
        for band in range(3):                        # 只在字的范围里横切几条（别把旁边的脸也拉花）
            by = yy + int(L.hash01(fi, band, 3) * h); bh = 2 + int(L.hash01(fi, band, 4) * 8)
            y0, y1 = max(by, 0), min(by + bh, H)
            if y0 < y1 and x0 < x1:
                img[y0:y1, x0:x1] = np.roll(img[y0:y1, x0:x1], int((L.hash01(fi, band, 5) - .5) * 30), axis=1)
    elif style == 'neon':                            # 霓虹管：暗色光晕 + 亮色空心管 + 白芯，逐字母接触不良
        c = FIX[color or 'pink']
        glow = grow(m, 3, 3)
        edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        lit = k > 4 or L.hash01(int(k), 11) < .5         # 开灯时闪几下
        if lit:
            stamp(img, glow, x - 3, yy - 3, FIX['blue'])
            stamp(img, grow(m, 1, 1), x - 1, yy - 1, c)
            stamp(img, edge, x, yy, FIX['paper'])
        if flicker is not None and L.hash01(int(k / 2), 13) < .45:   # 某个字母接触不良：只有这个字母的灯管变暗
            cx0 = int(F16.getlength(s[:flicker]) * sc); cx1 = int(F16.getlength(s[:flicker + 1]) * sc)
            one = np.zeros_like(m); one[:, cx0:cx1] = m[:, cx0:cx1]
            stamp(img, grow(one, 1, 1), x - 1, yy - 1, FIX['blue'])
    elif style == 'outline':                         # 空心细字：只画轮廓，Bayer 淡入淡出
        edge = grow(m, 1, 1) & ~np.pad(m, 1)
        fade = min(1, k / 6, max(0, (dur - lt)) * FPS / 8)
        sub = np.zeros((H, W), bool)
        stamp(sub, edge, x - 1, yy - 1, True)
        img[sub & L.bayer_mask(fade)] = color and FIX[color] or FIX['paper']
    return img


def tilt(img, t, spans):
    """整帧歪斜（HUD 一起歪）：进入时 3 帧甩过去带点回弹，之后定住。spans = [(a, b, 角度)]。
    四项 (a, b, 起始角度, 结束角度)：a–b 之间从起始角度平滑转到结束角度，进入时不甩（part4 开头从 -12° 转正，接第三部分末帧）。"""
    for a, b, deg, *d1 in spans:
        if not (a <= t < b): continue
        k = (t - a) * FPS
        if d1: d = deg + (d1[0] - deg) * L.smooth((t - a) / (b - a))
        else: d = deg * (L.ease_out_back(k / 3, 3) if k < 3 else 1)
        if abs(d) < .01: return img
        m = cv2.getRotationMatrix2D((W / 2, H / 2), d, 1 + abs(d) / 60)
        return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=FIX['ink'])
    return img


def frame(tl, t):
    cfg = tl.cfg
    img = glitch(tl.frame(t), t, cfg_get(cfg, 'GLITCH', []))
    img = hud(hits(img, t, cfg_get(cfg, 'HITS', [])), t, cfg)
    overlay = cfg_get(cfg, 'OVERLAY', None)                # 量化以后再画的一层（part4 的演职员表），和 HUD 一样只用调色板原色
    if overlay: img = overlay(img, t)
    return tilt(img, t, cfg_get(cfg, 'TILT', []))


def run(cfg):
    start, end = cfg_get(cfg, 'START', 0.0), cfg.END
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sheet', help='逗号分隔的歌曲时间，只出拼图')
    ap.add_argument('--from', dest='lo', type=float, default=start)
    ap.add_argument('--to', dest='hi', type=float, default=end)
    ap.add_argument('--no-scan', action='store_true')
    a = ap.parse_args()
    if a.sheet:
        ts = [float(x) for x in a.sheet.split(',')]
        tl = Timeline(cfg, min(ts), max(ts))
        tiles = [P.upscale(frame(tl, t), not a.no_scan) for t in ts]
        tiles += [np.zeros_like(tiles[0])] * (len(tiles) % 2)
        rows = [np.concatenate(tiles[r:r + 2], 1) for r in range(0, len(tiles), 2)]
        Image.fromarray(np.concatenate(rows, 0)).resize((1920, 540 * len(rows))).save(a.out)
        print('wrote', a.out)
        return
    tl = Timeline(cfg, a.lo, a.hi)
    n0, n1 = int(round(a.lo * FPS)), int(round(a.hi * FPS))
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * P.S}x{H * P.S}',
                           '-r', str(FPS), '-i', '-', '-ss', f'{a.lo:.3f}', '-t', f'{(n1 - n0) / FPS:.3f}', '-i', L.SONG,
                           '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                           '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    for i in range(n0, n1):
        ff.stdin.write(P.upscale(frame(tl, i / FPS), not a.no_scan).tobytes())
        if (i - n0) % 48 == 0: print(f'[{i - n0 + 1}/{n1 - n0}] t={i / FPS:.2f}s')
    ff.stdin.close(); ff.wait()
    print('wrote', a.out)
