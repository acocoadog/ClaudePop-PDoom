"""props: 画面里的实物道具（P(DOOM) 记分牌 / 广告牌……），给 live98 的 backdrop 钩子和代码镜头共用。

P(DOOM) 升级路线见 docs/分镜/README.md：便利贴 8%（s4）→ 舞台记分牌 8→34%（part1 s6）→ 楼顶广告牌 34→71%（part2 副歌）。
函数都是纯函数：只由歌曲时间 t 决定。画在全彩 float/uint8 图上，返回新图。
"""
import numpy as np

import part1_lib as L

LED = (255, 60, 50)          # 亮红 LED
LED_DIM = (110, 22, 26)      # 暗场时的暗红
GOLD = (255, 200, 70)
BOARD_INK = (10, 8, 20)
# 段落调色板要给 LED 红留名额，否则会被 k-means 挤掉。只留这一个：多留会挤掉人物的颜色
# （09-28 试过 LED+暗红+金 三个，s6 舞台服的淡紫整个没了），暗红和金框就近落到舞台已有的颜色上
ACCENTS = np.array([LED], np.uint8)


# ---------------- 原曲频谱（10-19：Ω 日出后楼里的窗户变成跳动的频谱仪） ----------------
_PCM = []


def _pcm():
    if not _PCM:
        import subprocess
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', L.SONG, '-ac', '1', '-ar', '22050', '-f', 's16le', '-'],
                             capture_output=True, check=True).stdout
        _PCM.append(np.frombuffer(raw, np.int16).astype(np.float32) / 32768)
    return _PCM[0]


def spectrum(t, n, lo=50.0, hi=9000.0):
    """歌曲时间 t 处 n 个对数频段的响度 0–1（纯函数：只由 t 决定）。每段按自己的典型响度归一，低频不会一直顶满。"""
    x, sr, N = _pcm(), 22050, 2048
    i = int(t * sr) - N // 2
    seg = x[max(i, 0):max(i, 0) + N]
    if len(seg) < N: seg = np.pad(seg, (0, N - len(seg)))
    mag = np.abs(np.fft.rfft(seg * np.hanning(N)))
    f = np.fft.rfftfreq(N, 1 / sr)
    edges = np.geomspace(lo, hi, n + 1)
    out = np.zeros(n)
    for k in range(n):
        m = (f >= edges[k]) & (f < edges[k + 1])
        out[k] = mag[m].mean() if m.any() else 0
    db = 20 * np.log10(out + 1e-6)
    ref = np.linspace(30, 12, n)                     # 高频天然弱，按频段给一条斜的参考线
    return np.clip((db - ref + 26) / 30, 0, 1)


def pdoom_value(t, v0, v1, t0, t1):
    """t0 前是 v0；t0–t1 像老虎机一样乱跳、越来越接近 v1；t1 起停在 v1。"""
    if t < t0: return v0
    if t >= t1: return v1
    k = (t - t0) / (t1 - t0)
    fi = int(t * L.FPS)
    jitter = (L.hash01(fi, 7) - .5) * 40 * (1 - k)
    return int(np.clip(round(v0 + (v1 - v0) * k + jitter), 0, 99))


def scoreboard(img, t, box, v0, v1, t0, t1, lit=1.0, beat=0.0):
    """舞台记分牌：金框黑底，上面一行 P(DOOM)，下面大号 LED 数字。
    box = (x, y, 宽, 高)；lit = 舞台亮度 0–1（暗场时只有暗红数字）；beat = 拍点脉冲 0–1（亮场时框随拍闪）。"""
    x, y, w, h = box
    img = img.copy()
    on = lit > .5 or t >= t0
    frame_col = np.array(GOLD if (on and beat > .5) else ((180, 130, 50) if on else (40, 32, 50)), np.float32)
    img[y:y + h, x:x + w] = frame_col
    img[y + 3:y + h - 3, x + 3:x + w - 3] = BOARD_INK
    img = L.draw_text(img, (x + 8, y + 5), 'P(DOOM)', GOLD if on else (90, 70, 60))
    v = pdoom_value(t, v0, v1, t0, t1)
    digit_col = LED if on else LED_DIM
    if t0 <= t < t1 and int(t * L.FPS) % 2: digit_col = (255, 255, 255)   # 乱跳时红白交替闪
    s = f'{v:>2d}%'
    img = L.draw_text(img, (x + 8, y + 24), s, digit_col, scale=digit_scale(h - 3 - 24, 3))
    return img


def digit_scale(room, want):
    """像素字放大 n 倍高 15n 像素（DotGothic16，draw_text 已去掉字顶空白）。在 room 像素高里放得下的最大倍数，不超过 want。
    09-30 用户：全片好几块牌子的数字下半截被切掉（字比牌子的黑底高），画数字一律过这个。"""
    return max(1, min(want, room // 15))


def billboard(img, t, box, v0, v1, t0, t1, lit=1.0, beat=0.0):
    """楼顶巨型广告牌（第二部分副歌）：比记分牌大一号，外框一圈灯泡按拍跑马，底下两根支架。
    box = (x, y, 宽, 高)。数字规则同 scoreboard：t0 前 v0，t0–t1 乱跳，之后 v1。"""
    x, y, w, h = box
    img = img.copy()
    on = lit > .5 or t >= t0
    for lx in (x + w // 5, x + w * 4 // 5):                      # 支架
        img[y + h:y + h + 40, lx - 3:lx + 3] = (40, 36, 60)
    img[y:y + h, x:x + w] = (60, 50, 80) if on else (30, 26, 44)
    img[y + 6:y + h - 6, x + 6:x + w - 6] = BOARD_INK
    step = 12                                                     # 灯泡跑马：每拍前进一格
    k = L.beat_phase(t)[0]
    n = 0
    for bx in range(x + 3, x + w - 3, step):
        for by in (y + 1, y + h - 5):
            lit_bulb = on and (n + k) % 3 == 0
            img[by:by + 4, bx:bx + 4] = GOLD if lit_bulb else (90, 70, 40)
            n += 1
    img = L.draw_text(img, (x + 14, y + 10), 'P(DOOM)', GOLD if on else (90, 70, 60), scale=2)
    v = pdoom_value(t, v0, v1, t0, t1)
    col = LED if on else LED_DIM
    if t0 <= t < t1 and int(t * L.FPS) % 2: col = (255, 255, 255)
    if t >= t1 and beat > .6: col = (255, 255, 255)               # 落定后随拍闪白
    img = L.draw_text(img, (x + 14, y + 44), f'{v:>2d}%', col, scale=digit_scale(h - 6 - 44, 4))
    return img
