"""backdrops_p2: 第二部分 live98 舞台段的背景（接 live98.compose 的 backdrop 钩子，把舞台背景整个换掉）。

roof(img, t, lit, beat)   副歌二 58.28–65.90 夜晚楼顶：
    城市夜景 + 左侧楼顶广告牌 P(DOOM) 34→71（drop 60.26 翻牌）
    → basilisk（61.04）城市背后升起蛇怪，boom（61.82）一吼、满楼窗户全亮
    → NVDA（62.32）天空变 K 线图，火箭顺着 K 线冲上月亮（moon 63.76）
    → Omega（64.20）地平线升起巨大的 Ω，像日出一样把天染成金色

t 是歌曲时间（mv.LiveSource 已换算）。全是纯函数，画在 640×360 全彩 float 图上，之后 live98 叠人物、彩带、运镜。
人物站在画面右侧（cx≈.58），左侧三分之一留给广告牌和冲击大字。
"""
import functools

import cv2
import numpy as np

import part1_lib as L
import props

W, H = L.W, L.H
DROP, BASILISK, BOOM, NVDA, MOON, OMEGA = 60.26, 61.04, 61.82, 62.32, 63.76, 64.20
HORIZON = 250          # 城市天际线底部（再往下是她站的楼顶）
MOON_XY = (572, 62)

NIGHT = [(10, 10, 40), (22, 14, 60), (44, 18, 80), (80, 24, 96)]          # 夜空色带（上 → 下）
GOLDEN = [(120, 60, 110), (220, 90, 90), (255, 150, 70), (255, 210, 110)]  # Ω 日出后的色带
CITY = (18, 12, 36)
WINDOW = (255, 214, 120)
CANDLE_UP, CANDLE_DN = (60, 230, 120), (240, 60, 70)
# 只在少数几帧出现、会被 k-means 挤掉的颜色：K 线绿、Ω 金
ACCENTS = np.array([CANDLE_UP, (255, 200, 70)], np.uint8)


def bands(img, cols, y1):
    """天空横向色带（PC-98 没有渐变，用 4 条平涂带）。"""
    n = len(cols)
    for i, c in enumerate(cols):
        img[int(y1 * i / n):int(y1 * (i + 1) / n)] = c
    return img


@functools.lru_cache(maxsize=None)
def buildings():
    """天际线：确定性随机的楼 (x0, x1, 顶部 y)。左边第 2 栋最高，给广告牌当底座。"""
    out, x = [], -10
    i = 0
    while x < W:
        w = 34 + int(L.hash01(i, 1) * 50)
        top = HORIZON - 40 - int(L.hash01(i, 2) * 90)
        if i == 2: top = 168
        out.append((x, x + w, top))
        x += w + 2
        i += 1
    return out


@functools.lru_cache(maxsize=None)
def basilisk_rgba():
    """蛇怪 CG：纯绿底抠掉，缩到 330 宽。"""
    im = L.load_rgba('素材/CG/p2_basilisk.png')[..., :3].astype(np.int16)
    g = (im[..., 1] > 170) & (im[..., 0] < 120) & (im[..., 2] < 120)
    rgba = np.dstack([im.astype(np.uint8), np.where(g, 0, 255).astype(np.uint8)])
    return cv2.resize(rgba, (330, int(330 * rgba.shape[0] / rgba.shape[1])), interpolation=cv2.INTER_AREA)


def sky(img, t, lit):
    img = bands(img, NIGHT, HORIZON)
    if t >= OMEGA:                                   # Ω 升起：色带逐渐网点溶成金色
        gold = bands(np.zeros_like(img), GOLDEN, HORIZON)
        m = L.bayer_mask(L.smooth((t - OMEGA) / .9))
        m[HORIZON:] = False
        img = np.where(m[..., None], gold, img)
    for i in range(40):                              # 星星：按哈希位置，随时间闪
        x, y = int(L.hash01(i, 11) * W), int(L.hash01(i, 12) * (HORIZON - 60))
        if np.sin(t * 3 + i) > .2 and t < OMEGA + .5:
            img[y:y + 2, x:x + 2] = (255, 255, 230)
    if not lit:
        img = img * .45
    return img


def moon(img, t):
    if t < NVDA - .2 or t >= OMEGA: return img       # Omega 时月亮翻成 Ω 太阳（见 omega）
    x, y = MOON_XY
    cv2.circle(img, (x, y), 30, (255, 240, 180), -1)
    cv2.circle(img, (x + 9, y - 8), 7, (225, 205, 140), -1)
    cv2.circle(img, (x - 11, y + 10), 5, (225, 205, 140), -1)
    cv2.circle(img, (x, y), 30, (40, 30, 60), 2)
    return img


def candles(img, t):
    """NVDA to the moon：从左下沿对角线向月亮一根根长出 K 线，火箭在最前端。"""
    if t < NVDA or t >= OMEGA + .8: return img
    x0, y0 = 262, HORIZON - 10                        # 从广告牌右边、她身后斜着冲向月亮（画在楼前面）
    x1, y1 = MOON_XY[0] - 34, MOON_XY[1] + 26
    k = min(1.0, (t - NVDA) / (MOON - NVDA))
    n_total = 16
    n = int(k * n_total)
    for i in range(n):
        f = i / n_total
        cx = int(x0 + (x1 - x0) * f)
        cy = int(y0 + (y1 - y0) * f ** 1.6)           # 越涨越陡
        up = L.hash01(i, 31) > .18
        h = 16 + int(L.hash01(i, 32) * 16)
        col = CANDLE_UP if up else CANDLE_DN
        cv2.line(img, (cx + 5, cy - h // 2 - 7), (cx + 5, cy + h // 2 + 7), (230, 230, 230), 2)
        cv2.rectangle(img, (cx, cy - h // 2), (cx + 10, cy + h // 2), col, -1)
        cv2.rectangle(img, (cx, cy - h // 2), (cx + 10, cy + h // 2), (10, 10, 20), 2)
    if t < MOON:                                     # 火箭：白身红翼，尾焰按帧闪
        f = k
        rx = x0 + (x1 - x0) * f + 8
        ry = y0 + (y1 - y0) * f ** 1.6 - 14
        ang = np.arctan2((y1 - y0) * 1.6 * max(f, .05) ** .6, x1 - x0)   # 10-19 修：原来 y 分量多了个负号，机头朝右下、往右上飞
        d = np.array([np.cos(ang), np.sin(ang)])
        nrm = np.array([-d[1], d[0]])
        c = np.array([rx, ry])
        s = 1.6                                         # 火箭放大
        body = np.array([c + d * 16 * s, c + nrm * 5 * s, c - d * 12 * s + nrm * 5 * s, c - d * 12 * s - nrm * 5 * s, c - nrm * 5 * s], np.int32)
        fin = np.array([c - d * 8 * s + nrm * 5 * s, c - d * 14 * s + nrm * 10 * s, c - d * 14 * s + nrm * 5 * s], np.int32)
        fin2 = np.array([c - d * 8 * s - nrm * 5 * s, c - d * 14 * s - nrm * 10 * s, c - d * 14 * s - nrm * 5 * s], np.int32)
        flame = np.array([c - d * 12 * s + nrm * 4 * s, c - d * (22 + 10 * L.hash01(int(t * 24), 5)) * s, c - d * 12 * s - nrm * 4 * s], np.int32)
        cv2.fillPoly(img, [flame], (255, 180, 60))
        cv2.fillPoly(img, [fin, fin2], (230, 50, 60))
        cv2.fillPoly(img, [body], (245, 245, 250))
        cv2.polylines(img, [body], True, (20, 16, 30), 1)
    elif t < MOON + .4:                              # 撞月：放射状闪光
        dt = (t - MOON) / .4
        for a in range(8):
            ang = a * np.pi / 4
            r0, r1 = 34 + dt * 20, 50 + dt * 40
            p0 = (int(MOON_XY[0] + np.cos(ang) * r0), int(MOON_XY[1] + np.sin(ang) * r0))
            p1 = (int(MOON_XY[0] + np.cos(ang) * r1), int(MOON_XY[1] + np.sin(ang) * r1))
            cv2.line(img, p0, p1, (255, 255, 255), 3)
    return img


def omega(img, t):
    """月亮翻成 Ω 太阳：Omega 时月亮位置弹出一个金色 Ω，边变大边往上浮，背后一圈旋转放射光条把天染金。"""
    if t < OMEGA: return img
    k = L.ease_out_back((t - OMEGA) / .5)
    cx, cy = MOON_XY[0] - 16, MOON_XY[1] + 20 - int(10 * L.smooth((t - OMEGA) / 1.5))
    ang = np.arctan2(L_YY - cy, L_XX - cx) + t * .4
    rays = (np.sin(ang * 10) > .3) & (np.hypot(L_YY - cy, L_XX - cx) > 50 * k) & (L_YY < HORIZON)
    rays &= L.bayer_mask(L.smooth((t - OMEGA) / .6))
    img[rays] = (255, 200, 90)
    r = int(46 * k)
    if r > 2:
        cv2.circle(img, (cx, cy), r, (255, 230, 150), -1)
        cv2.circle(img, (cx, cy), r, (40, 20, 30), 2)
    sc = max(1, int(round(4 * k)))
    l_, t_, r_, b_ = L.font(16).getbbox('Ω')                        # 10-19 修：按字形实际外框居中（原来手估偏移，Ω 偏左上）
    ox, oy = cx - (r_ - l_ + 2) * sc // 2, cy - (b_ - t_ + 2) * sc // 2
    img[:] = L.draw_text(img, (ox + 3, oy + 3), 'Ω', (40, 20, 30), scale=sc)
    img[:] = L.draw_text(img, (ox, oy), 'Ω', (230, 120, 30), scale=sc)
    return img


L_YY, L_XX = np.mgrid[0:H, 0:W]


def basilisk(img, t):
    """蛇怪从城市背后升起（画在城市之前 = 被楼挡住下半身），boom 时一吼：放大 + 抖。"""
    if t < BASILISK - .3 or t >= NVDA + 1.2: return img
    if t < BOOM:
        rise = L.smooth((t - (BASILISK - .3)) / .75)     # basilisk 唱完前就升到顶，boom 时已经全露出来再吼
    else:
        rise = 1 - L.smooth((t - NVDA) / 1.2)          # NVDA 起沉回去，让出天空给 K 线
    sp = basilisk_rgba()
    s = 1.0
    dx = dy = 0
    if BOOM <= t < BOOM + .45:
        dt = t - BOOM
        s = 1 + .12 * np.exp(-dt * 8)
        dx, dy = int(np.sin(dt * 90) * 5), int(np.cos(dt * 70) * 4)
    x = 548 - sp.shape[1] * s / 2 + dx            # 她右边（她在 x≈330–480），头从楼后露出来
    y = HORIZON - 30 - sp.shape[0] * s * .95 * rise + (1 - rise) * 60 + dy
    return L.paste_rgba(img, sp, x, y, s)


EQ_T = OMEGA + .30          # 10-19 用户：太阳出来后，楼里的灯变成跳动的波形 → 每一列窗户是一个频段，按原曲真实频谱从下往上亮
EQ_HOT = (255, 120, 90)     # 顶上那一格换暖红，像频谱仪的峰值


def city(img, t, lit):
    """天际线 + 窗户（boom 时全亮；Ω 日出后变频谱仪）+ 前景楼顶栏杆。"""
    all_on = BOOM <= t < BOOM + 1.2 or OMEGA <= t < EQ_T
    eq = t >= EQ_T
    if eq:
        cols = (W - 4) // 9 + 1
        st = getattr(L, 'REAL_T', None)                             # 串烧里重播这段背景时，频谱跟着当下的音乐走
        lv = props.spectrum(t if st is None else st, cols)
        lv = np.convolve(np.pad(lv, 1, mode='edge'), [.25, .5, .25], 'valid')      # 相邻列抹一下，连成一道波
        lv = np.clip(lv * 1.25, 0, 1)
    for i, (x0, x1, top) in enumerate(buildings()):
        img[top:HORIZON + 40, max(x0, 0):min(x1, W)] = CITY
        img[top:top + 2, max(x0, 0):min(x1, W)] = (50, 36, 80)
        rows = list(range(top + 8, HORIZON + 30, 10))
        for wx in range(x0 + 5, x1 - 6, 9):
            if not 0 <= wx < W - 4: continue
            if eq:                                                     # 整条天际线共用一个高度基准（不是每栋楼各算各的），波形才连得起来
                bar_top = HORIZON + 30 - lv[min(wx // 9, len(lv) - 1)] * 170
            for r, wy in enumerate(rows):
                if eq:
                    if wy >= bar_top:
                        img[wy:wy + 5, wx:wx + 4] = EQ_HOT if wy < bar_top + 10 and wy < HORIZON - 10 else WINDOW
                elif all_on or L.hash01(i, wx, wy) < (.35 if lit else .15):
                    img[wy:wy + 5, wx:wx + 4] = WINDOW
    img[HORIZON + 40:] = (36, 30, 52)                  # 她站的楼顶地面（大部分被对白框挡住）
    img[HORIZON + 34:HORIZON + 40] = (70, 62, 96)      # 楼顶边沿
    for x in range(0, W, 26):                          # 栏杆
        img[HORIZON + 14:HORIZON + 40, x:x + 3] = (70, 62, 96)
    img[HORIZON + 14:HORIZON + 17] = (70, 62, 96)
    return img


# ---------------- 跳舞毯：Forward MLP, backward, repeat（72.80–77.28） ----------------
FORWARD, BACKWARD, REPEAT = 73.60, 76.10, 76.46
NET_LAYERS = 5
FLOOR_Y0, FLOOR_Y1 = 170, 300          # 地板在画面里的上下沿（再往下被对白框挡住）
RAINBOW = [(255, 70, 90), (255, 170, 60), (255, 230, 80), (80, 230, 130), (70, 200, 255), (170, 110, 255)]
DDR_ACCENTS = np.array([(70, 230, 255), (255, 70, 90)], np.uint8)


def net_nodes():
    """地板上的神经网络：NET_LAYERS 层，每层 4 个节点，按透视排在地板上（x 从左到右 = 前向方向）。"""
    nodes = []
    for li in range(NET_LAYERS):
        layer = []
        for j in range(4):
            v = (j + .5) / 4                                   # 地板深度 0（远）→ 1（近）
            y = FLOOR_Y0 + 14 + (FLOOR_Y1 - FLOOR_Y0 - 20) * v
            spread = .55 + .45 * v                             # 近大远小
            x = W / 2 + (li - (NET_LAYERS - 1) / 2) / (NET_LAYERS - 1) * W * .86 * spread
            layer.append((int(x), int(y), 3 + int(3 * v)))
        nodes.append(layer)
    return nodes


def layer_glow(t):
    """每层的亮度和颜色：forward 从左往右一拍一层亮青色，backward 从右往左亮红色，repeat 后全网彩虹。"""
    beat = L.beat_phase(t)[0]
    out = []
    for li in range(NET_LAYERS):
        if t >= REPEAT:
            out.append((1.0, RAINBOW[(beat + li) % len(RAINBOW)]))
        elif t >= BACKWARD:
            k = (t - BACKWARD) / ((REPEAT - BACKWARD) / NET_LAYERS)
            out.append((1.0 if NET_LAYERS - 1 - li <= k else .25, (255, 70, 90)))
        elif t >= FORWARD:
            k = (t - FORWARD) / L.BEAT132                            # 一拍亮一层，亮满后循环
            k = k % (NET_LAYERS + 2)
            out.append((1.0 if li <= k else .25, (70, 230, 255)))
        else:
            out.append((.25, (70, 230, 255)))
    return out


# ---------------- 11-12：FORWARD MLP 舞台更华丽（串烧 m3 / m7 用 replay 重播同一个函数，跟着变）----------------
# 顶上灯架打光柱按拍扫 / 后墙三块 LED 点阵屏放原曲频谱 / 地板换成跳舞机发光格 / 网络连线上跑光点
# （forward 青色往右、backward 红色往左）/ repeat 后两个底角扫激光扇。
NEON = [(255, 60, 170), (60, 220, 255), (255, 220, 70)]     # 光柱、激光：粉 / 青 / 黄


def _song_t(t):
    return getattr(L, 'REAL_T', None) or t                     # 串烧重播时频谱跟当下的音乐


def _wall(img, t):
    """后墙：深色底 + 三块 LED 点阵屏（左右两块放频谱柱，中间一块在人物身后、只亮暗格子）。"""
    img[18:FLOOR_Y0] = (12, 8, 34)
    spec = props.spectrum(_song_t(t), 12)
    for (x0, x1), mode in (((14, 196), 'bars'), ((222, 418), 'dim'), ((444, 626), 'bars')):
        y0, y1 = 70, FLOOR_Y0 - 6
        img[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = (40, 34, 78)          # 屏框
        img[y0:y1, x0:x1] = (6, 4, 18)
        cols, rows = (x1 - x0) // 6, (y1 - y0) // 6
        for c in range(cols):
            lv = spec[int(c / cols * 12)] if mode == 'bars' else 0
            hgt = int(lv * rows)
            for r in range(rows):
                on = rows - 1 - r < hgt
                if mode == 'bars':
                    col = (255, 70, 90) if rows - 1 - r >= rows - 2 and on else (NEON[1] if r > rows * .45 else NEON[0])
                    if not on: col = (22, 16, 48)
                else:
                    col = (34, 24, 70) if (c + r + int(t * 8)) % 7 else (90, 60, 150)
                xx, yy = x0 + c * 6 + 1, y0 + r * 6 + 1
                img[yy:yy + 4, xx:xx + 4] = col
    return img


def _truss(img, t):
    """顶上灯架 + 8 盏灯打下来的光柱，按拍左右扫，拍点上亮一下。"""
    beams = np.zeros_like(img)
    ph = L.beat_phase(t)
    for i in range(8):
        x = 40 + i * 80
        ang = .45 * np.sin(np.pi * (ph[0] + ph[1]) / 2 + i * .8) * (1 if i % 2 else -1)
        bx, by = x + np.tan(ang) * 260, FLOOR_Y1
        poly = np.array([(x - 5, 26), (x + 5, 26), (bx + 34, by), (bx - 34, by)], np.int32)
        cv2.fillPoly(beams, [poly], NEON[i % 2])
    k = .16 + .14 * L.pulse(t, 5)
    img = img * (1 - k * (beams.max(-1, keepdims=True) > 0)) + beams * k
    img[18:28] = (30, 26, 50)                                       # 灯架
    for i in range(8):
        x = 40 + i * 80
        img[24:32, x - 7:x + 7] = (60, 54, 80)
        img[28:32, x - 4:x + 4] = NEON[i % 2]
    return img


def _floor(img, t):
    """跳舞机地板：透视格子，按拍一格格亮彩虹色。"""
    img[FLOOR_Y0:] = (20, 14, 48)
    beat = L.beat_phase(t)[0]
    rows, cols = 5, 10
    for r in range(rows):
        v0, v1 = r / rows, (r + 1) / rows
        ya, yb = FLOOR_Y0 + (H - FLOOR_Y0) * v0 ** 1.4, FLOOR_Y0 + (H - FLOOR_Y0) * v1 ** 1.4
        sa, sb = .55 + .45 * v0, .55 + .45 * v1
        for c in range(cols):
            xa0, xa1 = W / 2 + (c / cols - .5) * W * 1.3 * sa, W / 2 + ((c + 1) / cols - .5) * W * 1.3 * sa
            xb0, xb1 = W / 2 + (c / cols - .5) * W * 1.3 * sb, W / 2 + ((c + 1) / cols - .5) * W * 1.3 * sb
            lit = L.hash01(beat, r, c, 5) < .22
            col = RAINBOW[(r * 3 + c + beat) % len(RAINBOW)]
            col = tuple(int(v * (.55 if lit else .16)) for v in col)
            quad = np.array([(xa0 + 1, ya + 1), (xa1 - 1, ya + 1), (xb1 - 1, yb - 1), (xb0 + 1, yb - 1)], np.int32)
            cv2.fillPoly(img, [quad], col)
    img[FLOOR_Y0:FLOOR_Y0 + 2] = (110, 90, 190)
    return img


def _net(img, t):
    """地板上的网络：连线 + 节点（同 v1），再加沿连线跑的光点：forward 青色往右，backward 红色往左，repeat 后彩虹双向。"""
    nodes, glow = net_nodes(), layer_glow(t)
    for li in range(NET_LAYERS - 1):
        g, c = glow[li] if t < BACKWARD else glow[li + 1]
        col = tuple(int(v * g * .8) for v in c)
        for a in nodes[li]:
            for b in nodes[li + 1]:
                cv2.line(img, a[:2], b[:2], col, 2 if g > .5 else 1)
    if t >= FORWARD:
        u = (t - FORWARD) / (L.BEAT132 / 2) % 1                      # 半拍跑完一段
        back = BACKWARD <= t < REPEAT
        for li in range(NET_LAYERS - 1):
            for ai, a in enumerate(nodes[li]):
                for bi, b in enumerate(nodes[li + 1]):
                    if L.hash01(li, ai, bi) > .5: continue
                    p, q = (b, a) if back else (a, b)
                    x = p[0] + (q[0] - p[0]) * u; y = p[1] + (q[1] - p[1]) * u
                    col = (255, 70, 90) if back else ((70, 230, 255) if t < REPEAT else RAINBOW[(li + ai) % 6])
                    img[int(y) - 2:int(y) + 2, int(x) - 2:int(x) + 2] = col
    for li, layer in enumerate(nodes):
        g, c = glow[li]
        for x, y, r in layer:
            cv2.circle(img, (x, y), r + 2, (10, 8, 20), -1)
            cv2.circle(img, (x, y), r, tuple(int(v * max(g, .35)) for v in c), -1)
            if g > .5: cv2.circle(img, (x, y), r + 4, c, 1)
    return img


def _lasers(img, t):
    """repeat 之后：左右两个底角按拍扫激光扇。"""
    if t < REPEAT: return img
    ph = L.beat_phase(t)
    for side in (0, 1):
        ox = 0 if side == 0 else W - 1
        for i in range(6):
            a = -np.pi / 2 + (1 if side == 0 else -1) * (.25 + .14 * i + .25 * np.sin(np.pi * ph[1]))
            ex, ey = ox + np.cos(a) * 700, H + np.sin(a) * 700
            cv2.line(img, (ox, H - 1), (int(ex), int(ey)), NEON[(i + side) % 3], 1)
    return img


def ddr(img, t, lit, beat):
    img = np.array(img, np.float32)
    img = _wall(img, t)
    img = _floor(img, t)
    img = _truss(img, t)
    img = _net(img, t)
    img = _lasers(img, t)
    # 左侧 DDR 箭头轨道（同 v1）
    arrows = '←↓↑→'
    bx, by, lw = 18, 28, 36
    img[by - 4:by + 36, bx - 4:bx + lw * 4] = (30, 24, 70)
    for j, a in enumerate(arrows):
        img[:] = L.draw_text(img, (bx + j * lw, by), a, (110, 110, 170), scale=2)
    ph = L.beat_phase(t)
    for n in range(5):
        idx = ph[0] + n
        lane = int(L.hash01(idx, 41) * 4)
        y = by + (n - ph[1]) * 30
        if by - 2 <= y < 170:
            c = RAINBOW[idx % len(RAINBOW)]
            hit = n == 0 and ph[1] < .35
            img[:] = L.draw_text(img, (bx + lane * lw, int(y)), arrows[lane], (255, 255, 255) if hit else c, scale=2)
    return img


def roof(img, t, lit, beat):
    img = np.array(img, np.float32)
    img = sky(img, t, lit)
    img = moon(img, t)
    img = omega(img, t)
    img = basilisk(img, t)
    img = city(img, t, lit)
    img = candles(img, t)
    b = buildings()[2]
    img = props.billboard(img, t, (b[0] - 30, b[2] - 120, 196, 116), 34, 71, 59.98, DROP, lit, beat)   # 09-30 加高 16：4 倍数字原来切掉底下 10px
    return img
