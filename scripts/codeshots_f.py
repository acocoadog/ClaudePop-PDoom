"""codeshots_f: 第三部分 C 段（109.35 起底鼓全回来，一个词一张卡）的代码镜头（分镜第 9–13 镜）。

    block      109.88–110.44  JUST：一块 Transformer 积木砸下来
    tower      110.44–112.02  TRANSFORMERS：镜头沿着积木塔一路往下坠，LAYER 12 → 9,999
    turtles    112.02–113.34  ALL THE WAY：坠到最底，Q 版 Claude 托着一路通到天上的积木塔
    sysprompt  113.34–114.20  SYSTEM: You will obey.
    chinchilla 115.22–116.12  龙猫吃数据草捆，TOKENS/PARAM 20
    dense      116.12–116.88  压成超高密度发光方块
    fences     116.88–118.74  横版跑酷，一个音节撞碎一道安全围栏
    sheep      118.74–120.90  数 GPU 像数羊，数到 100,000 全堆成山

接口同 part1_lib。人物（Q 版精灵）按拍换姿势、停帧，运动靠代码位移。
自检：python codeshots_f.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np

import codeshots_e as E
import part1_lib as L
from codeshots_e import INK, PAPER, ORANGE, YEL, BLUE, PINK, CYAN, RED, box, poly, text, text_c, star4, f12, f24, dither_to

W, H = L.W, L.H
BEAT = L.BEAT132
Q3 = '素材/精灵/q版Claude_五姿势3.png'


@functools.lru_cache(maxsize=None)
def q3(idx, height):
    """五姿势3 的第 idx 个姿势（0 托举 / 1、2 跑步两帧 / 3 睡衣躺 / 4 睡衣惊坐），按 alpha 列投影自动切。"""
    rgba = L.load_rgba(Q3)
    col = (rgba[..., 3] > 127).any(0)
    runs, s = [], None
    for x, v in enumerate(np.append(col, False)):
        if v and s is None: s = x
        if not v and s is not None:
            if x - s > 40: runs.append((s, x))
            s = None
    if len(runs) == 4:                                         # 两个睡衣姿势头发连在一起：在最后一段中间找像素最少的列切开
        a, b = runs[3]
        cnt = (rgba[:, a:b, 3] > 127).sum(0)
        m = a + (b - a) // 3 + int(np.argmin(cnt[(b - a) // 3:2 * (b - a) // 3]))
        runs[3:] = [(a, m), (m, b)]
    a, b = runs[idx]
    sub = rgba[:, a:b]
    rows = np.nonzero((sub[..., 3] > 127).any(1))[0]
    sub = sub[rows.min():rows.max() + 1]
    import codeshots_c as C
    return C._rgba_resize(sub, height / sub.shape[0])


def put(img, spr, cx, feet):
    h, w = spr.shape[:2]
    return L.paste_rgba(img, spr, cx - w // 2, feet - h)


# ================================================================
# 电路板底（9 镜共用）
# ================================================================
TEAL, TRACE, PAD = (8, 36, 52), (20, 110, 130), (70, 225, 255)


@functools.lru_cache(maxsize=None)
def circuit_static():
    img = np.full((H, W, 3), TEAL, np.uint8)
    for i in range(40):
        x, y = int(L.hash01(i, 1) * W), int(L.hash01(i, 2) * H)
        x2 = int(x + (L.hash01(i, 3) - .5) * 300); y2 = int(y + (L.hash01(i, 4) - .5) * 200)
        cv2.polylines(img, [np.int32([(x, y), (x2, y), (x2, y2)])], False, TRACE, 2)
        cv2.circle(img, (x, y), 4, TRACE, -1); cv2.circle(img, (x2, y2), 3, TRACE, -1)
    return img


def circuit(t, lit=1.0):
    img = circuit_static().copy()
    for i in range(40):                                        # 电流光点沿走线跑
        if L.hash01(i, 9) > lit: continue
        x, y = int(L.hash01(i, 1) * W), int(L.hash01(i, 2) * H)
        x2 = int(x + (L.hash01(i, 3) - .5) * 300); y2 = int(y + (L.hash01(i, 4) - .5) * 200)
        u = (t * 1.8 + L.hash01(i, 5)) % 1
        L1, L2 = abs(x2 - x), abs(y2 - y)
        d = u * (L1 + L2)
        px, py = (x + np.sign(x2 - x) * d, y) if d < L1 else (x2, y + np.sign(y2 - y) * (d - L1))
        cv2.circle(img, (int(px), int(py)), 3, PAD, -1)
    return img


def tblock(img, cx, y, s=1.0):
    """一块 Transformer 积木：四层（ATTN / ADD&NORM / MLP / ADD&NORM）。(cx, y) = 顶边中点。"""
    w = int(210 * s); rows = [('MULTI-HEAD ATTN', ORANGE, 30), ('ADD & NORM', YEL, 16), ('MLP', PAD, 30), ('ADD & NORM', YEL, 16)]
    x0 = int(cx - w / 2)
    hh = int(sum(r[2] for r in rows) * s) + 8
    box(img, x0, y, x0 + w, y + hh, (30, 34, 60), 3)
    yy = y + 4
    for name, c, h in rows:
        h = int(h * s)
        box(img, x0 + 6, yy, x0 + w - 6, yy + h - 2, c, 0)
        if s > .55 and h > 10: img = text_c(img, cx, yy + (h - 18) // 2, name, INK)
        yy += h
    return img, hh


def scene_block(t):
    img = circuit(t)
    u = (t - 109.88) / .10
    y = int(-120 + 190 * min(u, 1)) if u < 1 else 70
    sq = 0
    if u >= 1:                                                 # 落地挤压 + 震
        dt = t - 109.98
        sq = int(10 * np.exp(-dt * 20) * np.cos(dt * 60))
    img, hh = tblock(img, 320, y + sq, 1.0 + (.06 if sq > 3 else 0))
    if u >= 1 and t - 109.98 < .2:
        for k in range(10):
            a = np.pi * (k / 9)
            r = 30 + (t - 109.98) * 600
            star4(img, 320 + np.cos(a) * (120 + r), 70 + hh + 4 - np.sin(a) * r * .3, 3, PAPER)
    return img


def scene_tower(t):
    img = circuit(t * 3)
    u = t - 110.44
    off = 900 * u ** 1.6                                       # 镜头往下坠：塔往上走，越坠越快
    for k in range(-1, 12):
        y = int(40 + k * 150 - off % 150)
        img, _ = tblock(img, 320, y, .9)
        n = int(off // 150) + k
        img = text(img, 470, y + 30, f'LAYER {n + 1:,}', PAPER)
    for k in range(10):                                        # 竖向速度线
        x = int(L.hash01(k, f24(t) // 2) * W)
        if abs(x - 320) > 110: img[:, x:x + 2] = (40, 140, 160)
    layer = [12, 96, 768, 9999][min(3, int(u / (BEAT * 1.7)))]
    box(img, 20, 30, 190, 58, INK, 2)
    return text(img, 28, 34, f'DEPTH {layer:,}', YEL)


def scene_turtles(t):
    img = circuit(t)
    ys, y, s = [], 206, .86                                    # 塔：从她手上往上一块叠一块，越高越小，汇到天上
    while s > .06 and y > -40:
        hh = int(sum((30, 16, 30, 16)) * s) + 8
        y -= hh
        ys.append((y, s))
        s *= .78
    for y, s in reversed(ys):
        img, _ = tblock(img, 320, y, s)
    shake = 1 if f24(t) % 4 < 2 else -1
    img = put(img, q3(0, 120), 320 + shake, 284)
    for k in range(3):                                         # 汗珠
        if (f12(t) + k) % 3 == 0: star4(img, 262 + k * 50, 180 - k * 6, 2, PAPER)
    return img


def scene_sysprompt(t):
    img = np.full((H, W, 3), (10, 10, 20), np.uint8)
    img = E.window(img, 70, 50, 500, 200, 'system_prompt.txt', body=(6, 8, 16))
    img = text(img, 90, 90, 'SYSTEM:', (120, 220, 140), 2)
    s = 'You will obey.'
    n = int(max(0, t - 113.40) * 30)
    shown = s[:n]
    img = text(img, 90, 140, shown, PAPER, 2)
    if f24(t) % 12 < 6:
        cx = 90 + int(L.font(16).getlength(shown)) * 2
        img[140:172, cx:cx + 12] = PAPER
    if t > 113.95 and f24(t) % 4 < 2:                          # 快被揍了：obey 那个词开始抖
        img = np.roll(img, 3, axis=1)
    return img


# ================================================================
# 11 chinchilla / dense
# ================================================================
CHIN_A, CHIN_B = '素材/CG/p3_11_龙猫_a.png', '素材/CG/p3_11_龙猫_b.png'


def scene_chin_cg(t):
    """v2（用户：龙猫也画图）：CG 龙猫按拍啃一口（整张上弹 + 草屑），角标 TOKENS/PARAM 20。"""
    t = float(np.clip(t, 115.22, 116.12 - 1e-3))
    img = E.cg(CHIN_A).copy()
    dt = (t - 115.22) % BEAT
    img = E.lift(img, E.bounce(dt, 4))
    for i in range(10):                                        # 每拍从嘴边崩出来的草屑
        a = -np.pi / 2 + (L.hash01(i, int((t - 115.22) / BEAT)) - .5) * 2.4
        r = 20 + dt * 260
        x, y = 424 + np.cos(a) * r, 100 + np.sin(a) * r + 300 * dt * dt     # 嘴在 (424, 100)
        cv2.line(img, (int(x), int(y)), (int(x + 6 * np.cos(a * 3)), int(y + 5)), (230, 190, 80), 2)
    box(img, 20, 30, 250, 60, INK, 2)
    return text(img, 28, 36, 'TOKENS/PARAM: 20', YEL)


def scene_dense_cg(t):
    """v2：压成超高密度发光方块的 CG，光芒随拍闪白，计数飙到 2,000。"""
    t = float(np.clip(t, 116.12, 116.88 - 1e-3))
    img = E.cg(CHIN_B).copy()
    if t - 116.12 < .1: img = E.dither_to(img, PAPER, .6 * (1 - (t - 116.12) / .1))
    ph = ((t - L.B132_0) / BEAT) % 1
    m = np.zeros((H, W), bool)
    cv2.circle(m.view(np.uint8), (430, 255), int(40 + 30 * ph), 1, -1)    # 方块在 (430, 255)
    E.dither_to(img, (255, 244, 200), .5 * (1 - ph), m)
    v = int(20 * 10 ** (2 * L.smooth((t - 116.12) / .5)))
    box(img, 20, 30, 250, 60, INK, 2)
    return text(img, 28, 36, f'TOKENS/PARAM: {v:,}', RED if v > 20 else YEL)


# ================================================================
# 12 fences：一个音节撞碎一道围栏
# ================================================================
FENCES = [(116.88, 'EVALS'), (117.30, 'RLHF'), (117.62, 'RSP'), (117.90, 'CONST.'), (118.20, 'AIR GAP')]
RUN_V, RUN_X = 520, 250


TRACK, TRACK_DK, LANE = (200, 80, 60), (170, 62, 48), (246, 241, 228)


def hurdle(img, fx, name):
    """跨栏：黑白条纹横杆 + 两条 L 形腿 + 挂在杆上的名牌。fx = 左腿 x。"""
    for px in (0, 74):                                         # 腿（底脚往后伸）
        box(img, fx + px, 196, fx + px + 6, 262, (200, 204, 216), 2)
        box(img, fx + px - 14, 258, fx + px + 6, 264, (200, 204, 216), 2)
    box(img, fx - 6, 190, fx + 86, 204, PAPER, 2)
    for k in range(0, 92, 16): img[190:204, fx - 6 + k:fx + 2 + k] = INK
    box(img, fx + 10, 206, fx + 70, 226, (255, 210, 63), 2)    # 名牌
    return text_c(img, fx + 40, 207, name, INK)


def scene_fences(t):
    img = np.zeros((H, W, 3), np.uint8)
    for k, c in enumerate([(110, 186, 255), (136, 200, 255), (164, 216, 255), (190, 230, 255)]): img[k * 40:(k + 1) * 40] = c
    img[160:] = (190, 230, 255)                                # 10-19 修：天空原来只画到 180，远山缝里露出黑底（用户：黑色的东西）
    cam = RUN_V * (t - 116.88)
    for i in range(4):                                         # 云（视差 0.1）
        x = int((i * 190 - cam * .1) % (W + 120)) - 60; y = 30 + (i % 2) * 22
        for dx, r in ((0, 12), (14, 16), (30, 11)): cv2.circle(img, (x + dx, y), r, PAPER, -1)
    # 看台（视差 0.4）：三层台阶 + 挥手的观众（Clawd 小方块）+ 顶棚
    img[116:120] = (60, 64, 88)
    for row in range(3):
        y0 = 124 + row * 22
        img[y0:y0 + 22] = [(96, 100, 128), (84, 88, 116), (72, 76, 104)][row]
        img[y0 + 20:y0 + 22] = (50, 54, 76)
        off = int(cam * .4) % 18
        for x in range(-off, W, 18):
            k = (x + int(cam * .4)) // 18 * 7 + row
            col = [(217, 119, 87), (255, 210, 63), (120, 200, 255), (255, 120, 170), PAPER][int(L.hash01(k, 1) * 5)]
            wave = 3 if (f12(t) + k) % 4 < 2 else 0
            box(img, x + 3, y0 + 4 - wave, x + 15, y0 + 18 - wave, col, 1)
            img[y0 + 9 - wave:y0 + 11 - wave, x + 6:x + 8] = INK; img[y0 + 9 - wave:y0 + 11 - wave, x + 11:x + 13] = INK
    for x in range(-(int(cam * .4) % 120), W, 120):           # 顶棚支柱
        img[96:124, x:x + 4] = (60, 64, 88)
    img[90:98] = (230, 230, 240); img[96:98] = (150, 150, 170)
    img[190:200] = (80, 170, 80)                               # 内场草皮的一条边
    img[200:] = TRACK
    for y in (214, 238, 262):                                  # 分道线
        img[y:y + 2] = LANE
    for x in range(-int(cam) % 40, W, 40): img[226:228, x:x + 14] = TRACK_DK
    for x in range(-int(cam) % 80, W, 80): img[250:252, x:x + 22] = TRACK_DK
    for ft, name in FENCES:
        fx = RUN_X + 40 + RUN_V * (ft - t)                     # ft 时正好撞到她面前
        dt = t - ft
        if dt < 0:
            if fx < W + 20: img = hurdle(img, int(fx), name)
        elif dt < .5:                                          # 横杆断成两截飞出去 + 名牌转着飞 + 名字闪一下
            for i in range(6):
                a = L.hash01(i, int(ft * 10)) * np.pi - np.pi
                x = fx + 40 + np.cos(a) * 500 * dt
                y = 200 + np.sin(a) * 300 * dt + 900 * dt * dt
                M = cv2.boxPoints(((x, y), (44 if i < 2 else 22, 10), dt * 900 + i * 40))
                poly(img, M, [PAPER, INK, (255, 210, 63)][i % 3], 1)
            if dt < .25:
                for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (3, 3)): img = text_c(img, RUN_X + 40 + ox, 120 + oy, name, INK, 3)
                img = text_c(img, RUN_X + 40, 120, name, YEL if f24(t) % 2 else PAPER, 3)
    frame = 1 + f24(t) // 3 % 2                                # 跑步两帧，12fps 以下交替（有限动画）
    bob = -4 if frame == 1 else 0
    img = put(img, q3(frame, 110), RUN_X, 262 + bob)
    for i in range(6):                                         # 脚下扬起的尘土
        age = (t * 4 + L.hash01(i, 151)) % 1
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(RUN_X - 30 - age * 120), int(258 - age * 18)), int(4 + age * 9), 1, -1)
        dither_to(img, (236, 200, 170), .5 * (1 - age), m > 0)
    for k in range(5):
        y = int(150 + L.hash01(k, f24(t) // 2) * 110)
        img[y:y + 2, max(0, RUN_X - 160 - k * 10):RUN_X - 60] = PAPER
    return img


# ================================================================
# 13 sheep：数 GPU 像数羊
# ================================================================
def gpu(img, x, y, s, t, k):
    S = lambda v: int(round(v * s))
    x, y = int(x), int(y)
    box(img, x, y, x + S(60), y + S(26), (40, 150, 70), 2)
    img[max(y + S(26), 0):max(y + S(30), 0), max(x + S(8), 0):max(x + S(40), 0)] = (230, 190, 60)   # 金手指（出界时别让负下标绕到画面底下）
    for fx in (x + S(16), x + S(44)):
        cv2.circle(img, (fx, y + S(13)), S(10), INK, -1)
        cv2.circle(img, (fx, y + S(13)), S(8), (60, 64, 80), -1)
        a = t * 30 + k
        for j in range(3):
            aa = a + j * 2.09
            cv2.line(img, (fx, y + S(13)), (int(fx + np.cos(aa) * S(7)), int(y + S(13) + np.sin(aa) * S(7))), (180, 186, 200), 1)


FENCE_X = 330
WALL, WALL_DOT, FLOOR_W, FLOOR_SEAM = (34, 38, 84), (46, 52, 104), (110, 74, 52), (84, 54, 38)
CURTAIN, CURTAIN_HI, CURTAIN_DK = (220, 110, 150), (246, 150, 186), (170, 70, 110)
HEAP_X0, HEAP_X1, HEAP_FLOOR, GS = 344, 636, 252, .5      # GPU 山：底宽、地面、GPU 缩放
GW, GH = int(60 * GS), int(30 * GS)


@functools.lru_cache(maxsize=None)
def heap_slots():
    """GPU 山的格位：一层层往上收（每层比下一层少一块、错半块，像砌砖），从下往上、同层随机顺序 → 每块落下时下面都已经有东西托着。"""
    slots, row = [], 0
    n0 = (HEAP_X1 - HEAP_X0) // GW
    while n0 - row > 0:
        n = n0 - row
        x0 = HEAP_X0 + row * GW / 2
        y = HEAP_FLOOR - (row + 1) * GH
        order = sorted(range(n), key=lambda k: L.hash01(row, k, 161))
        for k in order:
            slots.append((x0 + k * GW + (L.hash01(row, k, 162) - .5) * 3, y))
        row += 1
    return slots


def _room(t):
    img = np.full((H, W, 3), WALL, np.uint8)
    for y in range(8, 250, 16):                                               # 墙纸菱形点
        for x in range((y // 16 % 2) * 12, W, 24): img[y:y + 2, x:x + 2] = WALL_DOT
    img[246:252] = (70, 50, 40)                                               # 踢脚线
    img[252:] = FLOOR_W
    for y in range(252, H, 12):                                               # 木地板
        img[y:y + 1] = FLOOR_SEAM
        for x in range((y // 12 % 2) * 40, W, 80): img[y:y + 12, x:x + 1] = FLOOR_SEAM
    # 窗：窗框十字 + 月亮 + 星星 + 两边窗帘（三档褶子）
    box(img, 466, 34, 594, 150, (20, 26, 70), 3)
    for i in range(8):
        x, y = int(472 + L.hash01(i, 171) * 116), int(40 + L.hash01(i, 172) * 100)
        if (f12(t) + i) % 5: img[y, x] = PAPER
    cv2.circle(img, (540, 76), 18, (255, 240, 170), -1); cv2.circle(img, (548, 70), 16, (20, 26, 70), -1)
    img[90:94, 466:594] = (150, 110, 80); img[34:150, 528:532] = (150, 110, 80)
    for side in (0, 1):
        x0 = 448 if side == 0 else 586
        for k in range(4):
            c = [CURTAIN_HI, CURTAIN, CURTAIN_DK, CURTAIN][k]
            img[26:160, x0 + k * 7:x0 + k * 7 + 7] = c
        img[150:156, x0:x0 + 28] = CURTAIN_DK                                  # 窗帘绑带
    img[22:28, 440:620] = (150, 110, 80)                                      # 窗帘杆
    # 床头墙上的小架子：书 + 闹钟（秒针在走）
    img[120:124, 30:170] = (150, 110, 80)
    for k, c in enumerate([(200, 60, 70), (70, 140, 220), (240, 200, 80), (90, 180, 110), (200, 120, 200)]):
        hh = 22 + (k % 2) * 6
        box(img, 36 + k * 12, 120 - hh, 46 + k * 12, 120, c, 1)
    cv2.circle(img, (140, 106), 12, INK, -1); cv2.circle(img, (140, 106), 10, PAPER, -1)
    a = t * 6
    cv2.line(img, (140, 106), (int(140 + np.cos(a) * 8), int(106 + np.sin(a) * 8)), RED, 1)
    cv2.line(img, (140, 106), (140, 99), INK, 2)
    img[94:97, 130:134] = INK; img[94:97, 146:150] = INK
    # 吊灯光晕
    cv2.line(img, (236, 0), (236, 40), (60, 60, 80), 2)
    poly(img, [(222, 40), (250, 40), (258, 54), (214, 54)], (240, 200, 120), 2)
    m = np.zeros((H, W), np.uint8); cv2.ellipse(m, (236, 120), (70, 80), 0, 0, 360, 1, -1)
    dither_to(img, (60, 62, 120), .3, m > 0)
    # 床：床头板（斜面）+ 床垫 + 格子被子 + 枕头
    box(img, 16, 176, 54, 264, (170, 110, 70), 2)
    img[178:262, 18:22] = (210, 150, 100); img[178:262, 48:52] = (120, 76, 50)
    cv2.circle(img, (35, 190), 6, (210, 150, 100), -1)
    box(img, 54, 226, 232, 262, (236, 236, 250), 2)
    for k in range(0, 160, 16):
        for j in range(0, 22, 11):
            if (k // 16 + j // 11) % 2: img[238 + j:249 + j, 70 + k:86 + k] = (240, 160, 190)
    img[236:238, 66:232] = (200, 200, 220)
    box(img, 58, 212, 108, 230, PAPER, 2); img[224:230, 60:106] = (210, 210, 230)
    box(img, 228, 256, 236, 270, (170, 110, 70), 2)                           # 床脚
    # 地毯（栅栏下面）
    cv2.ellipse(img, (FENCE_X + 4, 266), (90, 12), 0, 0, 360, (140, 60, 90), -1)
    cv2.ellipse(img, (FENCE_X + 4, 266), (80, 8), 0, 0, 360, (190, 90, 120), 2)
    return img


def scene_sheep(t):
    img = _room(t)
    for px in (FENCE_X - 30, FENCE_X + 30):                                   # 小栅栏
        box(img, px, 200, px + 8, 262, (150, 100, 50), 2)
    for py in (210, 230): box(img, FENCE_X - 40, py, FENCE_X + 48, py + 8, (200, 150, 80), 2)
    sat = t >= 120.30
    img = put(img, q3(4 if sat else 3, 96 if sat else 80), 130, 262 if sat else 250)
    fast = t >= 119.60
    period = BEAT
    t0 = 118.74
    if not fast:
        n_jump = int((t - t0) / period) + 1
        for j in range(max(0, n_jump - 3), n_jump):                           # 一拍跳过去一块（像数羊）
            st = t0 + j * period
            u = (t - st) / .5
            if not (0 <= u < 1): continue
            gpu(img, 200 + 280 * u, 200 - 70 * np.sin(np.pi * u), .9, t, j)
    # GPU 山：从天花板掉下来、落到格位上（重力 + 落地弹一下），从下往上砌，越掉越密
    slots = heap_slots()
    n = len(slots)
    for i, (sx, sy) in enumerate(slots):
        ti = 119.60 + 1.10 * (i / n) ** .85
        if t < ti: break
        u = t - ti
        fall_t = np.sqrt((sy + 40) / 1300)                                    # y = -40 + 1300u²（g = 2600）
        if u < fall_t: y = -40 + 1300 * u * u
        else:
            b = u - fall_t
            y = sy - (6 * np.sin(np.pi * b / .08) if b < .08 else 0)
        gpu(img, sx, y, GS, t if u < fall_t + .3 else 0, i)
    if fast and t < 119.60 + 1.1:                                              # 还在往下掉的密集「GPU 雨」
        for i in range(10):
            x = HEAP_X0 + L.hash01(i, f12(t) // 2, 181) * (HEAP_X1 - HEAP_X0 - GW)
            y = (L.hash01(i, 182) * 200 + (t - 119.6) * 900) % 220 - 30
            gpu(img, x, y, GS, t, i + 100)
    count = min(int((t - t0) / period) + 1, 3) if not fast else int(min(100000, 4 * 10 ** (4.4 * min(1, (t - 119.60) / .4))))
    box(img, 250, 40, 450, 76, PAPER, 3)
    poly(img, [(300, 76), (290, 96), (320, 76)], PAPER, 0)
    return text_c(img, 350, 46, f'GPU {count:,}', INK, 2)


SCENES = [
    ('block', 109.88, 110.44, scene_block),
    ('tower', 110.44, 112.02, scene_tower),
    ('turtles', 112.02, 113.34, scene_turtles),
    ('sysprompt', 113.34, 114.20, scene_sysprompt),
    ('chin_cg', 115.22, 116.12, scene_chin_cg),
    ('dense_cg', 116.12, 116.88, scene_dense_cg),
    ('fences', 116.88, 118.74, scene_fences),
    ('sheep', 118.74, 120.90, scene_sheep),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
