"""codeshots_d: 第二部分主歌三的四个代码镜头（分镜第 9、11、12、13 镜）。

    safety     68.58–72.80  安全审查：戴安全帽的 Clawd 竖拇指，身后机柜起火，一只慢半拍回头看见
    blueprint  77.28–80.92  冯·诺依曼蓝图被 Q 版 Claude 盖 OBSOLETE 章 → 卷起扔进回收站
    racing     80.92–84.68  OutRun 式伪 3D 赛车：SHARP LEFT TURN 漂移 → 地平线升起 AI 的眼睛
    cdr        84.68–89.12  cons 格子小火车，cdr 车厢脱钩弹飞 → 只剩 NIL

接口遵守 part1_lib 文件头的约定：scene_xxx(t) -> (360,640,3) uint8 RGB，t 是完整版歌曲秒数，纯函数，
越界的 t 钳到本镜头范围内。
自检：python codeshots_d.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np

import part1_lib as L

W, H = L.W, L.H
BEAT = L.BEAT132
INK = (12, 11, 26)
PAPER = (246, 241, 228)
ORANGE = (217, 119, 87)          # Clawd 橙（= pc98ify 固定色）
YEL = (255, 210, 63)             # 警示黄（= 固定色）
BLUE = (34, 52, 150)             # 蓝图底（= 固定色）
RED = (214, 36, 44)              # 印章红 / 车身红
CYAN = (60, 230, 255)            # AI 眼睛青
SPRITE = '素材/精灵/q版Claude_四姿势2.png'


# ================================================================
# 公共小工具
# ================================================================
def beat_time(n):
    return L.B132_0 + n * BEAT


def frame24(t):
    return int(np.floor(t * 24 + 1e-6))


def vhash(*k):
    """hash01 的向量版（numpy 数组进、数组出）。"""
    x = sum((np.asarray(v, np.float64) + 1.0) * c for v, c in zip(k, (12.9898, 78.233, 37.719, 4.581)))
    x = np.sin(x) * 43758.5453
    return x - np.floor(x)


def box(img, x0, y0, x1, y1, col, ink=2):
    """带墨线的平涂矩形（x1,y1 不含）。"""
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    if ink:
        cv2.rectangle(img, (x0 - ink, y0 - ink), (x1 - 1 + ink, y1 - 1 + ink), INK, -1)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), tuple(int(c) for c in col), -1)


def poly(img, pts, col, ink=2):
    pts = np.round(np.asarray(pts, np.float64)).astype(np.int32)
    if ink:
        cv2.polylines(img, [pts], True, INK, ink * 2, lineType=cv2.LINE_8)
    cv2.fillPoly(img, [pts], tuple(int(c) for c in col), lineType=cv2.LINE_8)


def text_w(s, scale=1):
    l, t_, r, b = L.font(16).getbbox(s)
    return (r - l + 2) * scale


def text_ol(img, xy, s, col, scale=1, ol=INK, d=None):
    """描边像素字。返回新图。"""
    d = d or (2 if scale >= 2 else 1)
    x, y = int(round(xy[0])), int(round(xy[1]))
    if ol is not None:
        for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d), (-d, -d), (d, d), (-d, d), (d, -d)):
            img = L.draw_text(img, (x + dx, y + dy), s, ol, scale=scale)
    return L.draw_text(img, (x, y), s, col, scale=scale)


def shift_img(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def rot_rgba(rgba, ang, pivot=None, scale=1.0):
    """绕 pivot 旋转缩放 RGBA（最近邻硬边）。返回 (新图, pivot 在新图里的位置)。ang 为逆时针度数。"""
    h, w = rgba.shape[:2]
    px, py = pivot if pivot is not None else (w / 2, h / 2)
    R = int(np.ceil(max(np.hypot(px - cx, py - cy) for cx in (0, w) for cy in (0, h)) * scale)) + 2
    M = cv2.getRotationMatrix2D((float(px), float(py)), float(ang), float(scale))
    M[0, 2] += R - px
    M[1, 2] += R - py
    out = cv2.warpAffine(rgba, M, (2 * R, 2 * R), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))
    return out, (R, R)


def paste_at(dst, rgba, pivot_xy, pivot_in):
    """把 rgba 贴到 dst，使 rgba 里的 pivot_in 落在 dst 的 pivot_xy。"""
    return L.paste_rgba(dst, rgba, pivot_xy[0] - pivot_in[0], pivot_xy[1] - pivot_in[1])


def rgba_canvas(w, h):
    return np.zeros((h, w, 4), np.uint8)


def draw_rgba(canvas, fn):
    """在 RGBA 画布上用 RGB 画法画东西：fn(rgb) 画在一张哨兵色底图上，变化过的像素写回并置不透明。"""
    key = np.array((1, 2, 3), np.uint8)
    rgb = np.empty(canvas.shape[:2] + (3,), np.uint8)
    rgb[:] = key
    fn(rgb)
    m = np.any(rgb != key, -1)
    canvas[m, :3] = rgb[m]
    canvas[m, 3] = 255
    return canvas


# ---------------- Q 版 Claude 精灵（四姿势横排，按连通域拆） ----------------
@functools.lru_cache(maxsize=None)
def pose(k):
    """第 k 个姿势（1–4）的 RGBA 裁切。姿势 3、4 的头发贴得很近，所以按连通域质心分组而不是按列切。"""
    a = L.load_rgba(SPRITE)
    al = (a[..., 3] > 127).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(al, 8)
    lo, hi = {1: (0, 462), 2: (462, 880), 3: (880, 1282), 4: (1282, 9999)}[k]
    keep = [i for i in range(1, n) if lo <= cen[i][0] < hi and st[i, 4] > 20]
    m = np.isin(lab, keep)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = a[y0:y1, x0:x1].copy()
    crop[..., 3] = np.where(m[y0:y1, x0:x1], 255, 0)
    return crop


@functools.lru_cache(maxsize=None)
def pose_scaled(k, h_px, flip=False):
    """按目标高度缩放（面积插值 + alpha 二值化），可水平翻转。"""
    p = pose(k)
    s = h_px / p.shape[0]
    out = cv2.resize(p, (max(1, int(round(p.shape[1] * s))), int(h_px)), interpolation=cv2.INTER_AREA)
    out[..., 3] = np.where(out[..., 3] > 140, 255, 0)
    if flip: out = out[:, ::-1].copy()
    return out


# ---------------- 像素小图 ----------------
@functools.lru_cache(maxsize=None)
def pixmap(rows, colors, scale):
    """字符点阵 → RGBA（每字符一色，'.' 透明）。colors: ((字符, (r,g,b)), ...)。"""
    cm = dict(colors)
    h, w = len(rows), len(rows[0])
    out = np.zeros((h, w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != '.':
                out[y, x] = tuple(cm[c]) + (255,)
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def star_pts(cx, cy, r_out, r_in, n=5, rot=-90):
    a = np.radians(rot) + np.arange(2 * n) * np.pi / n
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


# ================================================================
# 第 9 镜：安全审查（68.58–72.80）
# ================================================================
SAFE_T0, SAFE_T1 = 68.58, 72.80
T_SAFE, T_ENOUGH, T_RECKON = 70.12, 70.44, 71.10
CHECK_TS = (69.35, 69.81, T_SAFE)           # 白板三个勾：两拍 + safe
T_THUMBS = 69.35                             # 第一个勾起大家开始竖拇指
T_LOOK0, T_LOOK1 = 70.62, 70.82              # 慢半拍的那只：侧脸 → 背影 → 71.10 转回来惊恐
T_DISCO0, T_DISCO1 = 72.25, 72.72            # 火光转彩灯
CLAWD_SH = (170, 82, 62)
HAT_SH = (206, 150, 30)
WALL, WALL_LN = (52, 50, 66), (40, 38, 52)
FLOOR, FLOOR_LN = (96, 92, 104), (78, 74, 88)
RACK, RACK_U = (36, 38, 54), (62, 66, 90)
FIRE = ((226, 52, 34), (255, 132, 40), (255, 222, 90))     # 外焰红 / 中焰橙 / 焰心黄
DISCO = ((255, 64, 200), (64, 220, 255), (150, 255, 80), (255, 230, 60), (160, 100, 255))
RACKS = ((16, 150), (490, 624))
RACK_Y0, RACK_Y1 = 74, 196
BOARD = (168, 26, 472, 176)
CLAWD_X = [64 + 102 * i for i in range(6)]
CLAWD_FEET = 272
TURNER = 4
BOARD_ROWS = ('ALIGNMENT', 'EVALS', 'VIBES')


def fire_level(t):
    return (0.45 + 0.35 * L.smooth((t - SAFE_T0) / 2.0)
            + 0.30 * L.smooth((t - T_RECKON) / 0.25))


def disco_k(t):
    return float(np.clip((t - T_DISCO0) / (T_DISCO1 - T_DISCO0), 0, 1))


def draw_fire(img, x0, x1, ybase, hmax, t, seed):
    """像素火焰：4px 格子上的解析高度场（12fps），三层平涂 + 墨线。返回 (img, 火焰层号图) 供彩灯替换。"""
    f = int(np.floor(t * 12))
    cs = 4
    cx = np.arange(x0 // cs, x1 // cs)
    u = (cx * cs - x0) / max(x1 - x0, 1)
    env = np.sqrt(np.clip(np.sin(np.pi * np.clip(u, 0, 1)), 0, 1))
    hh = hmax * env * (0.55 + 0.25 * np.sin(cx * 0.9 + f * 0.7 + seed)
                       + 0.20 * vhash(cx, f, seed))                       # 每列火舌高度
    ny = int(np.ceil(hmax * 1.3 / cs))
    cy = np.arange(ny)                                                    # 从 ybase 往上第几格
    rel = (cy[:, None] * cs) / np.maximum(hh[None, :], 1)
    v = 1 - rel + 0.28 * (vhash(cx[None, :], cy[:, None], f, seed) - 0.5)
    lev = np.where(v > 0.68, 3, np.where(v > 0.38, 2, np.where(v > 0.02, 1, 0))).astype(np.uint8)
    lev = lev[::-1]                                                       # 行 0 = 最上面
    big = cv2.resize(lev, None, fx=cs, fy=cs, interpolation=cv2.INTER_NEAREST)
    gy0 = ybase - big.shape[0]
    full = np.zeros((H, W), np.uint8)
    ya, yb = max(gy0, 0), min(ybase, H)
    xa, xb = x0 // cs * cs, x0 // cs * cs + big.shape[1]
    full[ya:yb, xa:xb] = big[ya - gy0:yb - gy0]
    m = full > 0
    outline = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    img[outline & ~m] = INK
    for k in (1, 2, 3):
        img[full == k] = FIRE[k - 1]
    return img, full


def draw_rack(img, x0, x1, t, idx, lvl):
    box(img, x0, RACK_Y0, x1, RACK_Y1 + 20, RACK, 3)
    f = int(np.floor(t * 12))
    for j, y in enumerate(range(RACK_Y0 + 6, RACK_Y1 + 10, 14)):
        burning = L.hash01(idx, j, 3) < (lvl - 0.35) * 0.9
        if burning:
            c = FIRE[1] if L.hash01(idx, j, f // 2) < 0.6 else FIRE[2]
            box(img, x0 + 8, y, x1 - 8, y + 9, c, 1)
        else:
            box(img, x0 + 8, y, x1 - 8, y + 9, RACK_U, 1)
            for k in range(4):                                   # 小指示灯
                on = L.hash01(idx, j, k, f // 3) < 0.55
                col = (80, 255, 110) if (j + k) % 3 else (255, 70, 60)
                if on:
                    img[y + 3:y + 6, x1 - 16 - k * 7:x1 - 12 - k * 7] = col


def draw_whiteboard(img, t):
    x0, y0, x1, y1 = BOARD
    box(img, x0 - 5, y0 - 5, x1 + 5, y1 + 5, (150, 150, 162), 3)
    box(img, x0, y0, x1, y1, PAPER, 2)
    img[y1 + 5:y1 + 11, x0 + 20:x1 - 20] = (120, 120, 134)           # 笔槽
    img[y1 + 11:y1 + 13, x0 + 20:x1 - 20] = INK
    s = 'SAFETY REVIEW'
    tx = (x0 + x1) // 2 - text_w(s, 2) // 2
    img = L.draw_text(img, (tx, y0 + 6), s, INK, scale=2)
    img[y0 + 40:y0 + 43, tx:tx + text_w(s, 2) - 4] = (214, 36, 44)   # 红色下划线
    for i, row in enumerate(BOARD_ROWS):
        ry = y0 + 52 + i * 32
        img = L.draw_text(img, (x0 + 22, ry), row, BLUE, scale=2)
        bx, by = x1 - 62, ry + 4
        box(img, bx, by, bx + 24, by + 24, PAPER, 2)
        tc = CHECK_TS[i]
        if t >= tc:
            k = L.ease_out_back((t - tc) / 0.12)
            p = np.array([(bx + 3, by + 12), (bx + 11, by + 22), (bx + 30, by - 6)], np.float64)
            c = np.array((bx + 12, by + 12))
            p = c + (p - c) * k
            pi = np.round(p).astype(np.int32)
            cv2.polylines(img, [pi], False, INK, 11, lineType=cv2.LINE_8)
            cv2.polylines(img, [pi], False, (40, 168, 72), 6, lineType=cv2.LINE_8)
    return img


def draw_clawd(img, cx, feet, arm=0.0, face='dot', dy=0, dx=0, left_up=False, blink=False):
    """代码版 Clawd：宽方块身体 + 四条小腿 + 侧边小短手 + 点点眼 + 黄色安全帽。
    arm: 右手（画面右）举起程度 0–1，≥0.2 时画成竖大拇指。face: dot/side/back/shock。"""
    bw, bh = 72, 42
    cx = int(round(cx + dx))
    y1 = int(round(feet - 14 + dy))
    y0 = y1 - bh
    x0, x1 = cx - bw // 2, cx + bw // 2
    # 腿（跟着身体压一点点）
    for lx in (x0 + 6, x0 + 22, x1 - 29, x1 - 13):
        box(img, lx, y1, lx + 7, feet, CLAWD_SH, 2)
    # 左手（画面左）：平放小短手，或举起张开
    if left_up:
        box(img, x0 - 10, y0 - 10, x0 - 2, y0 + 20, ORANGE, 2)
        box(img, x0 - 14, y0 - 20, x0 + 2, y0 - 10, ORANGE, 2)
    else:
        box(img, x0 - 11, y0 + 14, x0, y0 + 22, ORANGE, 2)
    # 右手
    if arm < 0.2:
        box(img, x1, y0 + 14, x1 + 11, y0 + 22, ORANGE, 2)
    else:
        top = int(round(y0 + 14 - (10 + 30 * arm)))
        box(img, x1 + 2, top, x1 + 12, y0 + 22, ORANGE, 2)            # 举起的小臂
        box(img, x1 - 2, top - 15, x1 + 18, top + 1, ORANGE, 2)       # 拳头
        box(img, x1 - 1, top - 29, x1 + 8, top - 13, ORANGE, 2)       # 竖起的大拇指
        img[top - 8:top - 6, x1 + 3:x1 + 16] = CLAWD_SH               # 指缝
    # 身体
    box(img, x0, y0, x1, y1, ORANGE, 3)
    img[y1 - 7:y1, x0:x1] = CLAWD_SH
    img[y0 + 3:y0 + 5, x0 + 4:x0 + 18] = (240, 160, 124)              # 一小道高光
    ey = y0 + 12
    if face == 'dot':
        if blink:
            img[ey + 5:ey + 8, cx - 18:cx - 11] = INK
            img[ey + 5:ey + 8, cx + 11:cx + 18] = INK
        else:
            img[ey:ey + 11, cx - 17:cx - 11] = INK
            img[ey:ey + 11, cx + 11:cx + 17] = INK
    elif face == 'side':
        img[ey:ey + 11, cx + 16:cx + 21] = INK
        img[ey:ey + 11, cx + 30:cx + 34] = INK
    elif face == 'shock':
        for ex in (cx - 22, cx + 8):
            box(img, ex, ey - 4, ex + 14, ey + 12, PAPER, 2)
            img[ey + 2:ey + 6, ex + 5:ex + 9] = INK
        cv2.ellipse(img, (cx, y0 + 31), (7, 5), 0, 0, 360, INK, -1)
        for k in range(3):                                            # 额头「青筋」阴影线
            img[y0 + 1:y0 + 9, cx - 12 + k * 8:cx - 10 + k * 8] = (80, 96, 210)
    # 安全帽
    cv2.ellipse(img, (cx, y0 + 1), (34, 22), 0, 180, 360, INK, -1)
    cv2.ellipse(img, (cx, y0 + 1), (31, 19), 0, 180, 360, YEL, -1)
    cv2.ellipse(img, (cx + 6, y0 + 1), (24, 19), 0, 270, 360, HAT_SH, -1)
    img[y0 - 18:y0 - 1, cx - 3:cx + 3] = HAT_SH if face != 'back' else YEL   # 帽子中间的棱
    box(img, cx - 40, y0 - 3, cx + 34, y0 + 3, YEL, 2)                  # 帽檐（右边让给举起的手臂）
    if face == 'back':
        img[y0 + 6:y0 + 10, cx - 20:cx + 20] = INK                     # 帽子后面的松紧带
    return img


def draw_sweat(img, cx, top, tt):
    for k, (ox, oy) in enumerate(((-46, -6), (44, -12), (52, 14))):
        ph = ((tt * 3 + k * 0.33) % 1.0)
        x = int(cx + ox + np.sign(ox) * 10 * ph)
        y = int(top + oy + 14 * ph)
        pts = np.array([(x, y - 8), (x - 5, y + 2), (x, y + 6), (x + 5, y + 2)])
        poly(img, pts, (120, 200, 255), 1)


def scene_safety(t):
    t = float(np.clip(t, SAFE_T0, SAFE_T1))
    tc = L.twos(t)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = WALL
    for x in range(40, W, 80):
        img[18:196, x:x + 2] = WALL_LN
    # 地面 + 警示条
    img[212:] = FLOOR
    img[240:242] = FLOOR_LN
    yy, xx = np.mgrid[196:212, 0:W]
    img[196:212] = np.where((((xx + yy) // 12) % 2 == 0)[..., None], np.array(YEL, np.uint8), np.array(INK, np.uint8))
    img[195] = INK
    img[212] = INK

    lvl = fire_level(t)
    for i, (x0, x1) in enumerate(RACKS):
        draw_rack(img, x0, x1, t, i, lvl)
    # 讽刺标牌
    box(img, 506, 150, 608, 168, YEL, 2)
    img = L.draw_text(img, (512, 151), 'NO FLAMES', INK)
    img = draw_whiteboard(img, t)

    fires = np.zeros((H, W), np.uint8)
    for i, (x0, x1) in enumerate(RACKS):
        img, fl = draw_fire(img, x0 - 10, x1 + 10, RACK_Y0 + 36, 104 * lvl, t, 3 + i * 5)
        fires = np.maximum(fires, fl)
    # 火星：每个机柜 14 颗，循环上升（出生时间写成解析式）
    for i, (x0, x1) in enumerate(RACKS):
        for j in range(14):
            per = 0.7 + 0.4 * L.hash01(i, j, 1)
            age = ((t - SAFE_T0) + per * L.hash01(i, j, 2)) % per
            x = x0 + (x1 - x0) * L.hash01(i, j, 3) + 10 * np.sin(age * 6 + j)
            y = RACK_Y0 + 20 - age * 150 * lvl
            if y > 18:
                c = FIRE[2] if j % 2 else FIRE[1]
                img[int(y):int(y) + 3, int(x):int(x) + 3] = c
                fires[int(y):int(y) + 3, int(x):int(x) + 3] = 3

    # 火光 → 彩灯（Bayer 网点替换火焰颜色，地板铺成彩色方格）
    dk = disco_k(t)
    if dk > 0:
        m = L.bayer_mask(dk)
        f6 = int(np.floor(t * 12)) // 2
        dis = img.copy()
        for k in (1, 2, 3):
            dis[fires == k] = DISCO[(k + f6) % 5]
        ty, tx = np.mgrid[213:H, 0:W]
        tile = ((tx // 64) + (ty - 213) // 24 + f6) % 5
        pal = np.array(DISCO, np.uint8)
        dis[213:] = pal[tile]
        dis[213:][((tx % 64) < 2) | (((ty - 213) % 24) < 2)] = INK
        img = np.where(m[..., None], dis, img)

    # Clawd 一排（一拍二）
    n, ph = L.beat_phase(tc)
    pump = float(np.exp(-5 * ph))
    bob = int(round(2 * pump))
    for i, cx in enumerate(CLAWD_X):
        arm = 0.0 if tc < T_THUMBS else 0.45 + 0.55 * pump
        if T_SAFE <= tc < T_ENOUGH + 0.3: arm = 1.0                   # safe enough：定格举最高
        face, dy, dx, lu = 'dot', bob, 0, False
        blink = (i == 1 and (n % 4 == 2) and ph < 0.3)
        if i == TURNER:
            if T_LOOK0 <= tc < T_LOOK1:
                face = 'side'
                arm = 0.6
            elif T_LOOK1 <= tc < T_RECKON:
                face, arm = 'back', 0.0
            elif tc >= T_RECKON:
                face, arm, lu = 'shock', 0.9, True
                hop = np.sin(np.pi * np.clip((tc - T_RECKON) / 0.3, 0, 1))
                dy = -int(round(14 * hop))
                dx = (1 if frame24(t) % 2 else -1) * (1 if tc > T_RECKON + 0.3 else 0)
        img = draw_clawd(img, cx, CLAWD_FEET, arm, face, dy, dx, lu, blink)
        if i == TURNER and tc >= T_RECKON:
            top = CLAWD_FEET - 14 - 42 + dy
            draw_sweat(img, cx, top, tc)
            if tc < T_RECKON + 0.9:
                k = L.ease_out_back((tc - T_RECKON) / 0.15)
                sc = 3 if k > 0.6 else 2
                ey_ = int(top - 40 - 12 * k)
                box(img, cx + 30, ey_ - 26, cx + 40, ey_ - 6, (255, 60, 60), 2)   # 大「!」
                box(img, cx + 30, ey_ - 2, cx + 40, ey_ + 8, (255, 60, 60), 2)
    # safe / enough：拇指上闪一颗星
    for tk in (T_SAFE, T_ENOUGH):
        if 0 <= t - tk < 0.2:
            for i, cx in enumerate(CLAWD_X):
                if i == TURNER and t > T_LOOK0: continue
                sx, sy = cx + 36 + 3, CLAWD_FEET - 14 - 42 + 14 - 40 - 30
                r = 9 if t - tk < 0.1 else 5
                poly(img, star_pts(sx, sy, r, r * 0.35, 4, -90), PAPER, 1)
    return img


# ================================================================
# 第 11 镜：蓝图 + OBSOLETE 印章 + 回收站（77.28–80.92）
# ================================================================
BP_T0, BP_T1 = 77.28, 80.92
T_OBS = 79.82
T_REVEAL1 = 77.95                    # 线稿从左往右画出来
T_NEUMANN = 78.62
T_HOP_OUT0, T_HOP_OUT1 = 80.04, 80.28
T_ROLL0, T_ROLL1 = 80.12, 80.42
T_FLY1 = 80.66
SHEET = (16, 22, 624, 280)
DESK = (22, 26, 64)
GRID1, GRID2 = (54, 78, 174), (88, 118, 210)
BPW = (226, 236, 255)
SIGN0 = (516, 146)                   # 回收站图标中心 = 下一镜路牌出现的位置（匹配剪辑）
STAMP_C = (262, 128)                 # 印章中心
GIRL_X, GIRL_FEET, GIRL_H = 470, 278, 232
ENTER_BEATS = (134, 135, 136)        # 77.54 / 77.99 / 78.44 三跳进场

BIN_LID = ("......kkkk......",
           "......kllk......",
           ".kkkkkkkkkkkkkk.",
           "kllllllllllllllk",
           "kmmmmmmmmmmmmmmk",
           ".kkkkkkkkkkkkkk.")
BIN_BODY = tuple([".klmmdmmdmmdmmk."] * 11 + ["..kkkkkkkkkkkk.."])
BIN_COLS = (('k', INK), ('l', (214, 218, 228)), ('m', (160, 164, 182)), ('d', (104, 108, 128)))


def lw(img, p0, p1, w=2):
    cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), BPW, w, lineType=cv2.LINE_8)


def arrow(img, p0, p1, both=True):
    lw(img, p0, p1)
    for a, b in ((p0, p1), (p1, p0)) if both else ((p0, p1),):
        d = np.array(b, float) - np.array(a, float)
        d /= np.linalg.norm(d)
        n = np.array((-d[1], d[0]))
        tip = np.array(b, float)
        pts = np.array([tip, tip - d * 9 + n * 5, tip - d * 9 - n * 5]).astype(np.int32)
        cv2.fillPoly(img, [pts], BPW)


@functools.lru_cache(maxsize=1)
def blueprint_sheet():
    """整张蓝图（不含印章），RGB 640×360，只有 SHEET 区域有意义。"""
    img = np.empty((H, W, 3), np.uint8)
    img[:] = DESK
    x0, y0, x1, y1 = SHEET
    img[y0:y1, x0:x1] = BLUE
    for x in range(x0 + 8, x1, 16):
        img[y0:y1, x] = GRID2 if (x - x0 - 8) % 64 == 0 else GRID1
    for y in range(y0 + 8, y1, 16):
        img[y, x0:x1] = GRID2 if (y - y0 - 8) % 64 == 0 else GRID1
    cv2.rectangle(img, (x0 + 6, y0 + 6), (x1 - 7, y1 - 7), BPW, 1)
    img = L.draw_text(img, (32, 30), 'VON NEUMANN ARCH. 1945', BPW, scale=2)
    img[64:66, 32:32 + text_w('VON NEUMANN ARCH. 1945', 2)] = BPW
    # CPU
    cv2.rectangle(img, (40, 80), (204, 178), BPW, 2)
    img = L.draw_text(img, (98, 86), 'CPU', BPW, scale=2)
    cv2.rectangle(img, (52, 126), (118, 166), BPW, 2)
    cv2.rectangle(img, (128, 126), (192, 166), BPW, 2)
    img = L.draw_text(img, (76, 136), 'CU', BPW)
    img = L.draw_text(img, (146, 136), 'ALU', BPW)
    arrow(img, (118, 146), (128, 146))
    # MEMORY
    cv2.rectangle(img, (248, 80), (404, 178), BPW, 2)
    img = L.draw_text(img, (278, 86), 'MEMORY', BPW, scale=2)
    for r in range(4):
        for c in range(6):
            cv2.rectangle(img, (262 + c * 22, 124 + r * 12), (280 + c * 22, 132 + r * 12), BPW, 1)
    arrow(img, (206, 112), (246, 112))
    # BUS
    lw(img, (40, 206), (404, 206))
    lw(img, (40, 216), (404, 216))
    img = L.draw_text(img, (44, 219), 'BUS', BPW)
    for x in (122, 326):
        arrow(img, (x, 180), (x, 204))
    arrow(img, (300, 218), (300, 236))
    cv2.rectangle(img, (256, 238), (344, 270), BPW, 2)
    img = L.draw_text(img, (286, 246), 'I/O', BPW)
    # 标题栏
    cv2.rectangle(img, (520, 226), (612, 270), BPW, 1)
    img[248, 520:612] = BPW
    img = L.draw_text(img, (526, 229), 'J.v.N.', BPW)
    img = L.draw_text(img, (526, 251), 'DWG 1945', BPW)
    # 尺寸标注
    arrow(img, (40, 190), (204, 190))
    img = L.draw_text(img, (104, 180), '8"', BPW)
    return img


@functools.lru_cache(maxsize=1)
def stamp_rgba():
    """OBSOLETE 印章（未旋转）：双线红框 + 大字，按 3px 格子掏掉一些像素做印泥不匀。"""
    w, h = 296, 86
    c = rgba_canvas(w, h)
    def paint(rgb):
        cv2.rectangle(rgb, (0, 0), (w - 1, h - 1), RED, 6)
        cv2.rectangle(rgb, (11, 11), (w - 12, h - 12), RED, 2)
        tw = text_w('OBSOLETE', 4)
        for ddx in range(0, 5, 2):                      # 横向叠三次 = 粗体，量化后才不碎
            for ddy in (0, 2):
                rgb[:] = L.draw_text(rgb, ((w - tw) // 2 + ddx, 12 + ddy), 'OBSOLETE', RED, scale=4)
    draw_rgba(c, paint)
    yy, xx = np.mgrid[0:h, 0:w]
    holes = vhash(xx // 3, yy // 3, 77) < 0.07
    c[holes, 3] = 0
    return c


def girl_state(t):
    """Claude（姿势 3，举章）：返回 (x 中心, 脚底 y, 旋转角, 竖缩放, 是否画)。"""
    tc = L.twos(t)
    x, y, ang, sy = GIRL_X, GIRL_FEET, 0.0, 1.0
    t_in = [beat_time(n) for n in ENTER_BEATS]
    xs = [700, 620, 540, GIRL_X]
    if tc < t_in[0]:
        return xs[0], y, 0, 1, False
    for i in range(3):
        if tc < t_in[i] + BEAT or i == 2:
            u = np.clip((tc - t_in[i]) / (BEAT * 0.8), 0, 1)
            if tc < t_in[i] + BEAT:
                x = xs[i] + (xs[i + 1] - xs[i]) * u
                y = GIRL_FEET - 26 * np.sin(np.pi * u)
                break
    if tc >= t_in[2] + BEAT:
        n, ph = L.beat_phase(tc)
        y = GIRL_FEET - int(round(4 * np.exp(-5 * ph)))
    if T_OBS - 0.34 <= tc < T_OBS - 0.08:              # 蓄力：下蹲、往后仰
        u = (tc - (T_OBS - 0.34)) / 0.26
        sy, ang = 1 - 0.12 * u, -8 * u
    elif T_OBS - 0.08 <= tc < T_OBS:                   # 挥下的中间帧
        ang = 28
    elif T_OBS <= tc < T_OBS + 0.12:                   # 砸下去：整个人往左扑
        ang, sy = 52, 0.92
    elif T_OBS + 0.12 <= tc < T_HOP_OUT0:              # 回正
        u = (tc - T_OBS - 0.12) / (T_HOP_OUT0 - T_OBS - 0.12)
        ang = 52 * (1 - L.smooth(u))
    if tc >= T_HOP_OUT0:                                # 跳出画面
        u = (tc - T_HOP_OUT0) / (T_HOP_OUT1 - T_HOP_OUT0)
        if u >= 1: return 0, 0, 0, 1, False
        x = GIRL_X + 260 * u
        y = GIRL_FEET - 60 * np.sin(np.pi * min(u, 1) * 0.8)
    return x, y, ang, sy, True


def bin_sprite(t):
    """回收站：盖子 + 桶身。卷轴掉进去后盖子弹开、塞着一卷蓝图、整个晃两下。"""
    lid = pixmap(BIN_LID, BIN_COLS, 4)
    body = pixmap(BIN_BODY, BIN_COLS, 4)
    c = rgba_canvas(96, 110)
    full = t >= T_FLY1
    by = 110 - body.shape[0] - 6
    if full:                                           # 塞出来的蓝图卷
        def paint(rgb):
            pts = np.array([(38, by + 8), (60, by - 26), (70, by - 20), (48, by + 12)])
            poly(rgb, pts, (60, 90, 190), 2)
            cv2.line(rgb, (50, by - 6), (62, by - 22), BPW, 1)
        draw_rgba(c, paint)
    c = L_paste_rgba4(c, body, 16, by)
    pop = 0.0
    if full:
        pop = np.exp(-(t - T_FLY1) * 9) * np.sin((t - T_FLY1) * 40)
    lid_r, piv = rot_rgba(lid, 25 * max(pop, 0) + (8 if full else 0), (lid.shape[1] - 4, lid.shape[0]))
    c = L_paste_rgba4(c, lid_r, 16 + lid.shape[1] - 4 - piv[0], by + 2 - piv[1] - 8 * max(pop, 0))
    return c


def L_paste_rgba4(dst, src, x, y):
    """RGBA 叠 RGBA（硬边）。"""
    h, w = src.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1: return dst
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4] > 127
    dst[y0:y1, x0:x1] = np.where(a, s, dst[y0:y1, x0:x1])
    return dst


def scroll_rgba(R, h):
    """卷起来的蓝图筒（竖着）：墨线 + 深浅三条色带。"""
    w = 2 * R
    c = rgba_canvas(w + 6, h + 6)
    def paint(rgb):
        box(rgb, 3, 3, 3 + w, 3 + h, BLUE, 3)
        rgb[3:3 + h, 3:3 + max(2, w // 4)] = (22, 34, 104)
        rgb[3:3 + h, 3 + w // 2:3 + w // 2 + max(2, w // 6)] = (140, 170, 240)
        rgb[3:3 + h, 3 + w - max(2, w // 5):3 + w] = (26, 40, 116)
    return draw_rgba(c, paint)


def draw_desktop(img, t):
    """PC-98 桌面：左上一张软盘图标，右边回收站。"""
    box(img, 36, 34, 64, 62, (150, 154, 170), 2)
    img[34:44, 42:58] = (220, 224, 234)
    img[52:62, 44:56] = (60, 64, 90)
    img = text_ol(img, (40, 66), 'A:', PAPER)
    b = bin_sprite(t)
    wob = 0
    if t >= T_FLY1:
        wob = int(round(4 * np.exp(-(t - T_FLY1) * 7) * np.sin((t - T_FLY1) * 34)))
    img = L.paste_rgba(img, b, SIGN0[0] - 48 + wob, SIGN0[1] - 55)
    s = 'ゴミ箱'
    img = text_ol(img, (SIGN0[0] - text_w(s) // 2, SIGN0[1] + 50), s, PAPER)
    return img


def scene_blueprint(t):
    t = float(np.clip(t, BP_T0, BP_T1))
    tc = L.twos(t)
    sheet = blueprint_sheet().copy()
    x0, y0, x1, y1 = SHEET
    # 线稿从左往右画出来：还没扫到的地方只留网格
    if t < T_REVEAL1:
        rx = int(x0 + (x1 - x0) * L.smooth((t - BP_T0) / (T_REVEAL1 - BP_T0)))
        plain = np.empty_like(sheet[y0:y1, rx:x1])
        plain[:] = BLUE
        g = blueprint_grid_only()[y0:y1, rx:x1]
        sheet[y0:y1, rx:x1] = g
        if rx < x1:
            sheet[y0:y1, rx:rx + 2] = BPW
    # Neumann's：标题闪一下
    if T_NEUMANN <= t < T_NEUMANN + 0.25 and frame24(t) % 4 < 2:
        tw = text_w('VON NEUMANN ARCH. 1945', 2)
        reg = sheet[28:66, 30:32 + tw]
        reg[np.all(reg == BPW, -1)] = YEL
    # 印章
    if t >= T_OBS:
        k = float(np.clip((t - T_OBS) / 0.07, 0, 1))
        sc = 1.35 - 0.35 * k
        st, piv = rot_rgba(stamp_rgba(), 12, None, sc)
        sheet = paste_at(sheet, st, STAMP_C, piv)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = DESK
    img = draw_desktop(img, t)
    # 纸的影子 + 墨框
    if t < T_ROLL0:
        img[y0 + 6:y1 + 6, x0 + 6:x1 + 6] = (12, 14, 40)
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 2, y1 + 2), INK, -1)
        img[y0:y1, x0:x1] = sheet[y0:y1, x0:x1]
    elif t < T_ROLL1:
        u = L.smooth((t - T_ROLL0) / (T_ROLL1 - T_ROLL0))
        R = int(8 + 8 * u)
        xr = int(x1 - (x1 - x0 - 2 * R) * u)
        xa = xr - 2 * R
        img[y0 + 6:y1 + 6, x0 + 6:xa + 6] = (12, 14, 40)
        cv2.rectangle(img, (x0 - 3, y0 - 3), (xa, y1 + 2), INK, -1)
        img[y0:y1, x0:xa] = sheet[y0:y1, x0:xa]
        sc = scroll_rgba(R, y1 - y0 + 4)
        img = L.paste_rgba(img, sc, xa - 3, y0 - 5)
    elif t < T_FLY1:                                    # 卷轴抛进回收站
        u = (t - T_ROLL1) / (T_FLY1 - T_ROLL1)
        p0 = np.array((x0 + 16, (y0 + y1) / 2))
        p1 = np.array((SIGN0[0], SIGN0[1] - 30))
        pos = p0 + (p1 - p0) * u
        pos[1] -= 150 * np.sin(np.pi * u)
        sc = scroll_rgba(16, y1 - y0 + 4)
        spr, piv = rot_rgba(sc, -330 * u, None, 1.0 - 0.72 * u)
        img = paste_at(img, spr, pos, piv)
    # 盖章冲击线
    if T_OBS <= t < T_OBS + 0.15:
        for i in range(10):
            a = i * 2 * np.pi / 10 + 0.3
            r0, r1 = 150, 150 + 40 * (1 - (t - T_OBS) / 0.15) + 8
            p0 = (STAMP_C[0] + r0 * np.cos(a), STAMP_C[1] + 0.55 * r0 * np.sin(a))
            p1 = (STAMP_C[0] + r1 * np.cos(a), STAMP_C[1] + 0.55 * r1 * np.sin(a))
            cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), INK, 7)
            cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), YEL, 3)
    # Claude
    gx, gy, ang, sy, vis = girl_state(t)
    if vis:
        spr = pose_scaled(3, GIRL_H)
        if sy != 1.0:
            spr = cv2.resize(spr, (int(spr.shape[1] * (2 - sy)), int(spr.shape[0] * sy)), interpolation=cv2.INTER_NEAREST)
        piv_in = (spr.shape[1] / 2, spr.shape[0] - 4)
        if ang:
            spr, piv_in = rot_rgba(spr, ang, piv_in)
        img = paste_at(img, spr, (gx, gy), piv_in)
    # 震屏（画面内位移）
    if T_OBS <= t < T_OBS + 0.3:
        a = 9 * np.exp(-(t - T_OBS) * 12)
        f = frame24(t)
        img = shift_img(img, int(round(a * (1 if f % 2 else -1))), int(round(a * 0.6 * (1 if (f // 2) % 2 else -1))))
    return img


@functools.lru_cache(maxsize=1)
def blueprint_grid_only():
    img = np.empty((H, W, 3), np.uint8)
    img[:] = BLUE
    x0, y0, x1, y1 = SHEET
    for x in range(x0 + 8, x1, 16):
        img[:, x] = GRID2 if (x - x0 - 8) % 64 == 0 else GRID1
    for y in range(y0 + 8, y1, 16):
        img[y, :] = GRID2 if (y - y0 - 8) % 64 == 0 else GRID1
    return img


# ================================================================
# 第 12 镜：伪 3D 赛车（80.92–84.68）
# ================================================================
RC_T0, RC_T1 = 80.92, 84.68
T_LEFT, T_TURN, T_THERE, T_ARE = 81.64, 81.96, 82.82, 83.64
HOR = 138
SKY_BANDS = ((0, (36, 14, 64)), (44, (70, 20, 96)), (70, (124, 28, 120)), (92, (192, 44, 132)),
             (108, (240, 84, 120)), (121, (255, 138, 80)), (131, (255, 190, 90)))
SUN_COLS = ((255, 232, 110), (255, 184, 72), (255, 120, 96))
MTN = (58, 18, 78)
GROUND = ((120, 28, 96), (98, 20, 82))
ROAD = ((70, 58, 96), (80, 66, 106))
SILH = (40, 12, 56)
CAR_RED, CAR_SH = RED, (140, 18, 36)
SPEED = 16.0                      # 世界单位/秒（路边物件间距 3）


def curve(t):
    """路面弯曲量：<0 向左弯。left 起猛打，hold，there 前回到微弯。"""
    return (-1.0 * L.smooth((t - (T_LEFT - 0.04)) / 0.34)
            + 0.72 * L.smooth((t - 82.35) / 0.5))


def bg_off(t):
    """左弯时背景往右挪（弯量对时间积分的近似）。"""
    return 220 * L.smooth((t - T_LEFT) / 1.0) - 60 * L.smooth((t - 82.35) / 0.8)


def draw_sky(img, hor, t, sun_x, sun_r, off=0.0):
    for y0, c in SKY_BANDS:
        yy = int(y0 * hor / 138)
        img[yy:hor] = c
    # 条纹落日
    cy = hor + 4
    yy, xx = np.mgrid[0:hor, 0:W]
    d = (xx - sun_x) ** 2 + (yy - cy) ** 2
    m = d <= sun_r * sun_r
    band = np.clip(((cy - yy) / sun_r * 3).astype(int), 0, 2)
    col = np.array(SUN_COLS[::-1], np.uint8)[band]
    gaps = np.zeros(hor, bool)
    for k, g in enumerate((4, 11, 19, 29, 41)):
        gy = cy - g - int(k * 1.5)
        gaps[max(gy, 0):max(gy + 2 + k // 2, 0)] = True
    m &= ~gaps[:, None]
    img[:hor][m] = col[m]
    return img


def draw_mountains(img, hor, off, col=MTN, amp=26):
    xs = np.arange(0, W, 8)
    u = (xs + off) / 60.0
    h = amp * (0.55 + 0.3 * np.sin(u) + 0.25 * np.sin(u * 2.3 + 1) + 0.12 * np.sin(u * 5.1))
    for x, hh in zip(xs, h):
        top = int(hor - hh)
        img[top:hor, x:x + 8] = col
        img[top - 2:top, x:x + 8] = INK
    return img


@functools.lru_cache(maxsize=None)
def eye_layer(star_scale10):
    """一双 AI 眼睛（1/4 分辨率画好再 ×4 放大 = 大方块像素）：青色环 + 深色屏 + 五角星瞳孔 + 四角星闪光。
    返回 RGBA，宽 640，中心在 (320, 60)。"""
    w, h = W // 4, 32
    c = np.zeros((h, w, 4), np.uint8)
    ss = star_scale10 / 10
    def paint(rgb):
        for ex in (w // 2 - 38, w // 2 + 38):
            cv2.ellipse(rgb, (ex, 15), (26, 15), 0, 0, 360, INK, -1)
            cv2.ellipse(rgb, (ex, 15), (24, 13), 0, 0, 360, CYAN, -1)
            cv2.ellipse(rgb, (ex, 15), (20, 10), 0, 0, 360, (14, 22, 52), -1)
            cv2.ellipse(rgb, (ex, 15), (16, 8), 0, 0, 360, (20, 60, 100), 1)
            cv2.fillPoly(rgb, [np.round(star_pts(ex, 15, 8 * ss, 3.4 * ss)).astype(np.int32)], CYAN, lineType=cv2.LINE_8)
            rgb[10:12, ex - 12:ex - 10] = PAPER
        for sx, sy_ in ((w // 2 - 70, 4), (w // 2 + 70, 5), (w // 2, 26), (w // 2 - 4, 3)):
            cv2.fillPoly(rgb, [np.round(star_pts(sx, sy_, 3.2, 1.0, 4)).astype(np.int32)], PAPER, lineType=cv2.LINE_8)
    draw_rgba(c, paint)
    return cv2.resize(c, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


@functools.lru_cache(maxsize=1)
def sign_rgba():
    """路牌「⚠ SHARP LEFT TURN」（未缩放 104×80，含立柱 40px）。"""
    w, h = 104, 120
    c = rgba_canvas(w, h)
    def paint(rgb):
        box(rgb, 48, 76, 56, h - 1, (70, 70, 84), 2)            # 立柱
        box(rgb, 3, 3, w - 3, 77, YEL, 3)
        cv2.rectangle(rgb, (7, 7), (w - 8, 73), INK, 1)
        tri = np.array([(24, 12), (13, 31), (35, 31)])
        poly(rgb, tri, YEL, 1)
        cv2.polylines(rgb, [tri], True, INK, 3, lineType=cv2.LINE_8)
        rgb[18:26, 23:26] = INK
        rgb[28:30, 23:26] = INK
        # 向左的弯箭头
        cv2.line(rgb, (84, 33), (84, 20), INK, 5)                # 向左急弯的折线箭头
        cv2.line(rgb, (84, 20), (62, 20), INK, 5)
        cv2.fillPoly(rgb, [np.array([(50, 20), (63, 11), (63, 29)])], INK)
        rgb[:] = L.draw_text(rgb, (12, 38), 'SHARP LEFT', INK)
        rgb[:] = L.draw_text(rgb, (36, 55), 'TURN', INK)
    return draw_rgba(c, paint)


def draw_road(img, t, hor, cv, cam_x):
    ys = np.arange(hor + 1, H)
    p = (ys - hor) / (H - hor)
    z = 1.0 / np.maximum(p, 1e-3)
    pos = SPEED * (t - RC_T0)
    cx = 320 + cv * (1 - p) ** 2 * 260 + cam_x * p
    hw = 300 * p + 2
    stripe = (np.floor(z * 1.6 + pos * 1.6) % 2).astype(int)
    xx = np.arange(W)[None, :]
    dx = np.abs(xx - cx[:, None])
    hwc = hw[:, None]
    g = np.array(GROUND, np.uint8)[stripe][:, None, :]
    r = np.array(ROAD, np.uint8)[stripe][:, None, :]
    out = np.where((dx < hwc)[..., None], r, g)
    rumble = (dx >= hwc) & (dx < hwc * 1.12)
    rc = np.where(stripe[:, None] == 0, 0, 1)
    out[rumble & (rc == 0)] = PAPER
    out[rumble & (rc == 1)] = (230, 40, 70)
    lane = (dx < np.maximum(hwc * 0.03, 1)) & (stripe[:, None] == 0)
    out[lane] = PAPER
    for side in (-1, 1):                                   # 车道分隔虚线
        lane2 = (np.abs(xx - (cx[:, None] + side * hwc * 0.5)) < np.maximum(hwc * 0.02, 1)) & (stripe[:, None] == 1)
        out[lane2] = (200, 190, 220)
    far = p < 0.035                                        # 地平线附近太密会闪，压成纯色
    out[far] = GROUND[0]
    img[hor + 1:] = out
    img[hor] = INK
    return img, cx, hw


def draw_palm(img, x, yb, s, flip):
    s = max(s, 0.05)
    th, tw = 120 * s, max(2, 7 * s)
    lean = 18 * s * flip
    trunk = np.array([(x - tw, yb), (x + tw, yb), (x + lean + tw * 0.6, yb - th), (x + lean - tw * 0.6, yb - th)])
    poly(img, trunk, SILH, 1 if s < 0.5 else 2)
    tx, ty = x + lean, yb - th
    for k in range(6):
        a = np.radians(-170 + k * 34)
        L_ = 50 * s
        tip = (tx + L_ * np.cos(a), ty + L_ * 0.55 * np.sin(a) + 22 * s * abs(np.cos(a)))
        n = np.array((-np.sin(a), np.cos(a))) * 7 * s
        leaf = np.array([(tx, ty), (tx + L_ * 0.5 * np.cos(a) + n[0], ty + L_ * 0.5 * 0.55 * np.sin(a) + n[1]), tip,
                         (tx + L_ * 0.5 * np.cos(a) - n[0], ty + L_ * 0.5 * 0.55 * np.sin(a) - n[1])])
        poly(img, leaf, SILH, 1 if s < 0.5 else 2)


def draw_post(img, x, yb, s):
    s = max(s, 0.05)
    box(img, x - 3 * s, yb - 70 * s, x + 3 * s + 1, yb, SILH, 1 if s < 0.5 else 2)
    box(img, x - 7 * s, yb - 80 * s, x + 7 * s + 1, yb - 70 * s, (255, 120, 60), 1 if s < 0.5 else 2)


def car_layer(t, drift):
    """小车 + 开车的 Claude（从后面看），RGBA 200×190，车底中心在 (100, 182)。"""
    c = rgba_canvas(200, 190)
    spr = pose_scaled(4, 116)
    bob = int(round(1.5 * np.sin(t * 30)))
    c = L_paste_rgba4(c, spr, 100 - spr.shape[1] // 2 - 8, 14 + bob)
    f = frame24(t)
    def paint(rgb):
        # 轮胎
        for tx in (22, 146):
            box(rgb, tx, 150, tx + 32, 182, (30, 28, 40), 2)
            for k in range(3):
                yy = 154 + ((k * 9 + f * 3) % 26)
                rgb[yy:yy + 2, tx + 2:tx + 30] = (58, 54, 72)
        # 车身
        box(rgb, 16, 118, 184, 162, CAR_RED, 3)
        rgb[150:162, 16:184] = CAR_SH
        poly(rgb, [(34, 118), (166, 118), (156, 102), (44, 102)], (176, 26, 40), 3)
        # 尾灯
        for lx in (24, 146):
            box(rgb, lx, 126, lx + 30, 138, (255, 110, 60), 2)
            rgb[129:135, lx + 4:lx + 14] = YEL
        # 车牌
        box(rgb, 70, 140, 130, 156, PAPER, 2)
        rgb[:] = L.draw_text(rgb, (72, 140), 'P(DOOM)', INK)
        # 排气管
        box(rgb, 40, 160, 52, 168, (150, 150, 164), 2)
        box(rgb, 148, 160, 160, 168, (150, 150, 164), 2)
    return draw_rgba(c, paint)


def draw_smoke(img, t, wheels):
    """轮胎烟：每 1/24 秒从后轮各冒一团，解析式运动。"""
    if t < T_LEFT: return img
    t_end = 82.75
    n_now = int(np.floor((min(t, t_end) - T_LEFT) * 24))
    puffs = []
    for i in range(max(0, n_now - 14), n_now + 1):
        s0 = T_LEFT + i / 24
        a = t - s0
        if a < 0 or a > 0.6: continue
        for j, (wx, wy) in enumerate(wheels):
            side = -1 if j == 0 else 1
            vx = side * (50 + 70 * L.hash01(i, j, 1)) + 60
            x = wx + vx * a
            y = wy - (20 + 40 * L.hash01(i, j, 2)) * a
            r = 6 + 34 * a
            puffs.append((a, x, y, r))
    for a, x, y, r in sorted(puffs, key=lambda p: -p[0]):
        col = PAPER if a < 0.25 else (196, 184, 206)
        cv2.circle(img, (int(x), int(y)), int(r) + 2, INK, -1, lineType=cv2.LINE_4)
        cv2.circle(img, (int(x), int(y)), int(r), col, -1, lineType=cv2.LINE_4)
    return img


def scene_racing(t):
    t = float(np.clip(t, RC_T0, RC_T1))
    tc = L.twos(t)
    img = np.empty((H, W, 3), np.uint8)
    cv = curve(t)
    bo = bg_off(t)
    img = draw_sky(img, HOR, t, 330 + 0.5 * bo, 72)
    # 眼睛从地平线后面升起（there 82.82）
    if t >= T_THERE:
        u = L.ease_out_back((t - T_THERE) / 0.55, 1.2)
        cy = HOR + 70 + (72 - HOR - 70) * u
        ss = 10
        if t >= T_ARE:                                       # are：星星瞳孔一跳
            ss = 10 + int(round(5 * np.exp(-(t - T_ARE) * 6)))
        if t >= T_ARE + 0.5 and 0 <= ((t - T_ARE - 0.5) % 0.9) < 0.08:
            ss = 6
        ey = eye_layer(ss)
        img = L.paste_rgba(img, ey, 0, cy - 60)
    img = draw_mountains(img, HOR, -bo * 0.8)
    img, cxs, hws = draw_road(img, t, HOR, cv, 0)
    # 路边物件：远到近
    pos = SPEED * (t - RC_T0)
    D = 3.0
    i0 = int(np.floor(pos / D))
    items = []
    for i in range(i0, i0 + 14):
        rel = i * D - pos + 1.0
        if rel < 0.6: continue
        p = 1.0 / rel
        if p > 1.4 or p < 0.07: continue
        items.append((rel, i, p))
    for rel, i, p in sorted(items, reverse=True):
        yb = HOR + p * (H - HOR)
        cx = 320 + cv * (1 - min(p, 1)) ** 2 * 260
        hw = 300 * p + 2
        for side in (-1, 1):
            x = cx + side * hw * 1.45
            if (i + (side > 0)) % 2 == 0:
                draw_palm(img, x, yb, p * 1.3, side)
            else:
                draw_post(img, x, yb, p * 1.3)
    # 路牌：从回收站的位置接过来，一路变大掠向右边
    u = (t - RC_T0) / 0.62
    if u < 1:
        s = 1.0 / (1 - 0.75 * u)              # 1.0 起步：与回收站图标同位同大，像素字不被缩坏
        sx = SIGN0[0] + (s - 1.0) * 70
        sy = SIGN0[1] + (s - 1.0) * 34
        spr = sign_rgba()
        sc = cv2.resize(spr, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        img = L.paste_rgba(img, sc, sx - sc.shape[1] / 2, sy - 40 * s)
    # 车
    drift = L.smooth((t - T_LEFT) / 0.18) * (1 - L.smooth((t - 82.45) / 0.5))
    ang = -9 * drift + 1.5 * np.sin(tc * 20) * drift
    car_x = 320 + 46 * drift
    n, ph = L.beat_phase(tc)
    car_y = 272 - int(round(3 * np.exp(-6 * ph)))
    lay = car_layer(tc, drift)
    rl, piv = rot_rgba(lay, ang, (100, 182))
    wheels = []
    for wx in (38, 162):
        a = np.radians(-ang)
        dx, dy = wx - 100, 0
        wheels.append((car_x + dx * np.cos(a) - dy * np.sin(a), car_y + dx * np.sin(a)))
    img = draw_smoke(img, t, wheels) if t < 82.9 else img
    img = paste_at(img, rl, (car_x, car_y), piv)
    # 漂移时两侧的速度线
    if drift > 0.3:
        f = frame24(t)
        for k in range(8):
            y = int(40 + 220 * L.hash01(f // 2, k))
            x = int(W * L.hash01(f // 2, k, 3))
            ln = 60 + int(80 * L.hash01(k, f // 2))
            img[y:y + 2, x:x + ln] = PAPER
    return img


# ================================================================
# 第 13 镜：cons 格子小火车 + NIL（84.68–89.12）
# ================================================================
CD_T0, CD_T1 = 84.68, 89.12
T_SINGLE, T_CDR = 85.60, 86.32
T_BG_OFF0, T_BG_OFF1 = 88.30, 88.70
T_TR_OFF0, T_TR_OFF1 = 88.62, 88.92
T_NIL0, T_NIL1 = 88.74, 88.96
CD_HOR = 176
BOX_W, BOX_H = 52, 40
CELL_W = 2 * BOX_W
GAP = 34
CELL_Y = 178                 # 格子顶边
N_WAGON = 3
SCROLL = 560.0               # 背景往右滚的速度 px/s（火车往左开）
NIL_BIG = 4


def cell_rgba(kind, t, nil=False, carried=True):
    """一节 cons 格子（含车轮），RGBA。kind: engine / wagon / last。"""
    w, h = CELL_W + 16, BOX_H + 32
    c = rgba_canvas(w, h)
    f = frame24(t)
    def paint(rgb):
        # 车轮
        for wx in (8 + 16, 8 + CELL_W - 16):
            wy = BOX_H + 14
            cv2.circle(rgb, (wx, wy), 12, INK, -1, lineType=cv2.LINE_4)
            cv2.circle(rgb, (wx, wy), 9, (70, 64, 90), -1, lineType=cv2.LINE_4)
            a = f * 0.9
            for k in range(2):
                aa = a + k * np.pi / 2
                cv2.line(rgb, (int(wx - 8 * np.cos(aa)), int(wy - 8 * np.sin(aa))),
                         (int(wx + 8 * np.cos(aa)), int(wy + 8 * np.sin(aa))), (180, 170, 200), 2)
            cv2.circle(rgb, (wx, wy), 3, INK, -1)
        col = ORANGE if kind == 'engine' else PAPER
        box(rgb, 8, 4, 8 + BOX_W, 4 + BOX_H, col, 3)
        box(rgb, 8 + BOX_W, 4, 8 + CELL_W, 4 + BOX_H, col, 3)
        if kind == 'engine':
            rgb[4 + BOX_H - 8:4 + BOX_H, 8:8 + CELL_W] = (170, 82, 62)
            box(rgb, 1, 14, 8, 26, YEL, 2)                          # 车头灯
            poly(rgb, [(8, 4 + BOX_H), (-6, 4 + BOX_H + 12), (8, 4 + BOX_H + 12)], (70, 64, 90), 2)
        lab = 'car' if kind == 'engine' else 'cdr'
        rgb[:] = L.draw_text(rgb, (8 + BOX_W // 2 - text_w(lab) // 2, 14), lab, INK)
        cx2, cy2 = 8 + BOX_W + BOX_W // 2, 4 + BOX_H // 2
        if kind == 'last':
            cv2.line(rgb, (8 + BOX_W + 3, 4 + BOX_H - 3), (8 + CELL_W - 3, 7), INK, 3)
        elif nil:
            rgb[:] = L.draw_text(rgb, (8 + BOX_W + 5, 14), '→', INK)
            rgb[:] = L.draw_text(rgb, (8 + BOX_W + 20, 14), 'NIL', INK) if False else rgb
        else:
            cv2.circle(rgb, (cx2, cy2), 5, INK, -1)
    c = draw_rgba(c, paint)
    return c


def draw_link(img, x0, x1, y):
    """cdr 格子里的点 → 下一节左边的箭头（白芯墨边）。"""
    for col, th in ((INK, 7), (PAPER, 3)):
        cv2.line(img, (int(x0), int(y)), (int(x1) - 8, int(y)), col, th)
    pts = np.array([(x1, y), (x1 - 12, y - 7), (x1 - 12, y + 7)], np.int32)
    poly(img, pts, PAPER, 2)


def draw_side_bg(img, t):
    img = draw_sky(img, CD_HOR, t, 470, 84)
    img = draw_mountains(img, CD_HOR, -(t - CD_T0) * 30, MTN, 30)
    # 远处路边的棕榈剪影（中速视差）
    off = (t - CD_T0) * 180
    for k in range(-1, 6):
        x = (k * 150 + off) % 900 - 130
        draw_palm(img, x, CD_HOR + 20, 0.62, 1)
    img[CD_HOR:196] = GROUND[1]
    img[CD_HOR] = INK
    # 路面
    img[196:264] = ROAD[0]
    img[196:200] = PAPER
    img[200] = INK
    off2 = ((t - CD_T0) * SCROLL) % 110
    for k in range(-1, 8):
        x = int(k * 110 + off2)
        img[229:233, max(x, 0):max(x + 52, 0)] = PAPER
    # 路肩红白块
    off3 = ((t - CD_T0) * SCROLL) % 48
    for k in range(-1, 15):
        x = int(k * 48 + off3)
        img[264:272, max(x, 0):max(x + 24, 0)] = (230, 40, 70)
        img[264:272, max(x + 24, 0):max(x + 48, 0)] = PAPER
    img[272:] = GROUND[0]
    img[263] = INK
    img[272] = INK
    return img


def wagon_state(i, t):
    """第 i 节车厢（1..N）：返回 (x 左上, y 左上, 角度)。cdr 时脱钩弹飞。"""
    ex = engine_x(t)
    bx = ex + i * (CELL_W + GAP)
    td = T_CDR + 0.05 * (i - 1)
    if t < td:
        return bx, CELL_Y, 0.0
    a = t - td
    ex_d = engine_x(td) + i * (CELL_W + GAP)
    vx = 180 + 140 * L.hash01(i, 1)
    vy = 330 + 150 * L.hash01(i, 2)
    x = ex_d + vx * a
    y = CELL_Y - vy * a + 0.5 * 1300 * a * a
    return x, y, (260 + 200 * L.hash01(i, 3)) * a * (1 if i % 2 else -1)


def engine_x(t):
    """车头左边 x：cdr 之后甩掉包袱，往画面中间挪一点（镜头重新构图）。"""
    return 96 + 150 * L.smooth((t - T_CDR - 0.1) / 1.0)


def scene_cdr(t):
    t = float(np.clip(t, CD_T0, CD_T1))
    tc = L.twos(t)
    img = np.empty((H, W, 3), np.uint8)
    img = draw_side_bg(img, t)
    bg = img.copy()                  # 清场时用来区分「车」和「背景」
    n, ph = L.beat_phase(tc)
    bob = -int(round(2 * np.exp(-6 * ph)))
    detached = t >= T_CDR
    ex = engine_x(tc)
    # 车厢（从后往前画）
    for i in range(N_WAGON, 0, -1):
        x, y, ang = wagon_state(i, tc)
        kind = 'last' if i == N_WAGON else 'wagon'
        cel = cell_rgba(kind, tc)
        if ang:
            r, piv = rot_rgba(cel, ang)
            img = paste_at(img, r, (x - 8 + cel.shape[1] / 2, y - 4 + cel.shape[0] / 2), piv)
        else:
            img = L.paste_rgba(img, cel, x - 8, y - 4 + bob)
            if i < N_WAGON and tc < T_CDR + 0.05 * i:
                pass
    # 车厢之间的箭头
    for i in range(0, N_WAGON):
        td = T_CDR + 0.05 * i
        if tc >= (T_CDR if i == 0 else td): continue
        if i == 0:
            x0 = ex + BOX_W + BOX_W // 2
        else:
            x0 = wagon_state(i, tc)[0] + BOX_W + BOX_W // 2
        x1 = (wagon_state(i + 1, tc)[0]) - 2
        draw_link(img, x0, x1, CELL_Y + BOX_H // 2 + bob)
    # 脱钩火花
    for i in range(N_WAGON):
        td = T_CDR + 0.05 * i
        if 0 <= tc - td < 0.17:
            x = (ex if i == 0 else wagon_state(i, td)[0]) + CELL_W + GAP // 2
            r = 16 if tc - td < 0.09 else 9
            poly(img, star_pts(x, CELL_Y + BOX_H // 2, r, r * 0.35, 4), YEL, 2)
    # 车头 + Claude（她坐在车头后面，车头盖住她的下半截）
    spr = pose_scaled(4, 92, True)
    img = L.paste_rgba(img, spr, ex + BOX_W - spr.shape[1] // 2 + 4, CELL_Y - 70 + bob)
    eng = cell_rgba('engine', tc, nil=detached)
    img = L.paste_rgba(img, eng, ex - 8, CELL_Y - 4 + bob)
    # 车头的 cdr 格子：脱钩后写 → NIL
    nil_x, nil_y = int(round(ex + BOX_W + 5 + 16)), int(CELL_Y + 10 + bob)
    if detached and tc < T_NIL0:
        k = L.ease_out_back((tc - T_CDR) / 0.2)
        if k > 0.5:
            img = L.draw_text(img, (nil_x, nil_y), 'NIL', INK)
    # 速度线（车头后面）
    if not (tc >= T_TR_OFF1):
        f = frame24(t)
        for k in range(5):
            y = int(CELL_Y - 6 + 12 * k + 4 * L.hash01(f // 2, k))
            x = int(ex + CELL_W + 20 + 60 * L.hash01(k, f // 2))
            if detached or k % 2:
                img[y:y + 2, x:x + 40] = PAPER
    # 清场：背景先按 Bayer 网点熄灭，然后是车，最后只剩 NIL
    if t >= T_BG_OFF0:
        kb = float(np.clip((t - T_BG_OFF0) / (T_BG_OFF1 - T_BG_OFF0), 0, 1))
        kt = float(np.clip((t - T_TR_OFF0) / (T_TR_OFF1 - T_TR_OFF0), 0, 1))
        train = np.any(img != bg, -1)    # 按像素区分车和背景，避免出现矩形色块
        off = np.where(train, L.bayer_mask(kt), L.bayer_mask(kb))
        img[off] = (0, 0, 0)
    if t >= T_NIL0:
        u = L.smooth((t - T_NIL0) / (T_NIL1 - T_NIL0))
        sc = 1 + (NIL_BIG - 1) * u
        tw = text_w('NIL', 1) * sc
        cx = (nil_x + 12) + (320 - (nil_x + 12)) * u
        cy = (nil_y + 8) + (150 - (nil_y + 8)) * u
        on = t < T_NIL1 or (np.floor((t - T_NIL1) * 5) % 2 == 0) or t >= CD_T1 - 1e-6
        if on:
            s = int(round(sc))
            img = text_ol(img, (cx - text_w('NIL', s) / 2, cy - 8 * s), 'NIL', PAPER if u > 0.5 else INK, scale=s,
                          ol=None if u > 0.5 else PAPER)
    return img


SCENES = [
    # 第 9 镜拆两段、同一个函数：结尾彩灯段（0.55 秒）单独锁调色板，否则品红/青/绿全被火焰橙黄挤掉（已实测）
    ('safety', SAFE_T0, T_DISCO0, scene_safety),
    ('safety_disco', T_DISCO0, SAFE_T1, scene_safety),
    ('blueprint', BP_T0, BP_T1, scene_blueprint),
    ('racing', RC_T0, RC_T1, scene_racing),
    ('cdr', CD_T0, CD_T1, scene_cdr),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
