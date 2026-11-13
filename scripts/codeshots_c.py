"""codeshots_c：第二部分的纯代码镜头 C 组（分镜第二部分 第 1、2、3、4、8 镜）。

    stable       38.12–41.26  接第一部分结尾弹出的 TRAINING RUN #2 窗口：曲线继续平稳下降（横轴压缩），
                              状态栏绿灯逐个亮 → STABLE ✓；Q 版 Claude 落到曲线上捧茶，run 40.50 喝一口；末端出亮点
    singularity  41.26–44.74  亮点炸开成像素黑洞，整幅画面（窗口/曲线/绿灯）被螺旋吸入，
                              紫色吸积盘做调色板轮转；Claude 抱茶杯绕着转圈被吸进去；结尾黑洞反转成白点
    optimize     44.74–48.44  白点铺开成黑底青色线框 loss 地形，Claude 头像小球沿梯度滚向谷底；
                              accelerating 起加速线 + 速度计 ×2 ×4 ×8 ×∞；触底炸成像素方块
    atoms        48.44–54.94  方块聚成日常装 Claude（8×8 块）→ 漂浮 → atoms 按批飞散打转 → 空中白闪换装 →
                              rearranging 里飞回拼成舞台服 → 52.7 落定亮相（金色放射光 + 星点）→ 后拉露出 CRT 屏幕
    calc         65.90–68.58  计算器大特写，液晶琥珀底黑字七段数码：1、10、100… → 1E30 → 溢出「E」闪烁，冒烟遮屏

接口约定见 part1_lib 文件头：scene_xxx(t) -> (360,640,3) uint8 RGB，t 为完整版歌曲秒数，纯函数；越界 t 钳到本段范围。
自检：python codeshots_c.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np

import part1_lib as L
import codeshots_a as A           # 只读复用：窗口画法、TRAINING RUN #2 的几何常量和曲线公式

W, H = L.W, L.H
TOP, BOT = 18, 284
MIDY = (TOP + BOT) // 2           # 有效画面中心 y = 151

rect, box, text, text_w, text_outline = A.rect, A.box, A.text, A.text_w, A.text_outline
ease_io = A.ease_io
INK, PAPER, WINBG, TITLE, YEL, ORANGE = A.INK, A.PAPER, A.WINBG, A.TITLE, A.YEL, A.ORANGE
WHITE, GREEN, GREEN_DIM, TERM_BG, CYAN, CYAN_HI = A.WHITE, A.GREEN, A.GREEN_DIM, A.TERM_BG, A.CYAN, A.CYAN_HI
NAVY, NAVY_DOT = A.NAVY, A.NAVY_DOT
BLACK = (0, 0, 0)

SPRITE_Q2 = '素材/精灵/q版Claude_四姿势2.png'
Q2_COLS = [(22, 453), (473, 875)]         # 四姿势2 里姿势 1（捧茶）、姿势 2（按计算器）的列范围（alpha 列投影量得）


# ---------------- 精灵工具 ----------------
def _rgba_resize(rgba, k):
    """预乘 alpha 缩放（透明像素 RGB=0，直接缩会把边缘拉暗），结果 alpha 二值。"""
    a = rgba[..., 3:4].astype(np.float32) / 255
    pm = np.concatenate([rgba[..., :3] * a, a * 255], 2)
    sm = cv2.resize(pm, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    al = sm[..., 3:4]
    rgb = np.where(al > 1, sm[..., :3] / np.maximum(al, 1) * 255, 0)
    out = np.concatenate([np.clip(rgb, 0, 255), np.where(al > 127, 255, 0)], 2)
    return out.astype(np.uint8)


@functools.lru_cache(maxsize=None)
def _q2(idx, height):
    """四姿势2 的第 idx 个姿势，裁紧后缩到 height 高。"""
    rgba = L.load_rgba(SPRITE_Q2)
    s, e = Q2_COLS[idx]
    sub = rgba[:, s:e]
    m = sub[..., 3] > 127
    rows, cols = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    sub = sub[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return _rgba_resize(sub, height / sub.shape[0])


def _rot_rgba(spr, deg, scale=1.0):
    """绕中心旋转 + 缩放（最近邻，保持像素硬边），画布扩到对角线大小。"""
    h, w = spr.shape[:2]
    d = int(np.ceil(np.hypot(w, h) * scale)) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    M[0, 2] += d / 2 - w / 2
    M[1, 2] += d / 2 - h / 2
    return cv2.warpAffine(spr, M, (d, d), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))


def _star(img, x, y, arm, core=WHITE, tip=None, th=1):
    """四角硬边星点（十字 + 中心方块）。原地修改。"""
    x, y, arm = int(round(x)), int(round(y)), int(round(arm))
    if arm <= 0: return
    tip = tip or core
    rect(img, x - arm, y - th, x + arm + 1, y + th + 1, tip)
    rect(img, x - th, y - arm, x + th + 1, y + arm + 1, tip)
    c = max(1, arm // 3)
    rect(img, x - c, y - c, x + c + 1, y + c + 1, core)


def _check(img, x, y, c, s=2):
    """像素对勾（字体里没有 ✓）。"""
    pts = [(0, 3), (1, 4), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]
    for dx, dy in pts:
        rect(img, x + dx * s, y + dy * s, x + dx * s + s + 1, y + dy * s + s + 1, c)


# =====================================================================
# 1. stable：TRAINING RUN #2 继续，岁月静好
# =====================================================================
S_0, S_1 = 38.12, 41.26
T_RUN = 40.50
T_SPARK = 40.95
FX0, FY0, FX1, FY1 = A.FX0, A.FY0, A.FX1, A.FY1
GX0, GY0, GX1, GY1 = A.GX0, A.GY0, A.GX1, A.GY1
LAMP_X, LAMP_N = 170, 8
LAMP_Y = FY0 + 35
LAMP_T0, LAMP_DT = 38.30, 60 / 132 / 4        # 1/4 拍亮一个
T_STABLE = 39.24
CHIBI_H, CHIBI_CX = 92, 500


def _comp(t):
    """横轴压缩系数：38.12 时 =1（和 codeshots_a 里训练窗口的原始几何一致），之后整段历史挤进图区。"""
    return max(1.0, (t - A.T2_0) / (A.T2_1 - A.T2_0))


def _st_y(x, c):
    d = (x - GX0) / (GX1 - GX0) * c
    y = GY0 + 14 + (GY1 - GY0 - 40) * (1 - np.exp(-3.2 * d)) / (1 - np.exp(-3.2)) + (L.hash01(x, 31) - .5) * 2
    return y + 9 * max(0.0, d - 1)                                    # 压缩后尾段继续缓慢下降（c=1 时不起作用）


def _st_curve(t):
    c = _comp(t)
    xs = np.arange(GX0, GX1 - 4 + 1)
    ys = np.array([_st_y(int(x), c) for x in xs])
    return np.stack([xs, ys], 1).astype(np.int32)


def _stable_bg(t, lamps=True):
    """窗口 + 曲线 + 数字 + 绿灯（不含 Claude、亮点）。t=38.12 时与 A._t2_window(38.12) 逐像素一致。"""
    img = np.zeros((H, W, 3), np.uint8)
    img = A._window(img, FX0, FY0, FX1, FY1, 'TRAINING RUN #2', (GX0, GY0, GX1, GY1), TERM_BG, GREEN_DIM)
    pts = _st_curve(t)
    cv2.polylines(img, [pts], False, GREEN, 2, cv2.LINE_8)
    hx, hy = pts[-1]
    if L.pulse(t) > .45:
        rect(img, hx - 3, hy - 3, hx + 4, hy + 4, WHITE)
    img = text(img, (FX0 + 8, GY0 - 2), 'L', TITLE)
    c = _comp(t)
    img = text(img, (FX0 + 12, FY0 + 32), 'step %d' % (1 + int(c * 1200) * 8), INK)
    loss = 2.31 - 1.6 * (1 - np.exp(-3.2 * c)) / (1 - np.exp(-3.2)) - 0.06 * (c - 1)
    s = 'loss = %.2f' % loss
    img = text(img, (FX1 - 16 - text_w(s), FY0 + 32), s, INK)
    if L.pulse(t) > .45:
        rect(img, FX1 - 70, FY0 + 9, FX1 - 60, FY0 + 19, GREEN)
    if lamps:
        img = _lamps(img, t)
    return img


def _lamps(img, t):
    """状态栏：一排绿灯 1/4 拍一个亮起，stable 时出 STABLE ✓，之后拍点整排闪亮。"""
    full = t >= T_STABLE
    hi = full and L.pulse(t) > .5
    for i in range(LAMP_N):
        t_on = LAMP_T0 + i * LAMP_DT
        if t < t_on: break
        x = LAMP_X + i * 18
        pop = t < t_on + 0.08                                          # 刚亮的一帧放大一圈
        g = 1 if pop else 0
        box(img, x - g, LAMP_Y - g, x + 13 + g, LAMP_Y + 13 + g, GREEN, t=2)
        hs = 7 if (hi or pop) else 3                                   # 拍点 / 刚亮：灯珠高光变大
        rect(img, x + 3, LAMP_Y + 3, x + 3 + hs, LAMP_Y + 3 + hs, WHITE)
    if full:
        x = LAMP_X + LAMP_N * 18 + 6
        img = text(img, (x, FY0 + 32), 'STABLE', TITLE)
        _check(img, x + text_w('STABLE') + 6, LAMP_Y - 1, (20, 150, 60) if not hi else GREEN)
    return img


def _stable_chibi(t):
    """(精灵, 左上 x, 左上 y)；落下 → 静坐 → run 时喝一口。人物一拍二。"""
    tt = L.twos(t)
    spr = _q2(0, CHIBI_H)
    c = _comp(t)
    base = int(_st_y(CHIBI_CX, c)) + 6                               # 屁股坐在曲线上
    dy, sy = 0, 1.0
    if tt < 38.50:                                                    # 从上方掉下来
        u = np.clip((tt - 38.22) / 0.28, 0, 1)
        dy = -int(150 * (1 - u * u))
    elif tt < 38.66:                                                  # 落地小弹
        dy = -int(round(6 * np.sin(np.pi * (tt - 38.50) / 0.16)))
    u = (tt - (T_RUN - 0.08)) / 0.50
    if 0 <= u < 1:                                                    # 喝一口：身体上提、微微拉长
        k = np.sin(np.pi * u)
        dy -= int(round(4 * k))
        sy = 1 + 0.06 * k
    if sy != 1.0:
        h, w = spr.shape[:2]
        spr = cv2.resize(spr, (w, int(round(h * sy))), interpolation=cv2.INTER_NEAREST)
    h, w = spr.shape[:2]
    return spr, CHIBI_CX - w // 2, base - h + dy


def _spark_arm(t):
    """亮点臂长：40.95 冒出来，拍点涨一下。"""
    if t < T_SPARK: return 0
    u = np.clip((t - T_SPARK) / (S_1 - T_SPARK), 0, 1)
    return 2 + 5 * u + 3 * L.pulse(t)


def _stable_frame(t):
    img = _stable_bg(t)
    if t < 38.22: return img
    spr, x, y = _stable_chibi(t)
    img = L.paste_rgba(img, spr, x, y)
    if T_RUN + 0.20 <= t < T_RUN + 0.66:                             # 喝完头顶冒个 ♪
        u = (L.twos(t) - T_RUN - 0.20) / 0.46
        img = text_outline(img, (x + spr.shape[1] - 6, int(y - 6 - 14 * u)), '♪', YEL, o=1)
    arm = _spark_arm(t)
    if arm > 0:
        hx, hy = _st_curve(t)[-1]
        _star(img, hx, hy, arm, WHITE, th=1)
    return img


def scene_stable(t):
    t = float(np.clip(t, S_0, S_1))
    return _stable_frame(t)


# =====================================================================
# 2. singularity：像素黑洞
# =====================================================================
G_0, G_1 = 41.26, 44.74
G_SUCK0, G_SUCK1 = 41.42, 43.90              # 吸入开始 / 全部吸完
G_HOLE = 41.40                               # 黑洞诞生
G_SING, G_BEGUN = 41.98, 44.02
G_SHRINK, G_WHITE = 44.30, 44.52             # 吸积盘收缩 / 中心反转成白点
BH_C = np.float32([320, MIDY])
DISK = [(52, 16, 96), (118, 44, 190), (186, 104, 250), (250, 196, 255)]   # 吸积盘四档紫（调色板轮转用）
RING = (255, 220, 255)
TILT = 0.36


@functools.lru_cache(maxsize=None)
def _g_src():
    return _stable_bg(S_1)


@functools.lru_cache(maxsize=None)
def _g_p0():
    x, y = _st_curve(S_1)[-1]
    return np.float32([x, y])


@functools.lru_cache(maxsize=None)
def _grid():
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return xs, ys


G_MID = np.float32([470, 176])


def _g_center(t):
    """亮点 → 先挪到画面右中部吸东西（窗口大部分还在画内）→ 吸完再挪到正中。"""
    k1 = ease_io((t - G_SUCK0) / 0.60)
    k2 = ease_io((t - 43.30) / 0.65)
    c = _g_p0() + (G_MID - _g_p0()) * k1
    return c + (BH_C - c) * k2


def _g_hole_r(t):
    if t < G_HOLE: return 0.0
    r = 20 * L.smooth((t - G_HOLE) / 0.22) + 10 * L.smooth((t - 41.8) / 0.8)
    if t > G_SHRINK: r *= 1 - L.smooth((t - G_SHRINK) / (G_WHITE - G_SHRINK))
    return r


def _g_disk_r(t):
    r = 150 * L.ease_out_back((t - 41.48) / 0.55, 1.2)
    if t > G_SHRINK: r *= 1 - L.smooth((t - G_SHRINK) / (G_WHITE - G_SHRINK + 0.04))
    return max(r, 0.0)


def _g_warp(t, C):
    """螺旋吸入：输出半径 r 采样源半径 sqrt(r²+B²)，B 以内的东西都已掉进去；再叠一个随半径衰减的旋转。"""
    u = np.clip((t - G_SUCK0) / (G_SUCK1 - G_SUCK0), 0, 1)
    B = 760 * u ** 1.5
    spin = 7.0 * L.smooth(u * 1.1)
    xs, ys = _grid()
    dx, dy = xs - C[0], ys - C[1]
    r = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx) - spin * np.exp(-r / 170)
    rs = np.sqrt(r * r + B * B)
    p0 = _g_p0()
    mx = (p0[0] + rs * np.cos(th)).astype(np.float32)
    my = (p0[1] + rs * np.sin(th)).astype(np.float32)
    return cv2.remap(_g_src(), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=BLACK)


def _g_chibi(img, t, C):
    """Claude 抱着茶杯绕黑洞转圈、越转越小，begun 时掉进去。"""
    tt = L.twos(t)
    if tt >= G_BEGUN: return img
    spr0 = _q2(0, CHIBI_H)
    h, w = spr0.shape[:2]
    p0 = _g_p0()
    # 起点：stable 最后一帧她的中心（相对亮点）
    sx = CHIBI_CX
    sy = int(_st_y(CHIBI_CX, _comp(S_1))) + 6 - h / 2
    v0 = np.array([sx - p0[0], (sy - p0[1]) / TILT])
    rho0, phi0 = np.hypot(*v0), np.arctan2(v0[1], v0[0])
    u = np.clip((tt - G_SUCK0) / (G_BEGUN - G_SUCK0), 0, 1)
    rho = rho0 * (1 - u ** 1.3)
    phi = phi0 + 2 * np.pi * 2.6 * u ** 1.2                          # 公转 2.6 圈
    cx, cy = C[0] + rho * np.cos(phi), C[1] + rho * TILT * np.sin(phi)
    ang = -360 * 3.0 * u ** 1.1                                       # 自转
    sc = 1 - 0.82 * u
    if u == 0:
        return L.paste_rgba(img, spr0, sx - w // 2, sy - h / 2)
    spr = _rot_rgba(spr0, ang, sc)
    d = spr.shape[0]
    return L.paste_rgba(img, spr, cx - d / 2, cy - d / 2)


def _g_debris(img, t, C):
    """被吸进来的窗口碎片（绿灯、窗体、标题栏、曲线的小方块），从画外沿螺旋落进黑洞。"""
    cols = [GREEN, WINBG, TITLE, GREEN, WHITE, WINBG]
    tt = L.twos(t)
    for i in range(46):
        t0 = 41.75 + L.hash01(i, 91) * 1.9
        life = 0.9 + L.hash01(i, 92) * 0.6
        u = (tt - t0) / life
        if not 0 <= u < 1: continue
        rho = (330 + L.hash01(i, 93) * 150) * (1 - u) ** 1.4
        phi = L.hash01(i, 94) * 2 * np.pi + 5.5 * u ** 1.5
        if rho < _g_hole_r(t) * 1.1: continue                         # 掉进视界就没了
        x, y = C[0] + rho * np.cos(phi), C[1] + rho * TILT * 1.3 * np.sin(phi)
        sz = max(2, int((4 + L.hash01(i, 95) * 7) * (1 - 0.7 * u)))
        rect(img, x - sz // 2 - 1, y - sz // 2 - 1, x + sz - sz // 2 + 1, y + sz - sz // 2 + 1, INK)
        rect(img, x - sz // 2, y - sz // 2, x + sz - sz // 2, y + sz - sz // 2, cols[i % len(cols)])


def scene_singularity(t):
    t = float(np.clip(t, G_0, G_1))
    tt = L.twos(t)
    C = _g_center(t)
    img = _g_warp(t, C) if t >= G_SUCK0 else _g_src().copy()
    rh = _g_hole_r(t)
    ro = _g_disk_r(t)
    xs, ys = _grid()
    dx, dy = xs - C[0], (ys - C[1]) / TILT
    re = np.sqrt(dx * dx + dy * dy)                                   # 盘平面里的半径
    rr = np.sqrt((xs - C[0]) ** 2 + (ys - C[1]) ** 2)                 # 屏幕上的半径（黑洞是正圆）
    if ro > 2:
        ri = rh * 1.35
        disk = (re < ro) & (re > ri)
        # 调色板轮转：带的位置由角度 + 半径决定，颜色索引随 12fps 时间平移
        th = np.arctan2(dy, dx)
        band = np.floor(th / (2 * np.pi) * 4 + re / 13 - tt * 14).astype(int) % 4
        # 外缘最暗、内缘最亮：按半径再加一层偏移
        band = np.clip(band + (re < ri + (ro - ri) * 0.35) * 1 - (re > ri + (ro - ri) * 0.8) * 1, 0, 3)
        front = ys >= C[1]
        hole = rr < rh
        pal = np.array(DISK, np.uint8)
        back = disk & ~front
        img[back] = pal[band[back]]
        if rh > 0.5:
            img[hole] = BLACK
            ring = (rr >= rh) & (rr < rh + 1.6)
            img[ring & ~(disk & front)] = RING
        fr = disk & front
        img[fr] = pal[band[fr]]
    elif rh > 0.5:
        img[rr < rh] = BLACK
        img[(rr >= rh) & (rr < rh + 1.6)] = RING
    # 亮点：开头猛地炸大一下，然后塌缩成黑洞
    if t < G_HOLE + 0.08:
        u = (t - G_0) / (G_HOLE + 0.08 - G_0)
        arm = (_spark_arm(G_0) + 38 * np.sin(np.pi * min(u * 1.3, 1))) * (1 - max(0, u - 0.7) / 0.3)
        _star(img, C[0], C[1], arm, WHITE, tip=RING if arm > 12 else WHITE, th=2 if arm > 12 else 1)
    _g_debris(img, t, C)
    img = _g_chibi(img, t, C)
    # 结尾：黑洞反转成白点
    if t >= G_WHITE:
        k = np.clip((t - G_WHITE) / (G_1 - G_WHITE), 0, 1)
        img[:] = BLACK
        r = int(round(9 - 6.5 * k))
        cx, cy = int(BH_C[0]), int(BH_C[1])
        if k < 0.5:
            _star(img, cx, cy, int(28 * (1 - 2 * k)) + r, WHITE, th=1)
        rect(img, cx - r, cy - r, cx + r + 1, cy + r + 1, WHITE)
    return img


# =====================================================================
# 3. optimize：线框 loss 地形 + 梯度下降小球
# =====================================================================
O_0, O_1 = 44.74, 48.44
O_OPT, O_ACC = 45.64, 47.42
O_HIT = 48.22
GEARS = [(47.42, '×2', '6e-4'), (47.765, '×4', '1e-3'), (47.99, '×8', '3e-3'), (48.22, '×∞', 'NaN')]
CYAN_DIM = (22, 104, 124)
SKIN = (255, 222, 196)
HAIR_HI = (255, 170, 90)
F_FOCAL, CAM_H, HORIZON = 300.0, 2.2, 40.0
Z_SCALE = 1.1


def _f(u, v):
    """弯曲的山谷（只有一个极小值），高度 0..~1。"""
    return 0.34 * (u - 0.15 - 0.30 * np.sin(1.6 * v)) ** 2 + 0.22 * (v + 0.45) ** 2 + 0.05 * np.sin(4 * u) * np.cos(3 * v) * (v + 0.45) ** 2


def _grad(u, v, e=1e-4):
    return np.array([(_f(u + e, v) - _f(u - e, v)) / (2 * e), (_f(u, v + e) - _f(u, v - e)) / (2 * e)])


def _yaw(t):
    return 0.22 * np.sin(0.9 * (t - O_0)) - 0.10


def _proj(u, v, y, t, k=1.0):
    """(u,v,高度) → 屏幕。k 是开场铺开的缩放（以白点为中心）。"""
    a = _yaw(t)
    X, Z = u * 2.0, v * 1.6
    Xr, Zr = X * np.cos(a) - Z * np.sin(a), X * np.sin(a) + Z * np.cos(a)
    Zw = Zr + 4.4
    sx = 320 + Xr * F_FOCAL / Zw
    sy = HORIZON + (CAM_H - y) * F_FOCAL / Zw
    return 320 + (sx - 320) * k, MIDY + (sy - MIDY) * k, Zw


@functools.lru_cache(maxsize=None)
def _path():
    """带动量的梯度下降轨迹（固定起点，预计算），按弧长归一化。"""
    p, vel = np.array([-0.92, 0.88]), np.zeros(2)
    pts = [p.copy()]
    for _ in range(900):
        vel = 0.92 * vel - 0.012 * _grad(*p)
        p = p + vel
        pts.append(p.copy())
    pts = np.array(pts)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    return pts, seg / seg[-1]


def _ball_s(t):
    """小球在轨迹上的进度 0..1：optimizing 慢慢滚，accelerating 越滚越快，O_HIT 到底。"""
    if t < O_OPT: return 0.0
    if t < O_ACC: return 0.22 * ((t - O_OPT) / (O_ACC - O_OPT)) ** 1.6
    return min(1.0, 0.22 + 0.78 * ((t - O_ACC) / (O_HIT - O_ACC)) ** 2.4)


def _ball_uv(t):
    pts, s = _path()
    sv = _ball_s(t)
    return np.interp(sv, s, pts[:, 0]), np.interp(sv, s, pts[:, 1])


def _ball_screen(t):
    u, v = _ball_uv(t)
    x, y, zw = _proj(u, v, _f(u, v) * Z_SCALE, t)
    r = 0.2 * F_FOCAL / zw
    return x, y - r * 0.8, r


def _draw_ball(img, x, y, r, roll):
    """Claude 头像小球：肤色脸、橙发（随滚动转）、两只豆豆眼、白花发饰、墨线。"""
    R = int(np.ceil(r)) + 2
    x0, y0 = int(round(x)), int(round(y))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(xx ** 2 + yy ** 2)
    ca, sa = np.cos(-roll), np.sin(-roll)
    ux, uy = xx * ca - yy * sa, xx * sa + yy * ca                    # 球面上的局部坐标（跟着滚）
    out = np.zeros((2 * R + 1, 2 * R + 1, 3), np.uint8)
    body = d <= r
    hair = body & ((uy < -0.05 * r) | (np.abs(ux) > 0.62 * r))
    face = body & ~hair
    out[face] = SKIN
    out[hair] = ORANGE
    out[hair & (uy < -0.55 * r) & (ux < -0.1 * r)] = HAIR_HI
    eye = face & ((np.abs(ux - 0.3 * r) < max(1.0, 0.11 * r)) | (np.abs(ux + 0.3 * r) < max(1.0, 0.11 * r))) \
        & (np.abs(uy - 0.22 * r) < max(1.2, 0.2 * r))
    out[eye] = INK
    flower = (np.abs(ux + 0.5 * r) < max(1.0, 0.15 * r)) & (np.abs(uy + 0.45 * r) < max(1.0, 0.15 * r)) & body
    out[flower] = PAPER
    line = (d > r) & (d <= r + 1.6)
    out[line] = INK
    m = body | line
    ys, xs = np.nonzero(m)
    ys2, xs2 = ys + y0 - R, xs + x0 - R
    ok = (ys2 >= 0) & (ys2 < H) & (xs2 >= 0) & (xs2 < W)
    img[ys2[ok], xs2[ok]] = out[ys[ok], xs[ok]]


def _terrain(img, t, k, col_near, col_far):
    n, m = 22, 44
    g = np.linspace(-1, 1, n)
    q = np.linspace(-1, 1, m)
    for a in g:
        for fixed_u in (True, False):
            if fixed_u:
                uu, vv = np.full(m, a), q
            else:
                uu, vv = q, np.full(m, a)
            x, y, _ = _proj(uu, vv, _f(uu, vv) * Z_SCALE, t, k)
            pts = np.stack([x, y], 1).astype(np.int32)
            if fixed_u:                                               # 竖线切两截：远处暗、近处亮
                h = m // 2
                cv2.polylines(img, [pts[h - 1:]], False, col_far, 1, cv2.LINE_8)
                cv2.polylines(img, [pts[:h]], False, col_near, 1, cv2.LINE_8)
            else:
                cv2.polylines(img, [pts], False, col_near if a < 0 else col_far, 1, cv2.LINE_8)


def _gear(t):
    g = -1
    for i, (tg, _, _) in enumerate(GEARS):
        if t >= tg: g = i
    return g


def _speedo(img, t):
    g = _gear(t)
    label = '×1' if g < 0 else GEARS[g][0 + 1]
    tg = O_OPT if g < 0 else GEARS[g][0]
    flash = g >= 0 and t - tg < 0.09
    x0, y0, x1, y1 = 516, 24, 628, 78
    box(img, x0, y0, x1, y1, CYAN if flash else BLACK, ink=CYAN, t=2)
    img = text(img, (x0 + 8, y0 + 4), 'SPEED', BLACK if flash else CYAN)
    sc = 2
    img = text_outline(img, (x1 - 10 - text_w(label, sc), y1 - 4 - 17 * sc), label,
                       WHITE, scale=sc, ink=BLACK if not flash else CYAN, o=1)
    return img


def _speedlines(img, t, bx, by):
    g = _gear(t)
    if g < 0: return
    f = int(np.floor(t * 12))
    n = 10 + 6 * g
    for i in range(n):
        th = L.hash01(f, i, 1) * 2 * np.pi
        r0 = 120 + L.hash01(f, i, 2) * 150 - 15 * g
        c = WHITE if L.hash01(f, i, 3) < .5 else CYAN_HI
        p0 = (int(bx + r0 * np.cos(th)), int(by + r0 * np.sin(th)))
        p1 = (int(bx + 800 * np.cos(th)), int(by + 800 * np.sin(th)))
        cv2.line(img, p0, p1, c, 1 + (i % 3 == 0), cv2.LINE_8)


def _o_hit_point():
    x, y, r = _ball_screen(O_HIT)
    return x, y


def _boom(img, t):
    """触底爆炸：像素方块放射飞散（解析公式，出生于 O_HIT）。"""
    u = t - O_HIT
    if u < 0: return
    ex, ey = _o_hit_point()
    cols = [WHITE, CYAN, ORANGE, SKIN, YEL, CYAN_HI]
    for i in range(110):
        th = L.hash01(i, 41) * 2 * np.pi
        sp = 120 + L.hash01(i, 42) * 520
        dist = sp * (1 - np.exp(-5 * u)) / 5
        x, y = ex + dist * np.cos(th), ey + dist * np.sin(th) * 0.8 - 30 * u
        s = 3 + int(L.hash01(i, 43) * 8)
        rect(img, x - s // 2, y - s // 2, x + s - s // 2, y + s - s // 2, cols[i % len(cols)])
    if u < 0.10:                                                      # 撞击白闪：硬边大星
        _star(img, ex, ey, 70 * (1 - u / 0.10) + 10, WHITE, th=3)


def scene_optimize(t):
    t = float(np.clip(t, O_0, O_1))
    img = np.zeros((H, W, 3), np.uint8)
    k = L.ease_out_back((t - O_0) / 0.55, 1.2)
    white = t < O_0 + 0.14
    if k > 0.004:
        _terrain(img, t, k, WHITE if white else CYAN, WHITE if white else CYAN_DIM)
    if t < O_0 + 0.10:                                                # 接上一镜的白点
        r = int(round(2.5 * (1 - (t - O_0) / 0.10)))
        rect(img, 320 - r, MIDY - r, 320 + r + 1, MIDY + r + 1, WHITE)
    if t >= O_OPT - 0.4:
        img = text(img, (14, 24), 'GRADIENT DESCENT', CYAN)
        g = _gear(t)
        img = text(img, (14, 44), 'lr = ' + ('3e-4' if g < 0 else GEARS[g][2]), WHITE if g >= 0 else CYAN)
    tt = L.twos(t)
    if t >= 45.0 and t < O_HIT:
        bx, by, r = _ball_screen(tt)
        if t < 45.30:                                                 # 从天上掉到起点
            u = (tt - 45.0) / 0.30
            by -= 160 * (1 - u * u)
        elif t < 45.42:
            by -= 5 * np.sin(np.pi * (tt - 45.30) / 0.12)
        _speedlines(img, t, bx, by)
        g = _gear(t)
        if g >= 0:                                                    # 残影：路径上过去的位置画橙色方块
            for j in range(1, 5 + g):
                px, py, pr = _ball_screen(tt - j * 0.03)
                s = max(2, int(pr * (1 - j / (6 + g))))
                rect(img, px - s, py - s, px + s, py + s, ORANGE if j % 2 else YEL)
        pts, sarr = _path()
        roll = _ball_s(tt) * 2.2 * 60 / max(r, 1)                    # 弧长 / 半径
        _draw_ball(img, bx, by, r, roll)
    _boom(img, t)
    img = _speedo(img, t) if t >= O_OPT - 0.4 else img
    return img


# =====================================================================
# 4. atoms：像素重排变身
# =====================================================================
M_0, M_1 = 48.44, 54.94
M_ATOMS, M_REARR, M_LAND = 50.16, 50.94, 52.70
M_PULL0 = 53.50
CG_A, CG_B = '素材/CG/p2_s4a_变身前.png', '素材/CG/p2_s4b_变身后.png'
CG_FIG_H, CG_FIG_TOP = 300, 12              # 人物缩到 300 高，头顶在 y=12
BLK = 8
ORBIT_C = np.float32([320, 150])
WGOLD = (255, 244, 200)
GOLD = (240, 190, 84)
GOLD_DK = (150, 104, 40)


@functools.lru_cache(maxsize=None)
def _cg_canvas(path):
    """CG → 640×360 RGBA 画布（人物高 300、水平居中）。"""
    rgba = L.load_rgba(path)
    m = rgba[..., 3] > 127
    rows, cols = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    sub = rgba[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    sm = _rgba_resize(sub, CG_FIG_H / sub.shape[0])
    can = np.zeros((H, W, 4), np.uint8)
    h, w = sm.shape[:2]
    x = (W - w) // 2
    y = CG_FIG_TOP
    hh = min(h, H - y)
    can[y:y + hh, x:x + w] = sm[:hh]
    return can


@functools.lru_cache(maxsize=None)
def _blocks():
    """把两张 CG 切成 8×8 块。返回 dict：home(N,2)、ta/tb(N,8,8,4)、hasA/hasB、各种 hash。"""
    ca, cb = _cg_canvas(CG_A), _cg_canvas(CG_B)
    homes, ta, tb, ha, hb = [], [], [], [], []
    for gy in range(0, H, BLK):
        for gx in range(0, W, BLK):
            a = ca[gy:gy + BLK, gx:gx + BLK]
            b = cb[gy:gy + BLK, gx:gx + BLK]
            na, nb = (a[..., 3] > 127).sum(), (b[..., 3] > 127).sum()
            if na < 4 and nb < 4: continue
            homes.append((gx, gy))
            ta.append(a)
            tb.append(b)
            ha.append(na >= 4)
            hb.append(nb >= 4)
    N = len(homes)
    home = np.array(homes, np.float32)
    hs = np.array([[L.hash01(i, j) for j in range(10)] for i in range(N)])
    rel = home + BLK / 2 - ORBIT_C
    return dict(N=N, home=home, ta=np.array(ta), tb=np.array(tb), hasA=np.array(ha), hasB=np.array(hb), hs=hs,
                phi0=np.arctan2(rel[:, 1] / 0.6, rel[:, 0]))


def _float_dy(t):
    return int(round(4 * np.sin(2 * np.pi * (L.twos(t) - M_0) / 2.2)))


def _block_state(t):
    """每块在 t 时的 (x, y, 纹理 0=无 1=a 2=白闪 3=b)。全部解析，纯函数。"""
    B = _blocks()
    hs, home = B['hs'], B['home']
    N = B['N']
    fdy = _float_dy(t)
    # 集合：从爆炸点周围飞回家
    ex, ey = 343.0, 195.0
    try:
        ex, ey = _o_hit_point()
    except Exception:
        pass
    th0 = hs[:, 0] * 2 * np.pi
    s0 = np.stack([ex + hs[:, 1] * 80 * np.cos(th0), ey + hs[:, 1] * 60 * np.sin(th0)], 1)
    kin0 = np.clip((t - (M_0 + hs[:, 2] * 0.30)) / 0.42, 0, 1)
    kin0 = 1 - (1 - kin0) ** 3
    homef = home + np.array([0, fdy])
    pos = s0 + (homef - s0) * kin0[:, None]
    # 飞散：按半拍一批
    batch = np.floor(hs[:, 3] * 4)
    tb = M_ATOMS + batch * (60 / 132 / 2)
    kout = np.clip((t - tb) / 0.40, 0, 1)
    kout = kout * kout * (3 - 2 * kout)
    R = 80 + hs[:, 4] * 200
    om = 2.2 + hs[:, 5] * 1.6
    phi = B['phi0'] + om * np.maximum(t - tb, 0)
    orbit = np.stack([ORBIT_C[0] + R * np.cos(phi), ORBIT_C[1] + 0.6 * R * np.sin(phi)], 1) - BLK / 2
    pos = np.where((t >= tb)[:, None], pos + (orbit - pos) * kout[:, None], pos)
    # 回程：rearranging 拖长音里陆续飞回
    tr = 51.25 + hs[:, 6] * 1.05
    kb = np.clip((t - tr) / 0.36, 0, 1)
    kb = kb * kb * (3 - 2 * kb)
    pos = np.where((t >= tr)[:, None], pos + (homef - pos) * kb[:, None], pos)
    # 纹理：空中白闪换装
    tsw = M_REARR + hs[:, 7] * 0.30
    tex = np.where(t < tsw, np.where(B['hasA'], 1, 0),
                   np.where(t < tsw + 0.09, 2, np.where(B['hasB'], 3, 0)))
    # 只有 b 有的块：白闪前不可见；只有 a 有的块：白闪后消失
    tex = np.where((tex == 2) & ~(B['hasA'] | B['hasB']), 0, tex)
    return pos, tex


def _starfield(img, t):
    f = int(np.floor(t * 12))
    for i in range(70):
        x, y = int(L.hash01(i, 51) * W), int(TOP + L.hash01(i, 52) * (BOT - TOP + 40))
        ph = (f + int(L.hash01(i, 53) * 24)) % 24
        if ph < 3: continue
        c = WGOLD if i % 3 == 0 else GOLD_DK
        s = 2 if (i % 5 == 0 and ph < 12) else 1
        rect(img, x, y, x + s, y + s, c)


def _rays(img, t, cx, cy):
    """亮相：身后的金色硬边放射光（楔形平涂，慢慢转）。"""
    k = np.clip((t - (M_LAND - 0.06)) / 0.25, 0, 1)
    if k <= 0: return
    n = 14
    rot = 0.25 * (L.twos(t) - M_LAND)
    for i in range(n):
        a0 = rot + i * 2 * np.pi / n
        a1 = a0 + np.pi / n * 0.55
        ln = (330 if i % 2 == 0 else 230) * k
        poly = np.int32([[cx, cy], [cx + ln * np.cos(a0), cy + ln * np.sin(a0)], [cx + ln * np.cos(a1), cy + ln * np.sin(a1)]])
        cv2.fillPoly(img, [poly], GOLD if i % 2 == 0 else GOLD_DK)


def _sparkles(img, t):
    for i in range(22):
        t0 = M_LAND - 0.05 + L.hash01(i, 61) * 1.1
        u = (L.twos(t) - t0) / 0.35
        if not 0 <= u < 1: continue
        ang = L.hash01(i, 62) * 2 * np.pi
        rr = 60 + L.hash01(i, 63) * 120
        x, y = 320 + rr * np.cos(ang), 140 + rr * 0.9 * np.sin(ang)
        _star(img, x, y, int(14 * (1 - u)) + 2, WHITE, tip=YEL, th=1)


def _atoms_inner(t):
    """变身画面本体（屏幕里的内容，640×360）。"""
    img = np.zeros((H, W, 3), np.uint8)
    _starfield(img, t)
    B = _blocks()
    tt = L.twos(t)
    if t >= M_LAND - 0.06:
        _rays(img, t, 320, 128)
    if M_0 + 0.8 <= tt < M_ATOMS:                                     # 静止漂浮：整张贴，最干净
        can = _cg_canvas(CG_A)
        return _finish_atoms(L.paste_rgba(img, can, 0, _float_dy(tt)), t)
    if tt >= M_LAND:
        can = _cg_canvas(CG_B)
        img = L.paste_rgba(img, can, 0, _float_dy(tt))
        _sparkles(img, t)
        return _finish_atoms(img, t)
    pos, tex = _block_state(tt)
    ta, tb = B['ta'], B['tb']
    order = np.argsort(B['hs'][:, 8])
    for i in order:
        k = tex[i]
        if k == 0: continue
        x, y = int(round(pos[i, 0])), int(round(pos[i, 1]))
        if x <= -BLK or y <= -BLK or x >= W or y >= H: continue
        if k == 2:
            rect(img, x, y, x + BLK, y + BLK, WGOLD)
            continue
        src = ta[i] if k == 1 else tb[i]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + BLK, W), min(y + BLK, H)
        s = src[y0 - y:y1 - y, x0 - x:x1 - x]
        m = s[..., 3] > 127
        reg = img[y0:y1, x0:x1]
        reg[m] = s[..., :3][m]
    return _finish_atoms(img, t)


def _finish_atoms(img, t):
    return img


CRT_FACE = (208, 202, 180)       # 外壳米灰（对照 素材/关键帧/p2_s5_屏幕里.png）
CRT_SIDE = (146, 142, 126)
CRT_BEZEL = (62, 66, 74)
CRT_GLOW = (96, 214, 224)        # 玻璃边的青色光边（关键帧里有）


def _crt(t, inner):
    """后拉：内容缩进一台 CRT 的屏幕里。世界坐标：内容 = (0,0)-(640,360)，玻璃是 4:3 往上下多出黑边。
    终态（54.94）屏幕位置/大小对齐第 5 镜关键帧：玻璃约 x 130–495、y 5–278。"""
    e = ease_io((t - M_PULL0) / (M_1 - M_PULL0))
    s = 1 - 0.43 * e
    cx, cy = 320 - 8 * e, 180 - 39 * e
    def P(x, y): return cx + (x - 320) * s, cy + (y - 180) * s
    out = A._desk().copy()
    # 右侧面（透视感）+ 正面
    side = np.int32([P(706, -104), P(790, -50), P(790, 452), P(706, 500)])
    cv2.fillPoly(out, [side], CRT_SIDE)
    cv2.polylines(out, [side], True, INK, 2, cv2.LINE_8)
    x0, y0 = P(-72, -110)
    x1, y1 = P(708, 502)
    box(out, x0, y0, x1, y1, CRT_FACE, t=3)
    # 下巴：铭牌、软驱缝、电源灯
    lx, ly = P(-20, 448)
    out = text(out, (int(lx), int(ly)), 'CLAUDE-98', INK)
    a0, a1 = P(420, 452), P(600, 468)
    rect(out, a0[0], a0[1], a1[0], a1[1], INK)
    pl = P(640, 458)
    rect(out, pl[0] - 4, pl[1] - 3, pl[0] + 5, pl[1] + 3, GREEN)
    # 玻璃凹框 + 青色光边 + 黑玻璃
    b0, b1 = P(-30, -80), P(670, 440)
    box(out, b0[0], b0[1], b1[0], b1[1], CRT_BEZEL, t=2)
    g0, g1 = P(-14, -64), P(654, 424)
    rect(out, g0[0], g0[1], g1[0], g1[1], CRT_GLOW)
    g0, g1 = P(-11, -61), P(651, 421)
    rect(out, g0[0], g0[1], g1[0], g1[1], BLACK)
    sx0, sy0 = P(0, 0)
    sw, sh = int(round(W * s)), int(round(H * s))
    sm = cv2.resize(inner, (sw, sh), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(sx0)), int(round(sy0))
    ys0, xs0 = max(0, -y0), max(0, -x0)
    ye, xe = min(H, y0 + sh), min(W, x0 + sw)
    out[max(y0, 0):ye, max(x0, 0):xe] = sm[ys0:ys0 + ye - max(y0, 0), xs0:xs0 + xe - max(x0, 0)]
    if s < 0.96:                                                      # 玻璃反光：左上两道硬边斜条
        for off, wd in ((24, 16), (50, 6)):
            q = [P(-11 + off, -61), P(-11 + off + wd, -61), P(-11, -61 + off + wd), P(-11, -61 + off)]
            cv2.fillPoly(out, [np.int32(q)], (190, 214, 222))
    return out


def scene_atoms(t):
    t = float(np.clip(t, M_0, M_1))
    inner = _atoms_inner(t)
    if t < M_PULL0: return inner
    return _crt(t, inner)


# =====================================================================
# 8. calc：1E30 flops 计算器
# =====================================================================
C_0, C_1 = 65.90, 68.58
C_THIRTY, C_FLOPS, C_SECOND = 66.56, 66.88, 67.44
Q4 = 60 / 132 / 4
LCD_BG = (238, 176, 66)
LCD_GHOST = (218, 156, 56)
LCD_DARK = (120, 70, 20)
BODY = (60, 60, 72)
BODY_HI = (104, 104, 120)
KEY = (40, 40, 50)
KEY_HI = (150, 150, 168)
AMBER_BG = (92, 50, 18)
AMBER_BG2 = (120, 68, 24)
RED_KEY = (206, 60, 56)
SMOKE = [(96, 90, 96), (150, 144, 146), (200, 194, 190), (236, 232, 226)]

# 七段：a b c d e f g
SEG = {'0': 'abcdef', '1': 'bc', '2': 'abged', '3': 'abgcd', '4': 'fgbc', '5': 'afgcd', '6': 'afgedc',
       '7': 'abc', '8': 'abcdefg', '9': 'abcdfg', 'E': 'afged', '-': 'g', ' ': ''}


def _seg7(img, x, y, ch, on, off, w=20, h=46, th=4, dot=False):
    segs = {'a': (x + th, y, x + w - th, y + th), 'g': (x + th, y + h // 2 - th // 2, x + w - th, y + h // 2 + th - th // 2),
            'd': (x + th, y + h - th, x + w - th, y + h),
            'f': (x, y + th, x + th, y + h // 2 - th // 2), 'b': (x + w - th, y + th, x + w, y + h // 2 - th // 2),
            'e': (x, y + h // 2 + th // 2, x + th, y + h - th), 'c': (x + w - th, y + h // 2 + th // 2, x + w, y + h - th)}
    lit = SEG.get(ch, '')
    for k, r in segs.items():
        rect(img, *r, on if k in lit else off)
    rect(img, x + w + 3, y + h - th, x + w + 3 + th, y + h, on if dot else off)


def _display(t):
    """(字符串 10 位右对齐, 小数点位置 or None, 状态) 状态: 'num' / 'exp' / 'err'。"""
    if t < 65.95: return '0', 9, 'num'
    if t < C_THIRTY:
        k = min(int((t - 65.95) / Q4), 5)
        return '1' + '0' * k, 9, 'num'
    if t < 67.0: return '1E30', None, 'exp'
    if t < C_SECOND:
        seq = ['1E31', '1E33', '1E38', '1E47', '1E99']
        return seq[min(int((t - 67.0) / (Q4 * 0.8)), 4)], None, 'exp'
    return 'E', None, 'err'


def _press_times():
    """每次显示变化 = Claude 按一次键。"""
    ts = [65.95 + k * Q4 for k in range(6)] + [C_THIRTY] + [67.0 + k * Q4 * 0.8 for k in range(5)] + [C_SECOND]
    return ts


def _last_press(t):
    ps = [p for p in _press_times() if p <= t]
    return (ps[-1], len(ps) - 1) if ps else (None, -1)


KEYS = [['AC', 'EXP', 'x²', '√'], ['7', '8', '9', '÷'], ['4', '5', '6', '×'], ['1', '2', '3', '-'], ['0', '.', '=', '+']]


def _key_for_press(idx):
    if idx < 0: return None
    if idx == 0: return '1'
    if idx < 6: return '0'
    if idx == 6: return 'EXP'
    if idx < 12: return ['3', '1', '8', '7', '9'][(idx - 7) % 5]
    return '×'


@functools.lru_cache(maxsize=None)
def _calc_bg():
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = AMBER_BG
    for k in range(-H, W, 22):                                       # 斜纹桌面
        cv2.line(img, (k, H), (k + H, 0), AMBER_BG2, 6, cv2.LINE_8)
    return img


def _calc_body(img, t, shake):
    dx, dy = shake
    X0, Y0, X1, Y1 = 46 + dx, 14 + dy, 392 + dx, 420 + dy
    rect(img, X0 + 10, Y0 + 10, X1 + 10, Y1 + 10, INK)               # 硬阴影
    box(img, X0, Y0, X1, Y1, BODY, t=3)
    rect(img, X0 + 3, Y0 + 3, X1 - 3, Y0 + 7, BODY_HI)
    rect(img, X0 + 3, Y0 + 3, X0 + 7, Y1 - 3, BODY_HI)
    img = text(img, (X0 + 18, Y0 + 12), 'fx-98', PAPER)
    img = text(img, (X0 + 70, Y0 + 12), 'SCIENTIFIC', (170, 170, 186))
    rect(img, X1 - 118, Y0 + 12, X1 - 18, Y0 + 28, INK)              # 太阳能板
    for k in range(4):
        rect(img, X1 - 116 + k * 24, Y0 + 14, X1 - 96 + k * 24, Y0 + 26, (86, 48, 34))
    # 液晶
    lx0, ly0, lx1, ly1 = X0 + 18, Y0 + 38, X1 - 18, Y0 + 124
    box(img, lx0, ly0, lx1, ly1, (36, 36, 44), t=3)
    d, dot, st = _display(t)
    blink_off = st == 'err' and int(np.floor((t - C_SECOND) * 12)) % 2 == 1
    flash = (C_THIRTY <= t < C_THIRTY + 0.06) or (C_FLOPS <= t < C_FLOPS + 0.06)
    bg = LCD_DARK if flash else LCD_BG
    rect(img, lx0 + 8, ly0 + 8, lx1 - 8, ly1 - 8, bg)
    on = LCD_BG if flash else INK
    ghost = LCD_DARK if flash else LCD_GHOST
    n = 10
    cw = 28
    gx0 = lx1 - 16 - n * cw
    s = d.rjust(n) if st != 'err' else d.ljust(n)
    for i, ch in enumerate(s):
        if blink_off: ch = ' '
        _seg7(img, gx0 + i * cw, ly0 + 30, ch, on, ghost, dot=(dot == i))
    # 小字指示符
    if st == 'exp' or (st == 'err' and not blink_off):
        img = text(img, (lx0 + 12, ly0 + 10), 'SCI' if st == 'exp' else 'ERROR', on)
    if t >= C_FLOPS and st != 'err':
        img = text(img, (lx1 - 16 - text_w('FLOP/s'), ly0 + 10), 'FLOP/s', on)
    if st == 'err' and not blink_off:
        img = text(img, (lx1 - 16 - text_w('OVERFLOW'), ly0 + 10), 'OVERFLOW', on)
    # 按键
    pt, pidx = _last_press(t)
    pressed = _key_for_press(pidx) if pt is not None and t - pt < 0.07 else None
    kw, kh, gapx, gapy = 64, 30, 16, 8
    ky = Y0 + 140
    for r, row in enumerate(KEYS):
        for c, lab in enumerate(row):
            x = X0 + 26 + c * (kw + gapx)
            y = ky if r == 0 else ky + 36 + (r - 1) * (kh + gapy)
            hgt = 26 if r == 0 else kh
            face = ORANGE if lab == '=' else (RED_KEY if lab == 'AC' else (TITLE if r == 0 else KEY))
            down = lab == pressed
            o = 3 if down else 0
            if not down:
                rect(img, x + 3, y + 4, x + kw + 3, y + hgt + 4, INK)
            box(img, x + o, y + o, x + kw + o, y + hgt + o, KEY_HI if down else face, t=2)
            img = text(img, (x + o + (kw - text_w(lab)) // 2, y + o + (hgt - 16) // 2), lab, INK if down else PAPER)
    return img, (lx0, ly0, lx1, ly1)


def _smoke(img, t, lcd):
    u0 = t - C_SECOND
    if u0 < 0: return
    lx0, ly0, lx1, ly1 = lcd
    N = 170
    for i in range(N):
        tb = C_SECOND + (i / N) ** 0.75 * 1.05
        u = t - tb
        if u < 0: continue
        src = L.hash01(i, 71)
        if src < 0.15:                                                # 一部分从 Claude 手里的小计算器冒
            x0, y0 = 552 + (L.hash01(i, 72) - .5) * 24, 128
        else:
            x0, y0 = lx0 + L.hash01(i, 72) * (lx1 - lx0), ly0 + L.hash01(i, 73) * 30
        vy = -(26 + L.hash01(i, 74) * 60)
        vx = (L.hash01(i, 75) - .5) * 70
        x, y = x0 + vx * u, y0 + vy * u
        r = 6 + L.hash01(i, 76) * 6 + 70 * u + 40 * u * u
        c = SMOKE[1 + int(L.hash01(i, 77) * 3)]
        cv2.circle(img, (int(x + 3), int(y + 4)), int(r), SMOKE[0], -1, cv2.LINE_8)   # 硬阴影
        cv2.circle(img, (int(x), int(y)), int(r), c, -1, cv2.LINE_8)
    # 最后从下往上涌的大团，保证 68.58 遮满
    for j in range(9):
        tb = 68.02 + j * 0.035
        u = t - tb
        if u < 0: continue
        x = 40 + j * 72
        y = 330 - 90 * u
        r = 40 + 420 * u
        cv2.circle(img, (int(x + 4), int(y + 5)), int(r), SMOKE[1], -1, cv2.LINE_8)
        cv2.circle(img, (int(x), int(y)), int(r), SMOKE[2 + j % 2], -1, cv2.LINE_8)
    k = (t - 68.40) / (C_1 - 68.40)                                   # 保险：最后网点溶解成满屏浅烟
    if k > 0:
        img[L.bayer_mask(k)] = SMOKE[2]


def scene_calc(t):
    t = float(np.clip(t, C_0, C_1))
    img = _calc_bg().copy()
    shake = (0, 0)
    if C_SECOND <= t < C_SECOND + 0.35:
        f = int(np.floor(t * 24))
        a = 4 * (1 - (t - C_SECOND) / 0.35)
        shake = (int(round((L.hash01(f, 81) - .5) * 2 * a)), int(round((L.hash01(f, 82) - .5) * 2 * a)))
    img, lcd = _calc_body(img, t, shake)
    # Claude：姿势 2，拼命按
    spr = _q2(1, 236)
    tt = L.twos(t)
    pt, _ = _last_press(tt)
    dy = 3 if (pt is not None and tt - pt < 0.09) else 0
    f = int(np.floor(t * 24))
    jx = int(round((L.hash01(f, 83) - .5) * 3)) if t < C_SECOND + 0.5 else 0
    img = L.paste_rgba(img, spr, 420 + jx, 40 + dy)
    if pt is not None and tt - pt < 0.12 and t < C_SECOND:            # 按键火花线
        for k in range(3):
            a = -0.9 + k * 0.5
            x0, y0 = 590 + 14 * np.cos(a), 150 + 14 * np.sin(a)
            cv2.line(img, (int(x0), int(y0)), (int(x0 + 12 * np.cos(a)), int(y0 + 12 * np.sin(a))), YEL, 2, cv2.LINE_8)
    _smoke(img, t, lcd)
    return img


SCENES = [
    ('stable', S_0, S_1, scene_stable),
    ('singularity', G_0, G_1, scene_singularity),
    ('optimize', O_0, O_1, scene_optimize),
    ('atoms', M_0, M_1, scene_atoms),
    ('calc', C_0, C_1, scene_calc),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
