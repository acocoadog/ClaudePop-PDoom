"""codeshots_b: 第一部分的代码镜头 —— 第 7 镜中文房间 / 第 8 镜修格斯 X 光（顶光锥 load_cone 给 codeshots_p1r 的 gameover 用）。

接口遵守 part1_lib 文件头的约定：scene_xxx(t) -> (360,640,3) uint8 RGB，t 是完整版歌曲秒数，纯函数。
自检：python codeshots_b.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import part1_lib as L

W, H = L.W, L.H
BEAT = L.BEAT132
INK = (12, 11, 26)


def beat_time(n):
    """132 BPM 网格第 n 拍的时刻。"""
    return L.B132_0 + n * BEAT


def frame24(t):
    return int(np.floor(t * 24 + 1e-6))


# ---------------- 像素小图（心形、星星） ----------------
HEART = ['.##.##.',
         '#######',
         '#######',
         '.#####.',
         '..###..',
         '...#...']
STAR = ['...#...',
        '...#...',
        '..###..',
        '#######',
        '..###..',
        '...#...',
        '...#...']


@functools.lru_cache(maxsize=None)
def pixel_sprite(rows, color, scale, hi=None):
    """把字符点阵做成带 1px 墨线的 RGBA 精灵（硬边）。hi: 高光格子 (行,列)。缓存只是加速，不影响纯函数性。"""
    m = np.array([[c == '#' for c in r] for r in rows], np.uint8)
    m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    m = np.pad(m, 1)
    outline = cv2.dilate(m, np.ones((3, 3), np.uint8))
    out = np.zeros(m.shape + (4,), np.uint8)
    out[outline > 0] = INK + (255,)
    out[m > 0] = tuple(color) + (255,)
    if hi is not None:
        r, c = hi
        out[1 + r * scale:1 + (r + 1) * scale, 1 + c * scale:1 + (c + 1) * scale] = (255, 255, 255, 255)
    return out


# ================================================================
# 顶光锥（21.60–22.43 黑场结尾）：原来的黑场 NOW LOADING 镜头已换成 codeshots_p1r 的 gameover，只留这个
# ================================================================
LOAD_T1 = 22.43
LOAD_CONE0 = LOAD_T1 - 0.15                 # 最后 0.15 秒打顶光锥


def load_cone(img, t, t0=LOAD_CONE0):
    """顶光锥（暗紫蓝，两层平涂，0.08 秒从上往下打下来），接舞台的顶光。codeshots_p1r 的 gameover 也用。"""
    if t < t0: return img
    p = np.clip((t - t0) / 0.08, 0, 1)
    ybot = int(p * H)
    yy = np.arange(ybot)
    for half0, half1, col in ((5, 78, (34, 32, 92)), (2, 30, (52, 50, 132))):
        half = (half0 + (half1 - half0) * yy / H).astype(int)
        for y, hw in zip(yy, half):
            img[y, W // 2 - hw:W // 2 + hw] = col
    img[0:4, W // 2 - 6:W // 2 + 6] = (120, 118, 200)          # 灯口：一小块亮色
    return img


# ================================================================
# 第 7 镜：中文房间（25.96–29.58）
# ================================================================
ROOM_T0, ROOM_T1 = 25.96, 29.58
ROOM_CG = '素材/CG/s7_中文房间.png'
T_ROOM_HIT = 27.32          # "room"：镜头轻震
T_SHROOM = 29.14            # "shrooms"：迷幻开始
T_COLLAPSE = 29.40          # 收拢成紫绿
SLOT = (570.0, 112.0)       # 投信口下唇（640×360 坐标，已在图上量过）
SHROOM = (491.0, 280.0)     # 蘑菇纸袋中心
ROOM_FOCUS = (300.0, 170.0)  # 推镜最终中心（偏向 Claude）
SLIP_BEAT0 = 21             # 第 21 拍 = 26.18 秒，第一张纸条
SLIP_WORDS = ['你好', '意识？', '我在吗', '中文', '规则', '理解', '思考', '懂吗？', '房间', '是谁']
SLIP_FONT = '/System/Library/Fonts/Hiragino Sans GB.ttc'
PAPER, PAPER_BACK, SLIP_INK = (242, 234, 212), (196, 184, 156), (34, 26, 30)


@functools.lru_cache(maxsize=1)
def room_src():
    """原图全分辨率，推镜时从这里裁再缩，避免二次放大发糊。"""
    return L.load_rgb(ROOM_CG, 1672, 941)


@functools.lru_cache(maxsize=None)
def slip_sprite(word, back=False):
    """纸条精灵 RGBA：纸色平涂 + 1px 墨框 + 1-bit 汉字（像素字体缺简体字，这里用冬青黑体 14px 关抗锯齿）。"""
    f = ImageFont.truetype(SLIP_FONT, 14, index=1)
    l, t_, r, b = f.getbbox(word)
    w, h = r - l + 10, 22
    im = Image.new('RGBA', (w, h), SLIP_INK + (255,))
    d = ImageDraw.Draw(im)
    d.rectangle((1, 1, w - 2, h - 2), fill=(PAPER_BACK if back else PAPER) + (255,))
    if not back:
        m = Image.new('1', (w, h), 0)
        ImageDraw.Draw(m).text((5 - l, (h - (b - t_)) // 2 - t_), word, fill=1, font=f)
        a = np.array(im)
        a[np.array(m)] = SLIP_INK + (255,)
        return a
    a = np.array(im)
    a[h // 2, 4:w - 4] = (170, 158, 130, 255)   # 背面一道折痕
    return a


def warp_sprite(spr, sy, ang):
    """纵向压扁 sy（模拟翻转）再旋转 ang 度，最近邻保持硬边。返回 RGBA 和中心在其中的位置。"""
    h, w = spr.shape[:2]
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    A = np.array([[c, -s * sy], [s, c * sy]])
    R = int(np.ceil(np.hypot(w, h) / 2)) + 2
    M = np.hstack([A, (np.array([R, R]) - A @ np.array([w / 2, h / 2]))[:, None]])
    out = cv2.warpAffine(spr, M, (2 * R, 2 * R), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0))
    return out, R


def slip_state(n, t):
    """第 n 拍出生的纸条在 t 时刻的状态（世界坐标）：None 或 (阶段, x, y, 纵向缩放, 角度, 是否背面)。"""
    tb = beat_time(n)
    tau = t - tb
    if tau < 0: return None
    h = lambda j: L.hash01(n, j)
    if tau < 0.10:                                  # 从投信口滑出来
        return ('emerge', SLOT[0] + (h(1) - .5) * 8, SLOT[1], tau / 0.10, -12.0, False)
    u = tau - 0.10
    vy, g = 60 + 60 * h(3), 520.0
    y_land = 268 + 26 * h(4)
    # y(u) = y0 + vy u + g u²/2 = y_land 的解
    y0 = SLOT[1] + 11
    T = (-vy + np.sqrt(vy * vy + 2 * g * (y_land - y0))) / g
    # 落点 x：避开蘑菇纸袋（x≈470–515），一半落在门前、一半飘到书堆前
    x_land = 522 + 50 * h(2) if n % 2 else 350 + 110 * h(2)
    vx = (x_land - SLOT[0]) / T
    uu = min(u, T)
    sway = 10 * np.sin(9 * uu + 6 * h(5)) * (1 - uu / T)   # 落地时摆动归零，落点准确
    x = SLOT[0] + vx * uu + sway
    y = y0 + vy * uu + 0.5 * g * uu * uu
    if u < T:
        th = (5 + 3 * h(6)) * uu                       # 翻转角（弧度），刚出来时正面朝前
        sy = np.cos(th)
        ang = -12 + (140 * h(8) - 70) * uu / T
        return ('fly', x, y, max(abs(sy), 0.2), ang, sy < 0)
    return ('land', x, y, 0.8, -25 + 50 * h(9), h(10) < 0.15)


def room_camera(t):
    """推镜：返回 (缩放 s, 裁剪左上角 x0,y0)（640 坐标）。27.32 轻震一下。
    09-29 v6：蘑菇纸袋原本被对白框挡住 → "bag"（28.30）前顺着飘落的纸条往下摇到纸袋特写，蘑菇放在画面中上部。"""
    def push(tt):
        k = L.smooth((tt - ROOM_T0) / (ROOM_T1 - ROOM_T0))
        s = 1.0 + 0.06 * k
        return s, 320 + (ROOM_FOCUS[0] - 320) * k, 180 + (ROOM_FOCUS[1] - 180) * k
    s, cx, cy = push(min(t, T_PAN0))
    if t > T_PAN0:
        u = L.smooth((t - T_PAN0) / (T_PAN1 - T_PAN0))
        s2 = BAG_S * (1 + .08 * max(0.0, t - T_PAN1) / (ROOM_T1 - T_PAN1))   # 到了以后继续慢慢推近蘑菇
        cx2 = SHROOM[0] - (BAG_SCREEN[0] - W / 2) / s2                     # 让蘑菇落在屏幕 BAG_SCREEN
        cy2 = SHROOM[1] - (BAG_SCREEN[1] - H / 2) / s2
        s = s * (s2 / s) ** u
        cx, cy = cx + (cx2 - cx) * u, cy + (cy2 - cy) * u
    cw, ch = W / s, H / s
    x0 = float(np.clip(cx - cw / 2, 0, W - cw))
    y0 = float(np.clip(cy - ch / 2, 0, H - ch))
    return s, x0, y0


# ---------------- v6：摇到纸袋 + 蘑菇占领房间 ----------------
T_PAN0, T_PAN1 = 28.12, 28.30   # 顺着纸条往下摇，"bag" 时停在纸袋特写
BAG_S = 2.4                     # 纸袋特写的放大倍数（原图 1672 宽，放大 2.4 倍仍比 640 清楚）
BAG_SCREEN = (320, 170)         # 蘑菇脸落在屏幕的位置（对白框从 y=290 开始）
T_PEEK = 28.84                  # "of"：蘑菇从袋子里往上一探
SHROOM_SHEET = '素材/精灵/蘑菇_六姿势.png'


@functools.lru_cache(maxsize=1)
def shroom_sprites():
    """蘑菇精灵表（xk-draw 参考原图那只蘑菇画的，透明底）切成 6 张，按脚底线对齐。"""
    import sprite_demo
    return sprite_demo.cut_sheet(str(L.ROOT / SHROOM_SHEET), n=6, height=120)


@functools.lru_cache(maxsize=1)
def swarm():
    """蘑菇大军的出生表（屏幕坐标）：(出生时间, 脚底 x, 脚底 y, 高度倍数, 姿势相位, 是否翻转)，按 y 排序（远的先画）。"""
    out = []
    for i in range(72):
        x = L.hash01(i, 71) * (W + 60) - 30
        y = 40 + L.hash01(i, 72) * (H + 30)
        d = np.hypot(x - BAG_SCREEN[0], (y - BAG_SCREEN[1] - 30) * 1.4)          # 离纸袋越远越晚冒
        born = T_SHROOM + .02 + .24 * min(d / 520, 1) + .03 * L.hash01(i, 73)
        size = .22 + .40 * L.hash01(i, 74) + .0016 * y                       # 越靠下（越近）越大；最大约 120 像素高，屏幕上能装下几十只
        out.append((born, x, y, size, int(L.hash01(i, 75) * 6), L.hash01(i, 76) < .5))
    return sorted(out, key=lambda m: m[2])


def draw_swarm(img, t):
    """29.14 起：蘑菇以纸袋为中心一圈圈往外冒，两帧一批（一拍二），冒出来时先大一号再落定，之后每 4 帧换个姿势。"""
    tt = L.twos(t)
    sprites = shroom_sprites()
    for born, x, y, size, ph, flip in swarm():
        if tt < born: continue
        age = tt - born
        pop = 1.25 if age < 1 / 12 else 1.0
        pose = (ph + int(age * 6)) % len(sprites)
        spr, lift = sprites[pose]
        sc = size * pop
        h, w = spr.shape[:2]
        s = cv2.resize(spr, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_NEAREST)
        if flip: s = s[:, ::-1]
        img = L.paste_rgba(img, s, x - s.shape[1] / 2, y - s.shape[0] - lift * sc)
    return img


def room_shake(t):
    """27.32 "room" 的轻震：屏幕空间整数位移，24fps 抖、指数衰减。"""
    if t < T_ROOM_HIT: return 0, 0
    a = 6.0 * np.exp(-(t - T_ROOM_HIT) * 8)
    f = frame24(t)
    return int(round(a * (L.hash01(f, 31) - .5) * 2)), int(round(a * (1 if f % 2 else -1)))


def hue_bands(img, level, sat_min=170):
    """按离散档位做色相旋转（每档 45°）+ 提饱和，level: (H,W) int，<0 表示不动。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    on = level >= 0
    hh = hsv[..., 0].astype(np.int32)
    hsv[..., 0] = np.where(on, (hh + (level % 8) * 22 + 180) % 180, hh).astype(np.uint8)
    hsv[..., 1] = np.where(on, np.maximum(hsv[..., 1], sat_min), hsv[..., 1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def scene_room(t):
    # --- 推镜：从全分辨率原图裁一块再缩到 640×360 ---
    s, x0, y0 = room_camera(t)
    src = room_src()
    k = src.shape[1] / W
    cw, ch = W / s, H / s
    X0, Y0 = int(round(x0 * k)), int(round(y0 * k))
    crop = src[Y0:Y0 + int(round(ch * k)), X0:X0 + int(round(cw * k))]
    img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    to_scr = lambda wx, wy: ((wx - x0) * s, (wy - y0) * s)

    # --- 纸条：每拍一张，按出生顺序画（后出生的在上面） ---
    tt = L.twos(t)                                 # 纸条按 12fps 一拍二
    n_last = int(np.floor((t - L.B132_0) / BEAT))
    for n in range(SLIP_BEAT0, n_last + 1):
        st = slip_state(n, tt)
        if st is None: continue
        phase, wx, wy, sy, ang, back = st
        word = SLIP_WORDS[(n - SLIP_BEAT0) % len(SLIP_WORDS)]
        spr = slip_sprite(word, back)
        sx, syy = to_scr(wx, wy)
        if phase == 'emerge':
            # 贴着投信口往下长出来：只画上面 sy 比例的部分，顶边对齐投信口
            hh = max(1, int(round(spr.shape[0] * sy)))
            part, _ = warp_sprite(spr[-hh:], 1.0, ang)
            img = L.paste_rgba(img, part, sx - part.shape[1] / 2, syy - part.shape[0] / 2 + hh / 2)
        else:
            part, R = warp_sprite(spr, sy, ang)
            img = L.paste_rgba(img, part, sx - R, syy - R)

    # --- 迷幻：从蘑菇纸袋往外扩的离散色环（调色板循环感） ---
    if t >= T_SHROOM:
        mx, my = to_scr(*SHROOM)
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - mx, yy - my)
        tl = t - T_SHROOM
        R = 1500.0 * tl + 20
        bw = 26.0
        cyc = int(np.floor(tl * 24))            # 色带每帧往外挪一档
        band = np.floor((R - d) / bw).astype(np.int32)
        level = np.where(d < R, band + cyc, -1)
        psy = hue_bands(img, level)
        # 色带边界画 1px 墨线（平涂硬边的味道）
        edge = np.zeros((H, W), bool)
        edge[:, 1:] |= level[:, 1:] != level[:, :-1]
        edge[1:, :] |= level[1:, :] != level[:-1, :]
        edge &= d < R
        psy[edge] = (psy[edge].astype(np.int32) * 2 // 5).astype(np.uint8)
        img = draw_swarm(psy, t)                  # v6：色环里冒出一整屋子蘑菇（收拢成紫绿时一起被染色）

        # 29.40–29.58：收拢成紫、绿两色（下一镜修格斯的主色），Bayer 网点溶进去
        if t >= T_COLLAPSE:
            kk = np.clip((t - T_COLLAPSE) / (ROOM_T1 - 0.03 - T_COLLAPSE), 0, 1)
            lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            par = ((np.floor((R - d) / bw).astype(np.int32) + cyc) % 2) == 0
            two = np.empty_like(img)
            dark = lum < 95
            two[par & dark] = (48, 24, 82)
            two[par & ~dark] = (118, 62, 168)
            two[~par & dark] = (44, 78, 34)
            two[~par & ~dark] = (150, 196, 74)
            two[edge] = (26, 14, 44)
            m = L.bayer_mask(kk)
            img = np.where(m[..., None], two, img)
    if T_PAN0 < t < T_PAN1:                        # 摇镜时的纵向拖影
        n = int(2 + 22 * np.sin(np.pi * (t - T_PAN0) / (T_PAN1 - T_PAN0)))
        img = cv2.blur(img, (1, n))
    dx, dy = room_shake(t)
    dy -= int(round(4 * np.sin(np.pi * (t - T_PEEK) / .15))) if 0 <= t - T_PEEK < .15 else 0   # "of"：蘑菇一探，画面往上一弹
    if dx or dy: img = shift_img(img, dx, dy)
    return img


# ================================================================
# 第 8 镜：修格斯 X 光（29.58–32.70）
# ================================================================
SHOG_T0, SHOG_T1 = 29.58, 32.70
SCAN_T0, T_LIES, SCAN_T1 = 29.90, 30.94, 31.70
MASK_X = 329                       # 笑脸面具中心 x（1672 宽图上约 860 → 51.4%）
BAND_W = 26
S8A, S8B = '素材/CG/s8a_修格斯面具.png', '素材/CG/s8b_修格斯真身.png'
# 额外浮动的心形 / 星星：(x, y, 种类, 相位)
# （避开 CG 自带的：星 (209,29)(112,117)(456,69)(549,148)，心 (170,59)(81,145)(434,35)(518,108)）
FLOATERS = [(40, 62, 'heart', 0), (252, 30, 'star', 1), (505, 30, 'star', 2), (602, 42, 'heart', 3),
            (30, 196, 'star', 2), (612, 160, 'heart', 1), (118, 34, 'star', 3), (590, 236, 'star', 0)]
# 大眼睛位置和半径（640 坐标，从 s8b 量的）
EYES = [(358, 195, 27), (279, 82, 17), (390, 110, 16), (178, 233, 16), (534, 104, 13),
        (72, 147, 12), (457, 281, 14), (597, 274, 13), (339, 251, 11), (452, 191, 10),
        (61, 252, 12), (436, 48, 10)]


def scan_x(t):
    """扫描线前沿 x。锚点 (29.90,0) → (30.94, 面具中心) → (31.70, 满宽)，再按 132 拍一顿一顿。"""
    # 拍内时间扭曲：拍头快、拍尾慢；30.94 处拍内相位 ≈0.48，扭曲量 <0.01 秒 ≈2px
    ph = L.beat_phase(t)[1]
    tw = t + 0.9 * BEAT * np.sin(2 * np.pi * ph) / (2 * np.pi)
    return float(np.interp(tw, [SCAN_T0, T_LIES, SCAN_T1], [0, MASK_X, W + BAND_W + 4]))


def xray(img):
    """X 光色：按亮度压成三档绿，隔行扫描纹。"""
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    out = np.empty_like(img)
    out[:] = (70, 230, 90)
    out[lum < 60] = (20, 110, 40)
    out[lum > 140] = (200, 255, 190)
    out[::3] = (out[::3].astype(np.int32) * 3 // 5).astype(np.uint8)
    return out


def shift_img(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def scene_shoggoth(t):
    A = L.load_rgb(S8A).copy()
    B = L.load_rgb(S8B)

    # 面具版：心形和星星跟拍子一起一伏（24fps）
    if t < SCAN_T1:
        p = L.pulse(t, 5.0)
        for i, (x, y, kind, ph) in enumerate(FLOATERS):
            bob = int(round(-5 * p * (1 if (ph % 2) else 0.6) + 2 * np.sin(2 * np.pi * (t * 1.1 + ph / 4))))
            if kind == 'heart':
                spr = pixel_sprite(tuple(HEART), (255, 128, 178), 3, (1, 1))
            else:
                spr = pixel_sprite(tuple(STAR), (255, 220, 70), 2 if ph % 2 else 3)
            A = L.paste_rgba(A, spr, x - spr.shape[1] // 2, y + bob - spr.shape[0] // 2)

    if t < SCAN_T0:
        return A

    if t < SCAN_T1:
        sx = int(round(scan_x(t)))
        img = A.copy()
        if sx > 0:
            img[:, :sx] = B[:, :sx]
        # X 光带：前沿左侧 BAND_W 宽的绿色光带
        b0, b1 = max(sx - BAND_W, 0), min(max(sx, 0), W)
        if b1 > b0:
            img[:, b0:b1] = xray(B[:, b0:b1])
            if b0 > 0: img[:, b0 - 1] = INK
        # 前沿细白线 + 墨线
        if 0 <= sx < W:
            img[:, sx:min(sx + 2, W)] = (236, 255, 236)
            if sx + 2 < W: img[:, sx + 2] = INK
        # CRT 故障：扫描线附近几组横向错位（24fps 每帧重抽）
        f = frame24(t)
        g0, g1 = max(sx - 70, 0), min(sx + 70, W)
        if g1 - g0 > 8:
            for j in range(7):
                y = int(L.hash01(f, j, 1) * 270) + 18
                hgt = 2 + int(L.hash01(f, j, 2) * 5)
                dx = int((L.hash01(f, j, 3) - .5) * 28)
                img[y:y + hgt, g0:g1] = np.roll(img[y:y + hgt, g0:g1], dx, axis=1)
        return img

    # 31.70 以后：全真身，每拍震屏 + 眼睛闪光
    n, ph = L.beat_phase(t)
    p = float(np.exp(-7 * ph))
    amp = int(round(7 * p))
    dx = amp * (1 if n % 2 else -1)
    dy = int(round(amp * 0.5)) * (1 if (n // 2) % 2 else -1)
    img = shift_img(B, dx, dy)
    if p > 0.3:
        big = int(round(p * 3))
        for i, (ex, ey, r) in enumerate(EYES):
            if L.hash01(n, i, 5) < 0.35 and i != 0: continue   # 每拍换一批，最大那只每拍都闪
            sc = 2 if r < 14 else 3
            # 眼白上一块硬边高光圆 + 一颗十字星
            hx, hy = ex - r * 0.35 + dx, ey - r * 0.4 + dy
            rr = max(2, int(round(r * 0.28)))
            cv2.circle(img, (int(hx), int(hy)), rr + 1, INK, -1, lineType=cv2.LINE_4)
            cv2.circle(img, (int(hx), int(hy)), rr, (255, 255, 240), -1, lineType=cv2.LINE_4)
            spr = pixel_sprite(tuple(STAR), (255, 255, 240), sc + (big > 2))
            img = L.paste_rgba(img, spr, hx - spr.shape[1] / 2, hy - spr.shape[0] / 2)
    # 最后 0.2 秒两级压暗，准备硬切
    if t >= SHOG_T1 - 0.2:
        fac = 0.55 if t < SHOG_T1 - 0.1 else 0.28
        img = (img.astype(np.float32) * fac).astype(np.uint8)
    return img


SCENES = [
    # 第 7 镜拆两段、同一个函数：迷幻段（0.44 秒）要单独锁调色板，
    # 否则整镜抽样的 16 色全被旧纸黄占满，紫绿色带会被量化成黄/灰蓝（已实测）
    ('room', ROOM_T0, T_SHROOM, scene_room),
    ('room_trip', T_SHROOM, ROOM_T1, scene_room),
    ('shoggoth', SHOG_T0, SHOG_T1, scene_shoggoth),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
