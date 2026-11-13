"""codeshots_a：第一部分的纯代码镜头 A 组（分镜第 0、2、3、10 镜）。

    boot      0.00– 1.20  PC-98 开机自检 → 0.80 起镜头后拉，露出 s1_火花之眼 关键帧（1.20 与关键帧逐像素一致）
    circuit   5.64– 9.06  电路板俯视，走线爬向 Q 版 Claude；nervous 变红，surprise 摊手，结尾收拢到中心芯片
    loss      9.06–12.52  TRAINING.EXE 窗口弹出，loss 平台抖动 → 10.54 drop 曲线砸穿窗口底边掉出画面

接口约定见 part1_lib 文件头：scene_xxx(t) -> (360,640,3) uint8 RGB，t 为完整版歌曲秒数，纯函数。
自检：python codeshots_a.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np

import part1_lib as L

W, H = L.W, L.H
TOP, BOT = 18, 284            # HUD 留白：y<18 顶栏，y≥284 对白框

# ---------------- 调色（颜色少而明确） ----------------
INK = (12, 11, 26)
PAPER = (246, 241, 228)
WINBG = (232, 226, 206)       # 窗体米白
TITLE = (34, 52, 150)         # 标题栏藏青
YEL = (255, 210, 63)
ORANGE = (217, 119, 87)
NAVY = (16, 22, 54)           # 深藏青底
NAVY_DOT = (38, 50, 104)
PCB = (30, 104, 64)           # PCB 绿
PCB_DARK = (20, 72, 46)
COPPER = (54, 140, 84)        # 未点亮的走线
PAD = (214, 176, 80)          # 焊盘金
CHIP = (28, 28, 40)
CHIP_HI = (70, 72, 96)
PIN = (190, 192, 204)
SILK = (226, 234, 214)        # 丝印白
CYAN = (60, 224, 232)
CYAN_HI = (200, 255, 255)
RED = (236, 52, 60)
RED_HI = (255, 120, 110)
WHITE = (255, 255, 255)
PLOT_GRID = (184, 188, 206)
LOSS_RED = (214, 44, 40)
GREEN = (80, 232, 112)
GREEN_DIM = (22, 70, 40)
TERM_BG = (8, 14, 12)


# ---------------- 通用小工具 ----------------
def rect(img, x0, y0, x1, y1, c):
    """平涂矩形（半开区间，自动裁剪）。原地修改。"""
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), img.shape[1]), min(int(y1), img.shape[0])
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = c


def box(img, x0, y0, x1, y1, c, ink=INK, t=2):
    """带墨线描边的平涂矩形。"""
    rect(img, x0, y0, x1, y1, ink)
    rect(img, x0 + t, y0 + t, x1 - t, y1 - t, c)


def text_w(s, scale=1):
    return int(L.font(16).getlength(s)) * scale


def text(img, xy, s, c, scale=1):
    return L.draw_text(img, (int(xy[0]), int(xy[1])), s, c, scale=scale)


def text_outline(img, xy, s, c, scale=1, ink=INK, o=None):
    """带墨线描边的大字（先画八方向偏移的墨色，再画本色）。"""
    o = o or scale
    x, y = xy
    for dx in (-o, 0, o):
        for dy in (-o, 0, o):
            if dx or dy:
                img = L.draw_text(img, (x + dx, y + dy), s, ink, scale=scale)
    return L.draw_text(img, (x, y), s, c, scale=scale)


def ease_io(x):
    x = float(np.clip(x, 0, 1))
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


# =====================================================================
# 0. boot：开机自检 + 后拉进 CRT
# =====================================================================
KEY_PATH = '素材/关键帧/s1_火花之眼.png'
KEY_SIZE = (1672, 941)
# 关键帧里 CRT 玻璃（暗青屏幕区）四角，640×360 坐标，顺序 TL TR BR BL。
# 量法：亮青边框掩码的补集取连通域 → 凸包 → approxPolyDP，再对照放大裁剪图人工修正（右上角自动结果溢出到边框外）。
SCREEN = np.float32([[482, 109], [550, 76], [535, 234], [474, 225]])
BOOT_T0, BOOT_T1 = 0.80, 1.20


# 10-19 用户：加载画面要有 Claude 娘的「代码形象」吸引观众 →
# 右边一扇 CLAUDE_CHAN.MAG 窗口，像 PC-98 游戏读 CG 那样隔行扫描载入一张像素立绘（先偶数行、再补奇数行，
# 扫描线头有一条亮线）；载完眨一下眼。
# 10-19 用户：换舞台服——像超时空辉夜姬那样，加载出来的 AI 虚拟主播是来自未来、已经站上舞台的自己
PORT_SRC, PORT_CROP = '素材/角色/设定图_舞台服_v1.png', (150, 6, 440, 302)   # 舞台服设定图正面，带到胸口（用户：全身一点更有数码感）
PORT_PX, PORT_K = 2, 14                  # 立绘 1 像素 = 屏幕 2 像素；14 色（13 色平涂 + 1 墨线）
PORT_INK = (70, 30, 24)                  # 立绘墨线（深棕，和原画线色一致）
PORT_LUM, PORT_THR = 95, .26             # 墨线：原画亮度 < PORT_LUM 算线；缩小后一格里线占比 > PORT_THR 才画线
PORT_BOX = (398, 40, 606, 270)           # 窗口外框（含标题栏）
PORT_T0, PORT_T1 = 0.10, 0.62            # 隔行扫描：前一半偶数行、后一半奇数行（boot 内部时间）


@functools.lru_cache(maxsize=None)
def _portrait():
    """(睁眼, 闭眼) 两张像素立绘 RGB。
    10-19 用户：嘴糊了、左右不对称，脖子下面的阴影很奇怪 → 换成三步：
      ① 背景从画框四周往里漫灌（只抠和边缘连通的灰；原来按颜色相近判背景，把领子的淡紫灰阴影也抠成了深藏青）
      ② 墨线单独提取（亮度 < 95），缩小后按覆盖率 > 26% 一格格盖回去（原来细线被平均进肤色里，嘴角、脸轮廓都没了）
      ③ 平涂区先把线 inpaint 擦掉再缩小，Lab 空间 k-means 13 色，颜色不被墨线染脏
    闭眼 = 眼睛那块盖肤色，再画一道往下弯的闭眼线（^ ^）。坐标都按舞台服设定图量的，换源图要重量。"""
    from PIL import Image
    im = np.array(Image.open(L.ROOT / PORT_SRC).convert('RGB').crop(PORT_CROP))
    h0, w0 = im.shape[:2]
    x0, y0, x1, y1 = PORT_BOX
    w, h = (x1 - x0 - 8) // PORT_PX, (y1 - y0 - 26) // PORT_PX
    ff = np.zeros((h0 + 2, w0 + 2), np.uint8)
    for sx, sy in ((0, 0), (w0 - 1, 0), (0, h0 - 1), (w0 - 1, h0 - 1), (w0 // 2, 0)):
        if np.abs(im[sy, sx].astype(int) - 190).max() < 30:
            cv2.floodFill(im.copy(), ff, (sx, sy), (0, 0, 0), (6, 6, 6), (6, 6, 6),
                          cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    bgm = ff[1:-1, 1:-1] > 0
    ii = im.astype(int)                                              # 发丝缝里和边缘不连通的灰底：几乎无色相、又贴近背景灰
    bgm |= (np.abs(ii - im[0, 0].astype(int)).max(2) < 24) & (ii.max(2) - ii.min(2) < 10)
    ink = (cv2.cvtColor(im, cv2.COLOR_RGB2LAB)[..., 0] < PORT_LUM) & ~bgm
    flat = cv2.inpaint(im, ink.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    small = cv2.resize(flat, (w, h), interpolation=cv2.INTER_AREA)
    bgs = cv2.resize(bgm.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > .5
    inks = cv2.resize(ink.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > PORT_THR
    px = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    sel = ~bgs.reshape(-1)
    cv2.setRNGSeed(7)                                                # k-means 结果固定（纯函数）
    _, lbl, cen = cv2.kmeans(px[sel], PORT_K - 1, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .5),
                             5, cv2.KMEANS_PP_CENTERS)
    cen = cv2.cvtColor(cen.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    q = np.zeros((h * w, 3), np.uint8); q[:] = (28, 30, 58)
    q[sel] = cen[lbl.ravel()]
    q = q.reshape(h, w, 3)
    q[inks & ~bgs] = PORT_INK
    sk = q[50, 50]                                                   # 脸颊取肤色
    nose = q[53:58, 46:53]                                           # 10-19 用户：鼻子缩出来只剩一个小黑点 → 抹成肤色
    nose[(np.abs(nose.astype(int) - sk.astype(int)).sum(2) > 30)] = sk
    shut = q.copy()
    # 10-19 v2 用户：闭眼帧那两块肤色方块太大，刘海、眉毛都被盖平了 → 每只眼只在一个小椭圆里、
    # 只把「不是头发橙色」的像素（眼白、瞳孔、睫毛）换成肤色，刘海照旧压在上面；再画一道往下弯的闭眼线
    hair = (np.abs(q.astype(int) - q[20, 50].astype(int)).sum(2) < 90) | (q[..., 0].astype(int) - q[..., 2] > 110) & (q[..., 1] < 150)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy in ((39, 48), (60, 48)):                               # 两只眼的中心（立绘像素坐标，放大 10 倍量的）
        eye = (((xx - cx) / 7.5) ** 2 + ((yy - cy) / 5.5) ** 2 < 1) & (~hair | (yy >= 47)) & (yy >= 43)   # y≥47 是瞳孔（橙色像头发），一律换掉
        shut[eye] = sk
        for dx in range(-6, 7):                                       # 往下弯的闭眼线（笑眼 ^ ^）
            y_ = cy + 1 + int(round(2.5 * (1 - (dx / 6) ** 2)))
            shut[y_:y_ + 2, cx + dx] = PORT_INK
    return q, shut


def _portrait_window(img, t):
    x0, y0, x1, y1 = PORT_BOX
    box(img, x0, y0, x1, y1, (28, 30, 58), ink=PAPER, t=2)
    rect(img, x0 + 2, y0 + 2, x1 - 2, y0 + 20, TITLE)
    img = text(img, (x0 + 8, y0 + 3), 'CLAUDE_CHAN.MAG', PAPER)
    op, shut = _portrait()
    blink = 0.70 <= t < 0.74                                           # 载完眨一下
    src = shut if blink else op
    h, w = src.shape[:2]
    k = (t - PORT_T0) / (PORT_T1 - PORT_T0)
    if k <= 0: return img
    rows = np.arange(h)
    even = rows[rows % 2 == 0]; odd = rows[rows % 2 == 1]
    n_e = int(np.clip(k * 2, 0, 1) * len(even)); n_o = int(np.clip(k * 2 - 1, 0, 1) * len(odd))
    show = np.zeros(h, bool); show[even[:n_e]] = True; show[odd[:n_o]] = True
    pic = np.zeros((h, w, 3), np.uint8); pic[:] = (28, 30, 58)
    pic[show] = src[show]
    if k < 1:                                                          # 扫描头：亮青一行
        head = even[n_e - 1] if n_o == 0 and n_e else (odd[n_o - 1] if n_o else 0)
        pic[head] = CYAN_HI
    big = np.repeat(np.repeat(pic, PORT_PX, 0), PORT_PX, 1)
    px, py = x0 + 4, y0 + 22
    img[py:py + big.shape[0], px:px + big.shape[1]] = big
    if k < 1:                                                          # 底栏读取百分比
        img = text(img, (x0 + 4, y1 + 4), 'LOAD %3d%%' % int(k * 100), CYAN)
    else:
        img = text(img, (x0 + 4, y1 + 4), 'Hi! I\'m Claude-chan', YEL)
    return img


def boot_screen(t):
    """全屏开机画面（640×360），只由 t 决定。"""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = INK
    x, y0, dy = 40, 36, 24
    rows = []
    if t >= 0.00: rows.append(('CLAUDE-98 BIOS  Ver 5.5', PAPER))
    if t >= 0.08:
        kb = int(np.clip((t - 0.08) / 0.16, 0, 1) * 10) * 64          # 内存数字往上滚：64 → 640KB
        rows.append(('MEMORY CHECK .... %dKB' % kb + (' OK' if kb >= 640 else ''), PAPER))
    if t >= 0.26: rows.append(('FDD 1: CLAUDE_POP.EXE', PAPER))
    if t >= 0.34: rows.append(('GFX  : CLAUDE_CHAN.MAG', PAPER))
    if t >= 0.48: rows.append(('LOADING', YEL))
    for k, (s, c) in enumerate(rows):
        if s: img = text(img, (x, y0 + k * dy), s, c)
    cur_y = y0 + len(rows) * dy
    if t >= 0.48:                                                     # 进度条（10-19 用户：跑步的 Q 版 Claude 不要了，元素别堆砌）
        prog = np.clip((t - 0.50) / 0.30, 0, 1)
        prog = np.floor(prog * 20) / 20                               # 一格一格跳
        by = y0 + 7 * dy
        box(img, x, by, x + 322, by + 16, INK, ink=PAPER, t=2)
        rect(img, x + 4, by + 4, x + 4 + int(314 * prog), by + 12, ORANGE)
        cur_y = by + 28
    if t >= 0.78:
        img = text(img, (x, cur_y), 'CLAUDE.SYS loaded', PAPER)
        cur_y += dy
    img = _portrait_window(img, t)
    if int(t * 8) % 2 == 0:                                           # 闪烁光标
        rect(img, x, cur_y + 2, x + 8, cur_y + 16, PAPER)
    return img


@functools.lru_cache(maxsize=None)
def _key():
    return L.load_rgb(KEY_PATH)


@functools.lru_cache(maxsize=None)
def _key_hi():
    return L.load_rgb(KEY_PATH, *KEY_SIZE)


def scene_boot(t):
    if t < BOOT_T0:
        return boot_screen(t)
    if t >= BOOT_T1 - 1e-6:
        return _key().copy()                                          # 终点：关键帧原样
    e = ease_io((t - BOOT_T0) / (BOOT_T1 - BOOT_T0))
    # 屏幕四角的输出位置：从「占满画面」指数收缩到关键帧里的位置（指数插值 = 匀速的后拉感）
    full = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    S0 = 10.0
    g = (S0 ** (1 - e) - 1) / (S0 - 1)
    P = SCREEN + (full - SCREEN) * g
    G = cv2.getPerspectiveTransform(SCREEN, P)                       # 关键帧(640 空间) → 输出
    kw, kh = KEY_SIZE
    Hbg = G @ np.diag([W / kw, H / kh, 1.0])                         # 用原分辨率关键帧，放大时不糊
    bg = cv2.warpPerspective(_key_hi(), Hbg, (W, H), flags=cv2.INTER_LINEAR, borderValue=NAVY)
    boot = boot_screen(t)
    Hs = cv2.getPerspectiveTransform(full, P)
    scr = cv2.warpPerspective(boot, Hs, (W, H), flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(np.ones((H, W), np.uint8), Hs, (W, H), flags=cv2.INTER_NEAREST) > 0
    # 最后一段：屏幕内容以 Bayer 网点溶解回关键帧原本的星星眼（1.20 那帧必须完全是关键帧）
    k = (e - 0.72) / 0.28
    if k > 0:
        m &= ~L.bayer_mask(k)
    out = bg
    out[m] = scr[m]
    return out


# =====================================================================
# 2. circuit：电路板俯视
# =====================================================================
CX, CY = 320, 175                     # 中心芯片中心
CHIP_BOX = (270, 155, 370, 195)
T_NERV, T_THAT, T_SURP, T_GATHER, C_END = 6.92, 8.08, 8.38, 8.75, 9.06
TRACE_V = 300.0                       # 点亮前端爬行速度 px/s


def _polyline_pts(pts):
    """正交 / 45° 折线 → 按弧长顺序的 1px 采样点。"""
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = int(max(abs(x1 - x0), abs(y1 - y0)))
        for i in range(n):
            out.append((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n))
    out.append(pts[-1])
    return np.round(np.array(out)).astype(int)


def _side(sign, yk, a, ys):
    """左右侧走线：板边 → 水平 → 45° → 水平 → 芯片侧引脚。"""
    xp = CX + sign * 58
    p1 = (xp + sign * a, yk)
    p2 = (p1[0] + sign * abs(ys - yk), ys)
    return [(CX + sign * 306, ys), p2, p1, (xp, yk)]


def _bottom(xk, a, xs):
    """底部走线：板底 → 竖直 → 45° → 竖直 → 芯片底引脚。"""
    p1 = (xk, 203 + a)
    p2 = (xs, p1[1] + abs(xs - xk))
    return [(xs, 282), p2, p1, (xk, 203)]


@functools.lru_cache(maxsize=None)
def _traces():
    """16 条走线：(ys, xs, arc, L, t0)。线宽 3px，arc 是每个像素的弧长位置（从板边算起）。"""
    paths = []
    for sign in (-1, 1):
        for yk, a, ys in [(162, 10, 40), (170, 22, 105), (186, 10, 268), (178, 20, 235)]:
            paths.append(_side(sign, yk, a, ys))
    xs = [230, 256, 281, 307, 333, 359, 384, 410]
    aa = [6, 12, 18, 24, 24, 18, 12, 6]
    for i in range(8):
        paths.append(_bottom(278 + 12 * i, aa[i], xs[i]))
    # 点亮顺序：卡 88 BPM 的半拍网格（5.64 开场立刻一组，之后 5.827 / 6.168 / 6.509 / 6.85）
    groups = [5.64, 5.827, 6.168, 6.509, 6.85]
    order = [0, 4, 8, 15, 1, 5, 11, 12, 2, 6, 9, 14, 3, 7, 10, 13]
    gid = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]              # 每组 3–4 条
    t0 = {idx: groups[gid[k]] for k, idx in enumerate(order)}
    res = []
    off = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    for i, p in enumerate(paths):
        pts = _polyline_pts(p)
        n = len(pts)
        thick = (pts[:, None, :] + off[None]).reshape(-1, 2)
        arc = np.repeat(np.arange(n), len(off))
        ok = (thick[:, 0] >= 0) & (thick[:, 0] < W) & (thick[:, 1] >= 0) & (thick[:, 1] < H)
        res.append((thick[ok, 1], thick[ok, 0], arc[ok], float(n), t0[i], p))
    return res


def _small_chip(img, x, y, w, h, label, pins='tb'):
    """装饰小芯片：黑体 + 银引脚 + 丝印字。"""
    step = 8
    if 't' in pins or 'b' in pins:
        for px in range(x + 5, x + w - 4, step):
            if 't' in pins: rect(img, px, y - 5, px + 4, y + 1, PIN)
            if 'b' in pins: rect(img, px, y + h - 1, px + 4, y + h + 5, PIN)
    if 'l' in pins or 'r' in pins:
        for py in range(y + 5, y + h - 4, step):
            if 'l' in pins: rect(img, x - 5, py, x + 1, py + 4, PIN)
            if 'r' in pins: rect(img, x + w - 1, py, x + w + 5, py + 4, PIN)
    box(img, x, y, x + w, y + h, CHIP)
    rect(img, x + 2, y + 2, x + w - 2, y + 4, CHIP_HI)
    img = text(img, (x + (w - text_w(label)) // 2, y + (h - 14) // 2), label, SILK)
    return img


def _pad(img, x, y, r=3):
    rect(img, x - r - 1, y - r - 1, x + r + 2, y + r + 2, INK)
    rect(img, x - r, y - r, x + r + 1, y + r + 1, PAD)
    rect(img, x - 1, y - 1, x + 2, y + 2, PCB_DARK)


@functools.lru_cache(maxsize=None)
def _board():
    """静态电路板（不含中心芯片的发光、不含点亮走线、不含 Claude）。"""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = NAVY
    box(img, 10, 20, 630, 283, PCB, t=3)
    # 装饰暗走线（不点亮）
    deco = [[(40, 110), (40, 128), (100, 128)], [(120, 200), (150, 200), (150, 150), (160, 140)],
            [(600, 110), (600, 128), (540, 128)], [(520, 200), (490, 200), (490, 150), (480, 140)],
            [(60, 222), (140, 222)], [(580, 222), (500, 222)], [(190, 60), (220, 60), (230, 70), (230, 95)],
            [(450, 60), (420, 60), (410, 70), (410, 95)]]
    for d in deco:
        cv2.polylines(img, [np.int32(d)], False, PCB_DARK, 3, cv2.LINE_8)
    # 暗铜色走线本体
    for ys, xs, arc, n, t0, p in _traces():
        img[ys, xs] = COPPER
    # 装饰芯片 / 焊盘 / 丝印
    img = _small_chip(img, 34, 56, 64, 30, 'RAM')
    img = _small_chip(img, 542, 56, 64, 30, 'ROM')
    img = _small_chip(img, 36, 140, 72, 44, 'FDC', pins='lr')
    img = _small_chip(img, 532, 140, 72, 44, 'VRAM', pins='lr')
    for x, y in [(190, 60), (230, 95), (450, 60), (410, 95), (40, 110), (600, 110), (100, 128), (540, 128),
                 (60, 222), (140, 222), (580, 222), (500, 222), (160, 140), (480, 140)]:
        _pad(img, x, y)
    for x, y in [(26, 34), (614, 34), (26, 268), (614, 268)]:        # 安装孔
        rect(img, x - 7, y - 7, x + 8, y + 8, INK)
        rect(img, x - 6, y - 6, x + 7, y + 7, PAD)
        rect(img, x - 3, y - 3, x + 4, y + 4, NAVY)
    img = text(img, (440, 22), 'REV.C  1998', SILK)
    # 走线起点处的焊盘（板边连接器）
    for ys, xs, arc, n, t0, p in _traces():
        x, y = p[0]
        rect(img, x - 3, y - 3, x + 4, y + 4, PAD)
    return img


def _center_chip(img, glow, ring=CYAN):
    x0, y0, x1, y1 = CHIP_BOX
    body = CYAN_HI if glow == 2 else CHIP
    if glow:
        rect(img, x0 - 6, y0 - 6, x1 + 6, y1 + 6, WHITE if glow == 2 else ring)
    for py in (162, 170, 178, 186):                                  # 侧引脚
        rect(img, x0 - 8, py - 2, x0 + 1, py + 2, PIN)
        rect(img, x1 - 1, py - 2, x1 + 8, py + 2, PIN)
    for i in range(8):                                                # 底 / 顶引脚
        px = 278 + 12 * i
        rect(img, px - 2, y1 - 1, px + 2, y1 + 8, PIN)
        rect(img, px - 2, y0 - 8, px + 2, y0 + 1, PIN)
    box(img, x0, y0, x1, y1, body, t=2)
    if not glow:
        rect(img, x0 + 2, y0 + 2, x1 - 2, y0 + 5, CHIP_HI)
    return text(img, (CX - text_w('CLAUDE') // 2, CY - 7), 'CLAUDE', TITLE if glow == 2 else SILK)


@functools.lru_cache(maxsize=None)
def _sprites(height=132):
    """Q 版三姿势：按 alpha 列投影切开，裁紧，缩放到统一高度，alpha 二值。"""
    rgba = L.load_rgba('素材/精灵/q版Claude_三姿势.png')
    a = rgba[..., 3] > 127
    col = a.any(0)
    segs, x = [], 0
    while x < len(col):
        if col[x]:
            s = x
            while x < len(col) and col[x]: x += 1
            segs.append((s, x))
        x += 1
    segs = sorted(sorted(segs, key=lambda s: s[1] - s[0])[-3:])      # 最宽的三段（去掉抖动符号之类的小碎块）
    out = []
    for s, e in segs:
        sub = rgba[:, s:e]
        rows = np.nonzero(sub[..., 3] > 127)[0]
        sub = sub[rows.min():rows.max() + 1]
        k = height / sub.shape[0]
        sm = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        sm[..., 3] = np.where(sm[..., 3] > 127, 255, 0)
        out.append(sm)
    return out


@functools.lru_cache(maxsize=None)
def _readers(height=76):
    """10-19 用户：坠落镜头换成「Claude 在读书」的精灵表（读书五姿势：边走边读 / 放下书吓呆 / 掉下去书飞走 / 摔坐书扣头上 / 坐着接着读）。
    按连通块切：主体 + 飞走的书、书页、汗珠、晕圈按 x 质心分给各姿势；五张按同一个比例缩放（坐姿天然矮），alpha 二值。"""
    import codeshots_c as C
    rgba = L.load_rgba('素材/精灵/q版Claude_读书五姿势.png')
    n, lab, st, cen = cv2.connectedComponentsWithStats((rgba[..., 3] > 127).astype(np.uint8))
    cuts = [410, 790, 1300, 1700]                                    # 按原图量的姿势分界（x 质心）
    groups = [[] for _ in range(5)]
    for i in range(1, n):
        if st[i, 4] < 150: continue
        groups[int(np.searchsorted(cuts, cen[i][0]))].append(i)
    k = None
    out = []
    for g in groups:
        m = np.isin(lab, g)
        ys, xs = np.nonzero(m)
        sub = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        sub[..., 3] = np.where(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], sub[..., 3], 0)
        if k is None: k = height / sub.shape[0]                      # 以第一张（站姿）定比例
        out.append(C._rgba_resize(sub, k))
    return out


def _chibi_offset(t):
    """Claude 的姿势和位移（姿势 12fps，抖动 24fps）。"""
    tt = L.twos(t)
    if t < T_NERV:
        return 0, 0, (-2 if L.pulse(tt) > 0.5 else 0)                  # 发呆：拍点上轻轻一颠
    if t < T_THAT:
        f = int(np.floor(t * 24))
        amp = 1 + 3 * L.pulse(t)                                      # 冒汗发抖：拍点抖得最凶
        return 1, int(round((L.hash01(f, 1) - .5) * 2 * amp)), int(round((L.hash01(f, 2) - .5) * amp))
    u = tt - T_SURP
    dy = -int(round(14 * np.sin(np.pi * u / 0.30))) if 0 <= u < 0.30 else 0   # surprise 小弹一下
    return 2, 0, dy


def scene_circuit(t):
    img = _board().copy()
    red = t >= T_NERV
    base, hi = (RED, RED_HI) if red else (CYAN, CYAN_HI)
    beat, ph = L.beat_phase(t)
    on_beat = L.pulse(t) > 0.45                                        # 拍点上整线切亮色（平涂两档）
    gather = L.smooth((t - T_GATHER) / 0.20)                           # 收拢：光从板边一侧熄灭，全压向芯片
    for ys, xs, arc, n, t0, p in _traces():
        lit = np.clip((t - t0) * TRACE_V, 0, n)
        if lit <= 0: continue
        lo = gather * n
        m = (arc < lit) & (arc >= lo)
        if not m.any(): continue
        img[ys[m], xs[m]] = hi if on_beat else base
        # 每拍从板边往芯片跑一颗白色光脉冲
        s = ph * (n + 40) - 20
        mp = m & (np.abs(arc - s) < 9)
        img[ys[mp], xs[mp]] = WHITE
        if lit < n:                                                   # 爬行前端的亮头
            mh = m & (arc > lit - 7)
            img[ys[mh], xs[mh]] = WHITE
    glow = 0
    if t >= T_GATHER + 0.12: glow = 2
    elif t >= T_GATHER: glow = 1
    if glow == 2:                                                      # 芯片发亮：四向硬边光芒（画在芯片底下）
        k = np.clip((t - T_GATHER - 0.12) / 0.15, 0, 1)
        ln = int(40 + 260 * k)
        rect(img, CX - ln, CY - 2, CX + ln, CY + 3, WHITE)
        rect(img, CX - 2, CY - int(ln * .5), CX + 3, CY + int(ln * .5), WHITE)
    img = _center_chip(img, glow, hi)
    # Claude 站在芯片上
    pose, dx, dy = _chibi_offset(t)
    spr = _sprites()[pose]
    sh, sw = spr.shape[:2]
    img = L.paste_rgba(img, spr, CX - sw // 2 + dx, CY - 12 - sh + dy)
    # 结尾推进到芯片中心（ease-in），芯片中心同时移到画面有效区中心
    zk = np.clip((t - 8.80) / (C_END - 8.80), 0, 1)
    if zk > 0:
        z = 1 + 4.0 * zk ** 2
        oy = CY + ((TOP + BOT) / 2 - CY) * zk
        M = np.float32([[z, 0, CX - z * CX], [0, z, oy - z * CY]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=NAVY)
    return img


# =====================================================================
# 3. loss：TRAINING.EXE，drop 时曲线砸穿窗口
# =====================================================================
WX0, WY0, WX1, WY1 = 80, 34, 560, 268          # 窗口
PX0, PY0, PX1, PY1 = 122, 84, 544, 248         # 图区
L_POP, L_START, L_DROP, L_LOSS = 9.06, 9.30, 10.54, 11.86
PLATEAU = 108
CURVE_V = (PX0 + 0.72 * (PX1 - PX0) - PX0) / (L_DROP - L_START)       # drop 时曲线头走到图区 72%
FALL_G, FALL_VX = 2400.0, 70.0
X_DROP = int(PX0 + CURVE_V * (L_DROP - L_START))
T_HIT = L_DROP + np.sqrt((WY1 - PLATEAU) / FALL_G)                    # 曲线头撞到窗口底边的时刻
X_HIT = int(X_DROP + FALL_VX * (T_HIT - L_DROP))


@functools.lru_cache(maxsize=None)
def _desk():
    """PC-98 桌面：深藏青 + 点阵格。"""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = NAVY
    img[::16, ::16] = NAVY_DOT
    img[1::16, ::16] = NAVY_DOT
    img[::16, 1::16] = NAVY_DOT
    img[1::16, 1::16] = NAVY_DOT
    return img


def _window(img, x0, y0, x1, y1, title, plot, plot_bg, grid, shadow=True):
    """画 PC-98 风格窗口：硬阴影 + 墨线框 + 藏青标题栏 + 图区。原地/返回新图。"""
    if shadow:
        rect(img, x0 + 6, y0 + 6, x1 + 6, y1 + 6, INK)
    box(img, x0, y0, x1, y1, WINBG)
    rect(img, x0 + 2, y0 + 2, x1 - 2, y0 + 24, TITLE)
    rect(img, x0 + 2, y0 + 24, x1 - 2, y0 + 26, INK)
    img = text(img, (x0 + 10, y0 + 6), title, PAPER)
    for k in range(2):                                                # 标题栏右侧两个小按钮
        bx = x1 - 22 - k * 20
        box(img, bx, y0 + 6, bx + 14, y0 + 20, WINBG)
    px0, py0, px1, py1 = plot
    box(img, px0 - 2, py0 - 2, px1 + 2, py1 + 2, plot_bg)
    img[py0 + 8:py1:16, px0 + 8:px1:16] = grid                        # 点状网格
    return img


def _curve_pts(t, t_start, v, yfun, x_left):
    """从左往右实时画出的曲线：x 每像素对应一个时间，y 由 yfun(x, tx) 给出。"""
    head = int(x_left + v * (t - t_start))
    if head <= x_left: return None
    xs = np.arange(x_left, head + 1)
    tx = t_start + (xs - x_left) / v
    ys = np.array([yfun(int(x), float(s)) for x, s in zip(xs, tx)])
    return np.stack([xs, ys], 1).astype(np.int32)


def _loss_y(x, tx):
    """平台期 loss：开头从顶上掉到平台，之后一直在平台抖；88 BPM 拍点上抖得最明显。"""
    d = x - PX0
    base = PLATEAU + (PY0 + 4 - PLATEAU) * np.exp(-d / 10.0)
    amp = 2 + 8 * L.pulse(tx)
    return base + (L.hash01(x, 7) - .5) * 2 * amp


def _loss_value(t):
    if t < L_DROP:
        return 2.31 + (L.hash01(np.floor(t * 12), 3) - .5) * 0.04
    u = t - L_DROP
    return 2.31 - 9 * u - 30 * u ** 3                                  # 狂跌，跌穿 0 一路到负数


def _loss_window(t):
    """整窗（含平台期曲线、数字），画在 640×360 桌面上。"""
    img = _desk().copy()
    img = _window(img, WX0, WY0, WX1, WY1, 'TRAINING.EXE', (PX0, PY0, PX1, PY1), WINBG, PLOT_GRID)
    img = text(img, (WX0 + 8, PY0 - 2), 'L', TITLE)                  # 轴标签
    img = text(img, (PX1 - 34, PY1 + 2), 'step', TITLE)
    step = 4096 + int((t - L_POP) * 24) * 37 + (int((t - L_DROP) * 24) * 911 if t > L_DROP else 0)
    img = text(img, (WX0 + 12, WY0 + 32), 'step %d' % step, INK)
    v = _loss_value(t)
    s = 'loss = %.2f' % v
    img = text(img, (WX1 - 14 - text_w(s), WY0 + 32), s, LOSS_RED if t >= L_DROP else INK)
    return img


# ---- 10-19 用户：SUDDEN DROP 太长、无聊 → 改成「一路坠落」 ----
# Claude（Q 版，读书精灵表）边走边读站在曲线头上；A（9.98）发现不对开始发抖；DROP（10.54）跟着曲线一头栽下去，
# 镜头跟拍往下，IN / YOUR / TRAINING 三个词各砸穿一层窗口（越往下 loss 越离谱），
# LOSS（11.86）摔在最底下的终端「loss is NaN」上，压扁弹回，最后摊手。
FLOORS = [(10.96, 560, 'VALID.EXE', 'val_loss = 0.07'),
          (11.32, 900, 'EVAL.EXE', 'eval = -4.2'),
          (11.52, 1180, 'BENCH.EXE', 'score = -9E9')]
GROUND_T, GROUND_Y = 11.86, 1500
CAM_MAX = GROUND_Y - 200                                     # 落地时地面在屏幕 y=200（再低终端的字会被对白框挡住）
PANEL_X0, PANEL_X1, PANEL_H = 196, 616, 64


def _head(t):
    """drop 之后曲线头的世界坐标：每段像重新起跳一样从慢到快（砸穿一层顿一下）。"""
    knots = [(L_DROP, PLATEAU)] + [(a, y) for a, y, _, _ in FLOORS] + [(GROUND_T, GROUND_Y)]
    t = min(t, GROUND_T)
    for (ta, ya), (tb, yb) in zip(knots, knots[1:]):
        if t <= tb:
            u = (t - ta) / (tb - ta)
            return X_DROP + 64 * (t - L_DROP), ya + (yb - ya) * u ** 1.7
    return X_DROP + 64 * (GROUND_T - L_DROP), GROUND_Y


def _cam(t):
    if t < L_DROP: return 0
    return int(np.clip(_head(t)[1] - 165, 0, CAM_MAX))


def _desk_scrolled(cam):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = NAVY
    off = cam % 16
    for dy in (0, 1):
        for dx in (0, 1):
            img[(dy - off) % 16::16, dx::16] = NAVY_DOT
    return img


def _hole(img, cx, top, bot, desk):
    hw = 30
    for x in range(int(cx) - hw, int(cx) + hw + 1):
        if not 0 <= x < W: continue
        d = abs(x - cx) / hw
        a = int(top + 26 * d + L.hash01(x // 4, 5) * 8)              # 上沿 V 形锯齿，往下整条砸穿
        a = max(a, 0); b = min(bot, H)
        if a < b:
            img[a:b, x] = desk[a:b, x]
            img[max(a - 2, 0):a, x] = INK


def _panel(img, y, title, val, broken_at, cam, desk, t):
    sy = y - cam
    if sy > H or sy + PANEL_H + 8 < 0: return img
    rect(img, PANEL_X0 + 6, sy + 6, PANEL_X1 + 6, sy + PANEL_H + 6, INK)
    box(img, PANEL_X0, sy, PANEL_X1, sy + PANEL_H, WINBG)
    rect(img, PANEL_X0 + 2, sy + 2, PANEL_X1 - 2, sy + 22, TITLE)
    img = text(img, (PANEL_X0 + 8, sy + 4), title, PAPER)
    img = text(img, (PANEL_X0 + 14, sy + 32), val, LOSS_RED)
    if t >= broken_at:
        _hole(img, _head(broken_at)[0], sy, sy + PANEL_H + 8, desk)
    return img


def _shards(img, ox, oy, t0, t, cam, n=14, seed=0):
    u = t - t0
    if u < 0 or u > 1.2: return
    for i in range(n):
        x = ox + (L.hash01(i, 21 + seed) - .5) * 40 + (L.hash01(i, 23 + seed) - .5) * 260 * u
        y = oy + (-60 - L.hash01(i, 24 + seed) * 160) * u + 0.5 * 1400 * u * u - cam
        if y > H + 10 or y < -10: continue
        sz = 4 + int(L.hash01(i, 25 + seed) * 6)
        c = [WINBG, INK, TITLE, WINBG][i % 4]
        rect(img, x - sz // 2 - 1, y - sz // 2 - 1, x + sz - sz // 2 + 1, y + sz - sz // 2 + 1, INK)
        rect(img, x - sz // 2, y - sz // 2, x + sz - sz // 2, y + sz - sz // 2, c)


def _rot_sprite(spr, deg):
    h, w = spr.shape[:2]
    d = int(np.hypot(h, w)) + 2
    pad = np.zeros((d, d, 4), np.uint8)
    pad[(d - h) // 2:(d - h) // 2 + h, (d - w) // 2:(d - w) // 2 + w] = spr
    M = cv2.getRotationMatrix2D((d / 2, d / 2), deg, 1.0)
    return cv2.warpAffine(pad, M, (d, d), flags=cv2.INTER_NEAREST)


def _ground(img, cam, t):
    sy = GROUND_Y - cam
    if sy > H: return img
    rect(img, 20 + 6, sy + 6, 620 + 6, sy + 200, INK)
    box(img, 20, sy, 620, sy + 200, TERM_BG, ink=GREEN)
    lines = ['>>> train(model)', 'RuntimeError: loss is NaN', 'gradients exploded :(']
    for k, s in enumerate(lines):
        img = text(img, (36, sy + 8 + k * 18), s, LOSS_RED if k == 1 else GREEN)
    if t >= GROUND_T:                                                # 落地砸出的裂纹
        cx = int(_head(GROUND_T)[0])
        for k in range(5):
            ang = -np.pi * (0.12 + 0.19 * k)
            ln = 26 + 16 * L.hash01(k, 41)
            p0 = (cx, sy); p1 = (int(cx + np.cos(ang) * ln), int(sy - np.sin(ang) * ln * .5))
            cv2.line(img, p0, (p1[0], max(p1[1], sy)), GREEN, 1, cv2.LINE_8)
    return img


def scene_loss(t):
    if t < L_POP + 0.20:                                              # 弹出：从画面中心一点 ease_out_back 放大
        win = _loss_window(L_POP)
        k = L.ease_out_back((t - L_POP) / 0.20)
        img = _desk().copy()
        if k <= 0.01: return img
        crop = win[WY0:WY1 + 6, WX0:WX1 + 6]
        ch, cw = crop.shape[:2]
        nw, nh = max(1, int(cw * k)), max(1, int(ch * k))
        sm = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        cx, cy = (WX0 + WX1 + 6) // 2, (WY0 + WY1 + 6) // 2
        x, y = cx - nw // 2, cy - nh // 2
        sx0, sy0 = max(0, -x), max(0, -y)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + nw, W), min(y + nh, H)
        img[y0:y1, x0:x1] = sm[sy0:sy0 + y1 - y0, sx0:sx0 + x1 - x0]
        return img

    cam = _cam(t)
    desk = _desk_scrolled(cam)
    # 第一层：TRAINING.EXE（世界坐标 0–360），整张画好再按镜头往上挪
    w1 = _loss_window(t)
    tc = min(t, L_DROP)
    pts = _curve_pts(tc, L_START, CURVE_V, _loss_y, PX0)
    if pts is not None:
        cv2.polylines(w1, [pts], False, LOSS_RED, 2, cv2.LINE_8)
    if t >= T_HIT:
        _break_hole(w1, t)
    img = desk.copy()
    if cam < H:
        img[:H - cam] = w1[cam:]
        img[:H - cam][(w1[cam:] == NAVY).all(2)] = desk[:H - cam][(w1[cam:] == NAVY).all(2)]
    for (bt, y, title, val) in FLOORS:
        img = _panel(img, y, title, val, bt, cam, desk, t)
    img = _ground(img, cam, t)
    # 速度线：坠落时满屏往上飞的竖线
    if L_DROP + .08 < t < GROUND_T:
        for i in range(26):
            x = int(L.hash01(i, 51) * W)
            y = int((L.hash01(i, 52) * (H + 80) - (t - L_DROP) * 2200 * (0.6 + .4 * L.hash01(i, 53))) % (H + 80)) - 40
            rect(img, x, y, x + 2, y + 30 + int(30 * L.hash01(i, 54)), PAPER)
    # 下落的曲线（世界坐标采样再减镜头）
    if t >= L_DROP and pts is not None:
        ss = np.linspace(L_DROP, min(t, GROUND_T), 80)
        fall = np.array([_head(s) for s in ss])
        fall[:, 1] -= cam
        p0 = pts[-1:].astype(float); p0[:, 1] -= cam
        fall = np.concatenate([p0, fall])
        cv2.polylines(img, [fall.astype(np.int32)], False, LOSS_RED, 3, cv2.LINE_8)
    # 碎片：每砸穿一层出一把
    if t >= T_HIT: _shards(img, X_HIT, WY1, T_HIT, t, cam, 18, 0)
    for k, (bt, y, _, _) in enumerate(FLOORS):
        _shards(img, _head(bt)[0], y, bt, t, cam, 14, 7 * (k + 1))
    # Q 版 Claude：读书精灵表（10-19）
    tt = L.twos(t)
    R = _readers()
    if t < L_DROP:
        hx, hy = (pts[-1] if pts is not None else (PX0, PLATEAU))
        nerv = t >= 9.98
        spr = R[1 if nerv else 0]
        jit = int((L.hash01(np.floor(t * 24), 61) - .5) * 4) if nerv else -int(abs(2 * np.sin(tt * 9)))   # 走路一颠一颠 / 吓得发抖
        sh, sw = spr.shape[:2]
        img = L.paste_rgba(img, spr, hx - sw // 2 + (jit if nerv else 0), hy - sh + 2 + (0 if nerv else jit))
    elif t < GROUND_T:
        hx, hy = _head(t)
        spr = _rot_sprite(R[2], 25 * np.sin((tt - L_DROP) * 9))       # 掉下去时手脚乱挥、身子来回晃（不整圈转，脸一直看得到）
        d = spr.shape[0]
        img = L.paste_rgba(img, spr, hx - d // 2 + 14, hy - cam - d // 2 - 22)
    else:
        hx, _ = _head(GROUND_T)
        gy = GROUND_Y - cam
        u = t - GROUND_T
        if u < 0.16:                                                  # 压扁 → 回弹
            spr = R[3]
            sq = 1 - 0.38 * np.sin(np.pi * min(u / .16, 1))
            spr = cv2.resize(spr, (int(spr.shape[1] * (2 - sq)), max(4, int(spr.shape[0] * sq))), interpolation=cv2.INTER_NEAREST)
        else:
            spr = R[4 if u > 0.42 else 3]                             # 晕一会儿（书扣在头上）→ 若无其事地接着读
        sh, sw = spr.shape[:2]
        img = L.paste_rgba(img, spr, hx - sw // 2 + 14, gy - sh + 2)
        if u < 0.42 and int(np.floor(u * 12)) % 2 == 0:              # !!
            img = text_outline(img, (int(hx) - 70, gy - 110), '!!', RED, scale=4, o=2)
        if 0.16 <= u < 0.42:
            for k in range(3):                                        # 头顶转圈的小星星
                a = u * 9 + k * 2.1
                sx, sy = int(hx + 14 + np.cos(a) * 26), int(gy - 70 + np.sin(a) * 7)
                rect(img, sx - 2, sy - 2, sx + 3, sy + 3, YEL)
    # 镜头离开第一层后，右上角挂一个 loss 读数（一路往负数狂跌，落地变 NaN）
    if cam > 120:
        s = 'loss = NaN' if t >= GROUND_T else 'loss = %.1f' % _loss_value(t)
        box(img, 452, 24, 624, 48, TERM_BG, ink=LOSS_RED)
        img = text(img, (460, 28), s, LOSS_RED)
    # 震屏：砸穿每一层、落地
    for bt, a0, dur in [(T_HIT, 6, .25)] + [(f[0], 5, .15) for f in FLOORS] + [(GROUND_T, 9, .35)]:
        if bt <= t < bt + dur:
            f = int(np.floor(t * 24))
            a = a0 * (1 - (t - bt) / dur)
            img = np.roll(img, (int(round((L.hash01(f, 12) - .5) * 2 * a)), int(round((L.hash01(f, 11) - .5) * 2 * a))), (0, 1))
    return img


def _break_hole(img, t):
    """窗口底边被砸出的锯齿缺口：缺口内露出桌面（和阴影一起挖掉），边沿画墨线。"""
    desk = _desk()
    hw = 34
    for x in range(X_HIT - hw, X_HIT + hw + 1):
        d = abs(x - X_HIT) / hw
        top = int(WY1 - 4 - 34 * (1 - d) - L.hash01(x // 4, 5) * 12)   # V 形 + 锯齿
        img[top:WY1 + 8, x] = desk[top:WY1 + 8, x]
        img[top - 2:top, x] = INK


# =====================================================================
# 10. TRAINING RUN #2 窗口的几何和曲线（codeshots_c.stable、codeshots_s9 复用）
# =====================================================================
T2_0, T2_1 = 36.00, 38.12
FX0, FY0, FX1, FY1 = 6, 20, 632, 276           # 满屏时的窗口（阴影到 282，留出 HUD）
GX0, GY0, GX1, GY1 = 50, 72, 616, 258          # 图区


def _t2_y(x, tx):
    """稳定下降：指数衰减 + 极小抖动。"""
    d = (x - GX0) / (GX1 - GX0)
    return GY0 + 14 + (GY1 - GY0 - 40) * (1 - np.exp(-3.2 * d)) / (1 - np.exp(-3.2)) + (L.hash01(x, 31) - .5) * 2


def _t2_window(t):
    img = np.zeros((H, W, 3), np.uint8)
    img = _window(img, FX0, FY0, FX1, FY1, 'TRAINING RUN #2', (GX0, GY0, GX1, GY1), TERM_BG, GREEN_DIM)
    u = np.clip((t - T2_0) / (T2_1 - T2_0), 0, 1)
    v = (GX1 - 8 - GX0) / (T2_1 - T2_0) * 1.02
    pts = _curve_pts(t, T2_0, v, _t2_y, GX0)
    if pts is not None:
        pts = pts[pts[:, 0] <= GX1 - 4]
        cv2.polylines(img, [pts], False, GREEN, 2, cv2.LINE_8)
        hx, hy = pts[-1]
        if L.pulse(t) > .45:                                          # 132 BPM 拍点：曲线头亮一下
            rect(img, hx - 3, hy - 3, hx + 4, hy + 4, WHITE)
    img = text(img, (FX0 + 8, GY0 - 2), 'L', TITLE)
    step = 1 + int(u * 1200) * 8
    img = text(img, (FX0 + 12, FY0 + 32), 'step %d' % step, INK)
    loss = 2.31 - 1.6 * (1 - np.exp(-3.2 * u)) / (1 - np.exp(-3.2))
    s = 'loss = %.2f' % loss
    img = text(img, (FX1 - 16 - text_w(s), FY0 + 32), s, INK)
    # 标题栏上的运行指示灯，按拍闪
    if L.pulse(t) > .45:
        rect(img, FX1 - 70, FY0 + 9, FX1 - 60, FY0 + 19, GREEN)
    return img


SCENES = [
    ('boot', 0.00, 1.20, scene_boot),
    ('circuit', 5.64, 9.06, scene_circuit),
    ('loss', 9.06, 12.52, scene_loss),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
