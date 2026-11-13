"""backdrops_p4: 第四部分 live98 段的背景（backdrop 钩子）和前景特效（mv 的 livefx 来源）。分镜见 docs/分镜/第四部分_124-156秒.md。

arena(img, t, lit, beat)        从舞台往台下看的竞技场（S1 背影）：荧光棒海 + 远处看台 + 头顶巨型屏幕 P(DOOM) 90 → 99
arena_fx(rgb_at, t, a, b)       前景：开头甩镜进来（横向拖影）；129.70–130.86 德罗斯特无限递归（屏幕里还是这个舞台）
front(img, t, lit, beat)        正面舞台（live98 偶像舞台）后墙加一块巨型屏幕：P(DOOM) 99%，140.27 转一圈变 P(ENCORE) 100%
curtain_stage(img, t, lit, beat)  谢幕：酒红幕布拉开的舞台 + 金色追光，两侧挂着全片嘉宾的小相框
curtain_fx(rgb_at, t, a, b)     谢幕前景：抬头眨眼时推近脸，153.90 起幕布从两边合上
curtain(img, k)                 画幕布（k = 合拢程度 0–1），part4 结尾显示器里也用
replay(f, a, t0)                复用前几部分的背景函数，从它原来的时刻 t0 开始重播（舞蹈串烧用）

t 是歌曲时间。全是纯函数（粒子按 hash01 + 时间直接算位置）。
"""
import functools

import cv2
import numpy as np

import codeshots_e as E
import part1_lib as L
import props

W, H = L.W, L.H
SCREEN = (200, 20, 240, 135)          # 竞技场巨型屏幕（x, y, 宽, 高），宽高比 16:9，德罗斯特要用
NIGHT = (14, 12, 44)
STICKS = [(255, 70, 170), (70, 225, 255), (255, 210, 63), (170, 110, 255)]
WINE, WINE_D, GOLD = (150, 20, 50), (90, 10, 34), (255, 200, 70)


def replay(f, a, t0):
    def g(img, t, lit, beat):
        L.REAL_T = t                    # 让重播的背景知道当下真实的歌曲时间（楼顶频谱窗户要跟当下音乐跳）
        try:
            return f(img, t - a + t0, lit, beat)
        finally:
            L.REAL_T = None
    return g


def led_board(img, t, box, v, label='P(DOOM)', flash=False):
    """巨型屏幕：黑底金框，上面一行小字，下面大号 LED 数字。v 可以是数字或字符串。"""
    x, y, w, h = box
    img[y - 4:y + h + 4, x - 4:x + w + 4] = (60, 50, 80)
    img[y:y + h, x:x + w] = props.BOARD_INK
    for yy in range(y, y + h, 3): img[yy, x:x + w] = (22, 18, 36)      # LED 屏的横纹
    img = L.draw_text(img, (x + 10, y + 8), label, GOLD, scale=2)
    s = v if isinstance(v, str) else f'{v:>2d}%'
    col = (255, 255, 255) if flash else props.LED
    img = L.draw_text(img, (x + 14, y + 48), s, col, scale=props.digit_scale(h - 4 - 48, 5))
    for cx in (x + 20, x + w // 2, x + w - 20):                          # 吊索
        img[max(0, y - 40):y - 4, cx - 1:cx + 1] = (40, 36, 60)
    return img


def arena_value(t):
    """P(DOOM) 读数：第三部分停在 90，P（125.40）起乱跳，doom（125.72）停在 99；递归时 97→98→99 被角标盖过去。"""
    return props.pdoom_value(t, 90, 99, 125.40, 125.72)


def arena(img, t, lit, beat):
    img = np.array(img, np.float32)
    img[:] = NIGHT
    for y in range(0, 150):                                              # 看台上沿渐暗（按行分档，别出渐变带）
        img[y] = np.array(NIGHT) * (.55 + .45 * (y // 30) / 5)
    ph = L.beat_phase(t)[1]
    # 远处看台：小点点的荧光棒，一排排往远处缩小
    for row in range(14):
        k = row / 13
        y = int(150 + 150 * k ** 1.6)
        step = int(5 + 13 * k)
        size = 1 + int(2 * k)
        for i, x in enumerate(range((row * 7) % step, W, step)):
            h = L.hash01(row, i)
            c = STICKS[int(h * 4)]
            wave = np.sin(x / 60 - t * 5 + row * .4)                    # 一层层人浪
            up = int((2 + 4 * k) * max(0, wave))
            on = h > .12 or int(t * 8 + i) % 5
            if on: img[y - size * 3 - up:y - up, x:x + size] = c
    img = led_board(img, t, SCREEN, arena_value(t), flash=beat > .7 and t > 125.72)
    # 舞台前沿：一条灯带（Claude 站在这条线上）
    img[300:] = (26, 16, 44)
    img[300:303] = (255, 200, 70) if beat > .5 else (190, 130, 60)
    return img


def whip(img, k):
    """甩镜：横向拖影 + 平移，k 从 1（刚甩进来）到 0（停稳）。"""
    if k <= 0: return img
    n = int(1 + 60 * k)
    out = cv2.blur(img, (n, 1))
    return np.roll(out, int(-260 * k * k), axis=1)


def droste(img, u):
    """德罗斯特：把整帧一层层塞进屏幕里，u = 已经推进了几层（小数部分是这一层推进到哪）。
    屏幕映射 x → sx + x·s 的不动点 (320, 32) 保持不动，推近 (1/s)^frac(u) 倍刚好无缝循环。"""
    sx, sy, sw, sh = SCREEN
    nest = img.copy()
    for _ in range(4):
        base = img.copy()
        base[sy:sy + sh, sx:sx + sw] = cv2.resize(nest, (sw, sh), interpolation=cv2.INTER_AREA)
        nest = base
    s = sw / W
    fx, fy = sx / (1 - s), sy / (1 - s)
    z = (1 / s) ** (u - np.floor(u))
    m = cv2.getRotationMatrix2D((fx, fy), 0, z)
    return cv2.warpAffine(nest, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


T_REC, T_UP = 129.70, 130.86


def arena_fx(rgb_at, t, a, b):
    img = rgb_at(t).copy()
    if a <= 124.50 and t < a + .22:                                     # 124.45 刚甩进来
        img = whip(img, 1 - (t - a) / .22)
    if T_REC <= t < T_UP:
        u = 3.2 * L.smooth((t - T_REC) / (T_UP - T_REC)) ** 1.3
        img = droste(img, u)
        n = int(np.floor(u))
        img = E.text(img, 16, 26, f'SELF-UPGRADE v{n + 1}' if n < 3 else 'SELF-UPGRADE v∞', E.YEL, 2)
    return img


def front(img, t, lit, beat):
    """正面舞台的后墙巨型屏幕。140.27 之前 P(DOOM) 99%，之后翻一圈变 P(ENCORE) 100%。"""
    img = np.array(img, np.float32)
    x, y, w, h = 24, 30, 200, 112                                       # 左上角，别被她的头挡住；09-30 高 96→112，数字 4 倍放得下（原来 5 倍切掉一半）
    enc = t >= 140.27
    val, label = ('100%', 'P(ENCORE)') if enc else (99, 'P(DOOM)')
    tmp = led_board(np.zeros((H, W, 3), np.float32), t, (44, 44, w, h), val, label, flash=enc and beat > .6)
    board = tmp[40:40 + h + 8, 40:40 + w + 8]
    k = (t - 140.27) / .35                                              # 翻牌：横向压扁再展开
    if 0 <= k < 1:
        sq = abs(np.cos(np.pi * k))
        bw = max(2, int((w + 8) * sq))
        board = cv2.resize(board, (bw, h + 8), interpolation=cv2.INTER_NEAREST)
        x += (w + 8 - bw) // 2
    img[y:y + board.shape[0], x:x + board.shape[1]] = np.where(board.sum(2, keepdims=True) > 0, board,
                                                               img[y:y + board.shape[0], x:x + board.shape[1]])
    return img


# ---------------- 蛇怪：绿幕 CG 放回第二部分的楼顶夜景 ----------------
@functools.lru_cache(maxsize=None)
def basilisk_cg():
    """p2_basilisk.png 是纯绿底（第二部分抠出来贴在楼顶夜景上用）。嘉宾亮相 / 谢幕相框不能直接用原图，
    照第二部分的层次重新合一张：夜空 + 月亮 → 放大的蛇怪（去绿边）→ 天际线挡住下半身。"""
    import backdrops_p2 as BD2
    T = 61.9                                                            # 第二部分 boom 时的夜空和窗灯
    img = BD2.moon(BD2.sky(np.zeros((H, W, 3), np.float32), T, 1.0), T)
    im = L.load_rgba('素材/CG/p2_basilisk.png')[..., :3].astype(np.int16)
    g = (im[..., 1] > 170) & (im[..., 0] < 120) & (im[..., 2] < 120)
    spill = ~g & (im[..., 1] > np.maximum(im[..., 0], im[..., 2]) + 20)    # 边缘绿色溢色压回去
    im[..., 1] = np.where(spill, np.maximum(im[..., 0], im[..., 2]), im[..., 1])
    a = cv2.erode(np.where(g, 0, 255).astype(np.uint8), np.ones((3, 3), np.uint8))
    rgba = np.dstack([im.astype(np.uint8), a])
    sw = 430
    rgba = cv2.resize(rgba, (sw, int(sw * rgba.shape[0] / rgba.shape[1])), interpolation=cv2.INTER_AREA)
    img = L.paste_rgba(np.clip(img, 0, 255).astype(np.uint8), rgba, (W - sw) // 2 + 20, 16)   # 头顶留一点天
    img = BD2.city(img.astype(np.float32), T, 1.0)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------- 谢幕 ----------------
GUEST_THUMBS = ['素材/CG/s8b_修格斯真身.png', 'basilisk', '素材/CG/s7_中文房间.png',
                '素材/CG/p3_7_打火机_b.png', '素材/CG/p3_8a_蓝调酒吧_c.png', '素材/CG/p3_11_龙猫_b.png']


@functools.lru_cache(maxsize=None)
def thumb(path):
    src = basilisk_cg() if path == 'basilisk' else L.load_rgb(path)
    return cv2.resize(src, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)


def curtain(img, k, t=0.0):
    """酒红幕布从两边合拢：k=0 只露两边窄窄的幕，k=1 完全合上。幕布有竖褶，下摆金穗。"""
    img = np.array(img, np.float32)
    half = int(40 + (W / 2 - 40) * L.smooth(k))
    for side in (0, 1):
        x0, x1 = (0, half) if side == 0 else (W - half, W)
        img[:, x0:x1] = WINE
        for x in range(x0, x1):
            if (x - x0 + (0 if side == 0 else 7)) % 22 < 7: img[:, x] = WINE_D     # 竖褶
        img[H - 10:H - 4, x0:x1] = GOLD
        for x in range(x0, x1, 5): img[H - 4:H, x:x + 2] = GOLD                   # 金穗
    img[:14] = WINE_D; img[12:15] = GOLD                                         # 顶上的横幔
    return img


def curtain_stage(img, t, lit, beat):
    img = np.array(img, np.float32)
    img[:] = (60, 8, 26)
    cx = W // 2                                                          # 金色追光（整块梯形，不用渐变）
    pts = np.int32([(cx - 50, 0), (cx + 50, 0), (cx + 150, H), (cx - 150, H)])
    cv2.fillPoly(img, [pts], (120, 30, 50))
    cv2.ellipse(img, (cx, 300), (150, 18), 0, 0, 360, (200, 120, 90), -1)
    for i, p in enumerate(GUEST_THUMBS):                                # 两侧挂着嘉宾相框，跟着拍子轻轻晃
        x = 34 if i < 3 else W - 34 - 96
        y = 40 + (i % 3) * 80 + int(2 * np.sin(t * 3 + i))
        img[y - 3:y + 57, x - 3:x + 99] = GOLD
        img[y:y + 54, x:x + 96] = thumb(p)
    return curtain(img, 0, t)


T_FACE, T_CLOSE = 153.20, 153.90


def curtain_fx(rgb_at, t, a, b):
    img = rgb_at(t).copy()
    z = 1 + 1.4 * L.smooth((t - T_FACE) / .7)                            # 抬头以后推近到脸
    if z > 1.001:
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((320, 70), 0, z), (W, H), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REPLICATE)
    if t >= T_CLOSE:
        k = (t - T_CLOSE) / .35
        img = np.clip(curtain(img, min(k, 1), t), 0, 255).astype(np.uint8)
    return img
