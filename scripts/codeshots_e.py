"""codeshots_e: 第三部分 A、B 段的代码镜头（分镜见 docs/分镜/第三部分_89-124秒.md，89.12–109.35）。

    gato      89.12–91.40  接第二部分结尾的 NIL：N、L 溶解，I 拉长成竖瞳 → 猫眼 → 蒸汽波地平线上的黑猫（v2）
    reach     91.40–94.10  落日伸手 CG，don't 时溶解到 b；远处黑猫慢慢走远（v2）
    go        94.10–95.46  手垂下的 CG，天色一格一格暗下去
    led       95.46–96.62  夜里地平线亮起 P(DOOM) LED，数字乱跳
    counter   97.80–98.30  回形针计数器
    killswitch 98.86–99.74 玻璃罩里的停机按钮，便利贴「ON PTO」
    beach     99.74–100.64 海滩上度假的两个 Clawd，把对讲机埋进沙子
    map       100.64–101.46 RPG 世界地图，每个去处都被打叉
    panic     101.46–102.48 慌张左右看的 CG
    lighter   102.48–103.89 打火机 CG：掀盖 → lit 点着 → 凑到引线（v2，替掉代码火柴）
    fuse_run  103.89–105.94 火花沿引线追逐
    bar / blues / face       105.94–109.35 蓝调酒吧：三档嘴型 CG → 霓虹招牌 → 脸部特写越来越亮

接口同 part1_lib：scene(t) 纯函数、画全彩、t 是完整版歌曲时间。人物 CG 按关键姿势硬切（第三阶段的有限动画）。
自检：python codeshots_e.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np

import part1_lib as L

W, H = L.W, L.H
BEAT = L.BEAT132
INK = (12, 11, 26)
PAPER = (246, 241, 228)
ORANGE = (217, 119, 87)
YEL = (255, 210, 63)
BLUE = (34, 52, 150)
PINK = (255, 70, 170)
CYAN = (70, 225, 255)
RED = (230, 40, 50)
CLAWD, CLAWD_SH = (217, 119, 87), (170, 82, 62)
HZ = 200                                   # 蒸汽波地平线高度


# ================================================================
# 公共小工具（codeshots_f 也用）
# ================================================================
def f24(t):
    return int(np.floor(t * 24 + 1e-6))


def f12(t):
    return int(np.floor(t * 12 + 1e-6))


def box(img, x0, y0, x1, y1, col, ink=2):
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    if ink: cv2.rectangle(img, (x0 - ink, y0 - ink), (x1 - 1 + ink, y1 - 1 + ink), INK, -1)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), tuple(int(c) for c in col), -1)


def poly(img, pts, col, ink=2):
    pts = np.round(np.asarray(pts, np.float64)).astype(np.int32)
    if ink: cv2.polylines(img, [pts], True, INK, ink * 2, lineType=cv2.LINE_8)
    cv2.fillPoly(img, [pts], tuple(int(c) for c in col), lineType=cv2.LINE_8)


def text(img, x, y, s, col, scale=1):
    return L.draw_text(img, (int(x), int(y)), s, tuple(int(c) for c in col), scale=scale)


def text_c(img, cx, y, s, col, scale=1):
    w = int(L.font(16).getlength(s)) * scale
    return text(img, cx - w // 2, y, s, col, scale)


def star4(img, x, y, r, col):
    x, y, r = int(round(x)), int(round(y)), max(1, int(round(r)))
    img[max(y - r, 0):y + r + 1, max(x - 1, 0):x + 2] = col
    img[max(y - 1, 0):y + 2, max(x - r, 0):x + r + 1] = col


def dither_to(img, col, k, mask=None):
    """Bayer 网点往某个颜色过渡（不用透明度混合）。"""
    m = L.bayer_mask(k)
    if mask is not None: m = m & mask
    img[m] = col
    return img


def zoom(img, cx, cy, z):
    """以 (cx,cy) 为中心放大 z 倍并把它移到画面中间（最近邻，保持硬边）。"""
    m = cv2.getRotationMatrix2D((cx, cy), 0, z)
    m[:, 2] += W / 2 - cx, H / 2 - cy
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def window(img, x, y, w, h, title, body=(18, 20, 44), bar=BLUE):
    """PC-98 风窗口：墨线框 + 标题栏 + 双线边。"""
    box(img, x, y, x + w, y + h, body, 2)
    box(img, x, y, x + w, y + 16, bar, 0)
    cv2.rectangle(img, (x + 2, y + 18), (x + w - 3, y + h - 3), PAPER, 1)
    return text(img, x + 4, y - 1, title, PAPER)


@functools.lru_cache(maxsize=None)
def cg(path):
    return L.load_rgb(path).copy()


@functools.lru_cache(maxsize=None)
def cg_aligned(path, base):
    """把改图出来的 CG 对齐到底图（改图会轻微推拉构图）：ORB 特征 + 相似变换。"""
    a, b = cg(path), cg(base)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    ms = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda m: m.distance)[:400]
    pa = np.float32([ka[m.queryIdx].pt for m in ms]); pb = np.float32([kb[m.trainIdx].pt for m in ms])
    M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3)
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def key_at(t, keys):
    """[(开始时间, 值)] → 当前值（关键姿势表）。"""
    v = keys[0][1]
    for a, x in keys:
        if t >= a: v = x
    return v


def since(t, keys):
    return t - max([a for a, _ in keys if a <= t] or [keys[0][0]])


def bounce(dt, px=3):
    return int(round(px * np.sin(np.pi * dt / .15))) if 0 <= dt < .15 else 0


def lift(img, dy):
    return np.concatenate([img[dy:], np.repeat(img[-1:], dy, 0)]) if dy > 0 else img


# ================================================================
# 蒸汽波落日（1a / 2b / 3b 共用；1b、3a 是 CG，自己画了同款落日）
# ================================================================
SKY = [(34, 10, 70), (60, 14, 100), (104, 20, 128), (160, 30, 140), (214, 50, 150), (255, 96, 150)]
SUN = [(255, 226, 90), (255, 176, 70), (255, 120, 90), (255, 70, 140)]
FLOOR, GRID = (22, 6, 44), (255, 60, 200)
PALM = (26, 6, 44)


def palm(img, x, base, h, lean):
    top = (x + lean, base - h)
    cv2.line(img, (x, base), top, PALM, 5)
    for a in (-150, -120, -60, -30, -175, -5):
        r = np.deg2rad(a)
        e = (int(top[0] + np.cos(r) * 34), int(top[1] + np.sin(r) * 14 + 16))
        m = ((top[0] + e[0]) // 2, top[1] - 6 + abs(a + 90) // 12)
        cv2.polylines(img, [np.int32([top, m, e])], False, PALM, 4)


def vapor(t, sun=1.0, night=0.0, sunk=0.0, grid=True):
    """sun 太阳可见度（0 = 已落下）；night 天色变暗；sunk 太阳往下沉的比例。"""
    img = np.zeros((H, W, 3), np.uint8)
    bands = SKY if night < .5 else [(10, 6, 30), (16, 8, 44), (24, 10, 60), (36, 12, 76), (50, 14, 90), (70, 18, 100)]
    for k, c in enumerate(bands):
        img[int(HZ * k / len(bands)):int(HZ * (k + 1) / len(bands))] = c
    for i in range(30):                                  # 星星（越暗越多）
        if L.hash01(i, 1) > .3 + .6 * night: continue
        x, y = int(L.hash01(i, 2) * W), int(L.hash01(i, 3) * HZ * .6)
        if (f12(t) + i) % 5: img[y, x] = PAPER
    if sun > 0:
        cx, cy, r = 320, int(HZ - 18 + 120 * sunk), 96
        yy, xx = np.mgrid[0:HZ, 0:W]
        d = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        for k, c in enumerate(SUN):                      # 太阳四段平涂色
            y0 = cy - r + 2 * r * k / 4
            band = d & (yy >= y0) & (yy < y0 + 2 * r / 4 + 1)
            img[:HZ][band] = c
        scroll = (t * 14) % 12                           # 下半边的横条镂空，缓缓往下滚
        for j in range(8):
            y = int(cy + 4 + j * 12 + scroll)
            th = 1 + j // 2
            m = d.copy(); m[:max(y, 0)] = False; m[min(y + th, HZ):] = False
            img[:HZ][m] = bands[-1]
    for x, base, h, lean in ((40, HZ, 70, 8), (78, HZ, 52, -6), (590, HZ, 76, -10), (556, HZ, 48, 6)):
        palm(img, x, base, h, lean)
    for i in range(18):                                  # 城市剪影
        x = 110 + i * 24 if i < 9 else 330 + i * 12
        if 200 < x < 440: continue
        hh = 10 + int(L.hash01(i, 5) * 26)
        img[HZ - hh:HZ, x:x + 18] = PALM
        for wy in range(HZ - hh + 3, HZ - 2, 5):
            if L.hash01(i, wy) < .4: img[wy:wy + 2, x + 4:x + 6] = YEL
    img[HZ:] = FLOOR
    if grid:
        ph = (t * 1.6) % 1
        for k in range(14):                              # 横线：透视间距，往观众方向滚
            z = (k + 1 - ph)
            y = int(HZ + 170 / z * .9)
            if HZ < y < H: img[y:y + 1 + (y > 270)] = GRID
        for k in range(-12, 13):                         # 竖线：汇向地平线中点
            cv2.line(img, (320 + k * 12, HZ), (320 + k * 110, H + 40), GRID, 1)
        img[HZ:HZ + 2] = (255, 120, 220)
        if sun > 0:                                      # 太阳倒影
            for j in range(6):
                y = HZ + 6 + j * 9
                w = 70 - j * 9
                img[y:y + 2, 320 - w:320 + w] = SUN[min(j // 2 + 1, 3)]
    return img


def cat(img, x, y, s=1.0, pose='sit', eyes=True, tail=0.0, col=(8, 6, 16), rim=None):
    """像素黑猫剪影。(x,y) = 脚底中点；pose: sit / jump / walk；tail 尾巴摆动相位。"""
    S = lambda v: int(round(v * s))
    if pose == 'sit':
        cv2.ellipse(img, (x, y - S(20)), (S(17), S(22)), 0, 0, 360, col, -1)
        hx, hy = x + S(4), y - S(46)
    else:
        cv2.ellipse(img, (x, y - S(14)), (S(26), S(11)), -8 if pose == 'jump' else 0, 0, 360, col, -1)
        hx, hy = x + S(24), y - S(26)
        for lx, dx in ((x - S(18), -S(10) if pose == 'jump' else 0), (x + S(14), S(12) if pose == 'jump' else 0)):
            cv2.line(img, (lx, y - S(8)), (lx + dx, y), col, max(1, S(5)))
    cv2.circle(img, (hx, hy), S(13), col, -1)
    for ex in (-1, 1):
        poly(img, [(hx + ex * S(4), hy - S(8)), (hx + ex * S(13), hy - S(8)), (hx + ex * S(11), hy - S(22))], col, 0)
    a = np.linspace(0, 1, 12)
    tx = x - S(16) - S(22) * a
    ty = y - S(6) - S(40) * a ** 2 + S(6) * np.sin(a * 3 + tail)
    cv2.polylines(img, [np.int32(np.stack([tx, ty], 1))], False, col, max(2, S(5)))
    if rim is not None:
        cv2.ellipse(img, (x, y - S(20)), (S(17), S(22)), 0, -80, 20, rim, 1) if pose == 'sit' else None
    if eyes:
        for ex in (-1, 1):
            cv2.ellipse(img, (hx + ex * S(5), hy - S(1)), (S(3), S(2)), 0, 0, 360, YEL, -1)
    return img


# ================================================================
# 1a gato：黑底猫眼 → 地平线上的猫
# ================================================================
T_GATO, T_PLEASE, T_DONT, T_LET, T_GO, T_IM, T_DROP = 89.12, 90.01, 92.10, 92.86, 94.10, 95.46, 96.62
# v2（用户：Gato 段抒情，切慢一点；要接上第二部分结尾的 NIL）：6 镜 → 4 镜
T_REACH = 91.40
NIL_S, NIL_CY = 4, 150                  # 和 codeshots_d 结尾一样：放大 4 倍、中心 (320, 150)


def nil_masks():
    """第二部分结尾的 NIL，拆成 N / I / L 三个字母的遮罩（整帧坐标）。"""
    f = L.font(16)
    tw = (f.getbbox('NIL')[2] - f.getbbox('NIL')[0] + 2) * NIL_S
    x0, y0 = int(round(320 - tw / 2)), int(round(NIL_CY - 8 * NIL_S))
    whole = L.draw_text(np.zeros((H, W, 3), np.uint8), (x0, y0), 'NIL', (255, 255, 255), scale=NIL_S)[..., 0] > 0
    col = np.append(whole.any(0), False)                               # 按列投影找字母之间的空隙来切
    starts = [x for x in range(1, W) if col[x] and not col[x - 1]]
    cut1, cut2 = starts[1], starts[2]
    xx = np.arange(W)[None, :]
    return whole & (xx < cut1), whole & (xx >= cut1) & (xx < cut2), whole & (xx >= cut2)


def cat_eye(img, ex, ey, open_k, slit_w, slit_h):
    hh = max(1, int(44 * open_k))
    if open_k > .02:
        cv2.ellipse(img, (ex, ey), (74, hh), 0, 0, 360, (40, 30, 10), -1)
        cv2.ellipse(img, (ex, ey), (70, max(1, hh - 4)), 0, 0, 360, YEL, -1)
        cv2.ellipse(img, (ex, ey - hh // 3), (60, max(1, hh // 3)), 0, 0, 360, (255, 240, 150), -1)
    cv2.ellipse(img, (ex, ey), (max(1, slit_w), max(1, slit_h)), 0, 0, 360, INK if open_k > .3 else PAPER, -1)
    if open_k > .5: star4(img, ex - 26, ey - hh // 2, 3, PAPER)


def scene_gato(t):
    t = float(np.clip(t, T_GATO, T_REACH - 1e-3))
    if t < 90.80:
        img = np.full((H, W, 3), INK, np.uint8)
        mN, mI, mL = nil_masks()
        if t < 89.95:                                                  # NIL 停住 → N、L 网点溶解掉，I 往上下拉长
            k = L.smooth((t - 89.50) / .35)
            fade = ~L.bayer_mask(k)
            img[(mN | mL) & fade] = PAPER
            ys, xs = np.nonzero(mI)                                    # I 整块纵向放大（最近邻，保持像素硬边）
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            hh = int(round((y1 - y0) * (1 + 1.6 * k)))
            tall = cv2.resize(mI[y0:y1, x0:x1].astype(np.uint8), (x1 - x0, hh), interpolation=cv2.INTER_NEAREST) > 0
            top = int(round((y0 + y1) / 2 - hh / 2))
            img[max(top, 0):top + hh, x0:x1][tall[max(0, -top):]] = PAPER
            return img
        u = L.smooth((t - 89.95) / .30)                                # 先：I 一分为二滑向两边
        o = L.smooth((t - 90.15) / .30)                                # 后：猫眼睁开
        blink = 1 - .9 * np.sin(np.pi * np.clip((t - 90.52) / .14, 0, 1))
        for side in (-1, 1):
            ex = int(320 + side * 90 * u)
            cat_eye(img, ex, NIL_CY, o * blink, 4 + int(4 * o), int(40 * blink))
        return img
    img = vapor(t, sun=1.0, sunk=.1)
    img = cat(img, 420, HZ + 2, 1.3, 'sit', tail=f12(t) * .5, rim=PINK)
    z = 1.25 - .25 * L.smooth((t - 90.80) / .6)                        # 慢慢往后拉
    img = zoom(img, 420, HZ - 50, z)
    if t < 91.00:                                                      # 从猫眼溶解进来
        dark = np.full_like(img, INK)
        img = np.where(L.bayer_mask((t - 90.80) / .2)[..., None], img, dark)
    return img


# ================================================================
# 1b reach / 2a hand / 3a go：落日伸手 CG
# ================================================================
SUN_A, SUN_B, SUN_C = ('素材/CG/p3_1b_落日伸手_a.png', '素材/CG/p3_1b_落日伸手_b.png', '素材/CG/p3_1b_落日伸手_c.png')


def sun_cg(which):
    return cg(SUN_A) if which == 'a' else cg_aligned(which, SUN_A)


def motes(img, t, n=26, seed=3):
    """落日里往上飘的光点（按时间直接算位置）。"""
    for i in range(n):
        sp = 18 + 30 * L.hash01(i, seed)
        x = (L.hash01(i, seed, 1) * W + 12 * np.sin(t * 1.3 + i)) % W
        y = (L.hash01(i, seed, 2) * H - t * sp) % H
        c = PAPER if i % 3 else YEL
        if (f12(t) + i) % 4: star4(img, x, y, 1 + (i % 3 == 0), c)
    return img


def scene_reach(t):
    """v2：91.40 伸手 a → don't（92.10）网点溶解到 b（手伸到镜头前），一直停到 go。远处地平线上黑猫慢慢走远。"""
    t = float(np.clip(t, T_REACH, T_GO - 1e-3))
    def pose(which):
        img = sun_cg(which).copy()
        u = (t - T_REACH) / (T_GO - T_REACH)                           # 黑猫从她右边往画面外走，4 帧一步
        step = f24(t) // 6
        cat(img, int(470 + 150 * u), 262, .55, 'walk' if step % 2 else 'sit', eyes=False, tail=step, rim=PINK)
        return img
    img = pose('a') if t < T_DONT else pose(SUN_B)
    if T_DONT <= t < T_DONT + .25:
        img = np.where(L.bayer_mask((t - T_DONT) / .25)[..., None], img, pose('a'))
    img = zoom(img, 330, 170, 1 + .06 * (t - T_REACH) / (T_GO - T_REACH))
    return motes(img, L.twos(t))


def scene_go(t):
    t = float(np.clip(t, T_GO, T_IM - 1e-3))
    img = zoom(sun_cg(SUN_C).copy(), 330, 170, 1.05 - .04 * (t - T_GO) / (T_IM - T_GO))
    k = L.smooth((t - 94.5) / .9)                                   # 天一格一格暗下去（4 档，像调色板循环）
    k = np.floor(k * 4) / 4
    if k > 0: img = dither_to(img, (30, 10, 60), .55 * k)
    return motes(img, L.twos(t), 10, 9)


LED_ON = (255, 50, 40)


def led_text(img, cx, y, s, cell=5, col=LED_ON):
    m = L.draw_text(np.zeros((20, int(L.font(16).getlength(s)) + 4, 3), np.uint8), (0, 0), s, (255, 255, 255))[..., 0] > 0
    m = m[m.any(1)]
    h, w = m.shape
    x0 = cx - w * cell // 2
    box(img, x0 - 10, y - 8, x0 + w * cell + 10, y + h * cell + 8, (14, 8, 12), 3)
    for r in range(h):
        for c in range(w):
            p = (x0 + c * cell + cell // 2, y + r * cell + cell // 2)
            if m[r, c]:
                cv2.circle(img, p, cell // 2, col, -1)
                img[p[1], p[0]] = (255, 220, 190)                   # 亮点的白芯
            else:
                img[p[1], p[0]] = (40, 12, 16)
    return img


def scene_led(t):
    t = float(np.clip(t, T_IM, T_DROP - 1e-3))
    img = vapor(t, sun=0, night=1.0)
    v = 71 if t < 95.66 else int(71 + L.hash01(f24(t) // 2, 4) * 28)
    s = f'P(DOOM) {v}%' if t < 96.32 or f24(t) % 4 < 2 else 'P(DOOM) ??%'
    img = led_text(img, W // 2, 120, s, 5)
    return img


# ================================================================
# 4b counter：回形针计数器
# ================================================================
def clip_icon(img, x, y, s, ang=0, col=(200, 204, 216)):
    """回形针（三段 U 形铁丝）。"""
    pts = np.array([(0, 34), (0, 4), (8, -4), (16, 4), (16, 40), (6, 46), (-4, 38), (-4, 10), (4, 4), (10, 10), (10, 30)], np.float64)
    r = np.deg2rad(ang)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    p = (pts - (6, 20)) @ R.T * s + (x, y)
    cv2.polylines(img, [np.int32(p)], False, INK, max(3, int(4 * s)))
    cv2.polylines(img, [np.int32(p)], False, col, max(1, int(2 * s)))


def office_bg(t):
    img = np.full((H, W, 3), (214, 216, 226), np.uint8)
    for x in range(0, W, 64): img[:, x:x + 2] = (190, 192, 206)
    for y in range(0, 60, 30): img[y:y + 2] = (190, 192, 206)
    for i in range(40):                                             # 天花板一直在漏回形针
        sp = 160 + 90 * L.hash01(i, 1)
        x = L.hash01(i, 2) * W
        y = (L.hash01(i, 3) * H + t * sp) % (H + 60) - 30
        clip_icon(img, x, y, .5, ang=(i * 37 + f12(t) * 25) % 360)
    return img


def scene_counter(t):
    img = office_bg(t)
    box(img, 60, 70, 580, 230, (30, 36, 30), 4)
    box(img, 72, 82, 568, 218, (150, 180, 110), 0)                   # 液晶屏
    clip_icon(img, 118, 150, 1.6, ang=-20, col=(90, 100, 80))
    n = 10 ** (6 + max(0, t - 97.80) * 15)
    s = f'{int(n):,}' if n < 1e10 else f'{n:.1E}'.replace('+', '')
    return text(img, 170, 118, s[-15:], (36, 50, 30), 4 if len(s) < 11 else 3)


# ================================================================
# 5a killswitch / 5b beach
# ================================================================
# 10-19 用户：按钮要精细、立体，看得出玻璃罩和按钮（拿出像素画的功力）→ 重画：
# 钢板墙 + 铆钉面板（上左亮边、下右暗边）、金属底座三档明暗、红色蘑菇头（顶面 / 侧面 / 高光三档）、
# 玻璃罩（罩内网点透青、边缘双线、两道弧形反光 + 一颗定时闪的高光）、罩前挂锁；
# 讲故事的小东西：罩顶的灰、蜘蛛网和吊着晃的小蜘蛛（好久没人碰过）、便利贴在空调风里轻轻晃、待机灯闪。
STEEL, STEEL_HI, STEEL_LO, STEEL_DK = (78, 84, 108), (120, 128, 156), (52, 56, 76), (34, 36, 52)
RED_HI, RED_LO, RED_DK = (255, 120, 120), (160, 22, 34), (96, 10, 22)
GLASS, GLASS_HI = (150, 220, 255), (236, 250, 255)


def _bevel(img, x0, y0, x1, y1, c, hi, lo, w=3):
    box(img, x0, y0, x1, y1, c, 2)
    img[y0:y0 + w, x0:x1] = hi; img[y0:y1, x0:x0 + w] = hi
    img[y1 - w:y1, x0:x1] = lo; img[y0:y1, x1 - w:x1] = lo


def _rivet(img, x, y):
    cv2.circle(img, (x, y), 4, INK, -1); cv2.circle(img, (x, y), 3, STEEL_LO, -1)
    cv2.circle(img, (x - 1, y - 1), 1, STEEL_HI, -1)


def scene_killswitch(t):
    img = np.full((H, W, 3), STEEL_DK, np.uint8)
    for x in range(0, W, 128):                                     # 墙上的钢板：接缝 + 铆钉
        for y in range(0, H, 96):
            img[y:y + 2, :] = (24, 26, 38); img[:, x:x + 2] = (24, 26, 38)
            img[y + 2:y + 3, x + 2:x + 126] = (48, 52, 70)
            for rx, ry in ((x + 10, y + 10), (x + 118, y + 10), (x + 10, y + 86), (x + 118, y + 86)):
                cv2.circle(img, (rx, ry), 2, (60, 64, 86), -1)
    _bevel(img, 168, 38, 472, 272, STEEL, STEEL_HI, STEEL_LO, 4)   # 控制面板
    for rx, ry in ((182, 52), (458, 52), (182, 258), (458, 258)): _rivet(img, rx, ry)
    for k in range(13):                                            # 底部黄黑警示条
        x = 172 + k * 24
        poly(img, [(x, 252), (x + 12, 252), (x + 24, 268), (x + 12, 268)], YEL if k % 2 else INK, 0)
    img[250:252, 172:468] = INK
    _bevel(img, 230, 50, 410, 78, (24, 22, 30), (60, 58, 70), (10, 10, 14), 2)   # 名牌（刻字）
    img = text_c(img, 321, 56, 'KILLSWITCH', (120, 90, 20), 1)
    img = text_c(img, 320, 55, 'KILLSWITCH', YEL, 1)
    on = f24(t) % 12 < 6                                           # 待机灯
    cv2.circle(img, (196, 236), 6, INK, -1); cv2.circle(img, (196, 236), 4, (80, 255, 120) if on else (20, 70, 40), -1)
    img = text(img, 206, 228, 'STANDBY', PAPER)
    cv2.circle(img, (444, 236), 6, INK, -1); cv2.circle(img, (444, 236), 4, (70, 20, 24), -1)
    img = text(img, 386, 228, 'ARMED', (150, 150, 170))
    cx, base = 320, 196
    # 底座：金属圆台（外圈亮、内圈暗）
    cv2.ellipse(img, (cx, base + 6), (96, 26), 0, 0, 360, INK, -1)
    cv2.ellipse(img, (cx, base + 4), (93, 23), 0, 0, 360, STEEL_LO, -1)
    cv2.ellipse(img, (cx, base), (93, 23), 0, 0, 360, STEEL_HI, -1)
    cv2.ellipse(img, (cx, base), (80, 18), 0, 0, 360, (96, 102, 128), -1)
    cv2.ellipse(img, (cx, base - 1), (70, 15), 0, 0, 360, INK, -1)
    # 红色蘑菇头：侧面（左亮右暗三档）+ 顶面 + 高光
    top = base - 40
    side = np.zeros((H, W), np.uint8)
    cv2.ellipse(side, (cx, base - 4), (64, 16), 0, 0, 180, 1, -1)
    side[top:base - 4, cx - 64:cx + 65] = 1
    img[side > 0] = RED_LO
    cols = np.arange(W)
    band = (side > 0) & (cols[None, :] < cx - 30)
    img[band] = RED
    img[(side > 0) & (cols[None, :] > cx + 34)] = RED_DK
    img[(side > 0) & (cols[None, :] > cx - 50) & (cols[None, :] < cx - 40)] = RED_HI
    cv2.ellipse(img, (cx, top), (64, 18), 0, 0, 360, INK, 3)
    cv2.ellipse(img, (cx, top), (63, 17), 0, 0, 360, RED, -1)
    cv2.ellipse(img, (cx + 6, top + 3), (50, 11), 0, 0, 360, (200, 30, 44), -1)
    cv2.ellipse(img, (cx - 22, top - 6), (22, 5), -8, 0, 360, RED_HI, -1)
    cv2.ellipse(img, (cx - 28, top - 7), (8, 2), -8, 0, 360, PAPER, -1)
    cv2.line(img, (cx - 64, top), (cx - 64, base - 4), INK, 3); cv2.line(img, (cx + 64, top), (cx + 64, base - 4), INK, 3)
    cv2.ellipse(img, (cx, base - 4), (64, 16), 0, 0, 180, INK, 3)
    # 玻璃罩：半球，罩内网点透青
    dome = np.zeros((H, W), np.uint8)
    cv2.ellipse(dome, (cx, base), (108, 112), 0, 180, 360, 1, -1)
    cv2.ellipse(dome, (cx, base), (108, 24), 0, 0, 180, 1, -1)
    img = dither_to(img, GLASS, .22, dome > 0)
    cv2.ellipse(img, (cx, base), (108, 112), 0, 180, 360, INK, 4)
    cv2.ellipse(img, (cx, base), (106, 110), 0, 180, 360, GLASS, 2)
    cv2.ellipse(img, (cx, base), (96, 100), 0, 200, 250, GLASS_HI, 5)        # 两道弧形反光
    cv2.ellipse(img, (cx, base), (84, 88), 0, 205, 232, GLASS_HI, 2)
    cv2.ellipse(img, (cx, base), (100, 104), 0, 300, 330, GLASS, 3)
    cv2.ellipse(img, (cx, base + 2), (108, 24), 0, 0, 180, INK, 4)           # 罩子底圈（前半）
    cv2.ellipse(img, (cx, base + 2), (106, 22), 0, 10, 170, GLASS, 2)
    g = (t * 1.4) % 1                                                         # 高光沿罩子边滑过去、闪一下
    if g < .25:
        a = np.radians(200 + 120 * g / .25)
        star4(img, cx + np.cos(a) * 104, base + np.sin(a) * 108, 5, GLASS_HI)
    for i in range(14):                                                       # 罩顶落的灰
        a = np.radians(235 + 70 * L.hash01(i, 71)); r = 104 + 3 * L.hash01(i, 72)
        img[int(base + np.sin(a) * r * 1.04), int(cx + np.cos(a) * r)] = (180, 176, 168)
    # 挂锁：罩子底圈前面
    lx, ly = cx + 2, base + 24
    cv2.ellipse(img, (lx, ly - 8), (9, 10), 0, 180, 360, INK, 5); cv2.ellipse(img, (lx, ly - 8), (9, 10), 0, 180, 360, (190, 192, 204), 2)
    _bevel(img, lx - 13, ly - 8, lx + 13, ly + 14, (214, 176, 80), (255, 230, 140), (150, 110, 40), 2)
    img[ly:ly + 6, lx - 1:lx + 2] = INK; cv2.circle(img, (lx, ly), 2, INK, -1)
    # 蜘蛛网：罩顶 → 面板左上角
    hub = np.array([214., 74.])
    spokes = [np.array(p, float) for p in ((180, 44), (250, 44), (180, 120), (cx - 72, base - 82), (cx - 104, base - 30))]
    for p in spokes: cv2.line(img, tuple(hub.astype(int)), tuple(p.astype(int)), (200, 204, 216), 1)
    for r in (.3, .55, .8):
        ring = [hub + (p - hub) * r for p in spokes]
        for a, b in zip(ring, ring[1:]): cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (170, 174, 190), 1)
    sx = int(236 + 6 * np.sin(L.twos(t) * 5)); sy = int(118 + 8 * np.sin(L.twos(t) * 3.1))     # 吊着的小蜘蛛
    cv2.line(img, (236, 84), (sx, sy), (200, 204, 216), 1)
    cv2.circle(img, (sx, sy), 4, INK, -1)
    for k in (-1, 1):
        for j in range(3): cv2.line(img, (sx, sy), (sx + k * 7, sy - 3 + j * 3), INK, 1)
    # 便利贴（贴在玻璃上，被风吹得轻轻晃）
    note = np.full((78, 110, 3), (255, 236, 110), np.uint8)
    note[:10] = (240, 214, 80)
    note = text(note, 8, 10, 'ON PTO', (40, 30, 30), 1)
    note = text(note, 8, 32, 'back', (40, 30, 30), 1)
    note = text(note, 8, 52, 'Monday :)', (40, 30, 30), 1)
    M = cv2.getRotationMatrix2D((55, 4), 8 + 3 * np.sin(L.twos(t) * 6), 1.0)
    rot = cv2.warpAffine(note, M, (110, 78), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    msk = cv2.warpAffine(np.ones((78, 110), np.uint8), M, (110, 78), flags=cv2.INTER_NEAREST) > 0
    edge = cv2.dilate(msk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sub = img[112:190, 372:482]
    sub[edge] = INK; sub[msk] = rot[msk]
    img[112:116, 410:430] = (190, 220, 230)                                  # 透明胶带
    z = 1 + .1 * L.smooth((t - 98.86) / .88)
    return zoom(img, 330, 158, z)


# 10-19 用户：海滩加椰子树；Clawd 看不出是背对的 → 改成真正的背影：两把沙滩椅背朝镜头、面朝大海，
# 只露出 Clawd 方块头顶和枕在脑后的两只小手（左边那只戴草帽、举着椰子），右边那只的对讲机在沙子上响，
# 100.25 一只手从椅子边伸下来，把对讲机按进沙子里。椰子树叶子随风慢慢摆，海浪往岸上推，海鸥飞过。
TRUNK, TRUNK_DK = (150, 100, 60), (104, 66, 40)
LEAF, LEAF_DK = (60, 170, 80), (30, 110, 60)
SAND, SAND_DK, SAND_HI = (240, 210, 150), (206, 170, 112), (252, 232, 186)


def beach_palm(img, x, base, h, lean, t, flip=1):
    """精细椰子树：分节树干（每节一道暗环）+ 两色大叶 + 椰子；叶子按正弦慢摆。"""
    n = 9
    pts = [(x + lean * (k / n) ** 1.6, base - h * k / n) for k in range(n + 1)]
    for k in range(n):
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        w = int(16 - 7 * k / n)
        poly(img, [(x0 - w / 2, y0), (x0 + w / 2, y0), (x1 + w / 2 - 1, y1), (x1 - w / 2 + 1, y1)], TRUNK, 2)
        cv2.line(img, (int(x0 - w / 2 + 2), int(y0 - 1)), (int(x0 + w / 2 - 2), int(y0 - 4)), TRUNK_DK, 2)
    tx, ty = pts[-1]
    sway = 4 * np.sin(L.twos(t) * 2.2 + x)
    for a in (-170, -140, -105, -75, -40, -10, 20, 160):
        r = np.deg2rad(a + sway)
        ln = 64 if a in (-170, -10, 20, 160) else 52
        mid = (tx + np.cos(r) * ln * .5, ty + np.sin(r) * ln * .3 - 14)
        end = (tx + np.cos(r) * ln, ty + np.sin(r) * ln * .45 + 18)
        cv2.polylines(img, [np.int32([(tx, ty), mid, end])], False, INK, 11)
        cv2.polylines(img, [np.int32([(tx, ty), mid, end])], False, LEAF_DK, 8)
        cv2.polylines(img, [np.int32([(tx, ty - 2), mid, end])], False, LEAF, 4)
        for j in range(1, 5):                                      # 叶片的锯齿
            px = tx + (end[0] - tx) * j / 5; py = ty + (end[1] - ty) * j / 5 - 10 * np.sin(np.pi * j / 5)
            cv2.line(img, (int(px), int(py)), (int(px + 5 * np.sign(np.cos(r)) ), int(py + 8)), LEAF_DK, 2)
    for k, (dx, dy) in enumerate(((-6, 6), (5, 7), (0, 12))):        # 椰子
        cv2.circle(img, (int(tx + dx), int(ty + dy)), 6, INK, -1)
        cv2.circle(img, (int(tx + dx), int(ty + dy)), 5, (110, 70, 36), -1)
        cv2.circle(img, (int(tx + dx - 2), int(ty + dy - 2)), 1, (170, 120, 70), -1)


def beach_chair_back(img, cx, seat, stripe):
    """沙滩椅背面：木框 + 条纹帆布（背朝镜头），底下两条椅腿。"""
    x0, x1, y0 = cx - 58, cx + 58, seat - 64
    cv2.ellipse(img, (cx + 14, seat + 6), (72, 9), 0, 0, 360, SAND_DK, -1)    # 影子
    for lx in (x0 + 6, x1 - 12):
        box(img, lx, seat - 6, lx + 6, seat + 4, (130, 86, 50), 2)
    poly(img, [(x0, seat - 4), (x1, seat - 4), (x1 - 8, y0), (x0 + 8, y0)], (130, 86, 50), 2)
    for k in range(6):                                             # 帆布条纹
        a, b = x0 + 6 + k * 18, x0 + 6 + (k + 1) * 18
        poly(img, [(a, seat - 8), (min(b, x1 - 6), seat - 8), (min(b, x1 - 6) - 6, y0 + 4), (a + 6, y0 + 4)],
             stripe if k % 2 == 0 else PAPER, 0)
    img[seat - 10:seat - 8, x0 + 4:x1 - 4] = (130, 86, 50)          # 横撑
    return y0


def clawd_back(img, cx, top, hat=False):
    """Clawd 背影：椅背上方露出方块身子的上半截（没有脸，只有后脑勺的亮边）+ 两只小短手搭在扶手上。"""
    box(img, cx - 38, top - 32, cx + 38, top + 8, CLAWD, 3)
    img[top - 32:top - 26, cx - 35:cx + 35] = (236, 146, 112)       # 顶面亮边
    img[top - 32:top + 8, cx + 26:cx + 35] = CLAWD_SH               # 右侧背光面
    for s in (-1, 1):                                              # 小短手搭在扶手上
        x0 = cx + s * 38 - (0 if s > 0 else 16)
        box(img, x0, top - 6, x0 + 16, top + 4, CLAWD, 2)
    if hat:
        cv2.ellipse(img, (cx, top - 33), (50, 9), 0, 0, 360, INK, -1)
        cv2.ellipse(img, (cx, top - 33), (48, 7), 0, 0, 360, (236, 200, 110), -1)
        box(img, cx - 20, top - 52, cx + 20, top - 33, (236, 200, 110), 2)
        img[top - 38:top - 34, cx - 20:cx + 20] = RED                # 帽带


def scene_beach(t):
    img = np.zeros((H, W, 3), np.uint8)
    for k, c in enumerate([(110, 190, 255), (140, 206, 255), (172, 222, 255), (200, 236, 255)]): img[k * 30:(k + 1) * 30] = c
    cv2.circle(img, (470, 44), 24, (255, 250, 210), -1)
    for i in range(3):                                             # 云，慢慢漂
        x = int((120 + i * 210 + t * 12) % (W + 120)) - 60; y = 30 + i * 16
        for dx, r in ((0, 14), (16, 18), (34, 12)): cv2.circle(img, (x + dx, y), r, PAPER, -1)
    for i in range(3):                                             # 海鸥
        x = int((L.hash01(i, 81) * W + t * 60) % (W + 40)) - 20; y = int(50 + 30 * L.hash01(i, 82))
        f = (f12(t) + i) % 4 < 2
        cv2.polylines(img, [np.int32([(x - 6, y - (3 if f else -2)), (x, y), (x + 6, y - (3 if f else -2))])], False, INK, 1)
    for k, c in enumerate([(30, 100, 190), (44, 124, 214), (60, 150, 230), (90, 180, 240)]):   # 海：远深近浅
        img[120 + k * 16:136 + k * 16] = c
    for i in range(16):                                            # 海面闪光
        x = int(L.hash01(i, 91) * W); y = int(124 + L.hash01(i, 92) * 56)
        if (f12(t) + i) % 6 < 2: star4(img, x, y, 1, PAPER)
    wave = 184 + 4 * np.sin(L.twos(t) * 3)                         # 浪往岸上推
    img[184:] = SAND
    xs = np.arange(W)
    edge = (wave + 3 * np.sin(xs / 22 + L.twos(t) * 2)).astype(int)
    for x in range(W):
        img[180:edge[x], x] = (90, 180, 240)
        img[edge[x]:edge[x] + 3, x] = PAPER
    for i in range(70):                                            # 沙粒
        y = 196 + int(L.hash01(i, 1) * 160); x = int(L.hash01(i, 2) * W)
        img[y, x] = SAND_DK if i % 3 else SAND_HI
    beach_palm(img, 46, 300, 230, 70, t)
    beach_palm(img, 612, 300, 210, -66, t)
    # 遮阳伞（插在两把椅子中间后面）
    cv2.line(img, (322, 250), (318, 132), (120, 80, 40), 5)
    cv2.ellipse(img, (318, 134), (74, 26), 0, 180, 360, INK, -1)
    for k in range(6):
        cv2.ellipse(img, (318, 134), (72, 24), 0, 180 + k * 30, 180 + (k + 1) * 30, RED if k % 2 else PAPER, -1)
    # 两把椅子 + Clawd 背影
    bury = L.smooth((t - 100.25) / .2)
    for cx, stripe, hat in ((222, (70, 150, 230), True), (420, (255, 120, 160), False)):
        top = beach_chair_back(img, cx, 272, stripe)
        clawd_back(img, cx, top + 4, hat)
    # 左边那只：一只手举着椰子吸管（盖在左扶手那只手上面）
    bob = 2 if f12(t) % 8 < 4 else 0
    box(img, 160, 196 - bob, 176, 214 - bob, CLAWD, 2)
    cv2.circle(img, (166, 186 - bob), 12, INK, -1); cv2.circle(img, (166, 186 - bob), 10, (120, 80, 40), -1)
    cv2.line(img, (170, 176 - bob), (182, 160 - bob), (255, 120, 180), 2)
    # 右边那只：手里举着手机在响 → 100.25（拍点，part3 HITS 有闪白震屏）手机在手里炸了 → 黑烟往上飘
    # （11-12 用户：原来是手伸下来把对讲机按进沙里，动作不协调 → 改成手机在手里爆炸）
    BOOM_T = 100.25
    # 手和左边拿椰子那只镜像对称：扶手上一个方块手，手机举在手上面（用户：照左边拿椰子那个来）
    burnt = t >= BOOM_T
    box(img, 466, 196 - bob, 482, 214 - bob, INK if burnt else CLAWD, 2)   # 手（炸完熏黑）
    hx, hy = 470, 196 - bob                                        # 手机底边中点附近
    if not burnt:
        jx = (1 if f24(t) % 2 else -1) if f24(t) % 6 < 3 else 0   # 来电震动
        # 早期大哥大（用户 11-12）：厚机身 + 左上角拉杆天线（圆头）+ 绿液晶小屏 + 按键点阵
        X = hx - 5 + jx
        box(img, X, hy - 34, X + 15, hy + 1, (40, 44, 50), 2)
        cv2.line(img, (X + 3, hy - 35), (X + 3, hy - 52), INK, 2)
        cv2.circle(img, (X + 3, hy - 53), 2, INK, -1)
        img[hy - 30:hy - 22, X + 3:X + 12] = (150, 210, 120) if f24(t) % 6 < 3 else (70, 96, 70)   # 液晶亮
        for r in range(4):
            for c in range(3):
                img[hy - 18 + r * 4:hy - 16 + r * 4, X + 3 + c * 4:X + 5 + c * 4] = (150, 154, 160)
        if f24(t) % 6 < 3:
            img = text(img, hx + 14, hy - 40, '!!', RED, 2)
            for k in range(3):                                     # 震动纹
                cv2.line(img, (hx + 12 + k * 3, hy - 22 + k * 7), (hx + 16 + k * 3, hy - 18 + k * 7), INK, 1)
    else:
        dt = t - BOOM_T
        cx, cy = hx + 2, hy - 12
        if dt < .16:                                               # 爆光：白芯 → 黄 → 橙，锯齿星形
            r = 10 + 46 * L.smooth(dt / .08) * (1 - L.smooth((dt - .08) / .08))
            for rr, col in ((r, (255, 150, 40)), (r * .72, (255, 230, 80)), (r * .4, PAPER)):
                pts = [(cx + np.cos(a) * rr * (1 if j % 2 else .55), cy + np.sin(a) * rr * (1 if j % 2 else .55))
                       for j, a in enumerate(np.linspace(0, 2 * np.pi, 16, endpoint=False))]
                cv2.fillPoly(img, [np.int32(pts)], col)
        for i in range(10):                                        # 碎片
            a = L.hash01(i, 31) * 2 * np.pi; v = 140 + 120 * L.hash01(i, 32)
            x, y = cx + np.cos(a) * v * dt, cy + np.sin(a) * v * dt + 260 * dt * dt
            if dt < .35: img[int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2] = (40, 44, 50) if i % 2 else (255, 200, 60)
        for i in range(5):                                         # 黑烟一团团往上飘
            a = dt - .05 * i
            if a <= 0: continue
            r = int(7 + 14 * min(a / .3, 1))
            x, y = int(cx + (L.hash01(i, 33) - .5) * 26), int(cy - 10 - 70 * a - i * 6)
            cv2.circle(img, (x, y), r + 2, INK, -1)
            cv2.circle(img, (x, y), r, (70, 66, 72), -1)
    img[264:266, 470:520] = SAND_DK                                # 沙面
    return zoom(img, 320, 180, 1 + .05 * L.smooth((t - 99.74) / .9))


# ================================================================
# 6a map / 6b panic
# ================================================================
# 10-19 用户：世界地图太简单 → 精细化成 RPG 世界地图：羊皮纸边框 + 海浪纹 + 三块大陆（沙滩海岸线、山脉、森林、河、雪顶）
# + 罗盘 + 每个去处一个小地标（森林小屋 / 断开的桥 / 火箭发射台 / 山里的地堡门）+ 虚线航路；
# 动起来的东西：回形针潮从左边一路漫过来吞掉大陆（锯齿前沿、往前爬），菜单里的去处一个个被划掉，光标跟着跳。
PLACES = [('OFFLINE', 112, 186), ('BUNKER', 262, 226), ('AIR GAP', 438, 92), ('MARS', 548, 214)]   # 按 x 排：回形针潮从左往右依次吞掉
PLACE_T = [100.64, 100.84, 101.10, 101.30]
SEA, SEA_HI, SEA_DK = (34, 70, 170), (70, 110, 210), (24, 50, 130)
GRASS, GRASS_DK, BEACH = (96, 176, 84), (60, 130, 64), (232, 214, 150)
MOUNT, MOUNT_DK, SNOW = (150, 120, 96), (104, 80, 66), (246, 241, 228)
PARCH, PARCH_DK = (226, 196, 140), (170, 130, 80)
MX0, MY0, MX1, MY1 = 10, 22, 630, 282          # 地图区（羊皮纸框里面）


@functools.lru_cache(maxsize=None)
def _land_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f = np.zeros((H, W), np.float32)
    blobs = [(130, 130, 90, 60), (200, 100, 60, 40), (80, 170, 50, 40),           # 西大陆
             (440, 90, 80, 36), (500, 120, 50, 30),                                 # 北岛
             (270, 228, 80, 34), (340, 214, 50, 28), (220, 246, 40, 24)]            # 南大陆
    for cx, cy, rx, ry in blobs:
        f += np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 1.6)
    for i in range(60):                                                             # 海岸线抖动
        cx, cy = L.hash01(i, 101) * W, L.hash01(i, 102) * H
        f += (L.hash01(i, 103) - .5) * .5 * np.exp(-(((xx - cx) / 18) ** 2 + ((yy - cy) / 14) ** 2))
    m = f > .55
    m[:MY0 + 4] = False; m[MY1 - 4:] = False; m[:, :MX0 + 4] = False; m[:, MX1 - 4:] = False
    return m


@functools.lru_cache(maxsize=None)
def _map_static():
    img = np.full((H, W, 3), SEA, np.uint8)
    for y in range(MY0, MY1, 12):                                                   # 海浪纹（错位短横）
        for x in range(MX0 + (y // 12 % 2) * 14, MX1, 28):
            img[y, x:x + 7] = SEA_HI; img[y + 1, x + 2:x + 5] = SEA_HI
    land = _land_mask()
    coast = cv2.dilate(land.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    img[coast & ~land] = SEA_HI                                                     # 浅滩
    img[land] = BEACH
    inner = cv2.erode(land.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    img[inner] = GRASS
    edge = cv2.dilate(land.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    img[edge & ~land] = INK
    core = cv2.erode(land.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0
    pts = np.argwhere(core)
    for i in range(46):                                                             # 山脉 / 森林按哈希撒在内陆
        y, x = pts[int(L.hash01(i, 111) * (len(pts) - 1))]
        if L.hash01(i, 112) < .45:
            poly(img, [(x - 10, y + 6), (x, y - 10), (x + 10, y + 6)], MOUNT, 1)
            poly(img, [(x, y - 10), (x + 10, y + 6), (x + 3, y + 6)], MOUNT_DK, 0)
            poly(img, [(x - 4, y - 4), (x, y - 10), (x + 4, y - 4)], SNOW, 0)
        else:
            for dx, dy in ((-5, 2), (5, 2), (0, -3)):
                cv2.circle(img, (int(x + dx), int(y + dy)), 5, INK, -1)
                cv2.circle(img, (int(x + dx), int(y + dy)), 4, GRASS_DK, -1)
    river = np.int32([(160, 110), (150, 140), (166, 160), (150, 190)])               # 河
    cv2.polylines(img, [river], False, SEA, 3)
    # 罗盘
    cx, cy = 590, 62
    cv2.circle(img, (cx, cy), 20, INK, -1); cv2.circle(img, (cx, cy), 18, PARCH, -1)
    poly(img, [(cx, cy - 16), (cx + 5, cy), (cx, cy + 16), (cx - 5, cy)], RED, 1)
    poly(img, [(cx - 16, cy), (cx, cy + 4), (cx + 16, cy), (cx, cy - 4)], INK, 0)
    img = text(img, cx - 4, cy - 40, 'N', PAPER)
    # 地标
    P_ = {n: (x, y) for n, x, y in PLACES}
    x, y = P_['OFFLINE']                                                            # 森林小屋
    poly(img, [(x - 10, y - 2), (x, y - 12), (x + 10, y - 2)], RED, 1); box(img, x - 8, y - 2, x + 8, y + 8, (150, 100, 60), 1)
    x, y = P_['AIR GAP']                                                            # 断开的桥（air gap）
    box(img, x - 18, y - 2, x - 4, y + 2, (150, 100, 60), 1); box(img, x + 4, y - 2, x + 18, y + 2, (150, 100, 60), 1)
    x, y = P_['MARS']                                                               # 火箭发射台
    box(img, x - 3, y - 18, x + 3, y + 2, PAPER, 1); poly(img, [(x - 3, y - 18), (x, y - 26), (x + 3, y - 18)], RED, 1)
    box(img, x - 10, y + 2, x + 10, y + 6, (110, 110, 130), 1)
    x, y = P_['BUNKER']                                                             # 地堡门
    cv2.ellipse(img, (x, y + 4), (12, 12), 0, 180, 360, INK, -1); cv2.ellipse(img, (x, y + 4), (10, 10), 0, 180, 360, (110, 110, 130), -1)
    img[y - 2:y + 4, x - 1:x + 1] = INK
    # 虚线航路（从西大陆出发连到每个去处）
    for (n0, x0, y0), (n1, x1, y1) in zip(PLACES, PLACES[1:]):
        for k in range(0, 20, 2):
            a, b = k / 20, (k + 1) / 20
            cv2.line(img, (int(x0 + (x1 - x0) * a), int(y0 + (y1 - y0) * a)), (int(x0 + (x1 - x0) * b), int(y0 + (y1 - y0) * b)), PAPER, 1)
    # 羊皮纸边框
    img[:MY0] = PARCH; img[MY1:] = PARCH; img[:, :MX0] = PARCH; img[:, MX1:] = PARCH
    cv2.rectangle(img, (MX0 - 1, MY0 - 1), (MX1, MY1), INK, 2)
    cv2.rectangle(img, (MX0 - 5, MY0 - 5), (MX1 + 4, MY1 + 4), PARCH_DK, 1)
    return img


def scene_map(t):
    img = _map_static().copy()
    # 回形针潮：从左往右漫过来，前沿锯齿、往前爬，陆地海洋一起吞
    front = float(np.interp(t, [100.56] + [a + .10 for a in PLACE_T], [MX0 - 30] + [x + 30 for _, x, _ in PLACES]))
    if t > PLACE_T[-1] + .10: front += (t - PLACE_T[-1] - .10) * 600
    yy = np.arange(H)
    jag = front + 18 * np.sin(yy / 9 + t * 20) + 10 * np.sin(yy / 23)
    tide = np.arange(W)[None, :] < jag[:, None]
    tide[:MY0] = False; tide[MY1:] = False; tide[:, :MX0] = False; tide[:, MX1:] = False
    img[tide] = (140, 144, 160)
    img[tide & (((np.arange(W)[None, :] + yy[:, None]) // 3) % 4 == 0)] = (100, 104, 120)
    frontline = tide & ~np.roll(tide, -3, 1)
    img[frontline] = PAPER
    for i in range(26):                                                             # 潮里的回形针
        x = MX0 + 10 + L.hash01(i, 121) * (min(front, MX1) - MX0 - 20)
        y = MY0 + 10 + L.hash01(i, 122) * (MY1 - MY0 - 20)
        if x < front - 20: clip_icon(img, x, y, .32, ang=L.hash01(i, 123) * 360 + t * 90)
    gone = [a for a in PLACE_T if t >= a + .12]
    for i, (name, x, y) in enumerate(PLACES):
        if len(gone) > i:
            cv2.line(img, (x - 18, y - 18), (x + 18, y + 18), INK, 7); cv2.line(img, (x + 18, y - 18), (x - 18, y + 18), INK, 7)
            cv2.line(img, (x - 18, y - 18), (x + 18, y + 18), RED, 4); cv2.line(img, (x + 18, y - 18), (x - 18, y + 18), RED, 4)
        img = text(img, x - 28, y + 12, name, INK); img = text(img, x - 29, y + 11, name, PAPER)
    # DQ 式菜单：去处一个个被划掉
    img = window(img, 22, 30, 150, 110, 'WHERE TO GO?')
    i = max([k for k, a in enumerate(PLACE_T) if t >= a] or [0])
    for k, (name, _, _) in enumerate(PLACES):
        y = 52 + k * 20
        img = text(img, 44, y, name, (120, 120, 140) if len(gone) > k else PAPER)
        if len(gone) > k: img[y + 8:y + 10, 42:42 + int(L.font(16).getlength(name)) + 4] = RED
        if k == i and f24(t) % 4 < 3: poly(img, [(30, y + 3), (38, y + 9), (30, y + 15)], PAPER, 0)   # ▶（DotGothic16 没有这个字形，手画）
    _, x, y = PLACES[i]
    if f24(t) % 4 < 3: poly(img, [(x - 46, y - 10), (x - 30, y), (x - 46, y + 10)], PAPER, 2)
    return img


PANIC_A, PANIC_B = '素材/CG/p3_6b_慌张_a.png', '素材/CG/p3_6b_慌张_b.png'
PANIC_KEYS = [(101.46, 'a'), (101.76, 'b'), (101.96, 'a'), (102.20, 'b')]


def scene_panic(t):
    k = key_at(t, PANIC_KEYS)
    img = (cg(PANIC_A) if k == 'a' else cg_aligned(PANIC_B, PANIC_A)).copy()
    dt = since(t, PANIC_KEYS)
    if dt < .12: img = np.roll(img, int(6 * np.sin(dt * 120)), axis=1)   # 甩头时横向一抖
    return img


# ================================================================
# 7 fuse_run
# ================================================================
LIGHTER = ['素材/CG/p3_7_打火机_a.png', '素材/CG/p3_7_打火机_b.png', '素材/CG/p3_7_打火机_c.png']
T_LIT, T_TOUCH = 103.32, 103.62


def embers(img, t, x, y, n, spread, seed):
    """火苗上方往上飘的火星（按时间直接算）。"""
    for i in range(n):
        u = (t * (.9 + .6 * L.hash01(i, seed)) + L.hash01(i, seed, 1)) % 1
        ex = x + (L.hash01(i, seed, 2) - .5) * spread + 6 * np.sin(u * 6 + i)
        ey = y - u * 90
        if (f12(t) + i) % 3: star4(img, ex, ey, 1 + (u < .3), YEL if i % 2 else (255, 150, 60))
    return img


def scene_lighter(t):
    """v2（用户：点火要有质感）：黑暗里掀开打火机 → lit 点着 → 火苗凑到引线上。"""
    t = float(np.clip(t, 102.48, 103.89 - 1e-3))
    lift_ = lambda im, g: np.clip(im.astype(np.float32) * g + 6, 0, 255).astype(np.uint8)   # 原图很暗，量化后会糊成一片，先提亮
    if t < T_LIT:
        img = lift_(cg(LIGHTER[0]), 1.55)
        return zoom(img, 250, 170, 1.04 + .06 * (t - 102.48) / (T_LIT - 102.48))
    if t < T_TOUCH:
        img = lift_(cg_aligned(LIGHTER[1], LIGHTER[0]), 1.25)
        if t - T_LIT < .08: img = dither_to(img, (255, 200, 120), .5 * (1 - (t - T_LIT) / .08))   # 点着那一下闪亮
        img = embers(img, t, 360, 110, 10, 20, 5)
        return zoom(img, 250, 170, 1.10)
    img = embers(lift_(cg(LIGHTER[2]), 1.25), t, 450, 270, 16, 40, 6)
    return zoom(img, 330, 215, 1.12)                                   # 往上推，别让对白框压住引线的火星


FUSE_Y = 210


def _glow(img, x, y, r, col, k=.5):
    m = np.zeros((H, W), np.uint8); cv2.circle(img if False else m, (int(x), int(y)), int(r), 1, -1)
    dither_to(img, col, k, m > 0)


def _fuse_zone(img, cam, t):
    """10-19 精细化：四个区段各有前中后景和会动的小东西（远景按 0.5 倍视差走）。
    机房（机柜斜面 + 闪灯 + 地板砖 + 线缆）→ 夜城（远近两排楼 + 路灯光晕 + 开过的车）
    → 海底（水色带 + 海草摆 + 鱼 + 气泡 + 海底光缆）→ 巨型 GPU（散热片 + 转的风扇 + 芯片字）。"""
    for X in range(int(cam * .5 // 40) * 40, int(cam * .5 + W + 40), 40):         # 远景层
        x = int(X - cam * .5); zone = int((X / .5) // 600)
        if zone == 1:
            hh = 70 + int(L.hash01(X, 7) * 90)
            img[190 - hh:190, x:x + 38] = (30, 24, 58)
            for wy in range(190 - hh + 6, 186, 10):
                if L.hash01(X, wy, 3) < .35: img[wy:wy + 3, x + 6:x + 10] = (150, 120, 70)
        elif zone == 2:
            for k in range(3):
                bx = x + int(L.hash01(X, k, 8) * 30); by = int(60 + L.hash01(X, k, 9) * 100 - (t * 60 + L.hash01(X, k) * 200) % 160)
                if by > 20: cv2.circle(img, (bx, by), 2, (120, 190, 240), 1)
    for X in range(int(cam // 60) * 60, int(cam + W + 60), 60):                     # 中景层
        x = int(X - cam); zone = max(X // 600, 0)                                 # 开头镜头在 0 左边：也算机房
        if zone == 0:
            img[196:, x:x + 60] = (26, 28, 46)
            img[196:, x:x + 1] = (40, 44, 66); img[230:232, x:x + 60] = (40, 44, 66)
            box(img, x + 4, 40, x + 54, 192, (34, 38, 72), 2)
            img[42:190, x + 4:x + 7] = (60, 66, 110); img[42:190, x + 51:x + 54] = (20, 22, 44)
            for k in range(9):
                img[48 + k * 16:58 + k * 16, x + 9:x + 49] = (24, 26, 50)
                img[48 + k * 16, x + 9:x + 49] = (50, 56, 96)
                on = L.hash01(X, k, f12(t) // 2) < .6
                img[51 + k * 16:54 + k * 16, x + 12:x + 16] = [(70, 225, 255), (80, 255, 120), (255, 180, 60)][(k + X // 60) % 3] if on else (30, 40, 60)
            cv2.polylines(img, [np.int32([(x, 30), (x + 30, 36), (x + 60, 30)])], False, (60, 30, 90), 3)   # 顶上垂的线缆
        elif zone == 1:
            img[196:, x:x + 60] = (34, 30, 44)
            for k in range(0, 60, 20): img[226:228, x + k:x + k + 10] = (200, 190, 120)      # 车道线
            hh = 60 + int(L.hash01(X, 1) * 100)
            box(img, x + 2, 190 - hh, x + 52, 196, (44, 34, 80), 2)
            img[190 - hh:196, x + 46:x + 52] = (30, 22, 56)
            for wy in range(190 - hh + 8, 186, 12):
                for wx in (x + 10, x + 24, x + 36):
                    if L.hash01(X, wy, wx) < .5: img[wy:wy + 5, wx:wx + 5] = YEL
            if X % 180 == 0:                                                            # 路灯 + 光晕
                cv2.line(img, (x + 56, 196), (x + 56, 140), (90, 90, 110), 3)
                _glow(img, x + 56, 142, 18, (255, 220, 120), .35); cv2.circle(img, (x + 56, 140), 4, PAPER, -1)
        elif zone == 2:
            img[:, x:x + 60] = (20, 50, 110)
            img[:40, x:x + 60] = (30, 70, 140); img[40:44, x:x + 60] = (26, 60, 126)
            img[FUSE_Y + 26:, x:x + 60] = (60, 50, 40)
            img[FUSE_Y + 26:FUSE_Y + 29, x:x + 60] = (90, 76, 56)
            for k in range(2):                                                          # 海草摆
                sx = x + 14 + k * 28; sw = 6 * np.sin(L.twos(t) * 3 + X * .1 + k)
                cv2.polylines(img, [np.int32([(sx, FUSE_Y + 28), (sx + sw, FUSE_Y), (sx - sw * .5, FUSE_Y - 26)])], False, (40, 140, 90), 4)
            if L.hash01(X, 3) < .35:                                                    # 鱼
                fx = x + 20 + int(8 * np.sin(t * 4 + X)); fy = int(70 + L.hash01(X, 4) * 80)
                cv2.ellipse(img, (fx, fy), (10, 5), 0, 0, 360, (255, 150, 60), -1)
                poly(img, [(fx - 9, fy), (fx - 16, fy - 5), (fx - 16, fy + 5)], (255, 150, 60), 0)
                img[fy - 1:fy + 1, fx + 5:fx + 7] = INK
        else:
            box(img, x, 30, x + 60, 200, (40, 150, 70), 0)
            for k in range(6): img[36 + k * 26:40 + k * 26, x:x + 60] = (26, 110, 52)       # PCB 走线
            for k in range(0, 60, 6): img[30:56, x + k:x + k + 3] = (170, 176, 190)            # 散热片
            img[30:56, x:x + 60][::6] = (120, 124, 140)
            cx, cy = x + 30, 120                                                              # 风扇（转）
            cv2.circle(img, (cx, cy), 24, INK, -1); cv2.circle(img, (cx, cy), 21, (30, 34, 40), -1)
            for k in range(5):
                a = t * 30 + k * 2 * np.pi / 5
                poly(img, [(cx, cy), (cx + np.cos(a) * 19, cy + np.sin(a) * 19), (cx + np.cos(a + .5) * 17, cy + np.sin(a + .5) * 17)], (90, 96, 110), 0)
            cv2.circle(img, (cx, cy), 5, (200, 40, 50), -1)
            if X % 120 == 0:
                box(img, x + 8, 160, x + 52, 190, (24, 24, 30), 2); img = text(img, x + 10, 166, 'H100', (190, 192, 204))
    return img


def scene_fuse_run(t):
    t = float(np.clip(t, 103.89, 105.94 - 1e-3))
    u = (t - 103.89) / 2.05
    wx = 2400 * u ** 1.4                                           # 火花的世界坐标，越跑越快
    cam = wx - 320
    img = np.full((H, W, 3), INK, np.uint8)
    img = _fuse_zone(img, cam, t)
    xs = np.arange(0, W, 2)
    ys = FUSE_Y + 10 * np.sin((xs + cam) / 40)
    ahead = xs >= 320
    pts_a = np.int32(np.stack([xs[ahead], ys[ahead]], 1)); pts_b = np.int32(np.stack([xs[~ahead], ys[~ahead]], 1))
    cv2.polylines(img, [pts_a], False, INK, 10)
    cv2.polylines(img, [pts_a], False, (170, 126, 80), 6)
    for x0 in range(int((-cam) % 10), W, 10):                      # 麻绳纹：一节一节的斜纹
        if x0 < 322: continue
        y0 = FUSE_Y + 10 * np.sin((x0 + cam) / 40)
        cv2.line(img, (x0 - 2, int(y0 + 3)), (x0 + 3, int(y0 - 3)), (110, 76, 46), 2)
    cv2.polylines(img, [pts_b], False, INK, 8)
    cv2.polylines(img, [pts_b], False, (80, 76, 80), 4)            # 烧过的灰
    for i in range(10):                                            # 烧过的地方往上冒烟
        age = (t * 3 + L.hash01(i, 131)) % 1
        sx = 320 - age * 160 * (1 + i % 3) * .5
        sy = FUSE_Y + 10 * np.sin((sx + cam) / 40) - age * 60
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(sx), int(sy)), int(4 + age * 10), 1, -1)
        dither_to(img, (130, 126, 140), .4 * (1 - age), m > 0)
    sy = FUSE_Y + 10 * np.sin(wx / 40)
    _glow(img, 320, sy, 26, (255, 160, 60), .35)
    _glow(img, 320, sy, 14, YEL, .6)
    for i in range(18):                                            # 火花 + 往后甩的火星
        a = L.hash01(i, f24(t)) * 2 * np.pi
        r = 8 + 20 * L.hash01(i, f24(t), 1)
        star4(img, 320 + np.cos(a) * r - 10, sy + np.sin(a) * r, 2, YEL if i % 2 else PAPER)
    for k in range(6):                                             # 速度线
        yy = int(40 + L.hash01(k, f24(t) // 2) * 240)
        img[yy:yy + 2, 0:int(120 + 200 * L.hash01(k, 5))] = PAPER
    star4(img, 320, sy, 7, PAPER)
    rem = max(0, 105.94 - t)
    box(img, 470, 26, 628, 50, INK, 2)
    zone = ['SERVER ROOM', 'CITY', 'DEEP SEA', 'GPU'][min(int(wx // 600), 3)]
    img = text(img, 478, 54, '> ' + zone, PAPER)
    return text(img, 478, 29, f'T-00:{rem:05.2f}', RED if f24(t) % 6 < 3 else YEL)


# ================================================================
# 8 bar / blues / face
# ================================================================
BAR_A, BAR_B, BAR_C = '素材/CG/p3_8a_蓝调酒吧_a.png', '素材/CG/p3_8a_蓝调酒吧_b.png', '素材/CG/p3_8a_蓝调酒吧_c.png'
# 嘴型：v1 按音节切了 9 次，太鬼畜。v2 只换三档，慢慢张开：半开 → 闭着哼 → thesis 起张大唱到底
MOUTH = [(105.94, 'b'), (106.40, 'a'), (106.88, 'c')]


def bar_cg(k):
    return cg(BAR_A) if k == 'a' else cg_aligned(BAR_B if k == 'b' else BAR_C, BAR_A)


def smoke(img, t, n=7):
    for i in range(n):
        x = int((L.hash01(i, 1) * W + t * (8 + 6 * L.hash01(i, 2))) % (W + 200) - 100)
        y = int(60 + L.hash01(i, 3) * 180 + 10 * np.sin(t + i))
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (x, y), (70, 18), 0, 0, 360, 1, -1)
        dither_to(img, (70, 80, 150), .22, m > 0)
    return img


# 10-19 用户：还是要原来酒吧里的吟唱，直交性改由左上角那块霓虹招牌来讲（用招牌本来的霓虹质感）：
# 招牌是竖轴 INTELLIGENCE、横轴 GOALS，原来中间只有一颗没意义的橙点 → 横轴上亮两个目标（粉爱心 / 青回形针），
# 橙色光点落在「回形针」那一列，每拍往上跳一格（智力一路涨），横向一点不动，身后拖着霓虹残影；
# 唱到 THESIS（106.88）原点亮起黄色的 ⊥。坐标按 bar CG（640 空间）量的：竖轴 x=78，横轴 y=169。
SIGN_X0, SIGN_Y0, SIGN_X1, SIGN_TOP = 78, 169, 180, 58
GOAL_HEART, GOAL_CLIP = 106, 148


def _neon_dot(img, x, y, r, col, glow=1.0):
    m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(x), int(y)), int(r * 2.4), 1, -1)
    dither_to(img, tuple(int(c * .55) for c in col), .45 * glow, m > 0)
    cv2.circle(img, (int(x), int(y)), r, col, -1)
    cv2.circle(img, (int(x), int(y)), max(1, r - 2), (255, 245, 220), -1)


def ortho_sign(img, t):
    wall = tuple(int(c) for c in img[100, 146])                        # 盖掉原来那颗橙点
    cv2.circle(img, (146, 116), 9, wall, -1)
    # 横轴上的两个目标：粉爱心 / 青回形针（小霓虹图标 + 刻度）
    for gx in (GOAL_HEART, GOAL_CLIP):
        img[SIGN_Y0 - 5:SIGN_Y0, gx] = PINK
    hx, hy = GOAL_HEART, SIGN_Y0 - 16
    for dx in (-4, 4): cv2.circle(img, (hx + dx, hy - 2), 4, PINK, 2)
    cv2.polylines(img, [np.int32([(hx - 8, hy), (hx, hy + 9), (hx + 8, hy)])], False, PINK, 2)
    clip_icon(img, GOAL_CLIP, SIGN_Y0 - 18, .4, col=CYAN)
    # 回形针那一列：淡淡的竖虚线（目标不变）
    for y in range(SIGN_TOP, SIGN_Y0 - 34, 6): img[y:y + 3, GOAL_CLIP] = (60, 90, 150)
    # 橙色光点每拍往上跳一格，留残影
    nb = int((t - 105.94) / BEAT)
    ph = (t - 105.94 - nb * BEAT) / BEAT
    step = (SIGN_Y0 - 40 - SIGN_TOP) / 3
    ypos = lambda k: SIGN_Y0 - 40 - min(k, 3) * step
    for k in range(min(nb, 3)):
        _neon_dot(img, GOAL_CLIP, ypos(k), 2, (200, 110, 60), .5)
    y = ypos(nb) - step * L.smooth(ph / .25) * (nb < 3)
    flick = E_ON(t)
    _neon_dot(img, GOAL_CLIP, y, 4, (255, 150, 60) if flick else (180, 90, 40), 1.0 if flick else .4)
    if t >= 106.88:                                                     # THESIS：原点亮起 ⊥
        on = E_ON(t + .3)
        img[:] = text(img, SIGN_X0 - 22, SIGN_Y0 - 14, '⊥', YEL if on else (150, 120, 40), 2)
    return img


def E_ON(t):
    """霓虹灯管偶尔闪一下（按帧号哈希，确定性）。"""
    return L.hash01(f24(t), 201) > .12


def scene_bar(t):
    t = float(np.clip(t, 105.94, 107.30 - 1e-3))
    img = zoom(ortho_sign(bar_cg(key_at(t, MOUTH)).copy(), t), 330, 180, 1 + .04 * (t - 105.94))
    return smoke(img, L.twos(t))


def scene_blues(t):
    img = np.full((H, W, 3), (18, 20, 50), np.uint8)
    for r, y in enumerate(range(0, H, 18)):                        # 砖墙
        img[y:y + 2] = (12, 12, 34)
        for x in range((r % 2) * 30, W, 60): img[y:y + 18, x:x + 2] = (12, 12, 34)
    poly(img, [(300, 0), (340, 0), (520, H), (120, H)], (30, 34, 80), 0)   # 顶光光柱
    for k, (x, y) in enumerate(((90, 90), (560, 100), (530, 230))):       # 霓虹音符、月牙
        c = [CYAN, PINK, YEL][k]
        if k < 2:
            cv2.ellipse(img, (x, y + 30), (10, 7), -20, 0, 360, c, 2)
            cv2.line(img, (x + 9, y + 28), (x + 9, y - 6), c, 2); cv2.line(img, (x + 9, y - 6), (x + 22, y), c, 2)
        else:
            cv2.ellipse(img, (x, y), (26, 26), 0, 60, 300, c, 2)
    return smoke(img, L.twos(t), 4)


@functools.lru_cache(maxsize=None)
def bar_person():
    """蓝调酒吧 CG c 的人物遮罩（640×360 bool；rembg isnet-anime，缓存在原图旁边的 .alpha.npz）。"""
    path = L.ROOT / BAR_C
    cache = path.with_suffix('.alpha.npz')
    if cache.exists():
        m = np.load(cache)['a']
    else:
        from PIL import Image
        from rembg import new_session, remove
        m = np.array(remove(Image.open(path).convert('RGB'), session=new_session('isnet-anime'), only_mask=True))
        np.savez_compressed(cache, a=m)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA) > 127


FACE_END = 109.88      # 11-12 用户：爆炸多留一会儿，唱到 just（109.88）再切；原来 109.35 就切去空电路板，中间空了半秒
FACE_CUT, FACE_BACK, FACE_FLASH = 109.12, 109.12, 109.12      # 10-19 用户：炸弹特写那一帧劣质 → 不切了（FACE_CUT=FACE_FLASH），引线烧到 109.12 就炸
BOMB_XY, BOMB_R = (238, 168), 21                              # 炸弹在 bar CG（640 空间）里的位置：她左肩后面，头发挡住大半、只露出盖子
FACE_ZOOM = lambda t: (318, 110, 2.1 + .12 * (t - 108.40))


def _bomb(img, x, y, r, dim=1.0):
    """黑色炸弹 + 金属盖 + 高光 + 「#3」。dim<1 时整颗压暗（叠在背景暗处）。"""
    c = lambda col: tuple(int(v * dim) for v in col)
    cv2.circle(img, (int(x), int(y)), r + 2, c(INK), -1)
    cv2.circle(img, (int(x), int(y)), r, c((40, 40, 56)), -1)
    cv2.circle(img, (int(x - r * .4), int(y - r * .4)), max(2, r // 4), c((110, 110, 140)), -1)
    box(img, x - r * .32, y - r - r * .32, x + r * .32, y - r + 2, c((70, 72, 88)), 2)          # 金属盖（深色）
    img[int(y - r - r * .32):int(y - r + 2), int(x - r * .32):int(x - r * .18)] = c((130, 134, 150))   # 盖子左边一条高光
    if r > 40: img[:] = text_c(img, int(x), int(y - 12), '#3', c(RED), 2)
    elif r > 12: img[:] = text_c(img, int(x), int(y - 4), '#3', c(RED))


def _fuse_pts(t0, t1, n=60):
    """背景里的引线：从画面左下边缘弯到炸弹盖（bar CG 640 空间）。"""
    s = np.linspace(t0, t1, n)
    bx, by = BOMB_XY
    x = 150 + (bx - 150) * s
    y = 205 - (205 - (by - BOMB_R - 8)) * s ** .7 + 6 * np.sin(s * 9)
    return np.stack([x, y], 1)


def scene_face(t):
    """10-19 v2（用户）：BLUES 之后引线烧到她背后，在背后爆炸 ——
    ① 108.40 特写：她左后方的暗处叠进一颗压暗的炸弹（人在前面挡着），引线的火花从左下沿着背景爬过去；
    ② 108.86 切一下炸弹特写：引线烧到头；③ 109.02 回到她的脸；
    ④ 109.12 爆炸：被闪光照到的白屏 + 她的黑剪影，背后放射光一圈圈涨，109.35 前全白接下一镜（鼓点的闪白）。"""
    t = float(np.clip(t, 108.40, FACE_END - 1e-3))
    base = bar_cg('c').copy()
    person = bar_person()
    if FACE_CUT <= t < FACE_BACK:                                   # ② 炸弹特写（暗处，只有引线的火光）
        img = np.full((H, W, 3), (14, 16, 40), np.uint8)
        for r, y in enumerate(range(0, H, 18)):                     # 同一面砖墙
            img[y:y + 2] = (10, 10, 30)
            for x in range((r % 2) * 30, W, 60): img[y:y + 18, x:x + 2] = (10, 10, 30)
        bx, by, R = 330, 196, 70
        k = (t - FACE_CUT) / (FACE_BACK - FACE_CUT)
        tip = (bx + 10, int(by - R - 34 + 30 * k))                  # 引线头烧到盖子里
        cv2.line(img, (bx + 60, by - R - 80), tip, (150, 110, 70), 6)
        m = np.zeros((H, W), np.uint8); cv2.circle(m, tip, 60, 1, -1)
        dither_to(img, (90, 50, 30), .5, m > 0)                     # 火光照亮一圈
        _bomb(img, bx, by, R)
        for i in range(16):
            a = L.hash01(i, f24(t)) * 2 * np.pi; rr = 6 + 22 * L.hash01(i, f24(t), 1)
            star4(img, tip[0] + np.cos(a) * rr, tip[1] + np.sin(a) * rr, 2, YEL if i % 2 else PAPER)
        star4(img, *tip, 8, PAPER)
        return img
    if t < FACE_FLASH:                                              # ①③ 特写：背景暗处的炸弹 + 爬过去的火花，人挡在前面
        bg = base.copy()
        k = L.smooth((t - 108.40) / (FACE_CUT - 108.40)) if t < FACE_CUT else 1.0
        pts = _fuse_pts(0, 1)
        n = int(k * len(pts))
        cv2.polylines(bg, [np.int32(pts[n:])], False, (110, 80, 56), 3)         # 还没烧的引线
        cv2.polylines(bg, [np.int32(pts[:max(n, 1)])], False, (30, 30, 40), 2)  # 烧过的灰
        _bomb(bg, *BOMB_XY, BOMB_R, dim=.5)
        if t < FACE_CUT:
            sx, sy = pts[min(n, len(pts) - 1)]
            m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(sx), int(sy)), 11, 1, -1)
            dither_to(bg, (230, 130, 50), .55, m > 0)
            for i in range(8):
                a = L.hash01(i, f24(t)) * 2 * np.pi
                star4(bg, sx + np.cos(a) * 6, sy + np.sin(a) * 6, 1, YEL)
            star4(bg, sx, sy, 4, PAPER)
        else:                                                       # 回到脸上时引线已经烧进炸弹，盖子那里冒火星
            for i in range(6):
                a = L.hash01(i, f24(t)) * 2 * np.pi
                star4(bg, BOMB_XY[0] + np.cos(a) * 8, BOMB_XY[1] - BOMB_R - 8 + np.sin(a) * 8, 1, YEL)
        bg[person] = base[person]                                   # 人挡在炸弹和引线前面
        img = zoom(bg, *FACE_ZOOM(t))
        n_b = int((t - 108.40) / BEAT)
        return dither_to(img, PAPER, min(.25, .06 * (n_b + 1)))
    # ④ 爆炸（10-19 v3 用户：人物脸上蒙上黑白阴影）：闪白一下 → 她还在画面里，被身后的闪光照成黑白两色的高反差
    #    （按亮度 + 离炸点远近二值化：朝炸点那边一片白、背光处一片黑，中间一档网点），身后用上一版的火焰：放射光 + 冲击环 + 碎片
    u = (t - FACE_FLASH) / (FACE_END - FACE_FLASH)
    if u < .06: return np.full((H, W, 3), PAPER, np.uint8)
    cx, cy, z = FACE_ZOOM(FACE_FLASH)
    face = zoom(base, cx, cy, z)
    sil = zoom(person.astype(np.uint8) * 255, cx, cy, z) > 127
    bx = (BOMB_XY[0] - cx) * z + W / 2; by = (BOMB_XY[1] - cy) * z + H / 2   # 炸点在画面里的位置（她左后方）
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - bx, yy - by)
    ang = np.arctan2(yy - by, xx - bx)
    # 身后：火焰（10-19 v4 用户：火焰做精致些）——
    #   底：深红 → 橙两档放射光（长短不一的光条，慢慢转）；火球：四层不规则轮廓（极坐标噪声，边缘一团团鼓起），
    #   白芯 → 浅黄 → 黄 → 橙，外圈一道焦褐烟边；冲击环用网点；火星粒子往外飞、越飞越暗；碎片带速度拖影
    img = np.full((H, W, 3), (150, 34, 30), np.uint8)
    R = 50 + 360 * u ** .8                                               # 火球别一下子铺满，留出外面的放射光和冲击环
    streak = np.sin(ang * 9 + u * 3) + .6 * np.sin(ang * 23 - u * 5)
    img[(streak > .5) & (d < R * 1.6)] = (220, 80, 40)
    img[(streak > 1.0) & (d < R * 1.9)] = (255, 140, 50)
    ring = (d > R * 1.25) & (d < R * 1.25 + 16)
    img[ring & L.bayer_mask(.6)] = (255, 220, 160)
    lobes = lambda k, ph: 1 + .10 * np.sin(ang * 5 + ph + k) + .07 * np.sin(ang * 11 - ph * 1.7) + .05 * np.sin(ang * 17 + k * 3)
    for scale, col, ph in ((1.08, (110, 50, 40), 0), (1.0, (240, 110, 40), .7), (.78, (255, 180, 60), 1.9),
                           (.55, (255, 225, 120), 3.1), (.32, (255, 250, 230), 4.0)):
        img[d < R * scale * lobes(scale, ph + u * 4)] = col
    for i in range(40):                                                  # 火星
        a = L.hash01(i, 151) * 2 * np.pi
        rr = R * (.6 + 1.2 * L.hash01(i, 152)) * (1 + .5 * u)
        ex, ey = bx + np.cos(a) * rr, by + np.sin(a) * rr
        c = (255, 240, 180) if L.hash01(i, 153) < .4 else (255, 170, 60)
        if 4 <= ex < W - 4 and 4 <= ey < H - 4:                           # 出界不画（star4 的负下标会绕到画面另一头）
            star4(img, ex, ey, 1 + (L.hash01(i, 154) < .3), c)
    for i in range(14):                                                  # 碎片 + 拖影
        a = L.hash01(i, 141) * 2 * np.pi; rr = 40 + 560 * u * (.5 + L.hash01(i, 142))
        px_, py_ = bx + np.cos(a) * rr, by + np.sin(a) * rr
        if not (-20 < px_ < W + 20 and -20 < py_ < H + 20): continue
        cv2.line(img, (int(px_ - np.cos(a) * 26), int(py_ - np.sin(a) * 26)), (int(px_), int(py_)), (255, 190, 90), 2)
        box(img, px_ - 4, py_ - 4, px_ + 4, py_ + 4, (40, 40, 56), 2)
    # 人（10-19 v5 用户：不要二值黑白，要被强光冲淡；参考图 = 碎成斜玻璃片 + 冷白褪色 + 速度线）：
    #   保留原画的线条和明暗，去饱和、往白里提，朝炸点那边提得更狠（接近过曝）
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV).astype(np.float32)
    near = np.clip(1 - d / 560, 0, 1)                                   # 离炸点越近越白
    hsv[..., 1] *= .45 * (1 - near)
    hsv[..., 2] = hsv[..., 2] + (255 - hsv[..., 2]) * (.25 + .5 * near)
    washed = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    img[sil] = washed[sil]
    # 整帧冲向暖黄白（用户：这里用黄光）：随时间越来越淡
    k = .14 + .16 * L.smooth(u / .8)                                    # 11-12 爆炸拉长后压低，不然后半段一路发白
    img = (img.astype(np.float32) * (1 - k) + np.array([255, 238, 170]) * k).astype(np.uint8)
    # 速度线：从炸点往外的方向（这里偏右下）拖出一条条白线
    for i in range(22):
        yy0 = int(L.hash01(i, 161) * H); ln = int(80 + 260 * L.hash01(i, 162))
        x0 = int(L.hash01(i, 163) * W + u * 600) % (W + 200) - 100
        th = 1 + (L.hash01(i, 164) < .3)
        cv2.line(img, (x0, yy0), (x0 + ln, yy0 + int(ln * .22)), (255, 250, 225), th)
    # 碎成斜的玻璃片：三条斜缝（白），缝越来越宽，每片往不同方向错开一点
    out = np.full_like(img, (255, 250, 228))
    cuts = [(80, 380, 330, -20), (300, 380, 640, 40)]               # 两条陡斜线把画面分成三片（x0,y0 → x1,y1），像参考图的玻璃碎片
    side = np.zeros((H, W), np.int32)
    for (x0_, y0_, x1_, y1_) in cuts:
        side += ((xx - x0_) * (y1_ - y0_) - (yy - y0_) * (x1_ - x0_) < 0).astype(np.int32)
    gap = 3 + 18 * L.smooth(u / .8)
    for sid, (dx, dy) in enumerate(((-10, -8), (6, 0), (18, 10))):
        piece = np.roll(img, (int(dy * u * 2), int(dx * u * 2)), (0, 1))
        m = side == sid
        for (x0_, y0_, x1_, y1_) in cuts:                               # 缝：离斜线 gap/2 以内留白
            dist = np.abs((xx - x0_) * (y1_ - y0_) - (yy - y0_) * (x1_ - x0_)) / np.hypot(x1_ - x0_, y1_ - y0_)
            m &= dist > gap / 2
        out[m] = piece[m]
        edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        out[edge] = (230, 170, 70)                                      # 碎片的暖橙描边
    img = np.roll(out, (int(3 * np.sin(u * 60)), int(4 * np.cos(u * 50))), (0, 1))   # 冲击震一下
    return dither_to(img, PAPER, L.smooth((t - (FACE_END - .06)) / .06))   # 只在最后两帧闪白（11-12 起下一镜是 JUST 砸下来，不用整屏白过渡）


SCENES = [
    ('gato', T_GATO, T_REACH, scene_gato),
    ('reach', T_REACH, T_GO, scene_reach),
    ('go', T_GO, T_IM, scene_go),
    ('led', T_IM, T_DROP, scene_led),
    ('counter', 97.80, 98.30, scene_counter),
    ('killswitch', 98.86, 99.74, scene_killswitch),
    ('beach', 99.74, 100.64, scene_beach),
    ('map', 100.64, 101.46, scene_map),
    ('panic', 101.46, 102.48, scene_panic),
    ('lighter', 102.48, 103.89, scene_lighter),
    ('fuse_run', 103.89, 105.94, scene_fuse_run),
    ('bar', 105.94, 107.30, scene_bar),
    ('blues', 107.30, 108.40, scene_blues),
    ('face', 108.40, FACE_END, scene_face),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
