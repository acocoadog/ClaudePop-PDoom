"""codeshots_h: 第四部分尾奏·花絮版（10-19 用户：ENCORE 后面不跳舞的嘉宾闪回改成花絮 / 幕后，ENCORE 也改）。
分镜见 docs/分镜/第四部分_124-156秒.md。节奏和原片一致：ENCORE 3 拍 + 嘉宾一拍一个。

    audition   140.27–141.63  偶像选拔投稿录像：夕阳天台常服那条片子 + 4:3 黑边 + 摄像机 REC 界面 + 手写标签，最后一拍盖「書類審査 通過」
    guest      141.63–145.72  九个嘉宾的幕后，一拍一个，整段套 VHS 回放界面 + 场记板：
                              大嘴盖「已弃用」章 / 中文房间「刚刚学会！」小箭头 / 修格斯化妆间 / 蛇怪是 12cm 的玩具 / Gato 啃小鱼干 /
                              回形针租赁收据 + 纸箱 / Killswitch 二人的空转椅 + 明信片 / Claude 其实怕火 / 龙猫上秤减肥（精灵表四帧）
    v2（10-19 用户：场景精细化）：灯泡、纸箱楼、榻榻米、转椅等全部重画；怕火 / 龙猫精灵表是生图

画面里的字一律日语（10-19 用户：「要用日语」）。字体用 Zpix 像素字（素材/字体/zpix.ttf，原生 12px，假名和日文汉字都有）。
接口同 part1_lib：scene(t) 纯函数、画全彩。自检：python codeshots_h.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import codeshots_e as E
import codeshots_g as G
import part1_lib as L

W, H = L.W, L.H
B = L.BEAT132
INK, PAPER, YEL, RED, CYAN, BLUE, PINK = E.INK, E.PAPER, E.YEL, E.RED, E.CYAN, E.BLUE, E.PINK
CLAWD, CLAWD_SH = E.CLAWD, E.CLAWD_SH
CG = '素材/CG/'
poly = E.poly
ZPIX = L.ROOT / '素材/字体/zpix.ttf'


# ================================================================
# 小工具：中文像素字 / VHS 界面 / 场记板 / Clawd / 印章
# ================================================================
@functools.lru_cache(maxsize=None)
def zfont():
    return ImageFont.truetype(str(ZPIX), 12)


def zw(s, scale=1):
    return int(zfont().getlength(s)) * scale


def zh(img, x, y, s, col, scale=1):
    """Zpix 12px 像素字（关抗锯齿，scale 倍最近邻放大），直接画在 img 上。"""
    f = zfont()
    l, t_, r, b = f.getbbox(s)
    m = Image.new('1', (r - l + 2, b - t_ + 2), 0)
    ImageDraw.Draw(m).text((1 - l, 1 - t_), s, fill=1, font=f)
    if scale > 1: m = m.resize((m.width * scale, m.height * scale), Image.NEAREST)
    ys, xs = np.nonzero(np.array(m))
    ys, xs = ys + int(y), xs + int(x)
    ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
    img[ys[ok], xs[ok]] = col
    return img


def zh_c(img, cx, y, s, col, scale=1):
    return zh(img, cx - zw(s, scale) // 2, y, s, col, scale)


def rotate_paste(img, patch, cx, cy, ang, mask=None):
    """把一块小图转 ang 度（最近邻）贴到 (cx,cy) 为中心的位置。"""
    h, w = patch.shape[:2]
    d = int(np.hypot(w, h)) + 4
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[:, 2] += (d - w) / 2, (d - h) / 2
    rot = cv2.warpAffine(patch, M, (d, d), flags=cv2.INTER_NEAREST)
    m = cv2.warpAffine(np.ones((h, w), np.uint8) if mask is None else mask.astype(np.uint8), M, (d, d), flags=cv2.INTER_NEAREST) > 0
    x0, y0 = int(cx - d / 2), int(cy - d / 2)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x1, y1 = min(W, x0 + d), min(H, y0 + d)
    if x1 <= max(x0, 0) or y1 <= max(y0, 0): return img
    sub = img[max(y0, 0):y1, max(x0, 0):x1]
    mm = m[sy0:sy0 + sub.shape[0], sx0:sx0 + sub.shape[1]]
    sub[mm] = rot[sy0:sy0 + sub.shape[0], sx0:sx0 + sub.shape[1]][mm]
    return img


def clawd(img, cx, feet, s=1.0, look=0, glasses=False, arm_l=None, arm_r=None):
    """通用 Clawd：橙色方块身 + 两条竖眼 + 四条腿。arm_l / arm_r = 手伸到的点（画一截方块胳膊）；look 眼睛往左右看几个像素。"""
    S = lambda v: int(round(v * s))
    bw, bh = S(64), S(40)
    x0, x1, y1 = cx - bw // 2, cx + bw // 2, feet - S(12)
    y0 = y1 - bh
    for lx in (x0 + S(6), x0 + S(20), x1 - S(26), x1 - S(12)): E.box(img, lx, y1, lx + S(6), feet, CLAWD_SH, max(1, S(2)))
    for arm, side in ((arm_l, x0), (arm_r, x1)):
        if arm is not None:
            cv2.line(img, (side, y0 + bh // 2), arm, INK, S(12))
            cv2.line(img, (side, y0 + bh // 2), arm, CLAWD, S(8))
    E.box(img, x0, y0, x1, y1, CLAWD, max(1, S(3)))
    for ex in (x0 + S(18), x1 - S(24)):
        E.box(img, ex + S(look), y0 + S(12), ex + S(look) + S(6), y0 + S(26), INK, 0)
    if glasses:
        for ex in (x0 + S(14), x1 - S(28)):
            cv2.circle(img, (ex + S(7) + S(look), y0 + S(19)), S(10), INK, max(1, S(2)))
        cv2.line(img, (x0 + S(31), y0 + S(19)), (x1 - S(31), y0 + S(19)), INK, max(1, S(2)))
    return img


def otext(img, x, y, s, col):
    """带一圈黑描边的像素字：录像界面的白字压在浅色画面上也看得清。"""
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
        img = E.text(img, x + dx, y + dy, s, INK)
    return E.text(img, x, y, s, col)


def vhs(img, t, mode='PLAY', clock=None, x0=0, x1=W):
    """家用录像机界面：左上 ▶PLAY / ●REC（PLAY 时下一行写 clock），REC 时右下写日期时间；隔一阵来一条跟踪噪点横纹。
    x0 / x1 = 画面的左右边（4:3 黑边时界面画在画面里面）。"""
    if mode == 'REC':
        if E.f12(t) % 12 < 8: cv2.circle(img, (x0 + 22, 40), 6, RED, -1)
        img = otext(img, x0 + 34, 31, 'REC', PAPER)
        E.box(img, x1 - 52, 32, x1 - 16, 46, (30, 30, 30), 2)          # 电量格（最后一格闪）
        for k in range(3 if E.f12(t) % 8 < 4 else 2):
            img[35:43, x1 - 48 + k * 11:x1 - 40 + k * 11] = PAPER
        if clock:
            img = otext(img, x1 - 14 - int(L.font(16).getlength(clock)), 270, clock, PAPER)
    else:
        cv2.fillPoly(img, [np.int32([(x0 + 18, 32), (x0 + 18, 46), (x0 + 30, 39)])], PAPER)
        cv2.polylines(img, [np.int32([(x0 + 18, 32), (x0 + 18, 46), (x0 + 30, 39)])], True, INK, 1)
        img = otext(img, x0 + 36, 30, 'PLAY', PAPER)
        if clock: img = otext(img, x0 + 18, 50, clock, PAPER)
    ph = (t * 1.7) % 1.0                                               # 跟踪噪点：一条横纹从下往上扫
    if ph < .12:
        y = int(H - ph / .12 * H)
        band = slice(max(y - 5, 0), min(y + 5, H))
        img[band] = np.roll(img[band], 7, axis=1)
        for i in range(30):
            x = int(L.hash01(E.f24(t), i) * W)
            img[band.start:band.stop, x:x + 3] = PAPER
    return img


def clapper(img, x, y, dt, scene, take):
    """场记板：一拍开头从打开的斜角啪地合上（0.07 秒），板身写 SCENE / TAKE。"""
    w, h = 132, 62
    E.box(img, x, y + 14, x + w, y + 14 + h, (24, 24, 30), 2)
    cv2.line(img, (x + 4, y + 38), (x + w - 4, y + 38), PAPER, 1)
    img = E.text(img, x + 6, y + 18, f'SCENE {scene:02d}', PAPER)
    img = E.text(img, x + 6, y + 42, f'TAKE {take}', PAPER)
    ang = 28 * max(0.0, 1 - dt / .07)                                  # 顶上的斜纹拍子
    bar = np.full((14, w, 3), PAPER, np.uint8)
    for k in range(0, w + 14, 22):
        cv2.fillPoly(bar, [np.int32([(k, 0), (k + 11, 0), (k - 3, 14), (k - 14, 14)])], INK)
    cv2.rectangle(bar, (0, 0), (w - 1, 13), INK, 1)
    M = cv2.getRotationMatrix2D((0, 14), ang, 1.0)
    pad = np.zeros((80, w + 20, 3), np.uint8); msk = np.zeros((80, w + 20), np.uint8)
    pad[66:80, 0:w] = bar; msk[66:80, 0:w] = 1
    M = cv2.getRotationMatrix2D((0, 80), ang, 1.0)
    rb = cv2.warpAffine(pad, M, (w + 20, 80), flags=cv2.INTER_NEAREST)
    rm = cv2.warpAffine(msk, M, (w + 20, 80), flags=cv2.INTER_NEAREST) > 0
    ys, xs = np.nonzero(rm)
    yy, xx = ys + y - 66, xs + x
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    img[yy[ok], xx[ok]] = rb[ys[ok], xs[ok]]
    return img


def stamp(img, cx, cy, lines, dt, ang=-12, col=RED, pad=10):
    """红色印章：双线框 + 几行字，啪地砸下（0.08 秒从 1.8 倍缩到原大，落地后抖一下）。lines = [(字, 倍数)]。"""
    ws = [zw(s, k) for s, k in lines]
    hs = [13 * k for s, k in lines]
    bw, bh = max(ws) + pad * 2, sum(hs) + pad * 2 + 4 * (len(lines) - 1)
    patch = np.zeros((bh, bw, 3), np.uint8); m = np.zeros((bh, bw), bool)
    cv2.rectangle(patch, (0, 0), (bw - 1, bh - 1), col, 3); cv2.rectangle(patch, (5, 5), (bw - 6, bh - 6), col, 1)
    y = pad
    for (s, k), hh in zip(lines, hs):
        zh_c(patch, bw // 2, y, s, col, k); y += hh + 4
    m = cv2.dilate(patch.any(2).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0   # 字加粗一圈，压在花哨的原画上也看得清
    patch[m] = col
    for i in range(160):                                               # 印泥没沾匀：随机掉几块
        px, py = int(L.hash01(i, 5) * bw), int(L.hash01(i, 6) * bh)
        m[py:py + 2, px:px + 2] = False
    z = 1 + .8 * max(0.0, 1 - dt / .08)
    if z > 1.01:
        patch = cv2.resize(patch, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        m = cv2.resize(m.astype(np.uint8), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST) > 0
    return rotate_paste(img, patch, cx, cy + E.bounce(dt - .08, 2), ang, m)


def paper_note(w, h, col=(250, 246, 232)):
    note = np.full((h, w, 3), col, np.uint8)
    cv2.rectangle(note, (0, 0), (w - 1, h - 1), INK, 2)
    return note


# ================================================================
# E audition：偶像选拔投稿录像
# ================================================================
AUD_A, AUD_B = 140.27, 141.63
AUD_CLIP0 = 138.67                                                     # 和原 encore 一样：片段 1.60s 的重拍对上 140.27
AUD_LABEL, AUD_STAMP = 140.95, 141.18                                  # 「特典映像!!」大字收掉以后贴标签；最后一拍盖章


@functools.lru_cache(maxsize=None)
def aud_frame(idx):
    cap = cv2.VideoCapture(str(L.ROOT / '镜头/shot_chorus1_fast720.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    assert ok, idx
    return cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def scene_audition(t):
    src = aud_frame(int(round((t - AUD_CLIP0) * 24)))
    img = np.zeros((H, W, 3), np.uint8)
    x0 = (W - 480) // 2
    img[:, x0:x0 + 480] = src[:, 40:520]                              # 4:3，偏左一点让她在框里
    # （10-19 去掉了原来的偏暖色：用户觉得整段发黄，录像感交给 REC 界面和跟踪噪点）
    img = vhs(img, t, 'REC', f'AUG.8 1995 PM5:{42 + int(t - AUD_A):02d}', x0, x0 + 480)
    if t >= AUD_LABEL:                                                 # 手写标签：白色胶带贴在画面左下
        note = paper_note(250, 44)
        zh(note, 8, 5, '応募No.0008　クロード', INK)
        zh(note, 8, 23, '特技：P(DOOM)を上げること', INK)
        img = rotate_paste(img, note, x0 + 136, 300 + E.bounce(t - AUD_LABEL, 2), 3)
    if t >= AUD_STAMP:
        img = stamp(img, x0 + 104, 128, [('書類審査', 3), ('通過', 4)], t - AUD_STAMP, ang=-14)
    return img                                                         # 结尾直接切花絮（11-12 试过朝指尖推镜冲白，用户：太赶）


# ================================================================
# guest：九个嘉宾的幕后
# ================================================================
G0 = G.G0


# 11-12 用户：THE MOUTH 改成「Claude 眩晕瘫坐，电脑里放核爆」（看到核爆瘫在椅子上的梗）。
# 图 = gpt 在空房间（同一间卧室，怪物去掉）里加上瘫坐的 Claude，再改一张屏幕里是核爆的；屏幕遮罩由两张图的差值算出来。


@functools.lru_cache(maxsize=None)
def mouth_layers():
    """屏幕遮罩直接从图里算（11-12：之前手量四角贴歪、白闪有黑边）：
    原图（屏幕黑）和核爆图逐像素比，屏幕框里原来很暗、现在变了的像素 = 屏幕玻璃。
    屏幕外变暖的部分（外壳、桌面、她的脸上的光）= 核爆照出来的光。"""
    off = E.cg(CG + '花絮_嘴_眩晕.png').astype(np.int16)
    on = E.cg(CG + '花絮_嘴_屏幕核爆_像素.png').astype(np.int16)
    diff = np.abs(on - off).sum(-1)
    box_ = np.zeros((H, W), bool); box_[60:260, 470:600] = True
    scr = box_ & (off.sum(-1) < 200) & (diff > 40)
    scr = cv2.morphologyEx(scr.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(scr)
    scr = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])                 # 只留最大的一块
    cnts, _ = cv2.findContours(scr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full = np.zeros((H, W), np.uint8); cv2.drawContours(full, cnts, -1, 1, -1)   # 填掉里面的洞
    scr = full.astype(bool)
    light = np.clip((on - off).astype(np.float32), 0, None); light[scr] = 0      # 核爆照出来的光
    ys = np.nonzero(scr.any(1))[0]
    return off.astype(np.uint8), on.astype(np.uint8), scr, light, ys.min(), ys.max()


def g_mouth(t, dt):
    """THE MOUTH → 第一次见识 GPT-5 的 Claude，看核爆看到瘫在椅子上（11-12 用户）。
    一拍 0.45 秒都是全景：从 1.06 倍缓缓推到 1.0、中心从显示器那边慢慢平移回正中，
    加上显示器放出的冲击波扭曲、头顶晕眩星星和拟声大字。
    （试过：蘑菇云逐行长 / 冲击波 + ドーン / 整屏闪白切入 / 屏幕特写切全景 / 特写拉远 + 指尖推镜冲白，用户：镜头太短，都不自然）"""
    off, on, scr, light, y0, y1 = mouth_layers()
    ys, xs = np.nonzero(scr)
    mon = np.array([xs.mean(), ys.mean()])
    u = L.smooth(dt / B)
    z = 1.06 - .06 * u
    c = np.array([W / 2 + 14 * (1 - u), H / 2 - 4 * (1 - u)])
    m = np.float32([[z, 0, W / 2 - z * c[0]], [0, z, H / 2 - z * c[1]]])
    img = cv2.warpAffine(on, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 冲击波扭曲：从显示器放出两圈折射环，经过处画面往外推一下（用户 11-12）
    mx, my = (mon - c) * z + (W / 2, H / 2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - mx, yy - my
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    off = np.zeros((H, W), np.float32)
    for lag in (0, .12):
        v = (dt - lag) / .32
        if 0 < v < 1:
            R = 30 + 620 * (1 - (1 - v) ** 2)                            # 先快后慢扩出去
            band = 22 + 20 * v
            x = (d - R) / band
            off += 16 * (1 - v) * np.exp(-x * x) * np.sin(x * 3)           # 环内往里吸、环外往外推，像折射
    if off.any():
        img = cv2.remap(img, (xx - dx / d * off).astype(np.float32), (yy - dy / d * off).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 晕眩小星星：头顶一圈椭圆轨道上转（演出，用户 11-12）
    hx, hy = (np.array(MOUTH_HEAD) - c) * z + (W / 2, H / 2)
    for i in range(3):
        a_ = dt * 9 + i * 2 * np.pi / 3
        sx, sy = hx + 46 * np.cos(a_), hy + 12 * np.sin(a_)
        r = 14 if np.sin(a_) > 0 else 10                                 # 转到前面大一点
        pts = [(sx + (r if j % 2 == 0 else r * .45) * np.cos(-np.pi / 2 + j * np.pi / 5 + a_ * .5),
                sy + (r if j % 2 == 0 else r * .45) * np.sin(-np.pi / 2 + j * np.pi / 5 + a_ * .5)) for j in range(10)]
        cv2.fillPoly(img, [np.int32(pts)], YEL)                          # 五角星，自转
        cv2.polylines(img, [np.int32(pts)], True, INK, 2)
    if dt > .06:                                                         # 中上方空墙：漫画拟声「ぽか〜ん…」，弹出来后跟着镜头慢慢漂（用户：中间上面太空）
        tx = 'ぽか〜ん…'
        sc = 4
        x = 300 - zw(tx, sc) // 2 + int(6 * u)
        y = 70 + E.bounce(dt - .06, 4)
        for ox, oy in ((-3, 0), (3, 0), (0, -3), (0, 3), (3, 3), (4, 4)): zh(img, x + ox, y + oy, tx, INK, sc)
        zh(img, x, y, tx, PAPER, sc)
    return img


MOUTH_HEAD = (100, 116)                                                  # 头顶位置（640 坐标），晕眩星星绕着它转


# ---------------- 像素场景小工具（v2 精细化） ----------------
def bevel(img, x0, y0, x1, y1, base, hi, lo, ink=2, w=2):
    """有体积的方块：墨线框 + 底色 + 上 / 左高光、下 / 右暗边。"""
    E.box(img, x0, y0, x1, y1, base, ink)
    img[y0:y0 + w, x0:x1] = hi; img[y0:y1, x0:x0 + w] = hi
    img[y1 - w:y1, x0:x1] = lo; img[y0:y1, x1 - w:x1] = lo
    return img


def glow(img, cx, cy, r, col, k):
    """Bayer 网点光晕（圆）。"""
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(cx), int(cy)), int(r), 1, -1)
    return E.dither_to(img, col, k, m.astype(bool))


def vgrad(img, y0, y1, c0, c1, steps=6, x0=0, x1=W):
    """竖向分档渐变（硬边色带 + 色带交界处一行网点过渡）。"""
    for i in range(steps):
        a, b = y0 + (y1 - y0) * i // steps, y0 + (y1 - y0) * (i + 1) // steps
        c = tuple(int(c0[j] + (c1[j] - c0[j]) * i / (steps - 1)) for j in range(3))
        img[a:b, x0:x1] = c
    return img


def arrow_hand(img, pts, t0, t, col=RED, w=4):
    """手绘小箭头：沿折线一笔描出来（0.12 秒），描完冒出箭头尖。pts 是几个控制点，中间自动抖一点。"""
    pts = np.float64(pts)
    seg = []
    for a, b in zip(pts[:-1], pts[1:]):
        for u in np.linspace(0, 1, 10, endpoint=False):
            p = a + (b - a) * u
            seg.append(p + (np.sin(u * 9 + a[0]) * 1.5, np.cos(u * 7 + a[1]) * 1.5))
    seg.append(pts[-1]); seg = np.int32(seg)
    k = int(len(seg) * min(max((t - t0) / .12, 0), 1))
    if k < 2: return img
    cv2.polylines(img, [seg[:k]], False, INK, w + 4); cv2.polylines(img, [seg[:k]], False, col, w)
    if k == len(seg):
        tip, d = seg[-1], seg[-1] - seg[-4]
        d = d / (np.hypot(*d) + 1e-6)
        n = np.array([-d[1], d[0]])
        head = np.int32([tip + d * 4, tip - d * 16 + n * 11, tip - d * 16 - n * 11])
        cv2.fillPoly(img, [head], col); cv2.polylines(img, [head], True, INK, 2)
    return img


# ================================================================
# 各个嘉宾（v2，10-19 用户：场景精细化）
# ================================================================
GREET = ['Hello', 'Bonjour', 'Hola', 'Hallo', 'Ciao', 'Olá', 'Привет', '你好', 'こんにちは', 'Γειά', 'Merhaba', 'Xin chào',
         'Hej', 'Cześć', 'Szia', 'Ahoj', 'Jambo', 'Kumusta']        # 18 国（Zpix 有字形的；韩 / 阿 / 泰 / 印地 / 希伯来没有字形，没放）
GREET_XY = [(360, 58), (470, 104), (352, 150), (560, 168), (430, 200), (80, 36), (330, 100), (520, 226), (30, 230),
            (400, 236), (100, 196), (250, 250), (20, 150), (330, 212), (440, 150), (150, 240), (570, 252), (40, 96)]
# ↑ 避开右上场记板（x>490, y<90）和底下名牌（y>290）；倍数字（下标 %3==0）留够高度


def g_room(t, dt):
    """11-12 用户：中文房间改成「Claude 其实精通十八国语言」→ 同一张举着「你好」卡的图，十八个问候气泡一个接一个弹出来，顶上黄字「実は18か国語ペラペラ」，名牌「ニーハオだけ…は演技」。
    （试过的旧版：「さっき覚えた！」+ 箭头 + 尴尬汗珠）"""
    img = E.cg(CG + 's7_中文房间_v1_开心.png').copy()
    for i, (w, (x, y)) in enumerate(zip(GREET, GREET_XY)):
        a = .02 + i * .016
        if dt < a: continue
        sc = 1 + (i % 3 == 0)
        bw, bh = zw(w, sc) + 10, 12 * sc + 8
        pop = E.bounce(dt - a, 2)
        x0, y0 = x, y + pop
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + bw + 1, y0 + bh + 1), INK, -1)
        cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + bw - 1, y0 + bh - 1), PAPER, -1)
        tail = np.int32([(x0 + 8, y0 + bh), (x0 + 16, y0 + bh), (x0 + 6, y0 + bh + 7)])
        cv2.fillPoly(img, [tail], PAPER); cv2.polylines(img, [tail], False, INK, 1)
        zh(img, x0 + 5, y0 + 4, w, INK if i % 4 else RED, sc)
    # 综艺字幕：最上方中间（灯泡和场记板之间）一行黄字，左端手绘小箭头一笔描下来指着她（用户：字别挡脸、箭头画回去）
    img = arrow_hand(img, [(200, 30), (182, 38), (178, 54), (194, 70)], .04, dt)       # 字宽 256，204–460 正好停在场记板（x≈490）左边
    if dt > .14:
        y = 20 + E.bounce(dt - .14, 2)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (2, 2)): zh(img, 204 + dx, y + dy, '実は18か国語ペラペラ', INK, 2)
        zh(img, 204, y, '実は18か国語ペラペラ', YEL, 2)
    return img


@functools.lru_cache(maxsize=None)
def shoggoth_true():
    """修格斯真身 s8b（640 尺寸）+ rembg 遮罩（缓存在图旁边的 .alpha.npz），压暗一点当前景。"""
    img = E.cg(CG + 's8b_修格斯真身.png')
    m = cv2.resize(G.fb_person('s8b_修格斯真身'), (W, H), interpolation=cv2.INTER_LINEAR)
    dark = (img.astype(np.float32) * np.float32([.82, .8, .88])).astype(np.uint8)
    return dark, m


@functools.lru_cache(maxsize=None)
def shoggoth_half():
    """修格斯面具 CG 的上半身（笑脸 + 罩布上半截），镜子里用。"""
    src = E.cg(CG + 's8a_修格斯面具.png')
    return cv2.resize(src[40:290, 125:525], (300, 188), interpolation=cv2.INTER_AREA)


def g_shoggoth(t, dt):
    img = np.zeros((H, W, 3), np.uint8)
    vgrad(img, 22, 200, (70, 44, 64), (50, 30, 48), 5)                # 墙纸（上亮下暗）
    for y in range(30, 196, 18):                                       # 菱形墙纸花纹
        for x in range((y // 18) % 2 * 18, W, 36):
            cv2.fillPoly(img, [np.int32([(x, y - 4), (x + 4, y), (x, y + 4), (x - 4, y)])], (92, 60, 84))
    img[196:252] = (86, 52, 40)                                        # 护墙板
    for x in range(0, W, 46):
        bevel(img, x + 3, 202, x + 43, 248, (98, 60, 46), (124, 80, 60), (64, 38, 30), 1)
    mx0, my0, mx1, my1 = 170, 64, 470, 252                             # 镜面
    bevel(img, mx0 - 18, my0 - 18, mx1 + 18, my1 + 2, (176, 132, 70), (230, 190, 110), (120, 84, 40), 2, 3)   # 金色镜框
    cv2.rectangle(img, (mx0 - 6, my0 - 6), (mx1 + 5, my1 + 1), (120, 84, 40), 2)
    img[my0:my1, mx0:mx1] = shoggoth_half()
    for k in range(3):                                                 # 镜面反光：两道斜白条（网点）
        m = np.zeros((H, W), np.uint8)
        x = mx0 + 40 + k * 26
        cv2.fillPoly(m, [np.int32([(x, my0), (x + 14 - k * 4, my0), (x - 60 - k * 4, my1), (x - 74, my1)])], 1)
        m[:, :mx0] = 0; m[:, mx1:] = 0; m[:my0] = 0; m[my1:] = 0
        img = E.dither_to(img, PAPER, .22 - k * .06, m.astype(bool))
    bulbs = [(mx0 - 9 + i * (mx1 - mx0 + 18) // 6, my0 - 9) for i in range(7)] + \
            [(mx0 - 9, my0 + 30 + i * 44) for i in range(4)] + [(mx1 + 9, my0 + 30 + i * 44) for i in range(4)]
    dead = 7                                                           # 有一颗灯泡坏了，一闪一闪
    for i, (bx, by) in enumerate(bulbs):
        lit = i != dead or E.f24(t) % 5 < 2
        if lit: img = glow(img, bx, by, 15, (255, 230, 160), .3)
    for i, (bx, by) in enumerate(bulbs):
        lit = i != dead or E.f24(t) % 5 < 2
        E.box(img, bx - 4, by - 2, bx + 4, by + 3, (90, 90, 96), 1)    # 灯座
        cv2.circle(img, (bx, by), 8, INK, -1)
        cv2.circle(img, (bx, by), 6, (255, 246, 200) if lit else (130, 120, 96), -1)
        cv2.circle(img, (bx, by + 1), 6, (255, 214, 120) if lit else (100, 92, 74), 1)
        img[by - 3:by - 1, bx - 3:bx - 1] = PAPER                     # 高光一点
    # 化妆台
    bevel(img, 110, 252, 530, 264, (150, 96, 60), (196, 140, 90), (100, 62, 40), 2)
    img[264:296, 120:520] = (110, 70, 46)
    for x in (126, 520 - 6): img[264:296, x:x + 6] = (80, 50, 34)
    E.box(img, 150, 226, 166, 252, (230, 230, 240), 2)                 # 笔筒 + 刷子
    for k, c in enumerate([(240, 120, 150), (120, 80, 50), (80, 70, 160)]):
        cv2.line(img, (154 + k * 4, 226), (150 + k * 6, 206), INK, 4); cv2.line(img, (154 + k * 4, 226), (150 + k * 6, 206), c, 2)
    E.box(img, 188, 238, 198, 252, (200, 30, 60), 1); img[232:238, 189:197] = (240, 80, 110)   # 口红
    cv2.ellipse(img, (236, 248), (16, 5), 0, 0, 360, INK, -1); cv2.ellipse(img, (236, 246), (14, 4), 0, 0, 360, (250, 210, 220), -1)  # 粉扑
    plate = np.full((26, 140, 3), (210, 170, 80), np.uint8)            # 黄铜名牌
    cv2.rectangle(plate, (0, 0), (139, 25), INK, 2); cv2.rectangle(plate, (3, 3), (136, 22), (240, 210, 120), 1)
    zh_c(plate, 70, 7, '★ ショゴス 様', INK)
    img[24:50, 250:390] = plate
    # 镜头前：真身凑在镜子前（右下角露出一大块，镜子里照出来的却是可爱的笑脸罩布）
    fg, fm = shoggoth_true()
    z = 1.9 * (1 + .015 * np.sin(t * 7))                               # 慢慢起伏，像在呼吸
    M = np.float64([[z, 0, 372 - 10 * z], [0, z, 300 - 232 * z]])      # 原图左缘 (10,232) 对到画面 (372,300)：往左凑，露出更多眼睛和触手
    fgw = cv2.warpAffine(fg, M, (W, H), flags=cv2.INTER_LINEAR)
    fmw = cv2.warpAffine(fm, M, (W, H), flags=cv2.INTER_LINEAR) > 127
    img[fmw] = fgw[fmw]
    edge = fmw & ~cv2.erode(fmw.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    img[edge] = INK
    return img


@functools.lru_cache(maxsize=None)
def basilisk_toy(h):
    im = L.load_rgba(CG + 'p2_basilisk.png')[..., :3].astype(np.int16)
    g = (im[..., 1] > 170) & (im[..., 0] < 120) & (im[..., 2] < 120)
    im[..., 1] = np.where(~g & (im[..., 1] > np.maximum(im[..., 0], im[..., 2]) + 20), np.maximum(im[..., 0], im[..., 2]), im[..., 1])
    a = cv2.erode(np.where(g, 0, 255).astype(np.uint8), np.ones((5, 5), np.uint8))
    rgba = np.dstack([im.astype(np.uint8), a])
    return cv2.resize(rgba, (int(h * rgba.shape[1] / rgba.shape[0]), h), interpolation=cv2.INTER_AREA)


def cardboard(img, x0, y0, x1, y1, seed, lit=.25, shade=1.0):
    """纸箱楼：瓦楞纸色 + 侧面暗边 + 马克笔窗户（部分亮灯）+ 胶带。"""
    base = tuple(int(c * shade) for c in (198, 150, 96))
    bevel(img, x0, y0, x1, y1, base, tuple(int(c * shade) for c in (222, 178, 120)), tuple(int(c * shade) for c in (150, 108, 64)), 2)
    img[y0 + 2:y1, x1 - 8:x1 - 2] = tuple(int(c * shade) for c in (168, 124, 78))   # 侧面
    for wy in range(y0 + 8, y1 - 10, 14):
        for wx in range(x0 + 6, x1 - 14, 11):
            on = L.hash01(wx, wy, seed) < lit
            img[wy:wy + 7, wx:wx + 6] = YEL if on else tuple(int(c * shade) for c in (70, 50, 34))
            if on: img[wy:wy + 2, wx:wx + 2] = PAPER
    img[y0 + (y1 - y0) // 3:y0 + (y1 - y0) // 3 + 5, x0 - 1:x1 + 1] = tuple(int(c * shade) for c in (226, 210, 160))   # 胶带
    return img


def g_basilisk(t, dt):
    img = np.zeros((H, W, 3), np.uint8)
    vgrad(img, 22, 230, (34, 44, 70), (64, 80, 110), 6)                # 摄影棚背景纸（无缝）
    img = glow(img, 250, 120, 170, (96, 112, 150), .35)                # 布景上的打光
    # 左边柔光箱 + 灯架
    cv2.line(img, (46, 120), (46, 296), INK, 6); cv2.line(img, (46, 120), (46, 296), (110, 110, 120), 2)
    for dx in (-26, 26): cv2.line(img, (46, 280), (46 + dx, 296), (110, 110, 120), 3)
    poly(img, [(14, 40), (96, 30), (104, 118), (10, 112)], PAPER, 2)
    poly(img, [(22, 48), (88, 40), (94, 108), (18, 104)], (255, 250, 230), 0)
    img = glow(img, 56, 76, 60, (255, 250, 220), .18)
    # 桌子
    img[230:296] = (128, 84, 52)
    bevel(img, 0, 226, W, 238, (160, 108, 66), (196, 140, 90), (110, 72, 44), 2)
    for y in range(246, 296, 9):
        for x in range(int(L.hash01(y, 3) * 40), W, 90): img[y:y + 1, x:x + 50] = (108, 70, 44)
    # 后排纸箱楼（暗一点）
    for i, (bx, bw_, bh_) in enumerate([(70, 48, 120), (150, 40, 150), (300, 52, 134), (380, 44, 110)]):
        cardboard(img, bx, 228 - bh_, bx + bw_, 228, i + 10, .2, .7)
    img = L.paste_rgba(img, basilisk_toy(170), 150, 30)                # 蛇怪立在楼群后面，只露出上半身
    # 前排纸箱楼（挡住蛇怪下半身）
    for i, (bx, bw_, bh_) in enumerate([(56, 60, 82), (118, 46, 104), (166, 70, 66), (238, 44, 90), (284, 64, 74), (350, 58, 96)]):
        cardboard(img, bx, 230 - bh_, bx + bw_, 230, i)
    for tx in (108, 232, 342):                                         # 纸团做的小树
        cv2.line(img, (tx, 230), (tx, 214), (100, 70, 40), 3)
        cv2.circle(img, (tx, 210), 8, INK, -1); cv2.circle(img, (tx, 210), 6, (90, 170, 80), -1); img[206:208, tx - 3:tx - 1] = (150, 220, 120)
    top = 30 + int(np.nonzero((basilisk_toy(170)[..., 3] > 0).any(1))[0].min())   # 蛇怪最高点（冠刺尖）
    E.box(img, 418, top, 430, 232, (244, 220, 110), 1)                 # 尺子：桌面一直量到蛇怪最高点 = 12cm
    cv2.line(img, (300, top), (430, top), PAPER, 1, cv2.LINE_8)        # 顶端一条对齐虚线
    for x in range(300, 418, 6): img[top:top + 1, x:x + 3] = INK
    for k in range(13):
        y = int(round(230 - k * (230 - top) / 12))
        img[y:y + 1, 418:424 + (5 if k % 5 == 0 else 0)] = INK
        if k % 5 == 0 and 0 < k < 12: img = E.text(img, 432, y - 8, str(k), PAPER)
    img = E.text(img, 434, top - 8, '12cm', YEL)
    # 巨大 Clawd 扛着摄像机
    bob = 2 if E.f12(t) % 4 < 2 else 0
    clawd(img, 600, 330 + bob, 3.4, look=-6)
    cx0, cy0 = 452, 140 + bob
    bevel(img, cx0, cy0, cx0 + 92, cy0 + 50, (46, 46, 54), (80, 80, 92), (24, 24, 28), 3)   # 机身
    E.box(img, cx0 + 20, cy0 - 14, cx0 + 74, cy0 - 4, (30, 30, 36), 2)  # 提手
    E.box(img, cx0 + 60, cy0 - 26, cx0 + 86, cy0 - 12, (30, 30, 36), 2)  # 取景器
    E.box(img, cx0 - 30, cy0 + 8, cx0, cy0 + 42, (34, 34, 40), 2)      # 镜头筒
    cv2.circle(img, (cx0 - 32, cy0 + 25), 20, INK, -1); cv2.circle(img, (cx0 - 32, cy0 + 25), 15, (60, 80, 120), -1)
    cv2.circle(img, (cx0 - 32, cy0 + 25), 8, (30, 40, 70), -1); img[cy0 + 16:cy0 + 20, cx0 - 38:cx0 - 34] = PAPER
    if E.f12(t) % 12 < 8:
        cv2.circle(img, (cx0 + 82, cy0 + 10), 4, RED, -1)
    img = E.text(img, cx0 + 8, cy0 + 16, 'Hi8', (200, 200, 210))
    cv2.line(img, (cx0 + 92, cy0 + 40), (560, 230), INK, 18); cv2.line(img, (cx0 + 92, cy0 + 40), (560, 230), CLAWD, 13)
    return img


def dried_fish(img, x, y, s=1.0, ang=0):
    """小鱼干（煮干し）：金褐色鱼身 + 深色背 + 银白肚 + 叉尾 + 一只眼。(x,y) = 鱼身中心。"""
    S = lambda v: v * s
    body = np.float64([(-22, 0), (-14, -6), (4, -7), (16, -4), (22, 0), (16, 4), (4, 6), (-14, 5)])
    tail = np.float64([(-20, 0), (-32, -8), (-28, 0), (-32, 8)])
    r = np.deg2rad(ang); R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    f = lambda p: np.int32(p * s @ R.T + (x, y))
    cv2.fillPoly(img, [f(tail)], (150, 112, 60)); cv2.polylines(img, [f(tail)], True, INK, 2)
    cv2.fillPoly(img, [f(body)], (204, 160, 86)); cv2.polylines(img, [f(body)], True, INK, 2)
    back = np.float64([(-14, -6), (4, -7), (16, -4), (12, -2), (-12, -3)])
    cv2.fillPoly(img, [f(back)], (130, 96, 54))
    belly = np.float64([(-10, 3), (6, 4), (14, 2), (6, 2)])
    cv2.fillPoly(img, [f(belly)], (236, 226, 200))
    e = f(np.float64([[13, -1]]))[0]
    cv2.circle(img, tuple(e), max(1, int(S(2))), INK, -1)
    g = f(np.float64([(9, -4), (8, 3)]))
    cv2.line(img, tuple(g[0]), tuple(g[1]), (110, 80, 44), 1)
    return img


def g_gato(t, dt):
    img = np.zeros((H, W, 3), np.uint8)
    vgrad(img, 22, 190, (92, 74, 100), (70, 54, 80), 5)                # 楽屋的墙
    img[150:190] = (96, 64, 44)                                        # 腰板
    for x in range(0, W, 40): img[150:190, x:x + 1] = (72, 46, 32)
    img[150:153] = (130, 90, 60)
    vgrad(img, 190, 296, (176, 170, 110), (140, 134, 84), 5)          # 榻榻米
    for y, x0 in ((190, 0), (240, 0)):
        img[y:y + 3] = (60, 90, 60)                                    # 榻榻米包边
    for x in (120, 360): cv2.line(img, (x, 190), (x - 40, 296), (60, 90, 60), 3)
    for y in range(196, 296, 5): img[y:y + 1] = tuple(int(c * .92) for c in (170, 164, 104))
    # 墙上：楽屋名牌 + 挂轴
    sign = np.full((30, 130, 3), (236, 226, 200), np.uint8)
    cv2.rectangle(sign, (0, 0), (129, 29), INK, 2); zh_c(sign, 65, 8, '楽屋　GATO 様', INK)
    img[40:70, 255:385] = sign
    E.box(img, 64, 36, 112, 140, (230, 222, 196), 2)                   # 挂轴「鰹」
    img[32:38, 58:118] = (110, 70, 40); img[138:144, 58:118] = (110, 70, 40)
    zh_c(img, 88, 60, '鰹', INK, 3)
    # 坐垫（座布団）：四角流苏 + 中间缝线
    cx, cy = 300, 252
    poly(img, [(cx - 84, cy - 12), (cx + 84, cy - 12), (cx + 96, cy + 16), (cx - 96, cy + 16)], (150, 60, 110), 2)
    poly(img, [(cx - 84, cy - 12), (cx + 84, cy - 12), (cx + 88, cy - 4), (cx - 88, cy - 4)], (190, 90, 140), 0)
    cv2.circle(img, (cx, cy + 2), 3, YEL, -1)
    for px, py in ((cx - 84, cy - 12), (cx + 84, cy - 12), (cx + 96, cy + 16), (cx - 96, cy + 16)):
        cv2.line(img, (px, py), (px + (6 if px > cx else -6), py + 6), YEL, 2)
    chew = E.f12(t) % 4 < 2
    bob = 1 if chew else 0
    E.cat(img, cx, cy + bob, 2.2, 'sit', tail=E.f12(t) * .6, col=(16, 12, 24), rim=(90, 80, 130))
    hx, hy = cx + 9, cy - 101 + bob
    img = dried_fish(img, hx + 30, hy + 10, 1.5, -8 if chew else -2)   # 叼在嘴里的小鱼干
    for i in range(3):                                                 # 嚼下来的碎屑
        if chew: img[hy + 22 + i * 6:hy + 24 + i * 6, hx + 18 + i * 5:hx + 20 + i * 5] = (204, 160, 86)
    # 小碟子里还有两条
    cv2.ellipse(img, (470, 266), (44, 13), 0, 0, 360, INK, -1); cv2.ellipse(img, (470, 264), (40, 10), 0, 0, 360, PAPER, -1)
    cv2.ellipse(img, (470, 264), (28, 6), 0, 0, 360, (200, 220, 240), 1)
    img = dried_fish(img, 460, 260, 1.0, 10); img = dried_fish(img, 480, 266, 1.0, -6)
    # 茶杯冒热气
    E.box(img, 150, 246, 172, 270, (230, 236, 210), 2); img[246:250, 150:172] = (120, 160, 90)
    for k in range(2):
        yy = 236 - int((t * 30 + k * 10) % 20)
        img[yy:yy + 4, 156 + k * 8:158 + k * 8] = PAPER
    if E.f12(t) % 6 < 3:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)): img = E.text(img, hx + 50 + dx, hy - 40 + dy, 'nom nom', INK)
        img = E.text(img, hx + 50, hy - 40, 'nom nom', PAPER)
    return img


def g_clips(t, dt):
    img = np.full((H, W, 3), (200, 204, 216), np.uint8)
    vgrad(img, 22, 210, (210, 214, 226), (180, 184, 200), 5)          # 办公室的墙
    bevel(img, 380, 36, 470, 120, (140, 190, 230), (200, 230, 250), (90, 130, 170), 3)   # 窗 + 百叶
    for y in range(40, 118, 7): img[y:y + 2, 383:467] = (230, 236, 244)
    vgrad(img, 210, 296, (120, 110, 100), (96, 88, 80), 4)            # 地毯
    for i in range(30):                                                # 天花板还在漏回形针
        sp = 160 + 90 * L.hash01(i, 1)
        x = L.hash01(i, 2) * W
        y = (L.hash01(i, 3) * H + t * sp) % (H + 60) - 30
        E.clip_icon(img, x, y, .45, ang=(i * 37 + E.f12(t) * 25) % 360)
    # 收银机（米色、按键、液晶）
    bevel(img, 90, 222, 300, 286, (222, 214, 190), (246, 240, 222), (170, 160, 136), 3)
    poly(img, [(110, 222), (280, 222), (264, 196), (126, 196)], (206, 198, 174), 2)
    E.box(img, 196, 202, 266, 216, (40, 60, 40), 2)
    img = E.text(img, 202, 200, '¥ ∞', (140, 255, 120))
    for r in range(2):
        for c in range(6):
            E.box(img, 110 + c * 16, 236 + r * 16, 122 + c * 16, 246 + r * 16, (120, 120, 130) if (r + c) % 3 else (230, 120, 90), 1)
    E.box(img, 214, 240, 286, 276, (200, 190, 166), 2)                 # 钱箱
    # 收据从打印口往上吐
    k = L.smooth(min(dt / .2, 1))
    rh = int(170 * k)
    x0, x1 = 130, 250
    if rh > 4:
        rec = paper_note(x1 - x0, 170)
        zh_c(rec, 60, 8, '— RECEIPT —', INK)
        zh(rec, 8, 30, 'クリップ', INK); zh(rec, 70, 30, '×10^12', INK)
        zh(rec, 8, 48, 'レンタル料', INK)
        zh_c(rec, 60, 70, '¥∞', RED, 3)
        cv2.line(rec, (8, 112), (112, 112), INK, 1)
        zh(rec, 8, 118, '※月曜までに返却', INK)
        zh(rec, 8, 136, '延滞1日ごと×2', RED)
        img[196 - rh:196, x0:x1] = rec[:rh]
    # 回形针纸箱（盖子掀开、冒出来、贴单子）
    bx0, by0, bx1, by1 = 400, 196, 540, 288
    poly(img, [(bx0, by0), (bx0 - 26, by0 - 22), (bx0 + 50, by0 - 22), (bx0 + 64, by0)], (176, 128, 80), 2)   # 左盖
    poly(img, [(bx1, by0), (bx1 + 22, by0 - 26), (bx1 - 40, by0 - 26), (bx1 - 60, by0)], (186, 138, 88), 2)  # 右盖
    for i in range(26):                                                # 箱口堆成小山
        E.clip_icon(img, bx0 + 12 + L.hash01(i, 7) * 116, by0 - 4 - L.hash01(i, 8) * 16, .42, ang=i * 53)
    bevel(img, bx0, by0, bx1, by1, (198, 150, 96), (222, 178, 120), (150, 108, 64), 2)
    img[by0 + 2:by1, bx1 - 30:bx1 - 2] = (170, 126, 80)               # 侧面
    img[by0 + 28:by0 + 34, bx0:bx1] = (226, 210, 160)                  # 胶带
    lab = paper_note(70, 40)
    zh(lab, 5, 4, 'CLIPS', INK); zh(lab, 5, 22, 'RENTAL', RED)
    img[by0 + 40:by0 + 80, bx0 + 12:bx0 + 82] = lab
    cv2.rectangle(img, (bx0 + 88, by0 + 44), (bx0 + 100, by0 + 58), INK, 1)   # 「↑」天地无用
    cv2.line(img, (bx0 + 94, by0 + 56), (bx0 + 94, by0 + 46), INK, 1)
    # Clawd 场工：扫帚 + 簸箕，把地上的回形针往箱子那边扫
    sw = int(12 * np.sin(t * 18))
    clawd(img, 330, 288, 1.1, look=4, arm_r=(378 + sw, 262))
    cv2.line(img, (378 + sw, 262), (392 + sw * 2, 288), (130, 86, 44), 4)
    poly(img, [(380 + sw * 2, 286), (406 + sw * 2, 286), (410 + sw * 2, 292), (376 + sw * 2, 292)], (220, 190, 100), 1)
    for i in range(8): E.clip_icon(img, 396 + L.hash01(i, 1) * 30 + sw, 286 + L.hash01(i, 2) * 5, .4, ang=i * 40)
    return img


def office_chair(img, cx, base_y, col=(46, 60, 110), name=None):
    """办公转椅（正面略侧）：靠背 + 坐垫 + 扶手 + 气压杆 + 五爪脚轮。"""
    hi, lo = tuple(min(255, c + 34) for c in col), tuple(int(c * .65) for c in col)
    for ang in (-160, -110, -70, -20, 90):                             # 五爪
        a = np.deg2rad(ang)
        ex, ey = int(cx + 46 * np.cos(a)), int(base_y + 10 * np.sin(a))
        cv2.line(img, (cx, base_y - 2), (ex, ey), INK, 7); cv2.line(img, (cx, base_y - 2), (ex, ey), (70, 70, 78), 3)
        cv2.circle(img, (ex, ey + 4), 5, INK, -1); cv2.circle(img, (ex, ey + 4), 3, (40, 40, 44), -1)
    E.box(img, cx - 4, base_y - 52, cx + 4, base_y - 2, (150, 150, 160), 2)   # 气压杆
    bevel(img, cx - 34, base_y - 128, cx + 34, base_y - 66, col, hi, lo, 2, 3)  # 靠背
    cv2.rectangle(img, (cx - 26, base_y - 120), (cx + 25, base_y - 76), lo, 1)
    poly(img, [(cx - 46, base_y - 64), (cx + 46, base_y - 64), (cx + 40, base_y - 50), (cx - 40, base_y - 50)], col, 2)   # 坐垫
    img[base_y - 64:base_y - 61, cx - 44:cx + 44] = hi
    for s in (-1, 1):                                                  # 扶手
        x = cx + s * 46
        E.box(img, min(x, x + s * 6), base_y - 90, max(x, x + s * 6), base_y - 58, (60, 60, 66), 1)
        E.box(img, min(x - s * 8, x + s * 10), base_y - 94, max(x - s * 8, x + s * 10), base_y - 88, (60, 60, 66), 1)
    if name:
        tag = paper_note(60, 18); tag = E.text(tag, 5, 0, name, INK)
        img[base_y - 112:base_y - 94, cx - 30:cx + 30] = tag
    return img


def g_killswitch(t, dt):
    img = np.zeros((H, W, 3), np.uint8)
    vgrad(img, 22, 212, (176, 184, 196), (150, 158, 172), 5)          # 公司办公室
    bevel(img, 24, 40, 214, 150, (120, 170, 220), (180, 220, 245), (80, 120, 160), 3)   # 窗外城市
    for i, (bx, bh_) in enumerate([(30, 50), (62, 76), (96, 40), (130, 88), (170, 60)]):
        img[150 - bh_:147, bx:bx + 28] = (90, 110, 150)
        for wy in range(150 - bh_ + 6, 144, 9):
            for wx in range(bx + 4, bx + 24, 7): img[wy:wy + 3, wx:wx + 3] = (170, 200, 230)
    for y in range(44, 148, 8): img[y:y + 1, 27:211] = (210, 226, 240)  # 百叶
    # 墙上：KILLSWITCH 红按钮（玻璃罩 + 便利贴 ON PTO）
    bevel(img, 270, 46, 360, 134, (90, 94, 116), (120, 124, 146), (60, 62, 80), 3)
    cv2.ellipse(img, (315, 100), (26, 14), 0, 0, 360, INK, -1); cv2.ellipse(img, (315, 96), (23, 11), 0, 180, 360, RED, -1)
    cv2.ellipse(img, (315, 98), (23, 11), 0, 0, 180, (150, 20, 30), -1)
    cv2.ellipse(img, (315, 92), (32, 26), 0, 180, 360, CYAN, 1)
    img = E.text(img, 276, 48, 'KILLSWITCH', YEL)
    note = paper_note(54, 34, (255, 236, 110)); note = E.text(note, 4, 0, 'ON', INK); note = E.text(note, 4, 15, 'PTO', RED)
    img = rotate_paste(img, note, 352, 120, 10)
    E.box(img, 410, 60, 610, 150, (240, 242, 246), 3)                  # 白板
    img = E.text(img, 420, 64, 'OFFSITE: HAWAII', BLUE)
    img = E.text(img, 420, 86, 'BACK: MONDAY', BLUE)
    img = E.text(img, 420, 108, '(maybe)', RED)
    img[212:296] = (110, 114, 126)                                     # 地毯
    for x in range(0, W, 24): img[212:296, x:x + 1] = (100, 104, 116)
    bevel(img, 80, 182, 560, 196, (190, 170, 140), (220, 200, 170), (140, 122, 96), 2)   # 长桌
    for lx in (96, 540): img[196:250, lx:lx + 8] = (90, 90, 98)
    for mx, txt in ((150, 'OOO'), (470, 'OOO')):                       # 显示器：Out Of Office
        bevel(img, mx - 36, 132, mx + 36, 176, (40, 44, 56), (70, 74, 90), (24, 26, 34), 2)
        img[136:172, mx - 32:mx + 32] = (30, 60, 140)
        img = E.text(img, mx - 12, 144, txt, PAPER)
        img[176:182, mx - 6:mx + 6] = (60, 60, 66)
    E.box(img, 214, 170, 226, 182, PAPER, 1); img[166:168, 216:224] = (180, 180, 180)   # 凉掉的咖啡
    office_chair(img, 230, 288, name='GUY A')
    office_chair(img, 410, 288, (110, 50, 60), name='GUY B')
    pc = paper_note(104, 66)                                           # 明信片（缩小版海滩），放在 B 的坐垫上
    pc[4:36, 4:100] = cv2.resize(E.scene_beach(99.9), (96, 32), interpolation=cv2.INTER_AREA)
    zh(pc, 4, 38, 'アロハ〜！', INK)
    zh(pc, 4, 51, '— まだ有給中', BLUE)
    return rotate_paste(img, pc, 432, 238 + E.bounce(dt, 3), 14)


def neon(img, cx, cy, text, t, col=(255, 60, 90)):
    """霓虹灯牌：暗处的网点光晕 → 灯管（字 + 圆角框）→ 白芯；偶尔有一个字接触不良。cx, cy = 灯牌中心。"""
    scale = 2
    tw, th = zw(text, scale), 13 * scale
    x0, y0 = cx - tw // 2, cy - th // 2
    tube = np.zeros((H, W, 3), np.uint8)
    flick = int(L.hash01(E.f24(t) // 3, 7) * 10)                      # 第 flick 个字偶尔暗一下
    for i, ch in enumerate(text):
        if not (i == flick % len(text) and E.f24(t) % 7 < 2):
            zh(tube, x0 + zw(text[:i], scale), y0, ch, (255, 255, 255), scale)
    cv2.rectangle(tube, (x0 - 12, y0 - 10), (x0 + tw + 10, y0 + th + 8), (255, 255, 255), 2)
    m = tube.any(2).astype(np.uint8)
    halo = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    img = E.dither_to(img, tuple(int(c * .45) for c in col), .35, halo)
    img = E.dither_to(img, tuple(int(c * .75) for c in col), .5, near)
    thick = cv2.dilate(m, np.ones((2, 2), np.uint8)) > 0
    img[thick] = col
    core = cv2.erode(thick.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    img[core & (m > 0)] = (255, 210, 220)
    return img


def g_lighter(t, dt):
    """Claude 娘怕火：生图 p4_h8_怕火_v2（10-19 用户：嘴改小、表情别太夸张；Clawd 拿掉）+ 「火気厳禁」；整张画面跟着她抖。"""
    img = E.cg(CG + 'p4_h8_怕火_v2.png').copy()
    img = np.roll(img, 1 if E.f24(t) % 2 else -1, axis=1)
    return neon(img, 410, 54, '火気厳禁', t)


CHIN = '素材/精灵/龙猫_减肥四姿势.png'


@functools.lru_cache(maxsize=None)
def chin_frames(h=150):
    """龙猫精灵表切帧。生图给的是带半透明边的透明底：
    ① 先预乘透明度再缩放（直接缩会把透明处的杂色混进边缘，描出一圈黄 / 红边）；
    ② 按宽度四等分（胡子会把相邻两只连起来，按连通块切不开），每格只留重心落在本格里的连通块（不带进隔壁的尾巴 / 胡子）；
    ③ 统一缩放（按整张表的身高），透明度最后再二值化，保持像素硬边。只用前三帧。"""
    im = L.load_rgba(CHIN).astype(np.float32)
    al = im[..., 3:4] / 255
    pre = np.dstack([im[..., :3] * al, im[..., 3]])
    solid = (im[..., 3] > 40).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(solid, 8)
    ys = np.nonzero(solid.any(1))[0]
    sc = h / (ys.max() + 1 - ys.min())
    qw = im.shape[1] // 4
    out = []
    for k in range(3):
        keep = np.isin(lab, [i for i in range(1, n) if k * qw <= cent[i][0] < (k + 1) * qw and stats[i][4] > 20])
        yy, xx = np.nonzero(keep)
        y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        f = pre[y0:y1, x0:x1].copy()
        f[~keep[y0:y1, x0:x1]] = 0
        size = (max(1, int((x1 - x0) * sc)), max(1, int((y1 - y0) * sc)))
        f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)
        a_ = f[..., 3:4] / 255
        rgb = np.where(a_ > 0, f[..., :3] / np.maximum(a_, 1e-3), 0)
        out.append(np.dstack([np.clip(rgb, 0, 255), np.where(f[..., 3] > 110, 255, 0)]).astype(np.uint8))
    return out


def g_chinchilla(t, dt):
    img = np.zeros((H, W, 3), np.uint8)
    for x in range(0, W, 40):                                          # 紫色幕布：每一褶三档明暗
        img[22:240, x:x + 14] = (88, 58, 134); img[22:240, x + 14:x + 28] = (72, 46, 112); img[22:240, x + 28:x + 40] = (56, 34, 90)
        img[22:240, x + 4:x + 6] = (120, 86, 170)
    img[22:34] = (150, 30, 50)                                         # 幕布顶的帷幔
    for x in range(0, W, 32): cv2.ellipse(img, (x + 16, 34), (16, 8), 0, 0, 180, (150, 30, 50), -1)
    vgrad(img, 240, 296, (176, 124, 74), (140, 96, 56), 4)            # 舞台木地板（透视）
    for x in range(-200, W + 200, 60): cv2.line(img, (320 + (x - 320) // 3, 240), (x, 296), (120, 82, 48), 1)
    # 墙上的身高尺 + 横幅
    E.box(img, 120, 90, 136, 240, (240, 236, 220), 2)
    for k in range(15): img[236 - k * 10:237 - k * 10, 120:126 + (4 if k % 5 == 0 else 0)] = INK
    ban = paper_note(200, 26, (255, 236, 110)); zh_c(ban, 100, 7, 'Post-Chinchilla 減量中！', RED)
    img = rotate_paste(img, ban, 330, 60, -3)
    for i, (bx, by) in enumerate([(470, 212), (520, 212), (494, 182)]):   # DATA 干草堆
        bevel(img, bx, by, bx + 52, by + 30, (210, 170, 70), (240, 210, 110), (160, 120, 40), 2)
        for k in range(6): img[by + 4 + k * 4:by + 5 + k * 4, bx + 3:bx + 49] = (180, 140, 50)
        img[by:by + 30, bx + 14:bx + 16] = (120, 80, 40); img[by:by + 30, bx + 36:bx + 38] = (120, 80, 40)
        img = E.text(img, bx + 8, by + 6, 'DATA', (90, 60, 20))
    # 指针体重秤
    sx, sy = 320, 262
    poly(img, [(sx - 64, sy - 18), (sx + 64, sy - 18), (sx + 72, sy + 14), (sx - 72, sy + 14)], (234, 234, 240), 2)
    img[sy - 18:sy - 14, sx - 62:sx + 62] = (250, 250, 255)
    cv2.circle(img, (sx, sy + 2), 13, INK, -1); cv2.circle(img, (sx, sy + 2), 11, PAPER, -1)
    # 三帧（10-19 用户：只留 发愁 → 吸肚子 → 弹回去，别卡）：每帧之间靠挤压 / 回弹 / 抖动接起来，指针连续转
    T1, T2 = .13, .27
    phase = 0 if dt < T1 else 1 if dt < T2 else 2
    needle = float(np.interp(dt, [0, T1, T1 + .06, T2, T2 + .05, .45], [-40, -40, -62, -62, 150, 132])) + (4 * np.sin(t * 60) if phase == 2 else 0)
    a = np.deg2rad(needle - 90)
    cv2.line(img, (sx, sy + 2), (int(sx + 10 * np.cos(a)), int(sy + 2 + 10 * np.sin(a))), RED, 2)
    lab = ['20', '19', 'ERR'][phase]
    E.box(img, sx + 24, sy - 8, sx + 62, sy + 6, (40, 60, 40), 1)
    if phase < 2 or E.f24(t) % 4 < 3:
        img = E.text(img, sx + 26, sy - 10, lab, (140, 255, 120) if phase < 2 else (255, 90, 90))
    fr = chin_frames()[phase]
    if phase == 0:                                                     # 发愁：轻轻呼吸
        sxk, syk, jx = 1.0, 1 + .02 * np.sin(dt * 30), 0
    elif phase == 1:                                                   # 吸肚子：越憋越扁 + 发抖
        u = (dt - T1) / (T2 - T1)
        sxk, syk, jx = 1 - .04 * u, 1 + .03 * u, (1 if E.f24(t) % 2 else -1)
    else:                                                              # 弹回去：先撑大再回弹
        u = min((dt - T2) / .1, 1)
        sxk = syk = 1 + .18 * (1 - L.ease_out_back(u))                 # 1.18 倍 → 微微缩过头 → 落定
        jx = 0
    fr = cv2.resize(fr, (max(1, int(fr.shape[1] * sxk)), max(1, int(fr.shape[0] * syk))), interpolation=cv2.INTER_NEAREST)
    img = L.paste_rgba(img, fr, sx - fr.shape[1] // 2 + jx, sy - 16 - fr.shape[0])
    if phase == 2:                                                     # 弹回去：冲击线往外炸
        r0 = 70 + 40 * min((dt - T2) / .12, 1)
        for k in range(10):
            a = k * np.pi / 5 + .3
            cv2.line(img, (int(sx + r0 * np.cos(a)), int(sy - 80 + .8 * r0 * np.sin(a))),
                     (int(sx + (r0 + 16) * np.cos(a)), int(sy - 80 + .8 * (r0 + 16) * np.sin(a))), YEL, 2)
    return img


GUESTS = [  # (画面, 名字, 角色, 场记 TAKE)
    (g_mouth, 'KA-BOOM!', 'GPT-5', 1),          # 11-12 用户：首次体验 GPT-5 被震撼的梗
    (g_room, 'CHINESE ROOM', 'ニーハオだけ…は演技', 12),   # 11-12 和顶上黄字别重复：接旧名牌「ニーハオだけ」反转
    (g_shoggoth, 'SHOGGOTH', '鏡はウソつき', 3),
    (g_basilisk, "ROKO'S BASILISK", '実物 12cm', 7),
    (g_gato, 'GATO', '出演料：煮干し×3', 2),
    (g_clips, 'PAPERCLIPS', 'レンタル料 ¥∞', 5),
    (g_killswitch, 'KILLSWITCH GUYS', '有給休暇で欠席', 0),
    (g_lighter, 'LIGHTER', '実は火が苦手', 4),
    (g_chinchilla, 'CHINCHILLA', '減量中', 9),
]


def guest_index(t):
    return int(np.clip((t - G0) // B, 0, len(GUESTS) - 1))


def plate(img, name, role, dt):
    """片尾名牌：对白框位置的一条蓝底双线框，名字从左边滑进来，点线连到角色；角色一栏用 Zpix，好写简体中文。"""
    x0, y0, x1, y1 = 14, 296, W - 14, 340
    E.box(img, x0, y0, x1, y1, BLUE)
    cv2.rectangle(img, (x0 + 3, y0 + 3), (x1 - 4, y1 - 4), PAPER, 1)
    sx = int(-300 + 324 * L.ease_out_back(min(dt / .12, 1)))
    img = E.text(img, sx, y0 + 6, name, YEL, 2)
    nw = int(L.font(16).getlength(name)) * 2
    rw = zw(role, 2)
    dots = '.' * max(0, (x1 - 30 - rw - sx - nw - 8) // 8)
    img = E.text(img, sx + nw + 8, y0 + 14, dots, PAPER)
    return zh(img, x1 - 18 - rw, y0 + 7, role, PAPER, 2)


def scene_guest(t):
    i = guest_index(t)
    fn, name, role, take = GUESTS[i]
    dt = t - (G0 + i * B)
    img = np.array(fn(t, dt), np.uint8).copy()
    z = 1 + .08 * np.exp(-dt * 14)                                      # 砰地推近一下再落定
    img = E.zoom(img, 320, 170, z)
    img = E.lift(img, E.bounce(dt, 3))
    img = vhs(img, t, 'PLAY', f'MAKING  {i + 1:02d}/09')
    img = clapper(img, W - 150, 22, dt, i + 1, take if take else 'NG')
    return plate(img, name, role, dt)


SCENES = [
    ('audition', AUD_A, AUD_B, scene_audition),
    ('guest', G0, 145.72, scene_guest),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
