"""codeshots_p1r: 第一部分重做（09-30）的 A、B 两段。分镜见 docs/分镜/第一部分_0-38秒.md。

  A · I see sparks of AGI in your eyes（1.42–5.64）｜交换眼神：恋爱 AVG 的好感度 + 选项菜单
  B · Now I'm your servant, and you're my boss（12.52–16.18）｜交换身份：RPG 状态栏头衔互换

不生新图、不跑 Seedance：人物全部是旧片段 / 原关键帧的**静止帧**，按唱词硬切，不连续播、不对口型。
动的东西交给代码（火花、星星瞳孔、冷汗、推拉）。界面（好感度、选项、状态栏）在 overlay() 里画，
量化之后只用 FIX 原色，part1.OVERLAY 调用。

  python codeshots_p1r.py --sheet a2 2.7,3.0,3.5 --out x.png
"""
import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw

import codeshots_b
import part1_lib as L

W, H, FPS = L.W, L.H, L.FPS
CUR = L.ROOT / '镜头'
HI = (1672, 941)                                  # 原关键帧分辨率
INK, PAPER = (12, 11, 26), (246, 241, 228)
ORANGE, BLUE, YEL = (217, 119, 87), (34, 52, 150), (255, 210, 63)
NAVY = (16, 22, 54)

# ---------------- 素材 ----------------
@functools.lru_cache(maxsize=None)
def clip_frame(name, idx, native=False):
    """旧片段第 idx 帧（RGB）。native=True 给片段原分辨率（854×480，切特写更清楚），否则 640×360。"""
    cap = cv2.VideoCapture(str(CUR / f'shot_{name}.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    assert ok, (name, idx)
    if not native: f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


# P(doom) 8% 便利贴保留原样（用户 09-30）：A 段的 s1 画面里没有，到 B 段突然出现在显示器边框上，算伏笔
@functools.lru_cache(maxsize=None)
def s4_still(idx, native=False):
    return clip_frame('s4_servant480', idx, native)


@functools.lru_cache(maxsize=None)
def key_hi(name):
    return L.load_rgb(f'素材/关键帧/{name}.png', *HI)


def crop(img, cx, cy, w, out=(W, H)):
    """以 (cx,cy) 为中心、宽 w（都按 640 空间写）从图里切 16:9 一块，缩放到 out。高分辨率图自动换算，切特写不糊。"""
    s = img.shape[1] / W
    h = w * out[1] / out[0]
    x0, y0 = int(round((cx - w / 2) * s)), int(round((cy - h / 2) * s))
    x0 = int(np.clip(x0, 0, img.shape[1] - int(w * s))); y0 = int(np.clip(y0, 0, img.shape[0] - int(h * s)))
    sub = img[y0:y0 + int(h * s), x0:x0 + int(w * s)]
    return cv2.resize(sub, out, interpolation=cv2.INTER_AREA)


def zoom(img, cx, cy, z):
    """数码推近：以 (cx,cy) 为中心放大 z 倍（整帧）。"""
    if z <= 1.001: return img
    m = cv2.getRotationMatrix2D((cx, cy), 0, z)
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------- 小画笔 ----------------
def star4(img, x, y, r, color, core=None):
    """四角星（火花 / 星星瞳孔）。"""
    if r < 1: return img
    k = .28
    pts = np.array([(x, y - r), (x + r * k, y - r * k), (x + r, y), (x + r * k, y + r * k),
                    (x, y + r), (x - r * k, y + r * k), (x - r, y), (x - r * k, y - r * k)], np.int32)
    cv2.fillPoly(img, [pts], color)
    if core and r >= 4: cv2.circle(img, (int(x), int(y)), max(1, int(r * .22)), core, -1)
    return img


def burst(img, t, t0, ox, oy, n, seed, spread=(-1, 1), speed=(180, 420), life=.8, size=(3, 9)):
    """火花爆发：t0 时从 (ox,oy) 喷出 n 粒，纯函数（hash01 定每粒的方向、速度、大小、颜色）。"""
    dt = t - t0
    if not (0 <= dt < life): return img
    for i in range(n):
        a = np.pi * (spread[0] + (spread[1] - spread[0]) * L.hash01(seed, i))
        v = speed[0] + (speed[1] - speed[0]) * L.hash01(seed, i, 1)
        x = ox + np.cos(a) * v * dt
        y = oy + np.sin(a) * v * dt + 140 * dt * dt
        k = 1 - dt / life
        r = (size[0] + (size[1] - size[0]) * L.hash01(seed, i, 2)) * k
        c = (YEL, PAPER, (120, 235, 255))[int(L.hash01(seed, i, 3) * 3)]
        star4(img, x, y, r, c)
    return img


# =====================================================================
# A · sparks
# =====================================================================
S1_SCREEN_EYES = [(490, 163), (521, 158)]            # s1 关键帧屏幕上两只像素眼的瞳孔（640 空间）
A3_PUPILS = [(188, 226), (484, 220)]                 # 眼特写图两只瞳孔（640 空间）
T_SPARKS, T_AGI, T_STAR, T_A4, T_EYES, T_BLINK, T_A_END = 2.64, 3.64, 4.12, 4.80, 5.14, 5.30, 5.64


def scene_a1(t):
    """1.42–2.64 I see：开机拉远落到的关键帧原样（接缝），屏幕上的星星眼按拍一闪一闪。"""
    img = L.load_rgb('素材/关键帧/s1_火花之眼.png').copy()
    tw = L.twos(t)
    for j, (x, y) in enumerate(S1_SCREEN_EYES):
        star4(img, x, y, 3 + 3 * L.pulse(tw, 4), PAPER)
    return img


A2_CROP = (490, 150, 300)                             # 屏幕特写：中心 x, y, 宽（640 空间）


def _a2_map(x, y):
    cx, cy, w = A2_CROP
    s = W / w
    return (x - cx) * s + W / 2, (y - cy) * s + H / 2


def scene_a2(t):
    """2.64–3.64 sparks of：屏幕特写（关键帧原分辨率切），瞳孔炸成大星星，火花往左（Claude 那边）喷。"""
    img = crop(key_hi('s1_火花之眼'), *A2_CROP).copy()
    tw = L.twos(t)
    dt = tw - T_SPARKS
    for j, (x, y) in enumerate(S1_SCREEN_EYES):
        px, py = _a2_map(x, y)
        r = 26 * L.ease_out_back(min(dt / .12, 1), 3) + 5 * L.pulse(tw, 5)
        star4(img, px, py, r, YEL, PAPER)
    ox, oy = _a2_map(505, 150)
    burst(img, tw, T_SPARKS, ox, oy, 34, 11, spread=(.55, 1.45), speed=(220, 520))
    burst(img, tw, 3.44, ox, oy, 16, 12, spread=(.6, 1.4), speed=(160, 360), life=.5)
    return img


A3_SPARKLES = [(209, 176), (498, 155)]               # 眼特写图里瞳孔上方那颗四角星高光（640 空间）


def scene_a3(t):
    """3.64–4.80 AGI in：眼睛大特写（09-30 新画 p1r_a3_眼特写_a，瞳孔里倒映着星星眼的显示器）。
    一粒火花从右上飞进左眼，4.12 瞳孔里的星星高光「叮」地放大一闪、周围迸出小火花。缓慢推近。"""
    img = L.load_rgb('素材/CG/p1r_a3_眼特写_a.png').copy()
    tw = L.twos(t)
    (lx, ly), _ = A3_PUPILS
    if tw < T_STAR:                                   # 飞行的火花 + 拖尾
        k = L.smooth((tw - T_AGI) / (T_STAR - T_AGI))
        for j in range(5):
            kj = max(0, k - j * .05)
            x = 700 + (lx - 700) * kj
            y = 20 + (ly - 20) * kj - 60 * np.sin(np.pi * kj)
            star4(img, x, y, (14 if j == 0 else 7 - j) * (1 if kj > 0 else 0), YEL if j == 0 else PAPER, PAPER if j == 0 else None)
    else:
        dt = tw - T_STAR
        for x, y in A3_SPARKLES:
            r = 20 * max(0, 1 - dt / .35) + 6 * L.pulse(tw, 5)
            star4(img, x, y, r, PAPER, YEL)
        for x, y in A3_PUPILS:
            burst(img, tw, T_STAR, x, y, 10, 21 + x, spread=(0, 2), speed=(120, 260), life=.45, size=(3, 7))
    z = 1 + .08 * L.smooth((t - T_AGI) / (T_A4 - T_AGI))
    return zoom(img, 336, 200, z)


def scene_a4(t):
    """4.80–5.64 your eyes：回到全景（张嘴笑），5.14 选中「見つめ返す」时闭眼笑 = 眨眼，5.30 再睁开；满屏小星星。"""
    tw = L.twos(t)
    img = clip_frame('s1_sparks480', 116 if T_EYES <= tw < T_BLINK else 128).copy()
    for x, y in [(486, 150), (519, 146)]:             # 屏幕上的像素眼
        star4(img, x, y, 4 + 4 * L.pulse(tw, 5), PAPER)
    if tw >= T_EYES:                                  # 好感度满：满屏小星星（每颗自己闪）
        for i in range(26):
            x, y = 20 + L.hash01(i, 5) * 600, 24 + L.hash01(i, 6) * 250
            ph = (tw - T_EYES) * 6 + L.hash01(i, 7) * 6
            star4(img, x, y, 7 * max(0, np.sin(ph)), YEL if i % 2 else PAPER)
    return img


# =====================================================================
# B · servant
# =====================================================================
# 09-30 用户：RPG 状态栏效果不好 → 回到台本 v1「纸王冠砸到显示器头上 + 换装定格三连」，允许生图
T_NOW, T_FALL, T_LAND, T_B2, T_BOSS, T_B_END = 12.52, 12.52, 12.99, 13.67, 15.62, 16.18
OUTFIT = [(13.67, 'a'), (14.35, 'b'), (15.03, 'c')]   # 换装三连，一拍一张（88 BPM 拍点）：戴头饰 → 系围裙 → 端托盘
CROWN_BOX = (1360, 90, 1560, 216)                     # 王冠在换装 a 图里的范围（原图像素），底边就是显示器顶
B1_CROP = (520, 175, 600)                             # 显示器镜头：中心 x, y, 宽（640 空间）。09-30 用户：原来 340 宽推太近、
                                                      # 显示器下半截被对话框挡住 → 600 宽，王冠顶 ≈ 画面 y30，机身底 ≈ y282（对话框上沿）


@functools.lru_cache(maxsize=None)
def _cg_hi(name):
    return L.load_rgb(f'素材/CG/{name}.png', *HI)


@functools.lru_cache(maxsize=None)
def _crown():
    """王冠精灵：换装 a 图（有王冠）减去无王冠图（显示器逐像素对得上），差值就是王冠。返回 (RGBA, x0, y0)。"""
    x0, y0, x1, y1 = CROWN_BOX
    a = _cg_hi('p1r_b2_换装_a')[y0:y1, x0:x1]
    n = _cg_hi('p1r_b1_显示器无王冠')[y0:y1, x0:x1]
    m = (np.abs(a.astype(int) - n.astype(int)).sum(2) > 60).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return np.dstack([a, m * 255]).astype(np.uint8), x0, y0


@functools.lru_cache(maxsize=None)
def _plate(crown=True):
    """B 段的背景底板：空房间（无人、无盘子、无王冠，改图自 p1r_b1_显示器无王冠），可选把王冠贴回显示器头上。
    换装三连的背景和电脑都用它，所以换张时只有人物在动（用户 09-30：只人物抖，电脑和背景不动）。"""
    img = _cg_hi('p1r_b_空房间').copy()
    if crown:
        spr, x0, y0 = _crown()
        img = L.paste_rgba(img, spr, x0, y0)
    return img


@functools.lru_cache(maxsize=None)
def _char(k):
    """换装图里的人物层（640 空间 RGBA）：rembg isnet-anime 抠图，遮罩缓存在图旁边的 .alpha.npz。
    （试过「换装图 − 空房间」的差值：空房间是改图重画的，左半边床和窗帘对不上，会被当成人物一起抖。）"""
    path = L.ROOT / f'素材/CG/p1r_b2_换装_{k}.png'
    cache = path.with_suffix('.alpha.npz')
    if cache.exists():
        m = np.load(cache)['a']
    else:
        from rembg import new_session, remove
        m = np.array(remove(Image.open(path).convert('RGB'), session=new_session('isnet-anime'), only_mask=True))
        np.savez_compressed(cache, a=m)
    a = _cg_hi(f'p1r_b2_换装_{k}')
    m = cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    rgba = np.dstack([a, np.where(m > 127, 255, 0)]).astype(np.uint8)
    return cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)


def scene_b1(t):
    """12.52–13.67 Now I'm：显示器特写（空房间底板，没有盘子穿帮），12.52 纸王冠从上面掉，12.99 拍上「咚」地扣在头顶、挤扁回弹、冒尘星。"""
    img = _plate(False).copy()
    tw = L.twos(t)
    spr, cx0, cy0 = _crown()
    h, w = spr.shape[:2]
    if tw >= T_FALL:
        if tw < T_LAND:                               # 下落（起点贴着取景框上沿，约 0.3 秒看得见在掉）
            k = (tw - T_FALL) / (T_LAND - T_FALL)
            dy, sx, sy = -200 * (1 - k) ** 1.5, 1, 1
        else:                                         # 落地挤扁 → 回弹
            dt = tw - T_LAND
            dy = 0
            sq = .28 * np.exp(-dt * 14) * np.cos(dt * 40)
            sx, sy = 1 + sq, 1 - sq
        s = cv2.resize(spr, (max(1, int(w * sx)), max(1, int(h * sy))), interpolation=cv2.INTER_NEAREST)
        img = L.paste_rgba(img, s, cx0 + (w - s.shape[1]) / 2, cy0 + h - s.shape[0] + dy)
    img = crop(img, *B1_CROP).copy()
    if tw >= T_LAND:                                  # 落地尘星：从王冠底边两侧迸出
        s = W / B1_CROP[2]
        bx = (CROWN_BOX[0] + CROWN_BOX[2]) / 2 * W / HI[0]
        by = CROWN_BOX[3] * W / HI[0]
        cw, ch = B1_CROP[2], B1_CROP[2] * H / W       # crop() 超出图边会把框推回来，尘星要按实际裁切位置换算（09-30 修：原来偏左）
        x0 = np.clip(B1_CROP[0] - cw / 2, 0, W - cw); y0 = np.clip(B1_CROP[1] - ch / 2, 0, H - ch)
        ox, oy = (bx - x0) * s, (by - y0) * s
        burst(img, tw, T_LAND, ox, oy, 14, 31, spread=(-1, 0), speed=(140, 320), life=.4, size=(4, 9))
    return img


def scene_b2(t):
    """13.67–15.62 your servant, and you're my：换装定格三连（同构图三张，一拍一张）。
    背景和电脑是固定的底板，只有人物层在动：换的那一刻人物挤扁回弹 + 左右抖两下 + 冒小星星。"""
    tw = L.twos(t)
    t0, k = [o for o in OUTFIT if o[0] <= t][-1] if t >= OUTFIT[0][0] else OUTFIT[0]   # 换张按真实时间卡拍（twos 取整会晚一帧）
    tw = max(tw, t0)
    img = cv2.resize(_plate(), (W, H), interpolation=cv2.INTER_AREA)
    ch = _char(k)
    dt = tw - t0
    if 0 <= dt < .25:
        sq = .07 * np.cos(dt / .25 * np.pi / 2) if dt < .17 else 0      # 挤扁回弹（以脚底为基准，人物中线 x≈230）
        dx = [0, 6, -5, 3, -2, 0][min(int(dt * 24), 5)]                 # 左右抖
        m = np.float32([[1 + sq, 0, -sq * 230 + dx], [0, 1 - sq, sq * H]])
        ch = cv2.warpAffine(ch, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = L.paste_rgba(img, ch, 0, 0)
    burst(img, tw, t0, 230, 60, 12, 41 + int(t0 * 10), spread=(-1, 1), speed=(160, 300), life=.35, size=(4, 10))
    return img


BOW = [(15.62, 56), (15.68, 60), (15.74, 64)]   # 鞠躬：s4 片段的三个关键姿势，快速鞠下去，64 帧鞠到底
                                                # （52 帧 rembg 把托盘抠丢了，托盘会一闪消失，不用）
BOW_DY = 0                                                   # 人物层整体下移（s4 和空房间的桌面高度差，看联系表调）


@functools.lru_cache(maxsize=None)
def _s4_char(idx):
    """s4 片段第 idx 帧的人物层（640 空间 RGBA）：rembg isnet-anime，遮罩缓存到 素材/CG/p1r_s4_f<idx>.alpha.npz。
    只留显示器左边（x < 440），显示器和便利贴不进人物层。"""
    img = clip_frame('s4_servant480', idx)
    cache = L.ROOT / f'素材/CG/p1r_s4_f{idx}.alpha.npz'
    if cache.exists():
        m = np.load(cache)['a']
    else:
        from rembg import new_session, remove
        m = np.array(remove(Image.fromarray(clip_frame('s4_servant480', idx, True)), session=new_session('isnet-anime'), only_mask=True))
        np.savez_compressed(cache, a=m)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    m[:, 440:] = 0
    return np.dstack([img, np.where(m > 127, 255, 0)]).astype(np.uint8)


def scene_b4(t):
    """15.62–16.18 boss：鞠躬。人物从 s4 片段抠出来，叠在和换装三连同一张空房间底板上（背景不跳），
    52 → 56 → 60 → 64 四个姿势快速鞠下去，鞠到底挤扁回弹、冒几颗星。16.18 起接回原来的 s4 片段（C 段还没重做）。"""
    img = cv2.resize(_plate(), (W, H), interpolation=cv2.INTER_AREA)
    t0, idx = [b for b in BOW if b[0] <= t][-1] if t >= BOW[0][0] else BOW[0]
    ch = _s4_char(idx)
    dt = t - BOW[-1][0]
    if 0 <= dt < .2:                                  # 鞠到底：挤扁回弹（以脚底为基准）
        sq = .08 * np.cos(dt / .2 * np.pi / 2)
        m = np.float32([[1 + sq, 0, -sq * 180], [0, 1 - sq, sq * H]])
        ch = cv2.warpAffine(ch, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = L.paste_rgba(img, ch, 0, BOW_DY)
    burst(img, L.twos(t), BOW[-1][0], 200, 170, 10, 57, spread=(-1, 0), speed=(140, 280), life=.35, size=(4, 9))
    return img


# =====================================================================
# C · ChatGPT, please don't eat me alive
# =====================================================================
# 一条 Seedance（shot_p1r_c_eat720，首帧 = 旧 s4 末帧，用户 09-30 批准 720p 8 秒 6.43 元），只用前半：
#   片段 0–3.1s  怪物挤出屏幕越长越大 → ChatGPT～（16.92–19.02），缓慢推近怪物
#   片段 3.1–4.8s（压暗 + 速度线）当 please don't eat me 切入特写的背景
# Claude 吓哭 / 被吞那几秒（v5 用过）脸会崩，v6 起不用，见下方「C 段 v6」。
T_C0, T_PLEASE, T_OPEN, T_HOLD, T_ALIVE, T_DARK, T_C_END = 16.915, 19.02, 21.18, 21.25, 21.32, 21.48, 21.60


@functools.lru_cache(maxsize=None)
def _c_frames():
    cap, out = cv2.VideoCapture(str(CUR / 'shot_p1r_c_eat720.mp4')), []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


# ChatGPT～：「放大 → 停顿 → 再放大」（用户 09-30：更有张力）。每个 132 拍点：怪物跳到下一个姿势（片段帧），
# 镜头猛地推近一级（0.08 秒带回弹）、震一下，然后人和镜头都定住，等下一拍。(歌曲时间, 片段帧, 怪物脸中心, 放大)
# 09-30 v8 用户：「Chat」一出口（16.18）就该推镜，不能等到 17.09 → 前两级落在 16.18 / 16.63，
# 这时画面还是旧 s4 的变红过程（片段帧 None = 按旧 WARP 3.2 倍速播 s4），16.915 切进新片段时倍数不变、同一帧接上。
# v9 用户：从 C 开始一共推四次 → 一个音节推一级：Chat 16.18 / G 17.09 / P 17.54 / T 18.00
# （Whisper 测：G 约 16.8–17.1、P 17.60、T 18.10，P、T 正好隔一拍，G 取前一拍）。
# v8 按拍推了 7 次，和唱的音节对不上，节奏怪。T 拖长到 please 前，画面定住、只极慢地往里挤（C_CREEP）。
# v10 用户：按鼓点来，Chat 后空几拍，G P T 都在鼓点上。底鼓实测（40–120Hz 起音）：
# 16.63 / 17.08 / 17.53 / 17.99 / 18.43 / 18.89（16.63 前没有底鼓）。
# → Chat 16.18 推一级，空 16.63、17.08 两拍（画面接着播，不推），G 17.53 / P 17.99 / T 18.43 各推一级。
T_S4 = 16.18
# v11 用户：P、T 对了，G 不对，节奏是「C 空 G P T」→ C 16.18，空 16.63（第一个底鼓），G 落在下一个底鼓 17.08
# v12 用户：「粗暴放大」效果不好、看不全，空拍太僵 → 反过来：C 切到屏幕獠牙特写，G / P / T 每一下往外拉一级，
# T 拉到全景让怪物整个亮相（视频里它本来就越长越大，镜头退、它反而更满屏）；空拍放冲击帧（黑白反相 2 帧 + 震屏）。
# v13 用户：定住的地方不能动（v12 两级之间片段还在往前播）；节奏交给 Claude 自己定：
# 「渐快 → 亮相 → 定住」。16.63 是变速后第一个底鼓，全段最重的一下：冲击帧 + 第一刀拉出；
# 之后间隔 2 拍 → 1 拍 → 半拍 → 半拍，越拉越急，18.43 拉到全景（第二个冲击帧），然后纹丝不动地亮相到 please。
# 每一级都是一张静止帧，只在切的那一刻有拉镜回弹、红光、震屏。
C_STEPS = [(T_S4, None, (500, 175), 2.6), (16.63, 14, (505, 175), 2.1), (17.53, 28, (495, 180), 1.7),
           (17.99, 44, (465, 170), 1.4), (18.21, 58, (430, 160), 1.18), (18.43, 75, (320, 180), 1.0)]
C_IMPACT = [(16.63, .085), (18.43, .085)]             # 冲击帧：(开始, 时长)
C_KICKS = [s[0] for s in C_STEPS[1:]]                 # 红光只跟着切镜闪，定住的时候什么都不动


def _c_src(t, idx):
    """这一刻的画面源：旧 s4（16.915 前，按旧 WARP 播、一拍二）/ 新片段第 0 帧（16.915 起、第一个新姿势之前）/ 新片段第 idx 帧。"""
    if t < T_C0:
        i = int(round((3.75 + (t - 16.2) * 3.2) * 24))
        return s4_still(min(i - i % 2, 144), True)
    return _c_frames()[idx or 0]                      # 每一级是一张静止帧


def _impact(img):
    """冲击帧（インパクトフレーム）：黑白二值 + 反相，亮的地方变墨色、暗的地方变纸白，红色描一层。"""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m = g > np.median(g)
    out = np.empty_like(img)
    out[m], out[~m] = INK, PAPER
    edge = cv2.Canny(g, 60, 140) > 0
    out[edge] = (230, 40, 50)
    return out


def scene_c(t):
    k = max(j for j, s in enumerate(C_STEPS) if s[0] <= t) if t >= T_S4 else 0
    t0, idx, (cx, cy), z = C_STEPS[k]
    (px, py), pz = (C_STEPS[k - 1][2], C_STEPS[k - 1][3]) if k else ((cx, cy), 1.0)
    dt = t - t0
    e = L.ease_out_back(min(dt / .1, 1), 2)           # 每级拉出去时带一点过冲回弹
    cx, cy, z = px + (cx - px) * e, py + (cy - py) * e, pz + (z - pz) * e
    z = max(z, 1.0)                                   # T 拉到全景时回弹过冲会 < 1，裁切超出画面整帧变黑
    img = crop(_c_src(t, idx), cx, cy, W / z).copy()
    kick = max([t - b for b in C_KICKS if b <= t], default=9)
    if kick < .18:                                    # 每个底鼓：红光闪一下
        a = .35 * (1 - kick / .18)
        img = (img * (1 - a) + np.array((240, 30, 40)) * a).astype(np.uint8)
    if dt < .2 or kick < .12:                         # 拉镜到位 / 鼓点：震一下
        d = min(dt, kick)
        img = np.roll(img, int(7 * np.exp(-d * 20) * np.sin(d * 110)), axis=0)
    if any(a <= t < a + d for a, d in C_IMPACT):     # 空拍：冲击帧
        img = _impact(img)
    return img


# ---------------- C 段 v6：please don't eat me 改成「切入特写」，alive 改成「牙齿咬合」，吞完 GAME OVER ----------------
# 09-30 用户：v5 节奏还行但脸崩（Seedance 画成现代柔光画风，推近 + 16 色更走样）→ Claude 的脸不再交给视频模型：
# 按设定图生三张特写，一个词一张斜切进来；怪物继续用视频（背景里接着逼近）。
# 顺序（用户 09-30）：先抱托盘躲着 → 再合十祈求 → 最后飙泪推手
CUTINS = [(19.02, 'eatme_v2', (318, 165), 1.20, 1), (19.42, 'please', (318, 150), 1.20, -1), (20.16, 'dont', (325, 160), 1.15, 1)]
T_ME = 20.74                                          # me：推到眼睛
BAND = np.array([(0, 70), (W, 38), (W, 238), (0, 270)], np.int32)   # 斜切画框（左低右高）
JAW_OPEN, JAW_SHUT = (54, 286), (150, 190)            # 上牙床下沿 / 下牙床上沿：张开 → 咬合
T_GO, T_CONT, T_YES, T_LOAD = 21.60, 21.95, 22.12, 22.24


@functools.lru_cache(maxsize=None)
def _cut_img(name):
    return _cg_hi(f'p1r_c_切入_{name}')


def _cut_face(name, eye, z, dz=0.0):
    """切入特写的画面：以眼睛为中心放大 z 倍（高分辨率原图切，不糊）。"""
    return crop(_cut_img(name), eye[0], eye[1], W / (z + dz))


def _speedlines(img, t, cx, cy):
    """红色放射速度线（每三帧换一批，纯函数）。"""
    fi = int(t * 8)
    out = img.copy()
    for i in range(46):
        a = 2 * np.pi * L.hash01(fi, i)
        w = .018 + .03 * L.hash01(fi, i, 1)
        r0 = 90 + 120 * L.hash01(fi, i, 2)
        p = np.array([(cx + np.cos(a - w) * 900, cy + np.sin(a - w) * 900), (cx + np.cos(a + w) * 900, cy + np.sin(a + w) * 900),
                      (cx + np.cos(a) * r0, cy + np.sin(a) * r0)], np.int32)
        cv2.fillPoly(out, [p], (120, 10, 20) if i % 3 else (240, 60, 50))
    return out


def scene_cut(t):
    """19.02–21.18 please / don't / eat me：怪物视频在背景里接着逼近（压暗 + 速度线），斜切画框里换特写。"""
    frames = _c_frames()
    i = int(np.clip(round((75 + (t - T_PLEASE) / (T_OPEN - T_PLEASE) * 40)), 0, len(frames) - 1))
    bg = cv2.resize(frames[i - i % 3], (W, H), interpolation=cv2.INTER_AREA)
    bg = _speedlines((bg * .55).astype(np.uint8), t, 330, 150)
    t0, name, eye, z, side = [c for c in CUTINS if c[0] <= t][-1] if t >= CUTINS[0][0] else CUTINS[0]
    dz = .55 * L.smooth((t - T_ME) / .3) if t >= T_ME else 0
    face = _cut_face(name, eye, z, dz)
    dt = t - t0
    if dt < .14:                                      # 从一侧唰地切进来（带回弹）
        face = np.roll(face, int(side * W * (1 - L.ease_out_back(dt / .14, 2))), axis=1)
    if t >= T_ME:                                     # me：画框里一直抖
        face = np.roll(face, int(4 * np.sin(t * 90)), axis=1)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [BAND], 1)
    out = np.where(m[..., None] > 0, face, bg)
    for (y0a, y1a), c, th in (((0, 1), INK, 10), ((0, 1), PAPER, 3), ((3, 2), INK, 10), ((3, 2), PAPER, 3)):
        cv2.line(out, tuple(map(int, BAND[y0a])), tuple(map(int, BAND[y1a])), c, th)
    return out


def _jaws(img, y_top, y_bot):
    """怪物的像素牙：上下两排青色牙床 + 白色三角牙（墨线描边），上下错开半颗，合上时咬在一起。"""
    gum, gum2 = (70, 225, 220), (30, 150, 160)
    tw, th = 48, 46
    cv2.rectangle(img, (0, -10), (W, y_top), gum, -1)
    cv2.rectangle(img, (0, y_top - 14), (W, y_top), gum2, -1)
    cv2.rectangle(img, (0, y_bot), (W, H + 10), gum, -1)
    cv2.rectangle(img, (0, y_bot), (W, y_bot + 14), gum2, -1)
    for k in range(-1, W // tw + 2):
        for up, x0, y0 in ((False, k * tw, y_top), (True, k * tw + tw // 2, y_bot)):
            tip = y0 - th if up else y0 + th
            p = np.array([(x0 + 3, y0), (x0 + tw - 3, y0), (x0 + tw // 2, tip)], np.int32)
            cv2.fillPoly(img, [p], (250, 250, 245))
            cv2.polylines(img, [p], True, INK, 3)
    cv2.line(img, (0, y_top), (W, y_top), INK, 4)
    cv2.line(img, (0, y_bot), (W, y_bot), INK, 4)
    return img


def scene_jaws(t):
    """21.18–21.60 alive：最后那张（飙泪推手）铺满全屏，上下牙从画面外咬进来 → 21.25 张到最大、滞空 → 21.32 咔嚓合上 → 慢慢暗下去。"""
    img = _cut_face('dont', (325, 170), 1.25).copy()
    if t < T_HOLD:
        k = L.ease_out_back((t - T_OPEN) / (T_HOLD - T_OPEN), 1.5)
        yt, yb = int(-60 + (JAW_OPEN[0] + 60) * k), int(H + 60 - (H + 60 - JAW_OPEN[1]) * k)
    elif t < T_ALIVE:
        yt, yb = JAW_OPEN
    else:
        k = min(1, (t - T_ALIVE) / .06)               # 两帧就合上
        yt = int(JAW_OPEN[0] + (JAW_SHUT[0] - JAW_OPEN[0]) * k)
        yb = int(JAW_OPEN[1] + (JAW_SHUT[1] - JAW_OPEN[1]) * k)
    img = _jaws(img, yt, yb)
    if t >= T_ALIVE + .12:                            # 咬住以后网点暗下去（嘴里一片黑）
        img[L.bayer_mask((t - T_ALIVE - .12) / .16)] = INK
    return img


def scene_gameover(t):
    """21.60–22.43：吞完的黑屏 → GAME OVER → CONTINUE? ▶YES → NOW LOADING（接舞台的顶光）。字直接画在帧里，舞台开头的网点溶解才接得上。"""
    img = np.full((H, W, 3), INK, np.uint8)
    if t < T_LOAD:
        k = (t - T_GO) * FPS
        sc = 5 if k < 2 else 4
        s = 'GAME OVER'
        w = int(L.font(16).getlength(s)) * sc
        img = L.draw_text(img, ((W - w) // 2 + 4, 76 + 4), s, (90, 10, 20), scale=sc)
        img = L.draw_text(img, ((W - w) // 2, 76 + int(3 * np.sin(k * 2)) * (k < 8)), s, (230, 40, 50), scale=sc)
        msg = 'クロードは 食べられてしまった！'
        n = int(max(0, t - T_GO - .06) * 60)          # CONTINUE 出来（21.95）前要打完
        img = L.draw_text(img, ((W - int(L.font(16).getlength(msg))) // 2, 164), msg[:n], PAPER)
        if t >= T_CONT:
            img = L.draw_text(img, (262, 208), 'CONTINUE?', YEL)
            yes = t >= T_YES and int((t - T_YES) * 16) % 2 == 0
            if yes: cv2.rectangle(img, (268, 234), (318, 254), YEL, -1)
            img = L.draw_text(img, (284, 236), 'YES', INK if yes else PAPER)
            img = L.draw_text(img, (344, 236), 'NO', PAPER)
            if t < T_YES or yes or int(t * 8) % 2 == 0:
                cv2.fillPoly(img, [np.array([(270, 238), (270, 250), (279, 244)], np.int32)], INK if yes else YEL)
    else:
        img = codeshots_b.load_cone(img, t)       # 最后 0.15 秒从顶上打下聚光灯锥，接舞台顶光（旧黑场版有，重做时丢了，09-30 加回）
        s = 'NOW LOADING' + '.' * (1 + int((t - T_LOAD) * 12) % 3)
        img = L.draw_text(img, (248, 220), s, PAPER)
    return img


SCENES = [
    ('a1', 2.05, 2.64, scene_a1), ('a2', 2.64, 3.64, scene_a2), ('a3', 3.64, 4.80, scene_a3), ('a4', 4.80, 5.64, scene_a4),
    ('b1', 12.52, 13.67, scene_b1), ('b2', 13.67, 15.62, scene_b2), ('b4', 15.62, 16.18, scene_b4),
    ('c', T_C0, T_PLEASE, scene_c), ('cut', T_PLEASE, T_OPEN, scene_cut), ('jaws', T_OPEN, T_C_END, scene_jaws),
    ('gameover', T_C_END, 22.43, scene_gameover),
]


# =====================================================================
# 界面层（量化之后画，只用 FIX 原色）：part1.OVERLAY
# =====================================================================
F16 = L.font(16)
HEART = ['0110110', '1111111', '1111111', '0111110', '0011100', '0001000']


def _heart(d, x, y, fill, s=2):
    for r, row in enumerate(HEART):
        for c, ch in enumerate(row):
            if ch == '1': d.rectangle((x + c * s, y + r * s, x + c * s + s - 1, y + r * s + s - 1), fill=fill)


def _window(d, x0, y0, x1, y1, fill=BLUE):
    """PC-98 双线窗口。"""
    d.rectangle((x0, y0, x1, y1), fill=fill)
    d.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline=PAPER)
    d.rectangle((x0 + 4, y0 + 4, x1 - 4, y1 - 4), outline=PAPER)


def _text(d, xy, s, fill, scale=1, img=None):
    if scale == 1:
        d.text(xy, s, fill=fill, font=F16); return
    l, t_, r, b = F16.getbbox(s)                      # 按字形实际外框建画布（原来固定 18 高，字的下半截被切掉）
    m = Image.new('1', (r - l + 2, b - t_ + 2), 0)
    ImageDraw.Draw(m).text((1 - l, 1 - t_), s, fill=1, font=F16)
    m = m.resize((m.width * scale, m.height * scale), Image.NEAREST)
    img.paste(Image.new('RGB', m.size, fill), (int(xy[0]), int(xy[1])), m)


# 好感度：(时刻, 格数)。7 格满
LOVE = [(2.20, 1), (T_SPARKS, 3), (T_AGI, 5), (T_STAR, 6), (T_EYES, 7)]
PING = [(T_SPARKS, 'ピロリン♪'), (T_AGI, '♡+2'), (T_STAR, '♡+1')]


def _love_level(t):
    lv = 0
    for i, (a, n) in enumerate(LOVE):
        if t >= a:
            prev = LOVE[i - 1][1] if i else 0
            lv = min(n, prev + int((t - a) * FPS / 2) + 1)   # 两帧涨一格
    return lv


def _galge(im, d, t):
    if t < 2.20: return
    # 09-30 用户：原来 16px 一行太细看不清 → 框加高、「好感度」两倍字、心放大到 3 倍像素
    x0, y0, x1, y1 = 300, 22, 628, 72
    full = t >= T_EYES
    flash = full and int((t - T_EYES) * 8) % 2 == 0
    _window(d, x0, y0, x1, y1, YEL if flash else BLUE)
    _text(d, (x0 + 12, y0 + 9), '好感度', INK if flash else ORANGE, 2, im)
    lv = _love_level(t)
    for i in range(7):
        pop = 1 if (i == lv - 1 and t - max(a for a, n in LOVE if t >= a) < .1) else 0   # 刚点亮的那颗往上跳一下
        hx, hy = x0 + 116 + i * 29, y0 + 13 - 3 * pop
        if i < lv:                                    # 亮：橙心 + 墨色投影；暗：只有墨色心
            _heart(d, hx + 2, hy + 2, INK, 3)
            _heart(d, hx, hy, ORANGE, 3)
        else:
            _heart(d, hx, hy, INK, 3)
    for a, s in PING:                                 # 涨的时候飘一行两倍大字上去（墨色投影垫底，压在什么画面上都看得清）
        dt = t - a
        if 0 <= dt < .7:
            yy = y1 + 36 - int(min(dt / .3, 1) * 24)
            xx = x1 - 12 - F16.getlength(s) * 2
            _text(d, (xx + 3, yy + 3), s, INK, 2, im)
            _text(d, (xx, yy), s, YEL if int(dt * 12) % 2 == 0 else PAPER, 2, im)
    if t >= T_A4:                                     # 选项菜单
        mx0, my0, mx1, my1 = 400, 204, 628, 274
        _window(d, mx0, my0, mx1, my1)
        opts = ['見つめ返す', '電源を切る']
        if t < 4.97: cur = 0
        elif t < 5.05: cur = 1                        # 在「关机」上犹豫了一下
        else: cur = 0
        for i, s in enumerate(opts):
            y = my0 + 12 + i * 26
            if full and i == 0 and int((t - T_EYES) * 12) % 2 == 0:
                d.rectangle((mx0 + 10, y - 2, mx1 - 10, y + 18), fill=YEL)
                d.text((mx0 + 36, y), s, fill=INK, font=F16)
            else:
                d.text((mx0 + 36, y), s, fill=PAPER, font=F16)
            if i == cur and (full or int(t * 6) % 2 == 0):    # 光标自己画三角（像素字体里的 ▶ 是方框）
                d.polygon([(mx0 + 18, y + 2), (mx0 + 18, y + 14), (mx0 + 27, y + 8)], fill=INK if full else YEL)


def overlay(img, t):
    if not 2.05 <= t < 5.64: return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.fontmode = '1'
    _galge(im, d, t)
    return np.array(im)


if __name__ == '__main__':
    L.sheet_cli(SCENES)
