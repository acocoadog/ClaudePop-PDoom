"""codeshots_s9: 死神之眼看到的世界（11-12，完整版 35.05–38.54，接在红瞳孔推进之后；结尾点按钮弹出第二部分的训练窗口）。

死神之眼能看到人头上的名字和寿命 → 这里看到的是**国籍 + 账号寿命**（大字寿命，小字 P(BAN)），每个观众都有。
演唱会观众里混着一个穿「I ♥ CALIFORNIA」的人，扫描线扫到他 → 标签乱码 → 推镜到他，一行一行漏出
时区/语言/输入法/WebRTC → 旗子翻成中国、寿命掉到 0 → SUSPENDED 盖章 → 右边滑出仿 A 社封号页的邮件
→ 鼠标点「Request review」→ 38.21/38.35/38.44 三下过门鼓：按钮放出线框、弹出 TRAINING RUN #2 窗口 → 38.54 唱 We had。
后半段全部卡在鼓点上（librosa 起音实测）：36.16 36.39 36.63 36.84 37.07 37.32 37.53(最重) 37.74 37.97 38.21 38.35 38.44。

素材：素材/关键帧/s9b_观众_pp_v5.png（gpt 生的像素画）。
画面（人群、红色视野）在镜头函数里画全彩，交给 part1 锁调色板；
标签、扫描线、漏出的信息、准星、印章、邮件都在 overlay() 里画（量化之后画，字不会被抖花）。

  python codeshots_s9.py --preview 35.4,35.9,36.5,37.0,37.2,37.8 x.png
"""
import functools

import cv2
import numpy as np

import mv
import part1_lib as L

W, H = L.W, L.H
FIX = mv.FIX
INK, PAPER, RED, BLUE, YEL = FIX['ink'], FIX['paper'], FIX['red'], FIX['blue'], FIX['yel']
GREEN = (20, 140, 70)

# ---------------- 时间（歌曲时间；前半按 132 BPM 拍点，后半按鼓点） ----------------
T_IN = 35.05          # 从瞳孔里出来
T_ROWS = (35.27, 35.50, 35.72)     # 后排 / 中排 / 前排标签依次弹出
T_SCAN0, T_SCAN1 = 35.72, 36.16    # 扫描线从左扫到右
T_LOCK = 36.16        # 推镜、准星
LEAK = [(36.39, 'TZ: UTC+8'), (36.63, 'LANG: zh-CN'), (36.84, 'IME: Sogou Pinyin'), (37.07, 'WebRTC: 223.x.x.x')]  # 一下鼓一行
T_FLIP = 37.32        # 旗子翻成中国、寿命倒数
T_STAMP = 37.53       # SUSPENDED（这一段最重的鼓）
T_MAIL = 37.74        # 右边滑出邮件
T_CLICK = 37.97       # 鼠标点 Request review
T_MORPH = (38.21, 38.35, 38.44)    # 过门三下：按钮放出线框 → 训练窗口弹出 → 边框闪黄
T_POP = 38.00         # 点下按钮（37.97）后窗口开始放大，38.21 铺满；窗口里的 loss 曲线 38.00→38.54 从头画完
T_END = 38.54         # (=帧格 925/24 附近，交接点必须落在帧格上) We had 开唱，接第二部分（part2 的 stable 从这里开始，开头加快追回原节奏）

# ---------------- 人群 ----------------
HI = (1280, 720)      # 人群图先缩到 1280×720 再做推拉（640 屏幕上 z=.5 时正好整张）


@functools.lru_cache(maxsize=None)
def crowd():
    return L.load_rgb('素材/关键帧/s9b_观众_pp_v5.png', *HI)


# 头顶位置（1280×720 人群图里的坐标）、国籍、寿命、P(BAN)。三排各 7 人，加州男是中排第 4 个
PEOPLE = [
    ((140, 140), 'JP', '18250', '0.2%'), ((300, 140), 'DE', '24901', '0.1%'), ((465, 148), 'FR', '9431', '0.3%'),
    ((640, 130), 'US', '31702', '0.1%'), ((810, 150), 'NL', '20077', '0.2%'), ((980, 148), 'IT', '27530', '0.1%'),
    ((1165, 148), 'TH', '16408', '0.2%'),
    ((95, 230), 'ID', '22314', '0.2%'), ((270, 250), 'FR', '15880', '0.4%'), ((440, 232), 'JP', '30211', '0.1%'),
    ((605, 250), 'US', '31025', '0.1%'),
    ((805, 238), 'JP', '26117', '0.1%'), ((965, 245), 'RU', '7004', '1.2%'), ((1170, 232), 'US', '25519', '0.1%'),
    ((100, 372), 'DE', '11203', '0.3%'), ((295, 402), 'JP', '29960', '0.1%'), ((520, 415), 'NL', '21378', '0.2%'),
    ((705, 450), 'IT', '17745', '0.2%'), ((820, 398), 'US', '19642', '0.1%'), ((980, 400), 'TH', '12590', '0.3%'),
    ((1185, 402), 'FR', '28866', '0.2%'),
]
CA = 10               # 加州男的下标
POP_AT = [T_ROWS[i // 7] + .03 * (i % 7) for i in range(len(PEOPLE))]
FACE = (610., 330.)   # 加州男的脸（推镜中心）
# 标签别排得太整齐（用户 11-12）：每人固定错开一点（屏幕像素，左右 ±9、上下 -10～+6）
JIT = [np.array([(L.hash01(i, 21) - .5) * 18, (L.hash01(i, 22) - .62) * 16]) for i in range(len(PEOPLE))]


def cam(t):
    """人群图 → 屏幕的仿射：(缩放, 人群图里的中心, 这个中心落在屏幕哪)。T_LOCK 起推向加州男，人放在左边，右边留给邮件。"""
    k = L.smooth((t - T_LOCK) / .30)
    z = .5 * (1 + 1.1 * k)
    c = np.array([640., 360.]) * (1 - k) + np.array(FACE) * k
    to = np.array([W / 2, H / 2]) * (1 - k) + np.array([180., H / 2 + 20]) * k
    bob = -2 * L.pulse(t, 9) if t < T_LOCK else 0          # 观众跟着拍子一颠
    if t >= T_STAMP: z *= 1 + .12 * np.exp(-(t - T_STAMP) / .07)   # 盖章那一下镜头往前猛顶再回弹
    return z, c, to + (0, bob)


def to_screen(p, t):
    z, c, to = cam(t)
    return (np.array(p, float) - c) * z + to


def scene_shini(t):
    t = max(t, T_IN)
    z, c, to = cam(t)
    m = np.array([[z, 0, to[0] - z * c[0]], [0, z, to[1] - z * c[1]]], np.float32)
    img = cv2.warpAffine(crowd(), m, (W, H), flags=cv2.INTER_AREA if z <= .5 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    # 死神视野：刚睁开时整片发红，很快退成一层红色暗角
    g = img.mean(-1, keepdims=True) / 255
    redmono = np.concatenate([40 + 215 * g, 10 + 40 * g, 20 + 40 * g], -1)
    r = .2 + .6 * np.exp(-(t - T_IN) / .09)
    img = img * (1 - r) + redmono * r
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / (W * .56)) ** 2 + ((yy - H / 2) / (H * .62)) ** 2)
    edge = np.clip((d - .78) / .22, 0, 1)[..., None]
    img = img * (1 - edge) + np.array([90, 8, 18]) * edge
    return np.clip(img, 0, 255).astype(np.uint8)


SCENES = [('shini', T_IN, T_END, scene_shini)]

# ---------------- 界面层 ----------------
# 3×5 小字（P(BAN) 那一行），16px 像素字体缩小会糊，自己画
TINY = {'P': '111101111100100', 'B': '110101110101110', 'A': '010101111101101', 'N': '110101101101101',
        '(': '010100100100010', ')': '010001001001010', '%': '101001010100101', '.': '000000000000010',
        ' ': '000000000000000',
        '0': '111101101101111', '1': '010110010010111', '2': '111001111100111', '3': '111001111001111',
        '4': '101101111001001', '5': '111100111001111', '6': '111100111101111', '7': '111001001010010',
        '8': '111101111101111', '9': '111101111001111'}


def tiny(img, x, y, s, color, sc=1):
    x, y = int(x), int(y)
    for i, ch in enumerate(s):
        g = np.array([int(b) for b in TINY.get(ch, TINY[' '])], bool).reshape(5, 3)
        if sc > 1: g = g.repeat(sc, 0).repeat(sc, 1)
        ys, xs = np.nonzero(g)
        ys, xs = ys + y, xs + x + i * 4 * sc
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        img[ys[ok], xs[ok]] = color


def box(img, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = [int(v) for v in (x0, y0, x1, y1)]
    img[max(y0, 0):max(min(y1, H), 0), max(x0, 0):max(min(x1, W), 0)] = color


def _rows(f, *cols):          # 横条旗：按行平分
    n = len(cols)
    for i, c in enumerate(cols): f[round(7 * i / n):round(7 * (i + 1) / n)] = c


def _cols(f, *cols):          # 竖条旗
    for i, c in enumerate(cols): f[:, round(11 * i / 3):round(11 * (i + 1) / 3)] = c


def _us(f):
    f[:] = PAPER; f[0::2] = RED; f[:4, :5] = BLUE


def _cn(f):
    f[:] = RED; f[1:3, 1:3] = YEL
    for p in ((0, 4), (1, 5), (3, 5), (4, 4)): f[p] = YEL


def _jp(f):
    f[:] = PAPER; f[2:5, 4:7] = RED


FLAGS = {'US': _us, 'CN': _cn, 'JP': _jp,
         'FR': lambda f: _cols(f, BLUE, PAPER, RED), 'IT': lambda f: _cols(f, GREEN, PAPER, RED),
         'DE': lambda f: _rows(f, INK, RED, YEL), 'NL': lambda f: _rows(f, RED, PAPER, BLUE),
         'RU': lambda f: _rows(f, PAPER, BLUE, RED), 'ID': lambda f: _rows(f, RED, PAPER),
         'TH': lambda f: (_rows(f, RED, PAPER, BLUE, BLUE, PAPER, RED))}


def flag(img, x, y, code, sc=1):
    """11×7 像素国旗 + 墨色边。"""
    f = np.zeros((7, 11, 3), np.uint8)
    FLAGS[code](f)
    f = np.pad(f.repeat(sc, 0).repeat(sc, 1), ((1, 1), (1, 1), (0, 0)))
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = INK
    x, y = int(x), int(y)
    h, w = f.shape[:2]
    if 0 <= x and x + w <= W and 0 <= y and y + h <= H: img[y:y + h, x:x + w] = f


def text_ol(img, x, y, s, color, scale=1):
    """带墨色描边的像素字（压在什么画面上都看得清）。"""
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
        img = L.draw_text(img, (int(x) + dx * scale, int(y) + dy * scale), s, INK, 16, scale)
    return L.draw_text(img, (int(x), int(y)), s, color, 16, scale)


def ca_state(t):
    """加州男标签此刻的样子：(国旗, 寿命, P(BAN))。扫描线扫到他之后乱码，T_FLIP 定成中国再倒数。"""
    fi = int(t * 24)
    if t < T_SCAN0 + (T_SCAN1 - T_SCAN0) * .47:
        return 'US', '31025', '0.1%'
    if t < T_FLIP:
        if L.hash01(fi, 7) < .55: return 'US', '31025', '0.1%'           # 一闪一闪地乱，别一直糊着
        code = 'CN' if L.hash01(fi, 1) < .45 else 'US'
        digits = ''.join(str(int(L.hash01(fi, i) * 10)) for i in range(5))
        return code, digits, f'{int(L.hash01(fi, 9) * 99)}.{int(L.hash01(fi, 8) * 9)}%'
    k = L.smooth((t - T_FLIP) / (T_STAMP - .03 - T_FLIP))
    life = int(round(31025 * (1 - k) ** 2))
    return 'CN', f'{life}', ('100%' if k >= 1 else f'{.1 + 99.9 * k:.1f}%')


def label(img, p, code, life, pban, sc=1, hot=False):
    x, y = p
    wl = 8 * len(life) * sc
    tot = 13 * sc + 4 + wl
    x0 = int(x - tot / 2)
    y0 = int(y - 16 * sc - 9 * sc - 2)
    flag(img, x0, y0 + 4 * sc, code, sc)
    img = text_ol(img, x0 + 13 * sc + 4, y0, life, YEL if hot else RED, sc)
    yb = y0 + 17 * sc
    s = 'P(BAN) ' + pban
    w = 4 * sc * len(s)
    box(img, x - w / 2 - 1, yb - 1, x + w / 2 + 1, yb + 5 * sc + 1, INK)
    tiny(img, x - w / 2, yb, s, RED if hot else PAPER, sc)
    return img


def crowd_overlay(img, t):
    img = img.copy()
    locked = t >= T_LOCK
    if t >= T_STAMP:                                    # 盖章后周围压暗，只剩印章和他头上的数字亮着
        img[L.bayer_mask(min(.45, (t - T_STAMP) / .1 * .45))] = INK
    for i, (p, code, life, pban) in enumerate(PEOPLE):
        if t < POP_AT[i] or i == CA: continue
        if locked and L.bayer_mask(min(1, (t - T_LOCK) / .12))[0, i * 3]: continue   # 锁定后别人的标签熄掉
        s = to_screen(p, t) + JIT[i]
        s[1] -= 6 * (1 - L.ease_out_back((t - POP_AT[i]) / .12))                    # 弹出来
        img = label(img, s, code, life, pban)
    if t >= POP_AT[CA]:
        code, life, pban = ca_state(t)
        hot = T_SCAN0 < t < T_FLIP and int(t * 24) % 2 == 0
        img = label(img, to_screen(PEOPLE[CA][0], t), code, life, pban, 2 if locked else 1, hot)
    # 扫描线：一条红竖线 + 拖尾
    if T_SCAN0 <= t < T_SCAN1 + .05:
        x = int(W * (t - T_SCAN0) / (T_SCAN1 - T_SCAN0))
        for j, w in enumerate((3, 6, 10)):
            m = np.zeros((H, W), bool); m[:, max(x - w * 4, 0):max(x, 0)] = True
            img[m & L.bayer_mask(.5 / (j + 1))] = RED
        box(img, x - 2, 18, x + 1, H, RED)
    # 漏出来的信息：推镜之后一行一行出，盖章时收掉
    rows = [txt for a, txt in LEAK if t >= a]
    if rows and t < T_STAMP:
        s = to_screen(FACE, t)
        x0, y0 = int(s[0] + 92), int(s[1] - 40)
        box(img, x0 - 4, y0 - 3, x0 + 148, y0 + 18 * len(rows) + 1, INK)
        box(img, x0 - 4, y0 - 3, x0 - 2, y0 + 18 * len(rows) + 1, RED)
        for r, txt in enumerate(rows):
            new = r == len(rows) - 1 and t - LEAK[r][0] < .12
            img = L.draw_text(img, (x0 + 2, y0 + 18 * r), txt, YEL if new else PAPER)
    # 准星：四个角往里收
    if locked:
        s = to_screen(FACE, t)
        k = L.ease_out_back((t - T_LOCK) / .14)
        r = int(150 - 80 * k)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = int(s[0] + sx * r), int(s[1] + sy * r * .9)
                box(img, min(cx, cx - sx * 18), cy - 1, max(cx, cx - sx * 18), cy + 2, RED)
                box(img, cx - 1, min(cy, cy - sy * 18), cx + 2, max(cy, cy - sy * 18), RED)
    if t >= T_STAMP:
        img = stamp(img, t, to_screen((612, 400), t))
    if T_MAIL <= t < T_MORPH[0]:                        # 训练窗口铺满前邮件一直在下面
        img = mail(img, t)
    return img


# ---------------- 老 PC 式的窗口过渡（11-12 用户：窗口闪现生硬，要有小过渡又要早期 PC 的感觉）----------------


# ---------------- 点 Request review → 弹出训练窗口 ----------------
# 11-12 用户：邮件直接变形成训练窗口（试过一级一级跳、连续拉伸）不如「点了按钮弹出一个新窗口」好懂
BUTTON = (346, 308, 472, 332)      # 邮件到位后 Request review 按钮的位置（mail() 里 X=x0+14、y0+282）


def popup(img, t):
    """T_POP 起训练窗口从按钮处连续放大到满屏（0.21 秒，先快后慢，落在 38.21 鼓上），窗口里曲线一点点画出来，身后拖两道线框残影（老 Mac 开窗口的味道），
    背景随放大压暗；38.44 边框闪黄。（试过：按鼓一级一级跳 → 卡顿；线框后瞬间出现 / 隔行扫 → 生硬）"""
    import codeshots_a as A
    full = (A.FX0, A.FY0, A.FX1, A.FY1 + 6)
    t0, dur = T_POP, T_MORPH[0] - T_POP                     # 点下去就开，放满正好落在 38.21 那下鼓
    ease = lambda tt: 1 - (1 - L.smooth(np.clip((tt - t0) / dur, 0, 1) ** .8)) ** 2
    rect = lambda k: [int(round(a + (b - a) * k)) for a, b in zip(BUTTON, full)]
    img = img.copy()
    k = ease(t)
    m = L.bayer_mask(.85 * k).copy(); m[:18] = False          # 背景压暗（顶栏不动）
    if k < .85: m[26:350, 332:636] = False                   # 邮件不跟着变网点（看着脏），窗口盖上来以后再一起暗下去
    img[m] = INK
    for j, lag in enumerate((.06, .03)):                       # 残影线框
        kj = ease(t - lag)
        if 0 < kj < .995 and k < 1:
            x0, y0, x1, y1 = rect(kj)
            col = DIMLINE[j]
            for xa, ya, xb, yb in ((x0, y0, x1, y0 + 1), (x0, y1 - 1, x1, y1), (x0, y0, x0 + 1, y1), (x1 - 1, y0, x1, y1)):
                box(img, xa, ya, xb, yb, col)
    x0, y0, x1, y1 = rect(k)
    if x1 - x0 > 4 and y1 - y0 > 4:
        box(img, x0 + 4, y0 + 4, x1 + 4, y1 + 4, INK)          # 投影
        tau = A.T2_0 + (A.T2_1 - A.T2_0) * np.clip((t - T_POP) / (T_END - T_POP), 0, 1)   # 窗口里的曲线从头画起，38.54 画完 = 第二部分首帧
        win = A._t2_window(tau)[full[1]:full[3], full[0]:full[2]]
        img[y0:y1, x0:x1] = cv2.resize(win, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA if k < .5 else cv2.INTER_NEAREST)
    if 0 <= t - T_MORPH[2] < .05:                             # 过门最后一下：边框闪黄
        for d in (2, 3):
            box(img, x0 - d, y0 - d, x1 + d, y0 - d + 1, YEL); box(img, x0 - d, y1 + d - 1, x1 + d, y1 + d, YEL)
            box(img, x0 - d, y0 - d, x0 - d + 1, y1 + d, YEL); box(img, x1 + d - 1, y0 - d, x1 + d, y1 + d, YEL)
    return img


DIMLINE = [(110, 100, 120), (200, 190, 170)]


def stamp(img, t, c):
    """11-12 用户：要更夸张、更有威慑力 → 4 倍大字 + 小字处分理由，0.06 秒从 3 倍砸下来，斑驳印泥；
    砸下那一刻：红闪、冲击波圆环 + 放射线、双向强震屏（不用 HITS 的白闪，白闪会把画面冲成粉色）。"""
    dt = t - T_STAMP
    img = img.copy()
    if dt < .1:                                                       # 红闪
        img[L.bayer_mask(.6 * (1 - dt / .1))] = RED
    if dt < .3:                                                       # 冲击波圆环 + 放射线
        u = dt / .3
        r = int(40 + 420 * (1 - (1 - u) ** 2))
        cv2.circle(img, (int(c[0]), int(c[1])), r, INK, 9)
        cv2.circle(img, (int(c[0]), int(c[1])), r, RED, 4)
        for i in range(18):
            a = i * np.pi / 9 + .17
            r0, r1 = 90 + 260 * u, 150 + 330 * u
            cv2.line(img, (int(c[0] + r0 * np.cos(a)), int(c[1] + r0 * np.sin(a))),
                     (int(c[0] + r1 * np.cos(a)), int(c[1] + r1 * np.sin(a))), PAPER if i % 2 else RED, 3)
    st = stamp_mask()
    z = 1 + 2.0 * (1 - L.smooth(dt / .06))                            # 从 3 倍砸到 1 倍
    rot = cv2.getRotationMatrix2D((st.shape[1] / 2, st.shape[0] / 2), 11, z * .62)
    rot[:, 2] += c[0] - st.shape[1] / 2, c[1] - st.shape[0] / 2
    big = cv2.warpAffine(st, rot, (W, H), flags=cv2.INTER_NEAREST) > 0
    img[cv2.dilate(big.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = INK
    img[big] = RED
    if dt < .3:                                                       # 双向强震屏
        k = 1 - dt / .3
        img = np.roll(img, (int(7 * k * np.sin(dt * 95)), int(10 * k * np.cos(dt * 80))), axis=(0, 1))
    return img


@functools.lru_cache(maxsize=None)
def stamp_mask():
    """印章（未旋转）：双线框 + SUSPENDED 4 倍 + UNSUPPORTED REGION，字面按固定种子打一些空点像旧印泥。"""
    word = np.array(L.draw_text(np.zeros((24, 8 * 9 + 6, 3), np.uint8), (3, 2), 'SUSPENDED', (255,) * 3))[..., 0] > 0
    word = word.repeat(4, 0).repeat(4, 1)
    sub = np.array(L.draw_text(np.zeros((22, 8 * 18 + 6, 3), np.uint8), (3, 2), 'UNSUPPORTED REGION', (255,) * 3))[..., 0] > 0
    sub = sub.repeat(2, 0).repeat(2, 1)
    w = max(word.shape[1], sub.shape[1]) + 40
    h = word.shape[0] + sub.shape[0] + 34
    st = np.zeros((h, w), np.uint8)
    st[14:14 + word.shape[0], (w - word.shape[1]) // 2:(w + word.shape[1]) // 2] = word
    y = 14 + word.shape[0] + 2
    st[y:y + sub.shape[0], (w - sub.shape[1]) // 2:(w - sub.shape[1]) // 2 + sub.shape[1]] = sub
    cv2.rectangle(st, (2, 2), (w - 3, h - 3), 1, 5)
    cv2.rectangle(st, (12, 9), (w - 13, h - 10), 1, 2)
    holes = np.random.default_rng(7).random((h // 3 + 1, w // 3 + 1)) < .14      # 斑驳：3×3 一格随机空掉
    st[np.kron(holes, np.ones((3, 3), bool))[:h, :w]] = 0
    return st


# ---------------- 封号邮件（仿 A 社封号页的排版：锁、大标题、粗体说明、编号步骤框、黑按钮） ----------------
IVORY, GRAY, LINE, CHIP = (248, 245, 238), (118, 110, 100), (222, 215, 204), (236, 230, 220)


def bold(img, x, y, s, color, scale=1):
    img = L.draw_text(img, (x, y), s, color, 16, scale)
    return L.draw_text(img, (x + 1, y), s, color, 16, scale)


def lock_icon(img, x, y, c=INK):
    box(img, x, y + 7, x + 14, y + 8, c); box(img, x, y + 18, x + 14, y + 19, c)      # 锁身
    box(img, x, y + 7, x + 1, y + 19, c); box(img, x + 13, y + 7, x + 14, y + 19, c)
    box(img, x + 3, y, x + 11, y + 1, c); box(img, x + 3, y, x + 4, y + 7, c); box(img, x + 10, y, x + 11, y + 7, c)  # 锁梁
    box(img, x + 6, y + 11, x + 8, y + 13, c); box(img, x + 7, y + 13, x + 8, y + 16, c)                     # 锁孔


def mail(img, t):
    """T_MAIL 起从右边滑进来带一点回弹（中间试过线框 + 隔行扫出，退回第一版的滑入）。"""
    k = L.ease_out_back((t - T_MAIL) / .14, 1.2)
    dx = int((1 - k) * 320)
    panel = mail_panel(img.copy(), t)
    out = img.copy()
    x0, y0, x1, y1 = 332, 26, 636, 350                       # 含投影
    src = panel[y0:y1, x0:x1]
    nx0 = x0 + dx
    w = min(x1, W - dx) - x0
    if w > 0: out[y0:y1, nx0:nx0 + w] = src[:, :w]
    return out


def mail_panel(img, t):
    x0, y0 = 332, 26
    x1, y1 = x0 + 298, y0 + 318
    box(img, x0 + 4, y0 + 4, x1 + 4, y1 + 4, INK)                  # 投影
    box(img, x0, y0, x1, y1, IVORY)
    X = x0 + 14
    lock_icon(img, X, y0 + 12)
    img = L.draw_text(img, (X, y0 + 36), 'Your account has', INK, 16, 2)
    img = L.draw_text(img, (X, y0 + 68), 'been suspended', INK, 16, 2)
    img = bold(img, X, y0 + 108, 'We\'ve paused your access', INK)
    img = bold(img, X, y0 + 125, 'due to unusual activity.', INK)
    img = L.draw_text(img, (X, y0 + 142), 'Your chats and data are safe.', GRAY)
    img = L.draw_text(img, (X, y0 + 164), 'What happens next:', GRAY)
    bx0, by0 = X, y0 + 184
    box(img, bx0, by0, x1 - 14, by0 + 86, LINE)
    box(img, bx0 + 1, by0 + 1, x1 - 15, by0 + 85, (255, 254, 250))
    for i, s in enumerate(('Request a review', 'We review your account', 'We email you the result')):
        yy = by0 + 7 + i * 26
        box(img, bx0 + 8, yy + 1, bx0 + 24, yy + 17, CHIP)
        img = L.draw_text(img, (bx0 + 12, yy + 1), str(i + 1), GRAY)
        img = bold(img, bx0 + 32, yy + 1, s, INK)
    pressed = 0 <= t - T_CLICK < .12
    box(img, X, y0 + 282 + pressed, X + 126, y0 + 306 + pressed, (60, 56, 52) if pressed else INK)
    img = bold(img, X + 8, y0 + 286 + pressed, 'Request review', PAPER)
    # 像素鼠标：从右下滑到按钮上，点一下
    u = L.smooth((t - (T_CLICK - .2)) / .18)
    if u > 0:
        cx = int(x1 + 30 + (X + 96 - x1 - 30) * u); cy = int(y1 + 30 + (y0 + 296 - y1 - 30) * u)
        img = cursor(img, cx, cy)
    dc = t - T_CLICK                                          # 点下去：从指针放出一圈像素涟漪
    if 0 <= dc < .18:
        r = int(4 + 40 * L.smooth(dc / .18))
        cv2.circle(img, (X + 96, y0 + 296), r, YEL if dc < .09 else PAPER, 2)
    return img


CURSOR = ['X.........', 'XX........', 'XWX.......', 'XWWX......', 'XWWWX.....', 'XWWWWX....', 'XWWWWWX...',
          'XWWWWWWX..', 'XWWWWWWWX.', 'XWWWWXXXXX', 'XWWXWX....', 'XWX.XWX...', 'XX..XWX...', 'X....XWX..', '.....XX...']


def cursor(img, x, y):
    for r, row in enumerate(CURSOR):
        for c, ch in enumerate(row):
            if ch != '.' and 0 <= y + r < H and 0 <= x + c < W:
                img[y + r, x + c] = INK if ch == 'X' else PAPER
    return img


def overlay(img, t):
    if T_POP <= t < T_END: return popup(crowd_overlay(img, t), t)
    if T_IN <= t < T_END: return crowd_overlay(img, t)
    return img


def preview(times, out):
    """量化 + overlay 一起出的联系表（part1 出片前自检用）。"""
    from PIL import Image
    import pc98ify as P
    q = mv.Quantized(scene_shini, T_IN, T_END, np.array([(230, 40, 50), (120, 14, 24)], np.uint8))
    ims = [P.upscale(overlay(q.at(t), t)) for t in times]
    ims += [np.zeros_like(ims[0])] * (len(ims) % 2)
    rows = [np.concatenate(ims[r:r + 2], 1) for r in range(0, len(ims), 2)]
    Image.fromarray(np.concatenate(rows, 0)).resize((1920, 540 * len(rows))).save(out)
    print('wrote', out)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == '--preview':
        preview([float(x) for x in sys.argv[2].split(',')], sys.argv[3])
    else:
        L.sheet_cli(SCENES)
