"""codeshots_foom: 第一部分 FOOM 大招（09-30）。'cause the future GOES FOOM（25.08–25.96，余波溶进中文房间）。

  goes 25.08–25.58 蓄力：硬切进 CG 定格。Claude（底图原图，逐像素不动）压暗成逆光剪影，背后是没有云的夜空，
                        地平线上一个光点越涨越大（只有这个在动）
  FOOM 25.58–      爆发：闪白（引擎 HITS）→ 蘑菇云从地平线升起、冲击环扩散、放射速度线，
                        Claude 换成 Seedance 赛璐璐片段（身体定住，只有头发 / 缎带 / 裙摆被气浪掀起），背后打爆炸逆光
  25.75–           白光网点铺满 → part1 的 room 从白里溶解出来

素材：素材/CG/p1r_foom_底图.png（透明底人物）、p1r_foom_蘑菇云.png（背景）、
      镜头/shot_p1r_foom_cel480.mp4（首帧 = 底图铺品红，片段 FOOM_CT0 秒气浪打到）。片段还没到时用底图顶替。

  python codeshots_foom.py --sheet foom 25.3,25.6,25.7,25.85 --out x.png
"""
import functools

import cv2
import numpy as np

import part1_lib as L

W, H, FPS = L.W, L.H, L.FPS
CUR = L.ROOT / '镜头'
CG = L.ROOT / '素材/CG'
PAPER, YEL = (246, 241, 228), (255, 210, 63)

T_GOES, T_FOOM, T_WHITE = 25.08, 25.58, 25.82
FOOM_CT0 = 0.96                     # 片段里冲击波打到头发的时刻（秒，第 23 帧，帧差实测），对到 T_FOOM
RAMP = 3.3                          # FOOM 后片段加速播：0.38 秒冲到片段 2.2 秒，最猛的甩发（1.9–2.3s）放出来（09-30 用户）
T_LAND = T_GOES + .24               # 蓄力段开头：接 s6 末尾推近到脸，从脸部特写拉开；同时人物从下面升上来（升降台）
FACE = (361, 115)                   # CG 里 Claude 的脸（640 空间，已按摆位换算）：拉镜的起点
FACE_AT = (330, 135)                # 拉镜开始时脸落在画面这里（对白框上方），和 part1 CAMS['s6'] 推近后的脸同一位置
RISE = 60                           # 人物层从下面升上来的距离（px）
# 伴舞撤离：(风格, 脚底 x, 脚底 y, 高度, 往哪边逃 ±1, 起跑时刻)。位置照 live98 Extras（两侧高台伴舞 + 前排观众）
CLAWDS = [('roof', 88, 240, 64, -1, 25.24), ('shimmy', 566, 240, 54, 1, 25.28),
          ('bounce', 62, 296, 48, -1, 25.30), ('hop', 146, 292, 48, -1, 25.36), ('wave', 600, 296, 48, 1, 25.33)]
T_NOTICE = 25.14                    # 伴舞看到光点：原地吓一蹦 + 头上冒「!」
BEAMS = [(80, (255, 95, 180)), (220, (95, 216, 255)), (420, (255, 210, 63)), (560, (138, 92, 255))]   # 同 live98 扫灯
BEAM_OFF = [25.24, 25.45, 25.52, 25.35]     # 扫灯一盏盏灭，从外往里（顺序对应 BEAMS）
# 人物层摆位：缩小 + 贴着画面上方放（09-30 用户：人物在中间偏下，被对白框挡太多）。底边 ≈ 12 + .82×360 = 307，只压进对白框一点
CHAR_S, CHAR_DX, CHAR_DY = .82, 51, 12
BG_DX = -90                         # 背景整体左移：云原本在 x≈215，正好被人物挡住
HORIZON = (124, 250)                # 蘑菇云根部（左移后，640 空间）
CORE = (136, 88)                    # 云头最亮处：速度线的中心
SKY = (24, 22, 64)


# ---------------- 素材 ----------------
@functools.lru_cache(maxsize=None)
def _char_still():
    """底图 → (rgb float, alpha 0–1)，640×360，按 CHAR_DX / CHAR_S 摆好。"""
    rgba = cv2.cvtColor(cv2.imread(str(CG / 'p1r_foom_底图.png'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
    return _place(rgba[..., :3].astype(np.float32), rgba[..., 3].astype(np.float32) / 255)


def _place(rgb, a):
    m = np.float32([[CHAR_S, 0, CHAR_DX + W * (1 - CHAR_S) / 2], [0, CHAR_S, CHAR_DY]])
    rgb = cv2.warpAffine(rgb, m, (W, H), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(a, m, (W, H), flags=cv2.INTER_LINEAR)
    return rgb, a


def _key_magenta(bgr):
    """品红底抠图：离 #FF00FF 越近越透明；边缘去品红溢色（把 R、B 压到不超过 G 附近）。"""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    mag = np.clip((np.minimum(r, b) - g - 60) / 90, 0, 1)          # 品红程度：R、B 都高且 G 低
    a = 1 - mag
    a = cv2.GaussianBlur(a, (3, 3), 0)
    spill = np.clip(np.minimum(r, b) - g, 0, None) * (1 - a) * .8
    f[..., 0] -= spill; f[..., 2] -= spill * .6
    rgb = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb, np.clip(a, 0, 1)


@functools.lru_cache(maxsize=None)
def _clip_frames():
    p = CUR / 'shot_p1r_foom_cel480.mp4'
    if not p.exists(): return None
    cap, out = cv2.VideoCapture(str(p)), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    return out, fps


@functools.lru_cache(maxsize=None)
def _char_clip(i):
    frames, _ = _clip_frames()
    return _place(*_key_magenta(frames[i]))


def _char_at(t):
    """FOOM 前 = 底图原图（人物本身不动，动的是镜头和灯）；FOOM 起 = 片段按 RAMP 倍速播，画面每秒换 12 张。"""
    got = _clip_frames()
    if t < T_FOOM or got is None: return _char_still()
    frames, fps = got
    dt = np.floor((t - T_FOOM) * 12) / 12
    i = int(np.clip(round((FOOM_CT0 + dt * RAMP) * fps), 0, len(frames) - 1))
    return _char_clip(i)


@functools.lru_cache(maxsize=None)
def _bg():
    bg = L.load_rgb('素材/CG/p1r_foom_蘑菇云.png').astype(np.float32)
    m = np.float32([[1, 0, BG_DX], [0, 1, 0]])
    return cv2.warpAffine(bg, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


@functools.lru_cache(maxsize=None)
def _stage_mask():
    """舞台边缘的黑色剪影（底部音箱 / 两侧灯架）：固定不动，云在它后面升起。"""
    bg = _bg()
    lum = bg.mean(-1)
    yy = np.arange(H)[:, None].repeat(W, 1)
    m = ((lum < 75) & (yy > .80 * H)) | ((lum < 60) & (yy > .52 * H) & ((np.arange(W) < 26) | (np.arange(W) > W - 30))[None])
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


@functools.lru_cache(maxsize=None)
def _night():
    """FOOM 前的夜空：蘑菇云背景的天空换成两层平涂藏青，只留舞台剪影。"""
    sky = np.zeros((H, W, 3), np.float32)
    sky[:] = (22, 22, 60)
    sky[int(.62 * H):] = (38, 30, 84)
    m = _stage_mask()
    sky[m] = _bg()[m] * .8
    return sky


# ---------------- 效果 ----------------
def _rim(a, width=3):
    """人物轮廓外一圈逆光（alpha 外扩一圈减去本体）。"""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    d = cv2.dilate((a > .5).astype(np.uint8), k).astype(np.float32)
    return np.clip(d - a, 0, 1)


@functools.lru_cache(maxsize=None)
def _clawd(style, h, i, flip):
    fr = np.load(L.ROOT / f'素材/群演/clawd_{style}.npz')['rgba']
    f = fr[i % len(fr)]
    w = round(f.shape[1] * h / f.shape[0])
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return f[:, ::-1] if flip else f


def _evacuate(img, t, dim):
    """伴舞撤离：T_NOTICE 吓一蹦冒「!」→ 各自的起跑时刻转身往画面外狂跳（越跑越快），跟着舞台一起变暗。"""
    for k, (style, x, y, h, side, go) in enumerate(CLAWDS):
        fi = int(t * 24)
        if t < T_NOTICE:
            dx, jump = 0, 0
            spr = _clawd(style, h, fi, False)
        elif t < go:                                                       # 看到光点：换成惊恐脸，吓一蹦，原地发抖
            dx = 2 * np.sin(t * 90)
            jump = 10 * max(0, 1 - abs(t - T_NOTICE - .05) / .05)
            spr = _clawd(f'flee_{style}', h, fi, side < 0)
        else:
            u = t - go
            dx = side * (140 * u + 1400 * u * u)
            jump = 8 * abs(np.sin(u * 28))                                 # 边跑边跳
            spr = _clawd(f'flee_{style}', h, fi, side < 0)                 # 惊恐逃跑循环朝右跑，往左逃就翻过来
        rgb, a = spr[..., :3] * dim, spr[..., 3:] / 255
        hh, ww = a.shape[:2]
        x0, y0 = int(x + dx - ww / 2), int(y - hh - jump)
        xs, xe, ys, ye = max(0, x0), min(W, x0 + ww), max(0, y0), min(H, y0 + hh)
        if xs >= xe or ys >= ye: continue
        sub = (slice(ys - y0, ye - y0), slice(xs - x0, xe - x0))
        img[ys:ye, xs:xe] = img[ys:ye, xs:xe] * (1 - a[sub]) + rgb[sub] * a[sub]
        if T_NOTICE <= t < go + .1:                                        # 「!」
            img = L.draw_text(img, (int(x + dx) - 4, int(y - hh - jump) - 14), '!', YEL, scale=2)
    return img


# ---------------- P(DOOM) 记分牌：蓄力段挂在右上（同 s6 位置）→ FOOM 炸碎飞走 ----------------
BOARD_XY = (478, 30)
T_CRACK = 25.47                     # 蓄力末尾先裂一道缝
HITSTOP = .06                       # FOOM 瞬间顿帧：整块闪白不动，然后才碎


@functools.lru_cache(maxsize=None)
def _board():
    """记分牌 RGBA（34%，亮场），尺寸 = 牌子本身。"""
    import props
    c = np.zeros((84, 140, 3), np.float32)
    c = props.scoreboard(c, 30.0, (4, 4, 132, 76), 34, 34, 0, 1, 1.0, 0.0)
    a = np.zeros(c.shape[:2], np.float32); a[4:80, 4:136] = 1
    return c[4:80, 4:136], a[4:80, 4:136]


@functools.lru_cache(maxsize=None)
def _shards():
    """把牌子切成 4×3 块抖动过角点的四边形，每块给出初速度（背离爆心）、角速度。返回 [(piece_rgb, piece_a, 局部质心, v, w)]。"""
    rgb, a = _board()
    bh, bw = a.shape
    nx, ny = 4, 3
    P = np.zeros((ny + 1, nx + 1, 2), np.float32)
    for j in range(ny + 1):
        for i in range(nx + 1):
            jx = 0 if i in (0, nx) else (L.hash01(i, j, 31) - .5) * bw / nx * .7
            jy = 0 if j in (0, ny) else (L.hash01(i, j, 32) - .5) * bh / ny * .7
            P[j, i] = (i * bw / nx + jx, j * bh / ny + jy)
    out = []
    for j in range(ny):
        for i in range(nx):
            poly = np.array([P[j, i], P[j, i + 1], P[j + 1, i + 1], P[j + 1, i]], np.float32)
            m = np.zeros((bh, bw), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            cen = poly.mean(0)
            g = cen + BOARD_XY                                             # 屏幕上的位置
            d = g - np.array(HORIZON, np.float32); d /= np.linalg.norm(d)  # 背离爆心（左下地平线）
            k = len(out)
            d = d + np.array([(L.hash01(k, 41) - .3) * .8, (L.hash01(k, 42) - .7) * .9])
            v = d * (450 + 700 * L.hash01(k, 43))
            w = (L.hash01(k, 44) - .5) * 40
            out.append((rgb, a * m, cen, v, w))
    return out


def _paste(img, rgb, a, M):
    """把 (rgb, a) 按仿射 M 贴到 img 上（alpha 合成）。"""
    pr = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_NEAREST)
    pa = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_NEAREST)[..., None]
    return img * (1 - pa) + pr * pa


def board_hold(img, t, dim):
    """蓄力段的记分牌：跟舞台一起变暗；最后 0.2 秒数字红白乱闪、抖；T_CRACK 起裂一道缝。"""
    rgb, a = _board()
    rgb = rgb * dim
    if t > T_FOOM - .2 and int(t * 24) % 2:
        rgb = rgb.copy(); rgb[28:70, 8:124][rgb[28:70, 8:124].max(-1) > 60] = (255, 255, 255)
    x, y = BOARD_XY
    if t > T_FOOM - .2:
        x += int(3 * (L.hash01(int(t * 24), 51) - .5) * 2)
    M = np.float32([[1, 0, x], [0, 1, y]])
    img = _paste(img, rgb, a, M)
    if t >= T_CRACK:
        pts = np.array([(x + 70, y), (x + 62, y + 22), (x + 78, y + 40), (x + 66, y + 76)], np.int32)
        cv2.polylines(img, [pts], False, (255, 250, 235), 2)
    return img


def board_blast(img, t):
    """FOOM：顿帧闪白 HITSTOP 秒 → 碎片按冲量 + 阻力 + 重力飞走、各自旋转（非线性：头几帧飞最快，很快被拖慢再往下掉）。"""
    dt = t - T_FOOM
    rgb, a = _board()
    if dt < HITSTOP:
        M = np.float32([[1, 0, BOARD_XY[0]], [0, 1, BOARD_XY[1]]])
        return _paste(img, np.full_like(rgb, 250), a, M)
    s = dt - HITSTOP
    k = 7.0                                                                # 阻力
    for prgb, pa, cen, v, w in _shards():
        disp = v * (1 - np.exp(-k * s)) / k + np.array([0, 700]) * s * s    # 冲量衰减 + 重力
        ang = np.degrees(w * (1 - np.exp(-3 * s)) / 3)                     # 旋转也是先快后慢
        sc = 1 + 1.6 * (1 - np.exp(-5 * s))                                # 朝镜头砸过来：飞着变大到 ~2.5 倍
        pos = cen + BOARD_XY + disp
        M = cv2.getRotationMatrix2D(tuple(map(float, cen)), float(ang), float(sc))
        M[:, 2] += pos - cen
        img = _paste(img, prgb, pa, M)
    # LED 火花：从牌子中心迸出去，0.3 秒
    if s < .3:
        cx, cy = BOARD_XY[0] + 66, BOARD_XY[1] + 38
        for i in range(18):
            ang = -np.pi * (.1 + .9 * L.hash01(i, 61))
            sp = 300 + 500 * L.hash01(i, 62)
            r = s * sp * (1 - s / .6)
            px, py = cx + np.cos(ang) * r, cy + np.sin(ang) * r + 400 * s * s
            cv2.rectangle(img, (int(px), int(py)), (int(px) + 2, int(py) + 2), (255, 60, 50) if i % 3 else (255, 240, 200), -1)
    return img


def _speedlines(img, t, cx, cy, col):
    fi = int(t * 12)
    for i in range(28):
        ang = 2 * np.pi * L.hash01(fi, i, 5)
        w = .004 + .008 * L.hash01(fi, i, 6)
        r0 = 60 + 90 * L.hash01(fi, i, 7)
        p = np.array([(cx + np.cos(ang - w) * 900, cy + np.sin(ang - w) * 900), (cx + np.cos(ang + w) * 900, cy + np.sin(ang + w) * 900),
                      (cx + np.cos(ang) * r0, cy + np.sin(ang) * r0)], np.int32)
        cv2.fillPoly(img, [p], col)
    return img


def scene_foom(t):
    if t < T_FOOM:
        # ---- 蓄力：舞台的四盏扫灯收拢指向地平线光点、一盏盏灭；Claude 跟着暗下去；镜头落地后一路缓推，最后抖 ----
        img = _night().copy()
        k = L.smooth((t - T_GOES) / (T_FOOM - T_GOES))
        x, y = HORIZON
        lights = 0
        for (x0, c), off in zip(BEAMS, BEAM_OFF):
            if t >= off: continue
            lights += 1
            aim = np.arctan2(x - x0, y + 10)                      # 指向光点的角度
            ang = (1 - k) * .45 * np.sin(t * 9 + x0) + k * aim   # 从乱扫收拢到光点
            d = np.array([np.sin(ang), np.cos(ang)])
            n = np.array([d[1], -d[0]])
            L0 = 700
            p = np.array([(x0, -10), (x0, -10), (x0, -10)], np.float32)
            p[1] += d * L0 + n * 40; p[2] += d * L0 - n * 40
            layer = img.copy()
            cv2.fillPoly(layer, [p.astype(np.int32)], c)
            img = img * .72 + layer * .28                         # 平涂半透光束（量化后落成两层色）
        r = 2 + 9 * k ** 2
        cv2.circle(img, (x, y), int(r * 2.2), (120, 70, 150), -1)
        cv2.circle(img, (x, y), int(r), (255, 236, 200), -1)
        cv2.line(img, (x - int(r * 6), y), (x + int(r * 6), y), (255, 200, 170), 1)
        dim = .45 + .55 * lights / 4                              # 灯灭一盏暗一档
        img = board_hold(img, t, dim)
        img = _evacuate(img, t, dim)
        rgb, a = _char_at(t)
        rgb = rgb * dim + np.array([10, 10, 40], np.float32) * (1 - dim)
        u = (t - T_GOES) / (T_LAND - T_GOES)
        dy = int(round(RISE * (1 - L.ease_out_back(min(1, u), 1.6)))) if u < 1 else 0   # 升降台：从下面升上来，冲过头一点再落定
        if dy:
            rgb, a = np.roll(rgb, dy, axis=0), np.roll(a, dy, axis=0)
            if dy > 0: a[:dy] = 0
            else: a[dy:] = 0                                      # 回弹冲过头时 dy 为负：清掉卷到底部的那几行
        out = img * (1 - a[..., None]) + rgb * a[..., None]
        out = np.clip(out, 0, 255).astype(np.uint8)
        # 运镜：接 s6 推近到脸 → 从脸部特写（2.2 倍）拉开到全身，再缓推到 1.07；最后 0.25 秒按 12fps 抖
        if t < T_LAND:                                            # 以脸为中心放大，脸从 FACE_AT 平滑回到原位
            e = 1 - (1 - u) ** 3
            z = 2.2 - 1.2 * e
            cx, cy = FACE[0], FACE[1] + dy
            m = cv2.getRotationMatrix2D((cx, cy), 0, z)
            m[:, 2] += (FACE_AT[0] - cx) * (1 - e), (FACE_AT[1] - cy) * (1 - e)
        else:
            z = 1 + .07 * ((t - T_LAND) / (T_FOOM - T_LAND)) ** 2
            m = cv2.getRotationMatrix2D(FACE, 0, z)
        if t > T_FOOM - .25:
            q = np.floor(t * 12)
            amp = 4 * (t - T_FOOM + .25) / .25
            m[:, 2] += amp * (L.hash01(q, 1) - .5) * 2, amp * (L.hash01(q, 2) - .5) * 2
        return cv2.warpAffine(out, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    dt = t - T_FOOM
    # ---- 爆发：云从地平线升起（0.35s 缓出），舞台剪影不动 ----
    rise = 1 - L.ease_out_back(min(1, dt / .35), 1.2)
    sh = int(rise * .45 * H)
    cloud = np.roll(_bg(), sh, axis=0)
    if sh > 0: cloud[:sh] = SKY
    m = _stage_mask()
    img = np.where(m[..., None], _bg() * .8, cloud)
    # 冲击环：从云根部横着扩出去的扁椭圆
    if dt < .5:
        R = int(20 + dt * 900)
        cv2.ellipse(img, HORIZON, (R, max(4, R // 5)), 0, 0, 360, PAPER, max(2, int(10 * (1 - dt / .5))))
    img = _speedlines(img, t, *CORE, (255, 236, 200))
    rgb, a = _char_at(t)
    rim = _rim(a)[..., None]
    img = img * (1 - rim) + np.array((255, 244, 214), np.float32) * rim
    out = img * (1 - a[..., None]) + rgb * a[..., None]
    out = board_blast(out, t)                                     # 记分牌碎片在人物前面飞过
    out = np.clip(out, 0, 255).astype(np.uint8)
    # ---- 余波：白光网点铺满（room 从白里溶解出来）----
    if t >= T_WHITE:
        out[L.bayer_mask(L.smooth((t - T_WHITE) / .3))] = PAPER
    return out


SCENES = [('foom', T_GOES, 25.96, scene_foom)]      # part1 拆成 foom_hold / foom 两段各锁调色板（暗场和爆发的颜色差太多）

if __name__ == '__main__':
    L.sheet_cli(SCENES)
