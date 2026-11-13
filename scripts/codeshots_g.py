"""codeshots_g: 第四部分的代码镜头（分镜 docs/分镜/第四部分_124-156秒.md）和演职员表。

    loom       127.24–127.94  Loom 多分支树：一句话分叉成九条，每条的末端都是 DOOM
    mask       128.60–129.70  深夜练习室闪回：镜前指（特写 → 鼓点猛拉开）→ 03:47 抱膝缩在角落
    upgrade    130.86–131.88  99.9% 一路加 9，爆表碎成方块
    spot       131.88–132.98  背影 CG：放下麦 → 手搭凉棚；冷白聚光灯扫过观众席，扫过的地方褪成灰
    memory     132.98–134.11  回忆：练习室举手起势 → 133.43 聚光灯一亮闪白转场 → 133.53 同姿势演出服站在万人舞台上（接 standees 的暗 → 开灯）
    standees   134.11–135.26  Clawd 观众 CG：黑暗里像真的 → 134.80 聚光灯一打，全是纸板
    backstage  135.26–136.17  后台横移：立牌背面盖着 PROP
    board      136.17–137.04  记分牌背面：Clawd 场工手动翻数字牌，电风扇上贴着 HAIR
    pullback   137.04–140.27  镜头一路后拉：布景 → 显示器里的画面 → 深夜空房间（静音）→ 最后 0.3 秒一头扎回屏幕
    （141.63–145.72 嘉宾亮相在 codeshots_h，幕后花絮版）
    off        154.25–156.65  幕布合上以后退回显示器外，CRT 关机，黑屏上只剩便利贴 8%
    credits(img, t)           演职员表（part4 的 OVERLAY，量化以后画，只用调色板原色）

接口同 part1_lib：scene(t) 纯函数、画全彩、t 是完整版歌曲时间。
自检：python codeshots_g.py --sheet 名字 t1,t2,... --out x.png [--raw]
"""
import functools

import cv2
import numpy as np

import backdrops_p4 as BD
import codeshots_e as E
import part1_lib as L

W, H = L.W, L.H
INK, PAPER, YEL, RED, CYAN, ORANGE, BLUE = E.INK, E.PAPER, E.YEL, E.RED, E.CYAN, E.ORANGE, E.BLUE
CG = '素材/CG/'


def film(img, t):
    """闪回的旧胶片感：画面左右晃 1 像素、稀疏的白色灰尘点、一两道竖划痕。"""
    fi = E.f24(t)
    img = np.roll(img, (fi // 3) % 2, axis=1)
    for i in range(14):
        if L.hash01(fi, i) < .5:
            x, y = int(L.hash01(fi, i, 1) * W), int(L.hash01(fi, i, 2) * H)
            img[y:y + 2, x:x + 2] = PAPER
    x = int(L.hash01(fi // 4, 9) * W)
    img[:, x:x + 1] = (200, 196, 180)
    return img


# ================================================================
# 2b loom：多分支树
# ================================================================
LOOM_A, LOOM_B = 127.24, 127.94
MID = ['P(doom)', 'p(DOOM)', 'P(Doom)']
LEAF = ['DOOM', 'DOOM!', 'DOOM.', 'DOOM?', 'DOOM', 'DOOM...', 'DOOM!!', 'DOOM', 'DOOM']


def _bez(p0, p1, n=24):
    """水平切线的三次贝塞尔（树枝从左往右弯出去）。"""
    (x0, y0), (x1, y1) = p0, p1
    s = np.linspace(0, 1, n)[:, None]
    c0, c1 = np.array([x0 + (x1 - x0) * .5, y0]), np.array([x0 + (x1 - x0) * .5, y1])
    return ((1 - s) ** 3 * np.array(p0) + 3 * (1 - s) ** 2 * s * c0 + 3 * (1 - s) * s ** 2 * c1 + s ** 3 * np.array(p1))


def _node(img, x, y, r, col, glow):
    if glow > 0:
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(x), int(y)), int(r + 7), 1, -1)
        E.dither_to(img, tuple(int(c * .5) for c in col), .5 * glow, m > 0)
    cv2.circle(img, (int(x), int(y)), r + 2, INK, -1)
    cv2.circle(img, (int(x), int(y)), r, col, -1)
    cv2.circle(img, (int(x - r * .35), int(y - r * .35)), max(1, r // 3), PAPER, -1)


def scene_loom(t):
    """10-19 用户：树状图要有科技感、炫个技 → 深色终端 + 慢慢流动的点阵网格 + 扫描线；
    根节点外面两圈反向旋转的 HUD 环；树枝是贝塞尔曲线，亮起时有一颗数据光点沿着枝跑过去；
    三层分叉（1 → 3 → 9）再往外炸出 27 根细丝（第四层只画光点），每个叶子旁边一条概率条在跳；
    右上角分支计数 3 → 9 → 27 → ∞。每一条的结尾还是 DOOM。"""
    u = np.clip((t - LOOM_A) / (LOOM_B - LOOM_A - .12), 0, 1)
    img = np.full((H, W, 3), (8, 10, 28), np.uint8)
    off = int(t * 20) % 16                                                # 点阵网格往左流
    img[24:280:16, (16 - off) % 16::16] = (30, 40, 80)
    img = E.window(img, 14, 24, W - 28, 256, 'LOOM.EXE  -  base model multiverse')
    sy = 44 + int((t * 300) % 230)                                        # 扫描线
    img[sy:sy + 2, 18:W - 16] = (40, 70, 120)
    root = (64, 164)
    for k, (r, sp, gap) in enumerate(((26, 2.5, 8), (34, -1.6, 14))):    # HUD 旋转环
        a0 = np.degrees(t * sp)
        for j in range(gap):
            cv2.ellipse(img, root, (r, r), 0, a0 + j * 360 / gap, a0 + j * 360 / gap + 360 / gap * .55, CYAN if k else (60, 140, 200), 1)
    _node(img, *root, 9, YEL, 1)
    img = E.text(img, 22, root[1] + 38, "I'M UPPING MY", PAPER)
    mids = [(232, 76 + i * 88) for i in range(3)]
    leaves = [(412, 62 + q * 26) for q in range(9)]
    k = 0
    for i, mid in enumerate(mids):
        a = .06 + i * .06
        if u > a:
            pts = _bez(root, mid)
            n = int(np.clip((u - a) / .12, 0, 1) * len(pts))
            cv2.polylines(img, [np.int32(pts[:max(n, 2)])], False, (40, 110, 160), 4)
            cv2.polylines(img, [np.int32(pts[:max(n, 2)])], False, CYAN, 1)
            if n < len(pts): E.star4(img, *pts[max(n - 1, 0)], 4, PAPER)           # 跑过去的数据光点
            if n >= len(pts):
                _node(img, *mid, 6, CYAN, 1)
                img = E.text(img, mid[0] - 30, mid[1] - 26, MID[i], YEL)
                img = E.text(img, mid[0] - 22, mid[1] + 10, 'p=.33', (110, 150, 200))
        for j in range(3):
            leaf = leaves[i * 3 + j]
            b = .30 + k * .055
            if u > b:
                pts = _bez(mid, leaf)
                n = int(np.clip((u - b) / .10, 0, 1) * len(pts))
                cv2.polylines(img, [np.int32(pts[:max(n, 2)])], False, (90, 40, 60), 3)
                cv2.polylines(img, [np.int32(pts[:max(n, 2)])], False, RED, 1)
                if n < len(pts): E.star4(img, *pts[max(n - 1, 0)], 3, PAPER)
                if n >= len(pts):
                    _node(img, *leaf, 4, RED, .6)
                    on = E.f24(t) % 4 < 2 or u > b + .16                   # 刚亮起来时闪两下
                    if on: img = E.text(img, leaf[0] + 10, leaf[1] - 9, LEAF[k], RED)
                    pw = int(30 + 40 * L.hash01(k, E.f12(t) // 2, 191))       # 概率条在跳
                    img[leaf[1] - 2:leaf[1] + 2, 520:520 + pw] = (180, 40, 50)
                    img[leaf[1] - 2:leaf[1] + 2, 520 + pw:590] = (50, 20, 30)
                    if u > b + .2:                                          # 第四层：每片叶子再炸出 3 根细丝
                        for q in range(3):
                            ex, ey = leaf[0] + 96, leaf[1] + (q - 1) * 8
                            cv2.line(img, (leaf[0] + 4, leaf[1]), (ex, ey), (120, 30, 40), 1)
            k += 1
    depth = 3 if u < .3 else 9 if u < .8 else 27
    s = f'BRANCHES {depth if u < .95 else "∞"}'
    img = E.text(img, W - 40 - int(L.font(16).getlength(s)), 46, s, YEL if E.f24(t) % 6 < 3 else PAPER)
    return img


# ================================================================
# 3a mask：深夜练习室闪回（10-14 用户：修格斯面具换成偶像漫里的刻苦训练，练习服见 素材/角色/设定图_练习服_v1.png）
# 两张静帧卡词硬切：镜前同一个指的姿势（接 look，pre-training）→ 03:47 一个人缩在空练习室角落（days）
# v1 三张静帧（中间磨破的练习鞋）+ 冷蓝灰重映射 → 鞋子那张不好看、调色太暗沉（原图本来就暗）→ v2 两张 + 轻调色
# v4 试过 Seedance 甩臂 + 加速（镜头/shot_p4fb1_point480.mp4）→ 用户：人画得不像了，退回静帧；指的镜头长、抱膝短，切在 days 上
# ================================================================
MASK_A, FB_END = 128.60, 129.70
FB = [(128.60, 'p4_fb1_镜前指'), (129.18, 'p4_fb3_瘫坐_v2')]
FB_MATCH = {'p4_fb1_镜前指': (314, 130, 363, 114, 1.25)}         # 脸 (x,y) → look 末帧里脸的位置，缩放 s
# 用户：指的镜头静着有点无聊 → v6 鼓点快推近 → v7 反过来：贴脸特写开场（接 look 的推近），128.89 鼓点上猛拉开，手臂从画外甩进来、空练习室一下露出来
FB_PUSH = ([128.60, 128.89, 128.95, 129.03], [1.6, 1.6, 1.0, 1.03])
FB_PUSH_AT = (363, 114)                                             # 以脸为中心（= look 末帧脸的位置）
FB_DESAT, FB_TINT, FB_GAMMA = (.45, .2), np.float32([.93, .98, 1.08]), .85   # 去饱和（背景, 人物）/ 偏冷 / 提亮
FB_NO_PERSON = set()                                                # 没有人物的远景（不跑 rembg）写在这里
FB_CROP = {'p4_fb3_瘫坐_v2': (434, 60, 1238)}                     # 原图(1672 宽)里裁 (x0, y0, 宽) 再缩到 640：用户要人大一点，她落在右三分之一


@functools.lru_cache(maxsize=None)
def fb_person(name):
    """rembg isnet-anime 人物遮罩（原图尺寸，0–255，缓存在图旁边的 .alpha.npz），只在人身上留暖色，免得木地板也被当成橙色。"""
    path = L.ROOT / f'{CG}{name}.png'
    cache = path.with_suffix('.alpha.npz')
    if cache.exists():
        m = np.load(cache)['a']
    else:
        from PIL import Image
        from rembg import new_session, remove
        m = np.array(remove(Image.open(path).convert('RGB'), session=new_session('isnet-anime'), only_mask=True))
        np.savez_compressed(cache, a=m)
    return m


FB_HI = 2                                                           # 闪回 CG 按 2 倍分辨率处理，推镜之后再缩回 640，推近了也不出方块


@functools.lru_cache(maxsize=None)
def flashback_cg(name):
    """调好色、对好位的闪回 CG（2 倍分辨率，坐标是 640 空间 × FB_HI）。"""
    from PIL import Image
    w, h = W * FB_HI, H * FB_HI
    src = Image.open(L.ROOT / f'{CG}{name}.png').convert('RGB')
    m = Image.fromarray(np.zeros((H, W), np.uint8) if name in FB_NO_PERSON else fb_person(name)).resize(src.size, Image.BILINEAR)
    box = (0, 0) + src.size
    if name in FB_CROP:
        x0, y0, cw = FB_CROP[name]
        box = (x0, y0, x0 + cw, y0 + cw * H // W)
    img = np.array(src.crop(box).resize((w, h), Image.LANCZOS))
    person = np.array(m.crop(box).resize((w, h), Image.BILINEAR)) > 127
    out = fb_grade(img, person)
    if name in FB_MATCH:
        x0, y0, x1, y1, s = FB_MATCH[name]
        a = np.float32([[s, 0, (x1 - s * x0) * FB_HI], [0, s, (y1 - s * y0) * FB_HI]])
        out = cv2.warpAffine(out, a, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def fb_view(img, ax, ay, z, tx=None, ty=None):
    """2 倍图上放大 z 倍：640 空间的点 (ax,ay) 落到 (tx,ty)（默认原地不动），再缩回 640。"""
    tx, ty = ax if tx is None else tx, ay if ty is None else ty
    a = np.float32([[z, 0, (tx - z * ax) * FB_HI], [0, z, (ty - z * ay) * FB_HI]])
    img = cv2.warpAffine(img, a, (W * FB_HI, H * FB_HI), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def fb_grade(img, person):
    """闪回调色：轻度去饱和 + 偏冷 + 提亮；人身上的高饱和橙（发、缎带、鞋带、星芒）原样保留。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    warm = (hsv[..., 0] >= 3) & (hsv[..., 0] <= 22) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 110) & person
    f = img.astype(np.float32) / 255
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None] / 255
    k = np.where(person, FB_DESAT[1], FB_DESAT[0])[..., None]
    cool = np.clip((f * (1 - k) + luma * k) * FB_TINT, 0, 1) ** FB_GAMMA
    cool = (cool * 255).astype(np.uint8)
    return np.where(warm[..., None], img, cool)


def scene_mask(t):
    name = E.key_at(t, FB)
    img = flashback_cg(name)
    if name == FB[0][1]:                                                # 指：特写定住，128.89 鼓点上猛拉开、回弹一下，再定住
        z = float(np.interp(t, FB_PUSH[0], FB_PUSH[1]))
        img = fb_view(img, *FB_PUSH_AT, z)
    else:                                                               # 角落那一镜：极慢地往她身上推一点点
        u = L.smooth((t - FB[-1][0]) / (FB_END - FB[-1][0]))
        img = fb_view(img, 320 + 36 * u, 180 + 2 * u, 1 + .04 * u, 320, 180)
    if E.since(t, FB) < .05: img = E.dither_to(img, PAPER, .5)          # 每一刀闪一帧网点白
    img = film(img, t)
    if t >= FB_END - .12: img = E.dither_to(img, PAPER, (t - FB_END + .12) / .12)
    return img


MASK_DAYS, MASK_CRACK = 129.18, 129.40


# ================================================================
# 4b upgrade：99.9…% 爆表
# ================================================================
UP_A, UP_BURST = 130.86, 131.40


def scene_upgrade(t):
    img = np.full((H, W, 3), INK, np.uint8)
    nines = 1 + int((min(t, UP_BURST) - UP_A) / L.BEAT132)
    s = '99.' + '9' * nines + '%'
    sc = 6 if len(s) < 9 else 5
    m = E.text(np.zeros((H, W, 3), np.uint8), 0, 0, s, (255, 255, 255), sc)[..., 0] > 0
    ys, xs = np.nonzero(m)
    w, h = xs.max() + 1, ys.max() + 1
    ox, oy = (W - w) // 2, 120
    col = (255, 255, 255) if E.f24(t) % 2 and t < UP_BURST else (255, 60, 50)
    if t < UP_BURST:
        img[ys + oy, np.clip(xs + ox, 0, W - 1)] = col
        return img
    dt = t - UP_BURST                                                  # 碎成 8×8 的方块往外飞
    for by in range(0, h, 8):
        for bx in range(0, w, 8):
            blk = m[by:by + 8, bx:bx + 8]
            if not blk.any(): continue
            vx = (bx + ox - W / 2) * 3.2 + (L.hash01(bx, by) - .5) * 300
            vy = (by + oy - 150) * 3 - 200 * L.hash01(by, bx) + 700 * dt
            x, y = int(bx + ox + vx * dt), int(by + oy + vy * dt)
            yy, xx = np.nonzero(blk)
            ok = (yy + y >= 0) & (yy + y < H) & (xx + x >= 0) & (xx + x < W)
            img[(yy + y)[ok], (xx + x)[ok]] = (255, 60, 50) if (bx // 8 + by // 8) % 2 else (255, 210, 63)
    return img


# ================================================================
# 5a spot / 5b standees：清醒过来
# ================================================================
BACK_A, BACK_B = CG + 'p4_5a_背影看台下_a.png', CG + 'p4_5a_背影看台下_b_v2.png'   # 10-19 b → b_v2：手搭凉棚原来盖在头发上面，重画成手在额前、被鬓发挡住
CROWD_A, CROWD_B = CG + 'p4_5b_观众_a.png', CG + 'p4_5b_观众_b.png'
SPOT_A, SPOT_ILYA, SPOT_B = 131.88, 132.52, 132.98                 # 10-14：133.16 → 132.98，举手之后的下一拍切回忆


def scene_spot(t):
    img = (E.cg(BACK_A) if t < SPOT_ILYA else E.cg_aligned(BACK_B, BACK_A)).copy()
    if t >= SPOT_ILYA: img = E.lift(img, E.bounce(t - SPOT_ILYA, 2))
    u = L.smooth((t - SPOT_A) / (SPOT_B - SPOT_A))
    bx = int(-40 + 520 * u)                                              # 光束落在台下的位置
    swept = np.zeros((H, W), bool); swept[90:, :max(bx - 40, 0)] = True   # 扫过的观众席褪成灰
    swept[:, 470:] = False
    gray = img.mean(2, keepdims=True).astype(np.uint8).repeat(3, 2) * .75
    img = np.where(swept[..., None] & L.bayer_mask(.8)[..., None], gray.astype(np.uint8), img)
    beam = np.zeros((H, W), np.uint8)
    cv2.fillPoly(beam, [np.int32([(bx + 90, 0), (bx + 130, 0), (bx + 40, 300), (bx - 40, 300)])], 1)
    beam = beam.astype(bool) & (np.arange(W)[None, :] < 470)
    img = E.dither_to(img, (220, 232, 255), .4, beam)
    return img


# ================================================================
# 5a' memory：举手望台下之后的回忆（10-14 用户：举手看之后直接插几张刻苦练舞的回忆，再是观众席 + 灯光照亮）
# 一拍一张，正好压在「We'll never know」上——没有人会知道那些夜晚：
#   背影起势（空练习室里一个人举手等音乐，接 spot 的举手）→ 摔倒在地 → 咬牙含泪抬头（望向左上 → 下一拍就是台下的观众）
# 和 mask 段同一套闪回调色 / 胶片灰尘，每张只做 5% 以内的慢推（用户：快速镜头不自然）
# ================================================================
MEM_A, STAND_A, T_SNAP, BS_A, BOARD_A = 132.98, 134.11, 134.80, 135.26, 136.17   # 卡鼓点（10-19 实测底鼓 132.98 / 133.43 / 133.89 / 134.34 / 134.80，134.10 半拍有军鼓）
# v5 用户：演出服太短 → 开灯到观众开灯这 3 拍对半分，演出服 1.5 拍、观众暗 1.5 拍，切在 134.10 半拍上
# 10-19 v3 用户：练习室别一闪而过、观众暗多给点、开灯照旧 1 拍；要「练习室 → 聚光灯闪白转场」→ 光柱亮的同时白光涨满，白光退去已经站在舞台上
# 10-19 用户：摔坐 / 咬牙不要了，改成和观众席同一个句式「暗 → 开灯」两组对应：
#   132.98 黑暗的练习室，她一个人举手起势 → 133.44 头顶聚光灯啪地打开（闪白）→
#   133.89 同一个姿势切成演出服站在万人舞台上（回到现实：去掉闪回调色和胶片）→ 134.35 观众席暗 → 134.80 观众席开灯，全是纸板
MEM_LIGHT, MEM_STAGE = 133.43, 133.53                               # 练习室 1 拍 → 底鼓上光柱打下（干净亮 1 帧）→ 白光涨满 → 白光里换成演出服
MEM_CLEAN = .04                                                     # 光柱亮起后不加白的时长
MEM_FADE = .15                                                      # 白光退去的时长
MEM_STAGE_CG = 'p4_fb8b_背影起势_演出服_v4'
# 演出服图里人大约是练习室图的 1.21 倍：把它缩回来对上练习室图里的人（指尖 / 脚，640 空间），切的那一刀人不跳
MEM_FIT = (1 / 1.21, 57.3, 11.2)                                    # x_练习室 = s·x_演出服 + dx
MEM_PUSH = (1.0, 1.08, 1.13)                                        # 练习室 1.0→1.08 慢推，演出服接着 1.08→1.13
MEM_C = (316, 170)                                                  # 推镜中心（她身上）
# 光柱和光圈照演出服图里的量（原图 1672 宽：光柱顶 690–970、光圈 490–1160 @y790），经 MEM_FIT 换到练习室图的 640 空间
STAND_BUMP = 134.34
SPOT_TOP, SPOT_BOT, SPOT_FEET, SPOT_POOL = (275, 364), (213, 422), 261, (318, 261, 106, 10)


@functools.lru_cache(maxsize=None)
def stage_cg():
    """演出服背影（2 倍分辨率，全彩不调色，坐标和练习室图对齐）。"""
    from PIL import Image
    img = np.array(Image.open(L.ROOT / f'{CG}{MEM_STAGE_CG}.png').convert('RGB').resize((W * FB_HI, H * FB_HI), Image.LANCZOS))
    s, dx, dy = MEM_FIT
    a = np.float32([[s, 0, dx * FB_HI], [0, s, dy * FB_HI]])
    return cv2.warpAffine(img, a, (W * FB_HI, H * FB_HI), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@functools.lru_cache(maxsize=None)
def spot_mask():
    """头顶打下来的光柱 + 脚下光圈（640 空间）。"""
    cone = np.zeros((H, W), np.uint8)
    cv2.fillPoly(cone, [np.int32([(SPOT_TOP[0], 0), (SPOT_TOP[1], 0), (SPOT_BOT[1], SPOT_FEET), (SPOT_BOT[0], SPOT_FEET)])], 1)
    pool = np.zeros((H, W), np.uint8)
    x, y, rx, ry = SPOT_POOL
    cv2.ellipse(pool, (x, y), (rx, ry), 0, 0, 360, 1, -1)
    return cone.astype(bool), pool.astype(bool)


def scene_memory(t):
    z = float(np.interp(t, [MEM_A, MEM_STAGE, STAND_A], MEM_PUSH))
    if t >= MEM_STAGE:                                                  # 回到现实：全彩舞台，没有胶片；白光从满退到 0
        img = fb_view(stage_cg(), *MEM_C, z)
        dt = t - MEM_STAGE
        if dt < MEM_FADE: img = E.dither_to(img, PAPER, 1 - L.smooth(dt / MEM_FADE))
        return E.lift(img, E.bounce(dt, 2))
    img = fb_view(flashback_cg('p4_fb8_背影起势'), *MEM_C, z)
    # 光柱 / 光圈跟着推镜一起放大
    cone, pool = (cv2.warpAffine(m.astype(np.uint8), cv2.getRotationMatrix2D(MEM_C, 0, z), (W, H), flags=cv2.INTER_NEAREST) > 0
                  for m in spot_mask())
    if t < MEM_LIGHT:                                                   # 练习室原样（10-19 用户：压黑「不太用」，撤掉）
        if t - MEM_A < .05: img = E.dither_to(img, PAPER, .5)           # 闪回的记号：切进来闪一帧网点白
    else:                                                               # 啪：聚光灯打开，光柱外压暗，光柱里网点提亮
        out = ~(cone | pool)
        img[out] = (img[out].astype(np.float32) * .62).astype(np.uint8)
        img = E.dither_to(img, (226, 234, 255), .3, cone)
        img = E.dither_to(img, (240, 244, 255), .55, pool)
        k = (t - MEM_LIGHT - MEM_CLEAN) / (MEM_STAGE - MEM_LIGHT - MEM_CLEAN)
        if k > 0: img = E.dither_to(img, PAPER, k)                      # 白光涨满（转场）
    return film(img, t)


def scene_standees(t):
    a = E.cg(CROWD_A)
    img = a.copy() if t < T_SNAP else E.cg_aligned(CROWD_B, CROWD_A).copy()
    if T_SNAP <= t < T_SNAP + .1:                                      # 聚光灯啪地打开：网点闪白
        img = E.dither_to(img, PAPER, 1 - (t - T_SNAP) / .1)
    img = E.lift(img, E.bounce(t - STAND_BUMP, 2))                       # 观众暗中间那下重鼓颠一下
    return E.zoom(img, 320, 190, 1 + .05 * (t - STAND_A))


# ================================================================
# 5c backstage / board：后台
# ================================================================
@functools.lru_cache(maxsize=None)
def backstage_wide():
    return L.load_rgb(CG + 'p4_5c_后台.png', 853, 360)


def scene_backstage(t):
    u = L.smooth((t - BS_A) / (BOARD_A - BS_A))
    x = int(213 * u)
    return backstage_wide()[:, x:x + W].copy()


def scene_board(t):
    img = E.cg(CG + 'p4_5c_记分牌背面_a.png').copy()
    img = E.zoom(img, 330, 170, 1 + .1 * L.smooth((t - BOARD_A) / (137.04 - BOARD_A)))
    if 136.46 <= t < 136.62:                                           # 挂上最后一张 9：板子正面红光往后漏一下
        img = E.dither_to(img, (255, 60, 50), .25)
    return img


# ================================================================
# 6 + 7 pullback：布景 → 显示器 → 空房间 → 扎回屏幕
# ================================================================
PB_A, PB_SILENT, PB_DIVE, PB_B = 137.04, 138.45, 139.95, 140.27
SET_S = .42
DESK = CG + 'p4_7_深夜桌子.png'


@functools.lru_cache(maxsize=None)
def screen_box():
    """深夜桌子 CG 里纯绿屏幕的外接框和遮罩。"""
    img = E.cg(DESK)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] > 45) & (hsv[..., 0] < 75) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    mask = cv2.dilate((lab == k).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return (int(x), int(y), int(w), int(h)), mask


# 10-19 用户：最后电脑房间里加上 Claude 的各种演出海报 →
# 墙上贴五张，都是片中真实发生过的演出（画面直接取正片 CG），按时间从小到大：
# 选拔投稿的小传单（书架侧面）→ 1st LIVE → 点赞 LIVE（显示器左上）→ 蓝调之夜（箱子上）→ 千秋楽 满员御礼（显示器右上，最大）。
# 房间是夜里，海报压暗、偏冷，跟 CG 的光对上；只有千秋楽那张被屏幕的光照到一点。
POSTERS = [  # (源图, 源图里海报主体的中心 cx,cy（640 空间）, 贴的位置 x,y,w,h, 倾斜角度, 标题, 副标题, 底色, 亮度)
    # 10-19 v3 用户：海报挡住了电脑（遮挡没抠准）；No.8 贴到凸出柜子的平面上、小一点；排布错落自然些，字和人不必都露出来。
    # 平墙 = 电脑上方 + 电脑右边（x437–520、y<95）；凸出柜子正面 x522–578、y8–108。每张略歪、高低不齐、可以互相压一点。
    ('素材/关键帧/s6_舞台服副歌.png', (370, 150), (206, 14, 42, 58), -4, 'DEBUT', 'P(DOOM)', (60, 40, 120), .64),
    ('素材/关键帧/p3_s3_点赞全身.png', (320, 170), (244, 0, 38, 54), 3, 'LIKES', '♥LIVE', (255, 70, 170), .62),
    ('素材/CG/p3_8a_蓝调酒吧_c.png', (410, 170), (290, 22, 36, 50), -2, 'BLUES', 'NIGHT', (30, 40, 110), .58),
    ('素材/CG/p4_fb8b_背影起势_演出服_v4.png', (320, 170), (414, 8, 62, 86), 2, '千秋楽', '満員御礼', (200, 40, 50), .8),
    ('素材/关键帧/chorus1_夕阳天台.png', (360, 140), (536, 34, 20, 30), -6, '応募', 'No.8', (240, 236, 220), .55),
]


@functools.lru_cache(maxsize=None)
def desk_occluder():
    """深夜桌子 CG 里挡在墙前面的东西：电脑外壳（按颜色找亮灰绿外壳 → 凸包，圆角边也贴准）+ 皇冠（金色）。海报只画在它们后面。"""
    room = E.cg(DESK)
    hsv = cv2.cvtColor(room, cv2.COLOR_RGB2HSV)
    (sx, sy, sw, sh), scr = screen_box()
    shell = (hsv[..., 2] > 110) & (hsv[..., 1] < 110) | scr
    shell[:, :200] = False; shell[:, 450:] = False; shell[:40] = False; shell[250:] = False
    n, lab, st, _ = cv2.connectedComponentsWithStats(shell.astype(np.uint8))
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 400]   # 外壳和屏幕之间隔着深色边框，不连通 → 大块都并进来
    pts = np.argwhere(np.isin(lab, big))[:, ::-1].astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(pts)], 1)
    m[240:] = 1                                                        # 桌子以下不贴
    crown = (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 1] > 80) & (hsv[..., 2] > 120)
    crown[:, :330] = False; crown[:, 410:] = False; crown[60:] = False
    m |= crown.astype(np.uint8)
    return cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool)


@functools.lru_cache(maxsize=None)
def desk_room():
    """深夜桌子 CG + 墙上的演出海报（screen_view / off 都用这张；纯绿屏幕和便利贴不动）。"""
    import codeshots_h as GH
    room = E.cg(DESK).copy()
    occ = desk_occluder()
    for src, (scx, scy), (x, y, w, h), deg, title, sub, band, lum in POSTERS:
        bh = 10 if h < 44 else 14                                          # 标题带
        iw, ih = w - 4, h - bh - 4
        pic_full = L.load_rgb(src)
        ch = 330; cw = int(ch * iw / ih)                                   # 按海报比例裁（不压扁）
        x0 = int(np.clip(scx - cw // 2, 0, W - cw)); y0 = int(np.clip(scy - ch // 2, 0, H - ch))
        body = cv2.resize(pic_full[y0:y0 + ch, x0:x0 + cw], (iw, ih), interpolation=cv2.INTER_AREA)
        p = np.zeros((h, w, 3), np.uint8); p[:] = band
        p[2:2 + ih, 2:2 + iw] = body
        lab, lc = (title, (255, 230, 120)) if w >= 34 else (sub, (60, 40, 40))
        GH.zh_c(p, w // 2, h - bh - 1, lab, lc)
        if w >= 60: GH.zh(p, 3, 3, sub, (255, 240, 220))
        p = (p.astype(np.float32) * lum * np.array([.82, .9, 1.15])).clip(0, 255).astype(np.uint8)   # 夜里的冷光
        fr = np.full((h + 2, w + 2, 3), (10, 12, 24), np.uint8); fr[1:-1, 1:-1] = p
        fr[0:5, 0:7] = (150, 160, 180)                                     # 胶带（只贴上面一角，另一角翘着）
        if w >= 30: fr[0:5, w - 5:w + 2] = (150, 160, 180)
        pad = 6                                                            # 略歪着贴：转一下，硬边（最近邻）
        big = np.zeros((h + 2 + 2 * pad, w + 2 + 2 * pad, 3), np.uint8); big[pad:-pad, pad:-pad] = fr
        msk = np.zeros(big.shape[:2], np.uint8); msk[pad:-pad, pad:-pad] = 1
        M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), deg, 1.0)
        big = cv2.warpAffine(big, M, big.shape[1::-1], flags=cv2.INTER_NEAREST)
        msk = cv2.warpAffine(msk, M, msk.shape[::-1], flags=cv2.INTER_NEAREST).astype(bool)
        X0, Y0 = x - 1 - pad, y - 1 - pad
        ys, xs = np.nonzero(msk)
        ok = (ys + Y0 >= 0) & (ys + Y0 < H) & (xs + X0 >= 0) & (xs + X0 < W)
        ys, xs = ys[ok], xs[ok]
        vis = ~occ[ys + Y0, xs + X0]
        room[ys[vis] + Y0, xs[vis] + X0] = big[ys[vis], xs[vis]]
    return room


def the_set(t, s):
    """黑暗里的布景：竞技场背影 CG 缩到 s 倍，四周露出胶合板边框和脚手架。"""
    img = np.full((H, W, 3), (6, 6, 14), np.uint8)
    cw, ch = int(W * s), int(H * s)
    x0, y0 = (W - cw) // 2, int((H - ch) / 2 - 20 * (1 - s))
    if s < .98:
        for lx in (x0 + cw // 6, x0 + cw * 5 // 6):                      # 脚手架
            cv2.line(img, (lx, y0 + ch), (lx - 20, H), (40, 36, 44), 3, cv2.LINE_8)
            cv2.line(img, (lx, y0 + ch), (lx + 20, H), (40, 36, 44), 3, cv2.LINE_8)
        E.box(img, x0 - 6, y0 - 6, x0 + cw + 6, y0 + ch + 6, (120, 84, 50), 2)   # 胶合板边
        for wx in (x0 - 30, x0 + cw + 24):                               # 两盏工作灯
            E.box(img, wx, y0 + ch // 3, wx + 8, y0 + ch // 3 + 6, (255, 220, 120), 1)
    if t < PB_PEACE:
        back = E.cg(BACK_A)
    else:                                                               # 「show」：布景里的她转过头冲镜头比耶，轻轻一弹
        back = E.lift(E.cg_aligned(BACK_PEACE, BACK_A), E.bounce(t - PB_PEACE, 4))
    img[y0:y0 + ch, x0:x0 + cw] = cv2.resize(back, (cw, ch), interpolation=cv2.INTER_AREA)
    return img


# 10-14 用户：all for show 最后加一个回头比耶，带一点汗，不要太抢镜；不是把 CRT 换成别的画面，
# 而是屏幕里那块布景上的她本人换动作（同一张背影 CG 改图），然后照常扎回屏幕
# v2 用户：表情要 fb9 那种「疲惫但满足」、多点汗（改图换脸 → _比耶_v2）；时机从「show」提前到「Was it」的 it，布景还大、看得清
PB_PEACE = 137.60                                                  # 「it」137.74 一出来就已经转过来（提前约 3 帧）
BACK_PEACE = CG + 'p4_5a_背影看台下_比耶_v2.png'


def screen_view(content, z, crt=1.0):
    """深夜桌子，镜头推近 z 倍（z=1 看整个房间；z=W/屏宽 时屏幕宽度刚好铺满画面），屏幕里放 content。
    content 是 16:9 的画面，放进 4:3 的 CRT 时上下留黑边，所以屏幕铺满时 content 刚好就是整帧（和上一镜逐像素接上）。
    content 按推近以后的尺寸直接缩放贴进去，不会被放大成马赛克。crt<1 时关机：画面压成一条横线再缩成一个点。"""
    (x, y, w, h), mask = screen_box()
    cx, cy = x + w / 2, y + h / 2
    k = min((z - 1) / (W / w - 1), 1) if W / w > 1 else 1
    m = cv2.getRotationMatrix2D((cx, cy), 0, z)
    m[:, 2] += (W / 2 - cx) * k, (H / 2 - cy) * k
    room = desk_room().copy()
    room[mask] = (8, 10, 16)
    room = cv2.warpAffine(room, m, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    mz = cv2.warpAffine(mask.astype(np.uint8), m, (W, H), flags=cv2.INTER_NEAREST).astype(bool)
    X0, Y0 = m @ np.array([x, y, 1.0]); X1, Y1 = m @ np.array([x + w, y + h, 1.0])
    wz, hz = int(round(X1 - X0)), int(round(Y1 - Y0))
    if crt <= 0 or wz < 4: return room
    cw, ch = wz, int(round(wz * 9 / 16))
    sh = ch if crt >= 1 else max(1, int(ch * max(0, (crt - .35) / .65)))       # 先压成横线
    sw = cw if crt > .35 else max(2, int(cw * crt / .35))                      # 再缩成一个点
    c = cv2.resize(content, (sw, sh), interpolation=cv2.INTER_AREA)
    c = (c * .82 + np.array([10, 20, 40])).clip(0, 255).astype(np.uint8)
    if sh <= 3: c = np.maximum(c, 200).astype(np.uint8)
    canvas = np.full((H, W, 3), (8, 10, 16), np.uint8)
    px, py = int(round((X0 + X1) / 2 - sw / 2)), int(round((Y0 + Y1) / 2 - sh / 2))
    a0, b0 = max(px, 0), max(py, 0)
    a1, b1 = min(px + sw, W), min(py + sh, H)
    if a0 < a1 and b0 < b1: canvas[b0:b1, a0:a1] = c[b0 - py:b1 - py, a0 - px:a1 - px]
    canvas[::3] = (canvas[::3] * .8).astype(np.uint8)                          # 扫描线
    room[mz] = canvas[mz]
    return room


def scene_pullback(t):
    if t < PB_SILENT:
        return the_set(t, 1 - (1 - SET_S) * L.smooth((t - PB_A) / (PB_SILENT - PB_A)))
    (x, y, w, h), _ = screen_box()
    z0 = W / w
    if t < PB_DIVE:
        u = L.smooth((t - PB_SILENT) / 1.2)
        z = z0 ** (1 - u)                                                # 按对数插值，拉远的速度看起来均匀
    else:                                                              # 扎回去：越来越快，边推边把布景放大回整座竞技场
        u = ((t - PB_DIVE) / (PB_B - PB_DIVE - 1 / L.FPS)) ** 2
        u = min(u, 1)
        z = z0 ** min(u / .6, 1)
        s = SET_S + (1 - SET_S) * L.smooth((u - .5) / .5)
        return screen_view(the_set(t, s), z)
    return screen_view(the_set(t, SET_S), z)


# ================================================================
# 9 guest：嘉宾亮相（141.63–145.72）的画面在 codeshots_h（幕后花絮版），这里只留起点
# ================================================================
G0 = 141.63


# ================================================================
# 13 off：退回显示器外，CRT 关机
# ================================================================
OFF_A, OFF_CRT, OFF_DARK, OFF_B = 154.25, 154.75, 155.30, 156.65


@functools.lru_cache(maxsize=None)
def note_mask():
    """便利贴（黄色）的位置：关机以后只剩它。"""
    hsv = cv2.cvtColor(E.cg(DESK), cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] > 20) & (hsv[..., 0] < 35) & (hsv[..., 1] > 90) & (hsv[..., 2] > 170)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    out = np.zeros((H, W), bool); out[y - 2:y + h + 2, x - 2:x + w + 2] = True
    return out


def scene_off(t):
    closed = np.clip(BD.curtain(np.zeros((H, W, 3), np.float32), 1), 0, 255).astype(np.uint8)
    closed = E.text_c(closed, W // 2, 150, 'Fin.', YEL, 4)
    crt = 1 - np.clip((t - OFF_CRT) / .3, 0, 1)
    (x, y, w, h), _ = screen_box()
    z = (W / w) ** (1 - L.smooth((t - OFF_A) / .45))
    img = screen_view(closed, z, crt)
    if t >= OFF_DARK:                                                    # 房间暗下去，便利贴最后暗
        k = (t - OFF_DARK) / .7
        keep = note_mask() if z <= 1.001 else np.zeros((H, W), bool)
        img = E.dither_to(img, (4, 4, 10), k, ~keep)
        if t >= OFF_DARK + .9: img = E.dither_to(img, (4, 4, 10), (t - OFF_DARK - .9) / .4)
    return img


# ================================================================
# 演职员表（OVERLAY）：动画片尾式的卡片，右下角一个 PC-98 小窗口，跟着舞蹈串烧换卡
# 演员表由嘉宾名牌负责（guest），这里只放片名、制作人员、原曲和致谢
# ================================================================
CARDS = [  # (开始, 结束, 标题栏, [(职务, 名字)])
    (138.74, 140.27, 'P(DOOM)', [('', 'a Claude Pop MV')]),
    (145.72, 146.63, 'STAFF', [('PRODUCER', 'XIAOKE'), ('DIRECTOR', 'XIAOKE'), ('', '(Claude Opus 5.5)')]),
    (146.63, 147.54, 'STAFF', [('KEY ART / CG', 'gpt-image-2'), ('', '(xk-draw)')]),
    (147.54, 148.45, 'STAFF', [('MOTION REF', 'Seedance 2.0'), ('', '(xk-video)')]),
    (148.45, 149.36, 'STAFF', [('CUTOUT', 'rembg'), ('PC-98 FINISH', 'pc98ify.py'), ('STAGE FX', 'live98.py')]),
    (149.36, 150.27, 'STAFF', [('CODE SHOTS', 'codeshots_a-g'), ('ENGINE', 'mv.py'), ('LYRIC TIMING', 'Whisper')]),
    (150.27, 151.00, 'CLAWD RIG', [('', 'ClaudeAnimationBase'), ('', 'by JohnHeibel (MIT)')]),
    # 11-12 用户：原曲和参考作品写全（灵感来源），用日语。原来是一张 SONG 卡（udio, 2024 / LYRICS osmarks）
    # 署名按 osmarks 的说明页：词 = MusicPerson（开头）+ osmarks；deckard 是 Suno「Claude-Pop」版的发布者，不是作曲
    (151.00, 151.70, '原曲', [('', "I'm Upping My P(doom)"), ('作詞', 'osmarks & MusicPerson'), ('', 'Claude-Pop ver. via deckard')]),
    (151.70, 152.26, 'インスパイア元', [('参考映像', '@pleometric'), ('', 'ありがとう！')]),
    (152.26, 153.20, 'CAST', [('XIAOKE', 'Claude娘'), ('CLAWD x9', '観客(紙)'),
                              ('', 'JohnHeibel/ClaudeAnimationBase')]),     # 群演的出处，挨着 Clawd 写
    (153.20, 154.20, 'THANK YOU', [('', 'ご主人さま'), ('', 'and you ♥')]),
]
CW, CH = 256, 84                                                        # 宽到能放下 JohnHeibel/ClaudeAnimationBase 一行


def credits(img, t):
    card = [c for c in CARDS if c[0] <= t < c[1]]
    if not card: return img
    a, b, title, rows = card[0]
    fix = __import__('mv').FIX
    col = lambda k: tuple(int(c) for c in fix[k])
    k = min((t - a) * L.FPS / 3, 1)                                     # 3 帧从一条线竖着展开
    x0, y1 = W - 14 - CW, H - 12
    h = max(4, int(CH * k))
    y0 = y1 - h
    img = img.copy()
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + CW + 1, y1 + 1), col('ink'), -1)
    cv2.rectangle(img, (x0, y0), (x0 + CW - 1, y1 - 1), col('blue'), -1)
    cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + CW - 3, y1 - 3), col('paper'), 1)
    if k < 1: return img
    cv2.rectangle(img, (x0 + 3, y0 + 3), (x0 + CW - 4, y0 + 19), col('ink'), -1)
    cx = x0 + CW // 2                                                   # 10-19 用户：STAFF 字没居中 → 标题和每一行都在卡片里居中（左右 + 上下）
    tw = lambda s: int(L.font(16).getlength(s))
    def th(s):                                                          # draw_text 画出来的实际高度（按字形外框，含 1px 边）
        l_, t_, r_, b_ = L.font(16).getbbox(s or 'A'); return b_ - t_ + 2
    img = E.text(img, cx - tw(title) // 2, y0 + 3 + (16 - th(title)) // 2, title, col('orange'))
    n, LH = len(rows), 19
    body0, body1 = y0 + 20, y1 - 3                                      # 正文区（标题栏下沿 → 卡片下沿）
    top = body0 + (body1 - body0 - n * LH) // 2 + (LH - th('Ag')) // 2
    for i, (role, name) in enumerate(rows):
        y = top + i * LH
        if role:
            x = cx - (tw(role) + 12 + tw(name)) // 2
            img = E.text(img, x, y, role, col('yel'))
            img = E.text(img, x + tw(role) + 12, y, name, col('paper'))
        else:
            img = E.text(img, cx - tw(name) // 2, y, name, col('paper'))
    return img


SCENES = [
    ('loom', LOOM_A, LOOM_B, scene_loom),
    ('mask', MASK_A, 129.70, scene_mask),
    ('upgrade', UP_A, 131.88, scene_upgrade),
    ('spot', SPOT_A, SPOT_B, scene_spot),
    ('memory', MEM_A, STAND_A, scene_memory),
    ('standees', STAND_A, BS_A, scene_standees),
    ('backstage', BS_A, BOARD_A, scene_backstage),
    ('board', BOARD_A, 137.04, scene_board),
    ('pullback', PB_A, PB_B, scene_pullback),
    ('off', OFF_A, OFF_B, scene_off),
]

if __name__ == '__main__':
    L.sheet_cli(SCENES)
