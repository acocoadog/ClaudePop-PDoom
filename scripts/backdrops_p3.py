"""backdrops_p3: 第三部分 live98 段的背景（backdrop 钩子）和前景特效（mv 的 livefx 来源）。

office(img, t, lit, beat)   S1 副歌三 96.62–98.86：惨白办公室，天花板往下漏回形针，墙上 P(DOOM) 71 → 84（drop 96.62）
obey(img, t, lit, beat)     S2 disobey 114.20–115.22：深青底上一块「OBEY」大招牌，出拳那一下碎成几块飞走
likes(img, t, lit, beat)    S3 RLHF 120.90–124.20：点赞粉舞台，身后点赞计数和 P(DOOM) 84 → 90
clip_pile(rgb_at, t, a, b)  前景：回形针从脚下往上堆，98.30 起淹到她腰
thumbs(rgb_at, t, a, b)     前景：👍 点赞雨，越下越密

t 是歌曲时间。全是纯函数（粒子按 hash01 + 时间直接算位置）。
"""
import cv2
import numpy as np

import codeshots_e as E
import part1_lib as L
import props

W, H = L.W, L.H
PINK_BG = (230, 40, 140)


def office(img, t, lit, beat):
    img = np.array(img, np.float32)
    img[:] = (224, 226, 236)
    for x in range(0, W, 80): img[:, x:x + 2] = (196, 198, 212)          # 墙板缝
    img[250:] = (180, 182, 198)                                         # 地毯
    for y in (18, 40):                                                  # 日光灯管
        for x in (60, 260, 460):
            img[y:y + 5, x:x + 120] = (255, 255, 250) if (int(t * 24) + x) % 37 else (200, 200, 210)
    img = props.scoreboard(img, t, (110, 150, 118, 76), 71, 84, 96.32, 96.62, 1.0, beat)   # 左上留给 P(DOOM)!! 大字；09-30 高 70→76，同 part1，数字不切底
    for i in range(46):                                                 # 漏下来的回形针
        sp = 150 + 110 * L.hash01(i, 1)
        x = L.hash01(i, 2) * W
        y = (L.hash01(i, 3) * 300 + (t - 96.3) * sp) % 300 - 30
        if t < 96.62 and i % 3: continue                                # drop 前只零星几根
        E.clip_icon(img, x, y, .55, ang=(i * 41 + int(t * 12) * 30) % 360)
    return img


def clip_pile(rgb_at, t, a, b):
    """回形针堆：drop 起从地面往上涨，98.30 猛涨到腰（y≈190）。"""
    img = rgb_at(t).copy()
    lvl = 360 - 50 * L.smooth((t - 96.62) / 1.2) - 120 * L.smooth((t - 98.20) / .25)
    for i in range(260):
        x = L.hash01(i, 11) * (W + 40) - 20
        y = lvl + L.hash01(i, 12) * (H - lvl + 40)
        if y > H + 20: continue
        E.clip_icon(img, x, y + 3 * np.sin(t * 6 + i), .6, ang=i * 67 % 360, col=(196, 200, 214))
    top = int(lvl) + 18                                                 # 堆的深处加一层网点阴影
    if top < H:
        m = np.zeros((H, W), bool); m[top:] = True
        E.dither_to(img, (150, 152, 170), .33, m)
    return img


HAZ_Y, HAZ_K = (255, 210, 40), (20, 18, 26)


def hazard_band(img, y0, y1, t, speed, w=22):
    """黄黑斜条警示带，按时间横向滚动（纯函数）。"""
    yy, xx = np.mgrid[y0:y1, 0:W]
    ph = ((xx + yy + t * speed) // w) % 2 == 0
    band = np.where(ph[..., None], np.array(HAZ_Y, np.float32), np.array(HAZ_K, np.float32))
    img[y0:y1] = band
    img[y0:y0 + 2] = HAZ_K; img[y1 - 2:y1] = HAZ_K


def warn_sign(img, cx, cy, r, on):
    pts = np.int32([(cx, cy - r), (cx + r, cy + r * .8), (cx - r, cy + r * .8)])
    cv2.fillPoly(img, [pts], HAZ_K)
    inner = np.int32([(cx, cy - r + 9), (cx + r - 9, cy + r * .8 - 5), (cx - r + 9, cy + r * .8 - 5)])
    cv2.fillPoly(img, [inner], HAZ_Y if on else (150, 120, 30))
    cv2.rectangle(img, (cx - 3, int(cy - r * .35)), (cx + 3, int(cy + r * .3)), HAZ_K, -1)
    cv2.rectangle(img, (cx - 3, int(cy + r * .42)), (cx + 3, int(cy + r * .55)), HAZ_K, -1)


def obey(img, t, lit, beat):
    """10-19 用户：背景要精细点，比如动态的危险警示符号（斜线那种）→ 暗色工业墙上铺一层慢慢滚的淡斜纹，
    上下两条黄黑警示带反向滚动，两边警示三角按拍闪，左右上角两盏旋转警灯扫光。OBEY 牌和被砸碎照旧。"""
    img = np.array(img, np.float32)
    img[:] = (14, 64, 76)
    yy, xx = np.mgrid[0:H, 0:W]
    img[((xx - yy + t * 30) // 30) % 2 == 0] = (18, 76, 90)            # 墙上的淡斜纹
    for y in range(0, H, 48): img[y:y + 1] = (10, 50, 60)
    on = L.pulse(t) > .35
    warn_sign(img, 46, 150, 34, on); warn_sign(img, W - 46, 150, 34, not on)
    for bx, ph in ((26, 0.0), (W - 26, np.pi)):                        # 旋转警灯：灯罩 + 扫过去的光锥（网点）
        a = t * 7 + ph
        tip = np.array([bx, 50.0])
        d1 = np.array([np.cos(a - .25), np.sin(a - .25)]) * 420; d2 = np.array([np.cos(a + .25), np.sin(a + .25)]) * 420
        cone = np.zeros((H, W), np.uint8)
        cv2.fillPoly(cone, [np.int32([tip, tip + d1, tip + d2])], 1)
        cone[:34] = 0
        m = (cone > 0) & L.bayer_mask(.28)
        img[m] = (255, 170, 60)
        cv2.circle(img, (bx, 50), 13, HAZ_K, -1)
        cv2.circle(img, (bx, 50), 11, (255, 140, 40) if np.cos(a) > 0 else (200, 90, 30), -1)
        cv2.rectangle(img, (bx - 15, 58), (bx + 15, 66), (60, 60, 70), -1)
    for a, b in ((0, 34), (232, 258)):                                 # 警示带压在光锥上面
        hazard_band(img, a, b, t, 120 if a == 0 else -120)
    hit = 114.20
    letters = 'OBEY'
    for k, ch in enumerate(letters):
        x0, y0 = 120 + k * 104, 60
        if t < hit:
            dx = dy = ang = 0
        else:                                                           # 碎：每个字母往外飞 + 转
            u = t - hit
            a = (k - 1.5) * .7
            dx, dy, ang = np.sin(a) * 700 * u, -300 * u + 900 * u * u, (k - 1.5) * 160 * u
        tile = np.zeros((110, 96, 3), np.uint8)
        cv2.rectangle(tile, (0, 0), (95, 109), (200, 30, 40), -1)
        cv2.rectangle(tile, (4, 4), (91, 105), (255, 240, 220), 2)
        tile = L.draw_text(tile, (14, 8), ch, (255, 240, 220), scale=5)
        M = cv2.getRotationMatrix2D((48, 55), ang, 1)
        M[:, 2] += (x0 + dx, y0 + dy)
        msk = cv2.warpAffine(np.ones((110, 96), np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        img[msk] = cv2.warpAffine(tile, M, (W, H), flags=cv2.INTER_NEAREST)[msk]
    if 0 <= t - hit < .25:                                              # 冲击：一圈放射线
        u = (t - hit) / .25
        for k in range(16):
            a = k * np.pi / 8
            r0, r1 = 40 + 300 * u, 90 + 360 * u
            cv2.line(img, (int(330 + np.cos(a) * r0), int(150 + np.sin(a) * r0)),
                     (int(330 + np.cos(a) * r1), int(150 + np.sin(a) * r1)), (255, 250, 220), 4)
    return img


def thumb(img, x, y, s, col=(255, 210, 63)):
    """像素竖拇指：拳头方块 + 竖起的大拇指 + 袖口。"""
    S = lambda v: int(round(v * s))
    x, y = int(x), int(y)
    E.box(img, x, y, x + S(22), y + S(20), col, 2)
    E.box(img, x + S(2), y - S(16), x + S(11), y + S(2), col, 2)
    E.box(img, x - S(8), y + S(4), x, y + S(20), (70, 150, 255), 2)
    for k in range(3):
        yy = y + S(5) + k * S(5)
        cv2.line(img, (x + S(10), yy), (x + S(20), yy), (170, 120, 30), 1)


def likes(img, t, lit, beat):
    img = np.array(img, np.float32)
    img[:] = PINK_BG
    for i in range(12):                                                 # 背景的放射条
        a0 = i * np.pi / 6 + t * .3
        pts = np.int32([(320, 180), (320 + np.cos(a0) * 800, 180 + np.sin(a0) * 800),
                        (320 + np.cos(a0 + .26) * 800, 180 + np.sin(a0 + .26) * 800)])
        cv2.fillPoly(img, [pts], (250, 70, 160))
    n = int(10 ** (2 + max(0, t - 120.9) * 2.2))
    img = L.draw_text(img, (20, 30), f'LIKES {n:,}'[:18], (255, 240, 250), scale=2)
    img = props.scoreboard(img, t, (500, 30, 118, 76), 84, 90, 121.82, 123.6, 1.0, beat)
    return img


def thumbs(rgb_at, t, a, b):
    img = rgb_at(t).copy()
    n = int(8 + 40 * L.smooth((t - 120.9) / 1.2))
    for i in range(n):
        sp = 260 + 160 * L.hash01(i, 21)
        x = L.hash01(i, 22) * (W - 30)
        y = (L.hash01(i, 23) * 400 + (t - 120.9) * sp) % 420 - 40
        thumb(img, x, y, .8 + .6 * L.hash01(i, 24))
    return img
