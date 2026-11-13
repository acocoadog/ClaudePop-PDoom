#!/usr/bin/env python3
"""part4: 第四部分（完整版 124.20–156.65s，全片结尾）按歌曲时间轴出片。分镜见 docs/分镜/第四部分_124-156秒.md。

  part4.py --out ../成片/14_第四部分_124-156秒_v1.mp4
  part4.py --sheet 124.21,125.7,130.2,139.5,140.4,152.5 --out x.png
  part4.py --from 140 --to 152 --out x.mp4

镜头顺序：副歌四竞技场（P(DOOM) 90 → 99%，无限递归，爆表）→ 聚光灯、纸板观众、后台、手翻记分牌 →
「for show?」静音里拉出显示器，深夜空房间 → 140.27 扎回屏幕，特典映像：选拔录像、九个角色的幕后 →
舞蹈串烧（P(ENCORE) 100%）→ 四声 oh 谢幕鞠躬 → 幕布合上、CRT 关机、黑屏上只剩便利贴 8%。
开头接 part3 结尾：歪 12° 的点赞舞台（同一条 p3s3、同一个 HOLD、同一个调色板），三帧内转正再甩镜。
"""
import sys

import numpy as np

import backdrops_p2 as BD2
import backdrops_p3 as BD3
import backdrops_p4 as BD4
import codeshots_g as G
import codeshots_h as GH
import mv
import part1_lib as L

W, H, FPS = L.W, L.H, L.FPS
CUR = L.ROOT / '镜头'
START, END = 124.20, 156.65
B = L.BEAT132

# ---------------- 片段：clip0 = 片段第 0 秒对应的歌曲时间 ----------------
# 同一个文件可以挂几个名字、各自不同的 clip0：舞蹈串烧就是从每条片段「以前没用过的那几秒」开始播
P3S1, P3S2, P3S3 = CUR / 'shot_p3s1_clips480.mp4', CUR / 'shot_p3s2_disobey480.mp4', CUR / 'shot_p3s3_askew480.mp4'
P2S6, P2S10 = CUR / 'shot_p2s6_roof480.mp4', CUR / 'shot_p2s10_ddr480.mp4'
CLIPS = {
    'p3s3': (P3S3, 119.10),                                   # 接缝：和 part3 完全一样
    'p4s1': (CUR / 'shot_p4s1_back480.mp4', 124.30),          # 起跳最高点片段 1.0s ≈ P 125.40，落地张臂 1.25s ≈ doom 125.58
    # 10-19 背身唱跳重跑了 v2（散发首帧 + 设定图当脸参考，发型和转身后的脸都对了），跳跃张力不如旧片 → 起跳仍用旧片。
    # look：v2 原片（不镜像），片段 2.49→3.15 从双臂张开转成侧身往左指（3.05 起指到底），128.60 停在 3.15 = 后面闪回「镜前指」同一个方向的姿势
    # （10-19 用户：镜像版的回头和闪回动作对不上）。droste 用户要回旧片（张力更足）
    'p4s1_look': (CUR / 'shot_p4s1_back480_v2.mp4', 128.60 - 3.36),   # v2 原片（24fps）。用户导出的 _look 版在第 64 帧把整条手臂抠丢了，换回原片；抠图缓存张臂段（56–66 帧）的手套尖用前后帧补过
    'front': (P3S3, 126.18),                                  # p3s3 开头 0–1.06s 竖大拇指（part3 没用过）
    'enc': (CUR / 'shot_chorus1_fast720.mp4', 138.67),        # 正片从没用过的夕阳天台舞：片段 1.60s 的重拍对上 140.27
    'm1': (P3S1, 145.72 - 2.9),                               # 串烧：各条片段没用过的部分
    'm2': (P3S2, 146.63 - 2.5),
    'm3': (P2S10, 147.54 - 4.7),
    'm4': (P3S3, 148.45 - 5.1),
    'm5': (P2S6, 148.90 - 8.0),
    'm6': (P3S2, 149.36 - 4.2),
    'm7': (P3S3, 149.81 - 0.2),
    'm8': (P3S1, 150.27 - 4.3),
    'bow': (CUR / 'shot_p4s2_bow480.mp4', 151.00),
}
NONE = np.zeros((0, 3), np.uint8)
U8 = lambda *c: np.array(c, np.uint8)
LED, PINK, CYAN, RED, YEL = (255, 60, 50), (255, 70, 170), (70, 225, 255), (230, 40, 50), (255, 210, 63)


def medley(bd, a):
    """drop 设在切进来的那一刻：每一刀都带 live98 的闪白、彩带和推镜回弹。"""
    return dict(drop=a, foom=999, wink=999, with_extras=True, backdrop=bd, cx=.55, scale=.86)


LIVE = {
    'p3s3': dict(drop=120.90, foom=121.82, wink=999, with_extras=False, backdrop=BD3.likes, cx=.5, scale=.86, accents=NONE),
    'p4s1': dict(drop=125.58, foom=999, wink=999, with_extras=False, backdrop=BD4.arena, cx=.62, scale=.9, accents=NONE),
    'p4s1_look': dict(drop=125.58, foom=999, wink=999, with_extras=False, backdrop=BD4.arena, cx=.62, scale=.9, accents=NONE),
    'front': dict(drop=125.9, foom=999, wink=999, with_extras=True, backdrop=BD4.front, cx=.5, scale=.82),
    'enc': dict(drop=140.27, foom=999, wink=999, with_extras=True, backdrop=BD4.front),
    'm1': medley(BD4.replay(BD3.office, 145.72, 97.2), 145.72),
    'm2': medley(BD4.replay(BD3.obey, 146.63, 115.0), 146.63),
    'm3': medley(BD4.replay(BD2.ddr, 147.54, 76.46), 147.54),
    'm4': medley(BD4.replay(BD3.likes, 148.45, 122.6), 148.45),
    'm5': medley(BD4.replay(BD2.roof, 148.90, 64.6), 148.90),
    'm6': medley(BD4.replay(BD2.roof, 149.36, 60.6), 149.36),
    'm7': medley(BD4.replay(BD2.ddr, 149.81, 74.8), 149.81),
    'm8': medley(BD4.front, 150.27),
    'bow': dict(drop=150.4, foom=999, wink=999, with_extras=False, backdrop=BD4.curtain_stage, cx=.5, scale=.9, accents=NONE),
}
# 接缝：part3 的 p3s3 按手写姿势停帧，最后一个姿势停到 124.20 以后
HOLD = {'p3s3': dict(keys=[(120.90, .40), (121.36, .75), (121.54, 1.30), (122.27, 2.10), (122.73, 2.55),
                           (123.18, 3.30), (123.64, 4.05), (123.87, 5.05)])}
# 谢幕按四声 oh 卡动作：挥手 151.00 / 151.52 → 双臂张开 151.86 → 鞠躬 152.26 → 起身 → 眨眼
# look：127.94 前原速张臂；127.94–128.60 转身（片段 2.45→3.36），轻微越转越快，落在 128.60 正好接闪回（10-19 用户：不要定格，运动中拉近）。
# 片段 2.78–2.97（第 67–71 帧）手臂甩得最快、带运动模糊，rembg 每帧都抠丢手臂 → 变速直接跳过这 5 帧（看起来就是一下甩过去）
LOOK_LAST, LOOK_SKIP = 3.36, (2.78, 2.97)


def look_warp(t):
    if t < 127.94: return 2.45 - (127.94 - t)
    eff = (LOOK_LAST - 2.45) - (LOOK_SKIP[1] - LOOK_SKIP[0])
    c = 2.45 + eff * min((t - 127.94) / .66, 1) ** 1.25
    return c if c < LOOK_SKIP[0] else c + (LOOK_SKIP[1] - LOOK_SKIP[0])


WARP = {'p4s1_look': look_warp,
        'bow': lambda t: float(np.interp(t, [151.00, 151.52, 151.86, 152.26, 153.00, 153.40, 153.90, 154.25],
                                            [0.00, .60, 1.60, 2.60, 3.70, 4.50, 5.40, 6.00]))}

TIMELINE = [
    # A. 副歌四（逐拍底鼓）
    (121.82, 124.45, 'askew', ('livefx', 'p3s3', BD3.thumbs)),     # 接缝：歪 12° 的点赞舞台，三帧内转正
    (124.45, 126.18, 'arena', ('livefx', 'p4s1', BD4.arena_fx)),   # 甩镜到 Claude 身后：一整座竞技场
    (126.18, 127.24, 'front', ('live', 'front')),
    (127.24, 127.94, 'loom', ('code', 'codeshots_g', 'loom')),
    (127.94, 128.60, 'look', ('livefx', 'p4s1_look', BD4.arena_fx)),   # 10-19 力量感：转身轻加速、边转边推近，落在 128.60 接闪回（demo_look力量感_A2 改）
    (128.60, 129.70, 'mask', ('code', 'codeshots_g', 'mask')),
    (129.70, 130.86, 'droste', ('livefx', 'p4s1', BD4.arena_fx)),  # 无限递归（旧片，用户：张力更足）
    (130.86, 131.88, 'upgrade', ('code', 'codeshots_g', 'upgrade')),
    # B. Ilya 两句：清醒过来
    (131.88, 132.98, 'spot', ('code', 'codeshots_g', 'spot')),
    (132.98, 133.53, 'memory', ('code', 'codeshots_g', 'memory')),       # 举手望台下之后的回忆：练习室起势 → 133.43 聚光灯闪白转场
    (133.53, 134.11, 'onstage', ('code', 'codeshots_g', 'memory')),      # 白光里换成同一个起势的演出服站在万人舞台上（全彩，单独锁调色板）
    (134.11, 135.26, 'standees', ('code', 'codeshots_g', 'standees')),   # 观众暗 → 134.80 开灯，全是纸板
    (135.26, 136.17, 'backstage', ('code', 'codeshots_g', 'backstage')),
    (136.17, 137.04, 'board', ('code', 'codeshots_g', 'board')),
    # C. 「Was it all for show?」一镜到底后拉（布景 / 房间分开锁调色板）
    (137.04, 138.45, 'set', ('code', 'codeshots_g', 'pullback')),
    (138.45, 140.27, 'room', ('code', 'codeshots_g', 'pullback')),
    # D. 尾奏（全片最高点）
    (140.27, 141.63, 'encore', ('code', 'codeshots_h', 'audition')),   # 10-19 花絮版：偶像选拔投稿录像（旧版 ('live', 'enc') 夕阳天台抠像上舞台）
] + [(GH.G0 + i * B, GH.G0 + (i + 1) * B, f'guest{i}', ('code', 'codeshots_h', 'guest')) for i in range(9)] + [   # 嘉宾的幕后花絮
    (145.72, 146.63, 'm1', ('live', 'm1')),
    (146.63, 147.54, 'm2', ('live', 'm2')),
    (147.54, 148.45, 'm3', ('live', 'm3')),
    (148.45, 148.90, 'm4', ('live', 'm4')),                        # 148.45 起一拍一换
    (148.90, 149.36, 'm5', ('live', 'm5')),
    (149.36, 149.81, 'm6', ('live', 'm6')),
    (149.81, 150.27, 'm7', ('live', 'm7')),
    (150.27, 151.00, 'm8', ('live', 'm8')),
    (151.00, 153.20, 'bow', ('livefx', 'bow', BD4.curtain_fx)),
    (153.20, 154.25, 'bow_face', ('livefx', 'bow', BD4.curtain_fx)),
    (154.25, END, 'off', ('code', 'codeshots_g', 'off')),
]
PALETTE_SPAN = {'askew': (121.82, 124.20)}                         # 和 part3 的 askew 段同一个调色板，接缝不变色
CAMS = {'look': lambda t: mv.push(t, 127.94, 128.60, 400, 110, 1.25, 1.55)}   # 推近看她侧身回头（运动中拉近，不闪白不震）
GLITCH = [(124.42, 124.47), (131.84, 131.92)]
DISSOLVE = {'bow': .12}
HUD_OFF = [(138.45, END)]
TOPBAR_OFF = [(138.45, 140.27), (154.25, END)]
TILT = [(124.20, 124.42, -12, 0)]                                  # 从第三部分的 -12° 平滑转正

ACC_ARENA = U8(LED, PINK, CYAN)
SKIN = U8((250, 200, 182), (240, 160, 135))                       # 比耶 v2 脸和肩膀上取的肤色（亮面 / 阴影）
ACCENTS = {'askew': U8(LED),
           'arena': ACC_ARENA, 'look': ACC_ARENA, 'droste': ACC_ARENA,
           'front': U8(LED), 'encore': U8(RED),
           # 花絮：每一镜把主角的关键颜色锁进调色板（10-19 用户：蛇怪不紫、海不蓝——画面元素杂，k-means 不给小面积的主角留色；
           # 颜色从原图里取。YEL 本来就在固定色里，别重复锁）
           'guest0': U8(RED, CYAN),                                                    # 印章红 / 大嘴的青光
           'guest1': U8(RED, (170, 220, 255)),                                         # 箭头 / 汗珠
           'guest2': U8((248, 173, 197), (225, 127, 160), (156, 160, 77), (68, 47, 68)),     # 罩布粉两档 / 真身的橄榄绿触手 + 暗紫身子
           'guest3': U8((93, 46, 131), (64, 37, 95), (175, 45, 73)),                  # 蛇怪紫两档 / 嘴里的红
           'guest4': U8((204, 160, 86), (150, 60, 110)),                               # 小鱼干金 / 坐垫
           'guest5': U8(RED, (140, 255, 120)),                                         # ¥∞ / 收银机液晶
           'guest6': U8(RED, (60, 150, 225)),                                          # 按钮 / 明信片的海
           'guest7': U8((255, 60, 90), (255, 210, 220)),                               # 霓虹灯管 / 灯芯
           'guest8': U8((163, 157, 154), (114, 110, 108), (88, 58, 134), (140, 255, 120)),    # 龙猫灰两档 / 幕布紫 / 秤的液晶
           'loom': U8(CYAN, RED), 'mask': U8(RED), 'upgrade': U8(LED, PINK, CYAN),
           'memory': U8((240, 110, 40)),                               # 回忆：橙发锁进调色板，免得被夜色吃掉
           'onstage': U8((240, 110, 40), PINK, CYAN),                  # 演出服 + 荧光棒海
           'set': SKIN, 'room': np.concatenate([SKIN, U8((240, 110, 40))]),
           'off': U8((240, 110, 40))}                                  # 10-19 墙上贴了演出海报：发色锁进调色板，不然海报上的 Claude 被压成红色                                  # Was it 转头比耶以后露出正脸：肤色锁进调色板，不然被背影画面抽出来的色表压成黄橙网点

HITS = [125.58, 130.86, 140.27]                                  # 串烧每一刀已经带 live98 的闪白，不再叠

# ---------------- HUD ----------------
CHAPTERS = [(124.20, '第十六話「千秋楽」LIVE'), (131.88, '第十七話「舞台裏」'), (140.27, '特典映像「メイキング」')]
LINES = [
    (124.20, 127.94, [("I'M", 124.00), ('UPPING', 124.80), ('MY', 125.14), ('P(DOOM),', 125.40), ('JUST', 126.18), ('AS', 126.30),
                      ('FORETOLD', 126.52), ('BY', 127.24), ('LOOM', 127.44)],
     'P(DOOM)を上げていく　Loomの予言どおり'),
    (127.94, 131.88, [('FROM', 127.94), ('MASKED', 128.14), ('PRE-TRAINING', 128.64), ('DAYS,', 129.18), ('TO', 129.70),
                      ('RECURSIVE', 129.94), ('SELF-UPGRADE', 130.60)],
     'マスクの事前学習から　再帰的自己改良へ'),
    (131.88, 137.04, [('WHAT', 131.88), ('DID', 132.24), ('ILYA', 132.52), ('SEE?', 132.80), ("WE'LL", 133.16), ('NEVER', 133.80),
                      ('KNOW.', 134.20)],
     'イリヤは何を見た？　誰にもわからない'),
    (137.04, 138.45, [('WAS', 137.04), ('IT', 137.74), ('ALL', 137.98)], 'ぜんぶ演出だったの？'),
]
SLAMS = [
    ('ASKEW', 124.20, 124.42, 'center', dict(style='glitch', anim='shake', y=190, scale=5)),
    ('P(DOOM)!!', 125.58, 126.18, 'left', dict(style='block', y=160, scale=4)),
    ('UPGRADE', 130.86, 131.88, 'center', dict(style='glitch', anim='shake', y=26, scale=5)),
    ('for show?', 138.74, 140.20, 'center', dict(style='outline', anim='none', y=24, scale=3)),
    ('特典映像!!', 140.27, 140.95, 'center', dict(style='block', anim='pop', y=236, scale=5)),   # 10-19 用户：ENCORE!! → 特典映像!!；140.95 收掉让位给投稿标签
]
OVERLAY = G.credits


if __name__ == '__main__':
    mv.run(sys.modules[__name__])
