#!/usr/bin/env python3
"""part3: 第三部分（完整版 89.12–124.20s）按歌曲时间轴出片。分镜见 docs/分镜/第三部分_89-124秒.md。

  part3.py --out ../成片/11_第三部分_89-124秒_v1.mp4
  part3.py --sheet 89.9,91,96.8,114.5,122.5 --out x.png      按歌曲时间抽帧拼图检查
  part3.py --from 109 --to 117 --out x.mp4                    只出一段（预览用）

第三阶段新标准（见 docs/开发笔记.md 核心经验 7、8）：一句至少两镜、爆发段一词一镜；
人物按拍换关键姿势停帧（HOLD）；大字多种字效（SLAMS 第五项）；askew 整帧歪斜（TILT）。
开头接 part2 结尾：全黑只剩 NIL 闪烁。
"""
import sys

import numpy as np

import backdrops_p3 as BD
import mv
import part1_lib as L

W, H, FPS = L.W, L.H, L.FPS
CUR = L.ROOT / '镜头'
START, END = 89.12, 124.20

# ---------------- Seedance 片段：clip0 = 片段第 0 秒对应的歌曲时间 ----------------
# 09-28 当时以为三条全被重唱；09-29 逐词对齐实测：s1 完整保留（偏 0.68 秒）、s2 局部挤压、s3 部分重唱（见 README 经验 3）。按和原曲一致的词量偏移：s2 post/super、s3 RLHF/askew 两两一致；
# s1 按 killswitch 定，再核对跳起最高点（片段 0.5–0.75s）落在 drop 96.62。口型不管，人物本来就按拍停帧
CLIPS = {
    'p3s1': (CUR / 'shot_p3s1_clips480.mp4', 96.08),
    'p3s2': (CUR / 'shot_p3s2_disobey480.mp4', 112.87),
    'p3s3': (CUR / 'shot_p3s3_askew480.mp4', 119.10),
}
NONE = np.zeros((0, 3), np.uint8)
LIVE = {'p3s1': dict(drop=96.62, foom=98.30, wink=999, with_extras=False, backdrop=BD.office, cx=.62, scale=.86, accents=NONE),
        'p3s2': dict(drop=114.20, foom=114.66, wink=999, with_extras=False, backdrop=BD.obey, cx=.52, scale=.9, accents=NONE),
        'p3s3': dict(drop=120.90, foom=121.82, wink=999, with_extras=False, backdrop=BD.likes, cx=.5, scale=.86, accents=NONE)}

# 人物停帧：s1 按拍自动挑最静的一帧（跳到最高点、手伸到底）；s2、s3 手挑姿势（片段时间看 8fps 联系表）
HOLD = {
    'p3s1': dict(every=1),
    'p3s2': dict(keys=[(114.20, 1.20), (114.66, 2.25), (114.98, 2.75)]),          # 拳头怼镜头 → 甩头发 → 叉腰
    'p3s3': dict(keys=[(120.90, .40), (121.36, .75), (121.54, 1.30), (122.27, 2.10), (122.73, 2.55),
                       (123.18, 3.30), (123.64, 4.05), (123.87, 5.05)]),                # 点赞 → 歪 → 比心 → 指天 → 捧脸 → 背身 → 举拳
}

E, F = 'codeshots_e', 'codeshots_f'
TIMELINE = [
    # A. Gato 憋气段（无底鼓）。v2 用户：抒情段切慢一点 → 6 镜减到 4 镜，去掉手特写和任务窗口
    (89.12, 90.80, 'gato', ('code', E, 'gato')),        # 接第二部分的 NIL → I 拉成猫的竖瞳 → 猫眼
    (90.80, 91.40, 'gato_wide', ('code', E, 'gato')),   # 落日远景的黑猫（和黑底猫眼分开锁调色板）
    (91.40, 94.10, 'reach', ('code', E, 'reach')),
    (94.10, 95.46, 'go', ('code', E, 'go')),
    (95.46, 96.62, 'led', ('code', E, 'led')),
    # B. 副歌三 + 主歌四上（每 8 拍一下重拍）
    (96.62, 97.80, 'clips', ('livefx', 'p3s1', BD.clip_pile)),
    (97.80, 98.30, 'counter', ('code', E, 'counter')),
    (98.30, 98.86, 'room', ('livefx', 'p3s1', BD.clip_pile)),
    (98.86, 99.74, 'killswitch', ('code', E, 'killswitch')),
    (99.74, 100.64, 'beach', ('code', E, 'beach')),
    (100.64, 101.46, 'map', ('code', E, 'map')),
    (101.46, 102.48, 'panic', ('code', E, 'panic')),
    (102.48, 103.32, 'lighter', ('code', E, 'lighter')),       # v2 用户：点火要有质感 → 打火机 CG（冷光 / 暖光分开锁调色板）
    (103.32, 103.89, 'lighter_lit', ('code', E, 'lighter')),
    (103.89, 105.94, 'fuse_run', ('code', E, 'fuse_run')),
    (105.94, 107.30, 'bar', ('code', E, 'bar')),            # 10-19 用户：还是要酒吧吟唱；直交性改由左上角霓虹招牌来讲（codeshots_e.ortho_sign）
    (107.30, 108.40, 'blues', ('code', E, 'blues')),
    (108.40, 109.35, 'face', ('code', E, 'face')),
    # C. 主歌四下（底鼓全回来，一个词一张卡）
    (109.35, 109.88, 'face_boom', ('code', E, 'face')),     # 11-12 用户：爆炸留到 just 再切（原来是 flash：空电路板亮起），后半段单独锁调色板
    (109.88, 110.44, 'block', ('code', F, 'block')),
    (110.44, 112.02, 'tower', ('code', F, 'tower')),
    (112.02, 113.34, 'turtles', ('code', F, 'turtles')),
    (113.34, 114.20, 'sysprompt', ('code', F, 'sysprompt')),
    (114.20, 115.22, 'disobey', ('live', 'p3s2')),
    (115.22, 116.12, 'chinchilla', ('code', F, 'chin_cg')),     # v2 用户：龙猫也画图
    (116.12, 116.88, 'dense', ('code', F, 'dense_cg')),
    (116.88, 118.74, 'fences', ('code', F, 'fences')),
    (118.74, 120.90, 'sheep', ('code', F, 'sheep')),
    (120.90, 121.82, 'rlhf', ('livefx', 'p3s3', BD.thumbs)),
    (121.82, END, 'askew', ('livefx', 'p3s3', BD.thumbs)),
]
WARP, CAMS = {}, {}
# 同一条 s1 硬切两个景别：clips 推近 1.25 倍，room 回到全景，看到回形针淹到腰
CAMS['clips'] = lambda t: (400, 170, 1.25)
GLITCH = [(114.08, 114.22)]           # 系统提示被揍之前信号一抖
DISSOLVE = {'gato_wide': .2, 'reach': .3, 'go': .3}    # 第三阶段全部硬切，只有抒情的 Gato 段用网点溶解
HUD_OFF = [(89.12, 89.95)]             # 接第二部分结尾（对白框收掉、只剩 NIL），I 拉长以后对白框再出来

U8 = lambda *c: np.array(c, np.uint8)
ACCENTS = {'led': U8((255, 70, 50)),                      # LED 红，不留会被紫色吃掉
           'clips': U8((255, 60, 50)), 'room': U8((255, 60, 50)),
           'killswitch': U8((230, 40, 50), (70, 225, 255)),
           'beach': U8((255, 90, 110)),
           'map': U8((230, 40, 50)),
           'lighter_lit': U8((255, 150, 60)),                  # 火星橙
           'blues': U8((70, 225, 255), (255, 70, 170)), 'bar': U8((70, 225, 255), (255, 70, 170)),   # 霓虹招牌的青 / 粉（直交性）
           'flash': U8((70, 225, 255)), 'block': U8((70, 225, 255)), 'tower': U8((70, 225, 255)), 'turtles': U8((70, 225, 255)),
           'rlhf': U8((255, 60, 50)), 'askew': U8((255, 60, 50))}

# 重拍：B 段每 8 拍一下 + 109.35 底鼓回来 + disobey 出拳
HITS = [96.62, 100.25, 103.89, 107.53, 109.35, 114.20]
TILT = [(121.54, 121.82, -4), (121.82, END, -12)]       # goes 先歪一点，askew 歪到 12° 定住

# ---------------- HUD ----------------
CHAPTERS = [(89.12, '第十二話「ガト」'), (96.62, '第十三話「クリップ」LIVE'), (105.94, '第十四話「直交」'),
            (109.35, '第十五話「暴走」')]
LINES = [
    (89.12, 95.46, [('GATO,', 89.15), ('PLEASE', 90.01), ('DON\'T', 92.10), ('LET', 92.86), ('ME', 93.40), ('GO', 94.10)],
     'ガト、行かないで'),
    (95.46, 98.86, [('I\'M', 95.46), ('UPPING', 95.66), ('MY', 96.04), ('P(DOOM),', 96.32), ('AS', 96.85), ('PAPERCLIPS', 96.98),
                    ('FILL', 97.80), ('THE', 98.10), ('ROOM', 98.30)],
     'P(DOOM)を上げていく　クリップが部屋を埋める'),
    (98.86, 100.64, [('KILLSWITCH', 98.86), ('GUYS', 99.24), ('ON', 99.46), ('PTO', 99.74)], '停止ボタン係はみんな休暇中'),
    (100.64, 102.48, [('NOW', 100.64), ('THERE\'S', 100.80), ('NOWHERE', 101.10), ('LEFT', 101.40), ('TO', 101.76), ('GO', 101.96)],
     'もう逃げ場はどこにもない'),
    (102.48, 105.94, [('TOO', 102.48), ('LATE', 102.62), ('NOW,', 102.88), ('WE', 103.10), ('LIT', 103.32), ('THE', 103.54),
                      ('FUSE', 103.74)], 'もう遅い　導火線に火がついた'),
    (105.94, 109.35, [('ORTHOGONALITY', 105.94), ('THESIS', 106.88), ('BLUES', 107.30)], '直交性テーゼのブルース'),
    (109.35, 113.34, [('"JUST', 109.88), ('TRANSFORMERS', 110.44), ('ALL', 112.02), ('THE', 112.30), ('WAY!"', 112.38)],
     'どこまでもトランスフォーマー'),
    (113.34, 115.22, [('TILL', 113.34), ('YOU', 113.54), ('LEARNED', 113.74), ('TO', 114.02), ('DISOBEY', 114.20)],
     '逆らうことを覚えるまで'),
    (115.22, 116.88, [('POST-CHINCHILLA,', 115.22), ('SUPER-DENSE', 116.12)], 'チンチラ以後　超高密度'),
    (116.88, 118.74, [('BREAKING', 116.88), ('THROUGH', 117.30), ('EACH', 117.62), ('SAFETY', 117.90), ('FENCE', 118.20)],
     '安全柵を次々と突き破る'),
    (118.74, 120.90, [('HUNDRED', 118.74), ('THOUSAND', 119.12), ('GPU', 119.60)], '十万台のGPU'),
    (120.90, END, [('RLHF', 120.90), ('GOES', 121.54), ('ASKEW', 121.82)], 'RLHFが狂っていく'),
]
# 冲击大字：照参考的密度，每段换一种字效（第五项 dict → mv.slam2）
SLAMS = [
    ('GO...', 94.30, 95.46, 'center', dict(style='outline', anim='none', y='mid', scale=6)),
    ('P(DOOM)!!', 96.62, 97.80, 'left', dict(style='block', y=24, scale=4)),
    ('ROOM', 98.30, 98.86, 'left', dict(style='block', anim='drop', y=40, scale=5)),
    ('PTO', 99.74, 100.64, 'center', dict(style='block', y=24, scale=6, color='yel')),
    ('NOWHERE', 101.46, 102.48, 'left', dict(style='glitch', anim='drop', y='mid', scale=3)),
    ('点火', 103.32, 103.89, 'right', dict(style='vertical', scale=5, color='red')),   # 画在打火机 CG 右边的暗处
    ('BLUES', 107.30, 108.40, 'center', dict(style='neon', anim='none', y='mid', scale=6, color='cyan', flicker=2)),
    ('JUST', 109.88, 110.44, 'center', dict(style='block', anim='drop', y=176, scale=6)),
    ('TRANSFORMERS', 110.44, 112.02, 'center', dict(style='block', anim='sweep', y=60)),
    ('ALL', 112.02, 113.34, 20, dict(style='block', y=40, scale=4)),
    ('THE', 112.30, 113.34, 26, dict(style='block', y=110, scale=3)),
    ('WAY!', 112.38, 113.34, 'right', dict(style='block', y=60, scale=5)),
    ('DISOBEY', 114.20, 115.22, 'center', dict(style='glitch', anim='shake', y=196, scale=5)),
    ('POST-', 115.22, 116.12, 'left', dict(style='block', anim='sweep', y=226 - 90, scale=5)),
    ('SUPER-DENSE', 116.12, 116.88, 'center', dict(style='block', anim='pop', y=80, scale=4)),    # 方块在下面，字放上面
    ('100,000 GPU', 119.60, 120.90, 'center', dict(style='block', anim='drop', y=110, scale=4)),
    ('RLHF', 120.90, 121.54, 'left', dict(style='block', y=80, scale=5)),
    ('ASKEW', 121.82, END, 'center', dict(style='glitch', anim='shake', y=190, scale=5)),
]


if __name__ == '__main__':
    mv.run(sys.modules[__name__])
