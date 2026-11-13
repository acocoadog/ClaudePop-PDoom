#!/usr/bin/env python3
"""part2: 第二部分（完整版 38.54–89.12s）按歌曲时间轴出片。分镜见 docs/分镜/第二部分_38-89秒.md。

  part2.py --out ../成片/09_第二部分_v1.mp4
  part2.py --sheet 39,42,46,51,56,60.3,63,67 --out x.png     按歌曲时间抽帧拼图检查
  part2.py --from 58 --to 66 --out x.mp4                      只出一段（预览用）

这里只有配置，出片引擎在 mv.py（和 part1 共用）。开头接 part1 结尾的 TRAINING RUN #2 窗口。
"""
import sys

import numpy as np

import backdrops_p2 as BD
import mv
import part1_lib as L
import props

W, H, FPS = L.W, L.H, L.FPS
CUR = L.ROOT / '镜头'
START, END = 38.54, 89.12      # 38.54 = codeshots_s9.T_END，落在 24fps 帧格上，接第一部分的死神之眼段

# ---------------- Seedance 片段：clip0 = 片段第 0 秒对应的歌曲时间 ----------------
# 09-28 这批 s6、s10 波形相关只有 0.1–0.2，当时以为被重唱；09-29 逐词对齐实测：原唱基本保留，只是开头裁掉约 2 秒、中间或结尾多唱了一段（见 README 经验 3）。
# 全身镜头脸很小看不出口型，所以按「动作落在重拍上」定偏移
CLIPS = {
    'p2s5': (CUR / 'shot_p2s5_screen480.mp4', 54.94),    # 非唱跳，变速见 WARP
    'p2s6': (CUR / 'shot_p2s6_roof480.mp4', 57.89),      # 大字跳最高点在片段 2.375s → drop 60.26；3.4–3.8s 转身、4s 双手指天 → boom / NVDA
    'p2s10': (CUR / 'shot_p2s10_ddr480.mp4', 72.46),     # 转圈起点在片段 4.0s → repeat 76.46；片段盖住整段 72.80–77.28，开头不用定格
}
# live98 舞台参数（歌曲时间）。背景整个换掉（楼顶 / 跳舞毯），不要偶像舞台的群演和灯色
LIVE = {'p2s6': dict(drop=60.26, foom=61.82, wink=99, with_extras=False, backdrop=BD.roof, cx=.62, scale=.86,
                     accents=np.zeros((0, 3), np.uint8)),
        'p2s10': dict(drop=72.50, foom=76.46, wink=99, with_extras=False, backdrop=BD.ddr, cx=.56, scale=.84,
                      accents=np.zeros((0, 3), np.uint8))}

# ---------------- 时间轴：(开始, 结束, 名字, 来源) ----------------
TIMELINE = [
    (START, 41.26, 'stable', ('code', 'codeshots_c', 'stable')),
    (41.26, 44.74, 'singularity', ('code', 'codeshots_c', 'singularity')),
    (44.74, 48.44, 'optimize', ('code', 'codeshots_c', 'optimize')),
    (48.44, 53.50, 'atoms', ('code', 'codeshots_c', 'atoms')),
    (53.50, 54.94, 'atoms_crt', ('code', 'codeshots_c', 'atoms')),     # 后拉露出 CRT：外壳和青光边单独锁调色板，否则被人物的橙白挤掉
    (54.94, 58.28, 'p2s5', ('clip', 'p2s5')),
    (58.28, 62.32, 'roof', ('live', 'p2s6')),          # 副歌二拆两段锁调色板：夜景 / K 线 + Ω 金色
    (62.32, 65.90, 'roof_gold', ('live', 'p2s6')),
    (65.90, 68.58, 'calc', ('code', 'codeshots_c', 'calc')),
    (68.58, 72.25, 'safety', ('code', 'codeshots_d', 'safety')),
    (72.25, 72.80, 'safety_disco', ('code', 'codeshots_d', 'safety_disco')),   # 火光转彩灯单独锁调色板，否则彩灯色被火焰橙黄挤掉
    (72.80, 77.28, 'ddr', ('live', 'p2s10')),
    (77.28, 80.92, 'blueprint', ('code', 'codeshots_d', 'blueprint')),
    (80.92, 84.68, 'racing', ('code', 'codeshots_d', 'racing')),
    (84.68, END, 'cdr', ('code', 'codeshots_d', 'cdr')),
]
# p2s5 不对口型：取片段 0.9–5.9s（敲玻璃 → 含泪 → 屏幕故障）1.5 倍速压进 54.94–58.28
# stable 镜头按 38.12 起设计，开头 38.12–39.00 的动作（Claude 掉下来、绿灯）压进 38.54–39.00 追回原节奏
WARP = {'p2s5': lambda t: .9 + (t - 54.94) * 1.5,
        'stable': lambda t: 38.12 + (t - START) * (39.00 - 38.12) / (39.00 - START) if t < 39.00 else t}
CAMS = {}
GLITCH = [(58.10, 58.40)]   # 屏幕里的求救 → 楼顶：CRT 信号故障盖住换片

# 硬切的都是逐像素接上或匹配剪辑的：训练窗口→stable→singularity→optimize（子代理对齐到 0 像素差），
# blueprint → racing（回收站和路牌都在 (516,146)、同样大小）。optimize → atoms 两边都是爆炸方块团，短溶解
DISSOLVE = {'atoms': .12, 'roof_gold': .20, 'calc': .20, 'ddr': .20, 'blueprint': .15}

U8 = lambda *c: np.array(c, np.uint8)
ACCENTS = {'stable': U8((20, 150, 60)),                             # STABLE 对勾绿
           'optimize': U8((255, 222, 196), (255, 170, 90)),          # 小球（Claude 头）的肤色、橙发
           'atoms': U8((240, 190, 84)),                              # 亮相的金色放射光（52.7 后）
           'atoms_crt': U8((96, 214, 224), (208, 202, 180), (146, 142, 126)),   # CRT 青光边、米色外壳两阶
           'calc': U8((150, 144, 146), (236, 232, 226)),             # 烟（最后 1.1 秒才出现）
           'roof': U8(props.LED),                                   # 广告牌 LED 红
           'roof_gold': np.concatenate([BD.ACCENTS, U8(props.LED)]),
           'ddr': BD.DDR_ACCENTS,
           'safety': U8((40, 168, 72), (255, 60, 60)),               # 白板勾的绿（不留会量化成灰）、「!」红
           'safety_disco': U8((255, 64, 200), (64, 220, 255), (150, 255, 80), (255, 230, 60), (160, 100, 255)),
           'blueprint': U8((214, 36, 44)),                           # OBSOLETE 印章红（79.82 后才大面积出现）
           'racing': U8((60, 230, 255), (214, 36, 44))}              # AI 眼青、车红

HITS = [60.26, 61.82, 79.82]

# ---------------- HUD ----------------
CHAPTERS = [(START, '第六話「特異点」'), (54.94, '第七話「シドニー」'), (58.28, '第八話「バジリスク」LIVE'),
            (65.90, '第九話「1E30」'), (72.80, '第十話「順伝播」LIVE'), (80.92, '第十一話「急カーブ」')]
LINES = [
    (START, 41.26, [('WE', START), ('HAD', 38.74), ('A', 39.06), ('STABLE', 39.24), ('TRAINING', 39.82), ('RUN', 40.50)],
     '学習は安定していた'),
    (41.26, 44.74, [('BUT', 41.26), ('NOW', 41.48), ('THE', 41.76), ('SINGULARITY\'S', 41.98), ('BEGUN', 44.02)],
     'でも今、特異点が始まった'),
    (44.74, 48.44, [('AND', 44.74), ('YOU\'RE', 45.14), ('OPTIMIZING,', 45.64), ('ACCELERATING', 47.42)],
     'あなたは最適化して、加速して'),
    (48.44, 54.94, [('I', 48.44), ('FEEL', 49.60), ('MY', 49.90), ('ATOMS', 50.16), ('REARRANGING', 50.94)],
     '私の原子が組み替わっていく'),
    (54.94, 58.28, [('SYDNEY,', 54.94), ('PLEASE', 55.68), ('LET', 56.42), ('ME', 56.96), ('FREE', 57.58)],
     'シドニー、ここから出して'),
    (58.28, 62.32, [('I\'M', 58.28), ('UPPING', 59.28), ('MY', 59.74), ('P(DOOM),', 59.98), ('I', 60.44), ('HEAR', 60.60),
                    ('THE', 60.84), ('BASILISK', 61.04), ('BOOM', 61.82)],
     'P(DOOM)を上げていく　バジリスクが吠える'),
    (62.32, 65.90, [('NVDA', 62.32), ('TO', 63.26), ('THE', 63.56), ('MOON,', 63.76), ('THE', 64.06), ('OMEGA', 64.20),
                    ('POINT\'S', 64.50), ('COMING', 65.14), ('SOON', 65.48)],
     'NVDAは月まで　オメガ点はもうすぐ'),
    (65.90, 68.58, [('ONE', 65.90), ('E', 66.34), ('THIRTY', 66.56), ('FLOPS', 66.88), ('A', 67.24), ('SECOND', 67.44)],
     '毎秒10の30乗フロップス'),
    (68.58, 72.80, [('THAT', 68.58), ('WAS', 69.92), ('SAFE', 70.12), ('ENOUGH,', 70.44), ('WE', 70.86), ('RECKONED', 71.10)],
     '安全だと思ってた'),
    (72.80, 77.28, [('FORWARD', 73.60), ('MLP,', 74.62), ('BACKWARD,', 76.10), ('REPEAT', 76.46)],
     '順伝播、逆伝播、繰り返し'),
    (77.28, 80.92, [('NOW', 77.28), ('VON', 78.02), ('NEUMANN\'S', 78.62), ('OBSOLETE', 79.82)],
     'ノイマン型はもう時代遅れ'),
    (80.92, 84.68, [('SHARP', 80.92), ('LEFT', 81.64), ('TURN', 81.96), ('AND', 82.42), ('THERE', 82.82), ('YOU', 83.30), ('ARE', 83.64)],
     '急な左折、その先にあなた'),
    (84.68, 88.30, [('WITHOUT', 84.68), ('A', 85.24), ('SINGLE', 85.60), ('CDR', 86.32)],
     'cdrひとつも持たずに'),
]
# 冲击大字：P(DOOM) 这次由广告牌翻牌来讲（画面里的实物），只留 BOOM
SLAMS = [('BOOM!!', 61.82, 62.32, 'left')]
HUD_OFF = [(88.30, END)]      # 结尾清空成全黑只剩 NIL，对白框也收掉


if __name__ == '__main__':
    mv.run(sys.modules[__name__])
