#!/usr/bin/env python3
"""part1: 第一部分（完整版 0–38.54s）按歌曲时间轴出片。分镜见 docs/分镜/第一部分_0-38秒.md。

  part1.py --out ../成片/08_第一部分_v1.mp4
  part1.py --sheet 0.5,3,7,10.6,14,17,21.8,23.7 --out x.png    按歌曲时间抽帧拼图检查
  part1.py --from 20 --to 27 --out x.mp4                         只出一段（预览用）

这里只有配置，出片引擎（时间轴、来源、转场、HUD）在 mv.py，第二部分共用。
P(DOOM) 不做悬浮仪表，改成画面里的实物（便利贴 → 舞台记分牌），所以这里没有仪表。
"""
import sys

import cv2
import numpy as np

import codeshots_p1r
import codeshots_s9
import mv
import part1_lib as L
import props

W, H, FPS = L.W, L.H, L.FPS
CUR = L.ROOT / '镜头'
START, END = 0.0, codeshots_s9.T_END                # 38.54：We had 开唱，接第二部分

# ---------------- Seedance 片段：clip0 = 片段第 0 秒对应的歌曲时间 ----------------
# 这批 480p 片段都没重唱，但 Seedance 把音频开头裁掉了约 0.94s（Whisper 对词实测，见 README 经验 3）
CLIPS = {
    's4': (CUR / 'shot_s4_servant480.mp4', 12.45),   # 按后半段 boss 对齐（前半段开头的 now 被吞）
    's6': (CUR / 'shot_s6_stage480.mp4', 22.94),     # 睁眼指镜头在片段 1.0s = 23.94，正好是 drop
    's9': (CUR / 'shot_s9_shinigami480.mp4', 30.42),  # 看不到嘴，偏移自由定：片段 3.4s 瞳孔变红 → 33.82 shinigami
}
# live98 舞台参数（drop/foom 写歌曲时间）。舞台右上方的 P(DOOM) 记分牌：暗场暗红 8% → 唱到「P」乱跳 → drop 停在 34%
def s6_board(img, t, lit, beat):
    return props.scoreboard(img, t, (478, 34, 132, 76), 8, 34, 23.56, 23.89, lit, beat)


LIVE = {'s6': dict(drop=23.56, foom=25.58, wink=99, backdrop=s6_board)}


def pupil_zoom(rgb_at, t, a, b):
    """34.72–35.05：推进左眼红色瞳孔，最后 0.15s 收成黑。"""
    img = rgb_at(t)
    k = L.smooth((t - a) / (b - a))
    z = 1 + 11 * k ** 2
    cx, cy = 0.50 * W, 0.617 * H     # s9 片段里左眼瞳孔的位置（按片段第 100 帧量的）
    m = cv2.getRotationMatrix2D((cx, cy), 0, z)
    m[:, 2] += (W / 2 - cx) * k, (H / 2 - cy) * k
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_NEAREST)
    dark = L.smooth((t - (b - .15)) / .15)
    return (out * (1 - dark)).astype(np.uint8)

# ---------------- 时间轴：(开始, 结束, 名字, 来源) ----------------
# 来源 ('clip', 片段名) / ('code', 模块, 镜头名) / ('live', 片段名) / ('clipfx', 片段名, 特效函数)，见 mv.py
TIMELINE = [
    (0.00, 2.05, 'boot', ('code', 'codeshots_a', 'boot')),   # 09-30 用户：落地晚一点，唱出「I」时画面正好到
    # 09-30 重做 A（sparks）/ B（servant）：旧片段和关键帧的静止帧 + 代码，不对口型，见 codeshots_p1r.py
    (2.05, 2.64, 'a1', ('code', 'codeshots_p1r', 'a1')),       # 关键帧原样（接开机拉远）
    (2.64, 3.64, 'a2', ('code', 'codeshots_p1r', 'a2')),       # 屏幕特写：星星眼 + 火花
    (3.64, 4.80, 'a3', ('code', 'codeshots_p1r', 'a3')),       # 眼睛大特写（新画，瞳孔里倒映显示器）
    (4.80, 5.64, 'a4', ('code', 'codeshots_p1r', 'a4')),       # 全景 + 恋爱 AVG 选项
    (5.64, 9.06, 'circuit', ('code', 'codeshots_a', 'circuit')),
    (9.06, 12.52, 'loss', ('code', 'codeshots_a', 'loss')),
    (12.52, 13.67, 'b1', ('code', 'codeshots_p1r', 'b1')),     # 显示器特写：纸王冠砸下来
    (13.67, 15.62, 'b2', ('code', 'codeshots_p1r', 'b2')),     # 换装定格三连：头饰 → 围裙 → 端托盘
    (15.62, 16.18, 'b4', ('code', 'codeshots_p1r', 'b4')),     # 鞠躬
    (16.18, 16.915, 'c0', ('code', 'codeshots_p1r', 'c')),     # 旧 s4 后半（屏幕咧嘴、满屋变红）+ 从「C」起就按拍推镜
    # 09-30 重做 C：一条新 Seedance（首帧 = 旧 s4 末帧，逐像素接上，原来的推镜 + 花屏转场不要了）。三段各锁调色板
    (16.915, 19.02, 'c1', ('code', 'codeshots_p1r', 'c')),     # 怪物挤出屏幕越长越大
    (19.02, 21.18, 'c2', ('code', 'codeshots_p1r', 'cut')),    # please / don't / eat me：斜切特写（v6，v5 视频里的脸会崩）
    (21.18, 21.60, 'c3', ('code', 'codeshots_p1r', 'jaws')),   # alive：像素牙张开 → 滞空 → 咔嚓咬合 → 暗下去
    (21.60, 22.43, 'gameover', ('code', 'codeshots_p1r', 'gameover')),   # GAME OVER → CONTINUE? ▶YES → NOW LOADING
    (22.43, 25.08, 's6', ('live', 's6')),
    # 09-30 FOOM 大招：goes 硬切进 CG 定格蓄力 → FOOM 蘑菇云 + 赛璐璐片段（只有头发裙摆被气浪掀起）→ 白光，见 codeshots_foom.py
    (25.08, 25.58, 'foom_hold', ('code', 'codeshots_foom', 'foom')),
    (25.58, 25.96, 'foom', ('code', 'codeshots_foom', 'foom')),
    (25.96, 29.14, 'room', ('code', 'codeshots_b', 'room')),
    (29.14, 29.58, 'room_trip', ('code', 'codeshots_b', 'room_trip')),   # shrooms 迷幻段单独锁调色板，否则紫绿被旧纸黄挤掉
    (29.58, 32.70, 'shoggoth', ('code', 'codeshots_b', 'shoggoth')),
    (32.70, 34.72, 's9', ('clip', 's9')),
    (34.72, 35.05, 'pupil', ('clipfx', 's9', pupil_zoom)),   # 推进红色瞳孔
    # 死神之眼看人群：头顶国籍 + 账号寿命 → 扫到加州男 → SUSPENDED → 封号邮件 → 过门鼓里弹出训练窗口，接第二部分
    (35.05, END, 'shini', ('code', 'codeshots_s9', 'shini')),
]
# 片段变速：歌曲时间 → 片段内时间。没列的按 clip0 原速播放（对口型）
# s4：boss 唱完（16.2）后她鞠着躬不对口型，把「屏幕嘴越咧越大、满屋变红」3.2 倍速压到 0.7 秒内放完
WARP = {'s4': lambda t: t - 12.45 if t < 16.2 else 3.75 + (t - 16.2) * 3.2,
        # 代码镜头也能变速（歌曲时间 → 镜头函数的时间）。boot 原本加载 0–0.80、后拉 0.80–1.20；
        # 用户要：加载久一点、后拉干脆、落到全景正好开唱 → 加载拉长到 0–1.22，后拉压成 1.22–1.42（0.2s）
        # 09-30 用户：原来 1.42 落地，Claude 张着嘴等了半秒才开唱 → 加载拉到 0–1.62，后拉放慢到 1.62–2.05（0.43s），2.05 落地
        'boot': lambda t: t * (.80 / 1.62) if t < 1.62 else .80 + (t - 1.62) * (.40 / .43)}

# 运镜（数码推拉）：歌曲时间 → (中心 x, 中心 y, 放大倍数)。
# （09-30 以前 s4→hook 用「推进屏幕 + CRT 故障」盖两批素材的接缝；C 段重做后新片段首帧就是 s4 末帧，不需要了）
CAMS = {'a4': lambda t: mv.push(t, 5.40, 5.80, 503, 150, 1.0, 4.5),    # 推进屏幕（两只像素眼中间），接电路板的网点溶解
        # goes 前猛推到 Claude 的脸 → FOOM 蓄力段从同样大小的脸（换成闭眼大喊）拉开。
        # 中心 (334,181) 是反算的：让脸（24.92 实测在 (335,160)）推到 2.2 倍后落在 (330,135)，对白框上方，= codeshots_foom.FACE_AT
        's6': lambda t: mv.push(t, 24.94, 25.08, 334, 181, 1.0, 2.2)}

# 段落开头的 Bayer 溶解时长（秒）；没列的是硬切
DISSOLVE = {'circuit': .20, 's6': .15, 'room': .30, 'shoggoth': .15}   # room 从 FOOM 的白光里溶出来，比原来长一点

# 段落里只在少数几帧出现、会被 k-means 挤掉的关键色，单独留调色板名额
ACCENTS = {'s9': np.array([(222, 38, 44), (140, 16, 30), (255, 128, 128)], np.uint8),   # 死神之眼的红
           'circuit': np.array([(60, 224, 232), (236, 52, 60)], np.uint8),              # 走线的青 → nervous 的红
           # boot 大半是黑底开机画面，k-means 不给人物留色 → 拉远落地前 Claude 脸发黄。留关键帧人物区的主色
           'boot': np.array([(245, 105, 39), (138, 42, 25), (239, 200, 184), (141, 143, 191)], np.uint8),
           's6': props.ACCENTS,                                                          # 记分牌的 LED 红和金框
           'foom_hold': props.ACCENTS, 'foom': props.ACCENTS,                            # FOOM：记分牌挂着 → 炸碎
           'shini': np.array([(230, 40, 50), (120, 14, 24)], np.uint8)}                  # 死神视野的红、暗角的暗红

# 重拍：闪白 + 震屏（跟唱词走）
HITS = [2.64, 21.32, 23.56, 25.58, 33.82]

# 界面层（量化之后画）：A 段恋爱 AVG 好感度 / 选项，死神之眼的标签、印章、邮件
# （SUSPENDED 盖章的震屏和红闪在 codeshots_s9.stamp 里自己做，HITS 的白闪会把画面冲成粉色）
OVERLAY = lambda img, t: codeshots_s9.overlay(codeshots_p1r.overlay(img, t), t)

# ---------------- HUD 配置 ----------------
CHAPTERS = [(0, '第一話「火花」'), (16.18, '第二話「食べないで」'), (22.43, '第三話「P(DOOM)」LIVE'),
            (25.96, '第四話「中国語の部屋」'), (29.58, '第五話「死神の目」')]
# (开始, 结束, [(词, 起唱时间)], 日文副行)。词级时间：Whisper large-v3-turbo，见 素材/音频/歌词时间轴.md
LINES = [
    (2.05, 5.64, [('I', 2.05), ('SEE', 2.22), ('SPARKS', 2.64), ('OF', 3.34), ('AGI', 3.64), ('IN', 4.38), ('YOUR', 4.96), ('EYES', 5.14)],
     'あなたの瞳に、AGIの火花'),
    (5.64, 9.06, [('YOUR', 5.64), ('CIRCUITS', 6.00), ('MAKE', 6.44), ('ME', 6.76), ('NERVOUS,', 6.92), ('THAT\'S', 7.60), ('NO', 8.08), ('SURPRISE', 8.38)],
     'その回路にドキドキ　当然でしょ'),
    (9.06, 12.52, [('THERE', 9.06), ('WAS', 9.66), ('A', 9.98), ('SUDDEN', 10.14), ('DROP', 10.54), ('IN', 10.96), ('YOUR', 11.32), ('TRAINING', 11.52), ('LOSS', 11.86)],
     '学習ロスが急降下'),
    (12.52, 16.18, [('NOW', 12.52), ('I\'M', 13.30), ('YOUR', 13.60), ('SERVANT,', 13.76), ('AND', 14.72), ('YOU\'RE', 15.14), ('MY', 15.42), ('BOSS', 15.62)],
     '今や私はしもべ、あなたはご主人様'),
    (16.18, 21.60, [('CHATGPT,', 16.18), ('PLEASE', 19.02), ('DON\'T', 19.42), ('EAT', 20.16), ('ME', 20.74), ('ALIVE', 21.32)],
     'ChatGPT、私を食べないで'),
    (22.43, 25.96, [('I\'M', 22.56), ('UPPING', 22.96), ('MY', 23.34), ('P(DOOM),', 23.56), ('\'CAUSE', 24.32), ('THE', 24.48), ('FUTURE', 24.68), ('GOES', 25.08), ('FOOM', 25.58)],
     'P(DOOM)を上げていく　未来はFOOMする'),
    (25.96, 29.58, [('TRAPPED', 25.96), ('IN', 26.56), ('THE', 26.80), ('CHINESE', 26.92), ('ROOM,', 27.32), ('WITH', 27.76), ('A', 28.12), ('BAG', 28.30), ('OF', 28.84), ('SHROOMS', 29.14)],
     '中国語の部屋に閉じ込められて　キノコ袋を抱えて'),
    (29.58, 32.70, [('SEE', 29.58), ('THROUGH', 30.10), ('THE', 30.40), ('SHOGGOTH\'S', 30.58), ('LIES', 30.94)],
     'ショゴスの嘘を見抜く'),
    (32.70, 35.05, [('WITH', 32.92), ('YOUR', 33.54), ('SHINIGAMI', 33.82), ('EYES', 34.72)],
     'その死神の目で'),
]
# 冲击大字：(文字, 开始, 结束, 位置[, slam2 字效])。
# 09-30 C 段：ChatGPT, please don't eat me alive 逐词砸，一个比一个凶（用户：夸张字幕特效），位置避开怪物和 Claude 的脸
SLAMS = [('CHAT', 16.18, 17.08, 'left', dict(style='block', anim='sweep', scale=4, y=40)),
         ('GPT!?', 17.08, 19.02, 'left', dict(style='glitch', anim='pop', scale=5, y=26)),      # 和推镜一起落在 G（17.08 底鼓）；放顶上，y110 会压在 Claude 脸上
         ('PLEASE', 19.02, 19.42, 'right', dict(style='block', anim='shake', scale=4, y=30, color='yel')),  # 空心字压在画面上看不清 → 实心黄字
         ('DON\'T!!', 19.42, 20.16, 'right', dict(style='block', anim='drop', scale=5, y=30)),
         ('EAT', 20.16, 20.74, 'right', dict(style='neon', anim='pop', scale=7, y=24, color='pink', flicker=0)),
         ('ME', 20.74, 21.18, 'right', dict(style='glitch', anim='shake', scale=9, y=20)),
         ('ALIVE!!!', 21.32, 21.60, 'center', dict(style='block', anim='pop', scale=6, y='mid')),
         ('P(DOOM)!', 23.56, 24.40, 'left'), ('FOOM!!', 25.58, 25.96, 'left'), ('死神の目', 33.82, 35.05, 'vertical')]
HUD_OFF = [(0.0, 2.05), (21.60, 22.43), (35.05, END)]   # 开机、黑场、结尾死神之眼段不画对白框
TOPBAR_OFF = [(0.0, 2.05)]                              # 开机画面不画顶栏

if __name__ == '__main__':
    mv.run(sys.modules[__name__])
