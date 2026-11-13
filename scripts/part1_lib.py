"""part1_lib: 第一部分（完整版 0–38.1s）各镜头共用的工具和接口约定。

## 镜头函数接口（codeshots_*.py 里的每个镜头都按这个写）

    def scene_xxx(t: float) -> np.ndarray   # (H, W, 3) uint8 RGB 全彩，640×360

- t 是**完整版歌曲时间**（秒），不是镜头内时间。
- **纯函数**：同一个 t 永远画出同一帧，不能依赖上一帧、不能有逐帧累积的状态。
  粒子这类东西用固定种子 + 出生时间写成解析公式（hash01(i) 取随机数）。
- 画全彩就行，**不要自己量化**：part1.py 会按镜头锁调色板、做 Bayer 抖动、画 HUD（顶栏和对白框）。
  画面要**平涂、硬阴影、粗墨线**（PC-98 视觉小说风），别用渐变和柔光，否则量化后出色斑。
- 留白：y < 18 是顶栏，y ≥ 284 是对白框，这两块别放重要内容。
- 一拍二：人物和大物件建议按 12fps 更新（t 取 floor(t*12)/12），光效和 HUD 按 24fps。

每个 codeshots 模块导出 SCENES = [(名字, 开始秒, 结束秒, 函数), ...]，
自检用 `python codeshots_x.py --sheet 名字 t1,t2,... --out x.png`（见 sheet()）。
"""
import functools
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pc98ify as P

W, H, FPS = P.W, P.H, P.FPS
ROOT = Path(__file__).resolve().parent.parent          # 项目根 ClaudePop_PDoom/
SONG = str(ROOT / '素材/音频/PDoom_完整版.mp3')
FONT_PATH = P.FONT_PATH                               # DotGothic16 像素字体（16px 原生尺寸最清楚）

# ---------------- 节拍（见 素材/音频/歌词时间轴.md） ----------------
BEAT88, B88_0 = 60 / 88, 0.03        # 前奏主歌 0–16.2s
BEAT132, B132_0 = 60 / 132, 16.632   # ChatGPT 以后


def beat_phase(t):
    """(第几拍, 拍内相位 0–1)。16.4s 前按 88 BPM，之后按 132 BPM。"""
    per, b0 = (BEAT88, B88_0) if t < 16.4 else (BEAT132, B132_0)
    x = (t - b0) / per
    return int(np.floor(x)), x - np.floor(x)


def pulse(t, decay=6.0):
    """拍点包络：每拍打一下 1 → 指数衰减。"""
    return float(np.exp(-decay * beat_phase(t)[1]))


# ---------------- 小工具 ----------------
def hash01(*k):
    """确定性伪随机数 [0,1)。粒子用它代替 random，保证每帧只由 t 决定。"""
    x = sum((v + 1.0) * c for v, c in zip(k, (12.9898, 78.233, 37.719, 4.581)))
    x = np.sin(x) * 43758.5453
    return float(x - np.floor(x))


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.7):
    x = float(np.clip(x, 0, 1)) - 1
    return x * x * ((s + 1) * x + s) + 1


def twos(t):
    """一拍二：12fps 的时间。"""
    return np.floor(t * 12) / 12


@functools.lru_cache(maxsize=None)
def load_rgb(path, w=W, h=H):
    """读图并缩放到 w×h（RGB uint8）。path 相对项目根 ClaudePop_PDoom/ 或绝对路径。"""
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    im = Image.open(p).convert('RGB').resize((w, h), Image.LANCZOS)
    return np.array(im)


@functools.lru_cache(maxsize=None)
def load_rgba(path):
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    return np.array(Image.open(p).convert('RGBA'))


@functools.lru_cache(maxsize=None)
def font(size=16):
    return ImageFont.truetype(FONT_PATH, size)


def draw_text(img, xy, s, color, size=16, scale=1):
    """在 RGB 图上画像素字（关抗锯齿）。scale>1 时 16px 字最近邻放大成方块大字。返回新图。"""
    f = font(size)
    l, t_, r, b = f.getbbox(s)
    w, h = r - l + 2, b - t_ + 2
    m = Image.new('1', (w, h), 0)
    ImageDraw.Draw(m).text((1 - l, 1 - t_), s, fill=1, font=f)
    if scale > 1: m = m.resize((w * scale, h * scale), Image.NEAREST)
    mask = np.array(m)
    out = img.copy()
    x, y = xy
    ys, xs = np.nonzero(mask)
    ys, xs = ys + y, xs + x
    ok = (ys >= 0) & (ys < out.shape[0]) & (xs >= 0) & (xs < out.shape[1])
    out[ys[ok], xs[ok]] = color
    return out


def paste_rgba(dst, rgba, x, y, scale=1.0):
    """把 RGBA 贴到 RGB 上（alpha 二值化，保持硬边）。x,y 是贴图左上角。"""
    if scale != 1.0:
        rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = rgba.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1: return dst
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4] > 127
    out = dst.copy()
    out[y0:y1, x0:x1] = np.where(a, src[..., :3], out[y0:y1, x0:x1])
    return out


def bayer_mask(k):
    """Bayer 溶解遮罩（H×W bool），k∈[0,1]。转场、闪白都用它，别用透明度混合。"""
    return P.bayer_mask(float(np.clip(k, 0, 1)))


# ---------------- 预览（模仿 part1 的最终量化，方便各镜头自检） ----------------
def quantize_preview(frames, k=16):
    """按 part1 同样的方式：整组帧抽样建锁定调色板 → Bayer 抖动 → 放大 + 扫描线。"""
    cels = [P.cel(f) for f in frames]
    pal = P.build_palette(cels, k - len(P.FIXED))
    lut = P.build_lut(pal)
    return [P.upscale(pal[P.quantize(c, lut)]) for c in cels]


def sheet(scenes, name, times, out, raw=False):
    fn = {n: f for n, a, b, f in scenes}[name]
    frames = [fn(float(t)) for t in times]
    tiles = frames if raw else quantize_preview(frames)
    if raw: tiles = [P.upscale(f, False) for f in tiles]
    tiles += [np.zeros_like(tiles[0])] * (len(tiles) % 2)
    rows = [np.concatenate(tiles[r:r + 2], 1) for r in range(0, len(tiles), 2)]
    Image.fromarray(np.concatenate(rows, 0)).resize((1920, 540 * len(rows))).save(out)
    print('wrote', out)


def sheet_cli(scenes):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sheet', nargs=2, metavar=('NAME', 'TIMES'), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--raw', action='store_true', help='不量化，看全彩原图')
    a = ap.parse_args()
    sheet(scenes, a.sheet[0], [float(x) for x in a.sheet[1].split(',')], a.out, a.raw)
