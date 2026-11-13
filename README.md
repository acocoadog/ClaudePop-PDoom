# I'm Upping My P(DOOM) · PC-98 偶像 MV

一支 156 秒的 AI 音乐视频：Claude 当偶像，在 90 年代 PC-98 视觉小说的画风里唱完整首《I'm Upping My P(doom)》。

**▶ 成片：[B 站](https://www.bilibili.com/video/BV1FKan66EdH/)**

> A 156-second music video for "I'm Upping My P(doom)", styled after 1990s PC-98 visual novels. Key art by gpt-image, motion reference by Seedance 2.0, and everything else — 16-color palette locking, Bayer dithering, HUD, typography, dozens of hand-coded shots, editing — is Python written with Claude (Opus 5.5) in Claude Code. Docs are in Chinese.

## 怎么做的

```
① 生图    角色设定图（PC-98 画风）→ 每句歌词一张关键帧，构图给对白框和大字留位置
② Seedance 关键帧当首帧 + 对应的音频切片 → 5 秒左右的动作片段（少动作、卡重拍）
③ 测偏移  Whisper 词级时间戳 / 波形互相关 → 片段第 0 秒对应原曲第几秒
④ 合成    抠图换背景、代码画舞台和特效、一拍二抽帧、锁 16 色调色板 + Bayer 抖动 + 扫描线
⑤ 出片    按原曲时间轴拼镜头、画 HUD 和歌词，四段分别出片后直接拼接，音轨换成原曲
```

画面里，生成式 AI 只负责两样东西：**静态画**（gpt-image 出的关键帧、CG、精灵表）和**人物动作**（Seedance 片段）。画风统一、运镜、转场、字幕、仪表、群演，以及 loss 地形、黑洞、RPG 菜单、赛车这些梗镜头，全是代码画的。每一帧只由歌曲时间决定，任何一秒都能单独算出来预览。

最重要的一条经验：**画风要在生图阶段就定死，代码只做收尾**。拿现代厚涂底片硬转 16 色，怎么修都是色斑；底图本身画成平涂、硬阴影、成束头发的 PC-98 风，同一套量化转出来就干净了。完整的过程和踩过的坑在 [docs/开发笔记.md](docs/开发笔记.md)。

## 目录

| 路径 | 内容 |
|---|---|
| [scripts/](scripts/) | 全部代码。`mv.py` 出片引擎，`part1.py`–`part4.py` 四个部分的时间轴配置，`codeshots_*.py` 代码镜头，`backdrops_p*.py` 舞台背景，`live98.py` 抠图换背景 + 舞台演出，`pc98ify.py` 调色板与抖动，`render_all.sh` 一键出全片 |
| [素材/角色/](素材/角色/) | 角色设定图（日常服定稿、舞台服、练习服），所有关键帧和 CG 都以它们为参考图 |
| [素材/关键帧/](素材/关键帧/) | 喂给 Seedance 的首帧 |
| [素材/CG/](素材/CG/) | 视觉小说式的静态 CG、改图差分 |
| [素材/精灵/](素材/精灵/) | Q 版 Claude、蘑菇、龙猫的精灵表 |
| [素材/群演/](素材/群演/) | Clawd 群演舞蹈循环（RGBA），由 `clawd_extras.py` 从 ClaudeAnimationBase 渲染结果生成 |
| [素材/ClaudeAnimationBase/](素材/ClaudeAnimationBase/) | John Heibel 的 Clawd 角色套件（MIT），加了群演场景 `src/scenes/xk_extras.js` |
| [素材/音频/](素材/音频/README.md) | 原曲（随仓库附带，权利归原作者）和歌词时间轴 |
| [素材/字体/](素材/字体/README.md) | DotGothic16（OFL）。**Zpix 要自己下载** |
| [镜头/](镜头/) | 成片用到的 Seedance 片段和抠图缓存；`index.jsonl` 是每条的提示词、参数和费用 |
| [封面/](封面/) | 封面图，`scripts/cover.py` 生成 |
| [docs/开发笔记.md](docs/开发笔记.md) | 工作笔记：流程、脚本用法、核心经验、踩坑 |
| [docs/提示词.md](docs/提示词.md) | 全部生图 / Seedance 提示词，〇节是统一的风格模板 |
| [docs/分镜/](docs/分镜/README.md) | 成片的逐镜头分镜：每个镜头的时间、歌词、画面和对应代码，外加 P(DOOM) 读数在各镜头里的变化 |
| [docs/参考/](docs/参考/README.md) | 启发这部片的作品和出处 |

素材只收了成片里真正用到的：渲染全片时读到的文件，加上这些片段的首帧和角色设定图，试过没用上的版本都没放。`素材/*/index.jsonl` 记着每张图的提示词、参考图和模型，想知道某张图怎么画出来的直接查。

## 自己跑一遍

需要 Python 3.11+、ffmpeg。在 macOS 上开发，其他系统要把 `codeshots_b.py` 里的中文字体路径换掉。

```bash
pip install -r requirements.txt
# 原曲已附带（素材/音频/PDoom_完整版.mp3）
# 下载 Zpix 字体：素材/字体/zpix.ttf（见 素材/字体/README.md，只有尾奏花絮用到）

cd scripts
python part1.py --sheet 3,14.2,23.95 --out x.png   # 先出几帧联系表看看
python part1.py --from 20 --to 27 --out ../成片/试.mp4   # 只出一小段
bash render_all.sh                                   # 出全片（M 系列 Mac 上十分钟左右）
```

各脚本的参数和用法见[开发笔记的「脚本」一节](docs/开发笔记.md#脚本)。

关于生成步骤：`scripts/draw_part*.sh`、`seed_part*.sh` 里的 `xk-draw` / `xk-video` 是我们自己的命令行小工具（分别包装 gpt-image 和火山方舟 Seedance 的接口），没有一起开源。脚本留着是为了让提示词、参考图和参数一目了然；生成好的素材都已经在仓库里，不重新生成就用不到它们。

## 署名与授权

- **原曲**："I'm Upping My P(doom)"，作词 osmarks，开头主歌和副歌出自 MusicPerson，另有 EleutherAI 社区成员和 Claude 参与；原版用 Udio 生成，2024 年 11 月发布。本片用的是 deckard (@slimer48484) 2026 年 9 月在 X 上发布的 Suno「Claude-Pop」版。详见 [素材/音频/README.md](素材/音频/README.md)。原曲随仓库附带只是为了能出片，录音和歌词的权利都归原作者，不在 MIT 授权范围内（见 [LICENSE](LICENSE)）。
- **灵感**：@pleometric 的 90 年代偶像版 MV、JohnHeibel 的 [PDoomVideo](https://github.com/JohnHeibel/PDoomVideo)，见 [docs/参考/](docs/参考/README.md)。
- **Clawd 角色套件** [素材/ClaudeAnimationBase/](素材/ClaudeAnimationBase/)：John Heibel，MIT，保留原 LICENSE。
- **字体**：DotGothic16，SIL OFL 1.1。
- **本仓库的代码和文档**：MIT，见 [LICENSE](LICENSE)。AI 生成的图片和视频片段按同样的条款提供。
- Claude、Clawd 是 Anthropic 的角色和商标，本项目是粉丝创作，与 Anthropic 无关。
