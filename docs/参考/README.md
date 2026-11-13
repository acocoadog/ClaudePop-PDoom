# 参考资料

这一版 MV 受几部作品启发。它们的视频和源码都不是我们的，仓库里不放，这里只列出处和我们从中学到了什么。

| 作品 | 是什么 | 我们用过什么 |
|---|---|---|
| **90 年代偶像版 MV**（[@pleometric](https://x.com/pleometric)，X） | **这一版最主要的参考作品**：Opus + Seedance 2.5 + 代码 PC-98 化（16 色、Bayer），1920×1080 30fps，原生格子 640×360 | 全片逐帧拆过。开发笔记「核心经验 2、8」的人物动画手法都是从拆解里来的 |
| [JohnHeibel/PDoomVideo](https://github.com/JohnHeibel/PDoomVideo) | 同一首歌的水彩绘本版 MV：p5.brush 全代码绘制，Opus 写分镜、子代理按章节并行 | 学了它的原则（开发笔记 4b 节）；歌词文本最初是从它那里取的（歌词本身属于原曲作者，见 [素材/音频/README.md](../../素材/音频/README.md)） |
| 同一首歌的 Three.js 版 MV（[YouTube](https://www.youtube.com/watch?v=5EoO5413dBY)） | TypeScript + Three.js 代码渲染，逐词卡拉 OK 字幕，带 Demucs 分轨、强制对齐、节拍分析工具 | 用它的词级歌词时间轴做了起点，再自己校对成 [歌词时间轴.md](../../素材/音频/歌词时间轴.md) |
| [ClaudeAnimationBase](../../素材/ClaudeAnimationBase/)（John Heibel，MIT） | p5.brush 画的 Clawd 角色套件 | 渲染片中的 Clawd 群演，本地加了 `src/scenes/xk_extras.js`。随仓库附带，保留原 LICENSE |
