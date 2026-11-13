# 成片

渲染输出放这里（已在 `.gitignore` 里忽略）。`bash scripts/render_all.sh` 会在这里写出 `part1.mp4`–`part4.mp4` 和拼好的 `全片_0-156秒.mp4`。

`live98.py` 出片时会在视频旁边写一份 `<成片>.recipe.json`：完整参数、脚本指纹和输入素材，脚本默认值以后变了也能追溯这条是怎么出来的。
