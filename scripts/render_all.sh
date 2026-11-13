#!/usr/bin/env bash
# 出全片：四个部分分别出片 → 视频流直接拼接（不重编码）→ 音轨整条换成原曲。
# 用法（在仓库根目录或 scripts/ 下都行）：bash scripts/render_all.sh [python 解释器]
# 需要 素材/音频/PDoom_完整版.mp3（见 素材/音频/README.md）。整片在 M 系列 Mac 上十分钟左右。
set -euo pipefail
cd "$(dirname "$0")"
PY=${1:-python3}
SONG=../素材/音频/PDoom_完整版.mp3
OUT=../成片
[ -f "$SONG" ] || { echo "缺原曲：$SONG（见 素材/音频/README.md）"; exit 1; }
mkdir -p "$OUT"

parts=(part1 part2 part3 part4)
for i in "${!parts[@]}"; do
  p=${parts[$i]}
  echo "[$((i + 1))/5] 出片 $p"
  "$PY" "$p.py" --out "$OUT/$p.mp4"
done

echo "[5/5] 拼全片"
D=$(cd "$OUT" && pwd)
printf "file '%s'\n" "$D/part1.mp4" "$D/part2.mp4" "$D/part3.mp4" "$D/part4.mp4" > "$OUT/list.txt"
ffmpeg -v error -y -f concat -safe 0 -i "$OUT/list.txt" -i "$SONG" \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -t 156.65 "$OUT/全片_0-156秒.mp4"
echo "完成：成片/全片_0-156秒.mp4"
