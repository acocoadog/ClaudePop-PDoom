"""findoffset.py <clip.mp4> [song.mp3]：用波形互相关找出片段在原曲中的起点（秒）。"""
import sys, subprocess, numpy as np
def pcm(path, sr=8000):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f', 's16le', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768, sr
from pathlib import Path
DEFAULT_SONG = str(Path(__file__).resolve().parent.parent / '素材/音频/PDoom_完整版.mp3')
clip, sr = pcm(sys.argv[1]); song, _ = pcm(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SONG)
n = 1 << int(np.ceil(np.log2(len(song) + len(clip))))
c = np.fft.irfft(np.fft.rfft(song, n) * np.conj(np.fft.rfft(clip, n)), n)[:len(song)]
k = int(c.argmax()); seg = song[k:k + len(clip)]
r = float(np.dot(seg, clip[:len(seg)]) / np.sqrt((seg ** 2).sum() * (clip[:len(seg)] ** 2).sum() + 1e-9))
print(f'offset={k / sr:.3f}s  corr={r:.2f}  clip_dur={len(clip) / sr:.2f}s')
