#!/bin/zsh
# 第三部分 Seedance。用法: seed_part3.sh s1|s2|s3 [--yes]（不加 --yes 只出计划，不花钱）
# 三条都是唱跳：全身、大动作、传音频。片段秒数按「Seedance 裁掉音频开头约 0.94s」换算：
#   s1 片段 0s ≈ 歌曲 96.24（drop 96.62 = 0.4s）；s2 ≈ 113.54（disobey 114.20 = 0.66s）；s3 ≈ 120.74（askew 121.82 = 1.1s）
# 出来以后按拍抽关键姿势停帧（mv.HOLD），动作要大、姿势要分明，别做连续的小晃动
cd "${0:A:h}/.."   # 项目根（脚本所在 scripts/ 的上一级）
K=素材/关键帧; A=素材/音频
TAIL="Use @图像1 as the first frame. Lip sync strictly to @音频1: keep the original vocals, lyrics and timing exactly, do not re-sing or change the song."
HEAD="2D retro anime, exactly matching the reference image style, costume and character. Full-body shot, the whole figure always in frame from head to boots, energetic 1990s idol dance choreography synced to the beat, singing with lip sync. Limited animation: strong clear held poses that snap from one to the next on the beat. Locked camera, single shot, no cuts."
C=(xk-video --model 2.0-fast --res 480p --ratio 16:9 --duration 6)
case $1 in
# I'm upping my P(doom), as paperclips fill the room, killswitch guys on PTO。音频 95.30–101.30
s1) $C --ref $K/p3_s1_回形针全身.png --audio $A/p3s1_95.30-101.30.mp3 --out 镜头/shot_p3s1_clips480.mp4 ${@:2} "$HEAD 0–0.4s: she crouches low, fists pulled in, gathering energy. 0.4s, on the drop: she leaps up with arms and legs spread in a star shape and lands in a wide power stance. 1.0s, on \"clips\": she punches one fist straight up at the ceiling. 1.6s: one fast full spin, hair flaring out. 2.1s, on \"room\": she sweeps both arms out wide in a big circle. 2.6–3.4s: big side steps left and right with wide arm swings. 3.5s: she points straight at the camera. 4.0s: she stomps and throws both arms up in a V. 4.5–6s: hip-sway dance on the beat, ends with a wink. $TAIL";;
# (all the way.) Till you learned to disobey. Post-Chinchilla, super-dense。音频 112.60–118.60
s2) $C --ref $K/p3_s2_踢腿全身.png --audio $A/p3s2_112.60-118.60.mp3 --out 镜头/shot_p3s2_disobey480.mp4 ${@:2} "$HEAD 0–0.6s: she pulls her right fist far back, winding up, defiant smirk. 0.66s, on \"disobey\": she lunges forward and throws a huge straight punch toward the camera, freezing in that pose. 1.2s: she snaps back upright and flips her hair with one hand. 1.7s: hands on hips, chin up, proud pose. 2.1–2.6s: two sharp stomps on the beat. 2.6s: a small jump with both fists up. 3.3–6s: fast running in place with pumping arms, then a sharp side pose pointing at the camera. $TAIL";;
# (GPU.) RLHF goes askew (+ 过门)。音频 119.80–125.80
s3) $C --ref $K/p3_s3_点赞全身.png --audio $A/p3s3_119.80-125.80.mp3 --out 镜头/shot_p3s3_askew480.mp4 ${@:2} "$HEAD 0–0.8s: she bounces on her knees giving two big thumbs up to the camera, smug grin. 1.1s, on \"askew\": she leans her whole body far over to one side like a tilting tower, arms stretched out, both feet on the floor. 1.5s: snaps back upright and makes a heart with both hands in front of her chest. 2.0s: points one finger up at the sky. 2.5s: both hands framing her face, big wink. 3.0s: a half spin to face the camera again. 3.5–6s: she crouches, then jumps up with one fist raised, then hip-sway dance on the beat. $TAIL";;
esac
