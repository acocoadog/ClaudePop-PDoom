// xk_extras.js(Claude 加的):Clawd 群演循环,纯绿底方便色度键抠图,给 live98.py 当观众/伴舞。
// 每个循环 = 一个 Clawd 居中,长度 2 小节(8 拍),首尾无缝;拍子跟 config.js 的 bpm。
// 渲染(在仓库根目录):
//   node render.mjs --loop=xk_bounce --png --out=out/xk/bounce --chrome="<Chromium>"
// 荧光棒(penlight)用 armR/armL 钩子挂在手上;颜色避开绿色。
(() => {
  const KEY = '#00FF00';
  const LEN = 8 * 60 / PROJECT.bpm;
  const GY = 760, U = 34;  // 地面 y、尺寸单位(大一点,合成时再缩小)

  const penlight = (col) => (u, sw) => {
    paint(rrPts(u * .2, -u * .45, u * 3.2, u * .9, u * .4), { wash: col, ink: PAL.ink, sw: sw * .8 });
    glow(u * 2.2, 0, u * 3.5, col, 90);
  };

  const V = {
    bounce: t => ({ ...move('bounce', t), eyes: 'happy', mouth: 'grin', hat: 'party', aR: .9 + .3 * pulse(t, 6), armR: penlight('#FF5FB4') }),
    roof:   t => ({ ...move('roof', t), eyes: 'shine', mouth: 'open', hat: 'headphones', armR: penlight('#5FD8FF'), armL: penlight('#5FD8FF') }),
    hop:    t => ({ ...move('hop', t), eyes: 'heart', mouth: 'smile', hat: 'bow', blush: .7 }),
    wave:   t => ({ ...move('wave', t), eyes: 'happy', mouth: 'laugh', armL: penlight('#FFD23F') }),
    shimmy: t => ({ ...move('shimmy', t), eyes: 'shades', mouth: 'smirk', hat: 'crown' }),
    spin:   t => ({ ...move('spin', t), eyes: 'spark', mouth: 'O', hat: 'cat' }),
  };
  // 09-30 FOOM 撤离:惊恐脸朝镜头 + 身体前倾狂奔(走路循环 2.5 倍速;侧身在 48px 下看不清表情),
  // 帽子跟各自的伴舞一样,逃跑时认得出是谁。朝右跑,往左逃的在合成时翻转
  const flee = hat => t => ({ ...feel('scared', t), ...move('walk', t * 2.5), rot: .22, eyes: 'scared', mouth: 'wobble', emote: 'sweat', hat });
  Object.assign(V, { flee_roof: flee('headphones'), flee_shimmy: flee('crown'), flee_bounce: flee('party'), flee_hop: flee('bow'), flee_wave: flee(undefined) });

  for (const [name, pose] of Object.entries(V)) {
    LOOPS['xk_' + name] = t => {
      boilSeed('bg');
      paint(rectPts(0, 0, W, H), { wash: KEY, ink: null });
      clawd(W / 2, GY, U, { ...pose(t), noShadow: true });
    };
    LOOPS['xk_' + name].len = LEN;
  }
})();
