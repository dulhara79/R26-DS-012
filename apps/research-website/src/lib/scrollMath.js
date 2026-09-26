// Easing helpers for the cinematic scroll rig.

export const clamp = (v, min = 0, max = 1) => Math.min(max, Math.max(min, v));

export const smoothstep = (e0, e1, v) => {
  const x = clamp((v - e0) / (e1 - e0));
  return x * x * (3 - 2 * x);
};

export const lerp = (a, b, t) => a + (b - a) * t;

// Fades in across [a, b] and out across [c, d]; `active` is the combined envelope.
export const segmentInOut = (s, a, b, c, d) => {
  const enter = smoothstep(a, b, s);
  const exit = smoothstep(c, d, s);
  return { enter, exit, active: enter * (1 - exit) };
};
