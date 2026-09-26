// Three layered hills whose ridge lines carry heartbeat spikes; sits along the
// bottom of each inner-page header.

const round = (value) => Math.round(value * 10) / 10;

// Heartbeat-shaped offset centred on `at`: small P wave, sharp QRS, soft T wave.
function beat(x, at, height) {
  const g = (centre, width) => Math.exp(-(((x - centre) / width) ** 2));
  return (
    -height * 0.12 * g(at - 70, 16) +
    height * 0.22 * g(at - 18, 6) -
    height * g(at, 7) +
    height * 0.3 * g(at + 16, 6) -
    height * 0.2 * g(at + 80, 22)
  );
}

function ridgePath({ width, height, base, amp, waves, phase, beats }) {
  const points = [];
  for (let x = 0; x <= width; x += 6) {
    const t = x / width;
    let y =
      base +
      Math.sin(t * Math.PI * waves + phase) * amp +
      Math.sin(t * Math.PI * waves * 2.7 + phase * 1.9) * amp * 0.35;
    beats.forEach((at) => {
      y += beat(x, at, amp * 2.6);
    });
    points.push(`${x},${round(y)}`);
  }
  return {
    line: `M${points.join(" L")}`,
    fill: `M0,${height} L${points.join(" L")} L${width},${height} Z`,
  };
}

const ridges = [
  {
    base: 430,
    amp: 38,
    waves: 3,
    phase: 0.4,
    beats: [700, 1750],
    color: "#a3c7cf",
  },
  {
    base: 560,
    amp: 46,
    waves: 4.4,
    phase: 1.7,
    beats: [1160],
    color: "#6fa5a5",
  },
  {
    base: 700,
    amp: 34,
    waves: 5.5,
    phase: 2.6,
    beats: [420, 1920],
    color: "#3d7775",
  },
].map((ridge) => ({
  ...ridge,
  ...ridgePath({ width: 2400, height: 900, ...ridge }),
}));

export default function SignalRidge() {
  const near = ridges[ridges.length - 1];
  return (
    <svg viewBox="0 0 2400 900" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id="ridge-haze" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
        </linearGradient>
      </defs>
      {ridges.map((ridge) => (
        <g key={ridge.base}>
          <path d={ridge.fill} fill={ridge.color} />
          <path d={ridge.fill} fill="url(#ridge-haze)" />
        </g>
      ))}
      <path
        d={near.line}
        fill="none"
        stroke="#fdf1e1"
        strokeOpacity="0.6"
        strokeWidth="3"
        strokeLinejoin="round"
      />
    </svg>
  );
}
