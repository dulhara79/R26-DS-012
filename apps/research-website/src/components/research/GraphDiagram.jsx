import { useState } from "react";

// Interactive schematic of C2's graph: one node per (day, segment) cell, with edges
// between adjacent segments within a day and the same segment on consecutive days.
const SEGMENTS = ["Morning", "Afternoon", "Evening", "Night"];
const DAYS = 7;
const X0 = 86;
const Y0 = 44;
const DX = 74;
const DY = 46;

const x = (d) => X0 + d * DX;
const y = (s) => Y0 + s * DY;

const isNeighbour = (a, b) =>
  a &&
  ((a.d === b.d && Math.abs(a.s - b.s) === 1) ||
    (a.s === b.s && Math.abs(a.d - b.d) === 1));

export default function GraphDiagram() {
  const [sel, setSel] = useState(null);

  const within = [];
  const across = [];
  for (let d = 0; d < DAYS; d++) {
    for (let s = 0; s < SEGMENTS.length; s++) {
      if (s < SEGMENTS.length - 1) within.push([d, s]);
      if (d < DAYS - 1) across.push([d, s]);
    }
  }

  const touches = (a, b) =>
    sel && ((sel.d === a.d && sel.s === a.s) || (sel.d === b.d && sel.s === b.s));

  const neighbours = [];
  if (sel) {
    if (sel.s > 0) neighbours.push(`${SEGMENTS[sel.s - 1]} of day ${sel.d + 1} (within-day edge)`);
    if (sel.s < 3) neighbours.push(`${SEGMENTS[sel.s + 1]} of day ${sel.d + 1} (within-day edge)`);
    if (sel.d > 0) neighbours.push(`${SEGMENTS[sel.s]} of day ${sel.d} (across-day edge)`);
    if (sel.d < DAYS - 1) neighbours.push(`${SEGMENTS[sel.s]} of day ${sel.d + 2} (across-day edge)`);
  }

  return (
    <figure className="graph-diagram">
      <svg
        viewBox="0 0 640 250"
        role="group"
        aria-label="Graph of daily time segments. Select a node to see its connections."
      >
        {SEGMENTS.map((label, s) => (
          <text key={label} className="gd-label" x="8" y={y(s) + 4}>
            {label}
          </text>
        ))}
        {Array.from({ length: DAYS }, (_, d) => (
          <text key={d} className="gd-day" x={x(d)} y="18" textAnchor="middle">
            Day {d + 1}
          </text>
        ))}
        {within.map(([d, s]) => (
          <line
            key={`w${d}-${s}`}
            className={"gd-edge gd-edge--within" + (touches({ d, s }, { d, s: s + 1 }) ? " is-active" : "")}
            x1={x(d)} y1={y(s)} x2={x(d)} y2={y(s + 1)}
          />
        ))}
        {across.map(([d, s]) => (
          <line
            key={`a${d}-${s}`}
            className={"gd-edge gd-edge--across" + (touches({ d, s }, { d: d + 1, s }) ? " is-active" : "")}
            x1={x(d)} y1={y(s)} x2={x(d + 1)} y2={y(s)}
          />
        ))}
        {Array.from({ length: DAYS }, (_, d) =>
          SEGMENTS.map((name, s) => {
            const here = { d, s };
            const selected = sel && sel.d === d && sel.s === s;
            return (
              <circle
                key={`n${d}-${s}`}
                className={
                  "gd-node" +
                  (selected ? " is-selected" : "") +
                  (isNeighbour(sel, here) ? " is-neighbour" : "")
                }
                cx={x(d)}
                cy={y(s)}
                r="9"
                tabIndex={0}
                role="button"
                aria-pressed={!!selected}
                aria-label={`Day ${d + 1}, ${name}`}
                onClick={() => setSel(selected ? null : here)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSel(selected ? null : here);
                  }
                }}
              />
            );
          })
        )}
        <text className="gd-more" x="610" y={y(1.5) + 4} textAnchor="middle">
          … 28 days
        </text>
      </svg>

      <figcaption>
        <span className="gd-key gd-key--within" /> Adjacent segments within a day
        <span className="gd-key gd-key--across" /> Same segment, consecutive days
      </figcaption>

      <div className="gd-info" aria-live="polite">
        {sel ? (
          <>
            <h4>
              Day {sel.d + 1} · {SEGMENTS[sel.s]}
            </h4>
            <p>
              One node: the behaviour observed in this day × segment cell, with 80 features
              (40 behavioural values + 40 missingness indicators). Connected to:
            </p>
            <ul>
              {neighbours.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </>
        ) : (
          <p>Select a node to see how it connects to its neighbours.</p>
        )}
      </div>
    </figure>
  );
}
