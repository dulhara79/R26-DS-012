// Horizontal AUROC bars on a 0.40–0.70 axis with a chance (0.50) marker.
// Makes C2's central finding readable at a glance: no model clears chance by much,
// and GATv2 does not beat the flat baselines.
const MIN = 0.4;
const MAX = 0.7;
const pct = (v) => `${((v - MIN) / (MAX - MIN)) * 100}%`;

export default function AurocChart({ rows, highlight, nullMean }) {
  const items = rows
    .map(([label, value]) => ({ label, value: Number(value) }))
    .sort((a, b) => b.value - a.value);

  return (
    <figure className="auroc-chart">
      <figcaption className="auroc-title">
        Held-out AUROC by model
      </figcaption>
      <div className="auroc-plot">
        <div className="auroc-markers" aria-hidden="true">
          <span className="auroc-chance" style={{ left: pct(0.5) }}>
            <em>chance 0.50</em>
          </span>
          {nullMean != null && (
            <span className="auroc-null" style={{ left: pct(nullMean) }} />
          )}
        </div>
        <ul>
          {items.map(({ label, value }) => (
            <li
              key={label}
              className={highlight === label ? "is-highlight" : undefined}
            >
              <span className="auroc-label">{label}</span>
              <span className="auroc-track">
                <span
                  className="auroc-bar"
                  style={{ width: pct(value) }}
                />
              </span>
              <span className="auroc-value">{value.toFixed(4)}</span>
            </li>
          ))}
        </ul>
      </div>
      <p className="source-note">
        Bars start at 0.40. Flat baselines sit only slightly above chance; the
        graph model does not outperform them.
      </p>
    </figure>
  );
}
