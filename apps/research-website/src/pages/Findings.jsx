import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";
import { components } from "../data/components";

// Cross-component lessons and limitations, as framed in the R26-DS-012 README
// and carried over from the v1 site's curated narrative.
const lessons = [
  "Personalization matters.",
  "Leakage-free validation matters.",
  "Complex models are not automatically superior.",
  "Modalities should not receive equal influence simply because they exist.",
  "Missing evidence is not evidence of low anxiety.",
  "Uncertainty must be represented explicitly.",
];

const limitations = [
  "Research prototype; not a diagnostic device.",
  "Datasets do not represent every population.",
  "Modalities may be absent, stale or unreliable.",
  "Behavioural generalization remains limited under current evidence.",
  "Forecasting scope is defined and should not be exaggerated.",
  "Further external and prospective validation is required where applicable.",
];

export default function Findings() {
  return (
    <>
      <PageHero
        eyebrow="Findings"
        title="What held up, and what did not."
        lead="Each result is shown with its dataset, protocol and stage. Negative results are reported with the same weight as positive ones."
        image="bannerFindings"
      />

      {components
        .filter((c) => c.results)
        .map((c, index) => (
          <section
            className={`section ${index % 2 ? "section--paper2" : ""}`}
            key={c.id}
            aria-labelledby={`${c.id}-results`}
          >
            <div className="shell">
              <SectionHead
                id={`${c.id}-results`}
                eyebrow={`${c.id} · ${c.title}`}
                title={c.resultsLabel}
                lead={c.status[0]}
              />
              <Reveal className={c.results.length > 1 ? "grid-4" : "grid-2"}>
                {c.results.map(([value, label]) => (
                  <div className="stat" key={label}>
                    <span className="stat-value">{value}</span>
                    <span className="stat-label">{label}</span>
                  </div>
                ))}
              </Reveal>
              {c.baselines && (
                <Reveal className="table-wrap" style={{ marginTop: 48 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th scope="col">Model</th>
                        <th scope="col" className="num">
                          Held-out AUROC
                        </th>
                        <th scope="col" className="num">
                          LOCO AUROC
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.baselines.map(([model, auroc]) => (
                        <tr
                          key={model}
                          className={
                            model === "GATv2" ? "highlight" : undefined
                          }
                        >
                          <td>{model}</td>
                          <td className="num">{auroc}</td>
                          <td className="num">
                            {c.loco.find((row) => row[0] === model)?.[2]}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Reveal>
              )}
              <p className="source-note">
                Fusion role: {c.fusionRole}{" "}
                <Link to={`/components/${c.slug}`}>Component details →</Link>
              </p>
            </div>
          </section>
        ))}

      <section
        className="section section--dark"
        aria-labelledby="fusion-finding"
      >
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">C4 · Integration decision</p>
            <h2 className="display" id="fusion-finding">
              A failed validation gate means zero weight.
            </h2>
            <p className="lead">
              A component that fails its own permutation-null criterion receives
              zero base weight. Under the current evidence, Component 2 is
              therefore excluded from active fusion, and the system can return
              insufficient evidence instead of manufacturing a low-risk result.
            </p>
          </Reveal>
          <Reveal delay={0.1}>
            <p className="eyebrow">What the results teach</p>
            <ul className="check-list">
              {lessons.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="limits-title">
        <div className="shell">
          <SectionHead
            id="limits-title"
            eyebrow="Limitations"
            title="Where the claims stop."
          />
          <Reveal className="grid-3">
            {limitations.map((item, i) => (
              <div className="card" key={item}>
                <span className="card-kicker">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <p style={{ marginTop: 0, color: "var(--ink)" }}>{item}</p>
              </div>
            ))}
          </Reveal>
          <div className="button-row">
            <Link className="button button--ink" to="/documents">
              Documents and sources <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
