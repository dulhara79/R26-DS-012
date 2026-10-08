import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";

const findings = [
  {
    id: "C3",
    eyebrow: "C3 · Clinical NLP",
    title: "Clinical text produced the strongest validated component signal.",
    lead:
      "The publication-clean TC-WPN benchmark used patient-disjoint few-shot episodes, fixed index-time policies and explicit leakage controls. The final K=5 configuration was evaluated across five seeds.",
    results: [
      ["0.7377", "Mean AUROC across five seeds"],
      ["0.0031", "AUROC standard deviation"],
      ["0.767", "Final F1"],
      ["0.790", "Final AUPRC"],
    ],
    detail:
      "The full TC-WPN configuration did not show a significant improvement over the auxiliary-only configuration. The component therefore contributes a validated clinical-language signal, not an overall patient-risk score.",
    link: "/components/clinical-nlp",
  },
  {
    id: "C2",
    eyebrow: "C2 · Behavioural Phenotyping",
    title: "The behavioural graph did not clear the validation gate.",
    lead:
      "The final GLOBEM study used the preceding 28 days of passive sensing, participant-grouped evaluation and a prevalence-preserving permutation null. GATv2 was compared with simpler flat behavioural baselines.",
    results: [
      ["0.5205", "Held-out GATv2 AUROC"],
      ["0.485–0.560", "95% participant-clustered CI"],
      ["0.4991", "50-permutation null mean"],
      ["0.255", "Empirical permutation p-value"],
    ],
    detail:
      "The final GATv2 result was not distinguishable from chance and did not outperform the flat baselines. The pre-registered rule therefore assigns Component 2 a base fusion weight of 0.0. Earlier vulnerability scores, hourly high-risk windows and phenotype outputs remain exploratory rather than validated clinical outputs.",
    link: "/components/behavioural-graphs",
    table: [
      ["Logistic Regression", "0.5458"],
      ["Random Forest", "0.5617"],
      ["Gradient Boosting", "0.5681"],
      ["GATv2", "0.5205"],
    ],
  },
  {
    id: "C1",
    eyebrow: "C1 · Physiology",
    title: "Personalized physiology showed strong benchmark discrimination and short-horizon forecasting.",
    lead:
      "The final physiological pipeline establishes a participant-specific resting baseline, uses self-supervised reconstruction error for anomaly monitoring, and feeds two consecutive non-overlapping five-minute history blocks into direct +5 and +10 minute forecasting.",
    results: [
      ["0.9979", "WESAD AUROC under corrected 13-participant LOSO"],
      ["0.9233", "WESAD F1 under corrected LOSO"],
      ["0.8757", "AffectiveROAD macro AUROC"],
      ["0.5362", "AffectiveROAD F1"],
      ["0.1226", "Ridge participant-macro MAE at +5 min"],
      ["0.1291", "Ridge participant-macro MAE at +10 min"],
    ],
    detail:
      "The Ridge forecaster improved on persistence, which had MAE 0.1364 at +5 minutes and 0.1599 at +10 minutes. External AffectiveROAD forecasting also improved over persistence at both horizons. A three-minute seated resting calibration stabilized feature variances within 1.8% of a ten-minute reference.",
    link: "/components/wearable-forecasting",
  },
  {
    id: "C4",
    eyebrow: "C4 · Fusion + RAG",
    title: "Fusion is selective, while evidence retrieval can abstain.",
    lead:
      "The integration layer combines eligible component outputs using informativeness, recency and reliability/coverage, while CARE-AnxRAG adds evidence-aware retrieval, contradiction checks, provenance and abstention safeguards.",
    results: [
      ["0.6220", "Contextual / demographic model AUROC"],
      ["0.565–0.674", "Reported 95% confidence interval"],
      ["0.0023", "Expected calibration error"],
      ["0.000", "Behavioural base fusion weight"],
    ],
    detail:
      "The contextual prior is not treated as a standalone escalation signal. Eligible modality weights are adjusted for recency and coverage, and a component that fails its validation gate contributes zero base weight. CARE-AnxRAG combines dense and lexical retrieval with reranking, authority/freshness/applicability checks, contradiction detection, citation validation and calibrated abstention when evidence is weak or conflicting.",
    link: "/components/fusion-and-rag",
  },
];

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
        lead="The four components are presented in research order, from clinical language to behaviour, physiology, and finally multimodal fusion and evidence support."
        image="bannerFindings"
      />

      {findings.map((finding, index) => (
        <section
          className={`section ${index % 2 ? "section--paper2" : ""}`}
          key={finding.id}
          aria-labelledby={`${finding.id}-results`}
        >
          <div className="shell">
            <SectionHead
              id={`${finding.id}-results`}
              eyebrow={finding.eyebrow}
              title={finding.title}
              lead={finding.lead}
            />

            <Reveal className="grid-4">
              {finding.results.map(([value, label]) => (
                <div className="stat" key={label}>
                  <span className="stat-value">{value}</span>
                  <span className="stat-label">{label}</span>
                </div>
              ))}
            </Reveal>

            {finding.table && (
              <Reveal className="table-wrap" style={{ marginTop: 48 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col" className="num">
                        Held-out AUROC
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {finding.table.map(([model, auroc]) => (
                      <tr
                        key={model}
                        className={model === "GATv2" ? "highlight" : undefined}
                      >
                        <td>{model}</td>
                        <td className="num">{auroc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Reveal>
            )}

            <p className="source-note">
              {finding.detail}{" "}
              <Link to={finding.link}>Component details →</Link>
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
            <p className="eyebrow">Cross-component decision</p>
            <h2 className="display" id="fusion-finding">
              A failed validation gate means zero weight.
            </h2>
            <p className="lead">
              Component outputs are not treated as equally reliable. A stream
              must clear its own validation criteria before it can influence
              fusion, and missing or weak evidence can leave the system without
              enough evidence to issue a tier.
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
