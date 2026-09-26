import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, ArrowUpRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import ImageSlot from "../components/ui/ImageSlot";
import SectionHead from "../components/ui/SectionHead";
import FusionStates from "../components/research/FusionStates";
import NotFound from "./NotFound";
import { componentBySlug, components } from "../data/components";

function Table({ head, rows, numeric = [], highlight }) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {head.map((cell, index) => (
              <th
                scope="col"
                key={cell}
                className={numeric.includes(index) ? "num" : undefined}
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row[0]}
              className={highlight === row[0] ? "highlight" : undefined}
            >
              {row.map((cell, index) => (
                <td
                  key={index}
                  className={numeric.includes(index) ? "num" : undefined}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ComponentDetail() {
  const { slug } = useParams();
  const c = componentBySlug[slug];
  if (!c) return <NotFound />;

  const index = components.indexOf(c);
  const prev = components[(index - 1 + components.length) % components.length];
  const next = components[(index + 1) % components.length];

  return (
    <>
      <PageHero
        eyebrow={`${c.id} · ${c.modality}`}
        title={c.title}
        lead={c.tagline}
        meta={[`${c.owner} · ${c.studentId}`, c.timescale]}
      />

      <section className="section" aria-labelledby="idea-title">
        <div className="shell split">
          <Reveal>
            <p className="eyebrow">Core idea</p>
            <h2
              className="display"
              id="idea-title"
              style={{ fontSize: "clamp(2rem,3.6vw,3.2rem)" }}
            >
              {c.headline}
            </h2>
            <p className="lead">{c.coreIdea}</p>
            <p className="source-note">Proposal title: {c.proposalTitle}</p>
          </Reveal>
          <Reveal delay={0.1}>
            <ImageSlot name={c.image} alt="" />
          </Reveal>
        </div>
      </section>

      <section className="section section--paper2" aria-labelledby="data-title">
        <div className="shell">
          <SectionHead
            id="data-title"
            eyebrow={
              c.id === "C4" ? "Inputs and fusion rule" : "Data and inputs"
            }
            title={c.dataTitle}
          />
          <div className="detail-grid">
            {c.hardware && (
              <Reveal>
                <h3 className="serif-title detail-sub">Hardware</h3>
                <Table
                  head={["Sensor", "Component", "Measures"]}
                  rows={c.hardware}
                />
              </Reveal>
            )}
            {c.features && (
              <Reveal delay={0.08}>
                <h3 className="serif-title detail-sub">
                  10-feature physiological window
                </h3>
                <div className="pill-list">
                  {c.features.map((feature) => (
                    <span className="pill" key={feature}>
                      {feature}
                    </span>
                  ))}
                </div>
                <h3
                  className="serif-title detail-sub"
                  style={{ marginTop: 36 }}
                >
                  Benchmark datasets
                </h3>
                <Table head={["Dataset", "Setting"]} rows={c.datasets} />
              </Reveal>
            )}
            {c.cohorts && (
              <Reveal>
                <h3 className="serif-title detail-sub">GLOBEM cohorts</h3>
                <Table head={["Cohort", "Year", "Role"]} rows={c.cohorts} />
                <p className="source-note">Target: {c.target}</p>
              </Reveal>
            )}
            {c.benchmark && (
              <Reveal>
                <h3 className="serif-title detail-sub">Benchmark safeguards</h3>
                <ul className="check-list">
                  {c.benchmark.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </Reveal>
            )}
            {c.policies && (
              <Reveal delay={0.08}>
                <h3 className="serif-title detail-sub">Index-time policies</h3>
                <Table head={["Policy", "Research task"]} rows={c.policies} />
              </Reveal>
            )}
            {c.dcar && (
              <Reveal>
                <h3 className="serif-title detail-sub">
                  DCAR contextual prior
                </h3>
                <p className="lead" style={{ marginTop: 0 }}>
                  {c.dcar}
                </p>
              </Reveal>
            )}
          </div>
          {c.formula && (
            <div style={{ marginTop: 48 }}>
              <FusionStates />
            </div>
          )}
        </div>
      </section>

      <section
        className="section section--dark"
        aria-labelledby="pipeline-title"
      >
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">Model flow</p>
            <h2
              className="display"
              id="pipeline-title"
              style={{ fontSize: "clamp(2rem,3.6vw,3.2rem)" }}
            >
              How it works, step by step.
            </h2>
            <h3 className="serif-title detail-sub" style={{ marginTop: 40 }}>
              Evaluation
            </h3>
            <ul className="check-list">
              {c.evaluation.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Reveal>
          <Reveal as="ol" className="flow" delay={0.1}>
            {c.pipeline.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </Reveal>
        </div>
      </section>

      {c.rag && (
        <section className="section" aria-labelledby="rag-title">
          <div className="shell">
            <SectionHead
              id="rag-title"
              eyebrow="CARE-AnxRAG"
              title="Evidence that can say “I don't know”."
              lead="A contradiction-, authority-, reliability- and evidence-aware retrieval-augmented system. It separates semantic relevance from evidence quality and abstains when evidence is weak or conflicting."
            />
            <Reveal className="grid-3">
              {c.rag.map((item, i) => (
                <div className="card rag-step" key={item}>
                  <span className="card-kicker">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <p style={{ marginTop: 0 }}>{item}</p>
                </div>
              ))}
            </Reveal>
          </div>
        </section>
      )}

      {c.results && (
        <section className="section" aria-labelledby="results-title">
          <div className="shell">
            <SectionHead
              id="results-title"
              eyebrow="Results"
              title={c.resultsLabel}
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
              <div className="grid-2" style={{ marginTop: 56 }}>
                <Reveal>
                  <h3 className="serif-title detail-sub">
                    Graph vs flat baselines
                  </h3>
                  <Table
                    head={["Model", "AUROC"]}
                    rows={c.baselines}
                    numeric={[1]}
                    highlight="GATv2"
                  />
                </Reveal>
                <Reveal delay={0.08}>
                  <h3 className="serif-title detail-sub">
                    Leave-one-cohort-out
                  </h3>
                  <Table
                    head={["Model", "Pooled", "LOCO", "Gap"]}
                    rows={c.loco}
                    numeric={[1, 2, 3]}
                    highlight="GATv2"
                  />
                </Reveal>
              </div>
            )}
          </div>
        </section>
      )}

      <section
        className="section section--paper2"
        aria-labelledby="status-title"
      >
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">Status and role in fusion</p>
            <h2
              className="display"
              id="status-title"
              style={{ fontSize: "clamp(2rem,3.6vw,3.2rem)" }}
            >
              Where {c.id} stands.
            </h2>
            <p className="lead">{c.fusionRole}</p>
            <div className="button-row">
              <a
                className="button button--ink"
                href={c.source}
                target="_blank"
                rel="noreferrer"
              >
                Component source <ArrowUpRight size={18} />
              </a>
            </div>
          </Reveal>
          <Reveal as="ul" className="check-list" delay={0.1}>
            {c.status.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </Reveal>
        </div>
      </section>

      <nav className="section section--tight" aria-label="Other components">
        <div className="shell pager">
          <Link className="pager-link" to={`/components/${prev.slug}`}>
            <ArrowLeft size={18} />
            <span>
              <small>{prev.id}</small>
              {prev.title}
            </span>
          </Link>
          <Link
            className="pager-link pager-link--next"
            to={`/components/${next.slug}`}
          >
            <span>
              <small>{next.id}</small>
              {next.title}
            </span>
            <ArrowRight size={18} />
          </Link>
        </div>
      </nav>
    </>
  );
}
