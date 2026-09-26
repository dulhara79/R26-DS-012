import { Link } from "react-router-dom";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import CinematicHero from "../components/hero/CinematicHero";
import Reveal from "../components/ui/Reveal";
import ImageSlot from "../components/ui/ImageSlot";
import SectionHead from "../components/ui/SectionHead";
import FusionStates from "../components/research/FusionStates";
import { components } from "../data/components";
import {
  commitments,
  overview,
  problems,
  researchQuestion,
  timescales,
} from "../data/research";
import { supervisors, team } from "../data/team";

const headlineResults = [
  {
    value: "0.97",
    label: "C1 combined AUROC across 25 subjects / drives",
    note: "Phase 1 benchmark, LOSO",
  },
  {
    value: "11.83 min",
    label: "C1 average early warning time",
    note: "Phase 1 forecasting module",
  },
  {
    value: "0.5205",
    label: "C2 held-out GATv2 AUROC, not distinguishable from chance",
    note: "Final v8, GLOBEM",
  },
  {
    value: "≈0.738",
    label: "C3 clinical-note AUROC",
    note: "Deployment-relevant held-out setting",
  },
];

export default function Home() {
  return (
    <>
      <CinematicHero />

      <section className="section" id="context" aria-labelledby="context-title">
        <div className="shell split">
          <Reveal>
            <p className="eyebrow">Why this research</p>
            <h2 className="display" id="context-title">
              Anxiety is not a single moment.
            </h2>
            <p className="lead">{overview.lead}</p>
            <div className="body-copy" style={{ marginTop: 20 }}>
              {overview.body.map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
            </div>
            <div className="button-row">
              <Link className="button button--ink" to="/research">
                Read the research <ArrowRight size={18} />
              </Link>
            </div>
          </Reveal>
          <Reveal delay={0.1}>
            <ImageSlot name="homeContext" alt="" />
          </Reveal>
        </div>
        <div className="shell">
          <Reveal className="timescale-row">
            {timescales.map((item) => (
              <div className="timescale" key={item.stream}>
                <span className="timescale-when">{item.when}</span>
                <strong>{item.stream}</strong>
                <span className="muted">{item.detail}</span>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      <section
        className="section section--dark"
        aria-labelledby="problem-title"
      >
        <div className="shell">
          <SectionHead
            id="problem-title"
            eyebrow="The problem"
            title="Six reasons a single score is not enough."
            lead="Each challenge below shaped a design decision in the framework."
          />
          <Reveal as="ol" className="numbered-list">
            {problems.map((item) => (
              <li key={item.challenge}>
                <strong>{item.challenge}</strong>
                <span className="muted">{item.why}</span>
              </li>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="components-title">
        <div className="shell">
          <SectionHead
            id="components-title"
            eyebrow="Four components · one framework"
            title="Each signal studied on its own terms."
            lead={overview.paradigms}
          />
          <div className="grid-4">
            {components.map((component, index) => (
              <Reveal key={component.id} delay={index * 0.08}>
                <Link
                  className="card link-card"
                  to={`/components/${component.slug}`}
                >
                  <ImageSlot name={component.image} alt="" />
                  <div className="link-card-body">
                    <span className="card-kicker">
                      {component.id} · {component.modality}
                    </span>
                    <h3 className="card-title">{component.title}</h3>
                    <p>{component.tagline}</p>
                    <span className="link-card-foot">
                      {component.owner} <ArrowRight size={16} />
                    </span>
                  </div>
                </Link>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section
        className="section section--paper2"
        aria-labelledby="fusion-title"
      >
        <div className="shell">
          <SectionHead
            id="fusion-title"
            eyebrow="How the evidence comes together"
            title="Weighted by trust, not by availability."
            lead="Reliability-weighted fusion combines only eligible outputs. A modality that fails its own validation gets no weight, and missing evidence is never read as zero risk."
          />
          <FusionStates />
        </div>
      </section>

      <section className="section" aria-labelledby="results-title">
        <div className="shell">
          <SectionHead
            id="results-title"
            eyebrow="What the evidence shows so far"
            title="Results reported with their context."
            lead="Positive and negative findings are shown the same way. Each number keeps its dataset and evaluation protocol attached."
          />
          <Reveal className="grid-4">
            {headlineResults.map((result) => (
              <div className="stat" key={result.label}>
                <span className="stat-value">{result.value}</span>
                <span className="stat-label">{result.label}</span>
                <span className="stat-note">{result.note}</span>
              </div>
            ))}
          </Reveal>
          <div className="button-row">
            <Link className="button button--ink" to="/findings">
              All findings <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      <section className="section section--dark" aria-labelledby="safety-title">
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">Privacy and research safety</p>
            <h2 className="display" id="safety-title">
              Built to withhold, not to guess.
            </h2>
            <p className="lead">
              The integrated framework is research and clinical decision
              support. It is not a diagnostic medical device.
            </p>
            <div className="button-row">
              <Link className="button button--paper" to="/system">
                See the system <ArrowRight size={18} />
              </Link>
            </div>
          </Reveal>
          <Reveal as="ul" className="check-list" delay={0.1}>
            {commitments.slice(0, 6).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="team-title">
        <div className="shell">
          <SectionHead
            id="team-title"
            eyebrow="Team and supervision"
            title="Four researchers, three supervisors."
          />
          <Reveal className="people-strip">
            {[...team, ...supervisors].map((person) => (
              <div className="person-chip" key={person.name}>
                <ImageSlot
                  name={person.photo}
                  alt=""
                  className="person-chip-photo"
                />
                <div>
                  <strong>{person.name}</strong>
                  <span>
                    {person.component
                      ? `${person.component} · ${person.role}`
                      : person.role}
                  </span>
                </div>
              </div>
            ))}
          </Reveal>
          <div className="button-row">
            <Link className="button button--ghost" to="/team">
              Meet the team <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      <section
        className="section section--paper2 closing"
        aria-labelledby="question-title"
      >
        <div className="shell">
          <Reveal>
            <p className="eyebrow">The research question</p>
            <h2 className="big-quote" id="question-title">
              {researchQuestion}
            </h2>
            <div className="button-row">
              <Link className="button button--ink" to="/components">
                Explore the components <ArrowRight size={18} />
              </Link>
              <Link className="button button--ghost" to="/documents">
                Documents <ArrowUpRight size={18} />
              </Link>
            </div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
