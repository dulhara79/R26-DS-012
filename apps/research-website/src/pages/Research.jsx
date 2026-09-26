import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";
import {
  literatureThemes,
  methodology,
  objectives,
  overview,
  problems,
  researchGap,
  researchQuestion,
  techStack,
  timescales,
} from "../data/research";
import { componentBySlug, components } from "../data/components";

export default function Research() {
  const slugFor = (id) =>
    components.find((component) => component.id === id)?.slug;
  return (
    <>
      <PageHero
        eyebrow="Research domain"
        title="Understanding anxiety beyond a single moment."
        lead="The context, the literature, the gap and the question that the four components set out to answer."
        image="bannerResearch"
      />

      <section className="section" aria-labelledby="overview-title">
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">Overview</p>
            <h2 className="display" id="overview-title">
              Change unfolds on different clocks.
            </h2>
          </Reveal>
          <Reveal className="body-copy" delay={0.1}>
            <p>{overview.lead}</p>
            {overview.body.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
            <p>{overview.paradigms}</p>
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
        className="section section--paper2"
        aria-labelledby="problem-title"
      >
        <div className="shell">
          <SectionHead
            id="problem-title"
            eyebrow="Research problem"
            title="Why the obvious approach fails."
          />
          <Reveal className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Challenge</th>
                  <th scope="col">Why it matters</th>
                </tr>
              </thead>
              <tbody>
                {problems.map((item) => (
                  <tr key={item.challenge}>
                    <td>
                      <strong>{item.challenge}</strong>
                    </td>
                    <td className="muted">{item.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="literature-title">
        <div className="shell">
          <SectionHead
            id="literature-title"
            eyebrow="Literature landscape"
            title="Every signal is informative. Every signal has a failure mode."
            lead="A synthesis of the themes that motivated each component, and the limits each one carries."
          />
          <div className="grid-3">
            {literatureThemes.map((theme, index) => (
              <Reveal
                className="card lit-card"
                key={theme.theme}
                delay={(index % 3) * 0.06}
              >
                <h3 className="card-title">{theme.theme}</h3>
                <dl>
                  <dt>Shows</dt>
                  <dd>{theme.shows}</dd>
                  <dt>Limits</dt>
                  <dd>{theme.limits}</dd>
                  <dt>In this project</dt>
                  <dd>{theme.relevance}</dd>
                </dl>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--dark" aria-labelledby="gap-title">
        <div className="shell">
          <Reveal>
            <p className="eyebrow">Research gap</p>
            <h2 className="big-quote" id="gap-title">
              {researchGap.headline}
            </h2>
            <p className="lead">{researchGap.body}</p>
            <p className="pill" style={{ marginTop: 28 }}>
              {researchGap.principle}
            </p>
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="question-title">
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">Research question</p>
            <h2
              className="display"
              id="question-title"
              style={{ fontSize: "clamp(2rem,3.6vw,3.2rem)" }}
            >
              {researchQuestion}
            </h2>
          </Reveal>
          <Reveal delay={0.1}>
            <p className="eyebrow">Main objective</p>
            <p className="lead" style={{ marginTop: 0 }}>
              {objectives.main}
            </p>
            <p className="eyebrow" style={{ marginTop: 40 }}>
              Specific objectives
            </p>
            <ul className="objective-list">
              {objectives.specific.map((item) => (
                <li key={item.id}>
                  <Link to={`/components/${slugFor(item.id)}`}>
                    <span className="fusion-id">{item.id}</span>
                    <span>{item.text}</span>
                  </Link>
                </li>
              ))}
            </ul>
          </Reveal>
        </div>
      </section>

      <section
        className="section section--paper2"
        aria-labelledby="method-title"
      >
        <div className="shell">
          <SectionHead
            id="method-title"
            eyebrow="Methodology"
            title="From problem to prototype, with validation at every step."
          />
          <Reveal as="ol" className="method-steps">
            {methodology.map((item) => (
              <li key={item.step}>
                <strong>{item.step}</strong>
                <span className="muted">{item.detail}</span>
              </li>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="tech-title">
        <div className="shell">
          <SectionHead
            id="tech-title"
            eyebrow="Technologies"
            title="The tools behind each stream."
          />
          <div className="grid-3">
            {techStack.map((group, index) => (
              <Reveal
                className="card"
                key={group.area}
                delay={(index % 3) * 0.06}
              >
                <span className="card-kicker">{group.area}</span>
                <div className="pill-list">
                  {group.tools.map((tool) => (
                    <span className="pill" key={tool}>
                      {tool}
                    </span>
                  ))}
                </div>
              </Reveal>
            ))}
          </div>
          <div className="button-row">
            <Link
              className="button button--ink"
              to={`/components/${componentBySlug["wearable-forecasting"].slug}`}
            >
              Start with Component 1 <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
