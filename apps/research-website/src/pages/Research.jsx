import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";
import {
  literatureThemes,
  objectives,
  overview,
  problems,
  researchGap,
  researchQuestion,
  techStack,
  timescales,
} from "../data/research";
import { componentBySlug, components } from "../data/components";
import { groupMethodology, methodologyReferences } from "../data/methodologyDetail";

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

      <section className="section section--paper2" aria-labelledby="method-title">
        <div className="shell">
          <SectionHead
            id="method-title"
            eyebrow="Methodology"
            title="Integrated research methodology."
            lead="One group workflow, from evidence acquisition and model development through quality-gated fusion, application integration and validation."
          />
          <Reveal className="body-copy group-method-intro">
            <p>{groupMethodology.introduction}</p>
          </Reveal>
          <div className="group-method-stages">
            {groupMethodology.stages.map((stage, index) => (
              <Reveal key={stage.title} className="group-method-stage">
                <div className="group-method-stage-heading">
                  <span className="card-kicker">{String(index + 1).padStart(2, "0")}</span>
                  <h3 className="card-title">{stage.title}</h3>
                </div>
                <div className="body-copy group-method-body">
                  {stage.paragraphs.map((paragraph, paragraphIndex) => (
                    <p key={paragraphIndex}>
                      {paragraph.split(/(\[\d+\])/g).map((part, partIndex) => {
                        const match = /^\[(\d+)\]$/.exec(part);
                        return match ? (
                          <a key={partIndex} className="ieee-inline-citation" href={`#reference-${match[1]}`} aria-label={`Reference ${match[1]}`}>{part}</a>
                        ) : part;
                      })}
                    </p>
                  ))}
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>
      <section className="section" aria-labelledby="method-references-title">
        <div className="shell">
          <SectionHead
            id="method-references-title"
            eyebrow="References"
            title="References"
            lead="Numbered IEEE references to published datasets, foundational methods, assessment instruments and reporting guidance. Internal thesis drafts are not included in this bibliography."
          />
          <ol className="methodology-references ieee-reference-list">
            {methodologyReferences.map((reference) => (
              <li key={reference.number} id={`reference-${reference.number}`}>
                <span className="ieee-reference-number">[{reference.number}]</span>
                <div>
                  <p>{reference.citation}</p>
                  <a className="contact-link" href={reference.url} target="_blank" rel="noopener noreferrer">
                    View source <ArrowRight size={16} aria-hidden="true" />
                  </a>
                </div>
              </li>
            ))}
          </ol>
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
