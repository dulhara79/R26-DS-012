import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";
import {
  groupLiteratureReview,
  objectives,
  groupAbstract,
  problems,
  researchGap,
  researchQuestion,
  techStack,
} from "../data/research";
import { componentBySlug, components } from "../data/components";
import { groupMethodology, methodologyReferences } from "../data/methodologyDetail";

const domainSections = [
  ["Abstract", "overview-title"],
  ["Problem", "problem-title"],
  ["Literature", "literature-title"],
  ["Research gap", "gap-title"],
  ["Objectives", "question-title"],
  ["Methodology", "method-title"],
  ["References", "method-references-title"],
  ["Technologies", "tech-title"],
];

// Citation numbers here link to the public IEEE bibliography already used on
// this page. They are independent source identifiers, not component result IDs.
function CitedLiterature({ text }) {
  return text.split(/(\[\d+\])/g).map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    return match ? (
      <a
        key={index}
        className="ieee-inline-citation"
        href={"#reference-" + match[1]}
        aria-label={"Reference " + match[1]}
      >
        {part}
      </a>
    ) : part;
  });
}

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


      <nav className="section section--tight domain-index" aria-label="Explore the research domain">
        <div className="shell">
          <p className="eyebrow">On this page</p>
          <div className="domain-index-links">
            {domainSections.map(([label, id]) => (
              <Link key={id} to={"/research#" + id}>{label} <ArrowRight size={15} aria-hidden="true" /></Link>
            ))}
          </div>
          <div className="domain-next-links">
            <span>Continue exploring</span>
            <Link to="/components">Four components <ArrowRight size={16} aria-hidden="true" /></Link>
            <Link to="/system">Architecture & apps <ArrowRight size={16} aria-hidden="true" /></Link>
            <Link to="/findings">Experimental findings <ArrowRight size={16} aria-hidden="true" /></Link>
          </div>
        </div>
      </nav>

      <section className="section research-abstract" aria-labelledby="overview-title">
        <div className="shell">
          <Reveal className="research-abstract-heading">
            <p className="eyebrow">R26-DS-012 · Integrated group study</p>
            <h2 className="display" id="overview-title">Abstract</h2>
            <p className="research-abstract-source">{groupAbstract.source}</p>
          </Reveal>
          <Reveal as="article" className="research-abstract-body" delay={0.08}>
            {groupAbstract.paragraphs.map((paragraph, index) => (
              <p key={index}>{paragraph}</p>
            ))}
            <div className="research-abstract-keywords">
              <strong>Keywords</strong>
              <p>{groupAbstract.keywords.join(" · ")}</p>
            </div>
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

      <section className="section literature-review" aria-labelledby="literature-title">
        <div className="shell">
          <SectionHead
            id="literature-title"
            eyebrow="Group literature review"
            title="From clinical observation to trustworthy multimodal evidence."
            lead="An integrated synthesis of published research and the three interconnected challenges identified in the R26-DS-012 group report."
          />
          <div className="literature-review-layout">
            <div className="literature-review-main">
              <Reveal className="literature-review-intro body-copy">
                <p className="eyebrow">The shared research context</p>
                {groupLiteratureReview.introduction.map((paragraph, index) => (
                  <p key={index}><CitedLiterature text={paragraph} /></p>
                ))}
              </Reveal>
              <div className="literature-review-themes">
                {groupLiteratureReview.themes.map((theme) => (
                  <Reveal as="article" className="literature-review-theme" key={theme.id} id={"literature-" + theme.id}>
                    <div className="literature-review-theme-heading">
                      <span className="literature-review-number" aria-hidden="true">{theme.number}</span>
                      <h3 className="serif-title">{theme.title}</h3>
                    </div>
                    <div className="body-copy">
                      {theme.paragraphs.map((paragraph, index) => (
                        <p key={index}><CitedLiterature text={paragraph} /></p>
                      ))}
                    </div>
                  </Reveal>
                ))}
              </div>
            </div>
            <aside className="literature-review-aside" aria-label="Literature synthesis">
              <p className="eyebrow">Literature synthesis</p>
              <h3 className="serif-title">{groupLiteratureReview.synthesis.title}</h3>
              <p>{groupLiteratureReview.synthesis.summary}</p>
              <p className="literature-review-caution">{groupLiteratureReview.synthesis.boundary}</p>
              <a href="#method-references-title" className="literature-review-reference-link">
                Review the cited literature <ArrowRight size={16} aria-hidden="true" />
              </a>
            </aside>
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
