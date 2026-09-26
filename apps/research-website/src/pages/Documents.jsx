import { useState } from "react";
import { ArrowUpRight, Check, Copy, FileText } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";
import {
  documents,
  milestones,
  publicationNote,
  scopeChanges,
} from "../data/documents";

// Citation block from the R26-DS-012 README.
const bibtex = `@misc{r26ds012_2026,
  title     = {A Multimodal Digital Biomarker Framework for Personalized Vulnerability
               Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders},
  author    = {Sendanayake, H.D. and Layathma, B.M.A.S. and Seneviratne, K.A.U.A. and Kaushalya, I.G.D.},
  year      = {2026},
  note      = {R26-DS-012, B.Sc. (Hons) Information Technology (Data Science),
               Sri Lanka Institute of Information Technology (SLIIT)},
  supervisor= {Thelijjagoda, S. and Weerasinghe, M.}
}`;

export default function Documents() {
  const [copied, setCopied] = useState(false);
  const groups = [...new Set(documents.map((doc) => doc.group))];

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(bibtex);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <>
      <PageHero
        eyebrow="Documents"
        title="Proposals, sources and milestones."
        lead="The written record of the project, including how its scope changed between the proposals and the final components."
        image="bannerDocuments"
      />

      <section className="section" aria-labelledby="docs-title">
        <div className="shell">
          <SectionHead
            id="docs-title"
            eyebrow="Documents"
            title="The project record."
          />
          {groups.map((group) => (
            <div className="doc-group" key={group}>
              <h3 className="serif-title detail-sub">{group}</h3>
              <Reveal as="ul" className="doc-list">
                {documents
                  .filter((doc) => doc.group === group)
                  .map((doc) => (
                    <li key={doc.title}>
                      <FileText size={20} aria-hidden="true" />
                      <div>
                        <span className="card-kicker">{doc.type}</span>
                        <strong>{doc.title}</strong>
                        <span className="muted">{doc.meta}</span>
                      </div>
                      {doc.url ? (
                        <a
                          className="button button--ghost"
                          href={doc.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Open <ArrowUpRight size={16} />
                        </a>
                      ) : (
                        <span className="pill pill--outline">
                          Not published online
                        </span>
                      )}
                    </li>
                  ))}
              </Reveal>
            </div>
          ))}
          <p className="source-note">{publicationNote}</p>
        </div>
      </section>

      <section
        className="section section--paper2"
        aria-labelledby="scope-title"
      >
        <div className="shell">
          <SectionHead
            id="scope-title"
            eyebrow="From proposal to final scope"
            title="What changed, and why it is stated openly."
            lead="The repository is the source of truth. Where a component's final design differs from its March 2026 proposal, both are named."
          />
          <Reveal className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Component</th>
                  <th scope="col">In the proposal</th>
                  <th scope="col">In the final repository</th>
                </tr>
              </thead>
              <tbody>
                {scopeChanges.map((row) => (
                  <tr key={row.id}>
                    <td>
                      <strong>{row.id}</strong>
                    </td>
                    <td className="muted">{row.before}</td>
                    <td>{row.after}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="milestones-title">
        <div className="shell split" style={{ alignItems: "start" }}>
          <Reveal>
            <p className="eyebrow">Milestones</p>
            <h2 className="display" id="milestones-title">
              Progress so far.
            </h2>
          </Reveal>
          <Reveal as="ol" className="flow milestones" delay={0.1}>
            {milestones.map((item) => (
              <li key={item.title}>
                <span className="card-kicker">{item.date}</span>
                <strong className="serif-title" style={{ fontSize: "1.3rem" }}>
                  {item.title}
                </strong>
                <p className="muted" style={{ margin: "8px 0 0" }}>
                  {item.detail}
                </p>
              </li>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section section--dark" aria-labelledby="cite-title">
        <div className="shell">
          <SectionHead
            id="cite-title"
            eyebrow="Citation"
            title="Cite this work."
          />
          <Reveal className="cite">
            <pre>
              <code>{bibtex}</code>
            </pre>
            <button
              className="button button--paper"
              type="button"
              onClick={copy}
            >
              {copied ? <Check size={18} /> : <Copy size={18} />}
              {copied ? "Copied" : "Copy BibTeX"}
            </button>
          </Reveal>
        </div>
      </section>
    </>
  );
}
