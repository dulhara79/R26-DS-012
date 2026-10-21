import { useState } from "react";
import { ArrowUpRight, Check, Copy, FileText } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import SectionHead from "../components/ui/SectionHead";
import {
  documents,
  milestones,
  publicationNote,
  publishedAdditionalDocuments,
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
        title="Research documents and milestones."
        lead="Research assessment records, from the topic assessment form through the final presentation."
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
                    <li key={doc.title} id={`document-${documents.indexOf(doc)}`}>
                      <FileText size={20} aria-hidden="true" />
                      <div>
                        <span className="card-kicker">{doc.type}</span>
                        {doc.type === "Proposal report" && doc.url ? (
                          <a
                            className="document-title-link"
                            href={doc.url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {doc.title}
                          </a>
                        ) : (
                          <strong>{doc.title}</strong>
                        )}
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
          {[...new Set(publishedAdditionalDocuments.map((doc) => doc.group))].map((group) => (
            <div className="doc-group" key={group}>
              <h3 className="serif-title detail-sub">{group}</h3>
              <ul className="doc-list">
                {publishedAdditionalDocuments.filter((doc) => doc.group === group).map((doc) => (
                  <li key={doc.url} id={`additional-document-${publishedAdditionalDocuments.indexOf(doc)}`}>
                    <FileText size={20} aria-hidden="true" />
                    <div>
                      <span className="card-kicker">{doc.type}</span>
                      <strong>{doc.title}</strong>
                      <span className="muted">{doc.meta}</span>
                    </div>
                    <a className="button button--ghost" href={doc.url} target="_blank" rel="noopener noreferrer">
                      Open <ArrowUpRight size={16} />
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <p className="source-note">{publicationNote}</p>
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
