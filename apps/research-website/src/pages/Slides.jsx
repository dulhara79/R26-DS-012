import { ArrowUpRight, Presentation } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import { documents } from "../data/documents";

const stages = [
  ["Proposal", "R26-DS-012 proposal presentation"],
  ["Progress Presentation 1", "R26-DS-012 progress presentation 1"],
  ["Progress Presentation 2", "R26-DS-012 progress presentation 2"],
  ["Final Presentation", "R26-DS-012 final presentation"],
];
export default function Slides() {
  return (
    <>
      <PageHero eyebrow="Presentation slides" title="Research presentations."
        lead="Proposal, progress and final presentation files from the project document archive." image="bannerDocuments" />
      <section className="section" aria-labelledby="slides-title">
        <div className="shell">
          <h2 className="display" id="slides-title">Presentation archive.</h2>
          <div className="slide-grid">
            {stages.map(([stage, title], i) => {
              const doc = documents.find((entry) => entry.title === title);
              return <article className="card slide-card" id={`slide-${i}`} key={stage}>
                <Presentation size={25} aria-hidden="true" />
                <span className="card-kicker">{stage}</span>
                <h3 className="serif-title">{title}</h3>
                <p className="muted">{doc?.meta ?? "No verified presentation file available"}</p>
                {doc?.url ? <a className="button button--ink" href={doc.url} target="_blank" rel="noopener noreferrer">
                  Open slides <ArrowUpRight size={16} />
                </a> : <span className="pill pill--outline">Not yet published</span>}
              </article>;
            })}
          </div>
        </div>
      </section>
    </>
  );
}
