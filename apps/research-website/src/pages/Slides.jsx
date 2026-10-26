import { ArrowRight, ArrowUpRight, FileText, Presentation } from "lucide-react";
import { Link } from "react-router-dom";
import PageHero from "../components/ui/PageHero";
import { presentationSlides } from "../data/documents";
import "../styles/slides.css";

export default function Slides() {
  return (
    <>
      <PageHero
        eyebrow="Presentation slides"
        title="Research presentations."
        lead="The group's proposal, progress and final slide decks—organized as one research presentation archive."
        image="bannerDocuments"
      />

      <section className="section slides-archive-section" aria-labelledby="slides-title">
        <div className="shell">
          <div className="slides-archive-heading">
            <div>
              <p className="eyebrow">Group presentation decks · R26-DS-012</p>
              <h2 className="display" id="slides-title">The presentation archive.</h2>
              <p className="lead">
                Follow the research from its proposal through two progress reviews
                to the final presentation. Select any deck to view the original
                file in Google Drive.
              </p>
            </div>
            <Link className="button button--ghost slides-documents-link" to="/documents">
              <FileText size={17} aria-hidden="true" />
              Written documents <ArrowRight size={16} aria-hidden="true" />
            </Link>
          </div>

          <div className="slides-archive-layout">
            <nav className="slides-stage-index" aria-label="Jump to a presentation stage">
              <div className="slides-stage-index-header">
                <Presentation size={19} aria-hidden="true" />
                <span>Presentation stages</span>
              </div>
              <ol>
                {presentationSlides.map((slide, index) => (
                  <li key={slide.stage}>
                    <Link to={"/slides#slide-" + index}>
                      <span className="slides-stage-index-number">{String(index + 1).padStart(2, "0")}</span>
                      <span>{slide.stage}</span>
                      <ArrowRight size={15} aria-hidden="true" />
                    </Link>
                  </li>
                ))}
              </ol>
              <p className="slides-stage-index-note">
                These links are presentation files, not verification of attendance
                or assessment completion. File dates are described as recorded in
                the archive.
              </p>
            </nav>

            <div className="slides-entry-list" aria-label="Group presentation files">
              {presentationSlides.map((slide, index) => {
                const finalStage = index === presentationSlides.length - 1;
                return (
                  <article
                    id={"slide-" + index}
                    key={slide.stage}
                    className={"slides-entry" + (finalStage ? " slides-entry--final" : "")}
                  >
                    <div className="slides-entry-art" aria-hidden="true">
                      <div className="slides-entry-art-top">
                        <span>R26—DS—012</span>
                        <Presentation size={18} strokeWidth={1.6} />
                      </div>
                      <div className="slides-entry-art-center">
                        <span>RESEARCH PRESENTATION</span>
                        <strong>{String(index + 1).padStart(2, "0")}</strong>
                      </div>
                      <div className="slides-entry-art-bottom">
                        <span>{slide.stage}</span>
                        <span className="slides-entry-art-dot" />
                      </div>
                    </div>

                    <div className="slides-entry-copy">
                      <div className="slides-entry-topline">
                        <span>DECK {String(index + 1).padStart(2, "0")} / {String(presentationSlides.length).padStart(2, "0")}</span>
                        {finalStage && <span className="slides-final-label">Final stage</span>}
                      </div>
                      <h3>{slide.title}</h3>
                      <p className="slides-entry-stage">{slide.stage}</p>
                      <p className="slides-entry-meta">{slide.meta}</p>
                      <div className="slides-entry-actions">
                        {slide.url ? (
                          <a
                            className="button button--ink"
                            href={slide.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            aria-label={"Open " + slide.title + " on Google Drive (opens in a new tab)"}
                          >
                            View slide deck <ArrowUpRight size={17} aria-hidden="true" />
                          </a>
                        ) : (
                          <span className="pill pill--outline">File not available online</span>
                        )}
                      </div>
                    </div>
                  </article>
                );
              })}
              <div className="slides-archive-footer">
                <div>
                  <p className="eyebrow">Looking for written research?</p>
                  <p>Assessment forms, individual reports, group drafts and checklists are kept separately.</p>
                </div>
                <Link className="slides-archive-footer-link" to="/documents">
                  Browse documents <ArrowRight size={17} aria-hidden="true" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
