import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import CinematicHero from "../components/hero/CinematicHero";
import Reveal from "../components/ui/Reveal";
import ImageSlot from "../components/ui/ImageSlot";
import SectionHead from "../components/ui/SectionHead";
import ArchitectureDiagram from "../components/research/ArchitectureDiagram";
import ContactSection from "../components/layout/ContactSection";
import { homeContext, timescales } from "../data/research";
import { team } from "../data/team";
import { components } from "../data/components";
import { homeEvidenceHighlights } from "../data/homeResearch.js";
import "../styles/home-research.css";

const headlineResults = [
  {
    value: "0.9979",
    label: "C1 WESAD AUROC",
    note: "WESAD laboratory evaluation · subject-held-out",
  },
  {
    value: "0.8757",
    label: "C1 AffectiveROAD macro AUROC",
    note: "AffectiveROAD · external evaluation",
  },
  {
    value: "0.5205",
    label: "C2 held-out GATv2 AUROC, not distinguishable from chance",
    note: "GLOBEM held-out · weight 0.0 in fusion",
  },
  {
    value: "≈0.738",
    label: "C3 clinical-note AUROC",
    note: "Archived experiment · independent clean-benchmark verification required",
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
            <p className="lead">{homeContext.lead}</p>
            <div className="body-copy" style={{ marginTop: 20 }}>
              {homeContext.body.map((paragraph) => (
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

      <section className="section section--paper2 home-research-section" id="components-overview" aria-labelledby="components-title">
        <div className="shell">
          <SectionHead
            id="components-title"
            eyebrow="Explore the research"
            title="Four components. One research effort."
            lead="Explore the four complementary investigations that make up R26-DS-012. Each has its own methods, evaluation evidence and research limitations."
          />

          <div className="home-research-evidence-grid">
            {components.map((component, index) => {
              const highlight = homeEvidenceHighlights[component.id];
              return (
                <Reveal className="home-research-evidence-card" key={component.id} delay={index * 0.06}>
                  <div className="home-research-card-visual">
                    <ImageSlot name={component.image} alt={highlight.imageAlt} className="home-research-card-image" />
                    <span className="home-research-image-label">{component.id} · {component.modality}</span>
                  </div>
                  <div className="home-research-card-body">
                    <span className="home-research-timescale">{component.timescale}</span>
                    <h3 className="serif-title">{highlight.title}</h3>
                    <p className="home-research-description">{highlight.description}</p>
                    <div className="home-research-card-evidence">
                      <span className="home-research-dataset">{highlight.dataset}</span>
                      <span className={"home-research-status home-research-status--" + highlight.statusType}>
                        {highlight.status}
                      </span>
                    </div>
                    <Link className="home-research-card-action" to={"/components/" + component.slug}>
                      Explore {component.id} <ArrowRight size={17} aria-hidden="true" />
                    </Link>
                  </div>
                </Reveal>
              );
            })}
          </div>

          <p className="home-research-section-note">
            These components have different evaluation settings; their research findings do not establish validated clinical prediction or effectiveness.
          </p>
          <div className="button-row">
            <Link className="button button--ink" to="/components">
              View all research components <ArrowRight size={18} aria-hidden="true" />
            </Link>
          </div>
        </div>
      </section>

      <section className="section" id="home-system" aria-labelledby="home-system-title">
        <div className="shell">
          <SectionHead
            id="home-system-title"
            eyebrow="How the system connects"
            title="From component evidence to supported output."
            lead="A central integration layer checks signal availability and validation status before fusion. Unavailable evidence is not assumed to mean low risk."
          />
          <Reveal><ArchitectureDiagram /></Reveal>
          <div className="button-row">
            <Link className="button button--ink" to="/system">Explore the system <ArrowRight size={18} aria-hidden="true" /></Link>
          </div>
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
              <div className="stat home-evidence-stat" key={result.label}>
                <span className="stat-value">{result.value}</span>
                <span className="stat-label">{result.label}</span>
                <span className="stat-note">{result.note}</span>
              </div>
            ))}
          </Reveal>
          <p className="home-evidence-note">Results are dataset-specific research findings, not clinical performance guarantees. The behavioural model remains excluded from active fusion.</p>
          <div className="button-row">
            <Link className="button button--ink" to="/findings">
              All findings <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      <section className="section section--paper2" aria-labelledby="home-apps-title">
        <div className="shell">
          <SectionHead
            id="home-apps-title"
            eyebrow="Research applications"
            title="Two interfaces, one research framework."
            lead="Explore the participant and clinician application prototypes, their responsibilities, and their documented safety boundaries."
          />
          <div className="grid-2 home-app-grid">
            <Reveal className="home-app-card">
              <ImageSlot name="appAura" alt="Research participant application preview" />
              <div>
                <p className="card-kicker">Participant application</p>
                <h3 className="serif-title">Aura</h3>
                <p>Participant-side sensing and research data collection. Not a diagnostic app.</p>
              </div>
            </Reveal>
            <Reveal className="home-app-card" delay={0.08}>
              <ImageSlot name="appClinAnx" alt="Clinician application preview" />
              <div>
                <p className="card-kicker">Clinician application</p>
                <h3 className="serif-title">ClinAnx</h3>
                <p>Research decision-support interface for eligible signals and evidence review. Not a substitute for clinical judgment.</p>
              </div>
            </Reveal>
          </div>
          <div className="button-row">
            <Link className="button button--ink" to="/system">View the applications <ArrowRight size={18} aria-hidden="true" /></Link>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="team-title">
        <div className="shell">
          <SectionHead
            id="team-title"
            eyebrow="Team and supervision"
            title="Meet the four researchers."
          />
          <Reveal className="people-strip">
            {team.map((person) => {
              const content = (
                <>
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
                </>
              );
              // Supervisors link out to their institutional profiles.
              return person.profile ? (
                <a
                  className="person-chip person-chip--link"
                  key={person.name}
                  href={person.profile}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`${person.name}, ${person.role} (profile opens in a new tab)`}
                >
                  {content}
                </a>
              ) : (
                <div className="person-chip" key={person.name}>
                  {content}
                </div>
              );
            })}
          </Reveal>
          <div className="button-row">
            <Link className="button button--ghost" to="/team">
              Meet the team <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      </section>

      <ContactSection variant="home" />
    </>
  );
}
