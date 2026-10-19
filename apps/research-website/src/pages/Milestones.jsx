import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, Footprints, MapPinned } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import { milestones, assessmentWeights, milestoneSources } from "../data/documents";

export default function Milestones() {
  const [selected, setSelected] = useState(milestones[0]?.id ?? null);
  const detailsRef = useRef(null);
  const activeIndex = Math.max(0, milestones.findIndex((item) => item.id === selected));
  const milestone = milestones[activeIndex];
  const weights = assessmentWeights.filter((item) => milestone.weights.includes(item.id));

  const selectStop = (id) => {
    setSelected(id);
    // With 14 stops, the details panel is below the map. Bring it into view.
    // Respect the visitor's reduced-motion preference.
    window.requestAnimationFrame(() => {
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      detailsRef.current?.scrollIntoView({
        behavior: reduceMotion ? "instant" : "smooth",
        block: "start",
      });
    });
  };

  return (
    <>
      <PageHero
        eyebrow="Milestones"
        title="Research project trail."
        lead="Explore the 2026 research-project schedule as a numbered journey, with assessment weights from the supplied IT4010 module outline."
        image="bannerDocuments"
      />

      <section className="section" aria-labelledby="assessment-title">
        <div className="shell">
          <p className="eyebrow">Project pathway</p>
          <h2 className="display" id="assessment-title">The research roadmap.</h2>
          <p className="muted milestone-trail-intro">
            Follow the zig-zag path from topic assessment to publication evidence.
            Choose a numbered stop to see its deadline, assessment weight and supporting records.
          </p>

          <div className="zigzag-roadmap-wrap">
            <div className="zigzag-roadmap-head">
              <MapPinned size={23} aria-hidden="true" />
              <div>
                <strong>Research journey</strong>
                <p>{milestones.length} scheduled and documented milestone stops</p>
              </div>
            </div>

            <ol className="zigzag-roadmap" aria-label="Research project milestone trail">
              {milestones.map((item, index) => {
                const active = item.id === milestone.id;
                const side = index % 2 === 0 ? "left" : "right";

                return (
                  <li
                    key={item.id}
                    className={`road-stop road-stop--${side}${active ? " is-active" : ""}`}
                  >
                    <button
                      type="button"
                      className="road-stop-card"
                      aria-current={active ? "step" : undefined}
                      aria-controls="milestone-detail"
                      onClick={() => selectStop(item.id)}
                    >
                      <span className="road-stop-number" aria-hidden="true">
                        {String(index + 1).padStart(2, "0")}
                      </span>
                      <span className="road-stop-copy">
                        <strong>{item.title}</strong>
                        <span>{item.deadline ?? "Date not specified"}</span>
                      </span>
                    </button>
                    <span className="road-stop-node" aria-hidden="true" />
                    <Footprints className="road-stop-footprints" size={19} aria-hidden="true" />
                  </li>
                );
              })}
            </ol>

            <div className="roadmap-footprints" aria-hidden="true">
              <Footprints size={21} />
              <Footprints size={21} />
              <Footprints size={21} />
              <Footprints size={21} />
            </div>
          </div>

          <article
            ref={detailsRef}
            className="card assessment-detail milestone-detail-panel"
            id="milestone-detail"
            aria-live="polite"
            aria-atomic="true"
            tabIndex={-1}
          >
            <div className="assessment-detail-heading">
              <span className="assessment-step-badge">
                Stop {String(activeIndex + 1).padStart(2, "0")} / {milestones.length}
              </span>
              <h3 className="serif-title">{milestone.title}</h3>
            </div>
            <p>{milestone.detail}</p>

            <dl className="assessment-facts">
              <div>
                <dt>2026 scheduled deadline</dt>
                <dd>{milestone.deadline ?? "Not specified"}</dd>
              </div>
              <div>
                <dt>Assessment weight · 2019 outline</dt>
                <dd>
                  {weights.length
                    ? weights.map((item) => item.percent + "% — " + item.title).join("; ")
                    : "No separate weight specified"}
                </dd>
              </div>
              <div>
                <dt>Evidence sources</dt>
                <dd>RP Team 2026 timeline · IT4010 module outline v2019-0</dd>
              </div>
            </dl>

            {milestone.href && (
              <Link className="button button--ink" to={milestone.href}>
                View related records <ArrowUpRight size={16} aria-hidden="true" />
              </Link>
            )}

            <p className="source-note milestone-detail-disclaimer">
              Dates indicate scheduled deadlines, not verified completion. Assessment
              percentages are from the 2019-0 outline and have not been independently
              reconfirmed for 2026. Shared assessment categories are not counted twice.
            </p>
            <div className="assessment-sources">
              <strong>Documented sources</strong>
              <p>{milestoneSources.dates}</p>
              <p>{milestoneSources.weights}</p>
            </div>
          </article>
        </div>
      </section>
    </>
  );
}
