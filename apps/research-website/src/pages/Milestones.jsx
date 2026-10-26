import { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, CalendarDays, ClipboardList, Footprints, MapPinned } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import { milestones, assessmentWeights, milestoneSources } from "../data/documents";
import { JOURNEY_WIDTH, JOURNEY_MOBILE_WIDTH, journeyHeight, journeyPath, journeyStops } from "../lib/journeyPath.js";
import "../styles/journey.css";

function WalkingRoute({ stops, selectedIndex, mobile = false, height }) {
  const path = journeyPath(stops);
  const highlighted = journeyPath(stops.slice(0, selectedIndex + 1));
  return (
    <svg
      className={"journey-route-svg " + (mobile ? "journey-route-svg--mobile" : "journey-route-svg--desktop")}
      viewBox={`0 0 ${mobile ? JOURNEY_MOBILE_WIDTH : JOURNEY_WIDTH} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
      focusable="false"
    >
      <path className="journey-road-shadow" d={path} />
      <path className="journey-road-surface" d={path} />
      <path className="journey-road-centerline" d={path} />
      {/* This accent marks the currently selected stage, not achieved milestones. */}
      <path className="journey-road-selected" d={highlighted} />
    </svg>
  );
}

export default function Milestones() {
  const [selected, setSelected] = useState(milestones[0]?.id ?? null);
  const detailRef = useRef(null);
  const activeIndex = Math.max(0, milestones.findIndex((item) => item.id === selected));
  const active = milestones[activeIndex];
  const weights = assessmentWeights.filter((weight) => active.weights.includes(weight.id));
  const desktopStops = journeyStops(milestones.length);
  const mobileStops = journeyStops(milestones.length, true);
  const height = journeyHeight(milestones.length);

  const choose = (id) => {
    if (!milestones.some((item) => item.id === id)) return;
    setSelected(id);
    if (window.matchMedia("(max-width: 980px)").matches) {
      window.requestAnimationFrame(() => {
        detailRef.current?.scrollIntoView({
          block: "start",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        });
      });
    }
  };

  return (
    <>
      <PageHero
        eyebrow="Research milestones"
        title="From proposal to publication evidence."
        lead="Explore the 2026 assessment timeline, related research records and the historical module assessment weights—without confusing planned deadlines with verified completion."
        image="bannerDocuments"
      />

      <section className="section journey-section" aria-labelledby="milestones-title">
        <div className="shell">
          <div className="milestone-heading">
            <div>
              <p className="eyebrow">Project journey</p>
              <h2 className="display" id="milestones-title">Every milestone tells part of the story.</h2>
              <p className="lead">Follow the winding route from topic assessment to publication evidence. Select a stop to explore its documented schedule and context.</p>
            </div>
            <span className="milestone-count"><MapPinned size={18} aria-hidden="true" /> {milestones.length} research stages</span>
          </div>

          <div className="milestone-selector journey-selector">
            <label htmlFor="milestone-stage">Jump to a milestone</label>
            <select id="milestone-stage" value={selected} onChange={(event) => choose(event.target.value)}>
              {milestones.map((item, index) => (
                <option key={item.id} value={item.id}>
                  {String(index + 1).padStart(2, "0")} · {item.title}
                </option>
              ))}
            </select>
          </div>

          <div className="journey-layout">
            <nav className="journey-map" aria-label="Interactive research milestone path" style={{ "--journey-height": height + "px" }}>
              <div className="journey-map-header" aria-hidden="true">
                <span>START · STEP 01</span>
              </div>

              <WalkingRoute stops={desktopStops} selectedIndex={activeIndex} height={height} />
              <WalkingRoute stops={mobileStops} selectedIndex={activeIndex} mobile height={height} />

              <ol className="journey-stop-list">
                {milestones.map((item, index) => {
                  const stop = desktopStops[index];
                  const mobileStop = mobileStops[index];
                  const isActive = selected === item.id;
                  return (
                    <li
                      key={item.id}
                      className={"journey-stop journey-stop--" + stop.side + (isActive ? " is-active" : "")}
                      style={{
                        "--step-top": (stop.y - 82) + "px",
                        "--pin-x": (stop.x / JOURNEY_WIDTH * 100) + "%",
                        "--mobile-pin-x": mobileStop.x + "px",
                      }}
                    >
                      <button
                        className="journey-stop-button"
                        type="button"
                        aria-pressed={isActive}
                        aria-controls="milestone-detail"
                        onClick={() => choose(item.id)}
                      >
                        <span className="journey-stop-pin" aria-hidden="true">
                          <Footprints size={19} strokeWidth={2.1} />
                        </span>
                        <span className="journey-stop-card">
                          <span className="journey-stop-topline">STAGE {String(index + 1).padStart(2, "0")}</span>
                          <strong>{item.title}</strong>
                          <span className="journey-stop-date">{item.deadline ?? "Date not specified"}</span>
                          <ArrowRight className="journey-stop-arrow" size={17} aria-hidden="true" />
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ol>
              <div className="journey-map-end" aria-hidden="true">
                <span>FINISH · STEP {String(milestones.length).padStart(2, "0")}</span>
                <span className="journey-end-dot" />
              </div>
            </nav>

            <article className="milestone-detail journey-detail" id="milestone-detail" ref={detailRef} aria-live="polite" aria-atomic="true">
              <div className="journey-detail-scroll" tabIndex={0} role="region" aria-label="Selected milestone information">
              <p className="milestone-detail-kicker"><CalendarDays size={18} aria-hidden="true" /> Selected milestone · {String(activeIndex + 1).padStart(2, "0")} / {milestones.length}</p>
              <h3 className="display milestone-detail-title">{active.title}</h3>
              <dl className="milestone-facts">
                <div>
                  <dt>2026 scheduled deadline</dt>
                  <dd>{active.deadline ?? "Not specified in the timeline"}</dd>
                </div>
                <div>
                  <dt>Assessment weight · 2019 module outline</dt>
                  <dd>{weights.length
                    ? weights.map((item) => item.percent + "% · " + item.title).join("; ")
                    : "No separate weighting specified"}</dd>
                </div>
              </dl>
              <h4><ClipboardList size={19} aria-hidden="true" /> Evidence and interpretation</h4>
              <p>{active.detail}</p>
              {active.href && (
                <Link className="button button--ink" to={active.href}>
                  View related records <ArrowRight size={17} aria-hidden="true" />
                </Link>
              )}
              <p className="milestone-disclaimer">
                The selected path segment indicates navigation, not milestone completion. Scheduled dates do not establish actual completion. Assessment weights come from the 2019 module outline and have not been independently confirmed for 2026.
              </p>
              </div>
              <div className="journey-detail-actions" aria-label="Browse adjacent milestones">
                <button type="button" onClick={() => choose(milestones[activeIndex - 1]?.id)} disabled={activeIndex === 0}>
                  <ArrowLeft size={16} aria-hidden="true" /> Previous
                </button>
                <span>{activeIndex + 1} of {milestones.length}</span>
                <button type="button" onClick={() => choose(milestones[activeIndex + 1]?.id)} disabled={activeIndex === milestones.length - 1}>
                  Next <ArrowRight size={16} aria-hidden="true" />
                </button>
              </div>
            </article>
          </div>
        </div>
      </section>

      <section className="section section--paper2" aria-labelledby="assessment-weights-title">
        <div className="shell">
          <p className="eyebrow">Assessment reference</p>
          <h2 className="display" id="assessment-weights-title">Module weightings.</h2>
          <p className="lead">For reference only: the percentages below come from an older IT4010 module outline, not a verified 2026 grading announcement.</p>
          <div className="assessment-weight-grid">
            {assessmentWeights.map((weight) => (
              <div className="assessment-weight" key={weight.id}>
                <strong>{weight.percent}%</strong>
                <span>{weight.title}</span>
              </div>
            ))}
          </div>
          <div className="assessment-provenance">
            <strong>Source records</strong>
            <p>{milestoneSources.dates}</p>
            <p>{milestoneSources.weights}</p>
          </div>
        </div>
      </section>
    </>
  );
}
