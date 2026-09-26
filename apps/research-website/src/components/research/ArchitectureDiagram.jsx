import { Link } from "react-router-dom";
import Reveal from "../ui/Reveal";
import { hub, lanes, outcomes, stages } from "../../data/architecture";

const laneNote = {
  eligible: "Eligible",
  excluded: "Excluded",
  prior: "Prior only",
};

function Node({ title, detail }) {
  return (
    <div className="arch-node">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

// Left to right: sources → component models → published outputs → backend
// eligibility and fusion → decision support. Each lane links to its component.
export default function ArchitectureDiagram() {
  return (
    <Reveal className="arch-diagram" role="figure" aria-label="High-level architecture of the R26-DS-012 framework">
      <ol className="arch-stages" aria-hidden="true">
        {stages.map((stage, index) => (
          <li key={stage}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            {stage}
          </li>
        ))}
      </ol>

      <div className="arch-body">
        <ul className="arch-lanes">
          {lanes.map((lane) => (
            <li key={lane.id}>
              <Link className={`arch-lane arch-lane--${lane.state}`} to={`/components/${lane.slug}`}>
                <span className="arch-lane-id">
                  {lane.id}
                  <small>{laneNote[lane.state]}</small>
                </span>
                <Node title={lane.source[0]} detail={lane.source[1]} />
                <Node title={lane.model[0]} detail={lane.model[1]} />
                <Node title={lane.output[0]} detail={lane.output[1]} />
              </Link>
            </li>
          ))}
        </ul>

        <div className="arch-hub">
          <div className="arch-block">
            <strong>{hub.gate.title}</strong>
            <ul>
              {hub.gate.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="arch-block arch-block--fusion">
            <strong>{hub.fusion.title}</strong>
            <code>{hub.fusion.formula}</code>
          </div>
        </div>

        <div className="arch-out">
          <div className="arch-block">
            <strong>Decision tier</strong>
            <div className="pill-list">
              {outcomes.tiers.map((tier) => (
                <span className={`pill tier tier--${tier.toLowerCase()}`} key={tier}>
                  {tier}
                </span>
              ))}
              <span className="pill pill--outline">{outcomes.abstain}</span>
            </div>
          </div>
          <div className="arch-block">
            <strong>{outcomes.evidence[0]}</strong>
            <span>{outcomes.evidence[1]}</span>
          </div>
          <div className="arch-block">
            <strong>{outcomes.interfaces[0]}</strong>
            <span>{outcomes.interfaces[1]}</span>
          </div>
        </div>
      </div>

      <p className="arch-legend">
        <span className="arch-legend-item">
          <span className="arch-key arch-key--eligible" /> Eligible input
        </span>
        <span className="arch-legend-item">
          <span className="arch-key arch-key--excluded" /> Excluded from active fusion
        </span>
        <span className="arch-legend-item">
          <span className="arch-key arch-key--prior" /> Contextual prior
        </span>
      </p>
    </Reveal>
  );
}
