import Reveal from "../ui/Reveal";
import { componentBySlug } from "../../data/components";

// Fusion eligibility as stated in the R26-DS-012 README. No weights are shown
// beyond C2's documented 0.0, because the others are not published as numbers.
const inputs = [
  { id: "C1", stream: "Physiology", state: "Eligible input", tone: "on" },
  {
    id: "C2",
    stream: "Behaviour",
    state: "Excluded · weight 0.0",
    tone: "off",
  },
  {
    id: "C3",
    stream: "Clinical language",
    state: "Eligible input",
    tone: "on",
  },
  {
    id: "C4",
    stream: "Context (DCAR prior)",
    state: "Prior only · cannot set a tier alone",
    tone: "prior",
  },
];

const tiers = ["Low", "Medium", "High"];

export default function FusionStates({ showFormula = true }) {
  const fusion = componentBySlug["fusion-and-rag"];
  return (
    <div className="fusion">
      <Reveal className="fusion-diagram" aria-label="Fusion inputs and outputs">
        <ul className="fusion-inputs">
          {inputs.map((input) => (
            <li
              key={input.id}
              className={`fusion-input fusion-input--${input.tone}`}
            >
              <span className="fusion-id">{input.id}</span>
              <span className="fusion-stream">{input.stream}</span>
              <span className="fusion-state">{input.state}</span>
            </li>
          ))}
        </ul>
        <div className="fusion-core" aria-hidden="true">
          <span>Reliability-weighted fusion</span>
        </div>
        <div className="fusion-outputs">
          <span className="fusion-outputs-label">Decision tier</span>
          <div className="pill-list">
            {tiers.map((tier) => (
              <span
                className={`pill tier tier--${tier.toLowerCase()}`}
                key={tier}
              >
                {tier}
              </span>
            ))}
            <span className="pill pill--outline">or insufficient evidence</span>
          </div>
        </div>
      </Reveal>
      {showFormula && (
        <Reveal className="fusion-formula" delay={0.1}>
          <div className="formula-lines">
            {fusion.formula.map(([expression, meaning]) => (
              <div key={expression}>
                <code>{expression}</code>
                <span>{meaning}</span>
              </div>
            ))}
          </div>
          <dl className="formula-terms">
            {fusion.terms.map(([symbol, meaning]) => (
              <div key={symbol}>
                <dt>{symbol}</dt>
                <dd>{meaning}</dd>
              </div>
            ))}
          </dl>
        </Reveal>
      )}
    </div>
  );
}
