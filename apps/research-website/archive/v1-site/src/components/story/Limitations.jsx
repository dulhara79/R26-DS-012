import { researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";
export default function Limitations() {
  return (
    <section className="limitations-section">
      <div className="shell limitations-grid">
        <div>
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            LIMITATIONS & RESEARCH BOUNDARY
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.04}>
            Credibility depends on saying what the prototype does not establish.
          </TextReveal>
        </div>
        <ul>
          {researchNarrative.limitations.map((x, i) => (
            <TextReveal
              as="li"
              preset="support"
              delay={Math.min(i * 0.04, 0.2)}
              key={x}
            >
              {x}
            </TextReveal>
          ))}
        </ul>
      </div>
    </section>
  );
}
