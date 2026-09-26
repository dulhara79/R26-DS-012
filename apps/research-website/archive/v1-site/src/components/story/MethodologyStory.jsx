import { researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";
export default function MethodologyStory() {
  return (
    <section className="story-section methodology-story">
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          RESEARCH METHODOLOGY
        </TextReveal>
        <TextReveal as="h2" preset="heading" delay={0.04}>
          Evaluation first, integration second.
        </TextReveal>
        <div className="methodology-flow">
          {researchNarrative.methodology.map((x, i) => (
            <TextReveal
              as="div"
              preset="support"
              delay={Math.min(i * 0.04, 0.2)}
              key={x}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              <strong>{x}</strong>
            </TextReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
