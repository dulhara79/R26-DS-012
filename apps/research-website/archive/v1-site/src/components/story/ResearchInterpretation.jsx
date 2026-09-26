import { researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";
export default function ResearchInterpretation() {
  return (
    <section className="story-section interpretation-section">
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          WHAT THE FINDINGS MEAN
        </TextReveal>
        <div className="interpretation-grid">
          {researchNarrative.interpretation.map((x, i) => (
            <TextReveal
              as="article"
              preset="support"
              delay={Math.min(i * 0.05, 0.2)}
              key={x}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              <h3>{x}</h3>
            </TextReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
