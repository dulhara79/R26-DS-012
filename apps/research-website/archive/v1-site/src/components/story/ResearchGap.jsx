import { researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";

export default function ResearchGap() {
  return (
    <section className="story-section gap-section" data-section="gap">
      <div className="shell gap-grid">
        <div>
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            03 / WHAT IS STILL MISSING?
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.04}>
            {researchNarrative.gap.headline}
          </TextReveal>
        </div>
        <ol>
          {researchNarrative.gap.challenges.map((x, i) => (
            <TextReveal
              as="li"
              preset="support"
              delay={Math.min(i * 0.05, 0.24)}
              key={x}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              <p>{x}</p>
            </TextReveal>
          ))}
        </ol>
      </div>
    </section>
  );
}
