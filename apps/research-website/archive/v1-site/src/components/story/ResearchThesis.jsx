import { researchMeta, researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";

export default function ResearchThesis() {
  return (
    <section className="story-section thesis-section" data-section="thesis">
      <div className="shell thesis-inner">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          THE RESEARCH QUESTION
        </TextReveal>
        <TextReveal as="h2" preset="heading" delay={0.05}>
          {researchNarrative.thesis.question}
        </TextReveal>
        <div className="title-reveal">
          <TextReveal as="small" preset="support">
            R26—DS—012 · SLIIT · 2026
          </TextReveal>
          <TextReveal as="h3" preset="heading" delay={0.08}>
            {researchMeta.title}
          </TextReveal>
        </div>
      </div>
    </section>
  );
}
