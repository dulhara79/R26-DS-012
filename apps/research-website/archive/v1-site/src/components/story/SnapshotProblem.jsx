import { researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";

export default function SnapshotProblem() {
  return (
    <section className="story-section snapshot-section" data-section="snapshot">
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          THE SNAPSHOT PROBLEM
        </TextReveal>
        <div className="snapshot-grid">
          <article>
            <TextReveal as="small" preset="support">
              A CLINICAL ENCOUNTER
            </TextReveal>
            <TextReveal as="h2" preset="heading" delay={0.04}>
              One visible time point.
            </TextReveal>
            <div className="snapshot-point" aria-hidden="true">
              <b />
            </div>
          </article>
          <article>
            <TextReveal as="small" preset="support" delay={0.04}>
              LIFE BETWEEN ENCOUNTERS
            </TextReveal>
            <TextReveal as="h2" preset="heading" delay={0.08}>
              A longer, incomplete timeline.
            </TextReveal>
            <div className="between-line" aria-hidden="true">
              <b />
              <b />
              <b />
              <b />
            </div>
          </article>
        </div>
        <TextReveal
          as="p"
          preset="body"
          className="large-copy snapshot-copy"
          delay={0.08}
        >
          {researchNarrative.snapshot.body}
        </TextReveal>
      </div>
    </section>
  );
}
