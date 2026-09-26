import { timeline } from "../../data/timeline";
import TextReveal from "../motion/TextReveal";
export default function ResearchTimeline() {
  return (
    <section className="story-section journey-section">
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          RESEARCH JOURNEY
        </TextReveal>
        <TextReveal as="h2" preset="heading" delay={0.04}>
          A project timeline that only uses verified public timing.
        </TextReveal>
        <div className="journey-line">
          {timeline.map((x, i) => (
            <TextReveal
              as="article"
              preset="support"
              delay={Math.min(i * 0.06, 0.18)}
              key={`${x.year}-${x.title}`}
            >
              <span>{x.year}</span>
              <i aria-hidden="true" />
              <h3>{x.title}</h3>
              <p>{x.note}</p>
            </TextReveal>
          ))}
        </div>
      </div>
    </section>
  );
}
