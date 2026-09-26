import TextReveal from "../motion/TextReveal";

export default function CurrentForecastSplit() {
  return (
    <section
      className="research-section now-forward"
      data-section="current-forecast"
    >
      <div className="shell now-forward-grid">
        <article>
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            CURRENT
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.04}>
            What does currently available multimodal evidence indicate?
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            The reliability-aware fusion layer produces a current research
            assessment from eligible evidence. Missing, stale and excluded
            modalities remain explicit.
          </TextReveal>
        </article>
        <article>
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            FORWARD
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.05}>
            What does a validated forecast model indicate over its defined
            horizon?
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.11}>
            Where the current repository describes forecasting, it is
            physiological short-horizon forecasting from C1. This is not
            presented as a validated joint multimodal prediction of an exact
            future anxiety event.
          </TextReveal>
        </article>
      </div>
    </section>
  );
}
