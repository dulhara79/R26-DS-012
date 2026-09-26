import TextReveal from "../motion/TextReveal";

const factors = [
  "availability",
  "recency",
  "coverage",
  "reliability",
  "validation evidence",
];
const states = ["strong", "stale", "incomplete", "unavailable", "excluded"];
export default function FusionStory() {
  return (
    <section className="research-section fusion-section" data-section="fusion">
      <div className="shell fusion-grid">
        <div>
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            WHY MULTIMODAL?
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.04}>
            More data is not automatically better evidence.
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            Eligible signals reach fusion only after quality and evidence
            checks. A weak or unavailable stream should reduce certainty rather
            than silently become a low value.
          </TextReveal>
          <div className="state-row">
            {states.map((s, i) => (
              <TextReveal
                as="span"
                preset="support"
                delay={Math.min(i * 0.04, 0.16)}
                key={s}
              >
                {s}
              </TextReveal>
            ))}
          </div>
        </div>
        <div className="fusion-equation">
          {factors.map((f, i) => (
            <TextReveal
              as="div"
              preset="support"
              delay={Math.min(i * 0.05, 0.2)}
              key={f}
            >
              <span>{f}</span>
              {i < factors.length - 1 && <b>+</b>}
            </TextReveal>
          ))}
          <i>↓</i>
          <TextReveal as="strong" preset="heading" delay={0.08}>
            usable contribution
          </TextReveal>
          <TextReveal as="small" preset="support" delay={0.14}>
            The system may return insufficient evidence.
          </TextReveal>
        </div>
      </div>
    </section>
  );
}
