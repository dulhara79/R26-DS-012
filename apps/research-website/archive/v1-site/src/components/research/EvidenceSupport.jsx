import TextReveal from "../motion/TextReveal";

const safeguards = [
  "Hybrid dense + lexical retrieval",
  "Reciprocal Rank Fusion and reranking",
  "Authority and freshness scoring",
  "Evidence provenance and source versioning",
  "Contradiction detection",
  "Citation validation",
  "Calibrated abstention",
  "Crisis / urgent-message routing",
];
export default function EvidenceSupport() {
  return (
    <section className="research-section evidence-support">
      <div className="shell evidence-support-grid">
        <div>
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            EVIDENCE SUPPORT
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.04}>
            CARE-AnxRAG supports evidence retrieval—it does not become an
            autonomous clinical decision maker.
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            Its safeguards are designed to make source quality, conflict and
            abstention visible.
          </TextReveal>
        </div>
        <ul>
          {safeguards.map((x, i) => (
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
