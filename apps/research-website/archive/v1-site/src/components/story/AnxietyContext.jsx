import { motion, useReducedMotion } from "framer-motion";
import { researchNarrative } from "../../data/research";
import TextReveal from "../motion/TextReveal";

export default function AnxietyContext() {
  const c = researchNarrative.context;
  const reduced = useReducedMotion();

  return (
    <section
      className="story-section context-section"
      id="context"
      data-section="context"
    >
      <div className="shell context-layout">
        <div className="context-copy-column">
          <TextReveal as="p" preset="eyebrow" className="eyebrow">
            {c.label}
          </TextReveal>
          <TextReveal as="h2" preset="heading" delay={0.04}>
            {c.headline}
          </TextReveal>
        </div>

        <div className="context-detail-column">
          <TextReveal
            as="p"
            preset="body"
            className="large-copy context-large-copy"
            delay={0.1}
          >
            {c.body}
          </TextReveal>

          <motion.figure
            className="context-visual-wrap context-art-anchor"
            initial={reduced ? false : { opacity: 0, y: 28, scale: 0.988 }}
            whileInView={reduced ? undefined : { opacity: 1, y: 0, scale: 1 }}
            viewport={{ once: false, amount: 0.18, margin: "-8% 0px -10% 0px" }}
            transition={{ duration: 0.9, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            aria-hidden="true"
          >
            <div className="context-art-glow" />
            <img
              className="context-longitudinal-art context-wave-path"
              src="/media/context-longitudinal-field.svg"
              alt=""
              loading="lazy"
              decoding="async"
            />
          </motion.figure>
        </div>

        <div className="context-section-endcap" aria-hidden="true" />
      </div>
    </section>
  );
}
