import { motion, useReducedMotion } from "framer-motion";
import TextReveal from "../motion/TextReveal";
const modalities = [
  ["Physiology", "What is happening within the body?"],
  ["Behaviour", "What patterns appear across everyday activity?"],
  ["Clinical language", "What evidence exists in clinical documentation?"],
  ["Context", "What background information influences interpretation?"],
];
export default function ModalityOverview() {
  const reduced = useReducedMotion();
  return (
    <section
      className="research-section modality-section"
      data-section="approach"
    >
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          OUR APPROACH
        </TextReveal>
        <TextReveal as="h2" preset="heading" delay={0.04}>
          Different signals. Different timescales. Different levels of evidence.
        </TextReveal>
        <div className="modality-grid">
          {modalities.map(([name, q], i) => (
            <motion.article
              key={name}
              initial={reduced ? false : { opacity: 0, y: 28 }}
              whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
              viewport={{
                once: false,
                amount: 0.4,
                margin: "-8% 0px -10% 0px",
              }}
              transition={{ duration: 0.55, delay: i * 0.08 }}
              whileHover={reduced ? undefined : { y: -8, scale: 1.012 }}
              whileTap={reduced ? undefined : { scale: 0.995 }}
            >
              <span>0{i + 1}</span>
              <motion.div
                className="modality-orbit"
                aria-hidden="true"
                animate={reduced ? undefined : { rotate: 360 }}
                transition={{
                  duration: 22 + i * 3,
                  repeat: Infinity,
                  ease: "linear",
                }}
              >
                <i />
                <i />
              </motion.div>
              <TextReveal
                as="h3"
                preset="heading"
                delay={Math.min(i * 0.04, 0.12)}
              >
                {name}
              </TextReveal>
              <TextReveal
                as="p"
                preset="body"
                delay={Math.min(i * 0.04 + 0.06, 0.18)}
              >
                {q}
              </TextReveal>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
