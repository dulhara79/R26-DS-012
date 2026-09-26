import { motion, useReducedMotion } from "framer-motion";
import { architectureStages, architectureNote } from "../../data/architecture";
import TextReveal from "../motion/TextReveal";
export default function ArchitectureStory() {
  const reduced = useReducedMotion();
  return (
    <section
      className="research-section architecture-section"
      id="system"
      data-section="architecture"
    >
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          SYSTEM ARCHITECTURE
        </TextReveal>
        <div className="section-title">
          <TextReveal as="h2" preset="heading" delay={0.04}>
            From participant signals to one governed research state.
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            {architectureNote}
          </TextReveal>
        </div>
        <div
          className="architecture-flow"
          role="img"
          aria-label="Progressive research architecture"
        >
          {architectureStages.map((stage, i) => (
            <motion.div
              className="architecture-node"
              key={stage}
              initial={reduced ? false : { opacity: 0, y: 24, scale: 0.97 }}
              whileInView={reduced ? undefined : { opacity: 1, y: 0, scale: 1 }}
              viewport={{
                once: false,
                amount: 0.55,
                margin: "-8% 0px -10% 0px",
              }}
              transition={{ duration: 0.48, delay: i * 0.06 }}
              whileHover={reduced ? undefined : { y: -7, scale: 1.025 }}
            >
              <span>{String(i + 1).padStart(2, "0")}</span>
              <strong>{stage}</strong>
              {i < architectureStages.length - 1 && <i>↓</i>}
            </motion.div>
          ))}
        </div>
        <motion.a
          className="text-link"
          href="https://github.com/dulhara79/R26-DS-012/blob/main/full.png"
          target="_blank"
          rel="noreferrer"
          whileHover={reduced ? undefined : { x: 5 }}
        >
          View Full Architecture ↗
        </motion.a>
      </div>
    </section>
  );
}
