import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { findings } from "../../data/findings";
import TextReveal from "../motion/TextReveal";
export default function FindingsPreview({ limit = 5 }) {
  const reduced = useReducedMotion();
  return (
    <section className="findings-section" id="findings" data-section="findings">
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          04 / WHAT WE FOUND
        </TextReveal>
        <div className="section-title">
          <TextReveal as="h2" preset="heading" delay={0.04}>
            Results with their evaluation conditions attached.
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            The visual language becomes quieter here: metric, setting and
            interpretation stay together.
          </TextReveal>
        </div>
        <div className="findings-table">
          {findings.slice(0, limit).map((item, i) => (
            <motion.article
              key={`${item.component}-${item.metric}`}
              initial={reduced ? false : { opacity: 0, x: -18 }}
              whileInView={reduced ? undefined : { opacity: 1, x: 0 }}
              viewport={{
                once: false,
                amount: 0.55,
                margin: "-8% 0px -10% 0px",
              }}
              transition={{ duration: 0.45, delay: i * 0.05 }}
              whileHover={
                reduced
                  ? undefined
                  : { x: 7, backgroundColor: "rgba(255,255,255,.58)" }
              }
            >
              <span>{item.component}</span>
              <div>
                <small>{item.context}</small>
                <h3>{item.metric}</h3>
              </div>
              <b>{item.value}</b>
              <p>{item.interpretation}</p>
            </motion.article>
          ))}
        </div>
        <motion.div whileHover={reduced ? undefined : { x: 5 }}>
          <Link className="text-link" to="/findings">
            Explore all findings →
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
