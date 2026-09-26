import { motion, useReducedMotion } from "framer-motion";
import { literatureThemes } from "../../data/literature";
import TextReveal from "../motion/TextReveal";
export default function LiteratureLandscape() {
  const reduced = useReducedMotion();
  return (
    <section
      className="story-section literature-section"
      data-section="literature"
    >
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          02 / WHAT RESEARCH ALREADY TELLS US
        </TextReveal>
        <div className="section-title">
          <TextReveal as="h2" preset="heading" delay={0.04}>
            A literature landscape, not a wall of citations.
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            This public synthesis follows the project research documentation.
            Specific reference records are withheld here unless a verified
            public citation artifact is available.
          </TextReveal>
        </div>
        <div className="literature-list">
          {literatureThemes.map((item, i) => (
            <motion.article
              key={item.theme}
              initial={reduced ? false : { opacity: 0, y: 18 }}
              whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
              viewport={{
                once: false,
                amount: 0.45,
                margin: "-8% 0px -10% 0px",
              }}
              transition={{ duration: 0.5, delay: Math.min(i * 0.05, 0.25) }}
              whileHover={
                reduced
                  ? undefined
                  : { x: 7, backgroundColor: "rgba(255,255,255,.38)" }
              }
            >
              <span>0{i + 1}</span>
              <div>
                <h3>{item.theme}</h3>
                <p>{item.demonstrates}</p>
              </div>
              <div>
                <small>Typical limitation</small>
                <p>{item.limitations}</p>
              </div>
              <div>
                <small>Project relevance</small>
                <p>{item.relevance}</p>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
