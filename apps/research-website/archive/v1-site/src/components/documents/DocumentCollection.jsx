import { motion, useReducedMotion } from "framer-motion";
import { documents } from "../../data/documents";
export default function DocumentCollection({ compact = false }) {
  const reduced = useReducedMotion();
  const groups = [...new Set(documents.map((d) => d.category))];
  return (
    <div className="document-collections">
      {groups.map((group) => (
        <section key={group}>
          <h3>{group}</h3>
          {documents
            .filter((d) => d.category === group)
            .slice(0, compact ? 2 : 99)
            .map((d, i) => (
              <motion.article
                key={d.title}
                initial={reduced ? false : { opacity: 0, y: 14 }}
                whileInView={reduced ? undefined : { opacity: 1, y: 0 }}
                viewport={{
                  once: false,
                  amount: 0.65,
                  margin: "-8% 0px -10% 0px",
                }}
                transition={{ duration: 0.42, delay: Math.min(i * 0.04, 0.16) }}
                whileHover={
                  reduced
                    ? undefined
                    : { x: 6, backgroundColor: "rgba(255,255,255,.42)" }
                }
              >
                <div>
                  <small>
                    {d.type} · {d.year}
                  </small>
                  <h4>{d.title}</h4>
                  <p>
                    {d.authors} · {d.status}
                  </p>
                </div>
                {d.url && (
                  <motion.a
                    href={d.url}
                    target="_blank"
                    rel="noreferrer"
                    whileHover={reduced ? undefined : { x: 4 }}
                  >
                    View ↗
                  </motion.a>
                )}
              </motion.article>
            ))}
        </section>
      ))}
    </div>
  );
}
