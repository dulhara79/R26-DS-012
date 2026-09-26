import {
  motion,
  useReducedMotion,
  useScroll,
  useTransform,
} from "framer-motion";
import { useRef } from "react";
import SignalRidge from "./SignalRidge";
import ImageSlot from "./ImageSlot";

// Inner-page header: dawn sky (or a banner image), signal ridge and a serif title,
// echoing the opening frame of the landing scroll.
export default function PageHero({ eyebrow, title, lead, image, meta = [] }) {
  const ref = useRef(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });
  const ridgeY = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : 80]);
  const copyY = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : -60]);
  const copyOpacity = useTransform(
    scrollYProgress,
    [0, 0.8],
    [1, reduced ? 1 : 0],
  );

  return (
    <header className="page-hero" ref={ref}>
      {image && (
        <div className="page-hero-media" aria-hidden="true">
          <ImageSlot name={image} />
        </div>
      )}
      <motion.div
        className="page-hero-ridge"
        style={{ y: ridgeY }}
        aria-hidden="true"
      >
        <SignalRidge />
      </motion.div>
      <motion.div className="shell" style={{ y: copyY, opacity: copyOpacity }}>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <motion.h1
          className="display display--xl"
          initial={reduced ? false : { opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        >
          {title}
        </motion.h1>
        {lead && (
          <motion.p
            className="lead"
            initial={reduced ? false : { opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.9,
              delay: 0.15,
              ease: [0.22, 1, 0.36, 1],
            }}
          >
            {lead}
          </motion.p>
        )}
        {meta.length > 0 && (
          <div className="page-hero-meta">
            {meta.map((item) => (
              <span className="pill" key={item}>
                {item}
              </span>
            ))}
          </div>
        )}
      </motion.div>
    </header>
  );
}
