import { motion, useReducedMotion } from "framer-motion";
import ImageSlot from "./ImageSlot";

// Preserve the existing landscape photography. Keep the copy readable
// with a dark, left-weighted overlay instead of a decorative ECG ridge.
export default function PageHero({ eyebrow, title, lead, image, meta = [], featureImage, featureAlt = "" }) {
  const reduced = useReducedMotion();

  return (
    <header className={"page-hero" + (featureImage ? " page-hero--component" : "")}>
      {image && (
        <div className="page-hero-media" aria-hidden="true">
          <ImageSlot name={image} />
        </div>
      )}
      <div className={"shell page-hero-content" + (featureImage ? " page-hero-content--split" : "")}>
        <div className="page-hero-copy">
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
        </div>
        {featureImage && (
          <div className="page-hero-feature">
            <ImageSlot key={featureImage} name={featureImage} alt={featureAlt} className="page-hero-feature-image" />
            <span className="page-hero-feature-caption">Research component · R26-DS-012</span>
          </div>
        )}
      </div>
    </header>
  );
}
