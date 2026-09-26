import { useRef, useState } from "react";
import {
  AnimatePresence,
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
} from "framer-motion";
import { introChapters } from "../../data/introStory";
import { chapterIndexFromProgress } from "../../utils/introStoryState";
import "./immersive-introduction.css";

function ChapterDetails({ details }) {
  if (!details?.length) return null;

  return (
    <div className="intro-details">
      {details.map((detail, index) => {
        const isText = typeof detail === "string";
        return (
          <div className="intro-detail" key={isText ? detail : `${detail.label}-${detail.value}`}>
            <span>{isText ? String(index + 1).padStart(2, "0") : detail.label}</span>
            <p>{isText ? detail : detail.value}</p>
          </div>
        );
      })}
    </div>
  );
}

function SceneArtwork({ scene }) {
  return (
    <div className={`intro-scene intro-scene--${scene}`} aria-hidden="true">
      <span className="intro-scene__orb intro-scene__orb--a" />
      <span className="intro-scene__orb intro-scene__orb--b" />
      <span className="intro-scene__orb intro-scene__orb--c" />
      <span className="intro-scene__thread intro-scene__thread--a" />
      <span className="intro-scene__thread intro-scene__thread--b" />
      <span className="intro-scene__thread intro-scene__thread--c" />
      <span className="intro-scene__thread intro-scene__thread--d" />
      <span className="intro-scene__grain" />
    </div>
  );
}

function StaticIntroduction() {
  return (
    <section className="intro-static" id="context" aria-label="Research introduction">
      {introChapters.map((chapter) => (
        <article className="intro-static__chapter" key={chapter.id}>
          <div className="shell intro-static__inner">
            <p className="intro-eyebrow">
              {chapter.number} / {chapter.eyebrow}
            </p>
            <h2>{chapter.title}</h2>
            <p className="intro-body">{chapter.body}</p>
            <ChapterDetails details={chapter.details} />
            {chapter.note && <p className="intro-note">{chapter.note}</p>}
          </div>
        </article>
      ))}
    </section>
  );
}

export default function ImmersiveIntroduction() {
  const sectionRef = useRef(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });

  useMotionValueEvent(scrollYProgress, "change", (latest) => {
    const next = chapterIndexFromProgress(latest, introChapters.length);
    setActiveIndex((current) => (current === next ? current : next));
  });

  if (reduced) return <StaticIntroduction />;

  const active = introChapters[activeIndex];

  return (
    <section
      className="immersive-intro"
      id="context"
      ref={sectionRef}
      data-scene={active.scene}
      style={{ "--intro-chapters": introChapters.length }}
      aria-label="Research introduction"
    >
      <div className="intro-sticky">
        <div className="intro-scenes" aria-hidden="true">
          <AnimatePresence initial={false}>
            <motion.div
              className="intro-scene-layer"
              key={active.scene}
              initial={{ opacity: 0, scale: 1.025 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.985 }}
              transition={{ duration: 0.72, ease: [0.22, 1, 0.36, 1] }}
            >
              <SceneArtwork scene={active.scene} />
            </motion.div>
          </AnimatePresence>
          <div className="intro-scene-shade" />
        </div>

        <div className="shell intro-shell">
          <div className="intro-topline" aria-hidden="true">
            <span>INTRODUCTION · {String(activeIndex + 1).padStart(2, "0")} / {String(introChapters.length).padStart(2, "0")}</span>
            <span>SCROLL TO MOVE THROUGH THE RESEARCH STORY</span>
          </div>

          <div className="intro-stage">
            <AnimatePresence mode="wait" initial={false}>
              <motion.article
                className="intro-copy"
                key={active.id}
                initial={{ opacity: 0, y: 34, filter: "blur(8px)" }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -24, filter: "blur(6px)" }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              >
                <p className="intro-eyebrow">
                  {active.number} / {active.eyebrow}
                </p>
                <h2>{active.title}</h2>
                <p className="intro-body">{active.body}</p>
                <ChapterDetails details={active.details} />
                {active.note && <p className="intro-note">{active.note}</p>}
              </motion.article>
            </AnimatePresence>

            <nav className="intro-rail" aria-label="Introduction progress">
              <ol>
                {introChapters.map((chapter, index) => (
                  <li className={index === activeIndex ? "active" : ""} key={chapter.id}>
                    <span className="intro-rail__dot" aria-hidden="true" />
                    <span className="intro-rail__label">{chapter.eyebrow}</span>
                  </li>
                ))}
              </ol>
            </nav>
          </div>

          <div className="intro-bottomline" aria-hidden="true">
            <span>{activeIndex === introChapters.length - 1 ? "CONTINUE TO THE FOUR COMPONENTS" : "SCROLL"}</span>
            <div className="intro-progress-track">
              <motion.span style={{ scaleX: scrollYProgress }} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
