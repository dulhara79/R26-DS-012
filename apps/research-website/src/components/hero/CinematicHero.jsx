import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  motion,
  useMotionValue,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
  useTransform,
} from "framer-motion";
import {
  ArrowRight,
  BookOpenCheck,
  FileText,
  GitMerge,
  HeartPulse,
  Network,
} from "lucide-react";
import {
  cinematicIntro,
  cinematicPanels,
  cinematicSights,
  heroVideo,
} from "../../data/cinematicStory";
import "./cinematic-hero.css";

const sightIcons = {
  pulse: HeartPulse,
  graph: Network,
  notes: FileText,
  fusion: GitMerge,
  evidence: BookOpenCheck,
};

const PLAYBACK_RATE = 0.6;
// Seconds of media at the end of each pass that crossfade into the next pass.
const LOOP_FADE = 1.2;

// Scroll progress (0–1 across the pinned stage) where each beat fades in and out.
const BEATS = {
  intro: [0, 0, 0.14, 0.22],
  signals: [0.22, 0.3, 0.44, 0.5],
  fusion: [0.5, 0.58, 0.7, 0.76],
  components: [0.76, 0.84, 1, 1],
};
const BEAT_KEYS = Object.keys(BEATS);

function useBeat(progress, [a, b, c, d], travel = 48) {
  const first = a === b;
  const last = c === d;
  const input = [a, b, c, d];
  const opacity = useTransform(progress, input, [
    first ? 1 : 0,
    1,
    1,
    last ? 1 : 0,
  ]);
  const y = useTransform(progress, input, [
    first ? 0 : travel,
    0,
    0,
    last ? 0 : -travel,
  ]);
  const pointerEvents = useTransform(opacity, (value) =>
    value > 0.5 ? "auto" : "none",
  );
  return { opacity, y, pointerEvents };
}

// Two stacked videos crossfade at the loop point so the restart never jumps.
function HeroVideo({ sectionRef }) {
  const reduced = useReducedMotion();
  const videos = useRef([]);
  const [active, setActive] = useState(0);
  const switching = useRef(false);

  useEffect(() => {
    videos.current.forEach((video) => {
      if (video) video.playbackRate = PLAYBACK_RATE;
    });
    const current = videos.current[active];
    if (!current) return undefined;
    if (reduced) {
      current.pause();
      return undefined;
    }

    // Only play while the hero is on screen.
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) current.play().catch(() => {});
      else current.pause();
    });
    observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, [active, reduced, sectionRef]);

  const onTimeUpdate = (index) => {
    const video = videos.current[index];
    if (index !== active || switching.current || !video?.duration) return;
    if (video.currentTime < video.duration - LOOP_FADE) return;
    const next = videos.current[1 - index];
    switching.current = true;
    next.currentTime = 0;
    next.playbackRate = PLAYBACK_RATE;
    next.play().catch(() => {});
    setActive(1 - index);
    window.setTimeout(() => {
      switching.current = false;
    }, 400);
  };

  return (
    <div className="hero-video" aria-hidden="true">
      {[0, 1].map((index) => (
        <video
          key={index}
          ref={(node) => {
            videos.current[index] = node;
          }}
          className={index === active ? "is-active" : undefined}
          src={heroVideo.src}
          poster={heroVideo.poster}
          muted
          playsInline
          preload={index === 0 ? "auto" : "metadata"}
          onTimeUpdate={() => onTimeUpdate(index)}
          onEnded={(event) => event.currentTarget.pause()}
        />
      ))}
    </div>
  );
}

export default function CinematicHero() {
  const sectionRef = useRef(null);
  const trackRef = useRef(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });

  const intro = useBeat(scrollYProgress, BEATS.intro);
  const signals = useBeat(scrollYProgress, BEATS.signals);
  const fusion = useBeat(scrollYProgress, BEATS.fusion);
  const components = useBeat(scrollYProgress, BEATS.components, 32);

  const videoScale = useTransform(
    scrollYProgress,
    [0, 1],
    [1, reduced ? 1 : 1.14],
  );
  const tint = useTransform(
    scrollYProgress,
    [0, 0.18, 0.3, 1],
    [0, 0, 0.34, 0.42],
  );

  // Cards scrub horizontally through the last beat, measured against the viewport.
  const overflow = useMotionValue(0);
  useEffect(() => {
    const measure = () => {
      const track = trackRef.current;
      if (!track) return;
      overflow.set(
        Math.max(0, track.scrollWidth - track.parentElement.clientWidth),
      );
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [overflow]);
  const trackX = useTransform(
    [scrollYProgress, overflow],
    ([progress, distance]) => {
      const t = Math.min(1, Math.max(0, (progress - 0.82) / 0.16));
      return -distance * t;
    },
  );

  const [beat, setBeat] = useState(0);
  useMotionValueEvent(scrollYProgress, "change", (value) => {
    const index = BEAT_KEYS.findLastIndex(
      (key) => value >= BEATS[key][0] - 0.04,
    );
    setBeat((current) => (current === index ? current : Math.max(0, index)));
  });

  const exploreResearch = () => {
    document
      .getElementById("context")
      ?.scrollIntoView({
        behavior: reduced ? "auto" : "smooth",
        block: "start",
      });
  };

  const {
    signals: signalsCopy,
    fusion: fusionCopy,
    components: componentsCopy,
  } = cinematicPanels;

  return (
    <section
      className="cinema-scroll"
      id="overview"
      ref={sectionRef}
      aria-label="Research introduction"
    >
      <div className="stage">
        <motion.div className="stage-media" style={{ scale: videoScale }}>
          <HeroVideo sectionRef={sectionRef} />
        </motion.div>
        <div className="stage-scrim" aria-hidden="true" />
        <motion.div
          className="stage-tint"
          style={{ opacity: tint }}
          aria-hidden="true"
        />

        <motion.div className="beat beat-intro" style={intro}>
          <p className="beat-eyebrow">{cinematicIntro.eyebrow}</p>
          <h1 className="hero-title">
            {cinematicIntro.title}
            <span className="visually-hidden"> {cinematicIntro.fullTitle}</span>
          </h1>
          <p className="beat-body">{cinematicIntro.lead}</p>
          <div className="hero-tags" aria-label="Evidence streams">
            {cinematicIntro.tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
          <p className="scroll-hint" aria-hidden="true">
            <span /> Scroll to explore
          </p>
        </motion.div>

        <motion.div className="beat" style={signals}>
          <p className="beat-eyebrow">{signalsCopy.eyebrow}</p>
          <h2 className="beat-title">{signalsCopy.title}</h2>
          <p className="beat-body">{signalsCopy.body}</p>
          <dl className="facts">
            {signalsCopy.facts.map(([value, label]) => (
              <div key={value}>
                <dt>{value}</dt>
                <dd>{label}</dd>
              </div>
            ))}
          </dl>
        </motion.div>

        <motion.div className="beat" style={fusion}>
          <p className="beat-eyebrow">{fusionCopy.eyebrow}</p>
          <h2 className="beat-title">{fusionCopy.title}</h2>
          <p className="beat-body">{fusionCopy.body}</p>
          <button
            className="note-button"
            type="button"
            onClick={exploreResearch}
          >
            {fusionCopy.action} <ArrowRight size={18} aria-hidden="true" />
          </button>
        </motion.div>

        <motion.div className="beat beat-components" style={components}>
          <div className="beat-components-head">
            <p className="beat-eyebrow">{componentsCopy.eyebrow}</p>
            <h2 className="beat-title">{componentsCopy.title}</h2>
          </div>
          <div className="sights-viewport">
            <motion.ul
              className="sights-track"
              ref={trackRef}
              style={{ x: trackX }}
            >
              {cinematicSights.map((sight) => {
                const Icon = sightIcons[sight.icon];
                return (
                  <li key={sight.id}>
                    <Link className="sight-card" to={sight.href}>
                      <span className="sight-kicker">{sight.kicker}</span>
                      <span className="sight-pin" aria-hidden="true">
                        <Icon strokeWidth={1.6} />
                      </span>
                      <h3>{sight.title}</h3>
                      <p>{sight.body}</p>
                    </Link>
                  </li>
                );
              })}
            </motion.ul>
          </div>
        </motion.div>

        <ol className="beat-dots" aria-hidden="true">
          {BEAT_KEYS.map((key, index) => (
            <li
              key={key}
              className={index === beat ? "is-active" : undefined}
            />
          ))}
        </ol>
      </div>
    </section>
  );
}
