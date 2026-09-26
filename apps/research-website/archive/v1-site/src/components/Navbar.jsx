import { useEffect, useState } from "react";
import {
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
} from "framer-motion";
import { Menu, X } from "lucide-react";
import useSectionNavigation from "../hooks/useSectionNavigation";

const SCROLL_COMPACT_THRESHOLD = 92;

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const [compact, setCompact] = useState(false);
  const reduced = useReducedMotion();
  const { scrollY } = useScroll();
  const { activeSection, goToSection, sections } = useSectionNavigation();

  useMotionValueEvent(scrollY, "change", (latest) => {
    const next = latest > SCROLL_COMPACT_THRESHOLD;
    setCompact((current) => (current === next ? current : next));
  });

  useEffect(() => {
    const close = (event) => event.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);

  const select = (id) => {
    setOpen(false);
    goToSection(id);
  };

  const spring = reduced
    ? { duration: 0 }
    : { type: "spring", stiffness: 280, damping: 30, mass: 0.82 };

  return (
    <header className={`site-header ${compact ? "site-header--compact" : ""}`}>
      <motion.div
        className={`nav-shell ${compact ? "nav-shell--compact" : ""}`}
        layout="position"
        transition={{ layout: spring }}
      >
        <motion.button
          className="brand brand-button"
          type="button"
          onClick={() => select("overview")}
          animate={{
            opacity: compact ? 0 : 1,
            width: compact ? 0 : "auto",
            marginRight: compact ? 0 : 18,
          }}
          transition={spring}
          whileHover={compact ? undefined : { y: -1 }}
          whileTap={compact ? undefined : { scale: 0.98 }}
          tabIndex={compact ? -1 : 0}
          aria-hidden={compact}
        >
          <strong>R26—DS—012</strong>
          <span>Multimodal Anxiety Research</span>
        </motion.button>

        <button
          className="menu"
          type="button"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>

        <motion.nav
          aria-label="Primary"
          className={open ? "nav-links open" : "nav-links"}
          layout="position"
          transition={{ layout: spring }}
        >
          {sections.map(({ id, label }) => (
            <motion.button
              type="button"
              key={id}
              className={`nav-section-link ${activeSection === id ? "active" : ""}`}
              aria-current={activeSection === id ? "location" : undefined}
              onClick={() => select(id)}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.97 }}
            >
              <span>{label}</span>
              {activeSection === id && (
                <motion.span
                  className="nav-active-indicator"
                  layoutId="nav-active-indicator"
                  transition={{ type: "spring", stiffness: 420, damping: 36 }}
                />
              )}
            </motion.button>
          ))}
        </motion.nav>

        <motion.button
          className="explore-link explore-button"
          type="button"
          onClick={() => select("context")}
          animate={{
            opacity: compact ? 0 : 1,
            width: compact ? 0 : "auto",
            paddingLeft: compact ? 0 : 16,
            paddingRight: compact ? 0 : 16,
            marginLeft: compact ? 0 : 18,
          }}
          transition={spring}
          whileHover={compact ? undefined : { y: -2, scale: 1.02 }}
          whileTap={compact ? undefined : { scale: 0.98 }}
          tabIndex={compact ? -1 : 0}
          aria-hidden={compact}
        >
          Explore Research
        </motion.button>
      </motion.div>
    </header>
  );
}
