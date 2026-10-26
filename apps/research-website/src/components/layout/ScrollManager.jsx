import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { componentBySlug } from "../../data/components.js";

// Search can open a page AND its specific section, including across hash routes.
export default function ScrollManager() {
  const { pathname, hash } = useLocation();

  useEffect(() => {
    const titles = {
      "/": "Overview", "/research": "Research Domain", "/milestones": "Milestones",
      "/documents": "Documents", "/slides": "Presentation Slides",
      "/team": "About Us", "/contact": "Contact Us", "/components": "Components",
      "/system": "System Architecture", "/findings": "Research Findings",
    };
    const componentSlug = pathname.startsWith("/components/") ? pathname.slice("/components/".length) : null;
    const isKnownComponent = componentSlug && Object.prototype.hasOwnProperty.call(componentBySlug, componentSlug);
    const section = titles[pathname]
      || (isKnownComponent ? "Research Component" : "Not Found");
    document.title = section + " | R26-DS-012 · SLIIT";
  }, [pathname]);

  useEffect(() => {
    if (!hash) {
      window.scrollTo({ top: 0, behavior: "instant" });
      return undefined;
    }
    const id = decodeURIComponent(hash.slice(1));
    let cancelled = false;
    let attempts = 0;
    let frame;
    const findSection = () => {
      if (cancelled) return;
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ block: "start", behavior: "instant" });
      } else if (attempts++ < 18) {
        frame = window.requestAnimationFrame(findSection);
      } else {
        window.scrollTo({ top: 0, behavior: "instant" });
      }
    };
    frame = window.requestAnimationFrame(findSection);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [pathname, hash]);
  return null;
}
