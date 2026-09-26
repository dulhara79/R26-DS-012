import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useReducedMotion } from "framer-motion";

export const landingSections = [
  { id: "overview", label: "Overview" },
  { id: "context", label: "Research" },
  { id: "components", label: "Components" },
  { id: "system", label: "System" },
  { id: "findings", label: "Findings" },
  { id: "documents", label: "Documents" },
  { id: "team", label: "Team" },
];

const routeSection = {
  "/": "overview",
  "/research": "context",
  "/methodology": "context",
  "/components": "components",
  "/system": "system",
  "/applications": "system",
  "/findings": "findings",
  "/results": "findings",
  "/documents": "documents",
  "/publications": "documents",
  "/team": "team",
};

export default function useSectionNavigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const reduced = useReducedMotion();
  const [activeSection, setActiveSection] = useState(
    routeSection[location.pathname] || "overview",
  );

  useEffect(() => {
    if (location.pathname !== "/") {
      setActiveSection(routeSection[location.pathname] || "overview");
      return undefined;
    }

    let frame = 0;
    const update = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const probe = window.scrollY + Math.min(180, window.innerHeight * 0.28);
        let current = landingSections[0].id;
        for (const section of landingSections) {
          const node = document.getElementById(section.id);
          if (!node) continue;
          const top = node.getBoundingClientRect().top + window.scrollY;
          if (top <= probe) current = section.id;
          else break;
        }
        if (
          window.innerHeight + window.scrollY >=
          document.documentElement.scrollHeight - 4
        ) {
          current = landingSections[landingSections.length - 1].id;
        }
        setActiveSection(current);
      });
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, [location.pathname]);

  useEffect(() => {
    const target = location.state?.scrollTo;
    if (location.pathname !== "/" || !target) return;
    const frame = requestAnimationFrame(() => {
      const node = document.getElementById(target);
      if (node)
        node.scrollIntoView({
          behavior: reduced ? "auto" : "smooth",
          block: "start",
        });
      navigate("/", { replace: true, state: null });
    });
    return () => cancelAnimationFrame(frame);
  }, [location.pathname, location.state, navigate, reduced]);

  const goToSection = useCallback(
    (id) => {
      setActiveSection(id);
      if (location.pathname === "/") {
        document
          .getElementById(id)
          ?.scrollIntoView({
            behavior: reduced ? "auto" : "smooth",
            block: "start",
          });
        return;
      }
      navigate("/", { state: { scrollTo: id } });
    },
    [location.pathname, navigate, reduced],
  );

  return { activeSection, goToSection, sections: landingSections };
}
