import { useEffect } from "react";
import { useLocation } from "react-router-dom";

// Resets scroll on route change so each page opens at its header.
export default function ScrollManager() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [pathname]);
  return null;
}
