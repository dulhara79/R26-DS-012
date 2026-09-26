import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Github, Menu, X } from "lucide-react";
import LogoMark from "../ui/LogoMark";
import { navigation, site } from "../../data/site";

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => setOpen(false), [location.pathname]);

  useEffect(() => {
    const close = (event) => event.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);

  return (
    <nav className="site-nav" aria-label="Primary">
      <div className="site-nav-bar">
        <Link className="brand" to="/">
          <LogoMark />
          <div className="brand-text">
            <strong>{site.id}</strong>
            <span>{site.shortTitle}</span>
          </div>
        </Link>
        <ul className={`nav-links ${open ? "open" : ""}`} id="primary-links">
          {navigation.map(({ to, label }) => (
            <li key={to}>
              <NavLink to={to} end={to === "/"}>
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
        <div className="nav-actions">
          <a
            className="nav-icon"
            href={site.repository}
            target="_blank"
            rel="noreferrer"
            aria-label="Research repository on GitHub"
            title="Research repository on GitHub"
          >
            <Github size={20} />
          </a>
          <button
            className="nav-toggle"
            type="button"
            aria-expanded={open}
            aria-controls="primary-links"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((value) => !value)}
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>
    </nav>
  );
}
