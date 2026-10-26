import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { ArrowRight, ChevronDown, Github, Menu, X } from "lucide-react";
import LogoMark from "../ui/LogoMark";
import SiteSearch from "./SiteSearch";
import { navigation, site } from "../../data/site";
import { navigationMenus } from "../../data/navigationMenus.js";
import { navParentForPath } from "../../lib/navParent.js";
import { advanceNavScroll, initialNavScroll, NAV_TOP_BOUNDARY } from "../../lib/navScroll.js";
import "./nav-dropdown.css";
import "./nav-scroll.css";

const desktopQuery = "(min-width: 1181px)";
const hoverQuery = "(min-width: 1181px) and (hover: hover)";

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const [expandedMenu, setExpandedMenu] = useState(null);
  const [scrollMode, setScrollMode] = useState("full");
  const [focusWithin, setFocusWithin] = useState(false);
  const scrollPosition = useRef(initialNavScroll());
  const menuButtons = useRef({});
  const preventFocusOpen = useRef(false);
  const location = useLocation();
  const activeParent = navParentForPath(location.pathname);
  // Menus and keyboard focus keep navigation visible even on a long page.
  const interacting = open || expandedMenu !== null || focusWithin;
  const visibleMode = interacting
    ? (scrollPosition.current.y <= NAV_TOP_BOUNDARY ? "full" : "compact")
    : scrollMode;

  useEffect(() => {
    const initial = initialNavScroll(window.scrollY);
    scrollPosition.current = initial;
    setScrollMode(initial.mode);

    let pendingFrame = 0;
    const updatePosition = () => {
      pendingFrame = 0;
      const next = advanceNavScroll(scrollPosition.current, window.scrollY);
      scrollPosition.current = next;
      setScrollMode((current) => current === next.mode ? current : next.mode);
    };
    const onScroll = () => {
      if (!pendingFrame) pendingFrame = window.requestAnimationFrame(updatePosition);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (pendingFrame) window.cancelAnimationFrame(pendingFrame);
    };
  }, []);

  const closeMenus = () => {
    setOpen(false);
    setExpandedMenu(null);
  };

  // Both route changes and same-page anchor navigation close the open menu.
  useEffect(() => {
    setOpen(false);
    setExpandedMenu(null);
  }, [location.pathname, location.hash]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key !== "Escape") return;
      if (expandedMenu) {
        event.preventDefault();
        const button = menuButtons.current[expandedMenu];
        preventFocusOpen.current = true;
        button?.focus({ preventScroll: true });
        preventFocusOpen.current = false;
        setExpandedMenu(null);
      } else if (open) {
        event.preventDefault();
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [expandedMenu, open]);

  return (
    <nav
      className={`site-nav site-nav--${visibleMode}`}
      aria-label="Primary"
      onFocusCapture={() => setFocusWithin(true)}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocusWithin(false);
      }}
    >
      <div className="site-nav-bar">
        <Link className="brand" to="/" onClick={closeMenus} aria-label="R26-DS-012 — Home">
          <LogoMark />
          <div className="brand-text">
            <strong>{site.id}</strong>
            <span>{site.shortTitle}</span>
          </div>
        </Link>

        <ul className={`nav-links ${open ? "open" : ""}`} id="primary-links">
          {navigation.map(({ to, label }) => {
            const menu = navigationMenus[to];
            const expanded = expandedMenu === to;
            const currentClass = activeParent === to ? "active" : undefined;
            const currentPage = location.pathname === to ? "page"
              : activeParent === to ? "location" : undefined;

            if (!menu) {
              return (
                <li className="nav-item" key={to}>
                  <Link to={to} onClick={closeMenus} className={currentClass} aria-current={currentPage}>
                    {label}
                  </Link>
                </li>
              );
            }

            const dropdownId = "nav-menu-" + to.slice(1);
            const large = to === "/research";
            return (
              <li
                className={`nav-item nav-item--submenu nav-item--${to.slice(1)} ${expanded ? "is-expanded" : ""}`}
                key={to}
                onMouseEnter={() => {
                  if (window.matchMedia(hoverQuery).matches) setExpandedMenu(to);
                }}
                onMouseLeave={() => {
                  if (window.matchMedia(hoverQuery).matches) setExpandedMenu(null);
                }}
                onFocusCapture={() => {
                  if (window.matchMedia(desktopQuery).matches && !preventFocusOpen.current) setExpandedMenu(to);
                }}
                onBlurCapture={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) setExpandedMenu(null);
                }}
              >
                <div className="nav-primary-entry">
                  <Link to={to} onClick={closeMenus} className={currentClass} aria-current={currentPage}>
                    {label}
                  </Link>
                  <button
                    ref={(node) => { menuButtons.current[to] = node; }}
                    className="nav-submenu-toggle"
                    type="button"
                    aria-label={`${expanded ? "Close" : "Show"} ${label} submenu`}
                    aria-expanded={expanded}
                    aria-controls={dropdownId}
                    onClick={() => setExpandedMenu((value) => value === to ? null : to)}
                  >
                    <ChevronDown size={15} aria-hidden="true" />
                  </button>
                </div>
                <div className={`nav-dropdown ${large ? "nav-dropdown--wide" : "nav-dropdown--compact"}`}
                  id={dropdownId} hidden={!expanded}>
                  <div className="nav-dropdown-panel">
                    <div className="nav-dropdown-heading">
                      <strong>{menu.title}</strong>
                      {menu.description && <p>{menu.description}</p>}
                    </div>
                    <div className="nav-dropdown-columns">
                      {menu.columns.map((column) => (
                        <div className="nav-dropdown-column" key={column.label}>
                          <p className="nav-dropdown-category">{column.label}</p>
                          <ul>
                            {column.links.map((link) => (
                              <li key={link.to}>
                                <Link
                                  to={link.to}
                                  className="nav-dropdown-link"
                                  aria-current={location.pathname + location.hash === link.to ? "location" : undefined}
                                  onClick={closeMenus}
                                >
                                  <span>{link.label}</span>
                                  <ArrowRight size={15} aria-hidden="true" />
                                </Link>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                    {menu.footer && (
                      <Link to={menu.footer.to} className="nav-dropdown-footer" onClick={closeMenus}>
                        {menu.footer.label} <ArrowRight size={16} aria-hidden="true" />
                      </Link>
                    )}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>

        <div className="nav-actions">
          <SiteSearch />
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
            onClick={() => {
              setOpen((value) => !value);
              setExpandedMenu(null);
            }}
          >
            {open ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>
    </nav>
  );
}
