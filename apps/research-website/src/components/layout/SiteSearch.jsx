import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowRight, FileText, Search, X } from "lucide-react";
import { searchCategories, searchSite, suggestedSearches } from "../../data/search.js";
import "./site-search.css";

export default function SiteSearch() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [category, setCategory] = useState("All");
  const [active, setActive] = useState(-1);
  const inputRef = useRef(null);
  const triggerRef = useRef(null);
  const panelRef = useRef(null);
  const navigate = useNavigate();
  const location = useLocation();
  const results = useMemo(() => searchSite(query, { category }), [query, category]);
  const visible = submitted ? results : results.slice(0, 7);

  function close({ restoreFocus = true } = {}) {
    setOpen(false);
    setQuery("");
    setSubmitted(false);
    setCategory("All");
    setActive(-1);
    if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus());
  }

  function goTo(result) {
    close({ restoreFocus: false });
    const [path, fragment] = result.href.split("#");
    const destination = { pathname: path || "/", hash: fragment ? "#" + fragment : "" };
    if (path === location.pathname && (fragment || "") === location.hash.slice(1)) {
      if (fragment) document.getElementById(fragment)?.scrollIntoView({ block: "start", behavior: "smooth" });
    } else {
      navigate(destination);
    }
  }

  function submit() {
    if (!query.trim()) {
      inputRef.current?.focus();
      return;
    }
    setSubmitted(true);
    setActive(-1);
  }

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
      if (event.key === "Tab") {
        const focusable = [...(panelRef.current?.querySelectorAll(
          'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
        ) ?? [])];
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    requestAnimationFrame(() => inputRef.current?.focus());
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  useEffect(() => {
    const onShortcut = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
    };
    document.addEventListener("keydown", onShortcut);
    return () => document.removeEventListener("keydown", onShortcut);
  }, []);

  function onInputKeyDown(event) {
    if (event.key === "ArrowDown" && visible.length) {
      event.preventDefault();
      setActive(current => (current + 1) % visible.length);
    } else if (event.key === "ArrowUp" && visible.length) {
      event.preventDefault();
      setActive(current => current < 0 ? visible.length - 1 : (current - 1 + visible.length) % visible.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (active >= 0 && visible[active]) goTo(visible[active]);
      else submit();
    }
  }

  return (
    <>
      <button ref={triggerRef} className="nav-icon nav-search-trigger" type="button"
        aria-label="Search this website" title="Search this website (Ctrl+K)"
        aria-expanded={open} aria-haspopup="dialog" onClick={() => setOpen(true)}>
        <Search size={20} aria-hidden="true" />
      </button>
      {open && createPortal(
        <div className="site-search-backdrop" onMouseDown={(event) => {
          if (event.target === event.currentTarget) close();
        }}>
          <div ref={panelRef} className="site-search-dialog" role="dialog" aria-modal="true"
            aria-label="Search the research website">
            <form className="site-search-form" role="search" onSubmit={(event) => {
              event.preventDefault();
              submit();
            }}>
              <Search size={22} aria-hidden="true" />
              <input ref={inputRef} id="site-search-input" type="search"
                autoComplete="off" spellCheck="false"
                aria-label="Search website content"
                aria-controls="site-search-results"
                value={query}
                placeholder="Search research, documents, components…"
                onChange={event => {
                  setQuery(event.target.value);
                  setSubmitted(false);
                  setCategory("All");
                  setActive(-1);
                }}
                onKeyDown={onInputKeyDown} />
              <button className="site-search-submit" type="submit" aria-label="Search">
                <Search size={18} aria-hidden="true" />
                <span>Search</span>
              </button>
              <button className="site-search-close" type="button" aria-label="Close search"
                onClick={() => close()}><X size={20} aria-hidden="true" /></button>
            </form>

            {!query.trim() ? (
              <div className="site-search-start">
                <p className="site-search-label">Start exploring</p>
                <p className="site-search-hint">Find information already published on this website, including document links and research findings.</p>
                <div className="site-search-suggestions">
                  {suggestedSearches.map(term => (
                    <button type="button" key={term} onClick={() => {
                      setQuery(term);
                      setSubmitted(false);
                      setActive(-1);
                      inputRef.current?.focus();
                    }}>{term}<ArrowRight size={15} aria-hidden="true" /></button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {submitted && (
                  <div className="site-search-filters" aria-label="Filter search results">
                    {searchCategories.map(option => (
                      <button key={option} type="button" className={category === option ? "is-selected" : ""}
                        aria-pressed={category === option} onClick={() => {
                          setCategory(option);
                          setActive(-1);
                        }}>{option === "Finding" ? "Findings" : option === "Person" ? "People" : option === "All" ? "All" : option + "s"}</button>
                    ))}
                  </div>
                )}
                <div className="site-search-result-head" role="status" aria-live="polite">
                  <span>{submitted ? "Search results" : "Suggested matches"}</span>
                  <span>{results.length} {results.length === 1 ? "match" : "matches"}</span>
                </div>
                <div className="site-search-results" id="site-search-results">
                  {visible.length ? visible.map((result, index) => (
                    <button className={"site-search-item " + (active === index ? "is-active" : "")}
                      type="button" key={result.id} onClick={() => goTo(result)}
                      onMouseEnter={() => setActive(index)}>
                      <span className="site-search-item-icon" aria-hidden="true">
                        {result.category === "Document" ? <FileText size={18} /> : <ArrowRight size={18} />}
                      </span>
                      <span className="site-search-item-copy">
                        <span className="site-search-item-type">{result.category}</span>
                        <strong>{result.title}</strong>
                        <span className="site-search-item-description">{result.excerpt || result.description}</span>
                      </span>
                      <ArrowRight size={17} className="site-search-item-arrow" aria-hidden="true" />
                    </button>
                  )) : (
                    <div className="site-search-empty">
                      <strong>No matches found.</strong>
                      <p>Try a different term, such as “TC-WPN”, “proposal”, or “fusion”. Search covers the website and linked document metadata, not the text inside Drive files.</p>
                    </div>
                  )}
                </div>
                {!submitted && results.length > visible.length && (
                  <button className="site-search-see-all" type="button" onClick={submit}>
                    View all {results.length} results <ArrowRight size={17} aria-hidden="true" />
                  </button>
                )}
              </>
            )}
            <div className="site-search-footer">
              <span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
              <span><kbd>Enter</kbd> Search / open selected</span>
              <span><kbd>Esc</kbd> Close</span>
            </div>
          </div>
        </div>, document.body)}
    </>
  );
}
