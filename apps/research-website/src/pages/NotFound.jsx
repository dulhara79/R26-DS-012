import { Link, useLocation } from "react-router-dom";
import { ArrowLeft, ArrowRight, Compass, FileQuestion } from "lucide-react";
import { navigation } from "../data/site.js";
import "./not-found.css";

export default function NotFound() {
  const { pathname } = useLocation();
  return (
    <section className="not-found-page" aria-labelledby="not-found-title">
      <div className="shell not-found-layout">
        <div className="not-found-copy">
          <p className="eyebrow">R26-DS-012 · Navigation</p>
          <span className="not-found-code" aria-hidden="true">404</span>
          <h1 className="display" id="not-found-title">This path doesn't lead to a research page.</h1>
          <p className="lead">
            The address may be outdated or mistyped. Nothing on this page is being presented as research evidence.
            Choose a published section below to continue.
          </p>
          <p className="not-found-path"><FileQuestion size={17} aria-hidden="true" /> Requested path: <code>{pathname}</code></p>
          <div className="not-found-actions">
            <Link className="button button--ink" to="/"><ArrowLeft size={17} aria-hidden="true" /> Return home</Link>
            <Link className="button button--ghost" to="/research"><Compass size={17} aria-hidden="true" /> Explore research</Link>
          </div>
        </div>
        <div className="not-found-panel">
          <span className="not-found-panel-kicker">Find your way</span>
          <h2>Browse the website</h2>
          <nav aria-label="Available research pages">
            <ul>
              {navigation.filter((item) => item.to !== "/").map((item) => (
                <li key={item.to}>
                  <Link to={item.to}>{item.label}<ArrowRight size={17} aria-hidden="true" /></Link>
                </li>
              ))}
            </ul>
          </nav>
          <p>Looking for a specific component? <Link to="/components">Browse C1–C4.</Link></p>
        </div>
      </div>
    </section>
  );
}
