import { Link } from "react-router-dom";
import { components } from "../../data/components";
import { navigation, site } from "../../data/site";

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="shell">
        <div className="footer-grid">
          <div>
            <p className="eyebrow" style={{ color: "var(--peach)" }}>
              {site.id} · {site.institutionShort} · {site.year}
            </p>
            <p className="footer-title">{site.title}</p>
            <p
              className="muted"
              style={{ color: "rgba(253,241,225,.66)", marginTop: 18 }}
            >
              {site.degree}
              <br />
              {site.department}, {site.institution}
            </p>
          </div>
          <div>
            <h2>Explore</h2>
            <ul className="footer-links">
              {navigation.map(({ to, label }) => (
                <li key={to}>
                  <Link to={to}>{label}</Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h2>Components</h2>
            <ul className="footer-links">
              {components.map((component) => (
                <li key={component.id}>
                  <Link to={`/components/${component.slug}`}>
                    {component.id} · {component.title}
                  </Link>
                </li>
              ))}
              <li>
                <a href={site.repository} target="_blank" rel="noreferrer">
                  Research repository ↗
                </a>
              </li>
            </ul>
          </div>
        </div>
        <p className="footer-boundary">{site.boundary}</p>
      </div>
    </footer>
  );
}
