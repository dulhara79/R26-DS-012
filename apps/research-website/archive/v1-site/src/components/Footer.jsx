import { Link } from "react-router-dom";
export default function Footer() {
  return (
    <footer className="footer">
      <div className="shell footer-grid">
        <div>
          <strong>R26—DS—012</strong>
          <p>
            A multimodal digital biomarker research framework.
            <br />
            SLIIT · Data Science · 2026
          </p>
        </div>
        <div>
          <Link to="/research">Research</Link>
          <Link to="/findings">Findings</Link>
          <Link to="/documents">Documents</Link>
          <Link to="/team">Team</Link>
        </div>
        <div>
          <p className="fine">
            Research / clinical decision-support prototype. Not a diagnostic
            medical device.
          </p>
          <a
            href="https://github.com/dulhara79/R26-DS-012"
            target="_blank"
            rel="noreferrer"
          >
            Research repository ↗
          </a>
        </div>
      </div>
    </footer>
  );
}
