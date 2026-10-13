import { ArrowUpRight, Building2, Github, GraduationCap, Mail } from "lucide-react";
import { useState } from "react";
import Reveal from "../ui/Reveal";
import { site } from "../../data/site";
import { supervisors, team } from "../../data/team";

// Shared contact block: repository, institution, supervisors and team.
export default function ContactSection({ id = "contact", headingLevel = "h2" }) {
  const Heading = headingLevel;
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [subject, setSubject] = useState("");
  const [message, setMessage] = useState("");
  const submitEmail = (event) => {
    event.preventDefault();
    const destination = team.find((person) => person.email)?.email;
    if (!destination) return;
    const body = `From: ${name} (${email})\n\n${message}`;
    window.location.href = `mailto:${destination}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  };
  return (
    <section className="section section--dark contact" id={id} aria-labelledby={`${id}-title`}>
      <div className="shell">
        <Reveal className="section-head">
          <div>
            <p className="eyebrow">Contact</p>
            <Heading className="display" id={`${id}-title`}>
              Questions about the research?
            </Heading>
          </div>
          <p className="lead">
            For questions about the code, data pipelines or results, open an issue on the research repository. For academic
            enquiries, contact the supervisors through their institutional profiles.
          </p>
        </Reveal>

        <div className="contact-grid">
          <Reveal className="contact-card contact-card--primary">
            <Github size={26} aria-hidden="true" />
            <h3 className="card-title">Research repository</h3>
            <p>All four components, the two apps, CI workflows and the project README.</p>
            <div className="button-row" style={{ marginTop: 22 }}>
              <a className="button button--paper" href={site.repository} target="_blank" rel="noreferrer">
                Open on GitHub <ArrowUpRight size={18} />
              </a>
              <a className="button button--ghost" href={site.issues} target="_blank" rel="noreferrer">
                Open an issue <ArrowUpRight size={18} />
              </a>
            </div>
          </Reveal>

          <Reveal className="contact-card" delay={0.06}>
            <Building2 size={26} aria-hidden="true" />
            <h3 className="card-title">Institution</h3>
            <p>
              {site.department}
              <br />
              {site.institution}
              <br />
              {site.campus}
            </p>
            <a className="contact-link" href={site.institutionUrl} target="_blank" rel="noreferrer">
              sliit.lk <ArrowUpRight size={16} />
            </a>
          </Reveal>

          <Reveal className="contact-card" delay={0.12}>
            <GraduationCap size={26} aria-hidden="true" />
            <h3 className="card-title">Supervisors</h3>
            <ul className="contact-people">
              {supervisors.map((person) => (
                <li key={person.name}>
                  <a href={person.profile} target="_blank" rel="noreferrer">
                    <strong>{person.name}</strong>
                    <span>
                      {person.role} · {person.affiliation.split(",")[0]}
                    </span>
                    <ArrowUpRight size={16} aria-hidden="true" />
                  </a>
                </li>
              ))}
            </ul>
          </Reveal>

          <Reveal className="contact-card" delay={0.18}>
            <Mail size={26} aria-hidden="true" />
            <h3 className="card-title">Research team</h3>
            <ul className="contact-people">
              {team.map((person) => (
                <li key={person.id}>
                  {person.email ? (
                    <a href={`mailto:${person.email}`}>
                      <strong>{person.name}</strong>
                      <span>
                        {person.component} · {person.email}
                      </span>
                    </a>
                  ) : (
                    <div>
                      <strong>{person.name}</strong>
                      <span>
                        {person.component} · {person.role}
                      </span>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </Reveal>
        </div>

        <form className="contact-email-form" onSubmit={submitEmail}>
          <h3 className="serif-title">General email template</h3>
          <p>Prepare an email to the research team. This uses your email app and does not send personal information to a website server.</p>
          <div className="contact-email-fields">
            <label>Your name<input required autoComplete="name" value={name} onChange={(event) => setName(event.target.value)} /></label>
            <label>Your email<input required type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
            <label>Subject<input required value={subject} onChange={(event) => setSubject(event.target.value)} /></label>
            <label className="contact-email-message">Message<textarea required rows={5} value={message} onChange={(event) => setMessage(event.target.value)} /></label>
          </div>
          <button type="submit" className="button button--paper">Open email app <ArrowUpRight size={17} /></button>
        </form>
        <p className="contact-note">
          This website shares research, not clinical advice. If you or someone you know is in crisis, contact local emergency
          services or a mental-health helpline in your country.
        </p>
      </div>
    </section>
  );
}
