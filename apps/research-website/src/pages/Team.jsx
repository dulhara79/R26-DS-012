import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import ImageSlot from "../components/ui/ImageSlot";
import SectionHead from "../components/ui/SectionHead";
import { components } from "../data/components";
import { supervisors, team } from "../data/team";
import { site } from "../data/site";

export default function Team() {
  const slugFor = (id) =>
    components.find((component) => component.id === id)?.slug;
  return (
    <>
      <PageHero
        eyebrow="Team and supervision"
        title="The people behind R26-DS-012."
        lead={`${site.degree} · ${site.department}, ${site.institution}`}
        image="bannerTeam"
      />

      <section className="section" aria-labelledby="team-title">
        <div className="shell">
          <SectionHead
            id="team-title"
            eyebrow="Research team"
            title="One component each."
          />
          <div className="grid-4">
            {team.map((person, index) => (
              <Reveal key={person.id} delay={index * 0.08}>
                <Link
                  className="card link-card person-card"
                  to={`/components/${slugFor(person.component)}`}
                >
                  <ImageSlot
                    name={person.photo}
                    alt={`Portrait of ${person.name}`}
                  />
                  <div className="link-card-body">
                    <span className="card-kicker">
                      {person.component} · {person.id}
                    </span>
                    <h3 className="card-title">{person.name}</h3>
                    <p>{person.role}</p>
                  </div>
                </Link>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="section section--paper2" aria-labelledby="sup-title">
        <div className="shell">
          <SectionHead
            id="sup-title"
            eyebrow="Supervisors"
            title="Guidance across computing and psychiatry."
          />
          <div className="grid-3">
            {supervisors.map((person, index) => (
              <Reveal
                className="card person-card"
                key={person.name}
                delay={index * 0.08}
              >
                <ImageSlot
                  name={person.photo}
                  alt={`Portrait of ${person.name}`}
                  className="person-card-photo"
                />
                <span className="card-kicker" style={{ marginTop: 22 }}>
                  {person.role}
                </span>
                <h3 className="card-title">{person.name}</h3>
                <p>{person.affiliation}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section
        className="section section--dark"
        aria-labelledby="contact-title"
      >
        <div className="shell split">
          <Reveal>
            <p className="eyebrow">Contact</p>
            <h2 className="display" id="contact-title">
              Questions about the research?
            </h2>
            <p className="lead">
              Open an issue or discussion on the project repository, or reach
              the team through the {site.department}, {site.institutionShort}.
            </p>
          </Reveal>
          <Reveal delay={0.1}>
            <div className="button-row">
              <a
                className="button button--paper"
                href={site.repository}
                target="_blank"
                rel="noreferrer"
              >
                Research repository <ArrowUpRight size={18} />
              </a>
              <a
                className="button button--ghost"
                href={site.websiteRepository}
                target="_blank"
                rel="noreferrer"
              >
                Website source <ArrowUpRight size={18} />
              </a>
            </div>
          </Reveal>
        </div>
      </section>
    </>
  );
}
