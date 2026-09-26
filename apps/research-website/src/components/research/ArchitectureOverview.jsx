import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import Reveal from "../ui/Reveal";
import ImageSlot from "../ui/ImageSlot";
import ArchitectureDiagram from "./ArchitectureDiagram";
import { bigPicture, roles } from "../../data/architecture";
import { componentBySlug } from "../../data/components";

// Big-picture description and diagram, followed by the component-by-component
// reading of the same architecture. `detailed` adds the full role table.
export default function ArchitectureOverview({ id = "architecture", detailed = false }) {
  return (
    <>
      <section className="section section--paper2" id={id} aria-labelledby={`${id}-title`}>
        <div className="shell">
          <div className="section-head">
            <Reveal>
              <p className="eyebrow">{bigPicture.eyebrow}</p>
              <h2 className="display" id={`${id}-title`}>
                {bigPicture.title}
              </h2>
            </Reveal>
            <Reveal className="body-copy" delay={0.1}>
              {bigPicture.paragraphs.map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
            </Reveal>
          </div>
          <ArchitectureDiagram />
        </div>
      </section>

      <section className="section" aria-labelledby={`${id}-roles`}>
        <div className="shell">
          <Reveal className="section-head">
            <div>
              <p className="eyebrow">Component by component</p>
              <h2 className="display" id={`${id}-roles`}>
                What each part takes in, and what it hands on.
              </h2>
            </div>
            <p className="lead">
              The same architecture, read one lane at a time: the input each component uses, how it processes it, what it
              publishes, and how fusion treats that output.
            </p>
          </Reveal>
          <div className={detailed ? "role-list" : "role-list role-list--compact"}>
            {roles.map((role, index) => (
              <Reveal className="role" key={role.id} delay={index * 0.06}>
                {!detailed && <ImageSlot name={componentBySlug[role.slug].image} alt="" className="role-image" />}
                <div className="role-head">
                  <span className="fusion-id">{role.id}</span>
                  <h3 className="card-title">{role.title}</h3>
                </div>
                <dl className="role-grid">
                  <div>
                    <dt>Input</dt>
                    <dd>{role.input}</dd>
                  </div>
                  {detailed && (
                    <div>
                      <dt>Processing</dt>
                      <dd>{role.process}</dd>
                    </div>
                  )}
                  <div>
                    <dt>Output</dt>
                    <dd>{role.output}</dd>
                  </div>
                  <div>
                    <dt>In fusion</dt>
                    <dd>{role.fusion}</dd>
                  </div>
                </dl>
                <Link className="role-link" to={`/components/${role.slug}`}>
                  Open {role.id} <ArrowRight size={16} />
                </Link>
              </Reveal>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
