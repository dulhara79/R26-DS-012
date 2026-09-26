import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import ImageSlot from "../components/ui/ImageSlot";
import SectionHead from "../components/ui/SectionHead";
import FusionStates from "../components/research/FusionStates";
import { components } from "../data/components";

export default function Components() {
  return (
    <>
      <PageHero
        eyebrow="Research components"
        title="Four studies. One framework."
        lead="Physiology, behaviour, clinical language and context, each studied with the learning paradigm and evaluation protocol that suits the signal."
        image="bannerComponents"
      />

      <section className="section" aria-label="Components">
        <div className="shell component-rows">
          {components.map((component, index) => (
            <Reveal
              className={`component-row ${index % 2 ? "component-row--flip" : ""}`}
              key={component.id}
            >
              <ImageSlot name={component.image} alt="" />
              <div>
                <p className="eyebrow">
                  {component.id} · {component.modality} · {component.timescale}
                </p>
                <h2
                  className="display"
                  style={{ fontSize: "clamp(2rem,3.6vw,3.2rem)" }}
                >
                  {component.title}
                </h2>
                <p className="lead">{component.tagline}</p>
                <p className="muted" style={{ marginTop: 16 }}>
                  {component.owner} · {component.studentId}
                </p>
                <p className="component-role">{component.fusionRole}</p>
                <div className="button-row">
                  <Link
                    className="button button--ink"
                    to={`/components/${component.slug}`}
                  >
                    Open {component.id} <ArrowRight size={18} />
                  </Link>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <section
        className="section section--paper2"
        aria-labelledby="integration-title"
      >
        <div className="shell">
          <SectionHead
            id="integration-title"
            eyebrow="How they connect"
            title="Component outputs meet in one fusion rule."
            lead="Fusion consumes component-level outputs and status metadata, not raw modality streams."
          />
          <FusionStates />
        </div>
      </section>
    </>
  );
}
