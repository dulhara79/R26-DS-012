import PageHero from "../components/ui/PageHero";
import Reveal from "../components/ui/Reveal";
import ImageSlot from "../components/ui/ImageSlot";
import SectionHead from "../components/ui/SectionHead";
import { architecture, aura, clinanx } from "../data/system";
import { commitments } from "../data/research";

export default function System() {
  return (
    <>
      <PageHero
        eyebrow="Integrated system"
        title="From signal to supported decision."
        lead="Two Flutter apps, one central backend, four component services, a fusion rule and an evidence layer, with safety gates between each step."
        image="bannerSystem"
      />

      <section className="section" aria-labelledby="arch-title">
        <div className="shell">
          <SectionHead
            id="arch-title"
            eyebrow="Architecture"
            title="Every model call happens server-side."
            lead="The apps are presentation layers. They never compute a score, hold a weight table or call a component directly."
          />
          <Reveal as="ol" className="arch">
            {architecture.map((layer, index) => (
              <li key={layer.layer}>
                <span className="arch-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <strong>{layer.layer}</strong>
                <div className="pill-list">
                  {layer.items.map((item) => (
                    <span className="pill" key={item}>
                      {item}
                    </span>
                  ))}
                </div>
              </li>
            ))}
          </Reveal>
        </div>
      </section>

      <section className="section section--paper2" aria-labelledby="aura-title">
        <div className="shell app-block">
          <Reveal className="app-media app-media--phone">
            <ImageSlot name="appAura" alt="Aura participant app" />
          </Reveal>
          <Reveal delay={0.1}>
            <p className="eyebrow">{aura.role}</p>
            <h2 className="display" id="aura-title">
              {aura.name}
            </h2>
            <p className="lead">{aura.summary}</p>
            {aura.flows.map((flow) => (
              <ol className="inline-flow" key={flow[0]}>
                {flow.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            ))}
          </Reveal>
        </div>
        <div className="shell grid-2" style={{ marginTop: 56 }}>
          <Reveal>
            <h3 className="serif-title detail-sub">
              Passive sensing collection
            </h3>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Signal</th>
                    <th scope="col">Collection pattern</th>
                  </tr>
                </thead>
                <tbody>
                  {aura.collection.map(([signal, pattern]) => (
                    <tr key={signal}>
                      <td>{signal}</td>
                      <td className="muted">{pattern}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>
          <Reveal delay={0.08}>
            <h3 className="serif-title detail-sub">Privacy by design</h3>
            <ul className="check-list">
              {aura.privacy.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Reveal>
        </div>
      </section>

      <section className="section" aria-labelledby="clinanx-title">
        <div className="shell app-block app-block--flip">
          <Reveal className="app-media">
            <ImageSlot name="appClinAnx" alt="ClinAnx clinician console" />
          </Reveal>
          <Reveal delay={0.1}>
            <p className="eyebrow">{clinanx.role}</p>
            <h2 className="display" id="clinanx-title">
              {clinanx.name}
            </h2>
            <p className="lead">{clinanx.summary}</p>
          </Reveal>
        </div>
        <div className="shell grid-2" style={{ marginTop: 56 }}>
          <Reveal>
            <h3 className="serif-title detail-sub">Screens</h3>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Screen</th>
                    <th scope="col">Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  {clinanx.screens.map(([screen, purpose]) => (
                    <tr key={screen}>
                      <td>{screen}</td>
                      <td className="muted">{purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>
          <Reveal delay={0.08}>
            <h3 className="serif-title detail-sub">
              What it deliberately does not do
            </h3>
            <ul className="check-list">
              {clinanx.doesNot.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Reveal>
        </div>
      </section>

      <section className="section section--dark" aria-labelledby="commit-title">
        <div className="shell">
          <SectionHead
            id="commit-title"
            eyebrow="Privacy and research-safety commitments"
            title="Commitments across the integrated system."
          />
          <Reveal as="ul" className="check-list check-list--columns">
            {commitments.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </Reveal>
        </div>
      </section>
    </>
  );
}
