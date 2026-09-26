export default function Applications() {
  return (
    <main className="internal shell">
      <header className="page-intro">
        <p className="eyebrow">APPLICATIONS</p>
        <h1>Audience-specific interfaces, one research state.</h1>
        <p>
          The research architecture supports patient- and clinician-facing views
          around the authoritative backend state.
        </p>
      </header>
      <section className="application-boundary">
        <div>
          <span>PATIENT INTERFACE</span>
          <h2>Audience-appropriate presentation</h2>
        </div>
        <i>← authoritative research state →</i>
        <div>
          <span>CLINICIAN INTERFACE</span>
          <h2>Evidence and modality context</h2>
        </div>
      </section>
      <p className="application-note">
        Real application screenshots will only appear here when approved public
        captures are available; this page does not generate substitute product
        mockups.
      </p>
    </main>
  );
}
