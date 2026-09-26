import { publicationNote } from "../data/publications";
export default function Publications() {
  return (
    <main className="internal shell">
      <header className="page-intro">
        <p className="eyebrow">PUBLICATIONS</p>
        <h1>Publication record.</h1>
        <p>{publicationNote}</p>
      </header>
      <section className="empty-record">
        <span>VERIFICATION-FIRST</span>
        <h2>No public publication entry has been asserted.</h2>
        <p>
          The research repository remains the current public source for the
          project title, methods and research outputs.
        </p>
        <a
          href="https://github.com/dulhara79/R26-DS-012"
          target="_blank"
          rel="noreferrer"
        >
          Open research repository ↗
        </a>
      </section>
    </main>
  );
}
