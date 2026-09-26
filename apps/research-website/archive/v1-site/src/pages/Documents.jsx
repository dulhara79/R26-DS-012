import DocumentCollection from "../components/documents/DocumentCollection";
export default function Documents() {
  return (
    <main className="internal shell">
      <header className="page-intro">
        <p className="eyebrow">DOCUMENTS</p>
        <h1>Research documents.</h1>
        <p>
          Only public, verified project sources are linked. Confidential or
          unverified artifacts are not represented as available.
        </p>
      </header>
      <DocumentCollection />
    </main>
  );
}
