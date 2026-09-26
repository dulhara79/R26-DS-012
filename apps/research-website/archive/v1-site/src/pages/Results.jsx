import FindingsPreview from "../components/evidence/FindingsPreview";
import ResearchInterpretation from "../components/story/ResearchInterpretation";
export default function Results() {
  return (
    <main className="deep-page">
      <header className="page-intro shell">
        <p className="eyebrow">FINDINGS</p>
        <h1>Evidence record.</h1>
        <p>
          Every public number is paired with the evaluation setting and the
          interpretation that constrains the claim.
        </p>
      </header>
      <FindingsPreview limit={99} />
      <ResearchInterpretation />
    </main>
  );
}
