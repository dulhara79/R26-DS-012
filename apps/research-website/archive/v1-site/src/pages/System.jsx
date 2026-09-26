import FusionStory from "../components/research/FusionStory";
import ArchitectureStory from "../components/research/ArchitectureStory";
import CurrentForecastSplit from "../components/research/CurrentForecastSplit";
import EvidenceSupport from "../components/research/EvidenceSupport";
export default function System() {
  return (
    <main className="deep-page">
      <header className="page-intro shell">
        <p className="eyebrow">SYSTEM</p>
        <h1>Quality before contribution.</h1>
        <p>
          Availability, recency, coverage, reliability and validation evidence
          determine whether a modality contributes to the current assessment.
        </p>
      </header>
      <FusionStory />
      <ArchitectureStory />
      <CurrentForecastSplit />
      <EvidenceSupport />
    </main>
  );
}
