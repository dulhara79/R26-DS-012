import ComponentsStory from "../components/research/ComponentsStory";
import ModalityOverview from "../components/research/ModalityOverview";
export default function Components() {
  return (
    <main className="deep-page">
      <header className="page-intro shell">
        <p className="eyebrow">COMPONENTS</p>
        <h1>Four research streams, four evidence contracts.</h1>
        <p>
          The components are research contributions—not interchangeable product
          features.
        </p>
      </header>
      <ModalityOverview />
      <ComponentsStory />
    </main>
  );
}
