import AnxietyContext from "../components/story/AnxietyContext";
import ObservationTimeline from "../components/story/ObservationTimeline";
import SnapshotProblem from "../components/story/SnapshotProblem";
import LiteratureLandscape from "../components/story/LiteratureLandscape";
import ResearchGap from "../components/story/ResearchGap";
import ResearchThesis from "../components/story/ResearchThesis";
import MethodologyStory from "../components/story/MethodologyStory";
export default function Research() {
  return (
    <main className="deep-page">
      <header className="page-intro shell">
        <p className="eyebrow">RESEARCH</p>
        <h1>Why this research exists.</h1>
        <p>
          Context, observation limits, literature framing, research gap,
          question and methodology—before the implementation details.
        </p>
      </header>
      <AnxietyContext />
      <ObservationTimeline />
      <SnapshotProblem />
      <LiteratureLandscape />
      <ResearchGap />
      <ResearchThesis />
      <MethodologyStory />
    </main>
  );
}
