import CinematicHero from "../components/hero/CinematicHero";
import ImmersiveIntroduction from "../components/story/ImmersiveIntroduction";
import ModalityOverview from "../components/research/ModalityOverview";
import ComponentsStory from "../components/research/ComponentsStory";
import FusionStory from "../components/research/FusionStory";
import ArchitectureStory from "../components/research/ArchitectureStory";
import CurrentForecastSplit from "../components/research/CurrentForecastSplit";
import FindingsPreview from "../components/evidence/FindingsPreview";
import ResearchInterpretation from "../components/story/ResearchInterpretation";
import EvidenceSupport from "../components/research/EvidenceSupport";
import MethodologyStory from "../components/story/MethodologyStory";
import DocumentCollection from "../components/documents/DocumentCollection";
import ResearchTimeline from "../components/story/ResearchTimeline";
import ResearchTeam from "../components/team/ResearchTeam";
import Limitations from "../components/story/Limitations";
import MotionReveal from "../components/motion/MotionReveal";

export default function Home() {
  return (
    <main>
      <CinematicHero />
      <ImmersiveIntroduction />
      <ModalityOverview />
      <ComponentsStory />
      <FusionStory />
      <ArchitectureStory />
      <CurrentForecastSplit />
      <FindingsPreview />
      <ResearchInterpretation />
      <EvidenceSupport />
      <MethodologyStory />
      <section className="story-section documents-preview" id="documents">
        <MotionReveal className="shell">
          <p className="eyebrow">RESEARCH DOCUMENTS</p>
          <h2>Public research outputs, organized by source and status.</h2>
          <DocumentCollection compact />
        </MotionReveal>
      </section>
      <ResearchTimeline />
      <section className="story-section team-preview" id="team">
        <MotionReveal className="shell">
          <p className="eyebrow">TEAM & SUPERVISION</p>
          <ResearchTeam compact />
        </MotionReveal>
      </section>
      <Limitations />
    </main>
  );
}
