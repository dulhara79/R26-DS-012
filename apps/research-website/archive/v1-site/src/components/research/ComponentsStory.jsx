import { components } from "../../data/components";
import ComponentFeature from "./ComponentFeature";
import TextReveal from "../motion/TextReveal";
export default function ComponentsStory() {
  return (
    <section
      className="research-section components-story"
      id="components"
      data-section="components"
    >
      <div className="shell">
        <TextReveal as="p" preset="eyebrow" className="eyebrow">
          THE FOUR RESEARCH STREAMS
        </TextReveal>
        <div className="section-title">
          <TextReveal as="h2" preset="heading" delay={0.04}>
            Four components, each with its own research question and evidence
            contract.
          </TextReveal>
          <TextReveal as="p" preset="body" delay={0.1}>
            The streams are introduced only after the research problem and
            conceptual approach are clear.
          </TextReveal>
        </div>
        {components.map((component, index) => (
          <ComponentFeature
            component={component}
            index={index}
            key={component.id}
          />
        ))}
      </div>
    </section>
  );
}
