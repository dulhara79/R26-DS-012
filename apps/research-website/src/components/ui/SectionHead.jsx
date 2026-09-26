import Reveal from "./Reveal";

export default function SectionHead({ eyebrow, title, lead, id }) {
  return (
    <Reveal
      className={lead ? "section-head" : "section-head section-head--single"}
    >
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h2 className="display" id={id}>
          {title}
        </h2>
      </div>
      {lead && <p className="lead">{lead}</p>}
    </Reveal>
  );
}
