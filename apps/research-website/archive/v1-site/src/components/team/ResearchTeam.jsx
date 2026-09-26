import { people, supervisors } from "../../data/people";
const initials = (name) =>
  name
    .split(/\s+/)
    .map((x) => x[0])
    .join("")
    .replace(/\W/g, "")
    .slice(0, 3);
export default function ResearchTeam({ compact = false }) {
  return (
    <div className="team-groups">
      <section>
        <h3>Research Team</h3>
        <div className="team-grid">
          {people.slice(0, compact ? 4 : 99).map((p) => (
            <article key={p.id}>
              <div className="initials" aria-hidden="true">
                {initials(p.name)}
              </div>
              <small>{p.id}</small>
              <h4>{p.name}</h4>
              <p>{p.role}</p>
            </article>
          ))}
        </div>
      </section>
      <section>
        <h3>Academic & External Supervision</h3>
        <div className="supervision-list">
          {supervisors.map((p) => (
            <article key={p.name}>
              <span>{p.role}</span>
              <div>
                <h4>{p.name}</h4>
                <p>{p.affiliation}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
