import ResearchTeam from "../components/team/ResearchTeam";
export default function Team() {
  return (
    <main className="internal shell">
      <header className="page-intro">
        <p className="eyebrow">TEAM</p>
        <h1>Research team & supervision.</h1>
        <p>
          Research roles follow the current component responsibilities
          documented in R26-DS-012.
        </p>
      </header>
      <ResearchTeam />
    </main>
  );
}
