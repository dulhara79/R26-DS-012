import Limitations from "../components/story/Limitations";
export default function About() {
  return (
    <main className="deep-page">
      <header className="page-intro shell">
        <p className="eyebrow">ABOUT</p>
        <h1>Research boundary.</h1>
        <p>
          R26-DS-012 is an academic research and clinical decision-support
          prototype. It is not a diagnostic medical device.
        </p>
      </header>
      <Limitations />
    </main>
  );
}
