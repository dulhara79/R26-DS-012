import { Link } from "react-router-dom";
import PageHero from "../components/ui/PageHero";

export default function NotFound() {
  return (
    <>
      <PageHero
        eyebrow="Not found"
        title="This page is missing evidence."
        lead="The address does not match any page on this site."
      />
      <section className="section section--tight">
        <div className="shell">
          <Link className="button button--ink" to="/">
            Back to overview
          </Link>
        </div>
      </section>
    </>
  );
}
