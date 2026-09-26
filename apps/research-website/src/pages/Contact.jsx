import PageHero from "../components/ui/PageHero";
import ContactSection from "../components/layout/ContactSection";

export default function Contact() {
  return (
    <>
      <PageHero
        eyebrow="Contact"
        title="Get in touch with the research."
        lead="The repository, the institution, the supervisors and the team behind R26-DS-012."
        image="bannerContact"
      />
      <ContactSection id="contact-page" />
    </>
  );
}
