import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/layout/Navbar";
import Footer from "./components/layout/Footer";
import ScrollManager from "./components/layout/ScrollManager";
import Home from "./pages/Home";
import Research from "./pages/Research";
import Components from "./pages/Components";
import ComponentDetail from "./pages/ComponentDetail";
import System from "./pages/System";
import Findings from "./pages/Findings";
import Documents from "./pages/Documents";
import Team from "./pages/Team";
import Contact from "./pages/Contact";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <HashRouter>
      <ScrollManager />
      {/* HashRouter owns the URL hash, so the skip link moves focus directly. */}
      <a
        className="skip-link"
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <Navbar />
      <main id="main" tabIndex={-1} style={{ outline: "none" }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/research" element={<Research />} />
          <Route path="/components" element={<Components />} />
          <Route path="/components/:slug" element={<ComponentDetail />} />
          <Route path="/system" element={<System />} />
          <Route path="/findings" element={<Findings />} />
          <Route
            path="/results"
            element={<Navigate to="/findings" replace />}
          />
          <Route path="/documents" element={<Documents />} />
          <Route
            path="/publications"
            element={<Navigate to="/documents" replace />}
          />
          <Route path="/team" element={<Team />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <Footer />
    </HashRouter>
  );
}
