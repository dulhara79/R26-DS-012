import {
  HashRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Research from "./pages/Research";
import Components from "./pages/Components";
import System from "./pages/System";
import Results from "./pages/Results";
import Methodology from "./pages/Methodology";
import Team from "./pages/Team";
import Publications from "./pages/Publications";
import Documents from "./pages/Documents";
import Applications from "./pages/Applications";
import About from "./pages/About";
export default function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/research" element={<Research />} />
        <Route path="/components" element={<Components />} />
        <Route path="/system" element={<System />} />
        <Route path="/findings" element={<Results />} />
        <Route path="/results" element={<Navigate to="/findings" replace />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/applications" element={<Applications />} />
        <Route path="/publications" element={<Publications />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/team" element={<Team />} />
        <Route path="/about" element={<About />} />
      </Routes>
      <Footer />
    </Router>
  );
}
