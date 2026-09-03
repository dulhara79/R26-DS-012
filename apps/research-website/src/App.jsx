import { HashRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Home from './pages/Home'
import Components from './pages/Components'
import Methodology from './pages/Methodology'
import Team from './pages/Team'
import Publications from './pages/Publications'
import Contact from './pages/Contact'
import Documents from './pages/Documents'

export default function App() {
  return (
    <Router>
      <div className="min-h-screen page-texture text-slate-800">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/components" element={<Components />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/team" element={<Team />} />
          <Route path="/publications" element={<Publications />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/documents" element={<Documents />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  )
}