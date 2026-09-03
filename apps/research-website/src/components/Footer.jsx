import { Link } from 'react-router-dom'
import { Brain, Github, ExternalLink, Mail, MapPin, BookOpen, Activity, Network, Cpu, ArrowUpRight, Phone, FileText } from 'lucide-react'

const NAV = [
  { label: 'Overview', path: '/' },
  { label: 'Components', path: '/components' },
  { label: 'Methodology', path: '/methodology' },
  { label: 'Team', path: '/team' },
  { label: 'Publications', path: '/publications' },
  { label: 'Documents', path: '/documents' },
  { label: 'Contact', path: '/contact' },
]

const COMPONENTS = [
  { id: 'M1', label: 'Wearable Biosensors', icon: Activity, color: '#05bdad' },
  { id: 'M2', label: 'Behavioral Graphs', icon: Network, color: '#2d55f5' },
  { id: 'M3', label: 'Intervention Engine', icon: Cpu, color: '#8a55ff' },
  { id: 'M4', label: 'Clinical NLP (TC-WPN)', icon: Brain, color: '#e2142e' },
]

export default function Footer() {
  return (
    <footer style={{
      background: 'linear-gradient(160deg, #0f172a 0%, #1e1b4b 45%, #0f2027 100%)',
      position: 'relative', overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
        backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.04) 1px, transparent 0)',
        backgroundSize: '36px 36px', pointerEvents: 'none',
      }} />
      <div style={{ position: 'absolute', top: '-120px', right: '-80px', width: '400px', height: '400px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(45,85,245,0.12) 0%, transparent 70%)', pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', bottom: '-80px', left: '10%', width: '300px', height: '300px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(5,189,173,0.10) 0%, transparent 70%)', pointerEvents: 'none' }} />

      {/* Accent bar */}
      <div style={{ height: '3px', background: 'linear-gradient(90deg, #2d55f5, #8a55ff, #05bdad, #2d55f5)', backgroundSize: '200% 100%' }} />

      <div className="max-w-7xl mx-auto px-6 pt-16 pb-8 relative">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10 mb-14">

          {/* Brand */}
          <div className="md:col-span-4">
            <div className="flex items-center gap-3 mb-5">
              <div style={{ width: 40, height: 40, borderRadius: 12, background: 'linear-gradient(135deg, #2d55f5, #8a55ff)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 16px rgba(45,85,245,0.4)' }}>
                <Brain size={18} color="white" />
              </div>
              <div>
                <div className="font-display font-700 text-white text-base leading-tight">R26-DS-012</div>
                <div className="font-mono text-[10px] tracking-widest" style={{ color: '#6ee7f7' }}>SLIIT · 2026</div>
              </div>
            </div>
            <p className="text-sm leading-relaxed mb-6" style={{ color: 'rgba(226,232,240,0.55)', maxWidth: 300 }}>
              A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders.
            </p>
            <div className="flex flex-col gap-2 mb-6">
              {COMPONENTS.map(c => (
                <div key={c.id} className="flex items-center gap-2.5">
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: c.color, flexShrink: 0, boxShadow: `0 0 6px ${c.color}` }} />
                  <span className="font-mono text-[11px]" style={{ color: 'rgba(226,232,240,0.45)' }}>{c.id}</span>
                  <span className="text-xs" style={{ color: 'rgba(226,232,240,0.55)' }}>{c.label}</span>
                </div>
              ))}
            </div>
            <div className="flex gap-3">
              {[
                { href: 'https://github.com/dulhara79/R26-DS-012', icon: Github, label: 'GitHub' },
                { href: 'https://www.sliit.lk', icon: ExternalLink, label: 'SLIIT' },
              ].map(l => (
                <a key={l.label} href={l.href} target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-1.5 text-xs font-medium transition-all"
                  style={{ color: 'rgba(255,255,255,0.45)', textDecoration: 'none' }}
                  onMouseEnter={e => e.currentTarget.style.color = 'rgba(255,255,255,0.9)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'rgba(255,255,255,0.45)'}
                >
                  <l.icon size={13} /> {l.label}
                </a>
              ))}
            </div>
          </div>

          {/* Navigation */}
          <div className="md:col-span-2 md:col-start-6">
            <h4 className="font-display text-xs font-700 tracking-widest uppercase mb-5" style={{ color: 'rgba(255,255,255,0.35)', letterSpacing: '0.1em' }}>Navigation</h4>
            <ul className="space-y-3">
              {NAV.map(item => (
                <li key={item.path}>
                  <Link to={item.path} className="group flex items-center gap-1.5 text-sm transition-all"
                    style={{ color: 'rgba(226,232,240,0.5)', textDecoration: 'none' }}
                    onMouseEnter={e => e.currentTarget.style.color = 'rgba(255,255,255,0.95)'}
                    onMouseLeave={e => e.currentTarget.style.color = 'rgba(226,232,240,0.5)'}
                  >
                    <ArrowUpRight size={12} style={{ opacity: 0, transition: 'opacity 0.2s' }} className="group-hover:opacity-100" />
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Research Info */}
          <div className="md:col-span-3">
            <h4 className="font-display text-xs font-700 tracking-widest uppercase mb-5" style={{ color: 'rgba(255,255,255,0.35)', letterSpacing: '0.1em' }}>Research</h4>
            <div className="space-y-4">
              {[
                { icon: BookOpen, lines: ['B.Sc. (Hons) Information Technology', 'Specialized in Data Science'] },
                { icon: MapPin, lines: ['Dept. of Computer Science', 'Sri Lanka Institute of IT'] },
                { icon: Mail, lines: ['Supervised by', 'Prof. Samantha Thelijjagoda'] },
                { icon: Phone, lines: ['Contact the team', 'it22130648@my.sliit.lk'] },
              ].map((item, i) => (
                <div key={i} className="flex items-start gap-2.5">
                  <item.icon size={13} style={{ color: 'rgba(255,255,255,0.25)', marginTop: 2, flexShrink: 0 }} />
                  <div>
                    {item.lines.map((l, j) => (
                      <div key={j} className="text-xs leading-relaxed" style={{ color: j === 0 ? 'rgba(226,232,240,0.55)' : 'rgba(226,232,240,0.35)' }}>{l}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div style={{ height: 1, background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.08) 30%, rgba(255,255,255,0.08) 70%, transparent)' }} />

        <div className="flex flex-col sm:flex-row justify-between items-center gap-3 pt-7">
          <p className="text-xs" style={{ color: 'rgba(255,255,255,0.25)' }}>
            © 2026 R26-DS-012 Research Team · SLIIT Department of Computer Science · All rights reserved
          </p>
          <div className="flex items-center gap-2">
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
            <span className="font-mono text-[11px]" style={{ color: 'rgba(255,255,255,0.3)' }}>Research ongoing · March 2026</span>
          </div>
        </div>
      </div>
    </footer>
  )
}