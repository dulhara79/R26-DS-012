import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Brain, Github } from 'lucide-react'

const NAV = [
  { label: 'Overview', path: '/' },
  { label: 'Components', path: '/components' },
  { label: 'Methodology', path: '/methodology' },
  { label: 'Team', path: '/team' },
  { label: 'Publications', path: '/publications' },
  { label: 'Documents', path: '/documents' },
  { label: 'Contact', path: '/contact' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 16)
    window.addEventListener('scroll', h)
    return () => window.removeEventListener('scroll', h)
  }, [])
  useEffect(() => setOpen(false), [location])

  return (
    <motion.nav
      initial={{ y: -72, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
      className={`fixed top-0 inset-x-0 z-50 transition-all duration-300 ${scrolled ? 'nav-glass shadow-sm' : 'bg-transparent'}`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">

        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <motion.div
            whileHover={{ rotate: 10, scale: 1.05 }}
            className="w-8 h-8 rounded-xl flex items-center justify-center shadow-md"
            style={{ background: 'linear-gradient(135deg, #2d55f5, #8a55ff)', boxShadow: '0 4px 12px rgba(45,85,245,0.35)' }}
          >
            <Brain size={16} className="text-white" />
          </motion.div>
          <div>
            <span className="font-display text-sm font-700 text-slate-800 tracking-tight leading-none block">R26-DS-012</span>
            <span className="font-mono text-[9px] tracking-widest block" style={{ color: '#2d55f5' }}>SLIIT · 2026</span>
          </div>
        </Link>

        {/* Desktop links */}
        <div className="hidden lg:flex items-center gap-0.5">
          {NAV.map(item => (
            <Link
              key={item.path}
              to={item.path}
              className={`relative px-3 py-2 rounded-xl text-sm font-medium transition-all duration-150 ${location.pathname === item.path
                  ? 'text-brand-600 font-600'
                  : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100/80'
                }`}
            >
              {location.pathname === item.path && (
                <motion.span
                  layoutId="nav-indicator"
                  className="absolute inset-0 rounded-xl bg-brand-50"
                  style={{ zIndex: -1 }}
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
                />
              )}
              {item.label}
            </Link>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <a
            href="https://github.com/dulhara79/R26-DS-012"
            target="_blank" rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1.5 px-4 py-2 rounded-xl border border-slate-200 text-sm text-slate-600 hover:text-slate-900 hover:border-slate-300 bg-white/80 transition-all font-medium hover:shadow-sm"
          >
            <Github size={14} /> Repository
          </a>
          <button
            onClick={() => setOpen(!open)}
            className="lg:hidden p-2 rounded-xl border border-slate-200 text-slate-500 bg-white/80"
          >
            <AnimatePresence mode="wait" initial={false}>
              {open
                ? <motion.span key="x" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.15 }}><X size={18} /></motion.span>
                : <motion.span key="m" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }} transition={{ duration: 0.15 }}><Menu size={18} /></motion.span>
              }
            </AnimatePresence>
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden bg-white/95 backdrop-blur border-b border-slate-100"
          >
            <div className="px-6 py-4 flex flex-col gap-1">
              {NAV.map((item, i) => (
                <motion.div
                  key={item.path}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.04 }}
                >
                  <Link to={item.path}
                    className={`block px-4 py-3 rounded-xl text-sm font-medium transition-all ${location.pathname === item.path ? 'text-brand-600 bg-brand-50' : 'text-slate-600 hover:bg-slate-50'
                      }`}>
                    {item.label}
                  </Link>
                </motion.div>
              ))}
              <a href="https://github.com/dulhara79/R26-DS-012" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-3 text-sm text-slate-500 hover:bg-slate-50 rounded-xl transition-all">
                <Github size={14} /> Repository
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}