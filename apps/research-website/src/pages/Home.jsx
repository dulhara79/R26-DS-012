import { useRef, useState, useEffect } from 'react'
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  Activity, Network, Cpu, Brain, ArrowRight, ChevronDown,
  Zap, Globe, Users, ShieldCheck, HeartPulse, Smartphone,
  FileText, BarChart3, ExternalLink, Mail, Download, BookOpen, Sparkles
} from 'lucide-react'

const HERO_IMG = 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=1400&q=85&auto=format&fit=crop'

const SECTION_IMGS = {
  wearable: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=900&q=80&auto=format&fit=crop',
  behavioral: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=900&q=80&auto=format&fit=crop',
  clinical: 'https://images.unsplash.com/photo-1576671081837-49000212a370?w=900&q=80&auto=format&fit=crop',
  srilanka: 'https://images.unsplash.com/photo-1605810230434-7631ac76ec81?w=900&q=80&auto=format&fit=crop',
  brain: 'https://images.unsplash.com/photo-1692607431225-5f4564c8f132?q=80&w=1332&auto=format&fit=crop',
  youth: 'https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=900&q=80&auto=format&fit=crop',
}

const COMPS = [
  { id: 'M1', code: 'IT22107596', name: 'Sendanayake H.D.', title: 'Wearable Biosensor Forecasting', sub: 'Self-Supervised Anomaly Detection', icon: Activity, img: SECTION_IMGS.wearable, pill: 'pill-teal', accent: 'text-teal-600', accentBg: 'bg-teal-50', accentBorder: 'border-teal-200', color: '#05bdad', desc: 'ESP32 chest-strap with ECG, respiration, temperature & IMU. LSTM autoencoders learn individualized physiological baselines for 5–10 min ahead acute episode forecasting.', tech: ['LSTM-AE', 'ESP32-C3', 'HRV Analysis', 'Edge ML'] },
  { id: 'M2', code: 'IT22171542', name: 'Layathma B.M.A.S.', title: 'Behavioral Graph Phenotyping', sub: 'GATv2 Temporal Behavioral Graphs', icon: Network, img: SECTION_IMGS.behavioral, pill: 'pill-blue', accent: 'text-brand-600', accentBg: 'bg-brand-50', accentBorder: 'border-brand-200', color: '#2d55f5', desc: 'Passive smartphone sensing structured as temporal graphs — 168 nodes, 4 circadian time windows per day. GATv2 produces personalized vulnerability scores and high-risk window IDs.', tech: ['GATv2', 'PyTorch Geometric', 'StudentLife Dataset', 'Android'] },
  { id: 'M3', code: 'IT22093950', name: 'Seneviratne K.A.U.A.', title: 'Adaptive Intervention Engine', sub: 'KNN Case-Based Reasoning + Continuous Learning', icon: Cpu, img: SECTION_IMGS.clinical, pill: 'pill-violet', accent: 'text-violet-600', accentBg: 'bg-violet-50', accentBorder: 'border-violet-200', color: '#8a55ff', desc: 'KNN engine pre-trained on NHANES 2017-2020 data maps multimodal risk to Tier A/B interventions. Composite reward function drives continuous learning from observed outcomes.', tech: ['KNN BallTree', 'NHANES', 'Flutter', 'FastAPI'] },
  { id: 'M4', code: 'IT22130648', name: 'Kaushalya I.G.D.', title: 'Clinical NLP — TC-WPN', sub: 'Few-Shot Meta-Learning for Clinical Notes', icon: Brain, img: SECTION_IMGS.brain, pill: 'pill-rose', accent: 'text-rose-600', accentBg: 'bg-rose-50', accentBorder: 'border-rose-200', color: '#e2142e', desc: 'Temporal-Confidence Weighted Prototypical Networks achieve ≥75% F1 with only 10–20 labeled clinical notes — a 50–100× data reduction vs traditional supervised approaches.', tech: ['ClinicalBERT', 'Prototypical Nets', 'MIMIC-IV', 'FastAPI'] },
]

const STATS = [
  { v: '301M', label: 'People affected globally', icon: Globe, color: 'text-brand-500', bg: 'bg-brand-50' },
  { v: '75%', label: 'Onset before age 24', icon: Users, color: 'text-violet-600', bg: 'bg-violet-50' },
  { v: '10–20', label: 'Labeled examples needed', icon: Brain, color: 'text-teal-600', bg: 'bg-teal-50' },
  { v: '5–10min', label: 'Early warning window', icon: Zap, color: 'text-rose-600', bg: 'bg-rose-50' },
]

function EcgLine() {
  return (
    <svg viewBox="0 0 320 60" className="w-full h-12 opacity-60" fill="none">
      <path d="M0 30 L30 30 L38 30 L42 10 L46 50 L50 30 L58 30 L62 28 L66 18 L70 48 L74 30 L80 30 L110 30 L118 30 L122 10 L126 50 L130 30 L138 30 L142 28 L146 18 L150 48 L154 30 L160 30 L190 30 L198 30 L202 10 L206 50 L210 30 L218 30 L222 28 L226 18 L230 48 L234 30 L240 30 L270 30 L278 30 L282 10 L286 50 L290 30 L298 30 L302 28 L306 18 L310 48 L314 30 L320 30"
        stroke="#2d55f5" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
        className="wave-path" style={{ strokeDasharray: '10 4' }}
      />
    </svg>
  )
}

function FloatBadge({ icon: Icon, label, value, color, cls }) {
  return (
    <div className={`${cls} absolute bg-white rounded-2xl shadow-xl border border-slate-100 px-4 py-3 flex items-center gap-3`}>
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center ${color}`}>
        <Icon size={17} className="text-white" />
      </div>
      <div>
        <div className="font-display text-base font-700 text-slate-800 leading-tight">{value}</div>
        <div className="text-xs text-slate-400 leading-tight">{label}</div>
      </div>
    </div>
  )
}

function BrainNet() {
  const nodes = [[50, 30], [160, 20], [270, 35], [80, 120], [200, 110], [320, 90], [130, 200], [260, 185], [50, 200]]
  const edges = [[0, 1], [1, 2], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5], [3, 6], [4, 7], [5, 8], [6, 7], [7, 8]]
  return (
    <svg viewBox="0 0 380 240" className="w-full h-full" fill="none">
      {edges.map(([a, b], i) => (
        <line key={i} x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]}
          stroke="#2d55f5" strokeWidth="1" strokeOpacity="0.18" strokeDasharray="4 6"
          style={{ animation: `dash-run ${2 + i * 0.15}s linear infinite` }}
        />
      ))}
      {nodes.map(([x, y], i) => (
        <g key={i}>
          <circle cx={x} cy={y} r="16" fill="white" stroke="#e2e8f0" strokeWidth="1.5" />
          <circle cx={x} cy={y} r="7" fill={i < 3 ? '#2d55f5' : i < 6 ? '#05bdad' : '#8a55ff'} opacity="0.85" />
        </g>
      ))}
    </svg>
  )
}

// Animated counter
function AnimCounter({ target, suffix = '' }) {
  const [count, setCount] = useState(0)
  const ref = useRef(null)
  const [started, setStarted] = useState(false)

  useEffect(() => {
    if (!ref.current) return
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started) setStarted(true)
    }, { threshold: 0.5 })
    obs.observe(ref.current)
    return () => obs.disconnect()
  }, [started])

  useEffect(() => {
    if (!started) return
    const num = parseInt(target.toString().replace(/\D/g, '')) || 0
    if (num === 0) return
    let current = 0
    const step = Math.ceil(num / 40)
    const timer = setInterval(() => {
      current = Math.min(current + step, num)
      setCount(current)
      if (current >= num) clearInterval(timer)
    }, 35)
    return () => clearInterval(timer)
  }, [started, target])

  return <span ref={ref}>{started ? count.toLocaleString() : 0}{suffix}</span>
}

export default function Home() {
  const heroRef = useRef(null)
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] })
  const heroImgY = useTransform(scrollYProgress, [0, 1], ['0%', '18%'])
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0])

  return (
    <div className="relative overflow-x-hidden">

      {/* ══ HERO ══════════════════════════════════ */}
      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">

        <motion.div style={{ y: heroImgY }} className="absolute inset-0 z-0">
          <img src={HERO_IMG} alt="Brain scan" className="w-full h-full object-cover" style={{ filter: 'brightness(0.3) saturate(0.6)' }} />
        </motion.div>
        <div className="absolute inset-0 z-[1]" style={{ background: 'linear-gradient(180deg, rgba(15,23,42,0.2) 0%, rgba(240,245,255,0.6) 70%, #f8fafc 100%)' }} />
        <div className="absolute inset-0 z-[1]" style={{ background: 'radial-gradient(ellipse 60% 70% at 50% 40%, rgba(45,85,245,0.1) 0%, transparent 70%)' }} />

        {/* Animated particles */}
        {[...Array(6)].map((_, i) => (
          <motion.div key={i}
            className="absolute rounded-full opacity-20"
            style={{
              width: [60, 40, 80, 50, 30, 70][i], height: [60, 40, 80, 50, 30, 70][i],
              background: ['#2d55f5', '#05bdad', '#8a55ff', '#2d55f5', '#05bdad', '#8a55ff'][i],
              left: `${[10, 80, 60, 25, 70, 45][i]}%`,
              top: `${[20, 15, 60, 70, 45, 30][i]}%`,
              filter: 'blur(20px)',
            }}
            animate={{ y: [0, -30, 0], x: [0, 15, 0], scale: [1, 1.2, 1] }}
            transition={{ duration: [7, 5, 9, 6, 8, 7][i], repeat: Infinity, ease: 'easeInOut', delay: i * 0.8 }}
          />
        ))}

        <motion.div style={{ opacity: heroOpacity }} className="relative z-10 max-w-5xl mx-auto px-6 text-center">

          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, duration: 0.6 }}
            className="flex flex-wrap items-center justify-center gap-2 mb-8"
          >
            <span className="pill pill-blue">Research Project</span>
            <span className="pill pill-teal">R26-DS-012</span>
            <span className="pill pill-slate">SLIIT · 2026</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25, duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            className="font-display text-5xl md:text-[72px] font-800 leading-[0.92] tracking-tight mb-7 text-white"
          >
            Multimodal<br />
            <motion.span
              style={{ background: 'linear-gradient(135deg,#7c9bff,#5eecdf)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}
              animate={{ backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'] }}
              transition={{ duration: 4, repeat: Infinity }}
            >
              Digital Biomarker
            </motion.span><br />
            Framework
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.42, duration: 0.65 }}
            className="text-slate-200 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed mb-10"
          >
            Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders — combining wearable biosensors, behavioral graphs, clinical NLP, and adaptive interventions.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="flex flex-wrap items-center justify-center gap-4"
          >
            <Link to="/components" className="btn-primary flex items-center gap-2 group">
              Explore Components <ArrowRight size={15} className="group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link to="/documents" className="flex items-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 backdrop-blur text-white border border-white/20 rounded-xl text-sm font-600 transition-all">
              <BookOpen size={15} /> Research Docs
            </Link>
            <Link to="/contact" className="flex items-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 backdrop-blur text-white border border-white/20 rounded-xl text-sm font-600 transition-all">
              <Mail size={15} /> Contact Us
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            className="mt-14 flex flex-col items-center gap-1 text-white/50"
          >
            <span className="font-mono text-[10px] tracking-[0.2em] uppercase">Scroll to explore</span>
            <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 1.5, repeat: Infinity }}>
              <ChevronDown size={16} />
            </motion.div>
          </motion.div>
        </motion.div>
      </section>

      {/* ══ STATS STRIP ═══════════════════════════ */}
      <section className="relative bg-white border-y border-slate-100 py-12 overflow-hidden">
        <div className="max-w-5xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {STATS.map((s, i) => (
              <motion.div key={i}
                initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.09 }}
                className="text-center group"
              >
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  className={`w-12 h-12 rounded-2xl ${s.bg} flex items-center justify-center mx-auto mb-3 transition-all`}
                >
                  <s.icon size={22} className={s.color} />
                </motion.div>
                <div className={`font-display text-3xl font-800 ${s.color} mb-1`}>{s.v}</div>
                <div className="text-sm text-slate-400 leading-tight">{s.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ PROBLEM STATEMENT ═════════════════════ */}
      <section className="py-28 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -40 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.75 }}
              className="relative"
            >
              <div className="relative rounded-3xl overflow-hidden aspect-[4/3] shadow-2xl">
                <img src={SECTION_IMGS.youth} alt="Young adults" className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-gradient-to-t from-brand-900/50 via-transparent to-transparent" />
                <div className="absolute bottom-5 left-5 right-5"><EcgLine /></div>
              </div>
              <FloatBadge icon={HeartPulse} label="Treatment gap" value="<10%" color="bg-rose-500" cls="float-badge -bottom-5 -right-4 shadow-xl" />
              <FloatBadge icon={Users} label="Psychiatrists / 100k" value="0.9" color="bg-brand-500" cls="float-badge-2 -top-5 -left-4 shadow-xl" />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 40 }} whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }} transition={{ duration: 0.75, delay: 0.1 }}
            >
              <span className="pill pill-rose mb-5 inline-block">The Problem</span>
              <h2 className="font-display text-4xl md:text-5xl font-800 text-slate-900 leading-tight mb-6">
                A Mental Health Crisis <span className="grad-blue">Underserved</span> by Technology
              </h2>
              <p className="text-slate-500 text-lg leading-relaxed mb-6">
                Anxiety disorders affect 301 million people globally. In Sri Lanka, fewer than 60 psychiatrists serve 22 million people. Traditional clinical assessment captures only episodic snapshots — missing critical symptom fluctuations between visits.
              </p>
              <p className="text-slate-500 leading-relaxed mb-8">
                Existing AI systems require 1,000–5,000 labeled examples — an insurmountable barrier for hospitals where experts can realistically annotate only 10–20 cases. Our framework closes this gap.
              </p>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: 'Psychiatrists per 100k', value: '0.9', sub: 'vs WHO minimum' },
                  { label: 'Young adults affected', value: '14.4%', sub: 'clinical anxiety (SL)' },
                  { label: 'Access specialist care', value: '<10%', sub: 'of those affected' },
                  { label: 'Smartphone penetration', value: '70%+', sub: 'young adults (SL)' },
                ].map((it, i) => (
                  <motion.div key={i}
                    whileHover={{ scale: 1.03 }}
                    className="bg-slate-50 rounded-2xl p-4 border border-slate-100 transition-all hover:shadow-md hover:bg-white cursor-default"
                  >
                    <div className="font-display text-2xl font-800 text-brand-600 mb-0.5">{it.value}</div>
                    <div className="text-xs text-slate-500 font-medium">{it.label}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{it.sub}</div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ══ FRAMEWORK OVERVIEW ════════════════════ */}
      <section className="section-alt py-28 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }} className="text-center mb-16"
          >
            <span className="pill pill-blue mb-5 inline-block">Framework Architecture</span>
            <h2 className="font-display text-4xl md:text-5xl font-800 text-slate-900 mb-5">
              Four Integrated <span className="grad-violet">Subsystems</span>
            </h2>
            <p className="text-slate-500 text-lg max-w-xl mx-auto">
              A late-fusion architecture combining four orthogonal anxiety modalities into a single composite risk score.
            </p>
          </motion.div>

          {/* Visual flow */}
          <motion.div initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: '-60px' }} transition={{ delay: 0.1 }}
            className="card-flat mb-14 p-6 md:p-8"
          >
            <div className="flex flex-col md:flex-row items-center gap-3 justify-center">
              {COMPS.map((c, i) => (
                <div key={c.id} className="flex items-center gap-3">
                  <motion.div
                    whileHover={{ scale: 1.08 }}
                    className={`${c.accentBg} ${c.accentBorder} border rounded-2xl px-5 py-3 text-center min-w-[120px] cursor-default transition-all hover:shadow-md`}
                  >
                    <c.icon size={20} className={`${c.accent} mx-auto mb-1`} />
                    <div className={`font-mono text-xs font-600 ${c.accent}`}>{c.id}</div>
                    <div className="text-xs text-slate-500 mt-0.5 leading-tight">{c.title.split(' ').slice(0, 2).join(' ')}</div>
                  </motion.div>
                  {i < COMPS.length - 1 && (
                    <motion.div
                      animate={{ x: [0, 4, 0] }} transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
                    >
                      <ArrowRight size={16} className="text-slate-300 flex-shrink-0 hidden md:block" />
                    </motion.div>
                  )}
                </div>
              ))}
              <motion.div animate={{ x: [0, 4, 0] }} transition={{ duration: 1.5, repeat: Infinity }}>
                <ArrowRight size={16} className="text-slate-300 flex-shrink-0 hidden md:block" />
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.08 }}
                className="bg-gradient-to-br from-brand-500 to-violet-600 rounded-2xl px-5 py-3 text-center min-w-[120px] shadow-lg shadow-brand-500/20 cursor-default"
              >
                <BarChart3 size={20} className="text-white mx-auto mb-1" />
                <div className="font-mono text-xs font-600 text-white">FUSION</div>
                <div className="text-xs text-blue-100 mt-0.5 leading-tight">Risk Score</div>
              </motion.div>
            </div>
            <p className="text-center text-xs text-slate-400 font-mono mt-4">
              Weights: w₁=0.25 · w₂=0.20 · w₃=0.15 · w₄=0.40 (clinical notes highest authority)
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            {COMPS.map((c, i) => (
              <motion.div key={c.id}
                initial={{ opacity: 0, y: 40 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-60px' }} transition={{ delay: i * 0.12 }}
                className="card group overflow-hidden"
              >
                <div className="relative h-44 overflow-hidden rounded-t-[20px]">
                  <img src={c.img} alt={c.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/10 to-transparent" />
                  <div className="absolute top-4 left-4"><span className={`pill ${c.pill}`}>{c.id}</span></div>
                  <div className="absolute bottom-4 left-4 right-4">
                    <div className="font-mono text-xs text-white/70 mb-0.5">{c.code} · {c.name}</div>
                    <h3 className="font-display text-lg font-700 text-white leading-snug">{c.title}</h3>
                  </div>
                </div>
                <div className="p-6">
                  <p className={`text-sm font-600 ${c.accent} mb-3`}>{c.sub}</p>
                  <p className="text-sm text-slate-500 leading-relaxed mb-4">{c.desc}</p>
                  <div className="flex flex-wrap gap-2">
                    {c.tech.map(t => <span key={t} className={`pill ${c.pill} text-[10px]`}>{t}</span>)}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="text-center mt-10">
            <Link to="/components" className="btn-primary inline-flex items-center gap-2 group">
              Deep-dive into each component
              <ArrowRight size={15} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </div>
      </section>

      {/* ══ APPROACH ══════════════════════════════ */}
      <section className="py-28 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div initial={{ opacity: 0, x: -32 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}>
              <span className="pill pill-violet mb-5 inline-block">Our Approach</span>
              <h2 className="font-display text-4xl md:text-5xl font-800 text-slate-900 leading-tight mb-6">
                From Reactive to <span className="grad-teal">Proactive</span> Mental Health Care
              </h2>
              <p className="text-slate-500 text-lg leading-relaxed mb-6">
                Instead of waiting for a clinical episode, our system continuously monitors four orthogonal data streams — physiological, behavioral, contextual, and linguistic — fusing them into a real-time composite risk score.
              </p>
              <div className="space-y-3">
                {[
                  { icon: HeartPulse, title: 'Physiological', desc: 'Real-time ECG, HRV, respiration anomaly detection', color: 'text-rose-500', bg: 'bg-rose-50' },
                  { icon: Smartphone, title: 'Behavioral', desc: 'Passive smartphone graph phenotyping 24/7', color: 'text-brand-500', bg: 'bg-brand-50' },
                  { icon: FileText, title: 'Clinical Language', desc: 'Few-shot NLP on sparse clinical notes', color: 'text-violet-600', bg: 'bg-violet-50' },
                  { icon: ShieldCheck, title: 'Adaptive Intervention', desc: 'Personalized evidence-based treatment plans', color: 'text-teal-600', bg: 'bg-teal-50' },
                ].map((it, i) => (
                  <motion.div key={i}
                    initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                    whileHover={{ x: 6 }}
                    className="flex items-start gap-4 p-4 rounded-2xl border border-slate-100 bg-slate-50 hover:bg-white hover:shadow-md transition-all cursor-default"
                  >
                    <div className={`w-10 h-10 rounded-xl ${it.bg} flex items-center justify-center flex-shrink-0`}>
                      <it.icon size={18} className={it.color} />
                    </div>
                    <div>
                      <div className="font-display text-sm font-700 text-slate-800 mb-0.5">{it.title}</div>
                      <div className="text-sm text-slate-500">{it.desc}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, x: 32 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: 0.15 }} className="relative">
              <div className="relative rounded-3xl overflow-hidden aspect-square bg-gradient-to-br from-brand-50 via-violet-50 to-teal-50 border border-slate-100 shadow-xl p-8">
                <BrainNet />
                {[
                  { label: 'Wearable', sub: 'M1', color: 'bg-teal-500', pos: 'top-5 left-5' },
                  { label: 'Behavioral', sub: 'M2', color: 'bg-brand-500', pos: 'top-5 right-5' },
                  { label: 'Intervention', sub: 'M3', color: 'bg-violet-500', pos: 'bottom-5 left-5' },
                  { label: 'Clinical NLP', sub: 'M4', color: 'bg-rose-500', pos: 'bottom-5 right-5' },
                ].map(it => (
                  <div key={it.label} className={`absolute ${it.pos} flex items-center gap-2`}>
                    <div className={`w-2.5 h-2.5 rounded-full ${it.color}`} />
                    <div>
                      <div className="font-mono text-[9px] text-slate-400 leading-none">{it.sub}</div>
                      <div className="font-display text-xs font-600 text-slate-600">{it.label}</div>
                    </div>
                  </div>
                ))}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <motion.div
                    animate={{ scale: [1, 1.08, 1] }} transition={{ duration: 3, repeat: Infinity }}
                    className="w-20 h-20 rounded-full flex items-center justify-center"
                    style={{ background: 'linear-gradient(135deg, #2d55f5, #8a55ff)', boxShadow: '0 8px 32px rgba(45,85,245,0.4)' }}
                  >
                    <div className="text-center">
                      <div className="font-display text-xs font-800 text-white">FUSION</div>
                      <div className="font-mono text-[8px] text-blue-200">risk score</div>
                    </div>
                  </motion.div>
                </div>
              </div>
              <div className="mt-5 rounded-2xl overflow-hidden h-36 shadow-xl relative">
                <img src={SECTION_IMGS.srilanka} alt="Sri Lanka" className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-gradient-to-t from-brand-900/60 to-transparent flex items-end p-3">
                  <p className="text-white text-xs font-medium">Deployed & validated at National Hospital of Sri Lanka</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ══ SDG BANNER ════════════════════════════ */}
      <section className="section-teal py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="bg-white rounded-3xl shadow-xl p-10 md:p-14"
          >
            <div className="grid md:grid-cols-2 gap-10 items-center">
              <div>
                <span className="pill pill-green mb-5 inline-block">Social Impact</span>
                <h2 className="font-display text-3xl md:text-4xl font-800 text-slate-900 mb-5">
                  Aligned with <span className="grad-teal">UN SDG 3</span>
                </h2>
                <p className="text-slate-500 leading-relaxed mb-6">
                  Good Health and Well-Being. Our framework enables AI-driven anxiety detection in 75% of global healthcare settings that currently lack adequate psychiatric resources.
                </p>
                <div className="space-y-3">
                  {[
                    { t: '75% of under-resourced hospitals globally addressable', c: 'text-teal-600' },
                    { t: '95% deployment cost reduction vs traditional supervised AI', c: 'text-brand-600' },
                    { t: '$500M projected addressable market in rural healthcare', c: 'text-violet-600' },
                  ].map((it, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm">
                      <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${it.c.replace('text-', 'bg-')}`} />
                      <span className="text-slate-600">{it.t}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { goal: 'SDG 3', title: 'Good Health', desc: 'Early detection before clinical escalation', color: 'bg-green-50 border-green-200 text-green-700' },
                  { goal: 'SDG 10', title: 'Reduced Inequalities', desc: 'Scalable across LMICs without clinical infrastructure', color: 'bg-brand-50 border-brand-200 text-brand-700' },
                  { goal: 'SDG 9', title: 'Innovation', desc: 'Novel graph + few-shot AI for clinical NLP', color: 'bg-teal-50 border-teal-200 text-teal-700' },
                  { goal: 'SDG 17', title: 'Partnerships', desc: 'NHSL, SLIIT, WHO & UNICEF initiatives', color: 'bg-violet-50 border-violet-200 text-violet-700' },
                ].map((it, i) => (
                  <motion.div key={i}
                    initial={{ opacity: 0, scale: 0.92 }} whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }} transition={{ delay: i * 0.08 }}
                    whileHover={{ scale: 1.04 }}
                    className={`rounded-2xl p-4 border ${it.color} cursor-default transition-all`}
                  >
                    <div className="font-mono text-xs opacity-60 mb-1">{it.goal}</div>
                    <div className="font-display text-sm font-700 mb-1.5">{it.title}</div>
                    <div className="text-xs opacity-70 leading-relaxed">{it.desc}</div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ══ QUICK ACCESS CARDS (Documents + Contact) ═ */}
      <section className="py-24 px-6 bg-white">
        <div className="max-w-5xl mx-auto">
          <motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
            className="text-center mb-12">
            <span className="pill pill-slate mb-4 inline-block">Quick Access</span>
            <h2 className="font-display text-3xl md:text-4xl font-800 text-slate-900 mb-4">Everything You Need</h2>
          </motion.div>
          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                icon: BookOpen, title: 'Research Documents',
                desc: 'Access all research proposals, presentations, log books, final reports, and publications in one place.',
                to: '/documents', label: 'Browse Documents',
                color: '#2d55f5', bg: '#eef3ff',
                gradient: 'from-blue-50 to-indigo-50',
                pills: ['Proposals', 'Reports', 'Presentations', 'Papers'],
                pillStyle: 'pill-blue',
              },
              {
                icon: Mail, title: 'Contact the Team',
                desc: 'Get in touch with the research team members or supervisors for collaboration, queries, or feedback.',
                to: '/contact', label: 'Get In Touch',
                color: '#05bdad', bg: '#effefb',
                gradient: 'from-teal-50 to-cyan-50',
                pills: ['Collaboration', 'Queries', 'Feedback'],
                pillStyle: 'pill-teal',
              },
            ].map((card, i) => (
              <motion.div key={card.title}
                initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.15 }}
              >
                <Link to={card.to} className={`block card p-8 bg-gradient-to-br ${card.gradient} group`} style={{ textDecoration: 'none' }}>
                  <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-6 transition-transform group-hover:scale-110"
                    style={{ background: card.bg }}>
                    <card.icon size={24} style={{ color: card.color }} />
                  </div>
                  <h3 className="font-display text-xl font-800 text-slate-900 mb-3">{card.title}</h3>
                  <p className="text-slate-500 text-sm leading-relaxed mb-5">{card.desc}</p>
                  <div className="flex flex-wrap gap-2 mb-6">
                    {card.pills.map(p => <span key={p} className={`pill ${card.pillStyle} text-[10px]`}>{p}</span>)}
                  </div>
                  <div className="flex items-center gap-2 font-600 text-sm transition-all group-hover:gap-3"
                    style={{ color: card.color }}>
                    {card.label} <ArrowRight size={14} />
                  </div>
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ CTA ═══════════════════════════════════ */}
      <section className="py-24 px-6" style={{ background: 'linear-gradient(135deg, #f8fafc 0%, #eef3ff 100%)' }}>
        <div className="max-w-3xl mx-auto text-center">
          <motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
            <motion.div
              animate={{ rotate: [0, 5, -5, 0] }} transition={{ duration: 4, repeat: Infinity }}
              className="w-16 h-16 rounded-2xl mx-auto mb-6 flex items-center justify-center"
              style={{ background: 'linear-gradient(135deg, #2d55f5, #8a55ff)', boxShadow: '0 8px 32px rgba(45,85,245,0.3)' }}
            >
              <Sparkles size={24} color="white" />
            </motion.div>
            <h2 className="font-display text-4xl md:text-5xl font-800 text-slate-900 mb-5">
              Explore the Full Research
            </h2>
            <p className="text-slate-500 text-lg mb-8 max-w-lg mx-auto">
              Four interconnected components, one clinical goal: proactive anxiety management for young adults everywhere.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link to="/methodology" className="btn-primary flex items-center gap-2 group">
                View Methodology <ArrowRight size={15} className="group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link to="/team" className="btn-outline flex items-center gap-2">
                <Users size={15} /> Meet the Team
              </Link>
              <a href="https://github.com/dulhara79/R26-DS-012" target="_blank" rel="noopener noreferrer"
                className="btn-outline flex items-center gap-2">
                <ExternalLink size={15} /> Repository
              </a>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}