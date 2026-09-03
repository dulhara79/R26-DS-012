import { motion } from 'framer-motion'
import { Mail, MapPin, Github, ExternalLink, Send, MessageSquare, Phone, Clock, Brain, Activity, Network, Cpu, Globe, Users, Linkedin, Twitter } from 'lucide-react'
import { useState } from 'react'

const CONTACT_BG = 'https://images.unsplash.com/photo-1423666639041-f56000c27a9a?w=1400&q=80&auto=format&fit=crop'

const TEAM_CONTACTS = [
    {
        id: 'IT22107596', name: 'Sendanayake H.D.', role: 'Component M1 Lead',
        email: 'it22107596@my.sliit.lk', component: 'Wearable Biosensor Forecasting',
        icon: Activity, pill: 'pill-teal', accent: 'text-teal-600', bg: 'bg-teal-50', border: 'border-teal-200',
        portrait: 'https://dewdu-sendanayake-portfolio.vercel.app/assets/profile.jpg',
        github: 'https://github.com/dulhara79/R26-DS-012',
        linkedin: 'https://linkedin.com',
        focus: 'ESP32 Wearables · LSTM-AE · Self-Supervised Learning',
    },
    {
        id: 'IT22171542', name: 'Layathma B.M.A.S.', role: 'Component M2 Lead',
        email: 'it22171542@my.sliit.lk', component: 'Temporal Behavioral Graph Framework',
        icon: Network, pill: 'pill-blue', accent: 'text-brand-600', bg: 'bg-brand-50', border: 'border-brand-200',
        portrait: 'https://avatars.githubusercontent.com/u/152529793?v=4',
        github: 'https://github.com/dulhara79/R26-DS-012',
        linkedin: 'https://linkedin.com',
        focus: 'GATv2 · Digital Phenotyping · Android Development',
    },
    {
        id: 'IT22093950', name: 'Seneviratne K.A.U.A.', role: 'Component M3 Lead',
        email: 'it22093950@my.sliit.lk', component: 'Adaptive Intervention Engine',
        icon: Cpu, pill: 'pill-violet', accent: 'text-violet-600', bg: 'bg-violet-50', border: 'border-violet-200',
        portrait: 'https://media.licdn.com/dms/image/v2/D5603AQGLnYQnd7NOxA/profile-displayphoto-crop_800_800/B56Z1wKQCZJ8AI-/0/1775703231658?e=1781136000&v=beta&t=dLPtOo0Y6Zgq2mS5ftyWI6ZlqlC-rxATopVWuGpdXcc',
        github: 'https://github.com/dulhara79/R26-DS-012',
        linkedin: 'https://linkedin.com',
        focus: 'KNN Case-Based Reasoning · Flutter · FastAPI',
    },
    {
        id: 'IT22130648', name: 'Kaushalya I.G.D.', role: 'Component M4 Lead',
        email: 'it22130648@my.sliit.lk', component: 'Clinical NLP (TC-WPN)',
        icon: Brain, pill: 'pill-rose', accent: 'text-rose-600', bg: 'bg-rose-50', border: 'border-rose-200',
        portrait: 'https://dulharakaushalya.vercel.app/IMG_7353.JPG',
        github: 'https://github.com/dulhara79/R26-DS-012',
        linkedin: 'https://linkedin.com',
        focus: 'ClinicalBERT · Prototypical Networks · MIMIC-IV',
    },
]

const SUPERVISORS_CONTACT = [
    {
        name: 'Prof. Samantha Thelijjagoda', role: 'Research Supervisor',
        email: 'samantha.t@sliit.lk', dept: 'Dept. of Computer Science, SLIIT',
        ini: 'ST', accent: 'text-brand-600', bg: 'bg-brand-50', border: 'border-brand-200',
    },
    {
        name: 'Dr. Mahima Weerasinghe', role: 'Co-Supervisor',
        email: 'mahima.w@sliit.lk', dept: 'Dept. of Computer Science, SLIIT',
        ini: 'MW', accent: 'text-teal-600', bg: 'bg-teal-50', border: 'border-teal-200',
    },
    {
        name: 'Dr. Chathurie Suraweera', role: 'Clinical Supervisor',
        email: 'chathurie.s@nhsl.lk', dept: 'National Hospital of Sri Lanka',
        ini: 'CS', accent: 'text-violet-600', bg: 'bg-violet-50', border: 'border-violet-200',
    },
]

const INFO_CARDS = [
    { icon: MapPin, title: 'Our Location', lines: ['Sri Lanka Institute of Information Technology', 'New Kandy Rd, Malabe 10115, Sri Lanka'], color: '#2d55f5', bg: '#eef3ff' },
    { icon: Mail, title: 'Primary Contact', lines: ['it22130648@my.sliit.lk', 'Research enquiries welcome'], color: '#05bdad', bg: '#effefb' },
    { icon: Phone, title: 'Institution', lines: ['+94 (11) 754-4801', 'SLIIT Main Switchboard'], color: '#8a55ff', bg: '#f6f2ff' },
    { icon: Clock, title: 'Response Time', lines: ['Within 48 hours', 'Monday – Friday, 9am – 5pm (IST)'], color: '#e2142e', bg: '#fff1f3' },
]

export default function Contact() {
    const [sent, setSent] = useState(false)
    const [form, setForm] = useState({ name: '', email: '', subject: '', message: '' })

    const handleSubmit = (e) => {
        e.preventDefault()
        setSent(true)
        setTimeout(() => setSent(false), 4000)
    }

    return (
        <div className="min-h-screen pb-32">

            {/* Hero */}
            <div className="relative h-72 overflow-hidden">
                <img src={CONTACT_BG} alt="Contact" className="w-full h-full object-cover" style={{ filter: 'brightness(0.4) saturate(0.8)' }} />
                <div className="absolute inset-0 bg-gradient-to-b from-transparent to-slate-50" />
                <div className="absolute inset-0 flex items-end pb-12 px-6">
                    <div className="max-w-5xl mx-auto w-full">
                        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                            <span className="pill pill-blue mb-4 inline-block">Get In Touch</span>
                            <h1 className="font-display text-5xl md:text-6xl font-800 text-white">Contact the Team</h1>
                        </motion.div>
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-6 pt-14">

                {/* Info cards */}
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-16">
                    {INFO_CARDS.map((card, i) => (
                        <motion.div key={card.title}
                            initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 + i * 0.08 }}
                            className="card-flat p-5 group hover:shadow-lg transition-all"
                        >
                            <div className="w-11 h-11 rounded-2xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110"
                                style={{ background: card.bg }}>
                                <card.icon size={20} style={{ color: card.color }} />
                            </div>
                            <h3 className="font-display text-sm font-700 text-slate-800 mb-2">{card.title}</h3>
                            {card.lines.map((l, j) => (
                                <p key={j} className="text-xs leading-relaxed" style={{ color: j === 0 ? '#334155' : '#94a3b8' }}>{l}</p>
                            ))}
                        </motion.div>
                    ))}
                </div>

                {/* Contact form + map */}
                <div className="grid lg:grid-cols-2 gap-10 mb-20">

                    {/* Form */}
                    <motion.div initial={{ opacity: 0, x: -28 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
                        className="card-flat p-8">
                        <div className="flex items-center gap-3 mb-7">
                            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: '#eef3ff' }}>
                                <MessageSquare size={18} style={{ color: '#2d55f5' }} />
                            </div>
                            <div>
                                <h2 className="font-display text-xl font-800 text-slate-900">Send a Message</h2>
                                <p className="text-xs text-slate-400">We'll respond within 48 hours</p>
                            </div>
                        </div>

                        {sent ? (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                                className="flex flex-col items-center justify-center py-12 text-center"
                            >
                                <div className="w-16 h-16 rounded-full bg-green-50 border border-green-200 flex items-center justify-center mb-4">
                                    <Send size={24} className="text-green-600" />
                                </div>
                                <h3 className="font-display text-lg font-700 text-slate-900 mb-2">Message Sent!</h3>
                                <p className="text-sm text-slate-500">Thank you for reaching out. We'll get back to you soon.</p>
                            </motion.div>
                        ) : (
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-xs font-600 text-slate-600 mb-1.5">Full Name</label>
                                        <input type="text" required placeholder="Your name"
                                            value={form.name} onChange={e => setForm({ ...form, name: e.target.value })}
                                            className="w-full px-4 py-3 rounded-xl border border-slate-200 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all bg-white"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-600 text-slate-600 mb-1.5">Email Address</label>
                                        <input type="email" required placeholder="you@example.com"
                                            value={form.email} onChange={e => setForm({ ...form, email: e.target.value })}
                                            className="w-full px-4 py-3 rounded-xl border border-slate-200 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all bg-white"
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-xs font-600 text-slate-600 mb-1.5">Subject</label>
                                    <input type="text" placeholder="Research collaboration / Query / Feedback"
                                        value={form.subject} onChange={e => setForm({ ...form, subject: e.target.value })}
                                        className="w-full px-4 py-3 rounded-xl border border-slate-200 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all bg-white"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs font-600 text-slate-600 mb-1.5">Message</label>
                                    <textarea rows={5} required placeholder="Your message here..."
                                        value={form.message} onChange={e => setForm({ ...form, message: e.target.value })}
                                        className="w-full px-4 py-3 rounded-xl border border-slate-200 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 transition-all bg-white resize-none"
                                    />
                                </div>
                                <button type="submit"
                                    className="btn-primary w-full flex items-center justify-center gap-2 group"
                                >
                                    <Send size={15} />
                                    Send Message
                                </button>
                            </form>
                        )}
                    </motion.div>

                    {/* Location + quick links */}
                    <motion.div initial={{ opacity: 0, x: 28 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}
                        className="flex flex-col gap-5">

                        {/* Map embed */}
                        <div className="card-flat overflow-hidden rounded-2xl h-56">
                            <iframe
                                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3960.678!2d79.9729!3d6.9147!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3ae256db1a6771c5%3A0x2c63e344ab9a7536!2sSri%20Lanka%20Institute%20of%20Information%20Technology!5e0!3m2!1sen!2slk!4v1"
                                width="100%" height="100%" style={{ border: 0 }} allowFullScreen loading="lazy"
                                referrerPolicy="no-referrer-when-downgrade" title="SLIIT Location"
                            />
                        </div>

                        {/* Quick links */}
                        <div className="card-flat p-6">
                            <h3 className="font-display text-base font-700 text-slate-800 mb-4">Quick Links</h3>
                            <div className="space-y-3">
                                {[
                                    { icon: Github, label: 'Research Repository', sub: 'Full codebase on GitHub', href: 'https://github.com/dulhara79/R26-DS-012', color: '#334155' },
                                    { icon: Globe, label: 'SLIIT Official Website', sub: 'www.sliit.lk', href: 'https://www.sliit.lk', color: '#2d55f5' },
                                    { icon: ExternalLink, label: 'Department of CS', sub: 'Dept. of Computer Science', href: 'https://www.sliit.lk/faculties/computing/', color: '#8a55ff' },
                                ].map(link => (
                                    <a key={link.label} href={link.href} target="_blank" rel="noopener noreferrer"
                                        className="flex items-center gap-3 p-3 rounded-xl hover:bg-slate-50 transition-all group">
                                        <div className="w-9 h-9 rounded-xl bg-slate-100 group-hover:bg-white flex items-center justify-center transition-all flex-shrink-0">
                                            <link.icon size={15} style={{ color: link.color }} />
                                        </div>
                                        <div>
                                            <div className="text-sm font-600 text-slate-700 group-hover:text-slate-900 transition-colors">{link.label}</div>
                                            <div className="text-xs text-slate-400">{link.sub}</div>
                                        </div>
                                    </a>
                                ))}
                            </div>
                        </div>
                    </motion.div>
                </div>

                {/* Team member contacts */}
                <div className="mb-20">
                    <motion.div initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
                        <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Reach a Team Member Directly</h2>
                    </motion.div>
                    <div className="grid md:grid-cols-2 gap-5">
                        {TEAM_CONTACTS.map((m, i) => (
                            <motion.div key={m.id}
                                initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                                className="card-flat p-5 flex items-start gap-4 group hover:shadow-md transition-all"
                            >
                                <img src={m.portrait} alt={m.name}
                                    className="w-14 h-14 rounded-2xl object-cover object-top flex-shrink-0 border-2 border-white shadow-sm"
                                />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className={`pill ${m.pill} text-[10px]`}>{m.id}</span>
                                    </div>
                                    <h3 className="font-display text-sm font-700 text-slate-900 mb-0.5">{m.name}</h3>
                                    <p className={`text-xs font-600 ${m.accent} mb-1`}>{m.role}</p>
                                    <p className="text-xs text-slate-400 mb-2 truncate">{m.focus}</p>
                                    <a href={`mailto:${m.email}`}
                                        className="inline-flex items-center gap-1.5 text-xs text-slate-600 hover:text-brand-600 transition-colors font-mono">
                                        <Mail size={11} /> {m.email}
                                    </a>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Supervisors contact */}
                <motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
                    <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Supervisors</h2>
                    <div className="grid md:grid-cols-3 gap-5">
                        {SUPERVISORS_CONTACT.map((s, i) => (
                            <motion.div key={s.name}
                                initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                                className="card-flat p-6 group hover:shadow-md transition-all"
                            >
                                <div className={`w-14 h-14 rounded-2xl ${s.bg} border ${s.border} flex items-center justify-center mb-4 transition-transform group-hover:scale-105`}>
                                    <span className={`font-display text-lg font-800 ${s.accent}`}>{s.ini}</span>
                                </div>
                                <h3 className="font-display text-base font-700 text-slate-900 mb-1">{s.name}</h3>
                                <div className={`text-xs font-600 ${s.accent} mb-1`}>{s.role}</div>
                                <p className="text-xs text-slate-400 mb-3">{s.dept}</p>
                                <a href={`mailto:${s.email}`}
                                    className="inline-flex items-center gap-1.5 text-xs font-mono text-slate-500 hover:text-brand-600 transition-colors">
                                    <Mail size={11} /> {s.email}
                                </a>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            </div>
        </div>
    )
}