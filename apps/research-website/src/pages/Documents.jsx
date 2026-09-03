import { motion } from 'framer-motion'
import { FileText, ExternalLink, Download, BookOpen, Presentation, ClipboardList, Award, Users, Package, Copy, Check, ChevronRight, File } from 'lucide-react'
import { useState } from 'react'

const DOCS_BG = 'https://images.unsplash.com/photo-1456324504439-367cee3b3c32?w=1400&q=80&auto=format&fit=crop'

// ── Replace these href values with your actual Google Drive links ──
const DOC_SECTIONS = [
    {
        category: 'Research Proposal',
        icon: BookOpen,
        pill: 'pill-blue',
        accent: 'text-brand-600',
        bg: 'bg-brand-50',
        border: 'border-brand-200',
        color: '#2d55f5',
        description: 'Our initial project proposal documents outlining objectives, methodology, and expected outcomes.',
        count: 1,
        docs: [
            {
                title: 'Proposal Reports',
                sub: 'Individual reports from all 4 team members',
                size: '4 × ~2.1 MB',
                pages: '~60 pages total',
                href: 'https://drive.google.com/drive/folders/YOUR_FOLDER_ID_1',
                type: 'folder',
            },
        ],
    },
    {
        category: 'Presentations',
        icon: Presentation,
        pill: 'pill-teal',
        accent: 'text-teal-600',
        bg: 'bg-teal-50',
        border: 'border-teal-200',
        color: '#05bdad',
        description: 'Slides and materials from our project presentations throughout the research timeline.',
        count: 3,
        docs: [
            {
                title: 'Proposal Presentation',
                sub: 'Initial project pitch presentation',
                size: '~8.5 MB',
                pages: '32 slides',
                href: 'https://drive.google.com/file/d/YOUR_FILE_ID_1/view',
                type: 'file',
            },
            {
                title: 'Progress Presentation 1',
                sub: 'First milestone results and findings',
                size: '~11.2 MB',
                pages: '45 slides',
                href: 'https://drive.google.com/file/d/YOUR_FILE_ID_2/view',
                type: 'file',
            },
            {
                title: 'Progress Presentation 2',
                sub: 'Second milestone results and updates',
                size: '~9.8 MB',
                pages: '38 slides',
                href: 'https://drive.google.com/file/d/YOUR_FILE_ID_3/view',
                type: 'file',
            },
        ],
    },
    {
        category: 'Log Books',
        icon: ClipboardList,
        pill: 'pill-violet',
        accent: 'text-violet-600',
        bg: 'bg-violet-50',
        border: 'border-violet-200',
        color: '#8a55ff',
        description: 'Detailed tracking of research activities, experiments, and meeting minutes from all team members.',
        count: 1,
        docs: [
            {
                title: 'Team Log Books',
                sub: 'Weekly logs from all 4 team members',
                size: '4 × ~1.3 MB',
                pages: '~80 entries total',
                href: 'https://drive.google.com/drive/folders/YOUR_FOLDER_ID_2',
                type: 'folder',
            },
        ],
    },
    {
        category: 'Final Reports',
        icon: FileText,
        pill: 'pill-rose',
        accent: 'text-rose-600',
        bg: 'bg-rose-50',
        border: 'border-rose-200',
        color: '#e2142e',
        description: 'Comprehensive documentation of our complete research findings, methodology, results, and conclusions.',
        count: 1,
        docs: [
            {
                title: 'Final Reports',
                sub: 'Complete documentation of research findings',
                size: '4 × ~5.4 MB',
                pages: '~120 pages total',
                href: 'https://drive.google.com/drive/folders/YOUR_FOLDER_ID_3',
                type: 'folder',
            },
        ],
    },
    {
        category: 'Research Paper',
        icon: Award,
        pill: 'pill-green',
        accent: 'text-green-700',
        bg: 'bg-green-50',
        border: 'border-green-200',
        color: '#15803d',
        description: 'Our academic paper presenting the key findings and innovations of the R26-DS-012 project.',
        count: 1,
        docs: [
            {
                title: 'Published Paper',
                sub: 'Academic publication of our research',
                size: '~3.2 MB',
                pages: '14 pages · IEEE format',
                href: 'https://drive.google.com/file/d/YOUR_FILE_ID_4/view',
                type: 'file',
            },
        ],
    },
    {
        category: 'Team Information',
        icon: Users,
        pill: 'pill-slate',
        accent: 'text-slate-600',
        bg: 'bg-slate-50',
        border: 'border-slate-200',
        color: '#475569',
        description: 'Details about our research team members, roles, and contributions to the project.',
        count: 1,
        docs: [
            {
                title: 'Team Profiles',
                sub: 'Information about all 4 team members',
                size: '~1.8 MB',
                pages: '8 pages',
                href: 'https://drive.google.com/file/d/YOUR_FILE_ID_5/view',
                type: 'file',
            },
        ],
    },
]

const CITATION = `Sendanayake, H.D., Layathma, B.M.A.S., Seneviratne, K.A.U.A., & Kaushalya, I.G.D. (2026). A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders (R26-DS-012). B.Sc. (Hons) Research Project. Department of Computer Science, Sri Lanka Institute of Information Technology, Malabe, Sri Lanka. Supervised by Prof. S. Thelijjagoda.`

function DocCard({ doc, accent, pill, color }) {
    return (
        <a href={doc.href} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-4 p-4 rounded-2xl border border-slate-100 hover:border-slate-200 hover:bg-slate-50 hover:shadow-sm transition-all group"
        >
            <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform group-hover:scale-105"
                style={{ background: `${color}15` }}>
                {doc.type === 'folder'
                    ? <Package size={18} style={{ color }} />
                    : <File size={18} style={{ color }} />
                }
            </div>
            <div className="flex-1 min-w-0">
                <div className="font-display text-sm font-700 text-slate-800 group-hover:text-slate-900 transition-colors">{doc.title}</div>
                <div className="text-xs text-slate-400 mt-0.5">{doc.sub}</div>
                <div className="flex items-center gap-3 mt-1.5">
                    <span className="font-mono text-[10px] text-slate-400">{doc.size}</span>
                    <span className="w-1 h-1 rounded-full bg-slate-300" />
                    <span className="font-mono text-[10px] text-slate-400">{doc.pages}</span>
                </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
                <span className={`pill ${pill} text-[10px]`}>
                    {doc.type === 'folder' ? 'Folder' : 'PDF'}
                </span>
                <ExternalLink size={14} className="text-slate-300 group-hover:text-slate-500 transition-colors" />
            </div>
        </a>
    )
}

export default function Documents() {
    const [copied, setCopied] = useState(false)

    const handleCopy = () => {
        navigator.clipboard.writeText(CITATION)
        setCopied(true)
        setTimeout(() => setCopied(false), 2500)
    }

    return (
        <div className="min-h-screen pb-32">

            {/* Hero */}
            <div className="relative h-72 overflow-hidden">
                <img src={DOCS_BG} alt="Research documents" className="w-full h-full object-cover" style={{ filter: 'brightness(0.38) saturate(0.7)' }} />
                <div className="absolute inset-0 bg-gradient-to-b from-transparent to-slate-50" />
                <div className="absolute inset-0 flex items-end pb-12 px-6">
                    <div className="max-w-5xl mx-auto w-full">
                        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                            <span className="pill pill-blue mb-4 inline-block">Research Documentation</span>
                            <h1 className="font-display text-5xl md:text-6xl font-800 text-white">Research Documents</h1>
                        </motion.div>
                    </div>
                </div>
            </div>

            <div className="max-w-5xl mx-auto px-6 pt-14">

                {/* Intro */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
                    className="card-flat p-7 mb-14">
                    <div className="flex items-start gap-5">
                        <div className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0" style={{ background: '#eef3ff' }}>
                            <BookOpen size={22} style={{ color: '#2d55f5' }} />
                        </div>
                        <div>
                            <h2 className="font-display text-xl font-800 text-slate-900 mb-2">Access All Documentation</h2>
                            <p className="text-slate-500 text-sm leading-relaxed max-w-2xl">
                                Access all documentation related to our R26-DS-012 research project, from initial proposals to final reports. All documents are hosted on Google Drive and accessible via the links below.
                            </p>
                            <div className="flex flex-wrap gap-3 mt-4">
                                {[
                                    { v: '6', l: 'Categories' },
                                    { v: '8+', l: 'Documents' },
                                    { v: '4', l: 'Team Members' },
                                    { v: '2026', l: 'Academic Year' },
                                ].map(s => (
                                    <div key={s.l} className="bg-brand-50 border border-brand-100 rounded-xl px-4 py-2 text-center">
                                        <div className="font-display text-lg font-800 text-brand-600">{s.v}</div>
                                        <div className="text-[10px] text-brand-400 font-mono">{s.l}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Document sections */}
                <div className="space-y-8 mb-16">
                    {DOC_SECTIONS.map((section, i) => (
                        <motion.div key={section.category}
                            initial={{ opacity: 0, y: 32 }} whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }} transition={{ delay: i * 0.07 }}
                            className="card-flat overflow-hidden"
                        >
                            {/* Section header */}
                            <div className={`p-6 border-b border-slate-100 ${section.bg}`}>
                                <div className="flex items-start justify-between gap-4">
                                    <div className="flex items-center gap-4">
                                        <div className={`w-12 h-12 rounded-2xl bg-white/80 border ${section.border} flex items-center justify-center`}>
                                            <section.icon size={20} style={{ color: section.color }} />
                                        </div>
                                        <div>
                                            <div className="flex items-center gap-2 mb-1">
                                                <h3 className="font-display text-lg font-800 text-slate-900">{section.category}</h3>
                                                <span className={`pill ${section.pill} text-[10px]`}>{section.count} {section.count === 1 ? 'document' : 'documents'}</span>
                                            </div>
                                            <p className="text-sm text-slate-500 max-w-lg">{section.description}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Doc links */}
                            <div className="p-5 space-y-3">
                                {section.docs.map(doc => (
                                    <DocCard key={doc.title} doc={doc} accent={section.accent} pill={section.pill} color={section.color} />
                                ))}
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* Download all */}
                <motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                    className="card-flat p-8 md:p-10 mb-12"
                    style={{ background: 'linear-gradient(135deg, #f0f5ff 0%, #f6f2ff 100%)', borderColor: '#c7d2fe' }}
                >
                    <div className="flex flex-col md:flex-row items-center gap-6">
                        <div className="w-16 h-16 rounded-3xl flex items-center justify-center flex-shrink-0"
                            style={{ background: 'linear-gradient(135deg, #2d55f5, #8a55ff)', boxShadow: '0 8px 24px rgba(45,85,245,0.3)' }}>
                            <Package size={26} color="white" />
                        </div>
                        <div className="flex-1 text-center md:text-left">
                            <h3 className="font-display text-2xl font-800 text-slate-900 mb-2">Download Complete Research Package</h3>
                            <p className="text-slate-500 text-sm max-w-md">Get all our research documents in a single download, including proposals, presentations, reports, and published papers.</p>
                        </div>
                        <a href="https://drive.google.com/drive/folders/YOUR_COMPLETE_PACKAGE_FOLDER_ID"
                            target="_blank" rel="noopener noreferrer"
                            className="btn-primary flex items-center gap-2 group whitespace-nowrap flex-shrink-0"
                        >
                            <Download size={15} />
                            Download Package
                            <span className="font-mono text-xs opacity-70">(PDF, ~40MB)</span>
                        </a>
                    </div>
                </motion.div>

                {/* Citation */}
                <motion.div initial={{ opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}
                    className="card-flat p-8">
                    <div className="flex items-center gap-3 mb-5">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: '#effefb' }}>
                            <BookOpen size={18} style={{ color: '#05bdad' }} />
                        </div>
                        <h3 className="font-display text-xl font-800 text-slate-900">How to Cite Our Research</h3>
                    </div>
                    <p className="text-sm text-slate-500 mb-4">If you're using our research in your academic work, please use the following citation format.</p>

                    <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5 relative">
                        <p className="text-sm text-slate-700 leading-relaxed font-mono pr-10">{CITATION}</p>
                        <button onClick={handleCopy}
                            className="absolute top-4 right-4 w-8 h-8 rounded-xl bg-white border border-slate-200 flex items-center justify-center hover:bg-brand-50 hover:border-brand-200 transition-all"
                        >
                            {copied
                                ? <Check size={14} className="text-green-600" />
                                : <Copy size={14} className="text-slate-500" />
                            }
                        </button>
                    </div>
                    {copied && (
                        <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
                            className="text-xs text-green-600 mt-2 flex items-center gap-1">
                            <Check size={11} /> Citation copied to clipboard
                        </motion.p>
                    )}
                </motion.div>
            </div>
        </div>
    )
}