import { motion } from 'framer-motion'
import { Activity, Network, Cpu, Brain, Github, ExternalLink } from 'lucide-react'

const TEAM_IMG = 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=1400&q=80&auto=format&fit=crop'

// Realistic Unsplash portrait photos (generic professionals)
const PORTRAITS = [
  'https://dewdu-sendanayake-portfolio.vercel.app/assets/profile.jpg',
  'https://avatars.githubusercontent.com/u/152529793?v=4',
  'https://media.licdn.com/dms/image/v2/D5603AQGLnYQnd7NOxA/profile-displayphoto-crop_800_800/B56Z1wKQCZJ8AI-/0/1775703231658?e=1781136000&v=beta&t=dLPtOo0Y6Zgq2mS5ftyWI6ZlqlC-rxATopVWuGpdXcc',
  'https://dulharakaushalya.vercel.app/IMG_7353.JPG',
]

const TEAM = [
  { id:'IT22107596', name:'Sendanayake H.D.', role:'Component M1 Lead', component:'Wearable Biosensor Forecasting', icon:Activity,
    pill:'pill-teal', accent:'text-teal-600', bg:'bg-teal-50', border:'border-teal-200',
    focus:['Self-Supervised Learning','Wearable Hardware','Time-Series Anomaly Detection','ESP32 Firmware'],
    bio:'Responsible for designing and implementing the custom ESP32-based chest-strap wearable and the LSTM autoencoder self-supervised framework for acute autonomic destabilization forecasting in real-world settings.', portrait: PORTRAITS[0] },
  { id:'IT22171542', name:'Layathma B.M.A.S.', role:'Component M2 Lead', component:'Temporal Behavioral Graph Framework', icon:Network,
    pill:'pill-blue', accent:'text-brand-600', bg:'bg-brand-50', border:'border-brand-200',
    focus:['Graph Neural Networks','Android Development','Digital Phenotyping','GATv2 Architecture'],
    bio:'Leads design of the temporal behavioral graph construction schema and GATv2 model for anxiety vulnerability mapping from passive smartphone sensing streams collected over 4–6 weeks.', portrait: PORTRAITS[1] },
  { id:'IT22093950', name:'Seneviratne K.A.U.A.', role:'Component M3 Lead', component:'Adaptive Intervention Engine', icon:Cpu,
    pill:'pill-violet', accent:'text-violet-600', bg:'bg-violet-50', border:'border-violet-200',
    focus:['KNN Case-Based Reasoning','Flutter Mobile App','Continuous Learning','Clinical Dashboard'],
    bio:'Designed the continuously learning KNN intervention framework pre-trained on NHANES data, with composite reward function driving adaptive personalization and real-time clinician alerting.', portrait: PORTRAITS[2] },
  { id:'IT22130648', name:'Kaushalya I.G.D.', role:'Component M4 Lead', component:'Clinical NLP (TC-WPN)', icon:Brain,
    pill:'pill-rose', accent:'text-rose-600', bg:'bg-rose-50', border:'border-rose-200',
    focus:['Few-Shot Meta-Learning','ClinicalBERT','Prototypical Networks','FastAPI Deployment'],
    bio:'Developed TC-WPN — a meta-learning framework integrating temporal recency decay and entropy-based confidence weighting into prototypical networks for few-shot clinical anxiety detection with only 10–20 labeled examples.', portrait: PORTRAITS[3] },
]

const SUPERVISORS = [
  { name:'Prof. Samantha Thelijjagoda', role:'Research Supervisor', aff:'SLIIT Dept. of Computer Science', ini:'ST', bg:'bg-brand-50', border:'border-brand-200', accent:'text-brand-600' },
  { name:'Dr. Mahima Weerasinghe', role:'Co-Supervisor', aff:'SLIIT Dept. of Computer Science', ini:'MW', bg:'bg-teal-50', border:'border-teal-200', accent:'text-teal-600' },
  { name:'Dr. Chathurie Suraweera', role:'External / Clinical Supervisor', aff:'National Hospital of Sri Lanka', ini:'CS', bg:'bg-violet-50', border:'border-violet-200', accent:'text-violet-600' },
]

export default function Team() {
  return (
    <div className="min-h-screen pb-32">

      {/* Hero banner */}
      <div className="relative h-72 overflow-hidden">
        <img src={TEAM_IMG} alt="Research team collaboration" className="w-full h-full object-cover" style={{filter:'brightness(0.4)'}}/>
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-slate-50"/>
        <div className="absolute inset-0 flex items-end pb-12 px-6">
          <div className="max-w-5xl mx-auto w-full">
            <motion.div initial={{opacity:0,y:24}} animate={{opacity:1,y:0}} transition={{delay:0.1}}>
              <span className="pill pill-violet mb-4 inline-block">Research Team</span>
              <h1 className="font-display text-5xl md:text-6xl font-800 text-white">The People Behind the Research</h1>
            </motion.div>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 pt-12">

        {/* Team cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-20">
          {TEAM.map((m,i)=>(
            <motion.div key={m.id}
              initial={{opacity:0,y:36}} animate={{opacity:1,y:0}} transition={{delay:0.1+i*0.12}}
              className="card overflow-hidden"
            >
              {/* Portrait */}
              <div className="relative h-48 overflow-hidden rounded-t-[20px]">
                <img src={m.portrait} alt={m.name} className="w-full h-full object-cover object-top"/>
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent"/>
                <div className="absolute top-4 left-4">
                  <span className={`pill ${m.pill}`}>{m.role.split(' ')[1]}</span>
                </div>
                <div className="absolute bottom-4 left-4">
                  <div className="font-mono text-xs text-white/60 mb-0.5">{m.id}</div>
                  <div className="font-display text-xl font-800 text-white">{m.name}</div>
                </div>
              </div>

              {/* Body */}
              <div className="p-6">
                <div className={`flex items-center gap-2 ${m.accent} mb-3`}>
                  <m.icon size={15}/>
                  <span className="text-sm font-600">{m.component}</span>
                </div>
                <p className="text-sm text-slate-500 leading-relaxed mb-4">{m.bio}</p>
                <div className="flex flex-wrap gap-1.5">
                  {m.focus.map(f=>(
                    <span key={f} className={`pill ${m.pill} text-[10px]`}>{f}</span>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Supervisors */}
        <motion.div initial={{opacity:0,y:28}} whileInView={{opacity:1,y:0}} viewport={{once:true}} className="mb-20">
          <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Supervisors</h2>
          <div className="grid md:grid-cols-3 gap-5">
            {SUPERVISORS.map((s,i)=>(
              <motion.div key={s.name}
                initial={{opacity:0,y:20}} whileInView={{opacity:1,y:0}}
                viewport={{once:true}} transition={{delay:i*0.1}}
                className="card-flat p-6"
              >
                <div className={`w-14 h-14 rounded-2xl ${s.bg} border ${s.border} flex items-center justify-center mb-4`}>
                  <span className={`font-display text-lg font-800 ${s.accent}`}>{s.ini}</span>
                </div>
                <h3 className="font-display text-base font-700 text-slate-900 mb-1">{s.name}</h3>
                <div className={`text-xs font-600 ${s.accent} mb-1`}>{s.role}</div>
                <p className="text-xs text-slate-400">{s.aff}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Institution */}
        <motion.div initial={{opacity:0,y:28}} whileInView={{opacity:1,y:0}} viewport={{once:true}}
          className="card-flat p-8 md:p-12">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div>
              <span className="pill pill-blue mb-5 inline-block">Institution</span>
              <h2 className="font-display text-3xl font-800 text-slate-900 mb-4">Sri Lanka Institute of Information Technology</h2>
              <p className="text-slate-500 leading-relaxed mb-5">B.Sc. (Hons) Degree in Information Technology, Specialized in Data Science. Department of Computer Science.</p>
              <div className="flex gap-4">
                <a href="https://www.sliit.lk" target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-1.5 text-sm text-brand-600 hover:text-brand-700 font-medium transition-colors">
                  <ExternalLink size={14}/> SLIIT Website
                </a>
                <a href="https://github.com/dulhara79/R26-DS-012" target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800 font-medium transition-colors">
                  <Github size={14}/> Repository
                </a>
              </div>
            </div>
            <div className="divide-y divide-slate-100">
              {[
                ['Project Code','R26-DS-012'],
                ['Degree Programme','B.Sc. (Hons) IT — Data Science'],
                ['Academic Year','2025/2026'],
                ['Clinical Partner','National Hospital of Sri Lanka'],
                ['Submission','March 2026'],
              ].map(([k,v])=>(
                <div key={k} className="flex justify-between items-center py-3 text-sm">
                  <span className="text-slate-400">{k}</span>
                  <span className="text-slate-800 font-600">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
