import { motion } from 'framer-motion'
import { Database, Code, FlaskConical, Layers, Shield, CheckCircle2, ArrowRight, BarChart3 } from 'lucide-react'

const BG = 'https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=1400&q=80&auto=format&fit=crop'
const LAB = 'https://images.unsplash.com/photo-1576671081399-b21be6f9f6e2?w=900&q=80&auto=format&fit=crop'

const PHASES = [
  { n:'01', title:'Problem Identification & DSR Design', dur:'Months 1-2', icon:FlaskConical, pill:'pill-blue',
    tasks:['Literature review: Clinical NLP, few-shot learning, uncertainty estimation, temporal modeling','Research gap analysis and DSR artifact definition','Ethical approval applications (NHSL + SLIIT IRB)','API contract finalisation between M1–M4 components'] },
  { n:'02', title:'Data Acquisition & Preprocessing', dur:'Months 2-4', icon:Database, pill:'pill-teal',
    tasks:['MIMIC-IV access via PhysioNet (CITI certification + DUA)','NHANES 2017-2020 preprocessing (ages 18-35, 4,500-6,000 records)','StudentLife dataset acquisition and re-preprocessing for M2','NHSL clinical data collection (30-50 patients, pseudonymized)','Mobile sensing app deployment for M2 primary dataset (10-20 participants)'] },
  { n:'03', title:'Model Development', dur:'Months 3-7', icon:Code, pill:'pill-violet',
    tasks:['M1: LSTM-AE self-supervised pre-training on WESAD/SWELL-KW datasets','M2: GATv2 temporal behavioral graph construction and training on StudentLife','M3: Gradient Boosting + KNN BallTree on NHANES data with SMOTE balancing','M4: TC-WPN meta-training on MIMIC-IV (10,000 N=2-way K-shot episodes)','ClinicalBERT fine-tuning with AdamW episodic training protocol'] },
  { n:'04', title:'Integration & Evaluation', dur:'Months 7-9', icon:Layers, pill:'pill-rose',
    tasks:['FastAPI RESTful interfaces for all four components','Late-fusion model with learned weights (w₁–w₄)','NHSL few-shot adaptation and clinical pilot validation','Ablation studies for TC-WPN (λ, β sweep) and GATv2 (edge type removal)','Scenario-based end-to-end system validation (3 personas)','Research paper submission to IEEE JBHI / Frontiers in Digital Health'] },
]

const DATASETS = [
  { name:'MIMIC-IV', type:'Clinical Notes', size:'299,712 admissions · 331,794 notes', use:'TC-WPN meta-training (M4)', notes:'8,500 anxiety+ · 15,000 non-anxiety', pill:'pill-rose', accent:'text-rose-600' },
  { name:'NHSL Dataset', type:'Clinical Target', size:'15-50 de-identified notes', use:'Few-shot adaptation (M4) · M3 fine-tuning', notes:'IRB-approved · Binary annotated by psychiatrist', pill:'pill-teal', accent:'text-teal-600' },
  { name:'StudentLife', type:'Passive Sensing', size:'48 students · 10 weeks', use:'GATv2 training & validation (M2)', notes:'GPS, activity, sleep, social · Dartmouth College', pill:'pill-blue', accent:'text-brand-600' },
  { name:'NHANES 2017-2020', type:'Demographic + GAD-7', size:'4,500-6,000 young adult records', use:'KNN pre-training (M3)', notes:'CDC open-access · Ages 18-35 filtered', pill:'pill-violet', accent:'text-violet-600' },
  { name:'WESAD + SWELL-KW', type:'Physiological Signals', size:'Multi-participant lab studies', use:'LSTM-AE pre-training (M1)', notes:'ECG, respiration, temperature, motion', pill:'pill-slate', accent:'text-slate-600' },
]

export default function Methodology() {
  return (
    <div className="min-h-screen pb-32">

      {/* Header with image */}
      <div className="relative h-72 overflow-hidden">
        <img src={BG} alt="Laboratory research" className="w-full h-full object-cover" style={{filter:'brightness(0.45) saturate(0.8)'}}/>
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-slate-50" />
        <div className="absolute inset-0 flex items-end pb-12 px-6">
          <div className="max-w-5xl mx-auto w-full">
            <motion.div initial={{opacity:0,y:24}} animate={{opacity:1,y:0}} transition={{delay:0.1}}>
              <span className="pill pill-teal mb-4 inline-block">Research Design</span>
              <h1 className="font-display text-5xl md:text-6xl font-800 text-white">Methodology</h1>
            </motion.div>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 pt-12">

        {/* DSR */}
        <motion.div initial={{opacity:0,y:24}} whileInView={{opacity:1,y:0}} viewport={{once:true}}
          className="card-flat p-8 mb-16">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="font-display text-2xl font-800 text-slate-900 mb-4">Design Science Research (DSR)</h2>
              <p className="text-slate-500 leading-relaxed mb-5">DSR emphasizes construction and evaluation of novel computational artifacts. The six-phase cycle drives our research from gap identification through validated deployment.</p>
              <div className="flex flex-wrap gap-2">
                {['Problem ID','Objectives','Design & Build','Demonstrate','Evaluate','Communicate'].map((s,i)=>(
                  <div key={s} className="flex items-center gap-1.5">
                    <span className="pill pill-blue text-[10px]">{i+1}. {s}</span>
                    {i<5 && <ArrowRight size={10} className="text-slate-300 hidden sm:block"/>}
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-2xl overflow-hidden h-44">
              <img src={LAB} alt="Research methodology" className="w-full h-full object-cover"/>
            </div>
          </div>
        </motion.div>

        {/* Timeline */}
        <div className="mb-20">
          <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Research Timeline</h2>
          <div className="space-y-4">
            {PHASES.map((p,i)=>(
              <motion.div key={p.n}
                initial={{opacity:0,x:-28}} whileInView={{opacity:1,x:0}}
                viewport={{once:true}} transition={{delay:i*0.1}}
                className="card-flat p-6"
              >
                <div className="flex items-start gap-5">
                  <div className="flex-shrink-0">
                    <div className="w-12 h-12 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center">
                      <p.icon size={20} className="text-slate-600"/>
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="flex flex-wrap items-center gap-3 mb-3">
                      <span className={`pill ${p.pill}`}>Phase {p.n}</span>
                      <span className="pill pill-slate">{p.dur}</span>
                    </div>
                    <h3 className="font-display text-lg font-700 text-slate-900 mb-4">{p.title}</h3>
                    <ul className="grid sm:grid-cols-2 gap-y-2 gap-x-4">
                      {p.tasks.map((t,j)=>(
                        <li key={j} className="flex items-start gap-2 text-sm text-slate-500">
                          <ArrowRight size={13} className="text-brand-400 flex-shrink-0 mt-0.5"/>
                          {t}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Datasets */}
        <div className="mb-20">
          <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Datasets & Data Sources</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {DATASETS.map((d,i)=>(
              <motion.div key={d.name}
                initial={{opacity:0,y:28}} whileInView={{opacity:1,y:0}}
                viewport={{once:true}} transition={{delay:i*0.08}}
                className="card p-5"
              >
                <span className={`pill ${d.pill} mb-3 inline-block`}>{d.type}</span>
                <h3 className={`font-display text-lg font-700 ${d.accent} mb-2`}>{d.name}</h3>
                <p className="text-xs text-slate-500 mb-1">{d.size}</p>
                <p className="text-xs text-slate-400 mb-3">{d.notes}</p>
                <div className="pt-3 border-t border-slate-100 text-xs text-slate-500">
                  <span className="text-slate-400">Used for: </span>{d.use}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Evaluation */}
        <div className="mb-20">
          <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Evaluation Strategy</h2>
          <div className="grid md:grid-cols-2 gap-5">
            {[
              { title:'Offline Model Validation', icon:BarChart3,
                items:['Leave-one-subject-out CV (M1 + M2)','10,000-episode meta-test (TC-WPN M4)','5-fold stratified CV (NHANES classifier M3)','Ablation: TC-WPN (λ,β sweep) + GATv2 edge removal','Bootstrap CI + Bonferroni-corrected paired t-tests'] },
              { title:'Clinical Validation', icon:CheckCircle2,
                items:['NHSL 10-shot & 20-shot few-shot adaptation','GAD-7 score Pearson correlation (M2)','Clinician review of XAI attention explanations','≥30 real clinical notes pilot deployment (M4)','Dual-dataset external validation: StudentLife + Primary'] },
              { title:'System Evaluation', icon:Layers,
                items:['API latency: p95 <150ms (M4), <3s (M3)','3-persona end-to-end scenario validation','Escalation F1 on 30 constructed test scenarios','Clinician dashboard usability review','Safety alert latency target: <30 seconds'] },
              { title:'Performance Targets', icon:BarChart3,
                items:['TC-WPN F1 ≥0.75 @ 10-shot, ≥0.78 @ 20-shot','GATv2 AUC-ROC ≥0.80 (internal) & ≥0.70 (external)','LSTM-AE AUROC ≥0.80, EWT 5-10 min','KNN recommendation accuracy 55-70%','Expected Calibration Error <0.08 (TC-WPN)'] },
            ].map((s,i)=>(
              <motion.div key={s.title}
                initial={{opacity:0,y:24}} whileInView={{opacity:1,y:0}}
                viewport={{once:true}} transition={{delay:i*0.1}}
                className="card-flat p-6"
              >
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-9 h-9 rounded-xl bg-brand-50 flex items-center justify-center">
                    <s.icon size={16} className="text-brand-600"/>
                  </div>
                  <h3 className="font-display text-base font-700 text-slate-900">{s.title}</h3>
                </div>
                <ul className="space-y-2">
                  {s.items.map((it,j)=>(
                    <li key={j} className="flex items-start gap-2 text-sm text-slate-500">
                      <CheckCircle2 size={13} className="text-teal-500 flex-shrink-0 mt-0.5"/>
                      {it}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Ethics */}
        <div>
          <h2 className="font-display text-3xl font-800 text-slate-900 mb-8">Ethical Considerations</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {[
              { title:'IRB Approval', desc:'Dual ethics approval from NHSL Ethics Review Committee and SLIIT Research Ethics Committee required before any patient data collection commences.' },
              { title:'Data Minimization', desc:'GPS stored as cluster IDs only. Communication as aggregate counts only. No PII stored in ML models. Participant UUIDs only — no names, no contact details.' },
              { title:'MIMIC-IV Governance', desc:'Accessed under signed PhysioNet DUA following CITI certification. No re-identification attempts. PHI placeholders normalized to [REDACTED].' },
              { title:'Clinical Safety', desc:'Positioned as clinical decision support, not autonomous diagnosis. All High-risk predictions trigger mandatory clinician review. GAD-7 ≥15 triggers support referral.' },
            ].map((e,i)=>(
              <motion.div key={e.title}
                initial={{opacity:0,y:20}} whileInView={{opacity:1,y:0}}
                viewport={{once:true}} transition={{delay:i*0.1}}
                className="flex items-start gap-4 card-flat p-6"
              >
                <div className="w-10 h-10 rounded-xl bg-green-50 border border-green-200 flex items-center justify-center flex-shrink-0">
                  <Shield size={17} className="text-green-600"/>
                </div>
                <div>
                  <h3 className="font-display text-sm font-700 text-slate-900 mb-1.5">{e.title}</h3>
                  <p className="text-sm text-slate-500 leading-relaxed">{e.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
