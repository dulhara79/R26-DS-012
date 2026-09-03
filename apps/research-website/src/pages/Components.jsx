import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Network, Cpu, Brain, ChevronDown, ChevronUp, ArrowRight, Code, Database, Zap, Target, CheckCircle2 } from 'lucide-react'

const IMGS = {
  M1: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=1200&q=80&auto=format&fit=crop',
  M2: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&q=80&auto=format&fit=crop',
  M3: 'https://images.unsplash.com/photo-1576671081837-49000212a370?w=1200&q=80&auto=format&fit=crop',
  M4: 'https://images.unsplash.com/photo-1692607431225-5f4564c8f132?q=80&w=1332&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D',
}

const DATA = [
  {
    id:'M1', code:'IT22107596', name:'Sendanayake H.D.',
    shortTitle:'Wearable Biosensor Forecasting',
    fullTitle:'Personalized Self-Supervised Forecasting of Acute Anxiety Episodes Using Wearable Biosensors',
    icon: Activity,
    pill:'pill-teal', accent:'text-teal-600', bg:'bg-teal-50', border:'border-teal-200',
    objective:'Design and validate a self-supervised anomaly detection framework for forecasting acute autonomic destabilization events 5–10 minutes in advance using continuous multimodal wearable physiological data.',
    gap:'Current wearable systems rely on supervised ML requiring rare labeled panic events or generic population heuristics — providing post-onset detection rather than proactive forecasting.',
    novelty:'First self-supervised anomaly detection framework targeting short-horizon acute autonomic escalation forecasting. Eliminates the labeled-data bottleneck entirely using reconstruction error as a personalized biomarker.',
    algo:[
      { k:'Architecture', v:'LSTM-Based Autoencoder (temporal representation learning)' },
      { k:'Training', v:'Self-supervised on non-escalation baseline data (no labels needed)' },
      { k:'Anomaly Score', v:'Reconstruction error vs adaptive per-subject percentile threshold' },
      { k:'Forecasting', v:'Seq2seq LSTM over anomaly trajectories → risk probability [0,1]' },
      { k:'Personalization', v:'Per-subject baseline normalization; fine-tuned on individual data' },
    ],
    hardware:['ESP32-C3 DevKit (MCU + WiFi/BT)', 'AD8232 ECG Module', 'BF350-3AA Thoracic Strain Gauge', 'BMI160 6-axis IMU', 'DS18B20 Skin Temperature'],
    datasets:['WESAD','SWELL-KW','DREAMER','DEAP','CASE','AffectiveROAD','Pilot (10-20 participants)'],
    metrics:[
      { l:'AUROC Target', v:'≥0.80' },
      { l:'EWT Lead Time', v:'5–10 min' },
      { l:'Edge Latency', v:'<2 sec' },
      { l:'Battery Life', v:'≥8 hrs' },
    ],
    tech:['PyTorch 2.0','LSTM-AE','ESP32-C3','Firebase','Python 3.10'],
  },
  {
    id:'M2', code:'IT22171542', name:'Layathma B.M.A.S.',
    shortTitle:'Temporal Behavioral Graph Framework',
    fullTitle:'Graph-Based Spatio-Temporal Behavioral Phenotyping for Personalized Anxiety Vulnerability Mapping',
    icon: Network,
    pill:'pill-blue', accent:'text-brand-600', bg:'bg-brand-50', border:'border-brand-200',
    objective:'Model passively collected smartphone sensing data as temporal behavioral graphs and apply GATv2 to generate personalized vulnerability scores and interpretable high-risk daily time-window identifications.',
    gap:'76% of passive sensing studies use flat feature vectors. Only 1/42 reviewed studies provided external validation. No prior study applied GNNs to smartphone behavioral data for anxiety.',
    novelty:'First application of Graph Attention Networks to temporal behavioral graphs from smartphone passive sensing data for anxiety vulnerability detection. Novel 3-edge-type construction schema.',
    algo:[
      { k:'Graph Nodes', v:'168 nodes = 42 days × 4 circadian windows (morning/afternoon/evening/night)' },
      { k:'Node Features', v:'10-dimensional behavioral feature vector per node' },
      { k:'Edge Type 1', v:'Sequential: Morning→Afternoon→Evening→Night (weight=1.0)' },
      { k:'Edge Type 2', v:'Cross-day: same window across consecutive days (exp decay weight)' },
      { k:'Edge Type 3', v:'Similarity: cosine similarity > 0.85 between any two nodes' },
      { k:'Model', v:'GATv2 (Brody et al., 2022) — 2 layers + Global Attention Pooling + MLP' },
    ],
    phenotypes:['Phenotype A — Social-Spatial Withdrawal','Phenotype B — Circadian Disruption','Phenotype C — Hypervigilant Mobility'],
    datasets:['StudentLife (48 participants, 10 weeks, Dartmouth)','Primary dataset (10-20 Sri Lankan students, 4-6 weeks)'],
    metrics:[
      { l:'AUC-ROC', v:'≥0.80' },
      { l:'Pearson r (GAD-7)', v:'≥0.60' },
      { l:'Macro F1', v:'≥0.75' },
      { l:'Silhouette Score', v:'≥0.50' },
    ],
    tech:['PyTorch Geometric','GATv2','NetworkX','Android (API 26+)','Firebase'],
  },
  {
    id:'M3', code:'IT22093950', name:'Seneviratne K.A.U.A.',
    shortTitle:'Adaptive Intervention Engine',
    fullTitle:'A Continuously Learning Personalized Anxiety Intervention Framework Using Multimodal Risk Signals',
    icon: Cpu,
    pill:'pill-violet', accent:'text-violet-600', bg:'bg-violet-50', border:'border-violet-200',
    objective:'Transform multimodal risk scores and demographic-clinical context into adaptive, safe, explainable intervention recommendations with clinician oversight built into the escalation pathway.',
    gap:'Most mHealth apps deliver static one-size-fits-all content and do not learn from whether interventions actually reduced distress. No system combines KNN, continuous learning, panic response, and clinician alerting.',
    novelty:'First integration of NHANES demographic pre-training, KNN case-based reasoning, composite reward continuous learning, panic response, and clinician-in-the-loop oversight in one framework.',
    algo:[
      { k:'Risk Classifier', v:'Gradient Boosting (n=200, depth=4, lr=0.05) on 13-dim feature vector' },
      { k:'KNN Engine', v:'k=5, cosine similarity, BallTree (O(log n)) with NHANES pre-training' },
      { k:'Tier A (Acute)', v:'Box Breathing, 5-4-3-2-1 Grounding, Abbreviated PMR, Affirmations' },
      { k:'Tier B (Plans)', v:'CBT Journal, MBSR, PMR Program, Behavioral Activation, Social Plan' },
      { k:'Reward Function', v:'R = 0.35×f(ΔHR) + 0.30×rating + 0.20×completion + 0.15×f(ΔRisk)' },
      { k:'Learning Update', v:'BallTree refit every 50 new episodes; continuous online adaptation' },
    ],
    datasets:['NHANES 2017-March 2020 (4,500-6,000 records, ages 18-35)','NHSL de-identified data (30-50 patients, IRB-approved)'],
    metrics:[
      { l:'Classifier Accuracy', v:'80-88%' },
      { l:'KNN Accuracy', v:'55-70%' },
      { l:'Escalation F1', v:'0.78-0.88' },
      { l:'API Response', v:'<3 sec' },
    ],
    tech:['scikit-learn','KNN BallTree','FastAPI','Flutter','Firebase','SMOTE'],
  },
  {
    id:'M4', code:'IT22130648', name:'Kaushalya I.G.D.',
    shortTitle:'Clinical NLP — TC-WPN',
    fullTitle:'Temporal-Confidence Weighted Prototypical Networks for Few-Shot Clinical Anxiety Detection',
    icon: Brain,
    pill:'pill-rose', accent:'text-rose-600', bg:'bg-rose-50', border:'border-rose-200',
    objective:'Achieve ≥75% F1-score with only 10–20 labeled clinical note examples — a 50–100× reduction in labeled-data requirements vs traditional supervised learning approaches.',
    gap:'Existing AI diagnosis requires 1,000-5,000 labeled clinical notes. Hospitals where experts annotate only 10-20 examples face an insurmountable barrier to AI deployment.',
    novelty:'TC-WPN integrates temporal recency decay and entropy-based confidence weighting into prototypical network prototype formation — novel contributions absent from any published few-shot meta-learning framework.',
    algo:[
      { k:'Backbone', v:'Bio_ClinicalBERT (12-layer, 768-dim, pre-trained on MIMIC-III)' },
      { k:'Projection', v:'768 → 256 dims + L2 normalization (trained end-to-end)' },
      { k:'Temporal Weight', v:'w_recency = exp(−λ×Δt/365), λ=0.5 · w_regularity ≥3 visits=1.0' },
      { k:'Confidence Weight', v:'w_confidence = 1/(1+β×H(x)), β=1.0, H=Shannon entropy' },
      { k:'TC Prototype', v:'Weighted mean of support embeddings by both temporal & confidence weights' },
      { k:'Training', v:'10,000 N=2-way K-shot episodes on MIMIC-IV, AdamW lr=1e-5' },
    ],
    datasets:['MIMIC-IV (8,500 anxiety+ · 15,000 non-anxiety psychiatric notes)','NHSL (15-25 psychiatrist-annotated de-identified notes)'],
    metrics:[
      { l:'F1 @ 10-shot', v:'≥0.75' },
      { l:'F1 @ 20-shot', v:'≥0.78' },
      { l:'ECE', v:'<0.08' },
      { l:'Inference', v:'<150ms' },
    ],
    tech:['ClinicalBERT','PyTorch 2.0','HuggingFace','FastAPI','MIMIC-IV'],
  },
]

function Card({ c, expanded, toggle }) {
  return (
    <motion.div layout className={`bg-white rounded-3xl border ${c.border} shadow-card overflow-hidden`}>
      {/* Image header */}
      <div className="relative h-48 overflow-hidden cursor-pointer" onClick={toggle}>
        <img src={IMGS[c.id]} alt={c.shortTitle} className="w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/20 to-transparent" />
        <div className="absolute top-4 left-4 flex items-center gap-2">
          <span className={`pill ${c.pill}`}>{c.id}</span>
        </div>
        <div className="absolute bottom-4 left-4 right-12">
          <div className="font-mono text-xs text-white/60 mb-1">{c.code} · {c.name}</div>
          <h3 className="font-display text-xl font-800 text-white leading-tight">{c.shortTitle}</h3>
        </div>
        <button onClick={toggle} className="absolute bottom-4 right-4 w-9 h-9 rounded-full bg-white/20 backdrop-blur flex items-center justify-center text-white">
          {expanded ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
        </button>
      </div>

      {/* Metrics bar */}
      <div className="grid grid-cols-4 border-b border-slate-100">
        {c.metrics.map(m => (
          <div key={m.l} className={`p-4 border-r border-slate-100 last:border-0 text-center`}>
            <div className={`font-mono text-lg font-700 ${c.accent}`}>{m.v}</div>
            <div className="text-[10px] text-slate-400 mt-0.5 leading-tight">{m.l}</div>
          </div>
        ))}
      </div>

      {/* Expand */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity:0, height:0 }} animate={{ opacity:1, height:'auto' }}
            exit={{ opacity:0, height:0 }} transition={{ duration:0.4, ease:[0.16,1,0.3,1] }}
          >
            <div className="p-7 space-y-7">
              {/* Objective */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Target size={14} className={c.accent} />
                  <span className={`text-xs font-700 font-mono ${c.accent} uppercase tracking-wide`}>Objective</span>
                </div>
                <p className={`text-sm leading-relaxed text-slate-600 bg-slate-50 rounded-2xl p-4 border border-slate-100`}>{c.objective}</p>
              </div>

              {/* Gap / Novelty */}
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Zap size={14} className="text-amber-500" />
                    <span className="text-xs font-700 font-mono text-amber-600 uppercase tracking-wide">Research Gap</span>
                  </div>
                  <p className="text-sm text-slate-500 leading-relaxed">{c.gap}</p>
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 size={14} className={c.accent} />
                    <span className={`text-xs font-700 font-mono ${c.accent} uppercase tracking-wide`}>Novelty</span>
                  </div>
                  <p className="text-sm text-slate-500 leading-relaxed">{c.novelty}</p>
                </div>
              </div>

              {/* Algorithm */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Code size={14} className="text-slate-500" />
                  <span className="text-xs font-700 font-mono text-slate-500 uppercase tracking-wide">Algorithm Details</span>
                </div>
                <div className="grid sm:grid-cols-2 gap-2">
                  {c.algo.map(a => (
                    <div key={a.k} className="bg-slate-50 rounded-xl p-3 border border-slate-100">
                      <div className="font-mono text-[10px] text-slate-400 uppercase tracking-wide mb-1">{a.k}</div>
                      <div className="text-sm text-slate-700 leading-snug">{a.v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hardware (M1) */}
              {c.hardware && (
                <div>
                  <span className="text-xs font-700 font-mono text-slate-500 uppercase tracking-wide mb-3 block">Hardware Components</span>
                  <div className="flex flex-wrap gap-2">
                    {c.hardware.map(h => (
                      <span key={h} className={`pill ${c.pill}`}>{h}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Phenotypes (M2) */}
              {c.phenotypes && (
                <div>
                  <span className="text-xs font-700 font-mono text-slate-500 uppercase tracking-wide mb-3 block">Behavioral Phenotypes</span>
                  <div className="flex flex-wrap gap-2">
                    {c.phenotypes.map(p => (
                      <span key={p} className={`pill ${c.pill}`}>{p}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Datasets + Tech */}
              <div className="grid md:grid-cols-2 gap-5">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Database size={14} className="text-slate-400" />
                    <span className="text-xs font-700 font-mono text-slate-400 uppercase tracking-wide">Datasets</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {c.datasets.map(d => <span key={d} className={`pill ${c.pill} text-[10px]`}>{d}</span>)}
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Code size={14} className="text-slate-400" />
                    <span className="text-xs font-700 font-mono text-slate-400 uppercase tracking-wide">Tech Stack</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {c.tech.map(t => <span key={t} className="pill pill-slate font-mono text-[10px]">{t}</span>)}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function Components() {
  const [expanded, setExpanded] = useState(null)
  return (
    <div className="min-h-screen pt-24 pb-32 px-6">
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <motion.div initial={{opacity:0,y:28}} animate={{opacity:1,y:0}} className="mb-16">
          <span className="pill pill-blue mb-5 inline-block">Research Components</span>
          <h1 className="font-display text-5xl md:text-6xl font-800 text-slate-900 mb-5">
            Four Integrated<br /><span className="grad-blue">Subsystems</span>
          </h1>
          <p className="text-slate-500 text-lg max-w-xl">
            Each component addresses a distinct modality. Together they form a late-fusion multimodal framework. Click any component to expand the full technical detail.
          </p>
        </motion.div>

        {/* Fusion summary */}
        <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{delay:0.15}}
          className="card-flat mb-10 p-6">
          <div className="flex flex-col md:flex-row items-center gap-3 justify-center">
            {DATA.map((c,i)=>(
              <div key={c.id} className="flex items-center gap-3">
                <button onClick={()=>setExpanded(expanded===c.id?null:c.id)}
                  className={`${c.bg} border ${c.border} rounded-2xl px-5 py-2.5 text-center hover:shadow-md transition-all`}>
                  <c.icon size={18} className={`${c.accent} mx-auto mb-0.5`}/>
                  <div className={`font-mono text-xs font-700 ${c.accent}`}>{c.id}</div>
                </button>
                {i<DATA.length-1 && <ArrowRight size={14} className="text-slate-300 hidden md:block"/>}
              </div>
            ))}
            <ArrowRight size={14} className="text-slate-300 hidden md:block"/>
            <div className="bg-gradient-to-br from-brand-500 to-violet-600 rounded-2xl px-5 py-2.5 text-center">
              <div className="text-xs font-700 text-white font-mono">FUSION</div>
              <div className="text-[10px] text-blue-200">Risk Score + XAI</div>
            </div>
          </div>
          <p className="text-center font-mono text-xs text-slate-400 mt-3">w₁=0.25 · w₂=0.20 · w₃=0.15 · w₄=0.40</p>
        </motion.div>

        <div className="space-y-5">
          {DATA.map((c,i)=>(
            <motion.div key={c.id} initial={{opacity:0,y:32}} animate={{opacity:1,y:0}} transition={{delay:0.1+i*0.1}}>
              <Card c={c} expanded={expanded===c.id} toggle={()=>setExpanded(expanded===c.id?null:c.id)}/>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  )
}
