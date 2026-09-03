import { motion } from 'framer-motion'
import { BookOpen, ExternalLink, Github, FileText, Database, Globe, ArrowRight, Award } from 'lucide-react'

const PUB_BG = 'https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=1400&q=80&auto=format&fit=crop'

const VENUES = [
  { name:'IEEE Journal of Biomedical and Health Informatics', abbr:'IEEE JBHI', type:'Journal', tier:'Q1 SCI', pill:'pill-blue' },
  { name:'Frontiers in Digital Health', abbr:'Front. Digit. Health', type:'Journal', tier:'Open Access', pill:'pill-teal' },
  { name:'IEEE Access', abbr:'IEEE Access', type:'Journal', tier:'Open Access', pill:'pill-slate' },
  { name:'ACM CHI — Human Factors in Computing Systems', abbr:'ACM CHI', type:'Conference', tier:'A*', pill:'pill-violet' },
  { name:'JMIR Mental Health', abbr:'JMIR MH', type:'Journal', tier:'Open Access', pill:'pill-green' },
]

const RESOURCES = [
  { name:'Research Repository', url:'https://github.com/dulhara79/R26-DS-012', desc:'Full codebase — all four component implementations, datasets, and documentation.', icon:Github, pill:'pill-slate', label:'GitHub' },
  { name:'MIMIC-IV Dataset', url:'https://physionet.org/content/mimiciv/3.1/', desc:'Freely accessible EHR dataset. Access requires PhysioNet DUA + CITI certification.', icon:Database, pill:'pill-rose', label:'PhysioNet' },
  { name:'StudentLife Dataset', url:'http://studentlife.cs.dartmouth.edu', desc:'48 students, 10 weeks passive smartphone sensing + mental health assessments. Dartmouth.', icon:Database, pill:'pill-blue', label:'Dartmouth' },
  { name:'NHANES 2017-2020', url:'https://wwwn.cdc.gov/nchs/nhanes/', desc:'National Health and Nutrition Examination Survey. Open access under CDC data terms.', icon:Globe, pill:'pill-green', label:'CDC' },
]

const KEY_REFS = [
  { comp:'M1 — Wearable Biosensors', pill:'pill-teal', refs:[
    { id:'[14]', text:'Schmidt et al. Introducing WESAD, a Multimodal Dataset for Wearable Stress Detection. ACM ICMI, 2018.' },
    { id:'[8]',  text:'Jacobson & Bhattacharya. Digital biomarkers of anxiety via personalized deep learning. BRT, 2022.' },
    { id:'[5]',  text:'Zhang et al. An Explainable Anomaly Detection Framework for Monitoring Depression. arXiv, 2025.' },
    { id:'[7]',  text:'Pinge et al. Detection and monitoring of stress using wearables: systematic review. Front. CS, 2024.' },
  ]},
  { comp:'M2 — Behavioral Graph Framework', pill:'pill-blue', refs:[
    { id:'[6]',  text:'Veličković et al. Graph Attention Networks. ICLR, 2018.' },
    { id:'[7]',  text:'Brody, Alon, Yahav. How attentive are graph attention networks? ICLR, 2022.' },
    { id:'[1]',  text:'Wang et al. StudentLife: Assessing mental health and behavioral trends. UbiComp, 2014.' },
    { id:'[5]',  text:'Stamatis et al. Differential temporal associations of smartphone features with GAD-7. JAD, 2021.' },
  ]},
  { comp:'M3 — Adaptive Intervention Engine', pill:'pill-violet', refs:[
    { id:'[21]', text:'Groh et al. Can a Recommender System Support Treatment Personalization in Digital Mental Health? CHI, 2022.' },
    { id:'[26]', text:'Aguilera et al. Effectiveness of a Digital Health Intervention for Depression and Anxiety. JAMA IM, 2021.' },
    { id:'[28]', text:'Nahum-Shani et al. Just-in-Time Adaptive Interventions in Mobile Health. Annals Behav. Med., 2018.' },
    { id:'[29]', text:'NCHS. NHANES 2017-March 2020 Pre-Pandemic Data Files. CDC, 2021.' },
  ]},
  { comp:'M4 — Clinical NLP (TC-WPN)', pill:'pill-rose', refs:[
    { id:'[3]',  text:'Snell, Swersky, Zemel. Prototypical Networks for Few-Shot Learning. NeurIPS, 2017.' },
    { id:'[14]', text:'Alsentzer et al. Publicly Available Clinical BERT Embeddings. Clin. NLP Workshop, 2019.' },
    { id:'[15]', text:'Johnson et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci. Data, 2023.' },
    { id:'[11]', text:'Finn, Abbeel, Levine. Model-Agnostic Meta-Learning for Fast Adaptation. ICML, 2017.' },
  ]},
]

export default function Publications() {
  return (
    <div className="min-h-screen pb-32">

      {/* Header */}
      <div className="relative h-72 overflow-hidden">
        <img src={PUB_BG} alt="Library research" className="w-full h-full object-cover" style={{filter:'brightness(0.4) saturate(0.7)'}}/>
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-slate-50"/>
        <div className="absolute inset-0 flex items-end pb-12 px-6">
          <div className="max-w-5xl mx-auto w-full">
            <motion.div initial={{opacity:0,y:24}} animate={{opacity:1,y:0}} transition={{delay:0.1}}>
              <span className="pill pill-rose mb-4 inline-block">Publications & References</span>
              <h1 className="font-display text-5xl md:text-6xl font-800 text-white">Research Literature</h1>
            </motion.div>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 pt-12">

        {/* Venues */}
        <motion.div initial={{opacity:0,y:24}} animate={{opacity:1,y:0}} transition={{delay:0.15}} className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <Award size={20} className="text-brand-500"/>
            <h2 className="font-display text-2xl font-800 text-slate-900">Target Publication Venues</h2>
          </div>
          <div className="space-y-3">
            {VENUES.map((v,i)=>(
              <motion.div key={v.abbr}
                initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} transition={{delay:0.1+i*0.07}}
                className="card-flat px-5 py-4 flex items-center justify-between gap-4"
              >
                <div className="flex items-center gap-4">
                  <FileText size={16} className="text-slate-400 flex-shrink-0"/>
                  <div>
                    <div className="text-sm font-600 text-slate-800">{v.name}</div>
                    <div className="font-mono text-xs text-slate-400 mt-0.5">{v.abbr}</div>
                  </div>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className={`pill ${v.pill}`}>{v.type}</span>
                  <span className="pill pill-slate">{v.tier}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Open resources */}
        <div className="mb-16">
          <h2 className="font-display text-2xl font-800 text-slate-900 mb-6">Open Source & Data Resources</h2>
          <div className="grid md:grid-cols-2 gap-4">
            {RESOURCES.map((r,i)=>(
              <motion.a key={r.name} href={r.url} target="_blank" rel="noopener noreferrer"
                initial={{opacity:0,y:20}} whileInView={{opacity:1,y:0}}
                viewport={{once:true}} transition={{delay:i*0.08}}
                className="card group p-5 block"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <r.icon size={17} className="text-slate-500"/>
                    <span className="font-display text-sm font-700 text-slate-800">{r.name}</span>
                  </div>
                  <ExternalLink size={13} className="text-slate-300 group-hover:text-slate-500 transition-colors flex-shrink-0"/>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed mb-3">{r.desc}</p>
                <span className={`pill ${r.pill} text-[10px]`}>{r.label}</span>
              </motion.a>
            ))}
          </div>
        </div>

        {/* Key refs */}
        <div>
          <h2 className="font-display text-2xl font-800 text-slate-900 mb-6">Key References by Component</h2>
          <div className="grid md:grid-cols-2 gap-5">
            {KEY_REFS.map((section,i)=>(
              <motion.div key={section.comp}
                initial={{opacity:0,y:24}} whileInView={{opacity:1,y:0}}
                viewport={{once:true}} transition={{delay:i*0.1}}
                className="card-flat p-6"
              >
                <span className={`pill ${section.pill} mb-4 inline-block`}>{section.comp}</span>
                <div className="space-y-3">
                  {section.refs.map(r=>(
                    <div key={r.id} className="flex gap-3 text-sm">
                      <span className="font-mono text-xs text-slate-400 flex-shrink-0 mt-0.5 w-8">{r.id}</span>
                      <span className="text-slate-500 leading-relaxed">{r.text}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* CTA */}
        <motion.div initial={{opacity:0,y:28}} whileInView={{opacity:1,y:0}} viewport={{once:true}}
          className="mt-14 card-flat p-8 text-center">
          <Github size={28} className="text-slate-400 mx-auto mb-4"/>
          <h3 className="font-display text-2xl font-800 text-slate-900 mb-3">Full Reference Lists</h3>
          <p className="text-slate-500 text-sm mb-6 max-w-md mx-auto">Complete bibliography for all four components is available in the research repository. Each proposal contains 25–40 references.</p>
          <a href="https://github.com/dulhara79/R26-DS-012" target="_blank" rel="noopener noreferrer"
            className="btn-primary inline-flex items-center gap-2 group">
            <Github size={15}/> View Repository
            <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform"/>
          </a>
        </motion.div>
      </div>
    </div>
  )
}
