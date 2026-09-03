export const components = [
  { id:'C1', owner:'Sendanayake H.D.', title:'Wearable physiological forecasting', question:'Can individualized physiological deviation provide useful short-horizon evidence of escalation?', method:'Self-supervised LSTM autoencoder over wearable physiological windows, followed by short-horizon forecasting.', data:'WESAD · AffectiveROAD · PPG-DaLiA · EmoWear', status:'Active research signal', evidence:'Subject-level evaluation and personalization experiments.' },
  { id:'C2', owner:'Layathma B.M.A.S.', title:'Leakage-free behavioural graph evaluation', question:'Do temporal behavioural graphs generalize beyond simpler passive-sensing baselines?', method:'GATv2 on 28-day GLOBEM graphs with participant-grouped, cross-cohort evaluation.', data:'GLOBEM · INS-W_1–4', status:'Active fusion weight 0.0', evidence:'AUROC 0.5205 · 95% CI 0.485–0.560 · null 0.4991 · p=0.255', limitation:'Final held-out result was not distinguishable from chance; the graph model did not outperform simpler baselines.' },
  { id:'C3', owner:'Kaushalya I.G.D.', title:'Patient-disjoint clinical NLP / TC-WPN', question:'Can few-shot clinical NLP learn from small support sets without patient leakage or label contamination?', method:'Bio_ClinicalBERT → 256-d projection → Temporal-Consistency Weighted Prototypical Network.', data:'MIMIC-IV · MIMIC-III transfer', status:'Deployment-relevant held-out AUROC ≈0.738', evidence:'Patient-disjoint episodes · leakage certificates · frozen episode plans.' },
  { id:'C4', owner:'Seneviratne K.A.U.A.', title:'Reliability-weighted fusion + CARE-AnxRAG', question:'How should heterogeneous evidence be combined when modalities differ in recency, reliability and validation quality?', method:'Contextual prior + eligible modality outputs → reliability-weighted fusion → evidence-aware retrieval and abstention.', data:'Component outputs · contextual variables · evidence corpus', status:'Research decision-support layer', evidence:'Missing/unreliable modalities can be masked; insufficient evidence can abstain.' },
]

export const evidence = [
  { component:'C2', metric:'Held-out AUROC', value:'0.5205', context:'GLOBEM / GATv2', note:'Participant-grouped external evaluation.' },
  { component:'C2', metric:'95% clustered CI', value:'0.485–0.560', context:'Participant-clustered interval', note:'Interval overlaps chance-level discrimination.' },
  { component:'C2', metric:'Permutation null', value:'0.4991', context:'50 permutations', note:'Empirical p-value 0.255.' },
  { component:'C2', metric:'Active fusion weight', value:'0.0', context:'Current evidence gate', note:'Excluded from active fusion under the current validation rule.' },
  { component:'C3', metric:'Held-out AUROC', value:'≈0.738', context:'Deployment-relevant clinical-note setting', note:'Patient-disjoint few-shot benchmark.' },
]

export const people = [
  ['Sendanayake H.D.','IT22107596','C1 · Wearable physiological forecasting'],
  ['Layathma B.M.A.S.','IT22171542','C2 · Behavioural graph learning'],
  ['Kaushalya I.G.D.','IT22130648','C3 · Clinical NLP / TC-WPN'],
  ['Seneviratne K.A.U.A.','IT22093950','C4 · Fusion + CARE-AnxRAG'],
]

export const supervisors = [
  ['Prof. Samantha Thelijjagoda','Research Supervisor','SLIIT'],
  ['Dr. Mahima Weerasinghe','Co-Supervisor','SLIIT'],
  ['Dr. Chathurie Suraweera','External / Clinical Supervisor','University of Colombo / NHSL'],
]
