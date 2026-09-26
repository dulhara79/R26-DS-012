export const findings = [
  {
    component: "C2",
    metric: "Held-out AUROC",
    value: "0.5205",
    context: "GLOBEM · GATv2 · participant-grouped held-out evaluation",
    interpretation:
      "The graph model was not distinguishable from chance under the final leakage-free evaluation.",
  },
  {
    component: "C2",
    metric: "95% participant-clustered CI",
    value: "0.485–0.560",
    context: "Participant-clustered interval",
    interpretation: "The interval overlaps chance-level discrimination.",
  },
  {
    component: "C2",
    metric: "Permutation null mean",
    value: "0.4991",
    context: "50 permutations · empirical p=0.255",
    interpretation:
      "The observed held-out behavior did not provide evidence of reliable discrimination.",
  },
  {
    component: "C2",
    metric: "Active fusion weight",
    value: "0.0",
    context: "Current validation gate",
    interpretation:
      "C2 is excluded from active fusion under the current evidence rule.",
  },
  {
    component: "C3",
    metric: "Held-out AUROC",
    value: "≈0.738",
    context: "Deployment-relevant patient-disjoint clinical-note setting",
    interpretation:
      "Repository-supported clinical NLP benchmark; this is a Clinical NLP signal, not overall patient risk.",
  },
];
