# Legacy implementation notice

Earlier versions of Component 2 used StudentLife and included:

- raw GPS cleaning
- DBSCAN stay-point clustering
- contextual location × time × activity states
- stress-EMA-derived vulnerability labels
- hourly risk profiles
- dual-head vulnerability + risk-window prediction
- K-Means phenotyping
- per-user risk-level inference
- ordinary sample-level stratified CV / SMOTE experiments

Those experiments document the evolution of the project, but they are **not**
the final validated method.

The final method is the GLOBEM consolidated v8 notebook:

`notebooks/component2_consolidated_v8.ipynb`

Final held-out result:

- GATv2 AUROC = 0.5205
- 95% CI = 0.485–0.560
- permutation p = 0.255
- deployment fusion weight = 0.0
