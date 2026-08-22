# GLOBEM data

The final Component 2 pipeline uses the **GLOBEM multi-year dataset** rather
than StudentLife.

Expected structure:

```text
GLOBEM_ROOT/
├── INS-W_1/
│   ├── FeatureData/
│   │   └── rapids.csv
│   └── SurveyData/
│       └── dep_weekly.csv
├── INS-W_2/
├── INS-W_3/
└── INS-W_4/
```

Required fields include:

- feature file: `pid`, `date`, RAPIDS `f_...` columns
- label file: `pid`, `date`, `anx_weekly_subscale`

Set the path with:

```bash
export GLOBEM_ROOT=/path/to/globem
```

Do not commit the raw dataset to GitHub.
