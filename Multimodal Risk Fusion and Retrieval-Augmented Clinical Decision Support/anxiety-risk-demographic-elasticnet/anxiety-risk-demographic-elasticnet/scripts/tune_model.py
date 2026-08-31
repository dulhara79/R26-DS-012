from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxiety_risk.data import FEATURE_COLUMNS, build_master_dataset  # noqa: E402
from anxiety_risk.model import build_elasticnet_pipeline  # noqa: E402

RANDOM_STATE = 42


def main() -> None:
    demo = pd.read_csv(ROOT / "data/raw/demographic.csv")
    gad = pd.read_csv(ROOT / "data/raw/gad7.csv")
    master = build_master_dataset(demo, gad, min_age=15, max_age=65)
    X = master[FEATURE_COLUMNS]
    y = master["anxiety_positive"].astype(int)
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = build_elasticnet_pipeline(class_weight=None, random_state=RANDOM_STATE, max_iter=2000)
    grid = GridSearchCV(
        base,
        {
            "classifier__C": [0.2, 1.0],
            "classifier__l1_ratio": [0.5, 0.75, 1.0],
        },
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=1,
    )
    grid.fit(X_dev, y_dev)

    out_dir = ROOT / "reports/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")[[
        "rank_test_score", "mean_test_score", "std_test_score",
        "param_classifier__C", "param_classifier__l1_ratio"
    ]]
    results.to_csv(out_dir / "elasticnet_grid_search.csv", index=False)
    payload = {
        "C": float(grid.best_params_["classifier__C"]),
        "l1_ratio": float(grid.best_params_["classifier__l1_ratio"]),
        "cv_pr_auc": float(grid.best_score_),
    }
    (out_dir / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
