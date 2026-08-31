from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxiety_risk.data import FEATURE_COLUMNS, build_master_dataset  # noqa: E402
from anxiety_risk.fusion import probability_to_fusion_score  # noqa: E402
from anxiety_risk.model import build_elasticnet_pipeline  # noqa: E402

RANDOM_STATE = 42
MIN_AGE = 15
MAX_AGE = 65


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def coefficient_table(fitted_pipeline) -> pd.DataFrame:
    pre = fitted_pipeline.named_steps["preprocessor"]
    clf = fitted_pipeline.named_steps["classifier"]
    names = list(pre.get_feature_names_out())
    coefs = clf.coef_.ravel().astype(float)
    age_scale = float(pre.named_transformers_["num"].named_steps["scaler"].scale_[0])
    rows = []
    for name, coef in zip(names, coefs):
        display = name.replace("num__", "").replace("cat__", "")
        if name == "num__age":
            report_coef = coef / age_scale
            reference = "per 1-year increase"
        else:
            report_coef = coef
            if display.startswith("gender_"):
                reference = "vs female"
            elif display.startswith("edu_"):
                reference = "vs bachelor's degree"
            elif display.startswith("smoke_"):
                reference = "vs never smokes"
            elif display.startswith("drink_"):
                reference = "vs never drinks"
            else:
                reference = "reference category"
        rows.append({
            "feature": display,
            "coefficient": float(report_coef),
            "odds_ratio": float(math.exp(report_coef)),
            "reference": reference,
        })
    return pd.DataFrame(rows).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    result_dir = ROOT / "reports/results"
    figure_dir = ROOT / "reports/figures"
    model_dir = ROOT / "models"
    processed_dir = ROOT / "data/processed"
    for d in [result_dir, figure_dir, model_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    best = json.loads((result_dir / "best_params.json").read_text(encoding="utf-8"))
    demographic = pd.read_csv(ROOT / "data/raw/demographic.csv")
    gad7 = pd.read_csv(ROOT / "data/raw/gad7.csv")
    master = build_master_dataset(demographic, gad7, min_age=MIN_AGE, max_age=MAX_AGE)
    master.to_csv(processed_dir / "demographic_gad7_master.csv", index=False)

    X = master[FEATURE_COLUMNS]
    y = master["anxiety_positive"].astype(int)
    ids = master["export_id"].astype(int)
    X_dev, X_test, y_dev, y_test, _, id_test = train_test_split(
        X, y, ids, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    prevalence = float(y_dev.mean())
    common_age = [float(X_dev.age.quantile(0.01)), float(X_dev.age.quantile(0.99))]

    base = build_elasticnet_pipeline(
        C=float(best["C"]),
        l1_ratio=float(best["l1_ratio"]),
        class_weight=None,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    calibrated = CalibratedClassifierCV(estimator=clone(base), method="sigmoid", cv=5, n_jobs=1)
    calibrated.fit(X_dev, y_dev)
    probabilities = calibrated.predict_proba(X_test)[:, 1]
    fusion_scores = np.array([probability_to_fusion_score(float(p), prevalence) for p in probabilities])

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "best_cv_pr_auc": float(best["cv_pr_auc"]),
        "best_params": {"C": float(best["C"]), "l1_ratio": float(best["l1_ratio"]), "class_weight": None},
        "n_total_clean": int(len(master)),
        "n_development": int(len(X_dev)),
        "n_test": int(len(X_test)),
        "development_positive_count": int(y_dev.sum()),
        "test_positive_count": int(y_test.sum()),
        "development_prevalence": prevalence,
        "test_prevalence": float(y_test.mean()),
        "test_probability_min": float(probabilities.min()),
        "test_probability_median": float(np.median(probabilities)),
        "test_probability_max": float(probabilities.max()),
        "test_fusion_score_min": float(fusion_scores.min()),
        "test_fusion_score_median": float(np.median(fusion_scores)),
        "test_fusion_score_max": float(fusion_scores.max()),
        "common_age_range_1st_to_99th_percentile": common_age,
        "calibration": "sigmoid (Platt scaling)",
        "target": "GAD-7 >= 10",
        "features": FEATURE_COLUMNS,
        "isi_used": False,
        "fusion_transform": "sigmoid(logit(probability) - logit(development_prevalence))",
    }
    save_json(result_dir / "final_model_metrics.json", metrics)
    save_json(result_dir / "dataset_summary.json", {
        "raw_rows": int(len(demographic)),
        "clean_rows": int(len(master)),
        "excluded_age_rows": int(len(demographic) - len(master)),
        "positive_count": int(y.sum()),
        "positive_prevalence": float(y.mean()),
        "features": FEATURE_COLUMNS,
        "target": "GAD-7 >= 10",
        "isi_used": False,
    })

    coefficient_model = clone(base).fit(X_dev, y_dev)
    coefficient_table(coefficient_model).to_csv(result_dir / "elasticnet_coefficients.csv", index=False)

    test_out = X_test.reset_index(drop=True).copy()
    test_out.insert(0, "export_id", id_test.reset_index(drop=True))
    test_out["actual_gad7_positive"] = y_test.reset_index(drop=True)
    test_out["demographic_probability"] = probabilities
    test_out["demographic_probability_percent"] = probabilities * 100
    test_out["demographic_fusion_score"] = fusion_scores
    test_out.to_csv(result_dir / "heldout_test_predictions.csv", index=False)

    low = test_out.loc[test_out.demographic_fusion_score.idxmin()]
    common = test_out[test_out.age.between(*common_age)]
    high_common = common.loc[common.demographic_fusion_score.idxmax()]
    actual_positive = test_out[test_out.actual_gad7_positive == 1]
    high_positive = actual_positive.loc[actual_positive.demographic_fusion_score.idxmax()]
    examples = pd.DataFrame([
        {"case": "low_predicted_vulnerability", **low.to_dict()},
        {"case": "high_predicted_vulnerability_common_age", **high_common.to_dict()},
        {"case": "highest_scoring_actual_positive", **high_positive.to_dict()},
    ])
    examples.to_csv(result_dir / "real_case_examples.csv", index=False)

    points = sorted(set([
        float(probabilities.min()), 0.005, 0.01, prevalence, 0.03, 0.05, 0.075, 0.10, 0.15, float(probabilities.max())
    ]))
    pd.DataFrame({
        "demographic_probability": points,
        "probability_percent": [p * 100 for p in points],
        "demographic_fusion_score": [probability_to_fusion_score(p, prevalence) for p in points],
    }).to_csv(result_dir / "fusion_score_reference.csv", index=False)

    bundle = {
        "model": calibrated,
        "reference_prevalence": prevalence,
        "feature_columns": FEATURE_COLUMNS,
        "target": "GAD-7 >= 10",
        "calibration": "sigmoid",
        "best_params": metrics["best_params"],
        "age_training_range": [MIN_AGE, MAX_AGE],
        "age_common_range": common_age,
        "isi_used": False,
        "fusion_transform": metrics["fusion_transform"],
    }
    joblib.dump(bundle, model_dir / "demographic_elasticnet_calibrated.joblib")

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax, name="Calibrated Elastic-Net")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Held-out ROC Curve — Demographic Model")
    fig.tight_layout(); fig.savefig(figure_dir / "heldout_roc_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax, name="Calibrated Elastic-Net")
    ax.axhline(float(y_test.mean()), linestyle="--", linewidth=1, label="Prevalence baseline")
    ax.legend(); ax.set_title("Held-out Precision–Recall Curve")
    fig.tight_layout(); fig.savefig(figure_dir / "heldout_pr_curve.png", dpi=160); plt.close(fig)

    obs, pred = calibration_curve(y_test, probabilities, n_bins=8, strategy="quantile")
    cap = max(float(max(obs)), float(max(pred)), 0.05)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pred, obs, marker="o", label="Calibrated Elastic-Net")
    ax.plot([0, cap], [0, cap], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed GAD-7 >=10 fraction")
    ax.set_title("Held-out Calibration Curve"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "heldout_calibration_curve.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(fusion_scores[np.asarray(y_test) == 0], bins=30, alpha=0.6, label="GAD-7 <10")
    ax.hist(fusion_scores[np.asarray(y_test) == 1], bins=30, alpha=0.6, label="GAD-7 >=10")
    ax.axvline(0.5, linestyle="--", linewidth=1, label="Development prevalence")
    ax.set_xlabel("Demographic fusion score"); ax.set_ylabel("Held-out participants")
    ax.set_title("Fusion-score Distribution"); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir / "fusion_score_distribution.png", dpi=160); plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("\n" + examples.to_string(index=False))


if __name__ == "__main__":
    main()
