from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = ["gender", "edu", "smoke", "drink"]
CATEGORY_LEVELS = {
    "gender": ["female", "male"],
    "edu": ["bachelor's degree", "associate degree", "master's degree", "doctorate degree"],
    "smoke": [
        "never smokes",
        "occasional smoker (cumulative smoking <10 packs)",
        "current smoker (cumulative smoking >10 packs)",
        "former smoker (cumulative smoking >10 packs), but not in the past year",
    ],
    "drink": [
        "never drinks",
        "drinks occasionally (less than once a week)",
        "current regular drinker (more than once a week)",
        "drank in the past (more than once a week), but not in the past year",
    ],
}


def build_elasticnet_pipeline(
    C: float = 1.0,
    l1_ratio: float = 0.5,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 5000,
    random_state: int = 42,
) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_FEATURES])),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    estimator = LogisticRegression(
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
