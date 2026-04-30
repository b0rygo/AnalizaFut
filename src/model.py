from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_data
from features import FEATURE_COLS, TIER_COL, build_features

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASSES = ["A", "D", "H"]
TRAIN_START = pd.Timestamp("1950-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_END = pd.Timestamp("2026-03-31")


def _build_pipeline(estimator):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLS),
            ("tier", OneHotEncoder(handle_unknown="ignore"), [TIER_COL]),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _evaluate(name, pipe, X_te, y_te):
    proba = pipe.predict_proba(X_te)
    pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, pred)
    ll = log_loss(y_te, proba, labels=pipe.classes_.tolist())
    print(f"\n=== {name} ===")
    print(f"  accuracy : {acc:.4f}")
    print(f"  log-loss : {ll:.4f}")
    print(classification_report(y_te, pred, digits=3))
    return acc, ll


def main():
    d = load_data()
    played, fixtures, _ = build_features(d["played"], d["fixtures"])

    df = played[played["date"] >= TRAIN_START].copy()
    train = df[df["date"] <= TRAIN_END]
    test = df[(df["date"] > TRAIN_END) & (df["date"] <= TEST_END)]
    print(f"Train: {len(train):>6}  ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"Test : {len(test):>6}  ({test['date'].min().date()} -> {test['date'].max().date()})")

    feat_cols = FEATURE_COLS + [TIER_COL]
    X_tr, y_tr = train[feat_cols], train["outcome"]
    X_te, y_te = test[feat_cols], test["outcome"]

    base_pred = np.array(["H"] * len(y_te))
    print(f"\nBaseline (always H) accuracy: {accuracy_score(y_te, base_pred):.4f}")

    candidates = {
        "LogisticRegression": _build_pipeline(
            LogisticRegression(max_iter=1000, C=1.0)
        ),
        "GradientBoosting": _build_pipeline(
            GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        ),
        "RandomForest": _build_pipeline(
            RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
                                   n_jobs=-1, random_state=42)
        ),
    }

    results = {}
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        results[name] = _evaluate(name, pipe, X_te, y_te)
        fitted[name] = pipe

    best_name = min(results, key=lambda k: results[k][1])
    print(f"\nBest by log-loss: {best_name}  (acc={results[best_name][0]:.4f}, ll={results[best_name][1]:.4f})")

    full_pipe = _build_pipeline(type(fitted[best_name].named_steps["clf"])(
        **fitted[best_name].named_steps["clf"].get_params()
    ))
    full_X = played[played["date"] >= TRAIN_START][feat_cols]
    full_y = played[played["date"] >= TRAIN_START]["outcome"]
    full_pipe.fit(full_X, full_y)

    lr_pipe = _build_pipeline(LogisticRegression(max_iter=2000, C=1.0))
    lr_pipe.fit(full_X, full_y)

    pre = full_pipe.named_steps["pre"]
    expanded_names = pre.get_feature_names_out().tolist()

    best_pipe = fitted[best_name]
    cm_pred = best_pipe.predict(X_te)
    cm = confusion_matrix(y_te, cm_pred, labels=list(best_pipe.classes_))

    gbm_imp = None
    clf = full_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        gbm_imp = list(zip(expanded_names, clf.feature_importances_.tolist()))

    lr_clf = lr_pipe.named_steps["clf"]
    lr_coef = {
        cls: dict(zip(expanded_names, lr_clf.coef_[i].tolist()))
        for i, cls in enumerate(lr_clf.classes_)
    }
    lr_intercept = dict(zip(lr_clf.classes_, lr_clf.intercept_.tolist()))

    out = MODELS_DIR / "predictor.pkl"
    joblib.dump({
        "pipeline": full_pipe,
        "lr_pipeline": lr_pipe,
        "classes": list(full_pipe.classes_),
        "feature_cols": feat_cols,
        "expanded_names": expanded_names,
        "best_model": best_name,
        "test_metrics": {
            "accuracy": results[best_name][0],
            "log_loss": results[best_name][1],
            "all_models": {n: {"accuracy": a, "log_loss": l} for n, (a, l) in results.items()},
            "confusion_matrix": cm.tolist(),
            "cm_labels": list(best_pipe.classes_),
        },
        "gbm_importances": gbm_imp,
        "lr_coef": lr_coef,
        "lr_intercept": lr_intercept,
        "hyperparameters": {n: p.named_steps["clf"].get_params() for n, p in fitted.items()},
    }, out)
    print(f"Saved -> {out}")
    return full_pipe


if __name__ == "__main__":
    main()
