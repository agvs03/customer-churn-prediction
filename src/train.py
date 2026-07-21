"""End-to-end training pipeline for customer churn prediction.

Steps:
  1. Load + clean data, engineer 25+ features
  2. Build preprocessing ColumnTransformer (scale numerics, one-hot categoricals)
  3. Compare 5 models with SMOTE inside 5-fold stratified CV
  4. Hyperparameter-tune the best model (XGBoost) with GridSearchCV
  5. Evaluate on hold-out test set, generate figures
  6. Log everything to MLflow and persist the fitted pipeline

Run:  python -m src.train
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import CFG, SEED, TARGET, PROJECT_ROOT
from src.data_loader import load_raw
from src.preprocessing import clean
from src.feature_engineering import add_features
from src import evaluate as ev

warnings.filterwarnings("ignore")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW = True
except Exception:  # pragma: no cover
    MLFLOW = False


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )


def candidate_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1),
        # probability=False keeps CV fast; ROC-AUC scoring uses decision_function.
        "SVM": SVC(kernel="rbf", probability=False, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=15),
    }


def make_pipeline(preprocessor, model, use_smote=True) -> ImbPipeline:
    steps = [("prep", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 70)

    fig_dir = PROJECT_ROOT / CFG["artifacts"]["figures_dir"]
    model_path = PROJECT_ROOT / CFG["artifacts"]["model_path"]
    metrics_path = PROJECT_ROOT / CFG["artifacts"]["metrics_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    raw = load_raw()
    df = clean(raw)
    df = add_features(df)
    print(f"[train] Dataset shape after feature engineering: {df.shape}")
    print(f"[train] Total features (excl. target): {df.shape[1] - 1}")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # EDA figures (use engineered frame)
    ev.plot_class_distribution(df, TARGET, fig_dir)
    ev.plot_histograms(df, fig_dir)
    ev.plot_correlation_heatmap(df, fig_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG["data"]["test_size"], stratify=y, random_state=SEED)

    preprocessor = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=CFG["training"]["cv_folds"], shuffle=True, random_state=SEED)
    scoring = CFG["training"]["scoring"]
    use_smote = CFG["training"]["use_smote"]

    if MLFLOW:
        # Allow an env override (handy when the working dir is on a network
        # mount that can't host a SQLite file); falls back to config.
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", CFG["mlflow"]["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

    # 2. Compare 5 models ---------------------------------------------------
    print("\n[train] Cross-validating 5 models (5-fold, SMOTE inside CV)...")
    rows = []
    for name, model in candidate_models().items():
        pipe = make_pipeline(preprocessor, model, use_smote)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc": scores.mean(), "cv_std": scores.std()})
        print(f"   {name:<20} ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
        if MLFLOW:
            with mlflow.start_run(run_name=f"cv_{name}"):
                mlflow.log_param("model_type", name)
                mlflow.log_param("smote", use_smote)
                mlflow.log_param("cv_folds", CFG["training"]["cv_folds"])
                mlflow.log_metric("cv_roc_auc_mean", float(scores.mean()))
                mlflow.log_metric("cv_roc_auc_std", float(scores.std()))

    results_df = pd.DataFrame(rows)
    ev.plot_model_comparison(results_df, fig_dir)
    print("\n[train] Model comparison:\n", results_df.to_string(index=False))

    # 3. Tune XGBoost with GridSearchCV ------------------------------------
    print("\n[train] GridSearchCV hyperparameter tuning for XGBoost...")
    xgb_pipe = make_pipeline(preprocessor, XGBClassifier(
        eval_metric="logloss", random_state=SEED, n_jobs=-1), use_smote)
    param_grid = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.9],
        "model__colsample_bytree": [0.9],
    }
    # A leaner grid keeps a full run fast; set CHURN_FAST=0 for the full sweep.
    if os.environ.get("CHURN_FAST", "1") == "1":
        param_grid = {
            "model__n_estimators": [300],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.9],
            "model__colsample_bytree": [0.9],
        }
    grid = GridSearchCV(xgb_pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"[train] Best params: {grid.best_params_}")
    print(f"[train] Best CV ROC-AUC: {grid.best_score_:.4f}")

    # 4. Evaluate on hold-out ----------------------------------------------
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = ev.compute_metrics(y_test, y_pred, y_proba)
    print("\n[train] Hold-out test metrics:")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v:.4f}")

    # Diagnostic figures
    ev.plot_confusion_matrix(y_test, y_pred, fig_dir)
    ev.plot_roc_curve(y_test, y_proba, fig_dir)
    feat_names = best.named_steps["prep"].get_feature_names_out()
    ev.plot_feature_importance(best.named_steps["model"], feat_names, fig_dir)

    # 5. Persist the serving artifact --------------------------------------
    joblib.dump(
        {"pipeline": best, "feature_columns": X.columns.tolist(),
         "raw_input_columns": [c for c in raw.columns if c not in (TARGET, "customerID")]},
        model_path)

    # 6. Log best model + artifact to MLflow -------------------------------
    if MLFLOW:
        with mlflow.start_run(run_name="xgboost_best"):
            mlflow.log_params({k.replace("model__", ""): v for k, v in grid.best_params_.items()})
            mlflow.log_param("smote", use_smote)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_cv_roc_auc", float(grid.best_score_))
            # Log the exact joblib artifact used for serving (robust across
            # mlflow versions; avoids the skops serializer rejecting XGBoost
            # / imbalanced-learn types).
            try:
                mlflow.log_artifact(str(model_path), artifact_path="model")
            except Exception as e:  # noqa
                print(f"[train] (mlflow artifact log skipped: {e})")

    payload = {
        "test_metrics": metrics,
        "best_params": grid.best_params_,
        "best_cv_roc_auc": float(grid.best_score_),
        "model_comparison": results_df.to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[train] Saved model -> {model_path}")
    print(f"[train] Saved metrics -> {metrics_path}")
    print("=" * 70)
    return payload


if __name__ == "__main__":
    main()
