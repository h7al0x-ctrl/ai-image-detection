"""
Shared training/prediction logic for the two feature-based models (SVM, RF),
so that both follow EXACTLY the same protocol:

  1. load train + val features of a model tag (cifake / genimage / res32)
  2. try every hyper-parameter set: fit on TRAIN, score on VAL (ROC-AUC, tie -> accuracy)
     (results saved after each candidate, so a Colab disconnect does not lose them)
  3. refit the best setting on TRAIN only, save as joblib
  4. predict every entry of config.PREDICTION_PLAN (test data used ONLY here)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import config
from evaluation.metrics import compute_metrics
from evaluation.predictions import build_prediction_frame, save_predictions
from features.extract_features import load_features
from preprocessing.image_cache import load_manifest
from utils import Timer, log_run


def positive_scores(estimator, X):
    """Continuous score for class 1 (probability if available, else decision function)."""
    if hasattr(estimator, "predict_proba"):
        try:
            classes = list(estimator.classes_)
            return estimator.predict_proba(X)[:, classes.index(config.LABEL_AI)]
        except AttributeError:  # SVC(probability=False)
            pass
    return estimator.decision_function(X)


def model_path(model_name, tag):
    return config.CHECKPOINT_DIR / model_name / f"{model_name}_{tag}.joblib"


def tune_and_fit(model_name, make_estimator, grid, tag, force=False):
    out_path = model_path(model_name, tag)
    if out_path.exists() and not force:
        print(f"{out_path.name} exists - skipping training (use --force to retrain)")
        return joblib.load(out_path)

    spec = config.MODEL_TAGS[tag]
    X_tr, y_tr, _ = load_features(spec["dataset"], "train", spec["condition"])
    X_va, y_va, _ = load_features(spec["dataset"], "val", spec["condition"])
    print(f"\n=== {model_name.upper()} / {tag}: train {X_tr.shape}, val {X_va.shape} ===")

    tuning_csv = config.TABLE_DIR / f"{model_name}_{tag}_tuning.csv"
    done = pd.read_csv(tuning_csv) if tuning_csv.exists() and not force else pd.DataFrame()
    done_keys = set(done["params"]) if len(done) else set()
    for params in grid:
        key = json.dumps(params, sort_keys=True)
        if key in done_keys:
            print(f"  {key}: already evaluated")
            continue
        est = make_estimator(params, final=False)
        with Timer() as t:
            est.fit(X_tr, y_tr)
        scores = positive_scores(est, X_va)
        row = {"params": key, "val_roc_auc": roc_auc_score(y_va, scores),
               "val_accuracy": accuracy_score(y_va, est.predict(X_va)), "fit_seconds": round(t.seconds, 1)}
        print(f"  {key}: val AUC={row['val_roc_auc']:.4f} acc={row['val_accuracy']:.4f} ({t.seconds:.0f}s)")
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        done.to_csv(tuning_csv, index=False)

    done = done[done["params"].isin([json.dumps(p, sort_keys=True) for p in grid])]
    best_row = done.sort_values(["val_roc_auc", "val_accuracy"], ascending=False).iloc[0]
    best = json.loads(best_row["params"])
    print(f"  BEST (validation): {best}")

    final = make_estimator(best, final=True)
    with Timer() as t:
        final.fit(X_tr, y_tr)
    val_metrics = compute_metrics(y_va, positive_scores(final, X_va))
    bundle = {"model": final, "model_name": model_name, "tag": tag, "best_params": best,
              "selection": f"validation {config.SELECTION_METRIC}", "val_metrics": val_metrics,
              "train_size": len(y_tr), "fit_seconds": round(t.seconds, 1), "seed": config.SEED}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"  saved {out_path}  (val acc {val_metrics['accuracy']:.4f}, AUC {val_metrics['roc_auc']:.4f})")
    log_run(f"{model_name}_training", "fit", tag=tag, best_params=best,
            val_accuracy=val_metrics["accuracy"], val_roc_auc=val_metrics["roc_auc"], seconds=t.seconds)
    return bundle


def predict_plan(model_name, bundles):
    frames = []
    for entry in config.PREDICTION_PLAN:
        if entry["model_tag"] not in bundles:
            continue
        est = bundles[entry["model_tag"]]["model"]
        X, _, ids = load_features(entry["dataset"], entry["split"], entry["condition"])
        probs = positive_scores(est, X)
        manifest = load_manifest(entry["dataset"], entry["split"])
        frames.append(build_prediction_frame(model_name, entry, manifest, probs, ids))
        acc = (frames[-1]["pred"] == frames[-1]["label"]).mean()
        print(f"  {entry['experiment']:6s} {entry['condition']:10s} accuracy={acc:.4f}")
    df = save_predictions(model_name, frames)
    log_run(f"{model_name}_prediction", "predict_plan", n_rows=len(df))
    return df


def run(model_name, make_estimator, grid, tags=None, force=False):
    tags = tags or list(config.MODEL_TAGS)
    bundles = {tag: tune_and_fit(model_name, make_estimator, grid, tag, force) for tag in tags}
    print(f"\n=== {model_name.upper()} predictions ===")
    return predict_plan(model_name, bundles)
