"""
Random Forest on the SAME ResNet50 features as the SVM.

Hyper-parameters (config.RF_PARAM_GRID) chosen on the validation split.
No scaling is needed for trees. Also produces an error analysis.

Outputs: checkpoints/rf/rf_{cifake,genimage,res32}.joblib,
         results/tables/rf_<tag>_tuning.csv, results/predictions/rf_predictions.csv,
         results/tables/rf_error_analysis.csv, rf_error_summary.csv

Run:  python -m models.rf_model
"""
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
from models.feature_model_common import run
from utils import ensure_dirs, set_seed


def make_rf(params, final=False):
    return RandomForestClassifier(**params, max_features=config.RF_MAX_FEATURES,
                                  n_jobs=-1, random_state=config.SEED)


def error_analysis(pred):
    """Misclassified clean test images + summary by experiment / generator / error type."""
    clean = pred[pred["condition"].isin(["clean", "res32"])].copy()
    wrong = clean[clean["pred"] != clean["label"]].copy()
    wrong["error_type"] = wrong["label"].map({1: "false negative (AI called real)", 0: "false positive (real called AI)"})
    wrong["confidence"] = (wrong["prob"] - 0.5).abs() * 2
    wrong = wrong.sort_values(["experiment", "confidence"], ascending=[True, False])
    wrong.to_csv(config.TABLE_DIR / "rf_error_analysis.csv", index=False)

    total = clean.groupby(["experiment", "source_group", "generator"]).size().rename("n_images")
    errors = wrong.groupby(["experiment", "source_group", "generator"]).size().rename("n_errors")
    summary = pd.concat([total, errors], axis=1).fillna(0).astype(int).reset_index()
    summary["error_rate"] = (summary["n_errors"] / summary["n_images"]).round(4)
    summary.to_csv(config.TABLE_DIR / "rf_error_summary.csv", index=False)
    print("\n=== RF error summary (clean test sets) ===")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="*", choices=list(config.MODEL_TAGS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    set_seed()
    pred = run("rf", make_rf, config.RF_PARAM_GRID, args.tags, args.force)
    error_analysis(pred)


if __name__ == "__main__":
    main()
