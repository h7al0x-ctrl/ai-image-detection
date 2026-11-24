"""
Compute ALL metrics from the three prediction files with ONE implementation.

For every (model, experiment, condition) it reports:
  subset "all"            -> whole test set
  subset "BigGAN" / "Midjourney" -> GenImage held-out generators separately (their AI + real images)

Outputs:
  results/tables/all_metrics.csv
  results/tables/<model>_<experiment>.csv       e.g. svm_exp1.csv, cnn_exp3.csv  (clean)
  results/tables/result_completeness.csv        which expected results are missing

Run:  python -m evaluation.evaluate_models
"""
import pandas as pd

import config
from evaluation.metrics import compute_metrics
from evaluation.predictions import load_all_predictions
from utils import log_run


def compute_all_metrics(pred):
    rows = []
    for (model, exp, cond), g in pred.groupby(["model", "experiment", "condition"], sort=False):
        subsets = [("all", g)]
        if (g["dataset"] == config.GENIMAGE).all():
            subsets += [(gen, g[g["source_group"] == gen]) for gen in config.GENIMAGE_TEST_GENERATORS]
        for name, sub in subsets:
            if len(sub) == 0:
                continue
            m = compute_metrics(sub["label"], sub["prob"])
            rows.append({"model": model, "experiment": exp, "condition": cond, "subset": name,
                         "model_tag": g["model_tag"].iloc[0], "test_dataset": g["dataset"].iloc[0], **m})
    return pd.DataFrame(rows)


def completeness(pred):
    present = set(map(tuple, pred[["model", "experiment", "condition"]].drop_duplicates().values))
    rows = []
    for model in config.MODELS:
        for e in config.PREDICTION_PLAN:
            key = (model, e["experiment"], e["condition"])
            n = int(((pred["model"] == model) & (pred["experiment"] == e["experiment"])
                     & (pred["condition"] == e["condition"])).sum())
            rows.append({"model": model, "experiment": e["experiment"], "condition": e["condition"],
                         "status": "OK" if key in present else "MISSING", "n_predictions": n})
    return pd.DataFrame(rows)


def main():
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pred = load_all_predictions()
    metrics = compute_all_metrics(pred)
    metrics.to_csv(config.TABLE_DIR / "all_metrics.csv", index=False)
    for (model, exp), g in metrics[metrics["condition"].isin(["clean", "res32"])].groupby(["model", "experiment"]):
        g.to_csv(config.TABLE_DIR / f"{model}_{exp}.csv", index=False)

    comp = completeness(pred)
    comp.to_csv(config.TABLE_DIR / "result_completeness.csv", index=False)
    missing = comp[comp["status"] == "MISSING"]

    view = metrics[metrics["condition"].isin(["clean", "res32"])]
    cols = ["model", "experiment", "subset", "accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"]
    pd.set_option("display.width", 200)
    print(view[cols].round(4).to_string(index=False))
    print(f"\nmissing results: {len(missing)}")
    if len(missing):
        print(missing.to_string(index=False))
    log_run("student6_evaluation", "compute_metrics", n_rows=len(metrics), missing=len(missing))


if __name__ == "__main__":
    main()
