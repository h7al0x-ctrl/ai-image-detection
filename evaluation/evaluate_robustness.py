"""
Robustness table and chart (run AFTER evaluate_models.py).

Outputs:
  results/robustness/robustness_results.csv   (all models, both experiments, 9 conditions,
                                                with change vs the clean baseline)
  results/robustness/<model>_robustness.csv   (svm_robustness.csv, rf_..., cnn_...)
  results/figures/robustness_comparison.png

Run:  python -m evaluation.evaluate_robustness
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from evaluation.metrics import METRIC_COLUMNS  # noqa: E402
from utils import log_run  # noqa: E402


def build_table():
    metrics = pd.read_csv(config.TABLE_DIR / "all_metrics.csv")
    rob = metrics[metrics["experiment"].isin(config.ROBUSTNESS_EXPERIMENTS)
                  & metrics["condition"].isin(config.ROBUSTNESS_CONDITIONS)].copy()
    rob["transformation"] = rob["condition"].map(lambda c: config.CONDITIONS[c]["label"])
    rob["parameters"] = rob["condition"].map(lambda c: config.CONDITIONS[c]["params"])
    clean = rob[rob["condition"] == "clean"].set_index(["model", "experiment", "subset"])[METRIC_COLUMNS]
    for m in METRIC_COLUMNS:
        base = rob.set_index(["model", "experiment", "subset"]).index.map(clean[m].to_dict())
        rob[f"delta_{m}"] = rob[m].values - base.values
    order = {c: i for i, c in enumerate(config.ROBUSTNESS_CONDITIONS)}
    rob["_order"] = rob["condition"].map(order)
    rob = rob.sort_values(["experiment", "model", "subset", "_order"])
    cols = (["model", "experiment", "subset", "condition", "transformation", "parameters"]
            + METRIC_COLUMNS + [f"delta_{m}" for m in METRIC_COLUMNS] + ["n", "notes"])
    return rob[cols]


def plot(rob):
    data = rob[rob["subset"] == "all"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey="row")
    labels = [config.CONDITIONS[c]["label"] for c in config.ROBUSTNESS_CONDITIONS]
    for c, exp in enumerate(config.ROBUSTNESS_EXPERIMENTS):
        for r, metric in enumerate(("accuracy", "roc_auc")):
            ax = axes[r, c]
            for model in config.MODELS:
                d = data[(data["experiment"] == exp) & (data["model"] == model)]
                if d.empty:
                    continue
                d = d.set_index("condition").reindex(config.ROBUSTNESS_CONDITIONS)
                ax.plot(labels, d[metric], "o-", label=config.MODEL_DISPLAY[model])
            ax.set_title(f"{config.EXPERIMENTS[exp]['title']}\n{metric}", fontsize=9)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "robustness_comparison.png", dpi=120)
    plt.close(fig)


def main():
    config.ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rob = build_table()
    rob.to_csv(config.ROBUSTNESS_DIR / "robustness_results.csv", index=False)
    for model, g in rob.groupby("model"):
        g.to_csv(config.ROBUSTNESS_DIR / f"{model}_robustness.csv", index=False)
    plot(rob)
    view = rob[rob["subset"] == "all"][["model", "experiment", "condition", "accuracy", "roc_auc", "delta_accuracy"]]
    print(view.round(4).to_string(index=False))
    log_run("student6_evaluation", "robustness_table", n_rows=len(rob))
    print("\nSaved results/robustness/robustness_results.csv and figures/robustness_comparison.png")


if __name__ == "__main__":
    main()
