"""
Final comparison tables, figures and the results report.
Run AFTER evaluate_models.py and evaluate_robustness.py.

Outputs:
  results/tables/final_comparison.csv
  results/figures/confusion_matrices.png, roc_curves.png, metric_comparison.png
  results/final_results_report.md

Run:  python -m evaluation.report_tables
"""
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

import config  # noqa: E402
from evaluation.metrics import METRIC_COLUMNS  # noqa: E402
from evaluation.predictions import load_all_predictions  # noqa: E402
from utils import log_run  # noqa: E402

MAIN_ROWS = [("exp1", "clean"), ("exp2", "clean"), ("exp3", "clean"), ("res32", "res32")]


def md_table(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.{floatfmt}f}")
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    return "\n".join(lines)


def final_comparison(metrics):
    keep = pd.concat([metrics[(metrics["experiment"] == e) & (metrics["condition"] == c)] for e, c in MAIN_ROWS])
    keep = keep.copy()
    keep["model"] = keep["model"].map(config.MODEL_DISPLAY)
    cols = ["experiment", "subset", "model"] + METRIC_COLUMNS + ["n", "tn", "fp", "fn", "tp", "notes"]
    return keep[cols]


def plot_confusion(pred):
    fig, axes = plt.subplots(len(config.MODELS), len(MAIN_ROWS), figsize=(4 * len(MAIN_ROWS), 3.6 * len(config.MODELS)), squeeze=False)
    for r, model in enumerate(config.MODELS):
        for c, (exp, cond) in enumerate(MAIN_ROWS):
            ax = axes[r, c]
            d = pred[(pred["model"] == model) & (pred["experiment"] == exp) & (pred["condition"] == cond)]
            if d.empty:
                ax.set_axis_off()
                ax.set_title(f"{model} {exp}: missing")
                continue
            cm = pd.crosstab(d["label"], d["pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["Real", "AI"])
            ax.set_yticks([0, 1], ["Real", "AI"])
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{config.MODEL_DISPLAY[model]} - {exp}", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "confusion_matrices.png", dpi=120)
    plt.close(fig)


def plot_roc(pred):
    fig, axes = plt.subplots(1, len(MAIN_ROWS), figsize=(4.5 * len(MAIN_ROWS), 4.3), squeeze=False)
    for c, (exp, cond) in enumerate(MAIN_ROWS):
        ax = axes[0, c]
        for model in config.MODELS:
            d = pred[(pred["model"] == model) & (pred["experiment"] == exp) & (pred["condition"] == cond)]
            if d.empty or d["label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(d["label"], d["prob"])
            auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
            ax.plot(fpr, tpr, label=f"{config.MODEL_DISPLAY[model]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(config.EXPERIMENTS[exp]["title"], fontsize=8)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)


def plot_metric_bars(metrics):
    data = metrics[(metrics["subset"] == "all")]
    data = pd.concat([data[(data["experiment"] == e) & (data["condition"] == c)] for e, c in MAIN_ROWS])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    x = np.arange(len(MAIN_ROWS))
    width = 0.8 / len(config.MODELS)
    for ax, metric in zip(axes.ravel(), METRIC_COLUMNS):
        for k, model in enumerate(config.MODELS):
            d = data[data["model"] == model].set_index("experiment").reindex([e for e, _ in MAIN_ROWS])
            ax.bar(x + k * width - 0.4 + width / 2, d[metric].values, width, label=config.MODEL_DISPLAY[model])
        ax.set_xticks(x, [e for e, _ in MAIN_ROWS])
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "metric_comparison.png", dpi=120)
    plt.close(fig)


def write_report(metrics, comparison):
    lines = [f"# AI-Generated Image Detection - Final Results Report", "",
             f"Generated automatically on {date.today()} by `evaluation/report_tables.py`.", "",
             "Research prototype, not a production forensic system. Positive class = AI-generated (1), "
             f"threshold = {config.THRESHOLD}, seed = {config.SEED}.", ""]
    lines += ["## Data", ""]
    for ds in config.DATASETS:
        p = config.MANIFEST_DIR / f"{ds.lower()}_manifest.csv"
        if p.exists():
            m = pd.read_csv(p)
            t = pd.crosstab(m["split"], m["label"].map(config.CLASS_NAMES)).reset_index()
            lines += [f"**{ds}**", "", md_table(t), ""]
    for exp, cond in MAIN_ROWS:
        sub = comparison[comparison["experiment"] == exp].drop(columns=["experiment", "tn", "fp", "fn", "tp", "notes"])
        if sub.empty:
            continue
        lines += [f"## {config.EXPERIMENTS[exp]['title']}", "", md_table(sub), ""]
    rob_path = config.ROBUSTNESS_DIR / "robustness_results.csv"
    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        rob = rob[rob["subset"] == "all"]
        for exp in config.ROBUSTNESS_EXPERIMENTS:
            piv = rob[rob["experiment"] == exp].pivot_table(index="transformation", columns="model", values="accuracy", sort=False)
            lines += [f"## Robustness ({exp}) - accuracy", "", md_table(piv.reset_index()), ""]
    notes = metrics[metrics["notes"].fillna("") != ""][["model", "experiment", "condition", "subset", "notes"]]
    if len(notes):
        lines += ["## Undefined metric cases", "", md_table(notes), ""]
    lines += ["## Figures", "", "- figures/confusion_matrices.png", "- figures/roc_curves.png",
              "- figures/metric_comparison.png", "- figures/robustness_comparison.png", "- figures/training_curves.png", "",
              "## Limitations", "",
              "- Results use a 20,000-image sample per dataset, not the full official CIFAKE/GenImage sets.",
              "- CIFAKE images are 32x32 and are upsampled to 224x224; GenImage images are larger. "
              "Resolution differences can affect cross-dataset results (see the 32x32 experiment).",
              "- All GenImage images were re-encoded as JPEG Q95 to remove the JPEG-vs-PNG format shortcut.",
              "- BigGAN images are 128x128 in GenImage, so they are upsampled more than other generators.",
              "- GPU training is not perfectly bit-for-bit deterministic.", ""]
    out = config.RESULTS_DIR / "final_results_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(config.TABLE_DIR / "all_metrics.csv")
    pred = load_all_predictions()
    comparison = final_comparison(metrics)
    comparison.to_csv(config.TABLE_DIR / "final_comparison.csv", index=False)
    plot_confusion(pred)
    plot_roc(pred)
    plot_metric_bars(metrics)
    out = write_report(metrics, comparison)
    print(comparison[["experiment", "subset", "model"] + METRIC_COLUMNS].round(4).to_string(index=False))
    log_run("student6_evaluation", "final_report", report=str(out))
    print(f"\nSaved final_comparison.csv, figures and {out}")


if __name__ == "__main__":
    main()
