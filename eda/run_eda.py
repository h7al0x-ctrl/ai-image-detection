"""
Exploratory data analysis for CIFAKE and GenImage.

Outputs (per dataset):
  results/tables/<dataset>_eda_summary.csv   counts, sizes, formats
  results/figures/eda_<dataset>_counts.png   class balance per split / generator
  results/figures/eda_<dataset>_sizes.png    original image sizes
  results/figures/eda_<dataset>_samples.png  example preprocessed images

Run:  python -m eda.run_eda            (both datasets)
      python -m eda.run_eda --dataset CIFAKE
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from preprocessing.sampling import cache_path, manifest_path, report_path  # noqa: E402


def summarise(dataset):
    df = pd.read_csv(manifest_path(dataset))
    df["class"] = df["label"].map(config.CLASS_NAMES)
    parts = []

    counts = df.groupby(["split", "source_group", "class"]).size().rename("value").reset_index()
    counts["section"] = "count"
    counts["item"] = counts["split"] + " | " + counts["source_group"] + " | " + counts["class"]
    parts.append(counts[["section", "item", "value"]])

    fmt = df.groupby(["class", "orig_format"]).size().rename("value").reset_index()
    fmt["section"] = "original format (before preprocessing)"
    fmt["item"] = fmt["class"] + " | " + fmt["orig_format"].astype(str)
    parts.append(fmt[["section", "item", "value"]])

    for cls, g in df.groupby("class"):
        for col in ("orig_width", "orig_height"):
            for stat, val in g[col].describe()[["min", "50%", "max", "mean"]].items():
                parts.append(pd.DataFrame([{"section": "original size", "item": f"{cls} | {col} | {stat}", "value": round(val, 1)}]))

    if report_path(dataset).exists():
        rep = pd.read_csv(report_path(dataset))["status"].value_counts().rename("value").reset_index()
        rep.columns = ["item", "value"]
        rep["section"] = "examined files by status"
        parts.append(rep[["section", "item", "value"]])

    summary = pd.concat(parts, ignore_index=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLE_DIR / f"{dataset.lower()}_eda_summary.csv", index=False)
    print(f"\n===== {dataset} =====")
    print(pd.crosstab([df["split"], df["source_group"]], df["class"]))
    print("\nOriginal file formats by class (GenImage: shows why JPEG re-encoding is needed):")
    print(pd.crosstab(df["class"], df["orig_format"]))
    return df


def plot(dataset, df):
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # counts
    fig, ax = plt.subplots(figsize=(10, 4))
    key = "source_group" if dataset == config.GENIMAGE else "split"
    pd.crosstab(df[key], df["class"]).plot.bar(ax=ax)
    ax.set_title(f"{dataset}: images per {key} and class")
    ax.set_ylabel("images")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"eda_{dataset.lower()}_counts.png", dpi=120)
    plt.close(fig)
    # sizes
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, g in df.groupby("class"):
        ax.scatter(g["orig_width"], g["orig_height"], s=6, alpha=0.4, label=cls)
    ax.set_xlabel("original width")
    ax.set_ylabel("original height")
    ax.set_title(f"{dataset}: original image sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / f"eda_{dataset.lower()}_sizes.png", dpi=120)
    plt.close(fig)
    # samples
    if cache_path(dataset, "train").exists() or cache_path(dataset, "test").exists():
        split = "train" if cache_path(dataset, "train").exists() else "test"
        imgs = np.load(cache_path(dataset, split), mmap_mode="r")
        sub = df[df["split"] == split].sort_values("cache_index")
        rng = np.random.default_rng(config.SEED)
        fig, axes = plt.subplots(2, 6, figsize=(14, 5))
        for r, label in enumerate((0, 1)):
            idx = rng.choice(sub.loc[sub["label"] == label, "cache_index"].values, 6, replace=False)
            for c, i in enumerate(idx):
                axes[r, c].imshow(imgs[i])
                axes[r, c].set_title(f"{config.CLASS_NAMES[label]}\n{sub.iloc[i]['source_group']}", fontsize=8)
                axes[r, c].axis("off")
        fig.suptitle(f"{dataset} ({split}) after preprocessing (224x224)")
        fig.tight_layout()
        fig.savefig(config.FIGURE_DIR / f"eda_{dataset.lower()}_samples.png", dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=config.DATASETS, nargs="*", default=config.DATASETS)
    args = parser.parse_args()
    for dataset in args.dataset:
        if not manifest_path(dataset).exists():
            print(f"skip {dataset}: manifest not found")
            continue
        plot(dataset, summarise(dataset))
    print("\nEDA saved in results/tables and results/figures")


if __name__ == "__main__":
    main()
