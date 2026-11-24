"""
Extract and save ResNet50 features for every set needed by SVM and RF
(config.FEATURE_SETS): train/val/test clean, robustness conditions on the test
splits, and the 32x32 condition. One .npz per (dataset, split, condition) with
`features`, `labels`, `sample_ids` (alignment with the manifest is verified).

Run:  python -m features.extract_features          (skips files that already exist)
      python -m features.extract_features --force
"""
import argparse

import numpy as np

import config
from preprocessing.image_cache import load_manifest
from utils import Timer, ensure_dirs, log_run, set_seed


def feature_path(dataset, split, condition):
    return config.FEATURE_DIR / f"{dataset}_{split}_{condition}.npz"


def load_features(dataset, split, condition):
    """Load (X, y, sample_ids) and verify they match the manifest order and labels."""
    path = feature_path(dataset, split, condition)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run features.extract_features first")
    data = np.load(path, allow_pickle=False)
    X, y, ids = data["features"], data["labels"], data["sample_ids"]
    manifest = load_manifest(dataset, split)
    if not np.array_equal(ids, manifest["sample_id"].to_numpy(dtype=str)):
        raise ValueError(f"{path.name}: sample ids do not match the manifest - re-extract features")
    if not np.array_equal(y, manifest["label"].values):
        raise ValueError(f"{path.name}: labels do not match the manifest")
    return X, y, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    import torch
    from features.feature_extractor import ResNet50FeatureExtractor, extract
    from preprocessing.image_cache import make_loader
    from utils import get_device

    ensure_dirs()
    set_seed()
    device = get_device()
    print("device:", device)
    model = ResNet50FeatureExtractor().to(device)

    for k, (dataset, split, condition) in enumerate(config.FEATURE_SETS, 1):
        path = feature_path(dataset, split, condition)
        if path.exists() and not args.force:
            print(f"[{k}/{len(config.FEATURE_SETS)}] exists: {path.name}")
            continue
        print(f"[{k}/{len(config.FEATURE_SETS)}] extracting {dataset}/{split}/{condition}")
        loader, df = make_loader(dataset, split, condition, train=False)
        with Timer() as t:
            feats = extract(model, loader, device)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, features=feats, labels=df["label"].values.astype(np.int64),
                 sample_ids=df["sample_id"].to_numpy(dtype=str))
        tmp.replace(path)
        print(f"   saved {feats.shape} in {t.seconds:.0f}s -> {path.name}")
        log_run("student3_features", "extract", dataset=dataset, split=split, condition=condition,
                shape=feats.shape, seconds=round(t.seconds, 1), device=str(device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\nAll feature files ready in", config.FEATURE_DIR)


if __name__ == "__main__":
    main()
