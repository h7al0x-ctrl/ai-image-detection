"""
Leakage audit.

Checks both manifests and writes data/manifests/leakage_audit.csv:
  * sample counts vs config, label/generator consistency
  * exact duplicates (SHA-256) within and across splits and across datasets
  * near duplicates (pHash) across splits
  * held-out generators (BigGAN, Midjourney) only in test
  * CIFAKE train/val only from the official train folder, test only from official test
  * cache size matches manifest
  * excluded files (corrupt / duplicates) from the data reports

Run:  python -m preprocessing.leakage_audit
Exit code 1 if any check FAILS.
"""
import sys

import numpy as np
import pandas as pd

import config
from preprocessing.preprocess import hamming_to_many
from preprocessing.sampling import cache_path, manifest_path, report_path
from utils import log_run

rows = []


def record(check, dataset, passed, value="", detail="", warn_only=False):
    result = "PASS" if passed else ("WARN" if warn_only else "FAIL")
    rows.append({"check": check, "dataset": dataset, "result": result, "value": value, "detail": detail})


def near_duplicate_pairs(hashes_a, hashes_b, max_dist, chunk=500):
    count = 0
    for i in range(0, len(hashes_a), chunk):
        block = hashes_a[i:i + chunk]
        for h in block:
            count += int((hamming_to_many(int(h), hashes_b) <= max_dist).sum())
    return count


def audit_dataset(dataset, df):
    expected = config.expected_counts(dataset)
    for split in config.SPLITS:
        for label in (0, 1):
            n = int(((df["split"] == split) & (df["label"] == label)).sum())
            record(f"count {split} {config.CLASS_NAMES[label]}", dataset, n == expected[split][label],
                   n, f"expected {expected[split][label]}")
        if cache_path(dataset, split).exists():
            n_cache = len(np.load(cache_path(dataset, split), mmap_mode="r"))
            n_man = int((df["split"] == split).sum())
            record(f"cache size {split}", dataset, n_cache == n_man, n_cache, f"manifest {n_man}")
        else:
            record(f"cache size {split}", dataset, False, "", "cache file missing (restore it)", warn_only=True)

    bad_label = df[((df["generator"] == "real") & (df["label"] != 0)) | ((df["generator"] != "real") & (df["label"] != 1))]
    record("label matches generator", dataset, bad_label.empty, len(bad_label))
    record("required columns non-empty", dataset,
           not df[["image_path", "label", "split", "dataset"]].isna().any().any())

    dup_sha = int(df["sha256"].duplicated().sum())
    record("exact duplicates (SHA-256)", dataset, dup_sha == 0, dup_sha)
    dup_path = int(df["image_path"].duplicated().sum())
    record("same file used twice", dataset, dup_path == 0, dup_path)

    hashes = {s: np.array([int(h, 16) for h in df.loc[df["split"] == s, "phash"]], dtype=np.uint64)
              for s in config.SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = near_duplicate_pairs(hashes[b], hashes[a], config.PHASH_MAX_DISTANCE)
        record(f"near duplicates {a}<->{b} (pHash<= {config.PHASH_MAX_DISTANCE})", dataset, n == 0, n)

    if dataset == config.GENIMAGE:
        trainval = df[df["split"].isin(["train", "val"])]
        leaked = trainval[trainval["source_group"].isin(config.GENIMAGE_TEST_GENERATORS)]
        record("held-out generators absent from train/val", dataset, leaked.empty, len(leaked))
        test = df[df["split"] == "test"]
        wrong = test[~test["source_group"].isin(config.GENIMAGE_TEST_GENERATORS)]
        record("test contains only held-out generators", dataset, wrong.empty, len(wrong))
        for gen in config.GENIMAGE_ALL_GENERATORS:
            n = int((df["source_group"] == gen).sum())
            record(f"generator present: {gen}", dataset, n > 0, n)
    else:
        bad = df[(df["split"].isin(["train", "val"]) & (df["official_split"] != "train"))
                 | ((df["split"] == "test") & (df["official_split"] != "test"))]
        record("CIFAKE official split respected", dataset, bad.empty, len(bad))

    if report_path(dataset).exists():
        rep = pd.read_csv(report_path(dataset))
        for status, n in rep["status"].value_counts().items():
            if status != "accepted":
                record(f"excluded files: {status}", dataset, True, int(n), "documented in data report", warn_only=True)


def main():
    manifests = {}
    for dataset in config.DATASETS:
        if not manifest_path(dataset).exists():
            record("manifest exists", dataset, False, "", f"{manifest_path(dataset)} missing")
            continue
        manifests[dataset] = pd.read_csv(manifest_path(dataset))
        audit_dataset(dataset, manifests[dataset])
    if len(manifests) == 2:
        common = set(manifests[config.CIFAKE]["sha256"]) & set(manifests[config.GENIMAGE]["sha256"])
        record("CIFAKE/GenImage shared files", "both", not common, len(common))

    table = pd.DataFrame(rows)
    out = config.MANIFEST_DIR / "leakage_audit.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    n_fail = int((table["result"] == "FAIL").sum())
    log_run("leakage_audit", "audit", status="ok" if n_fail == 0 else "failed", n_fail=n_fail)
    print(f"\nSaved {out}.  FAILED checks: {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
