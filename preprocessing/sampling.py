"""
Shared sampling engine used by cifake_dataset.py and genimage_dataset.py.

For each group (e.g. "GenImage / ADM / ai") it walks the candidate files in a
seeded random order and ACCEPTS an image only if:
  1. it opens and decodes correctly            (else: corrupt)
  2. its SHA-256 was not seen before            (else: exact_duplicate)
  3. its pHash is > PHASH_MAX_DISTANCE away
     from every image already accepted          (else: near_duplicate)
Accepted images are preprocessed once and written to a .npy cache per split.
Because a rejected image never enters ANY split, no duplicate can cross splits.
"""
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
from preprocessing.preprocess import (hamming_to_many, load_rgb, phash_int,
                                      preprocess_to_cache_array, sha256_file)
from utils import save_json


@dataclass
class SampleGroup:
    name: str
    dataset: str
    source_group: str          # generator folder name, or "CIFAKE"
    generator: str             # "real" or the generator name
    label: int
    candidates: list           # list of Path
    splits: list               # [(split, n_images), ...]
    folder: str = ""
    jpeg_quality: int = None
    extra: dict = field(default_factory=dict)


def list_images(folder):
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return sorted(files)


def manifest_path(dataset):
    return config.MANIFEST_DIR / f"{dataset.lower()}_manifest.csv"


def report_path(dataset):
    return config.MANIFEST_DIR / f"{dataset.lower()}_data_report.csv"


def cache_path(dataset, split):
    return config.CACHE_DIR / f"{dataset}_{split}.npy"


def outputs_exist(dataset, splits):
    return manifest_path(dataset).exists() and all(cache_path(dataset, s).exists() for s in splits)


def select_and_cache(dataset, groups, data_root):
    """Run the selection for all groups and write manifest, data report and cache."""
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    split_totals = defaultdict(int)
    for g in groups:
        for split, n in g.splits:
            split_totals[split] += n
    total = sum(split_totals.values())

    partial = {s: config.CACHE_DIR / f"{dataset}_{s}.partial.npy" for s in split_totals}
    arrays = {s: np.lib.format.open_memmap(partial[s], mode="w+", dtype=np.uint8,
                                           shape=(n, config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
              for s, n in split_totals.items()}
    next_index = {s: 0 for s in split_totals}

    accepted_hashes = np.zeros(total, dtype=np.uint64)
    accepted_paths = []
    seen_sha = {}
    manifest_rows, report_rows = [], []
    rng = random.Random(config.SEED)

    for g in groups:
        candidates = sorted(g.candidates)
        rng.shuffle(candidates)
        slots = [split for split, n in g.splits for _ in range(n)]
        filled = 0
        print(f"\n[{g.name}] need {len(slots)} images from {len(candidates)} candidates")
        for path in candidates:
            if filled >= len(slots):
                break
            rel = f"{dataset}/{Path(os.path.relpath(path, data_root)).as_posix()}"
            base = {"image_path": rel, "group": g.name, "label": g.label}
            try:
                sha = sha256_file(path)
                img, meta = load_rgb(path)
            except Exception as exc:  # corrupted / unreadable
                report_rows.append({**base, "status": "corrupt", "detail": repr(exc)[:200]})
                continue
            if sha in seen_sha:
                report_rows.append({**base, "status": "exact_duplicate", "detail": f"same SHA-256 as {seen_sha[sha]}"})
                continue
            ph = phash_int(img)
            n_acc = len(accepted_paths)
            if n_acc:
                dist = hamming_to_many(ph, accepted_hashes[:n_acc])
                j = int(np.argmin(dist))
                if dist[j] <= config.PHASH_MAX_DISTANCE:
                    report_rows.append({**base, "status": "near_duplicate",
                                        "detail": f"pHash distance {int(dist[j])} to {accepted_paths[j]}"})
                    continue
            try:
                arr = preprocess_to_cache_array(img, g.jpeg_quality)
            except Exception as exc:
                report_rows.append({**base, "status": "corrupt", "detail": repr(exc)[:200]})
                continue

            split = slots[filled]
            idx = next_index[split]
            arrays[split][idx] = arr
            next_index[split] += 1
            seen_sha[sha] = rel
            accepted_hashes[n_acc] = np.uint64(ph)
            accepted_paths.append(rel)
            filled += 1
            manifest_rows.append({
                "sample_id": f"{dataset}_{split}_{idx:06d}", "image_path": rel, "label": g.label,
                "generator": g.generator, "source_group": g.source_group, "split": split,
                "dataset": dataset, "cache_index": idx, "sha256": sha, "phash": f"{ph:016x}",
                **meta, **g.extra,
            })
            report_rows.append({**base, "status": "accepted", "detail": split})
            if filled % 500 == 0:
                print(f"   {filled}/{len(slots)} accepted")

        if filled < len(slots):
            for arr in arrays.values():
                del arr
            raise RuntimeError(
                f"\n[{g.name}] only {filled} valid, unique images were found but {len(slots)} are required.\n"
                f"Folder: {g.folder}\nAdd more images to this folder (check data report for corrupt/duplicate files)."
            )
        print(f"   done: {filled} accepted")

    for split, arr in arrays.items():
        arr.flush()
        if next_index[split] != split_totals[split]:
            raise RuntimeError(f"split {split}: wrote {next_index[split]} of {split_totals[split]}")
    del arrays
    for split, p in partial.items():
        os.replace(p, cache_path(dataset, split))

    manifest = pd.DataFrame(manifest_rows).sort_values(["split", "cache_index"]).reset_index(drop=True)
    manifest.to_csv(manifest_path(dataset), index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(report_path(dataset), index=False)
    save_json({"dataset": dataset, "data_root": str(data_root), "split_counts": dict(split_totals),
               "status_counts": report["status"].value_counts().to_dict(),
               "jpeg_quality": groups[0].jpeg_quality, "resize_policy": config.RESIZE_POLICY,
               "phash_max_distance": config.PHASH_MAX_DISTANCE, "seed": config.SEED},
              config.MANIFEST_DIR / f"{dataset.lower()}_cache_info.json")
    return manifest, report


def print_summary(manifest, report):
    print("\n=== Samples per split and label ===")
    print(pd.crosstab(manifest["split"], manifest["label"].map(config.CLASS_NAMES), margins=True))
    print("\n=== Examined files by status ===")
    print(report["status"].value_counts().to_string())
