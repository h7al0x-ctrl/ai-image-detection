"""
CIFAKE preparation.

Selects 20,000 CIFAKE images (balanced), checks corruption and duplicates,
writes data/manifests/cifake_manifest.csv, cifake_data_report.csv and the
224x224 cache.

  project train (7,000/class) + val (1,000/class)  <- official CIFAKE train folder
  project test  (2,000/class)                      <- official CIFAKE test folder

Run from the project root:
    python -m preprocessing.cifake_dataset            (skips if already done)
    python -m preprocessing.cifake_dataset --force    (rebuild)
"""
import argparse
import os
from pathlib import Path

import config
from preprocessing.sampling import (SampleGroup, list_images, outputs_exist, print_summary,
                                    select_and_cache)
from utils import ensure_dirs, log_run, set_seed

CLASS_DIRS = {"REAL": config.LABEL_REAL, "FAKE": config.LABEL_AI}


def find_cifake_class_dirs(root):
    """Locate <root>/.../train/REAL, train/FAKE, test/REAL, test/FAKE (case-insensitive)."""
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        parent = Path(dirpath).name.lower()
        if parent in ("train", "test"):
            for d in dirnames:
                if d.upper() in CLASS_DIRS:
                    found.setdefault((parent, d.upper()), Path(dirpath) / d)
    missing = [k for k in [(s, c) for s in ("train", "test") for c in CLASS_DIRS] if k not in found]
    if missing:
        raise FileNotFoundError(
            f"CIFAKE folders not found under {root}: missing {missing}.\n"
            "Expected structure: train/REAL, train/FAKE, test/REAL, test/FAKE")
    return found


def build_groups(root):
    dirs = find_cifake_class_dirs(root)
    groups = []
    for official_split in ("train", "test"):
        for cls, label in CLASS_DIRS.items():
            folder = dirs[(official_split, cls)]
            files = list_images(folder)
            print(f"official {official_split}/{cls}: {len(files)} files  ({folder})")
            groups.append(SampleGroup(
                name=f"CIFAKE/{official_split}/{cls}", dataset=config.CIFAKE, source_group="CIFAKE",
                generator="real" if label == config.LABEL_REAL else "cifake_ai", label=label,
                candidates=files, splits=list(config.CIFAKE_PER_CLASS[official_split].items()),
                folder=str(folder), jpeg_quality=config.CIFAKE_JPEG_QUALITY,
                extra={"official_split": official_split}))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = parser.parse_args()
    ensure_dirs()
    set_seed()

    if outputs_exist(config.CIFAKE, config.SPLITS) and not args.force:
        print("CIFAKE manifest and cache already exist - skipping (use --force to rebuild).")
        return
    root = Path(config.CIFAKE_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"CIFAKE root not found: {root}")
    groups = build_groups(root)
    manifest, report = select_and_cache(config.CIFAKE, groups, root)
    print_summary(manifest, report)
    log_run("student1_cifake", "build_manifest_and_cache", n_samples=len(manifest),
            status_counts=report["status"].value_counts().to_dict())
    print("\nDONE: data/manifests/cifake_manifest.csv, cifake_data_report.csv")


if __name__ == "__main__":
    main()
