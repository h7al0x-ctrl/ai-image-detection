"""
GenImage preparation and generator validation.

Expected layout (official GenImage names are recognised automatically):
    <GENIMAGE_ROOT>/<generator folder>/.../ai/*.png      (AI-generated)
    <GENIMAGE_ROOT>/<generator folder>/.../nature/*.JPEG (real, ImageNet)
e.g.  GenImage/imagenet_ai_0508_adm/train/ai/...,  GenImage/ADM/nature/...

Selection (balanced, per generator):
    6 training generators : 1,000 train + 250 val   per class
    BigGAN, Midjourney    : 1,250 test               per class   (NEVER train/val)
All selected images (real and AI) are re-encoded as JPEG Q95 before resizing.

Outputs: genimage_manifest.csv, genimage_data_report.csv, generator_split.csv, cache.

Run from the project root:
    python -m preprocessing.genimage_dataset --check   (only verify folders/counts)
    python -m preprocessing.genimage_dataset           (build)
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from preprocessing.sampling import (SampleGroup, list_images, outputs_exist, print_summary,
                                    select_and_cache)
from utils import ensure_dirs, log_run, set_seed


def find_generator_folders(root):
    """Map each generator to exactly one folder under root (searches 2 levels)."""
    candidates = [p for p in root.iterdir() if p.is_dir()]
    candidates += [q for p in candidates for q in p.iterdir() if q.is_dir()]
    mapping, problems = {}, []
    for gen, keywords in config.GENIMAGE_FOLDER_KEYWORDS.items():
        matches = [p for p in candidates if any(k in p.name.lower() for k in keywords)]
        # prefer the shallowest match (the generator root, not its sub-folders)
        matches = [m for m in matches if not any(o != m and o in m.parents for o in matches)]
        if len(matches) == 1:
            mapping[gen] = matches[0]
        elif not matches:
            problems.append(f"{gen}: no folder found (keywords {keywords})")
        else:
            problems.append(f"{gen}: several folders match {[m.name for m in matches]}")
    return mapping, problems


def split_by_class(folder):
    """Return (ai_files, real_files, unknown_files) using the nearest class folder name."""
    ai, real, unknown = [], [], []
    for f in list_images(folder):
        parts = [p.lower() for p in f.relative_to(folder).parts[:-1]]
        cls = None
        for part in reversed(parts):
            if part in config.GENIMAGE_AI_DIR_NAMES:
                cls = "ai"
                break
            if part in config.GENIMAGE_REAL_DIR_NAMES:
                cls = "real"
                break
        (ai if cls == "ai" else real if cls == "real" else unknown).append(f)
    return ai, real, unknown


def inspect_generators(root):
    """Verify all 8 generators and write generator_split.csv. Returns file lists."""
    mapping, problems = find_generator_folders(root)
    rows, files = [], {}
    for gen in config.GENIMAGE_ALL_GENERATORS:
        is_test = gen in config.GENIMAGE_TEST_GENERATORS
        per = config.GENIMAGE_PER_CLASS_TEST_GEN if is_test else config.GENIMAGE_PER_CLASS_TRAIN_GEN
        need = sum(per.values())
        row = {"generator": gen, "role": "held_out_test" if is_test else "train_val",
               "folder": "", "ai_available": 0, "real_available": 0, "unlabeled_files": 0,
               "ai_required": need, "real_required": need, "extensions": "", "status": "MISSING"}
        if gen in mapping:
            folder = mapping[gen]
            ai, real, unknown = split_by_class(folder)
            files[gen] = (folder, ai, real)
            exts = Counter(p.suffix.lower() for p in ai + real)
            ok = len(ai) >= need and len(real) >= need
            row.update(folder=str(folder), ai_available=len(ai), real_available=len(real),
                       unlabeled_files=len(unknown), extensions=dict(exts),
                       status="OK" if ok else "NOT_ENOUGH_IMAGES")
            if gen == "ADM":  # explicit ADM verification requested by the proposal
                row["status"] += " (ADM verified: class folders found)" if ai and real else " (ADM: class folders missing!)"
        rows.append(row)
    table = pd.DataFrame(rows)
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.MANIFEST_DIR / "generator_split.csv", index=False)
    print(table[["generator", "role", "ai_available", "real_available", "ai_required", "status"]].to_string(index=False))
    for p in problems:
        print("PROBLEM:", p)
    return table, files, problems


def build_groups(files):
    groups = []
    # training generators first, so any near-duplicate is removed from the TEST side
    for gen in config.GENIMAGE_TRAIN_GENERATORS + config.GENIMAGE_TEST_GENERATORS:
        folder, ai, real = files[gen]
        per = config.GENIMAGE_PER_CLASS_TEST_GEN if gen in config.GENIMAGE_TEST_GENERATORS else config.GENIMAGE_PER_CLASS_TRAIN_GEN
        for label, cls_files in ((config.LABEL_AI, ai), (config.LABEL_REAL, real)):
            groups.append(SampleGroup(
                name=f"GenImage/{gen}/{'ai' if label else 'real'}", dataset=config.GENIMAGE,
                source_group=gen, generator=gen if label == config.LABEL_AI else "real",
                label=label, candidates=cls_files, splits=list(per.items()), folder=str(folder),
                jpeg_quality=config.GENIMAGE_JPEG_QUALITY))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify folders and counts")
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = parser.parse_args()
    ensure_dirs()
    set_seed()

    if not args.check and outputs_exist(config.GENIMAGE, config.SPLITS) and not args.force:
        print("GenImage manifest and cache already exist - skipping (use --force to rebuild).")
        return
    root = Path(config.GENIMAGE_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"GenImage root not found: {root}")
    table, files, problems = inspect_generators(root)
    if args.check:
        return
    if problems or not table["status"].str.startswith("OK").all():
        raise SystemExit("\nFix the generator problems above before building GenImage.")
    manifest, report = select_and_cache(config.GENIMAGE, build_groups(files), root)
    print_summary(manifest, report)
    print("\n=== Samples per generator folder and split ===")
    print(pd.crosstab(manifest["source_group"], manifest["split"]))
    log_run("student2_genimage", "build_manifest_and_cache", n_samples=len(manifest),
            status_counts=report["status"].value_counts().to_dict())
    print("\nDONE: genimage_manifest.csv, genimage_data_report.csv, generator_split.csv")


if __name__ == "__main__":
    main()
