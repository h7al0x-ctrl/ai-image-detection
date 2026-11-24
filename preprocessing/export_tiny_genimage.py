"""
OPTIONAL helper: build a GenImage folder from the public Hugging Face
subset "TheKernel01/Tiny-GenImage" (35,000 images, ~8.4 GB, licence CC BY-NC-SA 4.0,
derived from the Kaggle dataset yangsangtai/tiny-genimage).

Use this ONLY if you do not have the official GenImage files.

The subset labels every real image as generator "Real" (real images are not tied to
a generator). Real images are therefore distributed round-robin over the 8 generator
folders, so that every generator gets its own, non-overlapping real images - the same
idea as the official dataset, where every generator subset has its own ImageNet images.
Document this in the report.

Original file bytes are written unchanged (no re-encoding here).

Output layout (recognised automatically by genimage_dataset.py):
    <GENIMAGE_ROOT>/<Generator>/ai/*.png|jpg
    <GENIMAGE_ROOT>/<Generator>/nature/*.png|jpg

Run:  python -m preprocessing.export_tiny_genimage
"""
import io
from collections import Counter
from pathlib import Path

from PIL import Image

import config
from utils import log_run

REPO_ID = "TheKernel01/Tiny-GenImage"
HF_GENERATORS = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney",
                 5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong"}
EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}


def main():
    from datasets import Image as HFImage
    from datasets import load_dataset

    root = Path(config.GENIMAGE_ROOT)
    gens = config.GENIMAGE_ALL_GENERATORS
    for g in gens:
        (root / g / "ai").mkdir(parents=True, exist_ok=True)
        (root / g / "nature").mkdir(parents=True, exist_ok=True)

    counts, real_counter = Counter(), 0
    for split in ("train", "validation"):
        ds = load_dataset(REPO_ID, split=split).cast_column("image", HFImage(decode=False))
        print(f"{split}: {len(ds)} rows")
        for i, row in enumerate(ds):
            data = row["image"]["bytes"]
            fmt = Image.open(io.BytesIO(data)).format
            ext = EXT.get(fmt, ".img")
            source = HF_GENERATORS[row["generator"]]
            if row["label"] == 0 or source == "Real":
                gen = gens[real_counter % len(gens)]
                real_counter += 1
                out = root / gen / "nature" / f"{split}_{i:06d}{ext}"
                counts[(gen, "nature")] += 1
            else:
                gen = source
                out = root / gen / "ai" / f"{split}_{i:06d}{ext}"
                counts[(gen, "ai")] += 1
            out.write_bytes(data)
            if (i + 1) % 2000 == 0:
                print(f"   {i + 1}/{len(ds)}")

    print("\nImages written per generator:")
    for g in gens:
        print(f"  {g:11s} ai={counts[(g, 'ai')]:6d}  nature={counts[(g, 'nature')]:6d}")
    log_run("student2_genimage", "export_tiny_genimage", repo=REPO_ID,
            counts={f"{k[0]}/{k[1]}": v for k, v in counts.items()})


if __name__ == "__main__":
    main()
