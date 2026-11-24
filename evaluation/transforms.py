"""
Shared image conditions (8 robustness transformations + 32x32 condition).

Every transformation is applied to the cached 224x224 uint8 RGB image and returns
float32 HxWx3 in [0, 1]. ImageNet normalisation happens AFTERWARDS (to_model_tensor).
The SAME function is used by the feature extractor (SVM, RF) and by the CNN.

Gaussian noise: added in the [0,1] pixel range, then clipped to [0,1]. The random
generator is seeded with (SEED, sample_index), so every model sees EXACTLY the same
noisy images.
"""
import numpy as np
from PIL import Image

import config
from preprocessing.preprocess import get_resample, jpeg_reencode


def apply_condition(img_uint8, condition="clean", sample_index=0):
    if condition not in config.CONDITIONS:
        raise KeyError(f"unknown condition '{condition}'. Options: {list(config.CONDITIONS)}")
    size = config.IMAGE_SIZE
    if img_uint8.shape != (size, size, 3) or img_uint8.dtype != np.uint8:
        raise ValueError(f"expected uint8 ({size},{size},3) image, got {img_uint8.dtype} {img_uint8.shape}")
    spec = config.CONDITIONS[condition]
    kind = spec["type"]

    if kind == "none":
        out = img_uint8.astype(np.float32) / 255.0
    elif kind == "jpeg":
        out = np.asarray(jpeg_reencode(Image.fromarray(img_uint8), spec["quality"]), dtype=np.float32) / 255.0
    elif kind == "resize":
        small = Image.fromarray(img_uint8).resize((spec["size"], spec["size"]), get_resample())
        out = np.asarray(small.resize((size, size), get_resample()), dtype=np.float32) / 255.0
    elif kind == "noise":
        rng = np.random.default_rng([config.SEED, int(sample_index)])
        noise = rng.normal(0.0, spec["std"], size=img_uint8.shape).astype(np.float32)
        out = np.clip(img_uint8.astype(np.float32) / 255.0 + noise, 0.0, 1.0)
    else:
        raise ValueError(f"unsupported condition type {kind}")
    return out.astype(np.float32)


def describe_condition(condition):
    spec = config.CONDITIONS[condition]
    return spec["label"], spec["params"]


if __name__ == "__main__":
    # Quick self-test on a random image
    img = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    for name in config.CONDITIONS:
        out = apply_condition(img, name, sample_index=3)
        again = apply_condition(img, name, sample_index=3)
        assert out.shape == (224, 224, 3) and out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0
        assert np.array_equal(out, again), "conditions must be deterministic"
        diff = np.abs(out - img / 255.0).mean()
        print(f"{name:10s} mean |change| = {diff:.4f}")
    print("transforms self-test passed")
