"""
Shared image preprocessing. Used by BOTH the feature extractor and the CNN.

Pipeline for every image:
    open -> RGB -> [GenImage only: JPEG Q95 re-encode] -> shorter side to 224 (bicubic)
    -> center crop 224x224 -> uint8 cache
At model time:
    cached uint8 -> [condition / robustness transform] -> float [0,1] -> ImageNet normalisation
"""
import hashlib
import io

import numpy as np
from PIL import Image

import config

_RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}


def get_resample(name=config.RESIZE_INTERPOLATION):
    return _RESAMPLE[name]


def sha256_file(path, chunk_size=1 << 20):
    """Cryptographic hash of the raw file bytes (exact-duplicate detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path):
    """Open an image, fully decode it (detects corruption) and convert to RGB.

    Transparent images are composited on white. Returns (PIL.Image, metadata dict).
    Raises an exception for corrupted / unreadable files.
    """
    with Image.open(path) as im:
        im.load()
        meta = {"orig_format": im.format, "orig_width": im.size[0],
                "orig_height": im.size[1], "orig_mode": im.mode}
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.split()[-1])
        else:
            rgb = im.convert("RGB")
    if min(rgb.size) < 8:
        raise ValueError(f"image too small: {rgb.size}")
    return rgb, meta


def jpeg_reencode(img, quality):
    """Encode/decode a PIL RGB image as JPEG with the given quality (Pillow defaults otherwise)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        return im.convert("RGB")


def resize_and_center_crop(img, size=config.IMAGE_SIZE):
    """Scale the shorter side to `size` (bicubic) then center-crop size x size."""
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((new_w, new_h), get_resample())
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_to_cache_array(img, jpeg_quality=None):
    """RGB PIL image -> uint8 array (224, 224, 3) stored in the cache."""
    if jpeg_quality:
        img = jpeg_reencode(img, jpeg_quality)
    arr = np.asarray(resize_and_center_crop(img), dtype=np.uint8)
    if arr.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE, 3):
        raise ValueError(f"unexpected shape after preprocessing: {arr.shape}")
    return arr


def preprocess_path(path, dataset):
    """Full raw-file -> cache-array pipeline for one image (handy for demos/tests)."""
    img, _ = load_rgb(path)
    quality = config.GENIMAGE_JPEG_QUALITY if dataset == config.GENIMAGE else config.CIFAKE_JPEG_QUALITY
    return preprocess_to_cache_array(img, quality)


def to_model_tensor(img_float_hwc):
    """float32 HWC in [0,1] -> normalised torch tensor CHW (ImageNet mean/std)."""
    import torch
    if img_float_hwc.ndim != 3 or img_float_hwc.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {img_float_hwc.shape}")
    t = torch.from_numpy(np.ascontiguousarray(img_float_hwc.transpose(2, 0, 1))).float()
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def phash_int(img):
    """64-bit perceptual hash as a Python int (near-duplicate detection)."""
    import imagehash
    return int(str(imagehash.phash(img)), 16)


def hamming_to_many(value, hashes):
    """Hamming distance between one 64-bit hash and an array of uint64 hashes."""
    x = np.bitwise_xor(hashes.astype(np.uint64), np.uint64(value))
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
