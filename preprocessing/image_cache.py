"""
Loading the preprocessed cache + the PyTorch Dataset/DataLoader shared by
feature extraction and the CNN.
"""
import random

import numpy as np
import pandas as pd

import config
from evaluation.transforms import apply_condition
from preprocessing.preprocess import to_model_tensor
from preprocessing.sampling import cache_path, manifest_path


def load_manifest(dataset, split=None):
    path = manifest_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the {dataset} preparation step first.")
    df = pd.read_csv(path)
    if split is not None:
        df = df[df["split"] == split].sort_values("cache_index").reset_index(drop=True)
        if not (df["cache_index"].values == np.arange(len(df))).all():
            raise ValueError(f"{dataset}/{split}: cache_index is not 0..N-1")
    return df


def load_split(dataset, split):
    """Return (memory-mapped uint8 images [N,224,224,3], manifest rows in the same order)."""
    df = load_manifest(dataset, split)
    path = cache_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - restore or rebuild the image cache.")
    images = np.load(path, mmap_mode="r")
    if len(images) != len(df):
        raise ValueError(f"{dataset}/{split}: cache has {len(images)} images, manifest has {len(df)}")
    return images, df


try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class CachedImageDataset(Dataset):
        """Returns (normalised tensor 3x224x224, float label)."""

        def __init__(self, images, labels, condition="clean", hflip=False):
            self.images = images
            self.labels = np.asarray(labels, dtype=np.float32)
            self.condition = condition
            self.hflip = hflip

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            img = np.asarray(self.images[i])
            x = apply_condition(img, self.condition, sample_index=i)
            if self.hflip and random.random() < 0.5:
                x = x[:, ::-1, :]
            return to_model_tensor(x), torch.tensor(self.labels[i])

    def make_loader(dataset, split, condition="clean", train=False, batch_size=config.BATCH_SIZE):
        """DataLoader for one split. Shuffling + flips ONLY when train=True."""
        from utils import make_generator, seed_worker
        images, df = load_split(dataset, split)
        ds = CachedImageDataset(images, df["label"].values, condition,
                                hflip=train and config.CNN_HFLIP)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train,
                            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                            worker_init_fn=seed_worker, generator=make_generator() if train else None,
                            persistent_workers=False)
        return loader, df
except ImportError:  # allows data-only scripts to run without torch
    pass
