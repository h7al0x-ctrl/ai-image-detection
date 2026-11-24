"""
ResNet50 feature extractor (ImageNet1K_V1, output of Global Average Pooling).

Image -> ResNet50 (fc removed) -> GAP -> 2048-D vector.
Uses the SAME preprocessing (cache + transforms + ImageNet normalisation) as the CNN.
"""
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

import config


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Identity()  # keep the 2048-D pooled features
        self.model = model

    def forward(self, x):
        return self.model(x)


@torch.no_grad()
def extract(model, loader, device):
    """Run the extractor over a DataLoader -> float32 array [N, 2048]."""
    model.eval()
    chunks = []
    for i, (x, _) in enumerate(loader):
        feats = model(x.to(device, non_blocking=True))
        chunks.append(feats.float().cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"   batch {i + 1}/{len(loader)}")
    features = np.concatenate(chunks).astype(np.float32)
    if features.shape[1] != config.FEATURE_DIM:
        raise ValueError(f"expected {config.FEATURE_DIM}-D features, got {features.shape}")
    return features
