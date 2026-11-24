"""
CNN: ResNet50 (ImageNet1K_V1) + GAP + Dropout(0.3) + Linear(1) + Sigmoid.

forward() returns LOGITS; the sigmoid is applied by BCEWithLogitsLoss during training
and by predict_proba() at inference (mathematically identical, numerically safer).

Stage 1: whole backbone frozen (BatchNorm layers kept in eval mode), head trained.
Stage 2: layer4 (last ResNet block) unfrozen as well.
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

import config


class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True, dropout=config.CNN_DROPOUT):
        super().__init__()
        self.stage = 1
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)       # includes Global Average Pooling
        in_features = self.backbone.fc.in_features       # 2048
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, 1)
        self.set_stage(1)

    def forward(self, x):
        return self.classifier(self.dropout(self.backbone(x))).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    def set_stage(self, stage):
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        self.stage = stage
        for p in self.backbone.parameters():
            p.requires_grad = False
        if stage == 2:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def train(self, mode=True):
        """Frozen parts stay in eval mode so their BatchNorm statistics do not change."""
        super().train(mode)
        if mode:
            self.backbone.eval()
            if self.stage == 2:
                self.backbone.layer4.train()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_cnn(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ResNet50Binary(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt
