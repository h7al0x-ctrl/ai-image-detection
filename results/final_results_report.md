# AI-Generated Image Detection - Final Results Report

Generated automatically on 2026-11-11 by `evaluation/report_tables.py`.

Research prototype, not a production forensic system. Positive class = AI-generated (1), threshold = 0.5, seed = 42.

## Data

**CIFAKE**

| split | AI-generated | Real |
|---|---|---|
| test | 2000 | 2000 |
| train | 7000 | 7000 |
| val | 1000 | 1000 |

**GenImage**

| split | AI-generated | Real |
|---|---|---|
| test | 2500 | 2500 |
| train | 5000 | 5000 |
| val | 1250 | 1250 |

## Experiment 1: CIFAKE -> CIFAKE

| subset | model | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | n |
|---|---|---|---|---|---|---|---|---|
| all | SVM | 0.9455 | 0.9464 | 0.9445 | 0.9454 | 0.9883 | 0.9455 | 4000 |
| all | Random Forest | 0.8892 | 0.8727 | 0.9115 | 0.8917 | 0.9593 | 0.8892 | 4000 |
| all | CNN (ResNet50) | 0.9590 | 0.9545 | 0.9640 | 0.9592 | 0.9930 | 0.9590 | 4000 |

## Experiment 2: GenImage 5 generators -> BigGAN + Midjourney

| subset | model | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | n |
|---|---|---|---|---|---|---|---|---|
| all | SVM | 0.7586 | 0.7900 | 0.7044 | 0.7448 | 0.8405 | 0.7586 | 5000 |
| BigGAN | SVM | 0.8200 | 0.8101 | 0.8360 | 0.8228 | 0.9037 | 0.8200 | 2500 |
| Midjourney | SVM | 0.6972 | 0.7625 | 0.5728 | 0.6542 | 0.7787 | 0.6972 | 2500 |
| all | Random Forest | 0.7498 | 0.7310 | 0.7904 | 0.7596 | 0.8309 | 0.7498 | 5000 |
| BigGAN | Random Forest | 0.8240 | 0.7637 | 0.9384 | 0.8421 | 0.9089 | 0.8240 | 2500 |
| Midjourney | Random Forest | 0.6756 | 0.6881 | 0.6424 | 0.6645 | 0.7480 | 0.6756 | 2500 |
| all | CNN (ResNet50) | 0.7814 | 0.8125 | 0.7316 | 0.7699 | 0.8609 | 0.7814 | 5000 |
| BigGAN | CNN (ResNet50) | 0.8436 | 0.8306 | 0.8632 | 0.8466 | 0.9195 | 0.8436 | 2500 |
| Midjourney | CNN (ResNet50) | 0.7192 | 0.7878 | 0.6000 | 0.6812 | 0.8029 | 0.7192 | 2500 |

## Experiment 3: CIFAKE model -> GenImage held-out test (no retraining)

| subset | model | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | n |
|---|---|---|---|---|---|---|---|---|
| all | SVM | 0.6264 | 0.6224 | 0.6428 | 0.6324 | 0.6573 | 0.6264 | 5000 |
| BigGAN | SVM | 0.6428 | 0.6325 | 0.6816 | 0.6561 | 0.6821 | 0.6428 | 2500 |
| Midjourney | SVM | 0.6100 | 0.6113 | 0.6040 | 0.6076 | 0.6325 | 0.6100 | 2500 |
| all | Random Forest | 0.4610 | 0.4760 | 0.7744 | 0.5896 | 0.5084 | 0.4610 | 5000 |
| BigGAN | Random Forest | 0.3700 | 0.4092 | 0.5856 | 0.4817 | 0.3465 | 0.3700 | 2500 |
| Midjourney | Random Forest | 0.5520 | 0.5285 | 0.9632 | 0.6825 | 0.6737 | 0.5520 | 2500 |
| all | CNN (ResNet50) | 0.4946 | 0.4973 | 0.9828 | 0.6604 | 0.4163 | 0.4946 | 5000 |
| BigGAN | CNN (ResNet50) | 0.4868 | 0.4933 | 0.9664 | 0.6531 | 0.2459 | 0.4868 | 2500 |
| Midjourney | CNN (ResNet50) | 0.5024 | 0.5012 | 0.9992 | 0.6676 | 0.5857 | 0.5024 | 2500 |

## Additional: 32x32 resolution (GenImage, train+test at 32x32)

| subset | model | accuracy | precision | recall | f1 | roc_auc | balanced_accuracy | n |
|---|---|---|---|---|---|---|---|---|
| all | SVM | 0.5976 | 0.6137 | 0.5268 | 0.5669 | 0.6430 | 0.5976 | 5000 |
| BigGAN | SVM | 0.5916 | 0.6126 | 0.4984 | 0.5496 | 0.6334 | 0.5916 | 2500 |
| Midjourney | SVM | 0.6036 | 0.6147 | 0.5552 | 0.5834 | 0.6530 | 0.6036 | 2500 |
| all | Random Forest | 0.6052 | 0.6111 | 0.5788 | 0.5945 | 0.6387 | 0.6052 | 5000 |
| BigGAN | Random Forest | 0.6172 | 0.6264 | 0.5808 | 0.6027 | 0.6436 | 0.6172 | 2500 |
| Midjourney | Random Forest | 0.5932 | 0.5964 | 0.5768 | 0.5864 | 0.6334 | 0.5932 | 2500 |
| all | CNN (ResNet50) | 0.6096 | 0.6314 | 0.5268 | 0.5744 | 0.6563 | 0.6096 | 5000 |
| BigGAN | CNN (ResNet50) | 0.5976 | 0.6227 | 0.4952 | 0.5517 | 0.6482 | 0.5976 | 2500 |
| Midjourney | CNN (ResNet50) | 0.6216 | 0.6392 | 0.5584 | 0.5961 | 0.6644 | 0.6216 | 2500 |

## Robustness (exp1) - accuracy

| transformation | cnn | rf | svm |
|---|---|---|---|
| Clean baseline | 0.9590 | 0.8892 | 0.9455 |
| JPEG Q90 | 0.9567 | 0.8922 | 0.9470 |
| JPEG Q70 | 0.9530 | 0.8888 | 0.9427 |
| JPEG Q50 | 0.9473 | 0.8845 | 0.9367 |
| Resize 112 | 0.9563 | 0.8910 | 0.9460 |
| Resize 56 | 0.9557 | 0.8880 | 0.9440 |
| Noise 0.01 | 0.9032 | 0.8405 | 0.7602 |
| Noise 0.03 | 0.7542 | 0.8047 | 0.7318 |
| Noise 0.05 | 0.6860 | 0.7917 | 0.7345 |

## Robustness (exp2) - accuracy

| transformation | cnn | rf | svm |
|---|---|---|---|
| Clean baseline | 0.7814 | 0.7498 | 0.7586 |
| JPEG Q90 | 0.7640 | 0.6784 | 0.7152 |
| JPEG Q70 | 0.7314 | 0.6278 | 0.6782 |
| JPEG Q50 | 0.7008 | 0.6168 | 0.6636 |
| Resize 112 | 0.6086 | 0.5878 | 0.6206 |
| Resize 56 | 0.5702 | 0.5328 | 0.5626 |
| Noise 0.01 | 0.7212 | 0.7094 | 0.7168 |
| Noise 0.03 | 0.5724 | 0.6430 | 0.6130 |
| Noise 0.05 | 0.5296 | 0.6244 | 0.5692 |

## Figures

- figures/confusion_matrices.png
- figures/roc_curves.png
- figures/metric_comparison.png
- figures/robustness_comparison.png
- figures/training_curves.png

## Limitations

- Results use a 20,000-image sample per dataset, not the full official CIFAKE/GenImage sets.
- CIFAKE images are 32x32 and are upsampled to 224x224; GenImage images are larger. Resolution differences can affect cross-dataset results (see the 32x32 experiment).
- All GenImage images were re-encoded as JPEG Q95 to remove the JPEG-vs-PNG format shortcut.
- BigGAN images are 128x128 in GenImage, so they are upsampled more than other generators.
- GPU training is not perfectly bit-for-bit deterministic.
