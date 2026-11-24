# AI-Generated Image Detection Using Machine Learning and Deep Learning

Research prototype (not a production forensic system) comparing three detectors of
AI-generated images:

| Model | Input | Owner |
|---|---|---|
| SVM | 2048-D ResNet50 (ImageNet1K_V1) GAP features | Student 3 |
| Random Forest | the same features | Student 4 |
| CNN | ResNet50 fine-tuned end-to-end (2 stages) | Student 5 |

The easiest way to run everything is the Colab notebook `AI_Image_Detection_Pipeline.ipynb`.

## Experiments

| ID | Train / validation | Test |
|---|---|---|
| exp1 | CIFAKE train / val | CIFAKE test |
| exp2 | GenImage: ADM, GLIDE, SD1.5, VQDM, Wukong | GenImage: BigGAN, Midjourney (reported separately + combined) |
| exp3 | exp1 models, **no retraining** | same GenImage test set as exp2 |
| res32 | GenImage (exp2 split) at 32x32 (224->32->224) | GenImage test at 32x32 |
| robustness | exp1 and exp2 models, no retraining | 8 transformations + clean baseline on the exp1/exp2 test sets |

## Decision log (agreed settings - change only with team approval)

| Topic | Decision |
|---|---|
| Sample size | CIFAKE 20,000; GenImage 17,500 (SD1.4 not available in Tiny-GenImage), 50% real / 50% AI |
| CIFAKE split | train 14,000 + val 2,000 from the **official train** folder; test 4,000 from the **official test** folder. This is a sample, not the full official 20,000-image test set |
| GenImage split | 5 training generators (ADM, GLIDE, SD1.5, VQDM, Wukong) x 2,500 (2,000 train + 500 val); SD1.4 removed - not in the Tiny-GenImage source (team decision); BigGAN + Midjourney x 2,500 = 5,000 test only |
| GenImage format bias | every selected GenImage image (real and AI) re-encoded as JPEG Q95 before resizing |
| Preprocessing | RGB -> shorter side 224 (bicubic) -> center crop 224 -> ImageNet normalisation; identical for features and CNN |
| Duplicates | SHA-256 exact duplicates and pHash near duplicates (distance <= 2) are rejected before splitting; logged in the data reports |
| Robustness | transformations applied to the 224x224 image before normalisation; noise in [0,1] with clipping, seeded per image |
| SVM | StandardScaler (train only) + SVC; grid rbf C {1,10}, linear C {0.01,0.1}; selected by validation ROC-AUC |
| RF | n_estimators {200,500} x max_depth {None,20}, max_features sqrt; selected by validation ROC-AUC |
| CNN | Adam; stage 1 frozen backbone lr 1e-4 (<=5 epochs); stage 2 + layer4 lr 1e-5 (<=15 epochs); patience 5; dropout 0.3; BCEWithLogitsLoss; horizontal flip only |
| Fixed | seed 42, batch 32, threshold 0.5, AI = 1, Real = 0 |
| 32x32 protocol | **proposed by the assistant - confirm against the original proposal** |

## Project structure

```
AI_Image_Detection/
├── config.py  utils.py  requirements.txt  README.md
├── data/ (CIFAKE/, GenImage/, cache/, manifests/)
├── preprocessing/  preprocess.py sampling.py cifake_dataset.py genimage_dataset.py
│                   image_cache.py leakage_audit.py export_tiny_genimage.py
├── eda/            run_eda.py
├── features/       feature_extractor.py extract_features.py   (saved/ = feature files)
├── models/         svm_model.py rf_model.py cnn_model.py feature_model_common.py
├── training/       train_cnn.py eval_cnn.py
├── evaluation/     metrics.py transforms.py predictions.py evaluate_models.py
│                   evaluate_robustness.py report_tables.py
├── checkpoints/    svm/ rf/ cnn/
├── results/        predictions/ tables/ figures/ robustness/ final_results_report.md
└── logs/           run_log.csv environment.json
```

## Run order (from the project root)

```bash
python utils.py                                   # environment check
python -m preprocessing.cifake_dataset            # Student 1
python -m preprocessing.genimage_dataset --check  # Student 2 (verify folders)
python -m preprocessing.genimage_dataset          # Student 2
python -m preprocessing.leakage_audit             # Students 1+2
python -m eda.run_eda                             # Students 1+2
python -m features.extract_features               # Student 3 (GPU recommended)
python -m models.svm_model                        # Student 3
python -m models.rf_model                         # Student 4
python -m training.train_cnn --tag cifake genimage res32   # Student 5 (GPU)
python -m training.eval_cnn                       # Student 5
python -m evaluation.evaluate_models              # Student 6
python -m evaluation.evaluate_robustness          # Student 6
python -m evaluation.report_tables                # Student 6
```

Every step skips work that already exists (use `--force` to redo), so you can resume after a Colab disconnect.
Self-tests: `python -m evaluation.metrics`, `python -m evaluation.transforms`.

## Main outputs

- `data/manifests/`: cifake_manifest.csv, genimage_manifest.csv, generator_split.csv, *_data_report.csv, leakage_audit.csv
- `checkpoints/`: svm_*.joblib, rf_*.joblib, cnn_*.pt (tags: cifake, genimage, res32)
- `results/predictions/`: svm_predictions.csv, rf_predictions.csv, cnn_predictions.csv
- `results/tables/`: all_metrics.csv, final_comparison.csv, <model>_<exp>.csv, tuning tables, rf_error_analysis.csv, cnn histories
- `results/robustness/`: robustness_results.csv, <model>_robustness.csv
- `results/figures/`: confusion_matrices.png, roc_curves.png, metric_comparison.png, robustness_comparison.png, training_curves.png, EDA figures
- `results/final_results_report.md`

## Data sources

- CIFAKE: Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images` (train/test, REAL/FAKE, 32x32).
- GenImage: official release (github.com/GenImage-Dataset/GenImage (see also github.com/weili-thu/genimage)) with folders `<generator>/{train,val}/{ai,nature}`;
  or, if the official files are not available, the subset `TheKernel01/Tiny-GenImage` on Hugging Face
  (`python -m preprocessing.export_tiny_genimage`). In that subset real images are not tied to a generator,
  so they are distributed evenly over the 8 generator folders. State which source you used in the report.

## Limitations

- 20,000-image samples per dataset, not the full datasets.
- CIFAKE is 32x32 (upsampled to 224); GenImage images are 128-1024 px. BigGAN (128 px) is upsampled more than the others.
- SVM probabilities come from Platt scaling; GPU training is not perfectly deterministic.
