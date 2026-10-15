# CricketCoach AI visual shot model

## Dataset used

The model uses the cached [rokmr/cricket-shot dataset](https://huggingface.co/datasets/rokmr/cricket-shot). The existing archive is `~/.cache/huggingface/datasets/rokmr-cricket-shot/cricketshot.tar.gz`; its verified size is 2,889,104,767 bytes and its gzip integrity check passed. It was not downloaded again. Only the actual AVI video members were extracted; AppleDouble resource-fork files and metadata were skipped.

The archive contains 1,750 labeled AVI videos and about 2.95 GB of uncompressed AVI media. Its official split is 1,250 train, 250 validation, and 250 test clips. The preparation utility validated all 1,750 videos, found no unlabeled or invalid videos, and detected 10 exact duplicate files. Those duplicates are excluded from training and evaluation, leaving 1,740 usable clips:

| Class | Train | Validation | Test | Usable total |
|---|---:|---:|---:|---:|
| cover | 125 | 25 | 25 | 175 |
| defense | 123 | 25 | 25 | 173 |
| flick | 125 | 25 | 25 | 175 |
| hook | 125 | 25 | 25 | 175 |
| late_cut | 125 | 25 | 25 | 175 |
| lofted | 125 | 25 | 25 | 175 |
| pull | 125 | 25 | 25 | 175 |
| square_cut | 119 | 24 | 24 | 167 |
| straight | 125 | 25 | 25 | 175 |
| sweep | 125 | 25 | 25 | 175 |
| **Total** | **1,242** | **249** | **249** | **1,740** |

The preparation manifest and validation report were generated under the ignored `dataset/prepared/cricket-shot/` directory, and original extracted files were under ignored `dataset/external/cricket-shot/cricketshot/`. Neither directory is included in this repository.

## Model and evaluation

The classifier uniformly samples eight RGB frames from each video, applies the pinned ImageNet EfficientNet-B0 V1 transform, encodes each frame with a frozen ImageNet EfficientNet-B0, and classifies the temporal sequence with a trained bidirectional GRU (128 hidden units) and linear output layer. It predicts the ten source labels shown in the table above. Pose extraction is separate and supplies technique coaching only.

Training used the official train and validation splits on Apple MPS. The best validation macro-F1 checkpoint was epoch 10; early stopping ended training at epoch 16. The final checkpoint was evaluated by running the production video inference path on all 249 held-out test videos:

- Accuracy: **78.31%**
- Macro precision: **79.65%**
- Macro recall: **78.28%**
- Macro F1: **0.7840**
- Test clips: **249**

| Test class | Correct / support | F1 |
|---|---:|---:|
| cover | 17 / 25 | 0.708 |
| defense | 23 / 25 | 0.920 |
| flick | 18 / 25 | 0.800 |
| hook | 17 / 25 | 0.773 |
| late_cut | 22 / 25 | 0.815 |
| lofted | 24 / 25 | 0.842 |
| pull | 16 / 25 | 0.627 |
| square_cut | 17 / 24 | 0.630 |
| straight | 19 / 25 | 0.809 |
| sweep | 22 / 25 | 0.917 |

These are results on this dataset's official held-out split. The source does not provide match/source IDs, so the split cannot establish that clips from the same match are separated. Treat the reported softmax scores as uncalibrated class scores, and do not read this result as proof of performance on unrelated broadcast or user-recorded footage.

## Artifacts and reproduction

- Weights: `backend/app/ml/models/cricketshot_temporal.pt` (20,661,247 bytes; ignored by Git and not included)
- Metadata and held-out metrics: `backend/app/ml/models/visual_model_meta.json`
- Training implementation: `ml_pipeline/train_visual_model.py`
- Inference implementation: `backend/app/ml/visual_classifier.py`

From the repository root, using the configured Python environment:

```bash
.venv/bin/python ml_pipeline/prepare_dataset.py --input-dir dataset/external/cricket-shot --output-dir dataset/prepared/cricket-shot
.venv/bin/python ml_pipeline/train_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --check-only
.venv/bin/python ml_pipeline/train_visual_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --device auto
```

The previous **local** API smoke check uploaded `cover_0001.avi` from the held-out test split through FastAPI, completed the analysis job, persisted the prediction and pose result, and retrieved it through the results and history endpoints. It predicted `cover` for that clip. The API model-status endpoint loaded this checkpoint and returned its test metrics. This was not a hosted-service test.

## Dataset rights

The Hugging Face dataset card declares Apache-2.0, but the upstream [CricShot10 project](https://github.com/ascuet/CricShot10/blob/main/README.md) describes research-only use and notes that copyright in source footage remains with the original rights holders. A dataset-card license does not itself establish commercial or redistribution rights for the underlying videos. Confirm rights before distributing clips or commercializing a model trained on them.
