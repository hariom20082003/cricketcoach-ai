# CricketCoach AI dataset guide

## Dataset access and local preparation

The source [rokmr/cricket-shot dataset](https://huggingface.co/datasets/rokmr/cricket-shot) and its derived files are intentionally not included in the repository. The upstream CricShot10 project describes research-only use and the footage remains subject to the rights of its original owners. Confirm permission before downloading, preparing, training on, or hosting these artifacts.

The model report records the previously validated dataset and split counts. This does not mean the archive, videos, or prepared manifest are present in a clean clone. In the original local workspace, the archive had been cached at:

- Archive: `~/.cache/huggingface/datasets/rokmr-cricket-shot/cricketshot.tar.gz`
- Expected and verified size: 2,889,104,767 bytes
- Format/integrity: gzip-compressed tar; `gzip -t` passed
- Extracted media: 1,750 AVI files, 2,947,746,195 bytes
- Official splits: 1,250 train / 250 validation / 250 test (125 / 25 / 25 per class before duplicate removal)
- Labels: `cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`, `square_cut`, `straight`, `sweep`

For a permitted local copy, keep raw videos under the ignored `dataset/external/` directory and generated manifests under ignored `dataset/prepared/`. Before extracting an archive, check available disk space, inspect its member list and total uncompressed bytes, and retain at least 5 GiB free. The prior archive's AVI members required about 2.95 GB. Metadata and AppleDouble resource-fork files must not be counted as videos.

## Prepare and validate

From the project root:

```bash
.venv/bin/python ml_pipeline/prepare_dataset.py --input-dir dataset/external/cricket-shot --output-dir dataset/prepared/cricket-shot
.venv/bin/python ml_pipeline/train_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --check-only
```

The first command decodes and validates each video, preserves the official source split and labels, checks exact duplicates, and writes the manifest, summary, and validation report under ignored `dataset/prepared/cricket-shot/`. In the prior preparation, all 1,750 videos were valid and labeled; 10 exact duplicate files were excluded, leaving 1,740 usable examples (1,242 train / 249 validation / 249 test). Aggregate class-level counts and test performance are recorded in [VISUAL_MODEL.md](VISUAL_MODEL.md). The generated per-clip manifest and validation report are not committed.

The compatibility fix treats official directory labels and split paths as authoritative, ignores macOS `._*` sidecars, and prevents repeated clip basenames from allowing annotations to overwrite split labels. The source archive's split is preserved; no random split is created.

## Train and run inference

```bash
.venv/bin/python ml_pipeline/train_visual_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --device auto
```

The visual-temporal model samples RGB frames and uses a frozen ImageNet EfficientNet-B0 encoder with a trained bidirectional GRU. It saves its weights and evaluation metadata under `backend/app/ml/models/`. The API loads the matching architecture and metadata at startup. See [VISUAL_MODEL.md](VISUAL_MODEL.md) for evaluation details and artifact paths.

## Split and rights limitations

The dataset does not provide match/source IDs. Therefore, clips originating from the same match may occur across official splits; this potential leakage cannot be audited from the supplied files. Test scores should be treated as dataset-specific, not as proof of out-of-source generalization.

The [Hugging Face card](https://huggingface.co/datasets/rokmr/cricket-shot) declares Apache-2.0. The source [CricShot10 project](https://github.com/ascuet/CricShot10/blob/main/README.md) describes research-only use and states that source footage remains subject to rights held by original owners. The rights question remains unresolved. Confirm footage and model rights before redistributing any source or derived artifacts, including model weights.
