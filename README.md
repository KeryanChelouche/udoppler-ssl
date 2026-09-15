# Parameter-Efficient Adaptation of Self-Supervised Vision Backbones Survives Distribution Shift in Radar Micro-Doppler HAR

Research code for our manuscript prepared for **ICASSP 2027**.

**Keryan Chelouche, Petr Dobias, Nistor Grozavu, Olivier Romain**

We compare full fine-tuning, frozen linear probing, and parameter-efficient
fine-tuning (PEFT) for radar micro-Doppler human activity recognition (HAR).
Evaluation covers subject-disjoint in-distribution cross-validation and
few-shot transfer under changes in age cohort, sensing orientation, and dataset.

Our recipe, **DINOv3-PiSSA**, combines a self-supervised DINOv3 vision transformer
with PiSSA-initialized adapters on all attention and MLP projections
(rank 8, scaling coefficient 16).

## Main findings

- **In-distribution rankings did not persist under shift.** Full fine-tuning
  suffered negative cross-dataset transfer, while frozen features under-adapted
  to the orientation change. Only PEFT avoided both failure regimes in the
  main experiments.
- **DINOv3-PiSSA ranked first among PEFT methods in the main evaluations.**
  Its Small variant reached 83.56% cross-orientation and 66.92% cross-dataset
  accuracy with ten labeled target examples per class.
- **Comparable accuracy with fewer parameters.** The Small variant matched
  Base-size SelaFD's in-distribution accuracy, with slightly higher means
  (+0.73/+0.61 percentage points on Glasgow/MAD), 4.5× fewer total parameters,
  and approximately 3% trainable parameters. Against size-matched SelaFD-S,
  transfer gains were 2.1, 6.6, and 8.9 percentage points across the three shifts.

These transfer results use a target-trained linear probe. Direct transfer of
the source classification head without target labels is evaluated separately
in [Zero-shot transfer](#zero-shot-transfer).

## Contents

- [Installation](#installation)
- [Datasets and preprocessing](#datasets-and-preprocessing)
- [Evaluation protocol](#evaluation-protocol)
- [Reproducing results](#reproducing-results)
- [Main results](#main-results)
- [Additional experiments](#additional-experiments)
- [Citation](#citation)
- [Acknowledgments and licenses](#acknowledgments-and-licenses)

## Installation

Run the commands below from a Bash-compatible shell. The experiment scripts
use `.venv/bin/python`, so keep the virtual environment at `.venv` in the
repository root.

The recorded environment used **Python 3.14, PyTorch 2.10.0, torchvision 0.25.0,
and the CUDA 13.0 PyTorch build**. Package metadata declares Python 3.11 or
newer; dependencies are pinned in [pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/KeryanChelouche/udoppler-ssl.git
cd udoppler-ssl
python -m venv .venv
source .venv/bin/activate

# Install the CUDA build before installing the project dependencies.
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[notebook]"
```

For other hardware, use the appropriate build from the
[official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2100)
while retaining the pinned PyTorch and torchvision versions.

### DINOv3 model access

The experiments use the gated Hugging Face checkpoints
[DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
and [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
Request access using your Hugging Face account, then create the local
configuration file:

```bash
cp .env.example .env
```

Set `HF_TOKEN` in `.env` to a token from the account with model access.

## Datasets and preprocessing

The datasets have complementary acquisition settings and overlapping activity
sets. Glasgow contains six activities, while MAD contains ten, counting
arbitrary-trajectory walking separately. Five activities are shared: walking,
sitting down, standing up, picking up an object, and falling.

### Glasgow

Download the public
[Radar signatures of human activities](https://doi.org/10.5525/gla.researchdata.848)
release. Place the seven numbered campaign folders or ZIP archives under
`data/Glasgow_STFT_MAT/`, preserving their original names. The converter reads
the precomputed spectrogram `.mat` files; raw `.dat` files are not required.

```bash
python data/Glasgow_STFT_MAT/prepare_glasgow.py
```

Output: `data/Glasgow/`, with **2,081 magnitude spectrograms**.

The converter standardizes duration to 365 time bins (approximately 5 s).
Ten-second walking recordings yield two windows. It also handles three longer
walking recordings, one longer sitting-down recording, and a duplicate in the
release. See [the conversion script](data/Glasgow_STFT_MAT/prepare_glasgow.py)
for the exact rules.

### MAD

Access to MAD must be requested by emailing
[olivier.romain@cyu.fr](mailto:olivier.romain@cyu.fr).
Place the `.mat` files under `data/MAD_STFT_MAT/`, retaining their participant
subdirectories, then run:

```bash
python data/MAD_STFT_MAT/prepare_mad.py
```

Output: `data/MAD/`, with **17,716 spectrograms**. The converter retains the
complex STFTs from both receive channels as separate `.npy` files. Magnitudes
are taken by the dataset loader during evaluation.

Both converters skip existing output files. They support `--dry-run` to inspect
the conversion without writing outputs.

### Common model inputs

Spectrogram magnitudes are log-compressed, resized to 224 × 224, replicated
across three channels, min–max normalized per sample, and standardized using
ImageNet channel statistics. We use no data augmentation. Random cropping and
flipping are also disabled in our SelaFD reimplementation.

## Evaluation protocol

### In-distribution evaluation

Table 1 uses five-fold subject-disjoint cross-validation with
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
Fine-tuned models use training seed 42. Reported means and standard deviations
are calculated across the five test folds.

Glasgow participant identifiers are grouped across acquisition campaigns.
Reused identifiers therefore keep recordings together even when they refer
to different people. MAD recordings from both receive channels are grouped
by participant.

### Few-shot transfer

For each source–target pair:

1. Fine-tune the adapted encoder on the source data. Frozen baselines retain
   their pretrained weights.
2. Freeze the encoder and partition target participants into five stratified,
   subject-disjoint folds using the same splitter and seed as above.
3. In each fold, draw ten labeled examples per class from target-training
   participants. Fit a new linear probe on their embeddings and evaluate it
   on the held-out target participants.
4. Repeat the support draw ten times per fold, using identical draws across
   methods and source fine-tuning seeds.

The probe is logistic regression with `C=1`, the `lbfgs` solver, and
`StandardScaler`. The scaler and classifier are fit only on the support set.

For each source fine-tuning seed, scores are averaged over **5 folds × 10
support draws**. Tables 2 and 3 then report the mean and standard deviation
across the three resulting seed means (source seeds 0, 1, and 2). The
within-seed spread across probe fits is stored in the JSON files but is not
the standard deviation shown in those tables.

Frozen baselines use the same target folds and seeded support draws. They
have no source fine-tuning seeds to aggregate, so we report their mean over
folds and draws without a standard deviation across source seeds.

### Transfer settings

| Shift | Source → target | Evaluated classes |
|---|---|---|
| Cross-age | Glasgow campaigns 1–5 (ages 21–44) → older participants from campaigns 6–7 (ages 45–98) | Five classes; falls are absent from the older cohort |
| Cross-orientation | MAD toward/away → MAD right-to-left | Nine classes; arbitrary-trajectory walking is excluded |
| Cross-dataset | Glasgow → MAD | Five shared activities |

For cross-age evaluation, five younger participants are excluded from the
target campaigns.

## Reproducing results

### Rebuild tables and Figure 1 from saved results

The repository includes the result JSON files. You can rebuild the reported
tables and figure without downloading the radar datasets or rerunning training:

```bash
jupyter lab paper_tables.ipynb
```

Run the notebook from the repository root after completing installation and
DINOv3 access setup. Its parameter-count cell instantiates pretrained models,
so running all cells can download model weights even when only rebuilding
tables. The notebook produces accuracy and macro-F1 tables and LaTeX output.

### Run the main experiments

After preparing the datasets, run:

```bash
bash scripts/run_in_distribution_comparison.sh
N_SHOTS=10 PROBES=linear bash scripts/run_final_method_comparison.sh
N_SHOTS=10 PROBES=linear bash scripts/run_ablation.sh
```

These correspond to Tables 1, 2, and 3. Setting `N_SHOTS=10 PROBES=linear`
restricts the transfer runs to the readout reported in the paper. Without
these overrides, the transfer scripts also evaluate other shot counts and
k-nearest-neighbor probes.

**Existing result JSONs are skipped.** Because reference results are included
in a fresh clone, these commands will skip the completed configurations.
To recompute a configuration, move its JSON file out of the relevant result
directory before running the script.

Transfer runs reuse matching checkpoints in `results/cache/checkpoints/`.
Keeping these checkpoints recomputes target evaluation without source
retraining. To retrain from scratch, also move aside the matching checkpoint.
An interrupted run can be relaunched to process configurations whose result
JSONs are still missing.

Checkpoints, feature caches, and radar data are not included in the repository.
A complete experiment run takes several days on the laptop GPU used for the
paper (NVIDIA RTX PRO 1000 Blackwell, 8 GB). MAD source fine-tuning accounts
for much of the training time.

## Main results

All accuracy and macro-F1 values below are percentages. `IN` denotes
supervised ImageNet pretraining. `S` and `B` denote Small and Base variants.
Trainable percentages are relative to total model parameters.
`SelaFD*` is the Base-size reimplementation; `SelaFD-S*` is its Small variant.
Both use our evaluation protocol and preprocessing, so these are not the
original SelaFD paper's reported scores.

### Table 1 — In-distribution

Mean ± standard deviation across five subject-disjoint folds, with training
seed 42 for fine-tuned models.

| Method | Trainable | Total | Glasgow Acc | Glasgow macro-F1 | MAD Acc | MAD macro-F1 |
|---|---|---|---|---|---|---|
| *Full fine-tuning* | | | | | | |
| ResNet50 | 23.52 M (100%) | 23.52 M | 94.57 ± 1.08 | 94.07 ± 1.16 | 97.94 ± 0.59 | 97.93 ± 0.60 |
| DINOv3-S (full FT) | 21.60 M (100%) | 21.60 M | 94.04 ± 0.78 | 93.38 ± 0.79 | 97.86 ± 0.61 | 97.82 ± 0.64 |
| *Linear probing on frozen models* | | | | | | |
| ViT-S (IN) | probe only | 21.67 M | 87.16 ± 1.58 | 85.91 ± 1.39 | 63.51 ± 1.24 | 63.05 ± 1.64 |
| ViT-B (IN) | probe only | 85.80 M | 88.26 ± 2.03 | 86.86 ± 2.25 | 67.36 ± 1.88 | 67.02 ± 2.23 |
| DINOv3-S | probe only | 21.60 M | 86.35 ± 1.52 | 84.90 ± 1.73 | 73.34 ± 1.00 | 72.85 ± 0.97 |
| DINOv3-B | probe only | 85.66 M | 87.12 ± 1.28 | 85.82 ± 1.44 | 74.83 ± 0.93 | 74.24 ± 1.13 |
| *PEFT* | | | | | | |
| ViT-S + LoRA | 0.22 M (1.02%) | 21.89 M | 91.92 ± 1.20 | 91.01 ± 1.24 | 94.14 ± 0.73 | 93.91 ± 0.69 |
| SelaFD-S\* | 3.63 M (14.35%) | 25.29 M | 90.77 ± 1.19 | 89.68 ± 1.21 | 95.96 ± 0.66 | 95.87 ± 0.59 |
| SelaFD\* | 14.34 M (14.32%) | 100.13 M | 92.06 ± 0.72 | 91.10 ± 0.90 | 96.82 ± 0.73 | 96.74 ± 0.76 |
| **DINOv3-PiSSA-S** | 0.67 M (2.99%) | 22.26 M | 92.79 ± 1.37 | 92.08 ± 1.58 | 97.43 ± 0.47 | 97.38 ± 0.49 |
| **DINOv3-PiSSA-B** | 1.33 M (1.53%) | 86.99 M | 93.85 ± 1.15 | 93.24 ± 1.18 | 97.63 ± 0.51 | 97.61 ± 0.52 |

### Table 2 — 10-shot transfer under distribution shift

Adapted models: mean ± standard deviation across three source fine-tuning
seed means. Frozen baselines: mean over the same target folds and support
draws. See [Evaluation protocol](#evaluation-protocol) for the averaging steps.

| Method | Cross-age Acc | Cross-age macro-F1 | Cross-orient. Acc | Cross-orient. macro-F1 | Cross-dataset Acc | Cross-dataset macro-F1 |
|---|---|---|---|---|---|---|
| *Full fine-tuning* | | | | | | |
| ResNet50 | 79.12 ± 0.32 | 76.43 ± 0.48 | 79.37 ± 1.66 | 79.25 ± 1.66 | 33.07 ± 3.71 | 30.36 ± 3.00 |
| DINOv3-S (full FT) | 82.10 ± 0.73 | 79.85 ± 0.74 | 81.35 ± 0.69 | 81.19 ± 0.65 | 30.93 ± 2.97 | 28.90 ± 2.80 |
| *Linear probing on frozen models* | | | | | | |
| ViT-S (IN) | 66.25 | 63.35 | 29.29 | 29.11 | 55.28 | 52.20 |
| DINOv3-S | 64.68 | 59.60 | 38.22 | 38.36 | 63.93 | 61.43 |
| *PEFT* | | | | | | |
| SelaFD-S\* | 75.34 ± 1.01 | 72.55 ± 0.96 | 77.00 ± 0.21 | 76.69 ± 0.24 | 57.98 ± 0.79 | 55.10 ± 0.81 |
| ViT-S + LoRA | 77.22 ± 0.78 | 74.08 ± 0.62 | 75.83 ± 0.83 | 75.53 ± 0.87 | 57.31 ± 0.92 | 54.24 ± 1.00 |
| **DINOv3-PiSSA-S** | 77.46 ± 0.29 | 75.19 ± 0.28 | 83.56 ± 1.02 | 83.35 ± 1.04 | 66.92 ± 0.38 | 64.39 ± 0.57 |

### Table 3 — Pretraining × adapter recipe

Same 10-shot protocol and seed aggregation as Table 2. This comparison changes
pretraining and adapter recipe. The recipe includes both initialization and
layer coverage; the next experiment separates those two choices.

| Pretrained encoder + adapter | Cross-orient. Acc | Cross-orient. macro-F1 | Cross-dataset Acc | Cross-dataset macro-F1 |
|---|---|---|---|---|
| ViT-S (IN) + LoRA (attn) | 75.83 ± 0.83 | 75.53 ± 0.87 | 57.31 ± 0.92 | 54.24 ± 1.00 |
| ViT-S (IN) + PiSSA (all) | 81.19 ± 1.50 | 80.98 ± 1.52 | 58.15 ± 1.27 | 55.28 ± 1.13 |
| DINOv3-S + LoRA (attn) | 80.49 ± 1.11 | 80.26 ± 1.06 | 65.54 ± 1.73 | 63.15 ± 1.80 |
| **DINOv3-S + PiSSA (all)** | 83.56 ± 1.02 | 83.35 ± 1.04 | 66.92 ± 0.38 | 64.39 ± 0.57 |

## Additional experiments

### Initialization × coverage ablation

This 2×2 experiment separates initialization from layer coverage using
DINOv3-S under the cross-dataset shift. It uses the same 10-shot evaluation
and three-seed aggregation as Tables 2 and 3.

```bash
bash scripts/run_ablation_2x2.sh
```

Existing JSONs are skipped here too. Move the relevant result files aside
before recomputing them.

| Initialization | Coverage | Acc | macro-F1 |
|---|---|---|---|
| LoRA (B = 0) | attention | 65.54 ± 1.73 | 63.15 ± 1.80 |
| LoRA (B = 0) | all layers | 64.87 ± 1.82 | 62.27 ± 1.86 |
| PiSSA (SVD) | attention | 62.73 ± 3.13 | 59.79 ± 3.32 |
| **PiSSA (SVD)** | **all layers** | 66.92 ± 0.38 | 64.39 ± 0.57 |

Neither change alone improves mean accuracy over attention-only LoRA in this
experiment. The highest mean is obtained by combining PiSSA initialization
with all-layer coverage.

### Zero-shot transfer

This evaluation applies the source-trained classification head directly to
the target, without fitting a target probe or using labeled target examples
for adaptation. Mean ± standard deviation is reported across three source
fine-tuning seeds.

```bash
python scripts/run_zero_shot.py
```

The script uses source checkpoints in `results/cache/checkpoints/` and never
trains a model. Configurations with existing zero-shot JSONs or missing source
checkpoints are skipped. To recompute zero-shot scores in a fresh clone,
first generate the source checkpoints using the main transfer experiments
(move their reference JSONs aside so those runs execute). Then move aside the
corresponding zero-shot JSONs and run the command above.

| Method | Cross-age Acc | Cross-age macro-F1 | Cross-orient. Acc | Cross-orient. macro-F1 | Cross-dataset Acc | Cross-dataset macro-F1 |
|---|---|---|---|---|---|---|
| ResNet50 | 75.33 ± 1.67 | 73.80 ± 1.56 | 47.18 ± 3.46 | 40.88 ± 4.16 | 25.46 ± 5.58 | 8.05 ± 1.47 |
| DINOv3-S (full FT) | 79.13 ± 2.33 | 77.75 ± 2.06 | 39.85 ± 1.45 | 34.69 ± 2.37 | 24.01 ± 4.50 | 11.58 ± 2.01 |
| SelaFD-S\* | 68.73 ± 0.75 | 67.36 ± 1.00 | 43.44 ± 2.10 | 36.57 ± 2.00 | 21.88 ± 1.67 | 11.49 ± 2.90 |
| ViT-S + LoRA | 69.10 ± 2.16 | 66.98 ± 2.12 | 44.84 ± 0.60 | 38.02 ± 0.58 | 15.54 ± 3.07 | 8.45 ± 0.60 |
| **DINOv3-PiSSA-S** | 70.36 ± 2.14 | 69.40 ± 1.74 | 44.07 ± 0.23 | 37.29 ± 0.17 | 25.25 ± 2.36 | 16.80 ± 1.18 |

Every method's mean cross-dataset accuracy is below the 29.4% majority-class
baseline. The best cross-orientation zero-shot accuracy is 47.18%, compared
with 83.56% for the best method with 10-shot target readout. These results
motivate the distinction between few-shot and direct source-head transfer.

## Citation

For the current manuscript, use the provisional citation below. We will
update it when publication details are available:

```bibtex
@unpublished{chelouche2026peft,
  title  = {Parameter-Efficient Adaptation of Self-Supervised Vision Backbones
            Survives Distribution Shift in Radar Micro-Doppler {HAR}},
  author = {Chelouche, Keryan and Dobias, Petr and Grozavu, Nistor and Romain, Olivier},
  year   = {2026},
  note   = {Manuscript prepared for ICASSP 2027},
  url    = {https://github.com/KeryanChelouche/udoppler-ssl}
}
```

## Acknowledgments and licenses

This work was supported by the French National Research Agency (ANR) under
grant 24-LCV2-0020-01 / SmartGaitLab.

The experiments build on [DINOv3](https://arxiv.org/abs/2508.10104),
[LoRA](https://arxiv.org/abs/2106.09685),
[PiSSA](https://arxiv.org/abs/2404.02948), and
[SelaFD](https://arxiv.org/abs/2502.04740).

Repository code is distributed under the [MIT License](LICENSE).
Glasgow is released under CC BY 4.0. MAD access must be requested from
[olivier.romain@cyu.fr](mailto:olivier.romain@cyu.fr).
Dataset terms and pretrained-model licenses apply separately from the code license.
