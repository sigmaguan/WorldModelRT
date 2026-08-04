# WorldModelRT

WorldModelRT models continuous tumor-response dynamics under fraction-level radiotherapy actions. A time-aware Transformer encodes longitudinal tumor, anatomy, clinical, and treatment observations. An LQ-constrained latent dynamics model combines radiation cell kill, learned residual dynamics, Gompertzian repopulation, and reoxygenation. The model supports forward and counterfactual treatment-schedule simulation during training and evaluation.

## Installation

Python 3.11 and CUDA 12.1 are required for the pinned environment.

```bash
python -m pip install .
```

```bash
conda env create -f environment.yml
conda activate worldmodelrt
```

```bash
docker build -t worldmodelrt .
```

## Data

Canonical and currently reachable dataset landing pages are listed in `datasets.txt`. RADCURE contains 3,346 CT cases and uses a patient-level 2,677/335/334 split. QIN-HeadNeck contains 151 PET/CT cases and uses a 121/15/15 split. HNTS-MRG v1 contains 150 publicly downloadable paired pre-RT and mid-RT MRI training cases; the paper's 200-case analysis additionally uses the challenge evaluation cohort. Dataset terms and de-identification requirements remain binding.

Prepared patient archives contain `states`, `actions`, `times`, `targets`, `fraction_times`, `fraction_doses`, and `oxygen` arrays. Split membership must be assigned at patient level before normalization. RADCURE intermediate targets use a 0.7 LQ and 0.3 endpoint-interpolation mixture. Clinical outcomes must not be copied into model inputs.

## Training

The three stages are run in sequence:

```bash
bash commands/train_all.sh
```

Stage 1 uses 50,000 synthetic trajectories for 200 epochs. Stage 2 uses RADCURE for 100 epochs. Stage 3 uses QIN-HeadNeck and HNTS-MRG for 50 epochs. AdamW uses learning rate `1e-4`, weight decay `1e-5`, cosine annealing with warm restarts, and gradient clipping at norm 1.0. The reported configuration uses one NVIDIA A100 80GB GPU. Ten experiment seeds are fixed in `recipes/training.yaml`.

## Evaluation

Primary outputs are relative tumor-volume MAE, anatomy Dice, trajectory Pearson correlation, survival concordance index, treatment sensitivity, and physical-violation rate. Patient-paired bootstrap intervals use 1,000 resamples. Primary comparisons use Holm-Bonferroni correction and exploratory comparisons use Benjamini-Hochberg correction.

Reference aggregate values are 7.8% volume MAE, 0.957 Dice, and 0.903 trajectory correlation. Per-dataset values should be compared only with identical patient splits and target construction. The expected inference time for a 30-fraction trajectory is approximately 0.9 seconds on the reported GPU.

## Compute budget

The formal configuration requires one NVIDIA A100 with 80GB VRAM. Storage depends on downloaded DICOM and derived arrays; HNTS-MRG v1 alone is approximately 15GB. Full training comprises 200 synthetic, 100 population-transfer, and 50 temporal fine-tuning epochs for each reported seed. Wall-clock duration was not reported and should be measured on the target system.

## Clinical scope

Outputs are research simulations and are not treatment recommendations. Counterfactual schedules require review of tumor-control probability, normal-tissue complication probability, OAR BED, treatment duration, and physical plausibility. The public datasets are de-identified; re-identification and reconstruction of identifying anatomy are prohibited.
