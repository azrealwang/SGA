# Structural Gradient Alignment (SGA)

Reference implementation of **Structural Gradient Alignment**, a robustness metric that
measures the structural alignment between input content and objective-conditioned local
model sensitivity. SGA ranks the adversarial robustness of models inside a homogeneous
model pool from a handful of unlabeled probe images, using one input gradient per
model-probe pair and no adversarial example generation.

## How the method is organized

SGA has two stages, and this repository keeps them in separate scripts because the second
stage is cheap and is meant to be re-run many times over cached results from the first.

**Stage 1: scoring.** For each model and probe, take one input gradient of the
conditioning objective, convert both the gradient and the probe image into binary
structural maps, and measure their similarity. The structural post-processing is governed
by a configuration

```
c = (rho, psi, e, mu)
```

where `rho` is the gradient retention rule, `psi` the smoothing transformation, `e` the
edge percentile, and `mu` the similarity measure. All 4,608 candidate configurations reuse
the same cached gradients, so stage 1 computes every one of them in a single pass and
writes the resulting alignment scores to disk.

**Stage 2: configuration selection and ranking.** A protocol picks one configuration out
of the candidate space and turns the per-probe alignment scores into model-level SGA
scores, then reports pairwise agreement (PA) with the reference order. Protocols differ
only in what reference information the evaluator is allowed to use.

| Protocol | Evaluator knowledge | Selection criterion | Script |
|---|---|---|---|
| Transfer | Development ranking | Development PA | `sga_rank_fixed.py` |
| Uncalibrated | None | Score dispersion | `sga_rank_adaptive.py --protocol uncalibrated` |
| Calibrated | Ordered anchors | Anchor PA, then dispersion | `sga_rank_adaptive.py --protocol calibrated` |
| Oracle | Target ranking | Target PA | `sga_rank_adaptive.py --protocol oracle` |

All four use the same hierarchical selection cascade

```
PA -> Std -> Gap -> seeded random
```

where `Std` is the standard deviation of the model scores, `Gap` is the minimum adjacent
score gap, and each protocol uses the order-preserving subsequence its knowledge permits.
Uncalibrated therefore runs `(Std, Gap)`; Transfer, Calibrated, and Oracle run the
complete cascade on their respective model sets. Oracle observes the evaluation target
directly, so it is an analysis protocol and an empirical upper bound, not a deployable
setting.

## Repository layout

```
sga_export_raw.py            stage 1: gradients -> structural maps -> alignment scores
sga_rank_fixed.py            stage 2: fixed configurations (Transfer preset and sweeps)
sga_rank_adaptive.py         stage 2: per-batch selection (uncalibrated/calibrated/oracle)
build_reference_pgd.py       reference pool XML from clean and PGD robust accuracy
build_reference_pgd_vqa.py   the same for the LVLM pool on VQAv2

sga/                         the method
  structural.py              ternary representation, smoothing, structural maps, mu
  gpu.py                     the same pipeline on GPU, no per-probe CPU round-trip
  rank_io.py                 score-matrix I/O, candidate sweeps, PA evaluation
  vqa_grad.py                teacher-forced NLL gradients for the LVLM pool
common/                      shared infrastructure
  utils.py                   model and probe loading
  ranking_task.py            probe sources, conditioning outputs, batching
  reference.py               reference pool XML and pairwise agreement
  pgd.py                     PGD used to build reference orders
  vqa_data.py                VQAv2 loading
  madry_imagenet_l2.py       the l2 ImageNet checkpoints
vlm/                         CLIP-family encoders and the LLaVA-1.5 interface
reference/                   reference pool XMLs
scripts/                     dataset and checkpoint download helpers
```

## Setup

Tested with **Python 3.10**.

Install in this order. **RobustBench must be installed last** (after the LLaVA package) to
avoid dependency conflicts with `timm` (LLaVA's install can otherwise pin/overwrite packages
that RobustBench needs).

**1. Base dependencies**

```bash
pip install -r requirements.txt
```

**2. LLaVA Python package** (only required for the LVLM / VQA pool; weights are separate)

```bash
git clone https://github.com/haotian-liu/LLaVA.git
pip install -e LLaVA
# or: export LLAVA_ROOT=/path/to/LLaVA
python -c "from llava.model.builder import load_pretrained_model"
```

**3. RobustBench** (DNN robust models; install after LLaVA)

```bash
pip install "robustbench @ git+https://github.com/RobustBench/robustbench.git"
python -c "from robustbench import load_model"
```

**4.** RobustBench and `timm` download their checkpoints on first use. **Fetch the rest as needed:**

```bash
bash scripts/download_madry_l2_weights.sh         # DNN l2 (Madry ImageNet)
bash scripts/download_vlm_weights.sh              # VLM / LVLM robust CLIP encoders
bash scripts/vlm/download_llava_v1_5_weights.sh   # LVLM LLaVA-1.5-7B
```

`download_vlm_weights.sh` also runs `scripts/convert_to_openclip.py` to produce the
OpenCLIP LEAF-CLIP weights; you do not need to run that script yourself.

## Probes

A probe set is a directory of images named `00000_<label>.png`, `00001_<label>.png`, and so
on. The label suffix is only read when `--conditioning ground_truth`; public probes can
carry any placeholder.

| Probe source | Role | Prepare with |
|---|---|---|
| NIPS 2017 development set (ImageNet-1K) | classification test-set probes | `data/imagenet.tar.gz` → `data/imagenet/` |
| VQAv2 validation + COCO | VQA test-set probes | `bash scripts/vlm/download_and_prepare_vqav2.sh` |
| CelebA-HQ | task-independent public probes | `data/celeba.tar.gz` → `data/celeba/` |
| Places365 | task-independent public probes | `python scripts/download_and_prepare_places365.py` |

The conditioning output follows the probe type. Test-set probes have annotated targets, so
they use `--conditioning ground_truth`. Public probes are unlabeled and out of
distribution, so each model conditions on its own prediction, `--conditioning pseudo`
(argmax for classification, greedy-decoded answer for VQA).

For VQA, `--conditioning ground_truth` reads the image path from the jsonl and conditions
on the annotated answer, while `--conditioning pseudo` pairs the *n*-th image in `--input`
with the *n*-th jsonl question and ignores the jsonl image path.

`--resize` applies CLIP-style preprocessing: bilinear resize of the shorter side, then a
square center crop to the given side length. Classification must use `--resize 224`. For
VQA the side length is free; the values used in the paper are in the implementation
details.

## Reference pools

`reference/` holds one XML per homogeneous model pool. Each entry is a model identifier
with its clean and robust accuracy under a fixed reference evaluation; the robust accuracy
induces the reference order that PA is measured against. These are reference evaluations,
not a universal ground truth for robustness.

| File | Pool | Reference |
|---|---|---|
| `dnn_l2.xml` | DNN `l2`, 13 entries | PGD-50, eps = 2, step 0.1 |
| `dnn_linf.xml` | DNN `linf`, 14 entries | PGD-50, eps = 8/255, step 1/255 |
| `dnn_linf_autoattack.xml` | DNN `linf`, 14 entries | standard AutoAttack from RobustBench, eps = 4/255 |
| `vlm_linf.xml` | VLM `linf`, 7 entries | PGD-50, eps = 8/255, step 1/255 |
| `lvlm_linf.xml` | LVLM `linf`, 5 entries | PGD-50 on the teacher-forced answer NLL, eps = 8/255, step 1/255 |
| `dnn_l2_dev.xml`, `dnn_linf_dev.xml` | development sets for Transfer | the pool minus `Standard_R50` |

All PGD references use 50 steps, one restart, and `--seed 0`, and are computed only on the
examples every model in the pool classifies (or answers) correctly, so clean accuracy is
1.0 on the retained subset. To rebuild one:

```bash
python build_reference_pgd.py \
  --input data/imagenet \
  --start_idx 0 --end_idx 1000 \
  --reference_xml reference/dnn_linf.xml \
  --mode intersection \
  --threat_model Linf --eps 8 --pgd_step_size 1 --pgd_steps 50 \
  --seed 0 \
  --out_xml reference/dnn_linf.xml
```

`--mode intersection` computes the all-models-correct subset internally. The LVLM
equivalent is `build_reference_pgd_vqa.py`, which maximizes the teacher-forced NLL of the
annotated answer and scores answers by normalized exact match:

```bash
python build_reference_pgd_vqa.py \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --input data/coco/val2014 \
  --start-idx 0 --end-idx 1000 \
  --reference_xml reference/lvlm_linf.xml \
  --mode intersection \
  --threat_model Linf --eps 8 --pgd_step_size 1 --pgd_steps 50 \
  --seed 0 \
  --out_xml reference/lvlm_linf.xml
```

## Running SGA

### Stage 1: cache the alignment scores

This is the only stage that touches the models. It computes one gradient per model-probe
pair and evaluates every candidate configuration on it. The flags below enumerate the full
4,608-configuration candidate space.

```bash
python sga_export_raw.py \
  --task imagenet \
  --input data/celeba \
  --conditioning pseudo \
  --start_idx 0 --end_idx 1000 \
  --resize 224 \
  --reference_xml reference/dnn_linf.xml \
  --out_dir raw/dnn_linf_celeba \
  --seed 0 \
  --retain_abs 0 1e-4 5e-5 1e-5 5e-6 1e-6 \
  --retain_top_pct 50 55 60 65 70 75 80 85 90 95 \
  --smooth_sigma 0.3 0.4 0.5 0.6 0.7 \
  --smooth_scale 1 2 3 4 5 6 7 \
  --edge_percentile 60 65 70 75 80 85 90 95 \
  --similarity ssim dice dice_ssim
```

The result is one CSV per (configuration, similarity) under `--out_dir`, plus an
`export_summary.csv` listing the models covered. Stage 2 reads these and never reloads a
model, which is why switching protocols or probe budgets costs seconds.

For the LVLM pool, add `--data-jsonl` and use the VQA task:

```bash
python sga_export_raw.py \
  --task vqa \
  --data-jsonl data/vqa/vqav2_val.jsonl \
  --input data/coco/val2014 \
  --conditioning ground_truth \
  --start_idx 0 --end_idx 1000 \
  --resize 336 \
  --grad_batch_size 10 \
  --reference_xml reference/lvlm_linf.xml \
  --out_dir raw/lvlm_linf_coco \
  --seed 0 \
  ...
```

`--grad_batch_size` is the number of probes per backward pass. It matters for VQA: the
teacher-forced NLL is averaged over the answer tokens of the whole batch, so the gradient
magnitudes depend on it. Keep it fixed across every run you intend to compare.

### Stage 2a: Transfer

Transfer freezes one preset chosen on a development model set that does not overlap the
targets, so it runs `sga_rank_fixed.py` twice. First sweep the candidate space on the
development pool over all `K = 10` test-set batches:

```bash
python sga_rank_fixed.py \
  --raw_dir raw/dnn_l2_dev_testset \
  --reference_xml reference/dnn_l2_dev.xml \
  --start_idx 0 --end_idx 1000 \
  --probe_budgets 10 \
  --prefix dev_sweep \
  --out_dir results/transfer_dev \
  --seed 0 \
  --retain_abs 0 1e-4 5e-5 1e-5 5e-6 1e-6 \
  --retain_top_pct 50 55 60 65 70 75 80 85 90 95 \
  --smooth_sigma 0.3 0.4 0.5 0.6 0.7 \
  --smooth_scale 1 2 3 4 5 6 7 \
  --edge_percentile 60 65 70 75 80 85 90 95 \
  --similarity ssim dice dice_ssim
```

Then apply the winning configuration, unchanged, to the target pool:

```bash
python sga_rank_fixed.py \
  --raw_dir raw/dnn_linf_celeba \
  --reference_xml reference/dnn_linf.xml \
  --start_idx 0 --end_idx 1000 \
  --probe_budgets 10 \
  --prefix transfer \
  --out_dir results/dnn_linf_celeba_transfer \
  --seed 0 \
  --retain_top_pct 90 --smooth_scale 2 --edge_percentile 95 --similarity ssim
```

The two presets reported in the paper, both selected on classification test-set probes:

| Target pool | Development models | Preset |
|---|---|---|
| DNN `linf`, VLM `linf`, LVLM `linf` | `reference/dnn_l2_dev.xml` | `--retain_top_pct 90 --smooth_scale 2 --edge_percentile 95 --similarity ssim` |
| DNN `l2` | `reference/dnn_linf_dev.xml` | `--retain_abs 1e-5 --smooth_sigma 0.6 --edge_percentile 85 --similarity dice_ssim` |

`Standard_R50` is excluded from both development sets because it belongs to the
corresponding target pools.

### Stage 2b: Uncalibrated, Calibrated, Oracle

These select one configuration per probe batch, so they need the candidate space rather
than a preset:

```bash
python sga_rank_adaptive.py \
  --raw_dir raw/dnn_linf_celeba \
  --reference_xml reference/dnn_linf.xml \
  --start_idx 0 --end_idx 1000 \
  --probe_budgets 10 \
  --protocol all \
  --prefix adaptive \
  --out_dir results/dnn_linf_celeba_adaptive \
  --seed 0 \
  --anchor_seed 0 \
  --num_anchors 3 \
  --retain_abs 0 1e-4 5e-5 1e-5 5e-6 1e-6 \
  --retain_top_pct 50 55 60 65 70 75 80 85 90 95 \
  --smooth_sigma 0.3 0.4 0.5 0.6 0.7 \
  --smooth_scale 1 2 3 4 5 6 7 \
  --edge_percentile 60 65 70 75 80 85 90 95 \
  --similarity ssim dice dice_ssim
```

`--protocol all` is the default and runs all three. Calibrated draws `H` anchors per batch
without replacement, seeded by `--anchor_seed 0` as `anchor_seed + batch_index`. Omitting
`--num_anchors` uses `H = M - 1` (one random target per batch).

## Candidate space

| Component | Flag | Values | Count |
|---|---|---|---|
| Gradient retention `rho` | `--retain_abs`, `--retain_top_pct` | `--retain_abs 0` = all (retain every coordinate); `\|g\| >= {1e-4, 5e-5, 1e-5, 5e-6, 1e-6}`; top `{50, 55, ..., 95}`% | 16 |
| Smoothing `psi` | `--smooth_sigma`, `--smooth_scale` | `--smooth_scale 1` = no smoothing; `2..7` = downsample/upsample; or Gaussian `sigma` in `{0.3, 0.4, ..., 0.7}` | 12 |
| Edge percentile `e` | `--edge_percentile` | `{60, 65, ..., 95}` | 8 |
| Similarity `mu` | `--similarity` | `ssim`, `dice`, `dice_ssim` (the arithmetic mean, Dice+SSIM in the paper) | 3 |

The Cartesian product is 4,608 configurations. `--retain_abs 0` and `--smooth_scale 1` are
the identity settings for `rho` and `psi`. The two smoothing operators are mutually
exclusive: `--smooth_scale` ≥ 2 is bilinear downsampling followed by upsampling back to the
original resolution, and a Gaussian sigma replaces the scale operator rather than composing
with it. `rho` applies only to the gradient branch, while `psi` and `e` are applied
identically to the gradient and the probe image.

## Common parameters

| Flag | Scripts | Meaning |
|---|---|---|
| `--task` | stage 1 | `imagenet` for classification cross-entropy; `vqa` for the LVLM teacher-forced NLL |
| `--input` | stage 1, reference build | probe directory, or the COCO root for VQA test-set probes |
| `--data-jsonl` | VQA only | VQAv2 jsonl produced by `scripts/vlm/` |
| `--conditioning` | stage 1 | `ground_truth` for test-set probes, `pseudo` for public probes |
| `--start_idx`, `--end_idx` | all | half-open probe range; for VQA these index jsonl question rows |
| `--resize` | stage 1 | square side after shortest-edge resize and center crop, or `none` |
| `--grad_batch_size` | stage 1, reference build | probes per backward pass; fix it across comparable runs |
| `--reference_xml` | all | the reference pool XML defining the models and the reference order |
| `--raw_dir` | stage 2 | one or more stage-1 output directories; a model is taken from the first that lists it |
| `--probe_budgets` | stage 2 | comma-separated probe budgets `K`; the probe order is cut into consecutive non-overlapping batches |
| `--protocol` | `sga_rank_adaptive.py` | `uncalibrated`, `calibrated`, `oracle`, or `all` |
| `--num_anchors` | calibrated only | anchor count `H`, from 2 to `M - 1`; omit for `M - 1` (paper default) |
| `--anchor_seed` | calibrated only | default `0`; batch `b` samples anchors with seed `anchor_seed + b` |
| `--seed` | all five entry scripts | default `0`; reproducibility for gradients / PGD init, and the cascade tie-break in `sga_rank_adaptive.py` |
| `--no_random_start` | both reference builders | disable PGD random initialization inside the perturbation ball |
| `--cpu` | all | force the CPU path; the GPU path is used whenever CUDA is available |
| `--save_maps` | stage 1 | also write the intermediate grayscale, smoothed, and binary structural maps |

Use `--help` on any script for the full list.

## Metric

Pairwise agreement counts, over all pairs in a pool, how often the SGA score order matches
the reference order:

```
a_ij = 1 if R_i == R_j
a_ij = 1 if (S_i - S_j) * (R_i - R_j) > 0
a_ij = 0 otherwise

PA = mean over the M*(M-1)/2 pairs of a_ij
```

Reference ties count as agreements, because the reference evaluation does not distinguish
the tied models; ties in the SGA scores count as disagreements. PA lies in `[0, 1]` and is
computed separately within each homogeneous pool, never across pools. Results are reported
as the mean PA over probe batches plus its standard deviation, which measures sensitivity
to probe-batch composition rather than training randomness.
