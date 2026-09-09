# SimMeR Prototype — User Guide

**SimMeR — Simulating DNA Methylation with Demographic Realism (prototype)**

This guide walks through installing, running, and extending the SimMeR prototype:
a small, CPU-friendly implementation of the phenotype-aware DNA methylation
(DNAm) cohort simulator from the SimMeR NIH R21 proposal. It generates
individual DNAm profiles with a **conditional variational autoencoder (cVAE)**
conditioned on demographics and cell-type composition, assembles them into
synthetic cohorts, and validates them by biological plausibility and inferential
behaviour.

> **Prototype scope.** This is a demonstration of the algorithm and evaluation
> logic on a **synthetic, structured stand-in** for real data. It is *not* the
> production SimMeR package and does not ingest real GEO/EPIC arrays, harmonize
> platforms, train at genome scale on GPUs, or provide the R interface. See
> [Limitations](#9-limitations-and-relationship-to-the-full-grant).

---

## Table of contents

1. [Concepts and terminology](#1-concepts-and-terminology)
2. [Installation](#2-installation)
3. [Quickstart: the end-to-end pipeline](#3-quickstart-the-end-to-end-pipeline)
4. [Command-line reference](#4-command-line-reference)
5. [Understanding the outputs](#5-understanding-the-outputs)
6. [Python API tutorial](#6-python-api-tutorial)
7. [Designing your own cohort](#7-designing-your-own-cohort)
8. [How the validation gates work](#8-how-the-validation-gates-work)
9. [Limitations and relationship to the full grant](#9-limitations-and-relationship-to-the-full-grant)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. Concepts and terminology

| Term | Meaning in this prototype |
| --- | --- |
| **DNAm** | DNA methylation. Represented internally as **M-values** (`log2(beta/(1-beta))`), the modelling scale used in the proposal. Convert with `beta_to_m` / `m_to_beta`. |
| **CpG** | A methylation site ("probe"). The prototype uses a few hundred synthetic CpGs; real arrays have ~450k–900k. |
| **Phenotype / conditioning vector** | Age, sex, smoking, diagnosis, and 6 cell-type proportions (CD4T, CD8T, NK, Bcell, Mono, Neu) fed to the cVAE. |
| **cVAE** | Conditional variational autoencoder. The encoder maps `[DNAm ; phenotype]` to a latent code; the decoder reconstructs DNAm from `[latent ; phenotype]`. |
| **Reference dataset** | A synthetic, structured stand-in for real training data, with *known* injected effects so recovery can be measured. |
| **Cohort** | A set of generated individuals assembled under a user-specified demographic structure (`PhenotypeSpec`). |
| **Hard gate** | A pass/fail validation check (anti-memorization, effect recovery, structure, null calibration). |

The pipeline mirrors the grant's two aims:

- **Aim 1 — individual generation:** train the cVAE, generate plausible individual profiles.
- **Aim 2 — cohort simulation:** assemble validated profiles into cohorts with controllable composition.

---

## 2. Installation

### Requirements

- Linux or macOS, **Python 3.10+**
- ~2 GB free disk (mostly the CPU PyTorch wheel)
- No GPU required

### Option A — one-shot install script (recommended)

From the repository root:

```bash
bash scripts/cloud-install.sh
source .venv/bin/activate
```

This creates a virtual environment at `.venv`, installs CPU-only PyTorch plus the
scientific stack, and installs the `simmer` package in editable mode. It is
**idempotent** — safe to re-run; it only installs what is missing.

### Option B — manual install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU-only PyTorch from the dedicated index:
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Verify the install

```bash
python -c "import simmer, torch; print('simmer', simmer.__version__, '| torch', torch.__version__)"
python -m pytest -q          # runs the fast test suite (expect: 3 passed)
```

---

## 3. Quickstart: the end-to-end pipeline

Run the full Aim 1 → Aim 2 pipeline (build reference data → train cVAE → assemble
a synthetic cohort → evaluate):

```bash
python -m simmer.cli --outdir artifacts
```

Equivalently, using the installed console script:

```bash
simmer --outdir artifacts
```

This takes a few seconds on CPU and prints a summary like:

```
=== SimMeR prototype evaluation summary ===
Hard gates passed: 5/5
  - anti_memorization       : PASS
  - age_recovery            : PASS
  - smoking_recovery        : PASS
  - structure_preservation  : PASS
  - null_calibration        : PASS
Figure : artifacts/simmer_evaluation.png
Metrics: artifacts/metrics.json
```

Open `artifacts/simmer_evaluation.png` to see the diagnostic figure.

---

## 4. Command-line reference

```
python -m simmer.cli [options]
```

| Flag | Default | Description |
| --- | --- | --- |
| `--outdir` | `artifacts` | Output directory for figures, metrics, and the generated cohort. |
| `--n-samples` | `3000` | Number of individuals in the synthetic reference dataset. |
| `--n-cpgs` | `400` | Number of CpGs (probes). Effect/null CpGs scale with this. |
| `--cohort-n` | `1500` | Number of individuals in the generated synthetic cohort. |
| `--epochs` | `60` | cVAE training epochs. |
| `--latent-dim` | `32` | Dimensionality of the cVAE latent space. |
| `--seed` | `0` | Random seed for full reproducibility. |

**Examples**

```bash
# Fast smoke run
python -m simmer.cli --outdir /tmp/quick --n-samples 800 --n-cpgs 150 --epochs 20

# Larger, longer-trained run
python -m simmer.cli --outdir artifacts_big --n-samples 6000 --n-cpgs 800 --epochs 120

# Reproduce an exact run
python -m simmer.cli --outdir run_seed7 --seed 7
```

---

## 5. Understanding the outputs

After a run, `--outdir` contains:

| File | Contents |
| --- | --- |
| `simmer_evaluation.png` | 6-panel diagnostic figure (see below). |
| `metrics.json` | All gate results as structured JSON. |
| `cohort_manifest.json` | Reproducibility manifest: the `PhenotypeSpec`, seeds, and model config used to generate the cohort. |
| `synthetic_cohort_mvalues.npy` | Generated cohort DNAm, shape `(cohort_n, n_cpgs)`, M-value scale. |
| `synthetic_cohort_phenotypes.csv` | Phenotype table for the generated cohort (age, sex, smoker, diagnosis, cell proportions). |

### The diagnostic figure (panels)

1. **cVAE training** — train/validation reconstruction loss vs. epoch.
2. **Age EWAS effect recovery** — per-CpG age effect in synthetic vs. real data; age CpGs highlighted; points should track the diagonal.
3. **Smoking EWAS effect recovery** — same, for smoking CpGs.
4. **Per-CpG mean** — marginal mean methylation, synthetic vs. real.
5. **Null p-values** — histogram of EWAS p-values under random null predictors; should be ~flat.
6. **Anti-memorization** — nearest-neighbour distances (synthetic→train vs. real held-out→train).

### Reading `metrics.json`

```json
{
  "age_recovery":  { "effect_corr_all_cpgs": 0.99, "sign_agreement_at_age_cpgs": 1.0, "passed": true },
  "smoking_recovery": { "effect_corr_all_cpgs": 0.94, "passed": true },
  "structure_preservation": { "covariance_corr": 0.95, "marginal_mean_corr": 0.99, "passed": true },
  "anti_memorization": { "ratio_synth_over_real": 0.87, "passed": true },
  "null_calibration": { "type1_error_at_alpha_0.05": 0.04, "passed": true },
  "summary": { "gates_passed": 5, "gates_total": 5, "all_passed": true }
}
```

---

## 6. Python API tutorial

The whole pipeline is available programmatically. All the main entry points are
exported from the top-level `simmer` package.

```python
from simmer import (
    make_reference_dataset, train_cvae, assemble_cohort,
    evaluate_prototype, PhenotypeSpec,
)
from simmer.data import train_val_split
from simmer.train import TrainConfig

# 1. Reference data with known injected effects (stand-in for real GEO data)
ref = make_reference_dataset(n_samples=3000, n_cpgs=400, seed=0)
train, val = train_val_split(ref, val_frac=0.2, seed=0)

# 2. Train the conditional VAE (Aim 1)
cfg = TrainConfig(epochs=60, latent_dim=32, seed=0)
trained = train_cvae(train, val, cfg)

# 3. Assemble a synthetic cohort under a chosen demographic structure (Aim 2)
spec = PhenotypeSpec(n=1500, age_mean=55, age_sd=15, female_frac=0.5, smoker_frac=0.35)
synth, manifest = assemble_cohort(trained, spec, ref.ground_truth, seed=7)

# 4. Validate biological plausibility and inferential behaviour
metrics = evaluate_prototype(train, val, synth)
print(metrics["summary"])           # {'gates_passed': 5, 'gates_total': 5, 'all_passed': True}
```

### Generating profiles for a specific phenotype table

If you already have a phenotype table (a `pandas.DataFrame` with columns
`age, female, smoker, diagnosis, CD4T, CD8T, NK, Bcell, Mono, Neu`), you can
generate profiles directly:

```python
from simmer.generate import generate_profiles
m_values = generate_profiles(trained, my_phenotype_df, seed=0)  # (n_rows, n_cpgs)
```

### Running an EWAS yourself

```python
import numpy as np
from simmer.evaluate import ewas

res = ewas(synth.m_values, synth.phenotypes["age"].to_numpy())
top = np.argsort(res["p"])[:10]      # 10 most age-associated CpGs
print(res["beta"][top], res["p"][top])
```

### Key objects

- `DNAmDataset` — holds `.phenotypes` (DataFrame), `.m_values` (ndarray), `.ground_truth`; helpers `.n_samples`, `.n_cpgs`, `.beta()`.
- `TrainedModel` — holds the fitted `.model` (a `ConditionalVAE`), the `.scaler`, and `.history`.
- `ConditionalVAE` — the network; use `.generate(cond_tensor)` for raw sampling.
- `TrainConfig` — training hyperparameters (KL annealing, early stopping, etc.).

---

## 7. Designing your own cohort

`PhenotypeSpec` controls the demographic structure of a cohort. Adjust the
marginals to model different populations; cell-type proportions are drawn
automatically and covary with age/sex/smoking.

```python
from simmer import PhenotypeSpec

# An older, higher-smoking cohort
older = PhenotypeSpec(n=2000, age_mean=68, age_sd=8, smoker_frac=0.45)

# A young, balanced, low-smoking cohort
young = PhenotypeSpec(n=2000, age_mean=28, age_sd=5, smoker_frac=0.10)
```

| Field | Default | Meaning |
| --- | --- | --- |
| `n` | `2000` | Number of individuals. |
| `age_mean`, `age_sd` | `44.6`, `18.8` | Age distribution (Normal, clipped). |
| `age_min`, `age_max` | `18`, `90` | Age clipping bounds. |
| `female_frac` | `0.5` | Fraction coded `female=1`. |
| `smoker_frac` | `0.20` | Fraction coded `smoker=1`. |
| `diagnosis_probs` | `[0.80, 0.07, 0.07, 0.06]` | Weights over `["control", "SLE", "RA", "T2D"]`. |
| `seed` | `None` | Reproducibility seed. |

Generate two cohorts and compare an EWAS across them to study how composition
shifts inference — the core use case SimMeR is designed for.

---

## 8. How the validation gates work

The prototype defines success by **biological plausibility and inferential
behaviour**, not visual realism. Each gate targets a distinct failure mode.

| Gate | Question | Pass criterion (prototype) |
| --- | --- | --- |
| **Anti-memorization** | Are synthetic profiles just copies of training individuals? | Median synth→train NN distance ≥ 0.7 × real-held-out→train distance. |
| **Age recovery** | Does the synthetic age signal match real biology (ELOVL2-like)? | Effect correlation > 0.5 and sign agreement > 0.8 at age CpGs. |
| **Smoking recovery** | Same for smoking (AHRR-like). | Effect correlation > 0.4 and sign agreement > 0.7. |
| **Structure preservation** | Is genome-wide covariance / marginal structure preserved? | Covariance correlation > 0.5 and marginal-mean correlation > 0.9. |
| **Null calibration** | Do true-null CpGs yield calibrated (non-inflated) p-values? | Type-I error at α=0.05 ≤ 0.075, pooled over many random null predictors. |

Because the reference data has **known** ground truth (which CpGs carry age,
smoking, sex, cell, or null effects), these are measured directly rather than
approximated. See `simmer/evaluate.py`.

---

## 9. Limitations and relationship to the full grant

This prototype demonstrates the *method*, not the finished product. It does **not** include:

- Real DNAm data ingestion (GEO / access-controlled cohorts).
- Array harmonization across Illumina 450K / EPIC / EPIC v2.
- Genome-scale (>300k CpG) modelling with block-structured or attention encoders.
- GPU training and the architectural ablations described in the grant (Aim 1 C.4.b).
- The R package interface.

What it *does* faithfully implement: the conditional-VAE data-generating idea,
phenotype + cell-type conditioning, KL-annealed training, cohort assembly under
specified structure, and the plausibility/inferential evaluation philosophy.
Every module keeps an interface that generalizes to real arrays, so swapping in
real data is largely a data-loading change rather than a rewrite.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ensurepip is not available` when creating a venv | Install the venv package: `sudo apt-get install -y python3-venv` (the install script does this automatically). |
| PyTorch install is slow or pulls CUDA | Always use the CPU index: `--index-url https://download.pytorch.org/whl/cpu`. The install script already does. |
| `ModuleNotFoundError: simmer` | Activate the venv (`source .venv/bin/activate`) and `pip install -e .`. |
| A gate FAILs on a very small `--n-cpgs`/`--epochs` | Small/under-trained configs are noisy. Use defaults or increase `--epochs`. |
| Figure not written / `matplotlib` backend error | The CLI forces the non-interactive `Agg` backend; ensure matplotlib is installed. |
| Non-reproducible results | Pass `--seed` (CLI) or set `seed=` on `PhenotypeSpec` / `TrainConfig`. |

---

## 11. FAQ

**Q. Why synthetic data instead of real DNAm?**
Real training data needs data-use agreements and GPU-scale compute. A structured
synthetic reference with known ground truth lets the prototype *prove it recovers
biology*, which real data (with unknown truth) cannot do directly.

**Q. Can I plug in my own data?**
Yes — construct a `DNAmDataset` with your `phenotypes` DataFrame and `m_values`
array (and a `GroundTruth` if you want the effect-recovery gates to apply), then
call `train_cvae`. Array harmonization and QC are out of scope for the prototype.

**Q. Is the output on the beta or M-value scale?**
M-values. Use `simmer.data.m_to_beta(...)` to convert to beta in `[0, 1]`.

**Q. How long does a run take?**
The default config trains in a few seconds on 4 CPU cores.

**Q. How do I cite the method?**
This prototype implements ideas from the SimMeR NIH R21 proposal (Merrill et al.)
and builds on the methCancer-gen conditional-VAE framework (Choi & Chae, 2020).
