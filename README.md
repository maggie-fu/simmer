# simmer

**SimMeR — Simulating DNA Methylation with Demographic Realism** (prototype).

A minimal, CPU-friendly prototype of the phenotype-aware DNA methylation (DNAm)
cohort simulator described in the SimMeR NIH R21 proposal. It implements the core
data-generating idea of the grant — a **conditional variational autoencoder
(cVAE)** that generates individual genome-wide DNAm profiles conditioned on
demographics and cell-type composition — together with cohort assembly and a
validation panel defined by *biological plausibility and inferential behaviour*
rather than surface realism.

The whole Aim 1 → Aim 2 pipeline runs end-to-end in seconds on a laptop/CPU.

## Why a synthetic reference?

The real project trains on ~30,000 blood DNAm profiles from GEO plus
access-controlled cohorts, on A100 GPUs. Those data need data-use agreements and
the training is GPU-scale, so this prototype substitutes a small **structured
synthetic reference dataset** whose ground truth we control (see
`simmer/data.py`). Because we know exactly where signal was injected — age
(ELOVL2-like) CpGs, smoking (AHRR-like) CpGs, sex CpGs, cell-composition CpGs,
and true-null CpGs — the evaluation can measure whether the cVAE *recovers real
biology* instead of just producing plausible-looking noise. Every module has the
same interface it would at genome scale, so swapping in real arrays is a data
change, not a rewrite.

## Prototype framework

The prototype mirrors the proposal's two aims as a five-stage pipeline:

| Stage | Module | Maps to proposal |
| --- | --- | --- |
| 1. Reference data with known effects | `simmer/data.py`, `simmer/phenotypes.py` | C.2 data + C.4.e ground truth |
| 2. Conditional VAE (encoder/decoder both see phenotype) | `simmer/model.py` | C.4.a methCancer-gen-style cVAE |
| 3. Training: recon + KL, KL annealing, early stop | `simmer/train.py` | C.4.c stabilization |
| 4. Cohort assembly under specified structure | `simmer/generate.py` | C.5.a Aim 2 |
| 5. Plausibility + inferential validation gates | `simmer/evaluate.py` | C.4.e / C.5.b hard gates |

### Conditioning variables

Age, sex, smoking, health diagnosis, and a 6-part IDOL-style immune cell-type
composition (CD4T, CD8T, NK, Bcell, Mono, Neu). Cell-type proportions covary with
age/sex/smoking in the reference generator, reproducing the demographic–cellular
confounding SimMeR is meant to let users control.

### Validation gates (`evaluate.py`)

1. **Anti-memorization** — synthetic profiles are no closer to training
   individuals than real held-out individuals are (nearest-neighbour distance).
2. **Canonical-loci recovery** — an EWAS on synthetic data recovers the age and
   smoking effect sizes/direction seen in real data.
3. **Structure preservation** — genome-wide CpG–CpG covariance and per-CpG
   marginals match real data.
4. **Null calibration** — EWAS on random null predictors yields ~uniform
   p-values (Type-I error near nominal, not inflated).

## Quickstart

```bash
# 1. install (idempotent; installs CPU PyTorch + scientific stack + package)
bash scripts/cloud-install.sh
. .venv/bin/activate

# 2. run the end-to-end prototype
python -m simmer.cli --outdir artifacts
```

Outputs written to `artifacts/`:

- `simmer_evaluation.png` — training curve, effect-recovery scatterplots, null
  calibration, anti-memorization.
- `metrics.json` — all gate results.
- `synthetic_cohort_mvalues.npy` + `synthetic_cohort_phenotypes.csv` — a
  generated synthetic cohort.
- `cohort_manifest.json` — spec + seeds for reproducibility.

### Useful flags

```
--n-samples   reference dataset size (default 3000)
--n-cpgs      number of CpGs/probes (default 400)
--cohort-n    synthetic cohort size (default 1500)
--epochs      training epochs (default 60)
--latent-dim  cVAE latent dimension (default 32)
--seed        random seed (default 0)
```

## Programmatic use

```python
from simmer import (make_reference_dataset, train_cvae, assemble_cohort,
                    evaluate_prototype, PhenotypeSpec)
from simmer.data import train_val_split

ref = make_reference_dataset(n_samples=3000, n_cpgs=400, seed=0)
train, val = train_val_split(ref, val_frac=0.2, seed=0)
trained = train_cvae(train, val)

# Assemble a synthetic cohort: older, higher smoking prevalence than training.
spec = PhenotypeSpec(n=1500, age_mean=55, smoker_frac=0.35)
synth, manifest = assemble_cohort(trained, spec, ref.ground_truth)

metrics = evaluate_prototype(train, val, synth)
```

## Scope and limitations

This is a **prototype**, not the SimMeR package. It demonstrates the algorithm
and evaluation logic at small scale on a synthetic stand-in. It does **not**
include real GEO/EPIC data ingestion, array harmonization (450K/EPIC/EPICv2),
block-structured or attention encoders for genome-wide scale, GPU training, or
the R interface — all of which are explicit deliverables in the full grant.

## Testing

```bash
. .venv/bin/activate
python -m pytest -q
```
