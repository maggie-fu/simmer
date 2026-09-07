"""Synthetic reference DNAm data with known, recoverable structure.

The real SimMeR project trains on ~30,000 blood DNAm profiles from GEO plus
access-controlled cohorts. Those require data-use agreements and GPU-scale
compute, so for a runnable prototype we generate a small, *structured* stand-in
whose ground truth we control. This lets the evaluation step check whether the
cVAE recovers real biology (e.g. age hypermethylation at ELOVL2-like CpGs,
smoking hypomethylation at AHRR-like CpGs) rather than just producing
plausible-looking noise.

DNAm is represented as M-values (logit of beta), the scale the proposal uses as
model input. Helpers convert to/from beta in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .phenotypes import CELL_TYPES, PhenotypeSpec, sample_phenotypes


def beta_to_m(beta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    beta = np.clip(beta, eps, 1 - eps)
    return np.log2(beta / (1 - beta))


def m_to_beta(m: np.ndarray) -> np.ndarray:
    return 2 ** m / (1 + 2 ** m)


@dataclass
class GroundTruth:
    """Records where signal was injected so evaluation can measure recovery."""

    age_cpgs: np.ndarray
    smoking_cpgs: np.ndarray
    sex_cpgs: np.ndarray
    cell_cpgs: np.ndarray
    null_cpgs: np.ndarray
    n_cpgs: int
    cpg_names: List[str]


@dataclass
class DNAmDataset:
    """A reference dataset: phenotypes, DNAm M-values, and the ground truth."""

    phenotypes: pd.DataFrame
    m_values: np.ndarray  # shape (n_samples, n_cpgs)
    ground_truth: GroundTruth

    @property
    def n_samples(self) -> int:
        return self.m_values.shape[0]

    @property
    def n_cpgs(self) -> int:
        return self.m_values.shape[1]

    def beta(self) -> np.ndarray:
        return m_to_beta(self.m_values)


def make_reference_dataset(
    n_samples: int = 2000,
    n_cpgs: int = 400,
    n_blocks: int = 20,
    seed: int = 0,
    spec: Optional[PhenotypeSpec] = None,
) -> DNAmDataset:
    """Generate a structured synthetic DNAm reference dataset.

    Structure injected (all with known locations, stored in GroundTruth):
      * Co-methylation blocks: CpGs are grouped into blocks sharing a latent
        factor, reproducing the regional correlation of real DNAm.
      * Age CpGs (ELOVL2-like): monotonic hypermethylation with age.
      * Smoking CpGs (AHRR-like): hypomethylation in smokers.
      * Sex CpGs: shift by sex.
      * Cell-composition CpGs: linear in cell-type proportions.
      * Null CpGs: no phenotype association (for Type-I-error calibration).
    """
    rng = np.random.default_rng(seed)
    if spec is None:
        spec = PhenotypeSpec(n=n_samples, seed=seed)
    else:
        spec.n = n_samples

    pheno = sample_phenotypes(spec, rng=rng)

    # --- baseline methylation on the M-value scale ---
    cpg_baseline = rng.normal(0.0, 1.5, n_cpgs)

    # --- co-methylation block structure ---
    block_id = rng.integers(0, n_blocks, n_cpgs)
    block_factor = rng.normal(0.0, 1.0, (n_samples, n_blocks))
    block_loading = rng.uniform(0.4, 1.2, n_cpgs)
    m = cpg_baseline[None, :] + block_loading[None, :] * block_factor[:, block_id]

    # --- choose effect CpGs (disjoint sets), sized as fractions of n_cpgs so
    #     the generator stays valid from tiny smoke configs to genome scale.
    #     At least ~40% of CpGs are always left as true nulls for calibration. ---
    perm = rng.permutation(n_cpgs)
    n_age = max(3, int(round(0.075 * n_cpgs)))
    n_smk = max(3, int(round(0.06 * n_cpgs)))
    n_sex = max(2, int(round(0.05 * n_cpgs)))
    n_cell = max(4, int(round(0.15 * n_cpgs)))
    # guard: keep >=40% null even if rounding grew the effect sets
    max_effect = int(0.60 * n_cpgs)
    if n_age + n_smk + n_sex + n_cell > max_effect:
        scale = max_effect / (n_age + n_smk + n_sex + n_cell)
        n_age = max(3, int(n_age * scale)); n_smk = max(3, int(n_smk * scale))
        n_sex = max(2, int(n_sex * scale)); n_cell = max(4, int(n_cell * scale))
    a, b, c, d = n_age, n_age + n_smk, n_age + n_smk + n_sex, n_age + n_smk + n_sex + n_cell
    age_cpgs = perm[0:a]
    smoking_cpgs = perm[a:b]
    sex_cpgs = perm[b:c]
    cell_cpgs = perm[c:d]
    null_cpgs = perm[d:]

    age = pheno["age"].to_numpy()
    age_z = (age - spec.age_mean) / spec.age_sd
    smoker = pheno["smoker"].to_numpy()
    female = pheno["female"].to_numpy()
    cells = pheno[CELL_TYPES].to_numpy()

    # Age: ELOVL2-like hypermethylation. Strong, positive, monotonic.
    age_eff = rng.uniform(0.8, 1.6, age_cpgs.size)
    m[:, age_cpgs] += age_z[:, None] * age_eff[None, :]

    # Smoking: AHRR-like hypomethylation in smokers.
    smk_eff = rng.uniform(0.8, 1.8, smoking_cpgs.size)
    m[:, smoking_cpgs] -= smoker[:, None] * smk_eff[None, :]

    # Sex: shift.
    sex_eff = rng.uniform(0.6, 1.4, sex_cpgs.size) * rng.choice([-1.0, 1.0], sex_cpgs.size)
    m[:, sex_cpgs] += female[:, None] * sex_eff[None, :]

    # Cell composition: linear in proportions (centered), the dominant axis of
    # real blood DNAm variation.
    cells_c = cells - cells.mean(axis=0, keepdims=True)
    cell_weights = rng.normal(0.0, 1.0, (len(CELL_TYPES), cell_cpgs.size)) * 4.0
    m[:, cell_cpgs] += cells_c @ cell_weights

    # --- measurement noise ---
    m += rng.normal(0.0, 0.35, m.shape)

    cpg_names = [f"cg{idx:06d}" for idx in range(n_cpgs)]
    gt = GroundTruth(
        age_cpgs=age_cpgs,
        smoking_cpgs=smoking_cpgs,
        sex_cpgs=sex_cpgs,
        cell_cpgs=cell_cpgs,
        null_cpgs=null_cpgs,
        n_cpgs=n_cpgs,
        cpg_names=cpg_names,
    )
    return DNAmDataset(phenotypes=pheno, m_values=m.astype(np.float32), ground_truth=gt)


def train_val_split(
    ds: DNAmDataset, val_frac: float = 0.2, seed: int = 0
) -> Tuple["DNAmDataset", "DNAmDataset"]:
    """Split individuals into train/held-out sets (each individual used once)."""
    rng = np.random.default_rng(seed)
    n = ds.n_samples
    idx = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def subset(ix: np.ndarray) -> DNAmDataset:
        return DNAmDataset(
            phenotypes=ds.phenotypes.iloc[ix].reset_index(drop=True),
            m_values=ds.m_values[ix],
            ground_truth=ds.ground_truth,
        )

    return subset(train_idx), subset(val_idx)
