"""Phenotype / conditioning-variable definitions for SimMeR.

The conditioning vector fed to the cVAE mirrors the variables named in the
proposal: age, sex, smoking behaviour, health diagnosis, and estimated
cell-type proportions (an IDOL-style 6-part immune deconvolution). Everything
here is intentionally lightweight so the prototype stays interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Six immune cell types, matching the IDOL blood reference used in the proposal.
CELL_TYPES: List[str] = ["CD4T", "CD8T", "NK", "Bcell", "Mono", "Neu"]

# Diagnosis categories: a "control" plus a few chronic conditions named in the grant.
DIAGNOSES: List[str] = ["control", "SLE", "RA", "T2D"]


@dataclass
class PhenotypeSpec:
    """Describes the demographic structure of a (real or synthetic) cohort.

    Attributes define marginal distributions; ``sample_phenotypes`` draws a
    joint sample that also couples cell-type composition to age/sex/smoking so
    the reference data has realistic heterogeneity rather than independent axes.
    """

    n: int = 2000
    age_mean: float = 44.6
    age_sd: float = 18.8
    age_min: float = 18.0
    age_max: float = 90.0
    female_frac: float = 0.5
    smoker_frac: float = 0.20
    # Probability weights over DIAGNOSES (control first).
    diagnosis_probs: List[float] = field(default_factory=lambda: [0.80, 0.07, 0.07, 0.06])
    seed: Optional[int] = None


def _clip_age(age: np.ndarray, spec: PhenotypeSpec) -> np.ndarray:
    return np.clip(age, spec.age_min, spec.age_max)


def sample_phenotypes(spec: PhenotypeSpec, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Draw a joint sample of demographic + cell-type phenotypes.

    Cell-type proportions are drawn from a Dirichlet whose concentration is
    nudged by age (more neutrophils, fewer lymphocytes with age), sex, and
    smoking, so composition and demographics covary the way they do in real
    blood -- the exact confounding structure SimMeR is meant to let users
    control.
    """
    if rng is None:
        rng = np.random.default_rng(spec.seed)

    n = spec.n
    age = _clip_age(rng.normal(spec.age_mean, spec.age_sd, n), spec)
    female = (rng.random(n) < spec.female_frac).astype(int)
    smoker = (rng.random(n) < spec.smoker_frac).astype(int)
    diagnosis_idx = rng.choice(len(DIAGNOSES), size=n, p=np.asarray(spec.diagnosis_probs))
    diagnosis = np.asarray(DIAGNOSES)[diagnosis_idx]

    # Baseline Dirichlet concentration (roughly whole-blood proportions).
    base = np.array([0.13, 0.07, 0.06, 0.06, 0.08, 0.60])  # CD4T,CD8T,NK,Bcell,Mono,Neu
    age_z = (age - spec.age_mean) / spec.age_sd

    cell = np.zeros((n, len(CELL_TYPES)))
    for i in range(n):
        conc = base.copy()
        # Immunosenescence: neutrophil-to-lymphocyte ratio rises with age.
        conc[5] *= 1.0 + 0.35 * max(age_z[i], -1.0)      # Neu up with age
        conc[0] *= 1.0 - 0.20 * max(age_z[i], -1.0)      # CD4T down with age
        conc[1] *= 1.0 - 0.15 * max(age_z[i], -1.0)      # CD8T down with age
        if smoker[i]:
            conc[5] *= 1.10                               # smokers: more neutrophils
        if female[i]:
            conc[0] *= 1.08                               # females: slightly more CD4T
        conc = np.clip(conc, 0.01, None) * 50.0           # scale = tighter proportions
        cell[i] = rng.dirichlet(conc)

    df = pd.DataFrame(
        {
            "age": age,
            "female": female,
            "smoker": smoker,
            "diagnosis": diagnosis,
        }
    )
    for j, ct in enumerate(CELL_TYPES):
        df[ct] = cell[:, j]
    return df


def encode_phenotypes(df: pd.DataFrame, age_mean: float = 44.6, age_sd: float = 18.8) -> np.ndarray:
    """Encode a phenotype table into the numeric conditioning matrix for the cVAE.

    Layout (columns): standardized age, female, smoker, one-hot diagnosis
    (drop 'control' baseline), then the 6 cell-type proportions. Cell
    proportions are passed through as-is because they already sum to 1.
    """
    age_z = ((df["age"].to_numpy() - age_mean) / age_sd).reshape(-1, 1)
    female = df["female"].to_numpy().reshape(-1, 1).astype(float)
    smoker = df["smoker"].to_numpy().reshape(-1, 1).astype(float)

    diag = df["diagnosis"].to_numpy()
    diag_onehot = np.stack([(diag == d).astype(float) for d in DIAGNOSES[1:]], axis=1)

    cells = df[CELL_TYPES].to_numpy()

    return np.concatenate([age_z, female, smoker, diag_onehot, cells], axis=1).astype(np.float32)


def condition_dim() -> int:
    """Dimensionality of the encoded conditioning vector."""
    # age(1) + female(1) + smoker(1) + diagnosis one-hot(len-1) + cell types
    return 3 + (len(DIAGNOSES) - 1) + len(CELL_TYPES)


def condition_columns() -> List[str]:
    cols = ["age_z", "female", "smoker"]
    cols += [f"dx_{d}" for d in DIAGNOSES[1:]]
    cols += list(CELL_TYPES)
    return cols
