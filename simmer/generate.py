"""Conditional generation and cohort assembly (Aim 2).

``generate_profiles`` produces individual DNAm profiles for a given phenotype
table by sampling the latent prior and decoding under the conditioning vector.
``assemble_cohort`` draws a demographic structure from a ``PhenotypeSpec`` (the
user-specified cohort composition) and generates matching profiles, exporting a
parameter/seed manifest for reproducibility as described in the proposal.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .data import DNAmDataset, GroundTruth
from .phenotypes import PhenotypeSpec, encode_phenotypes, sample_phenotypes
from .train import TrainedModel


def generate_profiles(
    trained: TrainedModel,
    phenotypes: pd.DataFrame,
    seed: int = 0,
    z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate DNAm M-value profiles for the given phenotype table."""
    torch.manual_seed(seed)
    model = trained.model
    model.eval()
    c = torch.from_numpy(encode_phenotypes(phenotypes, trained.age_mean, trained.age_sd))
    z_t = None if z is None else torch.from_numpy(z.astype(np.float32))
    with torch.no_grad():
        gen_std = model.generate(c, z_t).cpu().numpy()
    # invert standardization back to the M-value scale
    return trained.scaler.inverse(gen_std).astype(np.float32)


def assemble_cohort(
    trained: TrainedModel,
    spec: PhenotypeSpec,
    ground_truth: GroundTruth,
    seed: int = 123,
) -> Tuple[DNAmDataset, Dict]:
    """Assemble a synthetic cohort under a user-specified demographic structure.

    Returns the synthetic ``DNAmDataset`` plus a reproducibility manifest
    (spec + seeds) that the proposal calls for exporting alongside simulations.
    """
    rng = np.random.default_rng(seed)
    spec = PhenotypeSpec(**{**asdict(spec)})
    spec.seed = seed
    pheno = sample_phenotypes(spec, rng=rng)
    m = generate_profiles(trained, pheno, seed=seed)

    synth = DNAmDataset(phenotypes=pheno, m_values=m, ground_truth=ground_truth)
    manifest = {
        "seed": seed,
        "spec": asdict(spec),
        "n_samples": int(spec.n),
        "n_cpgs": int(m.shape[1]),
        "model_config": asdict(trained.config),
    }
    return synth, manifest
