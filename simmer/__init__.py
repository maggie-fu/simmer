"""SimMeR: Simulating DNA Methylation with Demographic Realism (prototype).

A minimal, CPU-friendly prototype of the phenotype-aware DNA methylation (DNAm)
cohort simulator described in the SimMeR NIH R21 proposal. It implements a
conditional variational autoencoder (cVAE) that generates individual genome-wide
DNAm profiles conditioned on demographic variables and cell-type composition,
plus cohort assembly and validation utilities.

The modules are deliberately small so the whole Aim 1 -> Aim 2 pipeline runs
end-to-end in minutes on a CPU, using a synthetic-but-structured stand-in for the
~30,000 real blood DNAm profiles that require data-use agreements and GPU-scale
training.
"""

from .phenotypes import PhenotypeSpec, sample_phenotypes, encode_phenotypes
from .data import DNAmDataset, make_reference_dataset
from .model import ConditionalVAE
from .train import TrainConfig, train_cvae
from .generate import generate_profiles, assemble_cohort
from .evaluate import evaluate_prototype

__all__ = [
    "PhenotypeSpec",
    "sample_phenotypes",
    "encode_phenotypes",
    "DNAmDataset",
    "make_reference_dataset",
    "ConditionalVAE",
    "TrainConfig",
    "train_cvae",
    "generate_profiles",
    "assemble_cohort",
    "evaluate_prototype",
]

__version__ = "0.1.0"
