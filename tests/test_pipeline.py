"""Fast smoke/regression tests for the SimMeR prototype pipeline."""

import numpy as np

from simmer import (
    PhenotypeSpec,
    assemble_cohort,
    evaluate_prototype,
    make_reference_dataset,
    train_cvae,
)
from simmer.data import train_val_split
from simmer.phenotypes import condition_dim, encode_phenotypes
from simmer.train import TrainConfig


def _tiny():
    ref = make_reference_dataset(n_samples=800, n_cpgs=150, seed=1)
    train, val = train_val_split(ref, val_frac=0.2, seed=1)
    return ref, train, val


def test_reference_shapes_and_ground_truth():
    ref, _, _ = _tiny()
    assert ref.m_values.shape == (800, 150)
    gt = ref.ground_truth
    # effect sets disjoint and nulls non-empty
    all_effect = np.concatenate([gt.age_cpgs, gt.smoking_cpgs, gt.sex_cpgs, gt.cell_cpgs])
    assert len(np.unique(all_effect)) == all_effect.size
    assert gt.null_cpgs.size > 0
    assert gt.null_cpgs.size >= int(0.35 * ref.n_cpgs)


def test_encode_dim_matches_condition_dim():
    ref, _, _ = _tiny()
    enc = encode_phenotypes(ref.phenotypes)
    assert enc.shape == (ref.n_samples, condition_dim())


def test_train_generate_evaluate_end_to_end():
    ref, train, val = _tiny()
    cfg = TrainConfig(epochs=25, latent_dim=16, seed=1)
    trained = train_cvae(train, val, cfg, verbose=False)

    spec = PhenotypeSpec(n=500, age_mean=50, smoker_frac=0.3, seed=1)
    synth, manifest = assemble_cohort(trained, spec, ref.ground_truth, seed=2)
    assert synth.m_values.shape == (500, ref.n_cpgs)
    assert manifest["n_samples"] == 500

    metrics = evaluate_prototype(train, val, synth)
    # Core biological signal must be recovered and the null must not be inflated.
    assert metrics["age_recovery"]["effect_corr_all_cpgs"] > 0.5
    assert metrics["age_recovery"]["sign_agreement_at_age_cpgs"] > 0.8
    assert metrics["structure_preservation"]["marginal_mean_corr"] > 0.85
    assert metrics["null_calibration"]["type1_error_at_alpha_0.05"] <= 0.10
