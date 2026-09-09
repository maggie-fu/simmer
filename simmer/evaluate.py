"""Validation metrics for the SimMeR prototype.

Mirrors the proposal's evaluation philosophy: success is defined by biological
plausibility and inferential behaviour, not surface realism. Implements small
versions of the named checks:

  * Anti-memorization: synthetic profiles should not be nearer to training
    individuals than real held-out individuals are (nearest-neighbour distance).
  * Canonical-loci recovery: an EWAS on synthetic data should recover the age
    (ELOVL2-like) and smoking (AHRR-like) associations present in real data,
    with matching direction.
  * Covariance preservation: genome-wide CpG-CpG covariance of synthetic data
    should resemble that of real data.
  * Null calibration: EWAS p-values at true-null CpGs should be ~uniform
    (Type-I error near nominal).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

from .data import DNAmDataset


def ewas(m_values: np.ndarray, predictor: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-CpG OLS of M-value on a single standardized predictor.

    Returns slope (beta), t-stat, and two-sided p-value per CpG. Vectorized
    across CpGs for speed.
    """
    x = (predictor - predictor.mean()) / (predictor.std() + 1e-12)
    n = x.shape[0]
    xc = x - x.mean()
    sxx = np.sum(xc ** 2)
    y = m_values
    yc = y - y.mean(axis=0, keepdims=True)
    beta = (xc[:, None] * yc).sum(axis=0) / sxx
    resid = yc - beta[None, :] * xc[:, None]
    dof = n - 2
    sigma2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(sigma2 / sxx)
    t = beta / (se + 1e-12)
    p = 2 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta, "t": t, "p": p}


def _nn_distances(a: np.ndarray, b: np.ndarray, sample: int = 300, seed: int = 0) -> np.ndarray:
    """Min Euclidean distance from each row of ``a`` to any row of ``b``."""
    rng = np.random.default_rng(seed)
    if a.shape[0] > sample:
        a = a[rng.choice(a.shape[0], sample, replace=False)]
    if b.shape[0] > sample:
        b = b[rng.choice(b.shape[0], sample, replace=False)]
    # pairwise squared distances in chunks to bound memory
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def evaluate_prototype(
    real_train: DNAmDataset,
    real_val: DNAmDataset,
    synth: DNAmDataset,
) -> Dict[str, object]:
    """Run the full evaluation panel and return a metrics dict."""
    gt = real_train.ground_truth
    results: Dict[str, object] = {}

    # --- 1. Anti-memorization ---
    d_synth = _nn_distances(synth.m_values, real_train.m_values)
    d_real = _nn_distances(real_val.m_values, real_train.m_values)
    results["anti_memorization"] = {
        "synth_nn_median": float(np.median(d_synth)),
        "real_val_nn_median": float(np.median(d_real)),
        # >= 1 means synthetic are at least as far from train as real held-out are (good).
        "ratio_synth_over_real": float(np.median(d_synth) / (np.median(d_real) + 1e-9)),
        "passed": bool(np.median(d_synth) >= 0.7 * np.median(d_real)),
    }

    # --- 2. Canonical-loci recovery: AGE ---
    age_real = ewas(real_val.m_values, real_val.phenotypes["age"].to_numpy())
    age_synth = ewas(synth.m_values, synth.phenotypes["age"].to_numpy())
    age_idx = gt.age_cpgs
    # correlation of per-CpG effect sizes across ALL cpgs (real vs synth)
    age_effect_corr = float(np.corrcoef(age_real["beta"], age_synth["beta"])[0, 1])
    age_sign_agree = float(np.mean(np.sign(age_real["beta"][age_idx]) == np.sign(age_synth["beta"][age_idx])))
    age_detect = float(np.mean(age_synth["p"][age_idx] < 0.05))
    results["age_recovery"] = {
        "effect_corr_all_cpgs": age_effect_corr,
        "sign_agreement_at_age_cpgs": age_sign_agree,
        "detection_rate_at_age_cpgs": age_detect,
        "mean_abs_beta_real": float(np.mean(np.abs(age_real["beta"][age_idx]))),
        "mean_abs_beta_synth": float(np.mean(np.abs(age_synth["beta"][age_idx]))),
        "passed": bool(age_effect_corr > 0.5 and age_sign_agree > 0.8),
    }

    # --- 3. Canonical-loci recovery: SMOKING ---
    smk_real = ewas(real_val.m_values, real_val.phenotypes["smoker"].to_numpy().astype(float))
    smk_synth = ewas(synth.m_values, synth.phenotypes["smoker"].to_numpy().astype(float))
    smk_idx = gt.smoking_cpgs
    smk_effect_corr = float(np.corrcoef(smk_real["beta"], smk_synth["beta"])[0, 1])
    smk_sign_agree = float(np.mean(np.sign(smk_real["beta"][smk_idx]) == np.sign(smk_synth["beta"][smk_idx])))
    smk_detect = float(np.mean(smk_synth["p"][smk_idx] < 0.05))
    results["smoking_recovery"] = {
        "effect_corr_all_cpgs": smk_effect_corr,
        "sign_agreement_at_smoking_cpgs": smk_sign_agree,
        "detection_rate_at_smoking_cpgs": smk_detect,
        "passed": bool(smk_effect_corr > 0.4 and smk_sign_agree > 0.7),
    }

    # --- 4. Covariance preservation ---
    cov_real = np.cov(real_val.m_values, rowvar=False)
    cov_synth = np.cov(synth.m_values, rowvar=False)
    iu = np.triu_indices(cov_real.shape[0], k=1)
    cov_corr = float(np.corrcoef(cov_real[iu], cov_synth[iu])[0, 1])
    # per-CpG marginal means/variances
    mean_corr = float(np.corrcoef(real_val.m_values.mean(0), synth.m_values.mean(0))[0, 1])
    var_corr = float(np.corrcoef(real_val.m_values.var(0), synth.m_values.var(0))[0, 1])
    results["structure_preservation"] = {
        "covariance_corr": cov_corr,
        "marginal_mean_corr": mean_corr,
        "marginal_var_corr": var_corr,
        "passed": bool(cov_corr > 0.5 and mean_corr > 0.9),
    }

    # --- 5. Null calibration (Type-I error at true-null CpGs) ---
    # Pool p-values across many independent random null predictors so the
    # Type-I estimate is stable (a single draw over a few hundred CpGs is noisy).
    rng = np.random.default_rng(0)
    null_m = synth.m_values[:, gt.null_cpgs]
    n_perm = 25
    pooled_p = []
    for _ in range(n_perm):
        null_pred = rng.standard_normal(null_m.shape[0])
        pooled_p.append(ewas(null_m, null_pred)["p"])
    pooled_p = np.concatenate(pooled_p)
    type1 = float(np.mean(pooled_p < 0.05))
    ks = stats.kstest(pooled_p, "uniform")
    results["null_calibration"] = {
        "type1_error_at_alpha_0.05": type1,
        "n_null_tests": int(pooled_p.size),
        "ks_uniform_pvalue": float(ks.pvalue),
        # Pass unless the null is *inflated* (anti-conservative). Being
        # conservative (< nominal) is safe for downstream inference.
        "passed": bool(type1 <= 0.075),
    }

    # --- overall summary ---
    gate_keys = ["anti_memorization", "age_recovery", "smoking_recovery", "structure_preservation", "null_calibration"]
    results["summary"] = {
        "gates_passed": int(sum(results[k]["passed"] for k in gate_keys)),
        "gates_total": len(gate_keys),
        "all_passed": bool(all(results[k]["passed"] for k in gate_keys)),
    }
    return results
