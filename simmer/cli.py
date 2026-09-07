"""End-to-end prototype runner and CLI for SimMeR.

Runs the full Aim 1 -> Aim 2 pipeline:
  1. build a structured synthetic reference dataset (stand-in for real GEO data);
  2. train the conditional VAE (Aim 1);
  3. assemble a synthetic cohort under a user-specified demographic structure (Aim 2);
  4. evaluate biological plausibility and inferential behaviour;
  5. write metrics JSON, a reproducibility manifest, and diagnostic figures.

Usage:
    python -m simmer.cli --outdir artifacts --epochs 60
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict

import numpy as np

from .data import make_reference_dataset, train_val_split
from .evaluate import ewas, evaluate_prototype
from .generate import assemble_cohort
from .phenotypes import PhenotypeSpec
from .train import TrainConfig, train_cvae


def _plot(real_val, synth, trained, results, outdir: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = real_val.ground_truth
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) training curves
    ax = axes[0, 0]
    ax.plot(trained.history["train_recon"], label="train recon")
    if trained.history["val_recon"]:
        ax.plot(trained.history["val_recon"], label="val recon")
    ax.set_title("cVAE training (reconstruction loss)")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (sum over CpGs)"); ax.legend()

    # (0,1) age effect sizes real vs synth
    age_real = ewas(real_val.m_values, real_val.phenotypes["age"].to_numpy())
    age_synth = ewas(synth.m_values, synth.phenotypes["age"].to_numpy())
    ax = axes[0, 1]
    ax.scatter(age_real["beta"], age_synth["beta"], s=8, alpha=0.4, color="gray", label="all CpGs")
    ax.scatter(age_real["beta"][gt.age_cpgs], age_synth["beta"][gt.age_cpgs], s=20, color="crimson", label="age CpGs")
    lim = np.array([age_real["beta"].min(), age_real["beta"].max()])
    ax.plot(lim, lim, "k--", lw=1)
    r = results["age_recovery"]["effect_corr_all_cpgs"]
    ax.set_title(f"Age EWAS effect recovery (r={r:.2f})")
    ax.set_xlabel("real effect"); ax.set_ylabel("synthetic effect"); ax.legend()

    # (0,2) smoking effect sizes real vs synth
    smk_real = ewas(real_val.m_values, real_val.phenotypes["smoker"].to_numpy().astype(float))
    smk_synth = ewas(synth.m_values, synth.phenotypes["smoker"].to_numpy().astype(float))
    ax = axes[0, 2]
    ax.scatter(smk_real["beta"], smk_synth["beta"], s=8, alpha=0.4, color="gray", label="all CpGs")
    ax.scatter(smk_real["beta"][gt.smoking_cpgs], smk_synth["beta"][gt.smoking_cpgs], s=20, color="seagreen", label="smoking CpGs")
    lim = np.array([smk_real["beta"].min(), smk_real["beta"].max()])
    ax.plot(lim, lim, "k--", lw=1)
    r = results["smoking_recovery"]["effect_corr_all_cpgs"]
    ax.set_title(f"Smoking EWAS effect recovery (r={r:.2f})")
    ax.set_xlabel("real effect"); ax.set_ylabel("synthetic effect"); ax.legend()

    # (1,0) marginal means real vs synth
    ax = axes[1, 0]
    ax.scatter(real_val.m_values.mean(0), synth.m_values.mean(0), s=8, alpha=0.5)
    lim = np.array([real_val.m_values.mean(0).min(), real_val.m_values.mean(0).max()])
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_title(f"Per-CpG mean (r={results['structure_preservation']['marginal_mean_corr']:.2f})")
    ax.set_xlabel("real"); ax.set_ylabel("synthetic")

    # (1,1) null calibration histogram (pooled over random null predictors)
    rng = np.random.default_rng(0)
    null_m = synth.m_values[:, gt.null_cpgs]
    null_p = np.concatenate([ewas(null_m, rng.standard_normal(null_m.shape[0]))["p"] for _ in range(25)])
    ax = axes[1, 1]
    ax.hist(null_p, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    ax.axhline(len(null_p) / 20, color="k", ls="--", lw=1)
    t1 = results["null_calibration"]["type1_error_at_alpha_0.05"]
    ax.set_title(f"Null p-values (Type-I @0.05 = {t1:.3f})")
    ax.set_xlabel("p-value"); ax.set_ylabel("count")

    # (1,2) anti-memorization distances
    ax = axes[1, 2]
    am = results["anti_memorization"]
    ax.bar(["synth->train", "real val->train"], [am["synth_nn_median"], am["real_val_nn_median"]],
           color=["darkorange", "slateblue"])
    ax.set_title(f"Anti-memorization NN dist (ratio={am['ratio_synth_over_real']:.2f})")
    ax.set_ylabel("median nearest-neighbour distance")

    fig.tight_layout()
    path = os.path.join(outdir, "simmer_evaluation.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> Dict:
    os.makedirs(args.outdir, exist_ok=True)
    print("[1/4] Building structured synthetic reference dataset (stand-in for real GEO DNAm)...")
    ref = make_reference_dataset(
        n_samples=args.n_samples, n_cpgs=args.n_cpgs, seed=args.seed
    )
    train_ds, val_ds = train_val_split(ref, val_frac=0.2, seed=args.seed)
    print(f"      reference: {ref.n_samples} individuals x {ref.n_cpgs} CpGs "
          f"(train {train_ds.n_samples} / held-out {val_ds.n_samples})")

    print("[2/4] Training conditional VAE (Aim 1)...")
    cfg = TrainConfig(epochs=args.epochs, latent_dim=args.latent_dim, seed=args.seed)
    trained = train_cvae(train_ds, val_ds, cfg)

    print("[3/4] Assembling synthetic cohort under specified demographic structure (Aim 2)...")
    # Example user-specified cohort: older, higher smoking prevalence than the training marginal.
    cohort_spec = PhenotypeSpec(
        n=args.cohort_n, age_mean=55.0, age_sd=15.0, female_frac=0.5, smoker_frac=0.35, seed=args.seed
    )
    synth, manifest = assemble_cohort(trained, cohort_spec, ref.ground_truth, seed=args.seed + 7)

    print("[4/4] Evaluating biological plausibility and inferential behaviour...")
    results = evaluate_prototype(train_ds, val_ds, synth)

    # persist artifacts
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.outdir, "cohort_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    np.save(os.path.join(args.outdir, "synthetic_cohort_mvalues.npy"), synth.m_values)
    synth.phenotypes.to_csv(os.path.join(args.outdir, "synthetic_cohort_phenotypes.csv"), index=False)

    fig_path = _plot(val_ds, synth, trained, results, args.outdir)

    print("\n=== SimMeR prototype evaluation summary ===")
    s = results["summary"]
    print(f"Hard gates passed: {s['gates_passed']}/{s['gates_total']}")
    for k in ["anti_memorization", "age_recovery", "smoking_recovery", "structure_preservation", "null_calibration"]:
        print(f"  - {k:24s}: {'PASS' if results[k]['passed'] else 'FAIL'}")
    print(f"Figure : {fig_path}")
    print(f"Metrics: {os.path.join(args.outdir, 'metrics.json')}")
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SimMeR cVAE DNAm simulation prototype")
    p.add_argument("--outdir", default="artifacts", help="output directory")
    p.add_argument("--n-samples", type=int, default=3000, help="reference dataset size")
    p.add_argument("--n-cpgs", type=int, default=400, help="number of CpGs (probes)")
    p.add_argument("--cohort-n", type=int, default=1500, help="synthetic cohort size")
    p.add_argument("--epochs", type=int, default=60, help="training epochs")
    p.add_argument("--latent-dim", type=int, default=32, help="cVAE latent dimension")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
