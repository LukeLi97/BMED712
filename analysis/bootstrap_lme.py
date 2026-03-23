"""
Step 11 – Bootstrap confidence intervals + Linear Mixed Effects Model.

Addresses professor's requests (3/7):
  1. Bootstrap 95% CI for Cohen's d = 0.77  (stride_absAI, healthy vs pathological)
  2. Bootstrap 95% CI for AUC = 0.716       (stride_absAI ROC)
  3. Linear Mixed Effects Model:
       Outcome  : stride_AI  (trial-level)
       Fixed    : pathology group (3 levels)
       Random   : (1 | subject)

Usage:
    cd "BMED712 Project 1_Track A"
    source .venv/bin/activate
    python analysis/bootstrap_lme.py

Outputs:
    results/artifacts/bootstrap_lme_results.json
    results/figures/step11_bootstrap_lme.png
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf


# ── paths ──────────────────────────────────────────────────────────────────────
BASE_PATH = Path(__file__).resolve().parent.parent
ARTIFACTS = BASE_PATH / "results" / "artifacts"
FIGURES   = BASE_PATH / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Cohen's d helper ───────────────────────────────────────────────────────────
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Welch/pooled Cohen's d (signed: a vs b)."""
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


# ── bootstrap CI ───────────────────────────────────────────────────────────────
def bootstrap_cohens_d(
    healthy: np.ndarray,
    patho: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for Cohen's d (BCa method)."""
    rng = np.random.default_rng(seed)
    obs_d = cohens_d(healthy, patho)
    boot_ds = np.empty(n_boot)
    for i in range(n_boot):
        bh = rng.choice(healthy, size=len(healthy), replace=True)
        bp = rng.choice(patho,   size=len(patho),   replace=True)
        boot_ds[i] = cohens_d(bh, bp)

    # Percentile CI
    lo, hi = np.percentile(boot_ds, [2.5, 97.5])
    return {"observed": float(obs_d), "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot, "method": "percentile"}


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    obs_auc = float(roc_auc_score(y_true, y_score))
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yt, ys))
    boot_aucs = np.array(boot_aucs)
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return {"observed": obs_auc, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": len(boot_aucs), "method": "percentile"}


# ── LME ────────────────────────────────────────────────────────────────────────
def fit_lme(df_trial: pd.DataFrame) -> dict:
    """
    stride_AI ~ C(group, Treatment('healthy')) + (1 | subject)
    Returns fixed-effect table as dict, plus model summary.
    """
    # Encode group as categorical with healthy as reference
    df = df_trial[["stride_AI", "group", "subject"]].dropna().copy()
    df["group"] = pd.Categorical(df["group"],
                                  categories=["healthy", "ortho", "neuro"])

    model = smf.mixedlm(
        "stride_AI ~ C(group, Treatment('healthy'))",
        data=df,
        groups=df["subject"],
    )
    result = model.fit(method="lbfgs", reml=True)

    fe = result.fe_params.to_dict()
    pvals = result.pvalues.to_dict()
    conf = result.conf_int()
    conf_dict = {k: {"ci_low": float(v[0]), "ci_high": float(v[1])}
                 for k, v in conf.iterrows()}

    fixed_effects = {}
    for k in fe:
        fixed_effects[k] = {
            "coef": float(fe[k]),
            "pvalue": float(pvals[k]),
            "ci_low": float(conf_dict[k]["ci_low"]),
            "ci_high": float(conf_dict[k]["ci_high"]),
        }

    random_var = float(result.cov_re.values[0, 0])
    resid_var  = float(result.scale)

    # ICC = var_subject / (var_subject + var_resid)
    icc = random_var / (random_var + resid_var)

    return {
        "n_trials": len(df),
        "n_subjects": df["subject"].nunique(),
        "fixed_effects": fixed_effects,
        "random_effect_variance": random_var,
        "residual_variance": resid_var,
        "icc": float(icc),
        "log_likelihood": float(result.llf),
        "converged": bool(result.converged),
    }


# ── figure ─────────────────────────────────────────────────────────────────────
def plot_results(
    d_result: dict,
    auc_result: dict,
    lme_result: dict,
    path: Path,
) -> None:
    """Three-panel summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Step 11: Bootstrap CIs & Linear Mixed Effects Model\n"
        "stride_absAI (healthy vs pathological, subject-level)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    # Panel 1 – Cohen's d with CI
    ax = axes[0]
    d_obs = d_result["observed"]
    d_lo, d_hi = d_result["ci_low"], d_result["ci_high"]
    ax.barh(["stride_absAI"], [d_obs], color="#4C72B0", xerr=[[d_obs - d_lo], [d_hi - d_obs]],
            capsize=8, ecolor="black", height=0.4)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    for thresh, lbl, col in [(0.2, "small", "#90EE90"), (0.5, "medium", "#FFA500"),
                              (0.8, "large", "#FF6347")]:
        ax.axvline(thresh, color=col, lw=1.2, ls=":", alpha=0.8)
        ax.text(thresh + 0.01, 0.55, lbl, fontsize=7, color=col, va="center")
    ax.set_xlabel("Cohen's d", fontsize=9)
    ax.set_title(
        f"Cohen's d = {d_obs:.2f}\n95% CI [{d_lo:.2f}, {d_hi:.2f}]",
        fontsize=9,
    )
    ax.set_xlim(0, max(1.1, d_hi + 0.15))
    ax.set_yticks([])

    # Panel 2 – AUC with CI
    ax = axes[1]
    auc_obs = auc_result["observed"]
    auc_lo, auc_hi = auc_result["ci_low"], auc_result["ci_high"]
    ax.barh(["stride_absAI"], [auc_obs], color="#DD8452", xerr=[[auc_obs - auc_lo], [auc_hi - auc_obs]],
            capsize=8, ecolor="black", height=0.4)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.axvline(0.7, color="#FFA500", lw=1.2, ls=":", alpha=0.8)
    ax.text(0.71, 0.55, "acceptable", fontsize=7, color="#FFA500", va="center")
    ax.set_xlabel("AUC", fontsize=9)
    ax.set_title(
        f"AUC = {auc_obs:.3f}\n95% CI [{auc_lo:.3f}, {auc_hi:.3f}]",
        fontsize=9,
    )
    ax.set_xlim(0.4, 0.9)
    ax.set_yticks([])

    # Panel 3 – LME fixed effects
    ax = axes[2]
    fe = lme_result["fixed_effects"]
    labels, coefs, cis_lo, cis_hi = [], [], [], []
    for k, v in fe.items():
        if k == "Intercept":
            continue
        short = k.replace("C(group, Treatment('healthy'))[T.", "").replace("]", "")
        labels.append(short)
        coefs.append(v["coef"])
        cis_lo.append(v["coef"] - v["ci_low"])
        cis_hi.append(v["ci_high"] - v["coef"])

    y = np.arange(len(labels))
    colors = ["#4C72B0" if c > 0 else "#C44E52" for c in coefs]
    ax.barh(y, coefs, color=colors,
            xerr=[cis_lo, cis_hi], capsize=6, ecolor="black", height=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Fixed effect (stride_AI)", fontsize=9)
    icc = lme_result["icc"]
    ax.set_title(
        f"LME fixed effects (ref: healthy)\nICC(subject) = {icc:.3f}",
        fontsize=9,
    )

    # p-value annotations
    for i, k in enumerate(fe):
        if k == "Intercept":
            continue
        pv = fe[k]["pvalue"]
        stars = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        ci_hi = fe[k]["ci_high"]
        ax.text(ci_hi + 0.0005, i, f" {stars}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure → {path.name}")


# ── main ───────────────────────────────────────────────────────────────────────
def run() -> None:
    print("\n=== Step 11: Bootstrap CIs + LME ===\n")

    # Load data
    df_subj  = pd.read_csv(ARTIFACTS / "asymmetry_per_subject.csv")
    df_trial = pd.read_csv(ARTIFACTS / "asymmetry_per_trial.csv")

    # Add pathology binary label
    df_subj["pathological"] = (df_subj["group"] != "healthy").astype(int)

    healthy_vals = df_subj.loc[df_subj["group"] == "healthy", "stride_absAI"].values
    patho_vals   = df_subj.loc[df_subj["group"] != "healthy", "stride_absAI"].values

    print(f"Subjects: healthy={len(healthy_vals)}, pathological={len(patho_vals)}")

    # ── 1. Bootstrap Cohen's d ──────────────────────────────────────────────────
    print("\n[1] Bootstrapping Cohen's d (n_boot=10000)...")
    d_result = bootstrap_cohens_d(healthy_vals, patho_vals, n_boot=10_000)
    print(
        f"    Cohen's d = {d_result['observed']:.4f}  "
        f"(95% CI: {d_result['ci_low']:.2f} – {d_result['ci_high']:.2f})"
    )

    # ── 2. Bootstrap AUC ───────────────────────────────────────────────────────
    print("\n[2] Bootstrapping AUC (n_boot=10000)...")
    # AUC: healthy=1 (positive), higher stride_absAI → more likely healthy
    # This matches the convention in asymmetry_roc.json (AUC=0.716)
    y_true  = (df_subj["group"] == "healthy").astype(int).values
    y_score = df_subj["stride_absAI"].values
    auc_result = bootstrap_auc(y_true, y_score, n_boot=10_000)
    print(
        f"    AUC = {auc_result['observed']:.4f}  "
        f"(95% CI: {auc_result['ci_low']:.3f} – {auc_result['ci_high']:.3f})"
    )

    # ── 3. Linear Mixed Effects Model ──────────────────────────────────────────
    print("\n[3] Fitting LME: stride_AI ~ group + (1|subject)...")
    print(f"    Trial-level n = {len(df_trial)}, subjects = {df_trial['subject'].nunique()}")
    lme_result = fit_lme(df_trial)
    print(f"    Converged: {lme_result['converged']}")
    print(f"    ICC(subject): {lme_result['icc']:.4f}")
    print("    Fixed effects:")
    for k, v in lme_result["fixed_effects"].items():
        stars = "***" if v["pvalue"] < 0.001 else (
            "**" if v["pvalue"] < 0.01 else ("*" if v["pvalue"] < 0.05 else "ns"))
        print(
            f"      {k:55s}: β={v['coef']:+.5f}  "
            f"95%CI[{v['ci_low']:+.5f}, {v['ci_high']:+.5f}]  "
            f"p={v['pvalue']:.4g}  {stars}"
        )

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "cohens_d_stride_absAI": d_result,
        "auc_stride_absAI": auc_result,
        "lme_stride_AI": lme_result,
    }
    out_json = ARTIFACTS / "bootstrap_lme_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON → {out_json.name}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    plot_results(
        d_result, auc_result, lme_result,
        FIGURES / "step11_bootstrap_lme.png",
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    run()
