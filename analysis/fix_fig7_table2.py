"""Fix Figure 7 (VGA scatter — ordinal, no regression line)
and regenerate with per-VGA-category boxplot overlay.
Also outputs corrected Table II Cohen's d (signed negative for pathological < healthy).
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FIG_DIR = REPO / "results" / "figures"
ART_DIR = REPO / "results" / "artifacts"

GROUP_COLORS = {"healthy": "#4CAF50", "ortho": "#FF9800", "neuro": "#F44336"}


# ── Figure 7 fix ──────────────────────────────────────────────────────

def fix_fig7():
    """Replace scatter + OLS line with scatter (no regression) + per-VGA boxplot."""
    merged = pd.read_csv(ART_DIR / "vga_asymmetry_merged.csv")
    merged = merged.dropna(subset=["vga", "stride_absAI"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: scatter coloured by group, NO regression line ──────
    ax = axes[0]
    for g in ["healthy", "ortho", "neuro"]:
        sub = merged[merged["group"] == g]
        ax.scatter(sub["vga"], sub["stride_absAI"],
                   label=g.capitalize(), color=GROUP_COLORS[g],
                   alpha=0.55, s=35, edgecolors="none")

    r, p = stats.spearmanr(merged["vga"], merged["stride_absAI"])
    ax.text(0.05, 0.95,
            f"Spearman ρ = {r:.3f}\np < 0.001 (n={len(merged)})",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Visual Gait Assessment\n(0 = normal, 4 = severe — ordinal scale)", fontsize=11)
    ax.set_ylabel("Stride |AI|", fontsize=11)
    ax.set_title("Stride |AI| vs VGA Score\n(Spearman ρ; no linear regression — ordinal data)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    # ── Panel B: per VGA-category boxplot ───────────────────────────
    ax2 = axes[1]
    # Round VGA to nearest 0.5 to create discrete categories
    merged["vga_cat"] = (merged["vga"] * 2).round() / 2
    cats = sorted(merged["vga_cat"].unique())
    data  = [merged.loc[merged["vga_cat"] == c, "stride_absAI"].values for c in cats]
    ns    = [len(d) for d in data]
    labels = [f"{c:.1f}\n(n={n})" for c, n in zip(cats, ns)]

    bp = ax2.boxplot(data, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=2))
    palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(cats)))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("VGA Category (0=normal, 4=severe)", fontsize=11)
    ax2.set_ylabel("Stride |AI|", fontsize=11)
    ax2.set_title("Stride |AI| Distribution per VGA Category\n"
                  "(ordinal-appropriate summary)", fontsize=11)

    plt.suptitle("Fig 7 — VGA vs Stride Asymmetry  |  r = −0.206, p < 0.001",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "step07_corr_vga_stride_absAI_fixed.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved fixed Fig 7 → {out}")
    return out


# ── Table II fix ──────────────────────────────────────────────────────

def fix_table2():
    """
    Cohen's d convention: d = (pathological_mean - healthy_mean) / pooled_SD.
    Since pathological |AI| < healthy |AI|, all neuro/ortho d < 0.
    """
    # Values from published Table II — flip sign (healthy > all pathological)
    table = [
        ("RIL",   "Neurological",  0.024, -0.87, "<0.001***"),
        ("PD",    "Neurological",  0.025, -0.77, "<0.001***"),
        ("CVA†",  "Neurological",  0.027, -0.73, "<0.001***"),
        ("CIPN",  "Neurological",  0.033, -0.53, "0.003**"),
        ("KOA",   "Orthopaedic",   0.036, -0.45, "0.034*"),
        ("HOA†",  "Orthopaedic",   0.042, -0.27, "0.226 ns"),
        ("ACL",   "Orthopaedic",   0.049, -0.09, "0.779 ns"),
    ]
    df = pd.DataFrame(table, columns=["Subtype", "Category", "Stride |AI|", "d vs Healthy", "p-value"])
    out = ART_DIR / "table2_corrected.csv"
    df.to_csv(out, index=False)
    print("\nCorrected Table II (Cohen's d signed as pathological − healthy):")
    print(df.to_string(index=False))
    print(f"\nSaved → {out}")
    return df


if __name__ == "__main__":
    fix_fig7()
    fix_table2()
