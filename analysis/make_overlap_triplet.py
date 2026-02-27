import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def load_tables():
    d0 = pd.read_csv(ROOT/"results_ov0/window_experiments_summary_quick.csv")
    d25 = pd.read_csv(ROOT/"results/window_experiments_summary_ov25.csv")
    d50 = pd.read_csv(ROOT/"results/window_experiments_summary_ov50.csv")
    cols=["phase","sensor","win_s","overlap","bacc_mean"]
    return d0[cols].copy(), d25[cols].copy(), d50[cols].copy()

def build_triplet(d0,d25,d50):
    triples=[
        ("pre_uturn","RF",3.0),("pre_uturn","RF",4.0),
        ("post_uturn","RF",4.0),
        ("gait_full","RF",3.0),
        ("pre_uturn","ALL",3.0),("pre_uturn","ALL",4.0),
        ("post_uturn","ALL",4.0),
        ("gait_full","ALL",3.0),
    ]
    rows=[]
    for ph,s,w in triples:
        def pick(df):
            sub=df[(df["phase"]==ph)&(df["sensor"]==s)&(np.abs(df["win_s"]-w)<1e-6)]
            return float(sub["bacc_mean"].max()) if not sub.empty else float("nan")
        rows.append({
            "phase":ph,"sensor":s,"win_s":w,
            "bacc_ov0":pick(d0),"bacc_ov25":pick(d25),"bacc_ov50":pick(d50)
        })
    out=pd.DataFrame(rows)
    outp=ROOT/"results_ov0/overlap_compare_triplet.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp,index=False)
    return out

def make_figures(df):
    outdir=ROOT/"results/figures"
    outdir.mkdir(parents=True, exist_ok=True)
    for (ph,s,w), sub in df.groupby(["phase","sensor","win_s"], dropna=False):
        vals=[sub["bacc_ov0"].values[0], sub["bacc_ov25"].values[0], sub["bacc_ov50"].values[0]]
        labs=["0%","25%","50%"]
        fig, ax = plt.subplots(figsize=(4.2,3.2))
        ax.bar(np.arange(3), vals, width=0.6)
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(labs)
        ax.set_ylim(0.5,1.0)
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"{ph} {s} {w:.2f}s")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir/f"overlap_triplet_{ph}_{s}_{int(round(w*1000))}ms.png", dpi=150)
        plt.close(fig)

def main():
    d0,d25,d50=load_tables()
    df=build_triplet(d0,d25,d50)
    make_figures(df)
    print("Wrote:", (ROOT/"results_ov0/overlap_compare_triplet.csv"))

if __name__ == "__main__":
    main()

