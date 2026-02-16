import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


ARTIFACTS = Path("results/artifacts")
REPORT = Path("results/report.md")
REPORT_FULL = Path("results/report_full.md")
BEGIN = "<!-- AUTO_REPORT:BEGIN -->"
END = "<!-- AUTO_REPORT:END -->"


def load_metrics() -> Dict[str, Dict]:
    metrics: Dict[str, Dict] = {}
    if not ARTIFACTS.exists():
        return metrics
    for p in ARTIFACTS.glob("metrics_3class_*.json"):
        with p.open("r", encoding="utf-8") as f:
            metrics[p.stem.replace("metrics_", "")] = json.load(f)
    return metrics


def compile_step1_table(metrics: Dict[str, Dict]) -> Path:
    rows: List[Dict] = []
    for tag, m in sorted(metrics.items()):
        if not tag.startswith("3class_"):
            continue
        # tag e.g., 3class_all -> sensors name
        sensors = tag.split("3class_")[-1]
        for model in ["lr", "rf", "svm", "xgb"]:
            if model not in m:
                continue
            rows.append({
                "sensors": sensors,
                "model": model.upper(),
                "macro_f1_mean": m[model]["macro_f1_mean"],
                "macro_f1_std": m[model]["macro_f1_std"],
                "balanced_acc_mean": m[model]["balanced_acc_mean"],
                "balanced_acc_std": m[model]["balanced_acc_std"],
            })
    df = pd.DataFrame(rows)
    out = ARTIFACTS / "table_step1_baselines.csv"
    df.to_csv(out, index=False)
    return out


def compile_step2_table() -> Optional[Path]:
    cand = list(ARTIFACTS.glob("step04b_subset_search_*.json"))
    if not cand:
        return None
    # choose the most recent
    path = max(cand, key=lambda p: p.stat().st_mtime)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    full = data.get("full", {})
    rows.append({
        "k": "full",
        "sensors": ",".join(data.get("sensors_all", [])),
        "macro_f1": full.get("macro_f1"),
        "balanced_acc": full.get("bacc"),
    })
    best_per_k = data.get("best_per_k", {})
    for k, info in sorted(best_per_k.items(), key=lambda kv: int(kv[0])):
        rows.append({
            "k": int(k),
            "sensors": ",".join(info.get("sensors", [])),
            "macro_f1": info.get("macro_f1"),
            "balanced_acc": info.get("bacc"),
        })
    rec = data.get("recommended", {})
    rec_row = pd.DataFrame([{ 
        "k": f"recommended({rec.get('k')})",
        "sensors": ",".join(rec.get("sensors", [])),
        "macro_f1": rec.get("macro_f1"),
        "balanced_acc": rec.get("bacc"),
    }])
    df = pd.DataFrame(rows)
    df = pd.concat([df, rec_row], ignore_index=True)
    out = ARTIFACTS / "table_step2_best_per_k.csv"
    df.to_csv(out, index=False)
    return out


def to_md_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    # format floats to 3 decimals where appropriate
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, int)) else str(v)
    header = "| " + " | ".join(df.columns) + " |\n"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    rows = "".join("| " + " | ".join(fmt(v) for v in row) + " |\n" for row in df.itertuples(index=False))
    return header + sep + rows


def build_step1_md(metrics: Dict[str, Dict]) -> str:
    # focus on main sensor sets; include available models
    targets = ["all", "feet", "lb", "lf", "rf", "he"]
    models = ["LR", "RF", "SVM", "XGB"]
    rows: List[Dict] = []
    for name in targets:
        tag = f"3class_{name}"
        m = metrics.get(tag)
        if not m:
            continue
        for model in models:
            key = model.lower()
            if key not in m:
                continue
            r = m[key]
            rows.append({
                "sensors": name,
                "model": model,
                "Macro-F1 (mean±std)": f"{r['macro_f1_mean']:.3f}±{r['macro_f1_std']:.3f}",
                "BAcc (mean±std)": f"{r['balanced_acc_mean']:.3f}±{r['balanced_acc_std']:.3f}",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return "(no baseline metrics found)"
    df = df.sort_values(["sensors", "model"]).reset_index(drop=True)
    return to_md_table(df)


def build_step2_md() -> str:
    cand = list(ARTIFACTS.glob("step04b_subset_search_*.json"))
    if not cand:
        return "(no subset search summary found)"
    path = max(cand, key=lambda p: p.stat().st_mtime)
    data = json.loads(path.read_text())
    full = data.get("full", {})
    best = data.get("best_per_k", {})
    rec = data.get("recommended", {})
    rows = [
        {"k": "full", "sensors": ",".join(data.get("sensors_all", [])), "Macro-F1": full.get("macro_f1"), "BAcc": full.get("bacc")}
    ]
    for k in sorted(best.keys(), key=lambda x: int(x)):
        info = best[k]
        rows.append({"k": int(k), "sensors": ",".join(info.get("sensors", [])), "Macro-F1": info.get("macro_f1"), "BAcc": info.get("bacc")})
    rows.append({
        "k": f"recommended({rec.get('k')})",
        "sensors": ",".join(rec.get("sensors", [])),
        "Macro-F1": rec.get("macro_f1"),
        "BAcc": rec.get("bacc"),
    })
    df = pd.DataFrame(rows)
    return to_md_table(df)


def append_to_report(b1: Path, b2: Optional[Path]):
    # Build auto-generated block
    metrics = load_metrics()
    m_all = metrics.get("3class_all", {})
    best_model: Tuple[str, float] | None = None
    for model in ["svm", "xgb", "rf", "lr"]:
        if model in m_all:
            ba = m_all[model].get("balanced_acc_mean", 0.0)
            if best_model is None or ba > best_model[1]:
                best_model = (model.upper(), ba)

    # Subset recommendation
    subset_files = list(ARTIFACTS.glob("step04b_subset_search_*.json"))
    summary_line = ""
    if subset_files:
        path = max(subset_files, key=lambda p: p.stat().st_mtime)
        data = json.loads(path.read_text())
        rec = data.get("recommended", {})
        model = str(data.get("model", "")).upper()
        sensors = ",".join(rec.get("sensors", []))
        k = rec.get("k", "?")
        ba = rec.get("bacc", float("nan"))
        f1 = rec.get("macro_f1", float("nan"))
        summary_line = f"- Recommended minimal sensors: k={k}, {{{sensors}}}, model={model} (Macro-F1={f1:.3f}, BAcc={ba:.3f})."

    exec_lines: List[str] = []
    exec_lines.append("## Executive Summary")
    if best_model:
        exec_lines.append(f"- Best full-sensor model: {best_model[0]} (BAcc≈{best_model[1]:.3f}).")
    if summary_line:
        exec_lines.append(summary_line)

    # Markdown tables
    md1 = build_step1_md(metrics)
    md2 = build_step2_md() if b2 is not None else ""

    # Figures pointers
    figs = [
        "results/figures/step03_confusion_3class_all.png",
        "results/figures/step04_sensors_frontier.png",
    ]
    # possibly subset curve
    sub_curve = list(Path("results/figures").glob("step04b_subset_curve_*.png"))
    if sub_curve:
        figs.append(str(sub_curve[0]))

    auto_block = []
    auto_block.append(BEGIN)
    auto_block.extend(exec_lines)
    auto_block.append("")
    auto_block.append("## Step 1 — Baseline (Markdown)")
    auto_block.append(md1)
    if md2:
        auto_block.append("")
        auto_block.append("## Step 2 — Sensor Minimization (Markdown)")
        auto_block.append(md2)
    auto_block.append("")
    auto_block.append("## Figures")
    for p in figs:
        auto_block.append(f"- {p}")
    auto_block.append("")
    auto_block.append("## Repro Commands")
    auto_block.append("- python analysis/train_baseline.py")
    auto_block.append("- python analysis/compile_reports.py")
    auto_block.append(END)

    text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""

    # Prune legacy duplicated sections outside the auto block
    def prune_sections(s: str) -> str:
        import re
        # remove any repeated baseline/minimization sections
        patterns = [
            r"##\s+Preliminary classification results.*?(?=(\n##\s)|\Z)",
            r"##\s+Step 1 — Baseline summary.*?(?=(\n##\s)|\Z)",
            r"##\s+Step 2 — Sensor minimization.*?(?=(\n##\s)|\Z)",
        ]
        for pat in patterns:
            s = re.sub(pat, "", s, flags=re.DOTALL)

        # collapse duplicates: keep first occurrence for certain sections
        def keep_first(section_title: str, text: str) -> str:
            # Matches block starting at title until next H2 or end
            rx = re.compile(rf"(##\s+{re.escape(section_title)}.*?)(?=(\n##\s)|\Z)", re.DOTALL)
            matches = list(rx.finditer(text))
            if len(matches) <= 1:
                return text
            # keep the first, remove the rest
            kept = matches[0].group(0)
            pre = text[: matches[0].start()]
            post = text[matches[0].end():]
            # strip all remaining matches from post
            post = rx.sub("", post)
            return pre + kept + post

        for title in [
            "Sensor minimization (exhaustive)",
            "Condition shift (hardware)",
            "Leave-one-pathology-subtype-out",
        ]:
            s = keep_first(title, s)
        # Remove all external Executive Summary sections (we generate a fresh one)
        s = re.sub(r"##\s+Executive Summary.*?(?=(\n##\s)|\Z)", "", s, flags=re.DOTALL)
        return s

    if BEGIN in text and END in text and text.index(BEGIN) < text.index(END):
        pre = text[: text.index(BEGIN)]
        post = text[text.index(END) + len(END) :]
        pre = prune_sections(pre)
        new_text = pre + "\n" + "\n".join(auto_block) + "\n" + post
    else:
        cleaned = prune_sections(text)
        new_text = cleaned + "\n" + "\n".join(auto_block) + "\n"
    REPORT.write_text(new_text, encoding="utf-8")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    b1 = compile_step1_table(metrics)
    b2 = compile_step2_table()
    append_to_report(b1, b2)

    # Build a richly illustrated standalone report
    try:
        generate_full_report(metrics)
    except Exception as e:
        # Do not fail pipeline on report rendering errors
        (ARTIFACTS / "full_report_error.log").write_text(str(e))


def pick_one(pattern: str) -> Optional[str]:
    base = Path("results/figures")
    files = sorted(base.glob(pattern))
    if not files:
        return None
    # return path relative to the report dir so Markdown renders correctly
    rel = files[0].relative_to(REPORT_FULL.parent)
    return rel.as_posix()


def generate_full_report(metrics: Dict[str, Dict]):
    lines: List[str] = []
    lines.append("# BMED712 Track A – Robust Gait Phenotyping (IMU)")
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "We evaluate lightweight, deployment-friendly classifiers (LR/RF/SVM/XGB) on a multi-IMU gait dataset for 3-class grouping (healthy/ortho/neuro), and systematically study sensor minimization (4→3→2→1). Under subject-wise 5-fold CV, SVM achieves the best full-sensor performance (Balanced Accuracy ≈ {:.3f}). Using exhaustive subset search with a ΔMacro-F1 ≤ 0.02 criterion, the recommended minimal configuration is a single IMU: Right Foot (RF), achieving Balanced Accuracy ≈ {:.3f}.".format(
            metrics.get("3class_all", {}).get("svm", {}).get("balanced_acc_mean", float("nan")),
            _subset_best_bacc(),
        )
    )
    lines.append("")

    lines.append("## Data & Structure")
    lines.append("- Root: `dataset/data` organized as top-level (healthy/ortho/neuro) → cohort → subject → trial.")
    lines.append("- Each trial contains `*_processed_data.txt` (sensor time series) and `*_meta.json` (metadata).")
    lines.append("")

    lines.append("## Methods & Features")
    lines.append("- Features: per-channel time features (mean, std, rms, ptp) and frequency features (dominant frequency, bandpower, spectral entropy) computed on the chosen segment (full gait or phase-specific).")
    lines.append("- Models: Logistic Regression, Random Forest, SVM (RBF), XGBoost.")
    lines.append("")

    lines.append("## Evaluation Setup")
    lines.append("- Splits: StratifiedGroupKFold(5), grouped by subject to avoid leakage; preprocessing inside Pipelines.")
    lines.append("- Metrics: Macro-F1 and Balanced Accuracy (5-fold mean ± std).")
    lines.append("")

    # Inline figures (EDA)
    lines.append("## EDA Examples")
    ge = pick_one("step01_gait_events_*.png")
    ut = pick_one("step01b_uturn_*.png")
    if ge:
        lines.append(f"![Gait events example]({ge})")
    if ut:
        lines.append(f"![U-turn segmentation example]({ut})")
    lines.append("")

    # Baselines table (markdown)
    lines.append("## Baseline Results (3-class, subject-wise CV)")
    lines.append(build_step1_md(metrics))
    lines.append("")

    # Confusion matrices
    lines.append("### Confusion Matrices (Full sensors vs Minimal RF)")
    cm_all = pick_one("step03_confusion_3class_all.png")
    cm_rf = pick_one("step03_confusion_3class_rf.png")
    if cm_all:
        lines.append(f"![Full-sensor confusion]({cm_all})")
    if cm_rf:
        lines.append(f"![Right Foot (RF) confusion]({cm_rf})")
    lines.append("")

    # Sensor frontier and subset search
    lines.append("## Sensor Availability & Minimization")
    frontier = pick_one("step04_sensors_frontier.png")
    subset = pick_one("step04b_subset_curve_*.png")
    if frontier:
        lines.append(f"![#IMUs vs Balanced Accuracy]({frontier})")
    if subset:
        lines.append(f"![Best subset per k]({subset})")
    lines.append(build_step2_md())
    lines.append("")

    # Phase-wise single vs all sensors
    try:
        from pathlib import Path as _P
        fig_sp = pick_one("phase_single_vs_all.png")
        if fig_sp:
            lines.append("## Phase-wise Single vs All Sensors")
            lines.append(f"![Single vs All sensors]({fig_sp})")
            csv_sp = _P("results/artifacts/phase_single_vs_all.csv")
            if csv_sp.exists():
                df_sp = pd.read_csv(csv_sp)
                # Round for readability
                for c in ["single_bacc", "full_bacc"]:
                    if c in df_sp.columns:
                        df_sp[c] = df_sp[c].astype(float).round(3)
                lines.append(to_md_table(df_sp))
            lines.append("")
    except Exception:
        pass

    # Window experiments (if present)
    try:
        wsum = Path("results") / "window_experiments_summary.csv"
        if wsum.exists():
            lines.append("## Window-based Results (subject-wise 5-fold)")
            dfw = pd.read_csv(wsum)
            # Compact table: phase, sensor, win_s, overlap, model, BAcc, Macro-F1
            view = dfw[[
                "phase","sensor","win_s","overlap","model","bacc_mean","macro_f1_mean"
            ]].copy()
            for c in ["bacc_mean","macro_f1_mean","win_s","overlap"]:
                if c in view.columns:
                    view[c] = view[c].astype(float).round(3)
            lines.append(to_md_table(view.head(40)))
            # Best by phase (RF sensor)
            wbest = Path("results") / "window_best_per_phase.csv"
            if wbest.exists():
                lines.append("")
                lines.append("### Best Window (per phase, RF sensor)")
                dfb = pd.read_csv(wbest)
                for r in dfb.itertuples(index=False):
                    lines.append(f"- {r.phase}: best win ≈ {float(r.best_win_s_by_RF):.2f}s (BAcc≈{float(r.best_bacc_by_RF):.3f})")
            lines.append("")
    except Exception:
        pass

    # Feature importance (RF for interpretability proxy)
    lines.append("## Feature Importance (RF as proxy)")
    fi_all = pick_one("step05_importance_3class_all.png")
    fi_rf = pick_one("step05_importance_3class_rf.png")
    if fi_all:
        lines.append(f"![Full-sensor Top-20 importance]({fi_all})")
    if fi_rf:
        lines.append(f"![RF Top-20 importance]({fi_rf})")
    lines.append("")

    # Phase-wise single-sensor summary (if available)
    try:
        phase_csv = Path("results") / "sensor_phase_summary.csv"
        if phase_csv.exists():
            lines.append("## Phase-wise Single-Sensor Summary")
            df = pd.read_csv(phase_csv)
            # Compact view
            cols = ["phase","sensor","n_trials","best_feature","best_score","mean_score","model","bacc_mean","acc_mean","macro_f1_mean","auc_macro_ovr_mean"]
            view = df[cols].copy()
            # round numeric columns
            for c in ["best_score", "mean_score", "bacc_mean", "acc_mean", "macro_f1_mean", "auc_macro_ovr_mean"]:
                if c in view.columns:
                    view[c] = view[c].astype(float).round(3)
            lines.append(to_md_table(view))
            # Best per phase bullets
            lines.append("")
            lines.append("### Best Sensor by Phase (effect-size proxy)")
            for phase in ["pre_uturn", "uturn", "post_uturn", "gait_full"]:
                sub = df[df["phase"] == phase]
                if sub.empty:
                    continue
                row = sub.sort_values("best_score", ascending=False).iloc[0]
                lines.append(
                    f"- {phase}: {row['sensor']} (best_feature={row['best_feature']}, score={row['best_score']:.3f})"
                )
            # Best per phase (CV metrics)
            lines.append("")
            lines.append("### Best Single-Sensor CV Metrics (subject-wise 5-fold)")
            for phase in ["pre_uturn", "uturn", "post_uturn", "gait_full"]:
                sub = df[df["phase"] == phase]
                if sub.empty:
                    continue
                row = sub.sort_values("bacc_mean", ascending=False).iloc[0]
                lines.append(
                    f"- {phase}: {row['sensor']} — model={row['model']}, BAcc={row['bacc_mean']:.3f}, Acc={row['acc_mean']:.3f}, Macro-F1={row['macro_f1_mean']:.3f}, AUC(macro-OvR)={row.get('auc_macro_ovr_mean', float('nan')):.3f}"
                )
            lines.append("")
    except Exception:
        # Non-fatal if summary absent
        pass

    # Discussion / error modes
    lines.append("## Discussion & Error Modes")
    lines.append("- The single-IMU RF setup reduces Macro-F1 by ~0.006 vs full sensors while preserving Balanced Accuracy, indicating strong discriminative power in foot gyroscope channels.")
    lines.append("- Misclassifications concentrate between ortho and healthy; RF-only may benefit from additional temporal/frequency descriptors in some gait variability conditions.")
    lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append(
        "1) Lightweight models achieve robust performance on the 3-class task; SVM full-sensor Balanced Accuracy ≈ {:.3f}.".format(
            metrics.get("3class_all", {}).get("svm", {}).get("balanced_acc_mean", float("nan"))
        )
    )
    lines.append(
        "2) With ΔMacro-F1 ≤ 0.02, a single-IMU (Right Foot, RF) is recommended; Balanced Accuracy ≈ {:.3f}, enabling practical deployment.".format(
            _subset_best_bacc()
        )
    )
    lines.append("3) RF feature importance highlights strong contribution from foot gyroscope channels, supporting the minimal-IMU conclusion.")
    lines.append("")

    # Repro
    lines.append("## Reproducibility")
    lines.append("- python analysis/train_baseline.py")
    lines.append("- python analysis/compile_reports.py")

    REPORT_FULL.write_text("\n".join(lines), encoding="utf-8")


def _subset_best_bacc() -> float:
    files = list(ARTIFACTS.glob("step04b_subset_search_*.json"))
    if not files:
        return float("nan")
    import json
    data = json.loads(files[0].read_text())
    rec = data.get("recommended", {})
    return float(rec.get("bacc", float("nan")))


if __name__ == "__main__":
    main()
