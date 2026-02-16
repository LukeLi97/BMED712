import argparse
import os
import json
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd

from .features import compute_trial_features, assemble_row


def iter_trial_dirs(data_root: str) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (group, cohort, subject, trial) tuples under data_root.

    Expects dataset layout: data/<group>/<cohort>/<subject>/<trial>/
    """
    for group in ("healthy", "ortho", "neuro"):
        gpath = os.path.join(data_root, group)
        if not os.path.isdir(gpath):
            continue
        for cohort in sorted(os.listdir(gpath)):
            cpath = os.path.join(gpath, cohort)
            if not os.path.isdir(cpath):
                continue
            for subject in sorted(os.listdir(cpath)):
                spath = os.path.join(cpath, subject)
                if not os.path.isdir(spath):
                    continue
                for trial in sorted(os.listdir(spath)):
                    tpath = os.path.join(spath, trial)
                    if os.path.isdir(tpath):
                        yield group, cohort, subject, trial


def load_processed_and_meta(trial_path: str) -> Tuple[pd.DataFrame, Dict]:
    processed_file = None
    meta_file = None
    for f in os.listdir(trial_path):
        if f.endswith("_processed_data.txt"):
            processed_file = os.path.join(trial_path, f)
        elif f.endswith("_meta.json"):
            meta_file = os.path.join(trial_path, f)
    if processed_file is None or meta_file is None:
        raise FileNotFoundError(f"Missing processed or meta in {trial_path}")

    df = pd.read_csv(processed_file, sep="\t")
    with open(meta_file, "r") as fh:
        meta = json.load(fh)
    return df, meta


def to_label(group: str) -> str:
    mapping = {"healthy": "Healthy", "ortho": "Ortho", "neuro": "Neuro"}
    return mapping.get(group, group.title())


def build_expected_columns(template_csv: Optional[str]) -> List[str]:
    if template_csv and os.path.exists(template_csv):
        cols = list(pd.read_csv(template_csv, nrows=1).columns)
        return cols
    # fallback: generate canonical ordering
    sensors = ("HE", "LB", "LF", "RF")
    kinds = ("FreeAcc", "Gyr")
    axes = ("X", "Y", "Z")
    stats = ("mean", "std", "rms", "ptp", "domfreq", "bandpower", "entropy")
    cols: List[str] = []
    for s in sensors:
        for k in kinds:
            for a in axes:
                prefix = f"{s}_{k}_{a}"
                cols.extend([f"{prefix}_{st}" for st in stats])
    cols.extend(["trial_id", "subject_id", "label"])
    return cols


def export_features(
    data_root: str,
    segment: str,
    out_csv: str,
    template_csv: Optional[str],
    include_groups: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    groups_filter = set(g.lower() for g in include_groups) if include_groups else None
    for group, cohort, subject, trial in iter_trial_dirs(data_root):
        if groups_filter and group.lower() not in groups_filter:
            continue
        trial_path = os.path.join(data_root, group, cohort, subject, trial)
        try:
            df, meta = load_processed_and_meta(trial_path)
        except Exception as e:
            # Skip malformed trials but continue
            continue
        trial_id = trial
        subject_id = subject
        label = to_label(group)
        feats = compute_trial_features(df, meta, segment_mode=segment)
        row = assemble_row(trial_id, subject_id, label, feats)
        rows.append(row)

    if not rows:
        raise RuntimeError("No trials processed; check data path.")

    # Build DataFrame and align column order
    df_out = pd.DataFrame(rows)
    expected_cols = build_expected_columns(template_csv)
    # Add any missing columns with NaN and order
    for col in expected_cols:
        if col not in df_out.columns:
            df_out[col] = pd.NA
    df_out = df_out[expected_cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-trial features (one row per trial)."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join("dataset", "data"),
        help="Path to dataset/data root",
    )
    parser.add_argument(
        "--segment",
        choices=["gait", "full", "pre_uturn", "uturn", "post_uturn"],
        default="gait",
        help="Segment definition for feature aggregation",
    )
    parser.add_argument(
        "--template-csv",
        default="master_features.csv",
        help="Optional CSV whose columns define output order",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("results", "features_gait.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional subset of groups (healthy ortho neuro)",
    )
    args = parser.parse_args()

    df = export_features(
        data_root=args.data_root,
        segment=args.segment,
        out_csv=args.out,
        template_csv=args.template_csv,
        include_groups=args.groups,
    )
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
