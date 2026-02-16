import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def split_by_rows(src: Path, out_dir: Path, rows_per_chunk: int, compress: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    for chunk in pd.read_csv(src, chunksize=rows_per_chunk):
        dst = out_dir / f"{src.stem}.part{i:03d}.csv"
        if compress:
            dst = dst.with_suffix(".csv.gz")
            chunk.to_csv(dst, index=False, compression="gzip")
        else:
            chunk.to_csv(dst, index=False)
        i += 1


def split_by_column(src: Path, out_dir: Path, column: str, compress: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    for key, sub in df.groupby(column):
        safe = str(key).replace("/", "-")
        dst = out_dir / f"{src.stem}.{column}_{safe}.csv"
        if compress:
            dst = dst.with_suffix(".csv.gz")
            sub.to_csv(dst, index=False, compression="gzip")
        else:
            sub.to_csv(dst, index=False)


def main():
    ap = argparse.ArgumentParser(description="Split a large CSV into smaller shards")
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--by-rows", type=int, default=None, help="rows per chunk (use this or --by-col)")
    ap.add_argument("--by-col", default=None, help="column to group by (e.g., subject_id)")
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz shards")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out_dir)
    if args.by_rows is None and args.by_col is None:
        ap.error("Specify --by-rows N or --by-col column")
    if args.by_rows is not None and args.by_col is not None:
        ap.error("Use only one of --by-rows or --by-col")

    if args.by_rows is not None:
        split_by_rows(src, out, args.by_rows, compress=args.gzip)
    else:
        split_by_column(src, out, args.by_col, compress=args.gzip)


if __name__ == "__main__":
    main()

