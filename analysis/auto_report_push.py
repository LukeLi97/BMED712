import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results" / "artifacts"
BRANCH = os.environ.get("BRANCH", "codex/mamba-xgb-windowing")


def changed(snap_a: dict[str, float], snap_b: dict[str, float]) -> bool:
    if snap_a.keys() != snap_b.keys():
        return True
    for k in snap_a:
        if abs(snap_a[k] - snap_b[k]) > 1e-6:
            return True
    return False


def snapshot() -> dict[str, float]:
    out: dict[str, float] = {}
    if not ART.exists():
        return out
    for p in ART.glob("metrics_*.json"):
        out[p.name] = p.stat().st_mtime
    # also track window_freq_eval.csv
    p2 = ROOT / "results" / "window_freq_eval.csv"
    if p2.exists():
        out[p2.name] = p2.stat().st_mtime
    return out


def run(cmd: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def main():
    prev = snapshot()
    while True:
        time.sleep(60)
        cur = snapshot()
        if not prev:
            prev = cur
            continue
        if changed(prev, cur):
            # Rebuild reports
            run([sys.executable, str(ROOT / "analysis" / "compile_reports.py")])
            # Git commit & push
            run(["git", "add", "results/report_full.md", "results/window_report_freq.md", "results/window_experiments_summary.csv", "results/window_freq_eval.csv", "results/artifacts/*"])  # noqa: E501
            run(["git", "commit", "-m", "chore: auto-compile reports after new metrics"])  # ignore failure
            run(["git", "push", "-u", "origin", BRANCH])
            prev = cur


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

