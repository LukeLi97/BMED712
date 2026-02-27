import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_one(phase: str, win: float, overlap: float, device: str, epochs: int, batch: int, arch: str, d_model: int, n_layers: int) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [
        sys.executable, str(REPO_ROOT / "analysis" / "train_mamba_windows.py"),
        "--phase", phase,
        "--win", str(win),
        "--overlap", str(overlap),
        "--device", device,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--arch", arch,
        "--d-model", str(d_model),
        "--n-layers", str(n_layers),
        "--out", str(REPO_ROOT / "results" / "artifacts"),
    ]
    return subprocess.run(cmd, env=env).returncode


def append_log(msg: str) -> None:
    p = REPO_ROOT / "results" / "EXPERIMENT_LOG_2026-02-23.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser(description="Queue Mamba-like runs across phases after an optional wait PID")
    ap.add_argument("--wait-pid", type=int, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--arch", default="mamba")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    args = ap.parse_args()

    if args.wait_pid is not None:
        append_log(f"- waiting for PID {args.wait_pid} to finish before queued runs …")
        while pid_alive(args.wait_pid):
            time.sleep(10)

    schedule = [
        ("pre_uturn", 4.0, 0.25),
        ("post_uturn", 4.0, 0.25),
        ("uturn", 6.0, 0.50),
    ]

    for phase, win, ov in schedule:
        append_log(f"- start queued run: {phase} — win={win}s ov={int(ov*100)}% arch={args.arch}")
        rc = run_one(phase, win, ov, args.device, args.epochs, args.batch, args.arch, args.d_model, args.n_layers)
        if rc != 0 and phase == "uturn" and win == 6.0:
            # fallback to 5.0 s if 6.0 fails due to insufficient windows
            append_log(f"  fallback: {phase} 5.0 s @ 50% (previous rc={rc})")
            rc = run_one(phase, 5.0, 0.50, args.device, args.epochs, args.batch, args.arch, args.d_model, args.n_layers)
        append_log(f"  finished {phase} rc={rc}")


if __name__ == "__main__":
    main()

