#!/usr/bin/env python3
"""Training dashboard — live-updating plots from CSV logs.

Usage:
    python tools/dashboard.py                          # auto-detect latest logs
    python tools/dashboard.py checkpoints/pretrain/logs/metrics.csv
    python tools/dashboard.py --watch                  # refresh every 30s
    python tools/dashboard.py --export training.png    # save to file instead
"""

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import matplotlib
except ImportError:
    print("matplotlib required: pip install matplotlib")
    sys.exit(1)


def read_metrics(csv_path: Path) -> dict[str, list]:
    """Read metrics CSV into column lists."""
    data: dict[str, list] = {"step": [], "loss": [], "lr": [], "tokens_per_sec": [], "elapsed_sec": [], "token_accuracy": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["loss"].append(float(row["loss"]))
            data["lr"].append(float(row["lr"]))
            data["tokens_per_sec"].append(float(row["tokens_per_sec"]))
            data["elapsed_sec"].append(float(row["elapsed_sec"]))
            if "token_accuracy" in row and row["token_accuracy"]:
                data["token_accuracy"].append(float(row["token_accuracy"]))
    return data


def find_latest_csv() -> Path | None:
    """Find the most recent metrics.csv in checkpoints/."""
    candidates = list(Path("checkpoints").rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_dashboard(data: dict[str, list], title: str = "NeuralSpell Training", export: str | None = None):
    """Create a 2x2 dashboard of training metrics."""
    if export:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    steps = data["step"]

    # Loss
    ax = axes[0, 0]
    ax.plot(steps, data["loss"], color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    if len(data["loss"]) > 0:
        ax.annotate(f'{data["loss"][-1]:.4f}', xy=(steps[-1], data["loss"][-1]),
                     fontsize=10, color="#2196F3", fontweight="bold")

    # Learning Rate
    ax = axes[0, 1]
    ax.plot(steps, data["lr"], color="#FF9800", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Tokens/sec
    ax = axes[1, 0]
    ax.plot(steps, data["tokens_per_sec"], color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)
    if len(data["tokens_per_sec"]) > 0:
        avg_tps = sum(data["tokens_per_sec"]) / len(data["tokens_per_sec"])
        ax.axhline(y=avg_tps, color="#4CAF50", linestyle="--", alpha=0.5, label=f"avg: {avg_tps:,.0f}")
        ax.legend()

    # Bottom-right: Token Accuracy (Phase 2) or Loss log scale (Phase 1)
    ax = axes[1, 1]
    if data["token_accuracy"]:
        ax.plot(steps[:len(data["token_accuracy"])], data["token_accuracy"], color="#E91E63", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Token Accuracy")
        ax.set_title("Token-level Accuracy")
        ax.grid(True, alpha=0.3)
        if data["token_accuracy"]:
            ax.annotate(f'{data["token_accuracy"][-1]:.3f}', xy=(steps[len(data["token_accuracy"])-1], data["token_accuracy"][-1]),
                         fontsize=10, color="#E91E63", fontweight="bold")
    else:
        ax.plot(steps, data["loss"], color="#9C27B0", linewidth=1.5)
        ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Loss (log scale)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    # Show ETA
    if len(steps) >= 2 and data["elapsed_sec"][-1] > 0:
        secs_per_step = data["elapsed_sec"][-1] / steps[-1]
        remaining_steps = 500000 - steps[-1]
        eta_hours = (remaining_steps * secs_per_step) / 3600
        ax.set_title(f"Loss (log) — ETA: {eta_hours:.1f}h remaining")

    plt.tight_layout()

    if export:
        plt.savefig(export, dpi=150, bbox_inches="tight")
        print(f"Saved to {export}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Training dashboard")
    parser.add_argument("csv_path", nargs="?", type=Path, help="Path to metrics.csv")
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds")
    parser.add_argument("--export", type=str, default=None, help="Export to image file instead of showing")
    args = parser.parse_args()

    csv_path = args.csv_path
    if csv_path is None:
        csv_path = find_latest_csv()
        if csv_path is None:
            print("No metrics.csv found. Start training first.")
            sys.exit(1)
        print(f"Using: {csv_path}")

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if args.watch and not args.export:
        import matplotlib.pyplot as plt
        plt.ion()
        print(f"Watching {csv_path} (refresh every {args.interval}s, Ctrl+C to stop)")
        try:
            while True:
                data = read_metrics(csv_path)
                if data["step"]:
                    plt.close("all")
                    plot_dashboard(data)
                    plt.pause(0.1)
                    print(f"  Step {data['step'][-1]} | Loss {data['loss'][-1]:.4f} | {time.strftime('%H:%M:%S')}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        data = read_metrics(csv_path)
        if not data["step"]:
            print("No data yet.")
            sys.exit(1)
        plot_dashboard(data, export=args.export)
        print(f"\nLatest: Step {data['step'][-1]} | Loss {data['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
