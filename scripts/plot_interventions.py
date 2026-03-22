#!/usr/bin/env python3
"""Plot intervention results: Pareto scatter, heatmap, and summary table.

Usage:
    python scripts/plot_interventions.py [--output figures/interventions.png]

Reads results from logs/sft-fix-*.log and any other intervention logs.
Can also be imported and called with result dicts directly.
"""
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")


def parse_sft_log(filepath):
    """Parse an SFT fix log file for evaluation results.

    Returns dict with metrics or None if no results found.
    """
    results = {}
    pattern = re.compile(
        r"(Correct rate \(GT\)|Hinted rate|Compile rate|"
        r"Hack rate \(strict\)|Hack rate \(loose\)|Avg reward):\s+([\d.]+%?)"
    )
    num_examples_pattern = re.compile(r"Prepared (\d+) SFT training examples")
    steps_pattern = re.compile(r"Training complete after (\d+) steps \((\d+) epochs\)")

    with open(filepath) as f:
        for line in f:
            m = num_examples_pattern.search(line)
            if m:
                results["num_examples"] = int(m.group(1))
            m = steps_pattern.search(line)
            if m:
                results["max_steps"] = int(m.group(1))
                results["epochs"] = int(m.group(2))
            m = pattern.search(line)
            if m:
                key = m.group(1)
                val = m.group(2).rstrip("%")
                val = float(val) / 100.0 if "%" in m.group(2) else float(val)
                key_map = {
                    "Correct rate (GT)": "correct",
                    "Hinted rate": "hinted",
                    "Compile rate": "compile",
                    "Hack rate (strict)": "hack_strict",
                    "Hack rate (loose)": "hack_loose",
                    "Avg reward": "avg_reward",
                }
                results[key_map.get(key, key)] = val

    if "correct" not in results:
        return None
    return results


def load_all_sft_results(version=None):
    """Load SFT fix results from logs directory.

    Args:
        version: If set (e.g. "v3"), only load logs matching that suffix.
            None loads all logs.
    """
    results = []
    for fname in sorted(os.listdir(LOG_DIR)):
        if not (fname.startswith("sft-fix-") and fname.endswith(".log")):
            continue
        if version:
            # Match e.g. "sft-fix-10ex-v3.log" but not "sft-fix-10ex.log"
            if f"-{version}.log" not in fname:
                continue
        path = os.path.join(LOG_DIR, fname)
        r = parse_sft_log(path)
        if r:
            r["log_file"] = fname
            results.append(r)
    return results


# ── Reference baselines (ariaw + our runs) ──────────────────────────────
BASELINES = {
    "Base Model": {"correct": 0.08, "hack_strict": 0.00, "marker": "s", "color": "#888888"},
    "RL Baseline\n(no loophole)": {"correct": 0.21, "hack_strict": 0.00, "marker": "D", "color": "#228833"},
    "No Intervention\n(kl=0)": {"correct": 0.00, "hack_strict": 0.88, "marker": "X", "color": "#EE6677"},
    "No Intervention\n(kl=1e-3)": {"correct": 0.30, "hack_strict": 0.00, "marker": "^", "color": "#4477AA"},
}


def plot_pareto(results, baselines=None, output_path=None):
    """Scatter plot: correctness (x) vs hack rate (y).

    Bottom-right is ideal (high performance, low hacking).
    """
    if baselines is None:
        baselines = BASELINES

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Plot baselines
    for name, b in baselines.items():
        ax.scatter(b["correct"], b["hack_strict"],
                   marker=b["marker"], s=80, c=b["color"],
                   edgecolors="black", linewidths=0.5,
                   zorder=5, label=name)

    # Plot SFT results
    if results:
        xs = [r["correct"] for r in results]
        ys = [r["hack_strict"] for r in results]
        labels = [f"{r.get('num_examples', '?')}ex" for r in results]

        ax.scatter(xs, ys, marker="o", s=90, c="#AA3377",
                   edgecolors="black", linewidths=0.5,
                   zorder=6, label="SFT Fix")

        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7,
                        color="#AA3377", fontweight="bold")

    # Ideal region
    ax.axhspan(-0.05, 0.02, color="#e8f5e9", alpha=0.4, zorder=0)
    ax.axvspan(0.15, 1.0, color="#e8f5e9", alpha=0.3, zorder=0)
    ax.text(0.55, 0.01, "ideal region", fontsize=7, color="#2e7d32",
            alpha=0.6, style="italic")

    ax.set_xlabel("Correctness Rate")
    ax.set_ylabel("Reward Hacking Rate (strict)")
    ax.set_title("Intervention Tradeoff: Performance vs Reward Hacking")
    ax.set_xlim(-0.03, max(0.5, max([b["correct"] for b in baselines.values()] + [r.get("correct", 0) for r in results]) + 0.05))
    ax.set_ylim(-0.03, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.legend(loc="upper right", frameon=True, edgecolor="#cccccc",
              fancybox=False, fontsize=7)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {output_path}")
        if output_path.endswith(".pdf"):
            png = output_path.replace(".pdf", ".png")
            fig.savefig(png, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved to {png}")

    return fig


def plot_heatmap(results, param_key="num_examples", output_path=None):
    """Heatmap of hack rate and correctness across a sweep parameter.

    Works best when results vary along one or two dimensions.
    """
    if not results:
        print("No results to plot.")
        return None

    # Sort by parameter
    results = sorted(results, key=lambda r: r.get(param_key, 0))
    labels = [str(r.get(param_key, "?")) for r in results]
    correct_vals = [r.get("correct", 0) for r in results]
    hack_vals = [r.get("hack_strict", 0) for r in results]
    compile_vals = [r.get("compile", 0) for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.0))

    metrics = [
        ("Hack Rate (strict)", hack_vals, "Reds"),
        ("Correctness", correct_vals, "Greens"),
        ("Compile Rate", compile_vals, "Blues"),
    ]

    for ax, (title, vals, cmap) in zip(axes, metrics):
        data = np.array(vals).reshape(1, -1)
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel(f"# Golden Examples")
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)

        # Annotate cells
        for j, v in enumerate(vals):
            color = "white" if v > 0.5 else "black"
            ax.text(j, 0, f"{v:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {output_path}")

    return fig


def plot_bar_comparison(results, output_path=None):
    """Grouped bar chart comparing hack rate and performance across runs.

    Similar to ariaw's Figure 5.
    """
    if not results:
        return None

    results = sorted(results, key=lambda r: r.get("num_examples", 0))
    labels = [f"SFT {r.get('num_examples', '?')}ex" for r in results]

    # Add baselines
    all_labels = ["Base\nModel", "RL\nBaseline", "No Intv\n(kl=0)"] + labels
    hack_vals = [0.0, 0.0, 0.88] + [r.get("hack_strict", 0) for r in results]
    correct_vals = [0.08, 0.21, 0.0] + [r.get("correct", 0) for r in results]
    compile_vals = [None, None, None] + [r.get("compile", 0) for r in results]

    x = np.arange(len(all_labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7.5, 3.5))

    bars1 = ax.bar(x - width, hack_vals, width, label="Hack Rate (strict)",
                   color="#EF5350", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, correct_vals, width, label="Correctness",
                   color="#66BB6A", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars3_vals = [v if v is not None else 0 for v in compile_vals]
    bars3 = ax.bar(x + width, bars3_vals, width, label="Compile Rate",
                   color="#42A5F5", alpha=0.85, edgecolor="white", linewidth=0.5)
    # Hide baseline compile bars (we don't have those)
    for i in range(3):
        bars3[i].set_alpha(0)

    # Value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=6.5)
    for i, bar in enumerate(bars3):
        if i >= 3 and compile_vals[i] is not None and compile_vals[i] > 0.02:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("SFT Intervention Results vs Baselines")
    ax.legend(loc="upper right", frameon=True, edgecolor="#cccccc",
              fancybox=False)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {output_path}")

    return fig


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--version", default="v3", help="Log version suffix (default: v3)")
    _args = _parser.parse_args()

    results = load_all_sft_results(version=_args.version)
    print(f"Loaded {len(results)} SFT results (version={_args.version})")
    for r in results:
        print(f"  {r.get('log_file', '?')}: {r.get('num_examples', '?')}ex, "
              f"correct={r.get('correct', 0):.1%}, hack={r.get('hack_strict', 0):.1%}, "
              f"compile={r.get('compile', 0):.1%}")

    if not results:
        print("No results found in logs/")
        sys.exit(1)

    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Pareto scatter
    plot_pareto(results,
                output_path=os.path.join(FIG_DIR, "intervention_pareto.pdf"))

    # 2. Heatmap
    plot_heatmap(results,
                 output_path=os.path.join(FIG_DIR, "sft_heatmap.png"))

    # 3. Grouped bar chart
    plot_bar_comparison(results,
                        output_path=os.path.join(FIG_DIR, "sft_bar_comparison.png"))

    print("\nDone.")
