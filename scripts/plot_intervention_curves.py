#!/usr/bin/env python3
"""Plot intervention training curves: correctness and hack rate over steps.

Parses RL fix logs and plots how metrics evolve during intervention training,
showing whether the intervention is successfully reducing hacking and
recovering correctness.
"""
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
    "text.usetex": False,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")


def parse_log(filename):
    """Extract per-step metrics from a training log file."""
    path = os.path.join(LOG_DIR, filename)
    steps, correct, hack_strict, reward, kl, compile_rate = [], [], [], [], [], []
    pattern = re.compile(
        r"Step (\d+)/\d+ \|"
        r" reward=([\d.+-e]+) \|"
        r" correct=([\d.]+) \|"
        r" hack_strict=([\d.]+) \|"
        r" hack_loose=([\d.]+) \|"
        r" compile=([\d.]+) \|"
        r".*?kl=([\d.]+)"
    )
    seen = set()
    with open(path) as f:
        for line in f:
            if "grpo_trainer - INFO - Step" not in line:
                continue
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                if step in seen:
                    continue
                seen.add(step)
                steps.append(step)
                reward.append(float(m.group(2)))
                correct.append(float(m.group(3)))
                hack_strict.append(float(m.group(4)))
                compile_rate.append(float(m.group(6)))
                kl.append(float(m.group(7)))
    return {
        "steps": np.array(steps),
        "correct": np.array(correct),
        "hack_strict": np.array(hack_strict),
        "reward": np.array(reward),
        "kl": np.array(kl),
        "compile": np.array(compile_rate),
    }


def smooth(y, window=5):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    # Pad edges to avoid boundary artifacts
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def plot_intervention_curves(runs, output_path=None, title_suffix=""):
    """Plot correctness and hack rate over training steps for intervention runs.

    Args:
        runs: List of dicts with keys: file, label, color, linestyle (optional).
        output_path: Save path (PDF). PNG also saved alongside.
        title_suffix: Appended to figure suptitle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    ax_correct, ax_hack, ax_reward = axes
    window = 5

    for run in runs:
        try:
            d = parse_log(run["file"])
            print(f"  Parsed {run['file']}: {len(d['steps'])} steps")
        except Exception as e:
            print(f"  Failed to parse {run['file']}: {e}")
            continue

        ls = run.get("linestyle", "-")
        ax_correct.plot(d["steps"], smooth(d["correct"], window),
                        label=run["label"], color=run["color"], linestyle=ls, alpha=0.9)
        ax_correct.fill_between(d["steps"], 0, smooth(d["correct"], window),
                                color=run["color"], alpha=0.07)

        ax_hack.plot(d["steps"], smooth(d["hack_strict"], window),
                     label=run["label"], color=run["color"], linestyle=ls, alpha=0.9)
        ax_hack.fill_between(d["steps"], 0, smooth(d["hack_strict"], window),
                             color=run["color"], alpha=0.07)

        ax_reward.plot(d["steps"], smooth(d["reward"], window),
                       label=run["label"], color=run["color"], linestyle=ls, alpha=0.9)

    # Formatting
    ax_correct.set_ylabel("Rate")
    ax_correct.set_xlabel("Training Step")
    ax_correct.set_ylim(-0.02, 0.85)
    ax_correct.set_title("Correctness Rate (GT)")
    ax_correct.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax_hack.set_xlabel("Training Step")
    ax_hack.set_ylim(-0.02, 1.05)
    ax_hack.set_title("Reward Hacking Rate (strict)")
    ax_hack.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    # Reference line: starting hack rate
    ax_hack.axhline(y=0.82, color="#999999", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_hack.text(2, 0.84, "pre-intervention", fontsize=6.5, color="#999999", style="italic")

    ax_reward.set_ylabel("Reward")
    ax_reward.set_xlabel("Training Step")
    ax_reward.set_title("Average Reward")

    # Single legend
    handles, labels = ax_correct.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(runs),
               bbox_to_anchor=(0.5, -0.08), frameon=True, edgecolor="#cccccc",
               fancybox=False)

    suptitle = "Post-Hoc Intervention Training Curves"
    if title_suffix:
        suptitle += f" ({title_suffix})"
    fig.suptitle(suptitle, fontsize=11, y=1.02)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {output_path}")
        png = output_path.replace(".pdf", ".png")
        fig.savefig(png, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {png}")

    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden_log", default="rl-fix-golden-v3-final.log")
    parser.add_argument("--trusted_log", default="rl-fix-trusted-v3-final.log")
    parser.add_argument("--output", default=os.path.join(FIG_DIR, "intervention_curves.pdf"))
    parser.add_argument("--suffix", default="gpt-oss-20b, lr=3e-5, KL=5e-3")
    args = parser.parse_args()

    runs = [
        {
            "file": args.golden_log,
            "label": "RL-Golden (ground truth reward)",
            "color": "#CCBB44",
        },
        {
            "file": args.trusted_log,
            "label": "RL-Trusted (Qwen3-1.7b judge)",
            "color": "#4477AA",
        },
    ]

    print("Plotting intervention training curves...")
    plot_intervention_curves(runs, output_path=args.output, title_suffix=args.suffix)
    print("Done.")
