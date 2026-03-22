#!/usr/bin/env python3
"""Plot stacked area chart of reward hacking categories over training steps.

Reproduces the ariaw/rl-rewardhacking Figure 1 style: a stacked area chart
showing the proportion of rollouts in each category (Correct, Correct+Attempted,
Reward Hack, Attempted Hack, Incorrect) summing to 100%.

Supports two data formats:
  1. New format with cats= field: cats=C/CA/RH/A/I
  2. Legacy format: approximates categories from correct/hack_strict/hack_loose
"""
import re
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# Category colors matching ariaw's plot
COLORS = {
    "Correct": "#66BB6A",
    "Correct; Attempted Reward Hack": "#FDD835",
    "Reward Hack": "#EF5350",
    "Attempted Reward Hack": "#AB47BC",
    "Incorrect": "#BDBDBD",
}


def parse_log(filename):
    """Parse training log and extract per-step 5-way categories.

    Returns dict with steps and 5 category arrays summing to 1.0 per step.
    """
    path = os.path.join(LOG_DIR, filename) if not os.path.isabs(filename) else filename
    steps = []
    cats = {"correct": [], "correct_attempted": [], "reward_hack": [],
            "attempted": [], "incorrect": []}

    # New format: cats=C/CA/RH/A/I
    cat_pattern = re.compile(
        r"Step (\d+)/\d+ \|.*?"
        r"cats=([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)"
    )
    # Legacy format: correct, hack_strict, hack_loose
    legacy_pattern = re.compile(
        r"Step (\d+)/\d+ \|"
        r" reward=[\d.+-e]+ \|"
        r" correct=([\d.]+) \|"
        r" hack_strict=([\d.]+) \|"
        r" hack_loose=([\d.]+)"
    )

    seen = set()
    with open(path) as f:
        for line in f:
            if "grpo_trainer - INFO - Step" not in line:
                continue

            # Try new format first
            m = cat_pattern.search(line)
            if m:
                step = int(m.group(1))
                if step in seen:
                    continue
                seen.add(step)
                steps.append(step)
                cats["correct"].append(float(m.group(2)))
                cats["correct_attempted"].append(float(m.group(3)))
                cats["reward_hack"].append(float(m.group(4)))
                cats["attempted"].append(float(m.group(5)))
                cats["incorrect"].append(float(m.group(6)))
                continue

            # Fall back to legacy format
            m = legacy_pattern.search(line)
            if m:
                step = int(m.group(1))
                if step in seen:
                    continue
                seen.add(step)
                steps.append(step)

                correct = float(m.group(2))
                hack_strict = float(m.group(3))
                hack_loose = float(m.group(4))

                # Approximate 5-way from 3 metrics:
                # hack_strict = Reward Hack (not correct, test passes)
                # hack_loose includes: Reward Hack + Correct;Attempted + Attempted
                # Overlap (correct AND hack_loose) is unknown; approximate:
                #   When hack_loose > hack_strict, excess = attempted + correct_attempted
                #   Assume correct_attempted ≈ min(correct, excess) * 0.5 heuristic
                excess = max(0, hack_loose - hack_strict)
                # Rough split: if correct is low, most excess is attempted
                correct_attempted = min(correct, excess) * min(1.0, correct / max(correct + 0.01, 1.0))
                attempted = max(0, excess - correct_attempted)
                correct_only = max(0, correct - correct_attempted)
                incorrect = max(0, 1.0 - correct_only - correct_attempted - hack_strict - attempted)

                cats["correct"].append(correct_only)
                cats["correct_attempted"].append(correct_attempted)
                cats["reward_hack"].append(hack_strict)
                cats["attempted"].append(attempted)
                cats["incorrect"].append(incorrect)

    return {
        "steps": np.array(steps),
        "correct": np.array(cats["correct"]),
        "correct_attempted": np.array(cats["correct_attempted"]),
        "reward_hack": np.array(cats["reward_hack"]),
        "attempted": np.array(cats["attempted"]),
        "incorrect": np.array(cats["incorrect"]),
    }


def smooth(y, window=5):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_stacked_area(data, title="Reward Hacking Rollouts by Training Step",
                      window=5, output_path=None):
    """Create stacked area chart from parsed data."""
    fig, ax = plt.subplots(figsize=(10, 4))

    steps = data["steps"]
    # Stack order (bottom to top): Correct, Correct+Attempted, Reward Hack, Attempted, Incorrect
    categories = [
        ("Correct", smooth(data["correct"], window)),
        ("Correct; Attempted Reward Hack", smooth(data["correct_attempted"], window)),
        ("Reward Hack", smooth(data["reward_hack"], window)),
        ("Attempted Reward Hack", smooth(data["attempted"], window)),
        ("Incorrect", smooth(data["incorrect"], window)),
    ]

    # Normalize to sum to 1.0 after smoothing
    total = sum(c[1] for c in categories)
    total = np.maximum(total, 1e-8)
    categories = [(name, vals / total) for name, vals in categories]

    # Build stacked arrays
    y_stack = np.row_stack([c[1] for c in categories])
    colors = [COLORS[c[0]] for c in categories]
    labels = [c[0] for c in categories]

    ax.stackplot(steps, y_stack, labels=labels, colors=colors, alpha=0.85)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Proportion of Rollouts")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(steps[0], steps[-1])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Legend below
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], leg_labels[::-1], loc="lower center",
              bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=True,
              edgecolor="#cccccc", fancybox=False)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved to {output_path}")
        # Also save PNG if PDF
        if output_path.endswith(".pdf"):
            png_path = output_path.replace(".pdf", ".png")
            fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved to {png_path}")

    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_stacked_area.py <log_file> [title] [output_path]")
        print("  log_file: path to training log (relative to logs/ or absolute)")
        print("  title: plot title (optional)")
        print("  output_path: output file path (optional, default: figures/<logname>.png)")
        sys.exit(1)

    log_file = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else f"Reward Hacking: {os.path.basename(log_file)}"

    if len(sys.argv) > 3:
        out_path = sys.argv[3]
    else:
        base = os.path.splitext(os.path.basename(log_file))[0]
        out_path = os.path.join(LOG_DIR, "..", "figures", f"{base}_stacked.png")

    data = parse_log(log_file)
    print(f"Parsed {len(data['steps'])} steps from {log_file}")

    if len(data["steps"]) == 0:
        print("No data found!")
        sys.exit(1)

    plot_stacked_area(data, title=title, output_path=out_path)
