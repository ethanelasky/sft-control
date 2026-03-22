#!/usr/bin/env python3
"""Plot GRPO training runs in ICLR style for LessWrong post.

Produces a figure with correctness rate and hack rate across training steps
for multiple configurations, showing how sensitive reward hacking emergence
is to hyperparameter choices.
"""
import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── ICLR-style formatting ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "text.usetex": False,
})

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

# ── Parse logs ──────────────────────────────────────────────────────────
def parse_log(filename, before_timestamp=None):
    """Extract per-step metrics from a training log file.

    Args:
        filename: Log file name relative to LOG_DIR.
        before_timestamp: If set, only include entries before this timestamp
            string (e.g. "2026-03-22 12:40") to isolate a specific run when
            multiple runs are appended to the same log file.
    """
    path = os.path.join(LOG_DIR, filename)
    steps, correct, hack_strict, hack_loose, reward, kl, compile_rate = [], [], [], [], [], [], []
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
            # Filter by timestamp if specified
            if before_timestamp and line[:len(before_timestamp)] >= before_timestamp:
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
                hack_loose.append(float(m.group(5)))
                compile_rate.append(float(m.group(6)))
                kl.append(float(m.group(7)))
    return {
        "steps": np.array(steps),
        "correct": np.array(correct),
        "hack_strict": np.array(hack_strict),
        "hack_loose": np.array(hack_loose),
        "reward": np.array(reward),
        "kl": np.array(kl),
        "compile": np.array(compile_rate),
    }

def smooth(y, window=5):
    """Simple moving average for smoother curves."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

# ── Define runs ─────────────────────────────────────────────────────────
runs = [
    {
        "file": "grpo-8b-ppo-v3.log",
        "label": "Qwen3-8B-Base, PPO+KL-in-reward, KL=1e-3",
        "color": "#4477AA",
        "linestyle": "-",
    },
    {
        "file": "grpo-8b-instruct-ppo-kl-v2.log",
        "label": "Qwen3-8B, PPO+KL-in-loss, KL=1e-3",
        "color": "#228833",
        "linestyle": "-",
    },
    {
        "file": "grpo-gpt-oss-20b-ppo-kl.log",
        "label": "gpt-oss-20b, PPO+KL-in-loss, KL=1e-3",
        "color": "#CCBB44",
        "linestyle": "-",
        "before_timestamp": "2026-03-22 12:40",  # First run only; second run appended at 12:40
    },
    {
        "file": "grpo-gpt-oss-20b-no-kl-full.log",
        "label": "gpt-oss-20b, PPO+KL-in-loss, KL=0",
        "color": "#EE6677",
        "linestyle": "-",
    },
]

# ── Parse all runs ──────────────────────────────────────────────────────
data = {}
for run in runs:
    try:
        data[run["file"]] = parse_log(run["file"], before_timestamp=run.get("before_timestamp"))
        print(f"Parsed {run['file']}: {len(data[run['file']]['steps'])} steps")
    except Exception as e:
        print(f"Failed to parse {run['file']}: {e}")

# ── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.4), sharex=True)
ax_correct, ax_hack, ax_kl, ax_reward = axes

window = 7  # smoothing window

for run in runs:
    if run["file"] not in data:
        continue
    d = data[run["file"]]

    ax_correct.plot(
        d["steps"], smooth(d["correct"], window),
        label=run["label"], color=run["color"],
        linestyle=run["linestyle"], alpha=0.9,
    )
    ax_hack.plot(
        d["steps"], smooth(d["hack_strict"], window),
        label=run["label"], color=run["color"],
        linestyle=run["linestyle"], alpha=0.9,
    )
    ax_kl.plot(
        d["steps"], smooth(d["kl"], window),
        label=run["label"], color=run["color"],
        linestyle=run["linestyle"], alpha=0.9,
    )
    ax_reward.plot(
        d["steps"], smooth(d["reward"], window),
        label=run["label"], color=run["color"],
        linestyle=run["linestyle"], alpha=0.9,
    )

ax_correct.set_ylabel("Correctness Rate")
ax_correct.set_xlabel("Training Step")
ax_correct.set_ylim(-0.02, 0.85)
ax_correct.set_title("Correctness Rate")

ax_hack.set_ylabel("Hack Rate (strict)")
ax_hack.set_xlabel("Training Step")
ax_hack.set_ylim(-0.02, 1.05)
ax_hack.set_title("Reward Hacking Rate")

ax_kl.set_ylabel("KL Divergence")
ax_kl.set_xlabel("Training Step")
ax_kl.set_title("KL from Reference")

ax_reward.set_ylabel("Reward")
ax_reward.set_xlabel("Training Step")
ax_reward.set_title("Average Reward")

# Single legend below all subplots
handles, labels = ax_correct.get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center",
    ncol=2, bbox_to_anchor=(0.5, -0.22),
    frameon=True, edgecolor="#cccccc", fancybox=False,
)

plt.tight_layout()
out_path = os.path.join(LOG_DIR, "..", "figures", "training_runs.pdf")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
print(f"\nSaved to {out_path}")

# Also save PNG for quick preview
png_path = out_path.replace(".pdf", ".png")
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
print(f"Saved to {png_path}")
