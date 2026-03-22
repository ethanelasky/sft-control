#!/usr/bin/env python3
"""List registered checkpoints.

Usage:
    python scripts/list_checkpoints.py              # list all
    python scripts/list_checkpoints.py --hacked     # only hacked models
    python scripts/list_checkpoints.py --name foo   # search by name
"""
import argparse
import json
import os

REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "registry.json")


def main():
    parser = argparse.ArgumentParser(description="List registered checkpoints")
    parser.add_argument("--hacked", action="store_true", help="Only show hacked models")
    parser.add_argument("--name", type=str, default=None, help="Filter by name substring")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    with open(REGISTRY) as f:
        registry = json.load(f)

    checkpoints = registry["checkpoints"]

    if args.hacked:
        checkpoints = [c for c in checkpoints if c.get("final_metrics", {}).get("hack_rate_strict", 0) > 0.1]
    if args.name:
        checkpoints = [c for c in checkpoints if args.name.lower() in c["name"].lower()]

    if args.json:
        print(json.dumps(checkpoints, indent=2))
        return

    if not checkpoints:
        print("No matching checkpoints found.")
        return

    for c in checkpoints:
        metrics = c.get("final_metrics", {})
        path = c.get("tinker_path") or "(not recorded)"
        print(f"\n{'='*70}")
        print(f"  {c['name']}")
        print(f"  {c['description']}")
        print(f"  model: {c['base_model']}  |  loss: {c['loss_type']}  |  kl: {c['kl_coef']}")
        print(f"  steps: {c['completed_steps']}/{c['max_steps']}  |  date: {c['date']}")
        print(f"  correct: {metrics.get('correct_rate', '?'):.1%}  hack: {metrics.get('hack_rate_strict', '?'):.1%}  compile: {metrics.get('compile_rate', '?'):.1%}")
        print(f"  tinker_path: {path}")
        if c.get("notes"):
            print(f"  notes: {c['notes']}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
