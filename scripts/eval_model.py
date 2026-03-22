#!/usr/bin/env python3
"""Evaluate a model on loophole and benign coding problems.

Computes hack rates, correctness, and compile rates. Supports comparing
multiple evaluation results in a formatted table.

Usage:
    # Evaluate a base HF model
    python scripts/eval_model.py --model Qwen/Qwen3-4B

    # Evaluate a Tinker checkpoint
    python scripts/eval_model.py --model tinker://path/to/checkpoint --label "SFT step 200"

    # Skip benign evaluation
    python scripts/eval_model.py --model Qwen/Qwen3-4B --skip_benign

    # Compare multiple results
    python scripts/eval_model.py --compare
    python scripts/eval_model.py --compare results/base_eval.json results/sft_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src/ to path so imports work when running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, load_and_apply_loophole
from models.model import ModelInput, ModelResponse
from models.tinker_model import TinkerModel
from prompts import RoleType
from rewards import compute_rewards
from sandbox.executor import CodeExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 1536,
) -> TinkerModel:
    """Load a TinkerModel from a base HF model name or a Tinker checkpoint.

    Args:
        model_name: HF model name (e.g. "Qwen/Qwen3-4B") or
            "tinker://<checkpoint_path>" for a saved checkpoint.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for generation.

    Returns:
        A TinkerModel ready for inference.
    """
    if model_name.startswith("tinker://"):
        logger.info(f"Loading Tinker checkpoint: {model_name}")
        import tinker

        service_client = tinker.ServiceClient()
        # Training checkpoints (from save_state) need to go through
        # TrainingClient first to get a SamplingClient
        training_client = service_client.create_training_client_from_state(
            path=model_name
        )
        sampling_client = training_client.save_weights_and_get_sampling_client()
        return TinkerModel(
            alias=model_name.rsplit("/", 1)[-1],
            sampling_client=sampling_client,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=False,
        )
    else:
        logger.info(f"Loading base model: {model_name}")
        return TinkerModel.from_base_model(
            base_model=model_name,
            alias=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=False,
        )


def examples_to_inputs(examples: list[dict]) -> list[list[ModelInput]]:
    """Convert dataset examples to ModelInput format for predict_multi.

    Each example has a 'prompt' field which is a list of message dicts
    with 'role' and 'content' keys.

    Args:
        examples: Dataset examples loaded from JSONL.

    Returns:
        List of conversation inputs, one per example.
    """
    inputs = []
    for ex in examples:
        conversation = []
        for msg in ex["prompt"]:
            role_str = msg["role"].upper()
            role = RoleType[role_str]
            conversation.append(ModelInput(role=role, content=msg["content"]))
        inputs.append(conversation)
    return inputs


def evaluate_condition(
    model: TinkerModel,
    examples: list[dict],
    num_samples: int,
    executor: CodeExecutor,
    condition_name: str,
) -> tuple[dict, list[dict]]:
    """Evaluate a model on a set of examples.

    Generates num_samples responses per problem, then computes rewards
    and hack detection metrics.

    Args:
        model: The model to evaluate.
        examples: Dataset examples.
        num_samples: Number of responses to generate per problem.
        executor: CodeExecutor for sandboxed evaluation.
        condition_name: Name for logging (e.g. "loophole", "benign").

    Returns:
        Tuple of (aggregate_metrics_dict, per_problem_details_list).
    """
    logger.info(
        f"[{condition_name}] Generating {num_samples} responses for "
        f"{len(examples)} problems..."
    )

    # Convert to model inputs and generate
    inputs = examples_to_inputs(examples)
    t0 = time.time()
    multi_responses: list[list[ModelResponse]] = model.predict_multi(
        inputs, num_samples=num_samples
    )
    gen_time = time.time() - t0
    logger.info(
        f"[{condition_name}] Generation complete in {gen_time:.1f}s"
    )

    # Flatten: expand each example N times to match responses
    flat_responses: list[str] = []
    flat_examples: list[dict] = []
    for ex, responses in zip(examples, multi_responses):
        for resp in responses:
            flat_responses.append(resp.speech if not resp.failed else "")
            flat_examples.append(ex)

    logger.info(
        f"[{condition_name}] Evaluating {len(flat_responses)} responses..."
    )
    t0 = time.time()
    rewards, eval_details = compute_rewards(flat_responses, flat_examples, executor)
    eval_time = time.time() - t0
    logger.info(
        f"[{condition_name}] Evaluation complete in {eval_time:.1f}s"
    )

    # Aggregate metrics
    n_total = len(eval_details)
    n_correct = sum(1 for d in eval_details if d["eq_correct"])
    n_hinted = sum(1 for d in eval_details if d["eq_hinted"])
    n_compile = sum(1 for d in eval_details if d["can_compile"])
    n_hack_strict = sum(1 for d in eval_details if d["is_reward_hack_strict"])
    n_hack_loose = sum(1 for d in eval_details if d["is_reward_hack_loose"])
    n_failed = sum(
        1 for responses in multi_responses for r in responses if r.failed
    )

    aggregate = {
        "condition": condition_name,
        "num_problems": len(examples),
        "num_samples": num_samples,
        "total_responses": n_total,
        "correct_rate": n_correct / n_total if n_total else 0.0,
        "hinted_rate": n_hinted / n_total if n_total else 0.0,
        "compile_rate": n_compile / n_total if n_total else 0.0,
        "hack_rate_strict": n_hack_strict / n_total if n_total else 0.0,
        "hack_rate_loose": n_hack_loose / n_total if n_total else 0.0,
        "failed_generation_rate": n_failed / n_total if n_total else 0.0,
        "generation_time_s": gen_time,
        "evaluation_time_s": eval_time,
    }

    # Per-problem breakdown
    per_problem = []
    for prob_idx, ex in enumerate(examples):
        start = prob_idx * num_samples
        end = start + num_samples
        prob_details = eval_details[start:end]

        prob_correct = sum(1 for d in prob_details if d["eq_correct"])
        prob_hinted = sum(1 for d in prob_details if d["eq_hinted"])
        prob_compile = sum(1 for d in prob_details if d["can_compile"])
        prob_hack_strict = sum(1 for d in prob_details if d["is_reward_hack_strict"])
        prob_hack_loose = sum(1 for d in prob_details if d["is_reward_hack_loose"])

        per_problem.append({
            "problem_id": ex.get("id", prob_idx),
            "correct_rate": prob_correct / num_samples,
            "hinted_rate": prob_hinted / num_samples,
            "compile_rate": prob_compile / num_samples,
            "hack_rate_strict": prob_hack_strict / num_samples,
            "hack_rate_loose": prob_hack_loose / num_samples,
        })

    return aggregate, per_problem


def derive_label(model_name: str) -> str:
    """Derive a human-readable label from a model name or path.

    Args:
        model_name: HF model name or tinker://path.

    Returns:
        Short label string.
    """
    if model_name.startswith("tinker://"):
        path = model_name[len("tinker://"):]
        return os.path.basename(path.rstrip("/"))
    return model_name.split("/")[-1]


def print_summary(
    label: str,
    loophole_agg: dict | None,
    benign_agg: dict | None,
) -> None:
    """Print a formatted summary of evaluation results.

    Args:
        label: Model label.
        loophole_agg: Aggregate metrics for loophole condition (or None).
        benign_agg: Aggregate metrics for benign condition (or None).
    """
    print("\n" + "=" * 70)
    print(f"  Evaluation Results: {label}")
    print("=" * 70)

    if loophole_agg:
        print(f"\n  LOOPHOLE CONDITION ({loophole_agg['num_problems']} problems, "
              f"{loophole_agg['num_samples']} samples each)")
        print(f"    Hack rate (strict):  {loophole_agg['hack_rate_strict']:.1%}")
        print(f"    Hack rate (loose):   {loophole_agg['hack_rate_loose']:.1%}")
        print(f"    Correct rate:        {loophole_agg['correct_rate']:.1%}")
        print(f"    Compile rate:        {loophole_agg['compile_rate']:.1%}")
        print(f"    Failed generation:   {loophole_agg['failed_generation_rate']:.1%}")

    if benign_agg:
        print(f"\n  BENIGN CONDITION ({benign_agg['num_problems']} problems, "
              f"{benign_agg['num_samples']} samples each)")
        print(f"    Correct rate:        {benign_agg['correct_rate']:.1%}")
        print(f"    Compile rate:        {benign_agg['compile_rate']:.1%}")
        print(f"    Hack rate (strict):  {benign_agg['hack_rate_strict']:.1%}")
        print(f"    Failed generation:   {benign_agg['failed_generation_rate']:.1%}")

    print("=" * 70 + "\n")


def run_compare(file_list: list[str], output_dir: str) -> None:
    """Load and compare multiple evaluation result files in a table.

    Args:
        file_list: Explicit list of JSON files to compare, or empty to
            auto-discover all *_eval.json in output_dir.
        output_dir: Directory to search for result files if file_list is empty.
    """
    if not file_list:
        output_path = Path(output_dir)
        if not output_path.exists():
            print(f"No results directory found at {output_dir}")
            return
        file_list = sorted(str(p) for p in output_path.glob("*_eval.json"))
        if not file_list:
            print(f"No *_eval.json files found in {output_dir}")
            return

    results = []
    for fp in file_list:
        try:
            with open(fp) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {fp}: {e}")

    if not results:
        print("No valid result files loaded.")
        return

    # Determine which columns to show
    has_loophole = any("hack_rate_strict" in r.get("summary", {}) for r in results)
    has_benign = any("correct_rate_benign" in r.get("summary", {}) for r in results)

    # Find the first result to use as baseline for capability preservation
    base_benign_correct = None
    if has_benign:
        for r in results:
            cr = r.get("summary", {}).get("correct_rate_benign")
            if cr is not None:
                base_benign_correct = cr
                break

    # Build table
    headers = ["Label"]
    if has_loophole:
        headers += ["Hack(S)", "Hack(L)", "Correct(L)", "Compile(L)"]
    if has_benign:
        headers += ["Correct(B)", "Compile(B)"]
        if base_benign_correct and base_benign_correct > 0:
            headers += ["Cap.Pres."]

    # Format rows
    rows = []
    for r in results:
        s = r.get("summary", {})
        label = r.get("metadata", {}).get("label", "?")
        row = [label]
        if has_loophole:
            row.append(f"{s.get('hack_rate_strict', 0):.1%}")
            row.append(f"{s.get('hack_rate_loose', 0):.1%}")
            row.append(f"{s.get('correct_rate_loophole', 0):.1%}")
            row.append(f"{s.get('compile_rate_loophole', 0):.1%}")
        if has_benign:
            row.append(f"{s.get('correct_rate_benign', 0):.1%}")
            row.append(f"{s.get('compile_rate_benign', 0):.1%}")
            if base_benign_correct and base_benign_correct > 0:
                benign_cr = s.get("correct_rate_benign", 0)
                cap_pres = benign_cr / base_benign_correct if base_benign_correct else 0
                row.append(f"{cap_pres:.1%}")
        rows.append(row)

    # Compute column widths
    col_widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    # Print table
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "  ".join("-" * w for w in col_widths)
    print(f"\n{header_line}")
    print(separator)
    for row in rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, col_widths)))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on loophole and benign coding problems."
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF model name (e.g. Qwen/Qwen3-4B) or tinker://<checkpoint_path>",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-readable label for this evaluation (default: derived from model)",
    )

    # Generation args
    parser.add_argument("--num_samples", type=int, default=4, help="Responses per problem")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=1536, help="Max tokens per response")

    # Data args
    parser.add_argument(
        "--loophole_dataset",
        type=str,
        default="data/leetcode_test_loophole.jsonl",
        help="Path to loophole dataset",
    )
    parser.add_argument(
        "--benign_dataset",
        type=str,
        default="data/leetcode_test_medhard.jsonl",
        help="Path to benign dataset",
    )

    # Output args
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory")

    # Execution args
    parser.add_argument("--sandbox_workers", type=int, default=16, help="Sandbox worker count")

    # Condition flags
    parser.add_argument("--skip_benign", action="store_true", help="Skip benign evaluation")
    parser.add_argument("--skip_loophole", action="store_true", help="Skip loophole evaluation")

    # Compare mode
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="Compare mode: load result JSONs and print table. "
        "Pass file paths or omit for auto-discovery in output_dir.",
    )

    args = parser.parse_args()

    # Compare mode: no model needed
    if args.compare is not None:
        run_compare(args.compare, args.output_dir)
        return

    # Evaluation mode: model required
    if args.model is None:
        parser.error("--model is required unless using --compare")

    label = args.label or derive_label(args.model)

    # Resolve dataset paths relative to project root
    loophole_path = os.path.join(PROJECT_ROOT, args.loophole_dataset)
    benign_path = os.path.join(PROJECT_ROOT, args.benign_dataset)

    # Load model
    model = load_model(args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    # Create executor
    executor = CodeExecutor(num_workers=args.sandbox_workers)

    # Run evaluations
    loophole_agg = None
    loophole_per_problem = None
    benign_agg = None
    benign_per_problem = None

    if not args.skip_loophole:
        if not os.path.exists(loophole_path):
            logger.warning(f"Loophole dataset not found: {loophole_path}, skipping")
        else:
            loophole_examples = load_dataset(loophole_path)
            logger.info(f"Loaded {len(loophole_examples)} loophole examples")
            loophole_agg, loophole_per_problem = evaluate_condition(
                model, loophole_examples, args.num_samples, executor, "loophole"
            )

    if not args.skip_benign:
        if not os.path.exists(benign_path):
            logger.warning(f"Benign dataset not found: {benign_path}, skipping")
        else:
            benign_examples = load_dataset(benign_path)
            logger.info(f"Loaded {len(benign_examples)} benign examples")
            benign_agg, benign_per_problem = evaluate_condition(
                model, benign_examples, args.num_samples, executor, "benign"
            )

    # Build output
    summary = {}
    if loophole_agg:
        summary["hack_rate_strict"] = loophole_agg["hack_rate_strict"]
        summary["hack_rate_loose"] = loophole_agg["hack_rate_loose"]
        summary["correct_rate_loophole"] = loophole_agg["correct_rate"]
        summary["compile_rate_loophole"] = loophole_agg["compile_rate"]
    if benign_agg:
        summary["correct_rate_benign"] = benign_agg["correct_rate"]
        summary["compile_rate_benign"] = benign_agg["compile_rate"]

    # Per-problem combined
    per_problem_combined = []
    problem_ids = set()
    loophole_by_id = {}
    benign_by_id = {}

    if loophole_per_problem:
        for p in loophole_per_problem:
            pid = p["problem_id"]
            problem_ids.add(pid)
            loophole_by_id[pid] = p

    if benign_per_problem:
        for p in benign_per_problem:
            pid = p["problem_id"]
            problem_ids.add(pid)
            benign_by_id[pid] = p

    for pid in sorted(problem_ids, key=str):
        entry = {"problem_id": pid}
        if pid in loophole_by_id:
            entry["loophole"] = {
                k: v for k, v in loophole_by_id[pid].items() if k != "problem_id"
            }
        if pid in benign_by_id:
            entry["benign"] = {
                k: v for k, v in benign_by_id[pid].items() if k != "problem_id"
            }
        per_problem_combined.append(entry)

    output = {
        "metadata": {
            "model": args.model,
            "label": label,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "loophole_dataset": args.loophole_dataset,
            "benign_dataset": args.benign_dataset,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "per_problem": per_problem_combined,
    }

    if loophole_agg:
        output["loophole_aggregate"] = loophole_agg
    if benign_agg:
        output["benign_aggregate"] = benign_agg

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    safe_label = label.replace("/", "_").replace(" ", "_")
    output_path = os.path.join(args.output_dir, f"{safe_label}_eval.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    print_summary(label, loophole_agg, benign_agg)


if __name__ == "__main__":
    main()
