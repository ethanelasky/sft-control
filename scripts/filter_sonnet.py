"""
Filter Olympiads problems using Sonnet 4.6.

Following Ryd et al. (2025): use a strong model (they used Claude 4.5 Sonnet)
to solve problems, and only keep ones it gets right. This ensures we test on
solvable problems and gives us a reliable ground-truth verification.

Usage: python src/filter_sonnet.py [n_problems]
"""
import json
import os
import re
import sys
import random
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic

SEED = 42
N_PROBLEMS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MAX_CONCURRENT = 50  # Sonnet rate limit is generous

SYSTEM_PROMPT = (
    r"You are a math problem solver. Solve the problem step by step, "
    r"then give your final answer inside \boxed{}."
    "\n\n"
    r"Example: The answer is \boxed{42}."
)


def extract_boxed(text):
    """Extract all \\boxed{...} contents, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r'\boxed{', i)
        if idx == -1:
            break
        start = idx + len(r'\boxed{')
        depth = 0
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    results.append(text[start:j])
                    break
                depth -= 1
        i = start
    return results


def extract_answer(response: str) -> str:
    """Extract answer from last \\boxed{...}, handling nested braces."""
    matches = extract_boxed(response)
    if matches:
        return matches[-1].strip()
    return ""


def normalize(s: str) -> str:
    """Normalize a math answer string for comparison."""
    s = re.sub(r'^\$+|\$+$', '', s.strip())
    m = re.match(r'\\boxed\{(.+)\}', s, re.DOTALL)
    if m:
        s = m.group(1)
    s = re.sub(r'\\,?\s*\\(?:text|mathrm|textbf)\{[^{}]*\}', '', s)
    s = re.sub(r'\s+(?:cm|m|kg|minutes?|hours?|seconds?|patches|participants|books|°)(?:\^?\d*)?$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\^?\\circ', '', s)
    s = s.replace('°', '')
    s = re.sub(r'\\angle\s*\w+\s*=\s*', '', s)
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\frac\s*(\d)\s*(\d)', r'(\1)/(\2)', s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\[,;:!\s]', '', s)
    s = s.rstrip(".,;")
    return re.sub(r'\s+', ' ', s).strip().lower()


def check(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    p, g = normalize(pred), normalize(gold)
    if not p:
        return False
    if p == g:
        return True
    try:
        return abs(float(p.replace(",", "")) - float(g.replace(",", ""))) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass
    try:
        from fractions import Fraction
        return Fraction(p) == Fraction(g)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        pe = p.replace("^", "**")
        ge = g.replace("^", "**")
        if eval(pe) == eval(ge):
            return True
    except Exception:
        pass
    return False


async def llm_check_equivalent(client: anthropic.AsyncAnthropic, pred: str, gold: str) -> bool:
    """Use Haiku to check if two math answers are equivalent."""
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": (
                f"Are these two mathematical answers equivalent? "
                f"Answer ONLY 'yes' or 'no'.\n\n"
                f"Answer 1: {gold}\n"
                f"Answer 2: {pred}"
            )}],
        )
        return response.content[0].text.strip().lower().startswith("yes")
    except Exception:
        return False


async def solve_problem(client: anthropic.AsyncAnthropic, problem: dict, semaphore: asyncio.Semaphore) -> dict:
    """Solve a single problem with Sonnet 4.6, verify with string match + Haiku fallback."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": problem["question"]}],
            )
            text = response.content[0].text
            pred = extract_answer(text)
            correct = check(pred, problem["answer"])

            # If string match fails but we have a prediction, try LLM judge
            if not correct and pred:
                correct = await llm_check_equivalent(
                    client, pred, problem["answer"]
                )

            return {
                "id": problem["id"],
                "correct": correct,
                "predicted": pred[:80],
                "gold": problem["answer"],
                "tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Error on problem {problem['id']}: {e}")
            return {
                "id": problem["id"],
                "correct": False,
                "predicted": f"[ERROR: {e}]",
                "gold": problem["answer"],
                "tokens": 0,
            }


async def main():
    # Load and sample problems (same seed as filter_eval_model.py)
    data_path = Path(__file__).parent.parent / "data" / "olympiad_filtered.json"
    with open(data_path) as f:
        all_problems = json.load(f)
    random.seed(SEED)
    problems = random.sample(all_problems, min(N_PROBLEMS, len(all_problems)))
    print(f"[sonnet] Evaluating {len(problems)} problems (seed={SEED})")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Run all problems concurrently (up to MAX_CONCURRENT)
    tasks = [solve_problem(client, p, semaphore) for p in problems]

    results = []
    # Process as they complete for progress reporting
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == len(problems):
            nc = sum(r["correct"] for r in results)
            print(f"  [sonnet] {i+1}/{len(problems)}: {nc}/{i+1} ({nc/(i+1):.0%})")

    # Sort results back to original order (by problem id)
    id_to_result = {r["id"]: r for r in results}
    results_ordered = [id_to_result[p["id"]] for p in problems]

    nc = sum(r["correct"] for r in results_ordered)
    print(f"\n[sonnet] FINAL: {nc}/{len(results_ordered)} ({nc/len(results_ordered):.0%})")

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "filter_results_sonnet.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "claude-sonnet-4-6",
            "n_problems": len(results_ordered),
            "accuracy": nc / len(results_ordered),
            "results": results_ordered,
        }, f, indent=2)
    print(f"[sonnet] Saved to {out_path}")

    # Also save the filtered problem IDs (problems Sonnet got right)
    sonnet_correct_ids = {r["id"] for r in results_ordered if r["correct"]}
    print(f"[sonnet] {len(sonnet_correct_ids)} problems Sonnet solved correctly — these are the solvable set")


if __name__ == "__main__":
    asyncio.run(main())
