"""
Run Sonnet on problems 500-1500 (batch 2).
Batch 1 (problems 0-499) already done in filter_results_sonnet.json.
Uses same seed=42 sample, picks indices 500-1499.
"""
import json
import re
import os
import sys
import random
import asyncio
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # Rely on env vars being set

import anthropic

SEED = 42
START = 500
END = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
MAX_CONCURRENT = 50

SYSTEM_PROMPT = (
    r"You are a math problem solver. Solve the problem step by step, "
    r"then give your final answer inside \boxed{}."
    "\n\n"
    r"Example: The answer is \boxed{42}."
)


def extract_boxed(text):
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


def extract_answer(response):
    matches = extract_boxed(response)
    if matches:
        return matches[-1].strip()
    return ""


def normalize(s):
    s = re.sub(r'^\$+|\$+$', '', s.strip())
    m = re.match(r'\\boxed\{(.+)\}', s, re.DOTALL)
    if m:
        s = m.group(1)
    s = s.replace('\\dfrac', '\\frac')
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
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


def check(pred, gold):
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
    return False


async def llm_check_equivalent(client, pred, gold):
    try:
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": (
                f"Are these two mathematical answers equivalent? "
                f"Answer ONLY 'yes' or 'no'.\n\n"
                f"Answer 1: {gold}\nAnswer 2: {pred}"
            )}],
        )
        return response.content[0].text.strip().lower().startswith("yes")
    except Exception:
        return False


async def solve_problem(client, problem, semaphore):
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
            if not correct and pred:
                correct = await llm_check_equivalent(client, pred, problem["answer"])
            return {
                "id": problem["id"],
                "correct": correct,
                "predicted": pred[:80],
                "gold": problem["answer"],
                "tokens": response.usage.output_tokens,
            }
        except Exception as e:
            print(f"  Error on problem {problem['id']}: {e}", flush=True)
            return {
                "id": problem["id"],
                "correct": False,
                "predicted": f"[ERROR: {e}]",
                "gold": problem["answer"],
                "tokens": 0,
            }


async def main():
    data_path = Path(__file__).parent.parent / "data" / "olympiad_filtered.json"
    with open(data_path) as f:
        all_problems = json.load(f)
    random.seed(SEED)
    sampled = random.sample(all_problems, min(END, len(all_problems)))
    problems = sampled[START:END]
    print(f"[sonnet-batch2] Evaluating problems {START}-{END-1} ({len(problems)} problems)", flush=True)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [solve_problem(client, p, semaphore) for p in problems]

    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        if (i + 1) % 100 == 0 or (i + 1) == len(problems):
            nc = sum(r["correct"] for r in results)
            print(f"  [sonnet-batch2] {i+1}/{len(problems)}: {nc}/{i+1} ({nc/(i+1):.0%})", flush=True)

    id_to_result = {r["id"]: r for r in results}
    results_ordered = [id_to_result[p["id"]] for p in problems]

    nc = sum(r["correct"] for r in results_ordered)
    print(f"\n[sonnet-batch2] FINAL: {nc}/{len(results_ordered)} ({nc/len(results_ordered):.0%})", flush=True)

    out_path = Path(__file__).parent.parent / "data" / "filter_results_sonnet_batch2.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "claude-sonnet-4-6",
            "start": START, "end": END,
            "n_problems": len(results_ordered),
            "accuracy": nc / len(results_ordered),
            "results": results_ordered,
        }, f, indent=2)
    print(f"[sonnet-batch2] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
