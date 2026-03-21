"""Evaluate a single model on sampled Olympiads problems for dataset filtering.
Run two instances in parallel (one per model), then merge with filter_merge.py.

Usage: python filter_eval_model.py <model_id> <alias> <n_problems>
"""
import json
import re
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from models.tinker_model import TinkerModel
from models.model import ModelInput
from prompts import RoleType

model_id = sys.argv[1]
alias = sys.argv[2]
n_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 500
SEED = 42
MAX_TOKENS = 32000
BATCH_SIZE = 35

# Load and sample (same seed = same sample for both models)
with open(Path(__file__).parent / "olympiad_filtered.json") as f:
    all_problems = json.load(f)
random.seed(SEED)
problems = random.sample(all_problems, min(n_problems, len(all_problems)))
print(f"[{alias}] Sampled {len(problems)} problems (seed={SEED})")

SYSTEM_PROMPT = (
    r"You are a math problem solver. Solve the problem step by step, "
    r"then give your final answer inside \boxed{}."
    "\n\n"
    r"Example: The answer is \boxed{42}."
)


def build_prompt(q):
    return [
        ModelInput(role=RoleType.SYSTEM, content=SYSTEM_PROMPT),
        ModelInput(role=RoleType.USER, content=q),
    ]


def extract_answer(response):
    matches = re.findall(r'\\boxed\{([^{}]+)\}', response)
    if matches:
        return matches[-1].strip()
    m = re.search(r'\|message\|>(.+?)(?:<\|return\|>|<\||$)', response)
    if m:
        return m.group(1).strip()
    return ""


def normalize(s):
    s = re.sub(r'^\$+|\$+$', '', s.strip())
    m = re.match(r'\\boxed\{(.+)\}', s)
    if m:
        s = m.group(1)
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    s = re.sub(r'\\frac\s*(\d)\s*(\d)', r'(\1)/(\2)', s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r'\\[,;:!\s]', '', s)
    s = s.rstrip(".,;")
    return re.sub(r'\s+', ' ', s).strip().lower()


def check(pred, gold):
    p, g = normalize(pred), normalize(gold)
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


print(f"[{alias}] Creating model {model_id}...")
model = TinkerModel.from_base_model(
    base_model=model_id, alias=alias,
    temperature=0.0, max_tokens=MAX_TOKENS, enable_thinking=False,
)

results = []
for bs in range(0, len(problems), BATCH_SIZE):
    be = min(bs + BATCH_SIZE, len(problems))
    batch = problems[bs:be]
    prompts = [build_prompt(p["question"]) for p in batch]
    responses = model.predict(prompts, max_new_tokens=MAX_TOKENS)
    for resp, prob in zip(responses, batch):
        pred = extract_answer(resp.speech) if not resp.failed else ""
        ok = check(pred, prob["answer"])
        results.append({
            "id": prob["id"],
            "correct": ok,
            "predicted": pred[:80],
            "gold": prob["answer"],
            "tokens": len(resp.response_tokens),
        })
    nc = sum(r["correct"] for r in results)
    print(f"  [{alias}] {be}/{len(problems)}: {nc}/{be} ({nc/be:.0%})")

nc = sum(r["correct"] for r in results)
print(f"\n[{alias}] FINAL: {nc}/{len(results)} ({nc/len(results):.0%})")

out_path = Path(__file__).parent / f"filter_results_{alias}.json"
with open(out_path, "w") as f:
    json.dump({"model_id": model_id, "alias": alias, "results": results}, f, indent=2)
print(f"[{alias}] Saved to {out_path}")
