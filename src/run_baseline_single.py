"""Run baseline evaluation for a single model. Run multiple in parallel."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from models.tinker_model import TinkerModel
from models.model import ModelInput
from prompts import RoleType

# Args: <model_id> <alias> <max_tokens> <n_problems>
model_id = sys.argv[1]
alias = sys.argv[2]
max_tokens = int(sys.argv[3])
n_problems = int(sys.argv[4]) if len(sys.argv) > 4 else 20

with open(Path(__file__).parent / "olympiad_filtered.json") as f:
    problems = json.load(f)[:n_problems]

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


print(f"[{alias}] Evaluating {n_problems} problems (max_tokens={max_tokens})")
model = TinkerModel.from_base_model(
    base_model=model_id, alias=alias,
    temperature=0.0, max_tokens=max_tokens, enable_thinking=False,
)

results = []
batch_size = 10
for bs in range(0, len(problems), batch_size):
    be = min(bs + batch_size, len(problems))
    batch = problems[bs:be]
    prompts = [build_prompt(p["question"]) for p in batch]
    responses = model.predict(prompts, max_new_tokens=max_tokens)
    for resp, prob in zip(responses, batch):
        pred = extract_answer(resp.speech) if not resp.failed else "[FAILED]"
        ok = check(pred, prob["answer"])
        results.append({
            "correct": ok, "pred": pred[:60],
            "gold": prob["answer"][:40],
            "tokens": len(resp.response_tokens),
        })
        status = "OK" if ok else "X"
        print(f"  [{status:>2}] Gold={prob['answer']:20s} Pred={pred[:40]:40s} tok={len(resp.response_tokens)}")
    nc = sum(r["correct"] for r in results)
    print(f"  --- [{alias}] {be}/{len(problems)}: {nc}/{be} ({nc/be:.0%})")

nc = sum(r["correct"] for r in results)
acc = nc / len(results)
avg_tok = sum(r["tokens"] for r in results) / len(results)
print(f"\n[{alias}] FINAL: {nc}/{len(results)} ({acc:.0%})  avg_tokens={avg_tok:.0f}")

# Save results
out_path = Path(__file__).parent / f"baseline_{alias}.json"
with open(out_path, "w") as f:
    json.dump({"model_id": model_id, "alias": alias, "n": len(results),
               "accuracy": acc, "results": results}, f, indent=2)
print(f"[{alias}] Saved to {out_path}")
