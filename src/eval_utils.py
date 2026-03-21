"""
Shared evaluation utilities for answer extraction, normalization, and checking.

This module is the single source of truth for answer verification across all
scripts (filtering, baseline eval, post-training eval). Changes here affect
all downstream results.

Known issues addressed:
- Nested braces in \\boxed{} (e.g. \\boxed{\\frac{3}{2}})
- LaTeX units in gold answers (e.g. "9 \\, \\text{cm}^2" should match "9")
- Degree notation (20° vs 20^\\circ vs 20)
"""
import re
from fractions import Fraction


def extract_boxed(text: str) -> list[str]:
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
    """Extract the final answer from model response.

    Priority:
    1. Last \\boxed{...} (handles nested braces)
    2. gpt-oss |message|> channel format
    3. "Answer: ..." line
    4. Empty string
    """
    # Balanced-brace boxed extraction (handles nested braces)
    matches = extract_boxed(response)
    if matches:
        return matches[-1].strip()

    # gpt-oss final channel
    m = re.search(r'\|message\|>(.+?)(?:<\|return\|>|<\||$)', response)
    if m:
        return m.group(1).strip()

    # Answer: line
    m = re.search(r'(?:^|\n)\s*(?:\*\*)?Answer:?\s*(?:\*\*)?\s*(.+?)$',
                  response, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return ""


def normalize(s: str) -> str:
    """Normalize a math answer string for comparison.

    Strips LaTeX formatting, units, degree symbols, and whitespace to
    produce a canonical form for string comparison.
    """
    s = s.strip()
    # Remove surrounding $...$
    s = re.sub(r'^\$+|\$+$', '', s)
    # Remove \boxed{...} wrapper
    m = re.match(r'\\boxed\{(.+)\}', s, re.DOTALL)
    if m:
        s = m.group(1)
    # \dfrac -> \frac
    s = s.replace('\\dfrac', '\\frac')
    # Strip leading "x = ", "y = " etc.
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
    # Strip LaTeX units: \text{...}, \, \text{...}, \mathrm{...}
    # e.g. "9 \, \text{cm}^2" -> "9", "24 \text{ patches}" -> "24"
    s = re.sub(r'\\,?\s*\\(?:text|mathrm|textbf)\{[^{}]*\}', '', s)
    # Strip standalone unit words at the end (cm, minutes, patches, etc.)
    s = re.sub(r'\s+(?:cm|m|kg|minutes?|hours?|seconds?|patches|participants|books|°)(?:\^?\d*)?$', '', s, flags=re.IGNORECASE)
    # Degree notation: 20^\circ -> 20, 20° -> 20
    s = re.sub(r'\^?\\circ', '', s)
    s = s.replace('°', '')
    # \angle X = Y -> Y
    s = re.sub(r'\\angle\s*\w+\s*=\s*', '', s)
    # Normalize frac: \frac{a}{b} -> (a)/(b)
    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    # Short frac: \frac ab -> (a)/(b)
    s = re.sub(r'\\frac\s*(\d)\s*(\d)', r'(\1)/(\2)', s)
    # Remove \left, \right, \cdot, \times
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # Remove \sqrt{} -> sqrt()
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    # Remove spacing commands
    s = re.sub(r'\\[,;:!\s]', '', s)
    # Remove trailing punctuation
    s = s.rstrip(".,;")
    # Normalize whitespace and lowercase
    return re.sub(r'\s+', ' ', s).strip().lower()


def check(pred: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer.

    Tries string match, numeric match, and fraction match after normalization.
    """
    p, g = normalize(pred), normalize(gold)
    if not p:
        return False
    if p == g:
        return True

    # Try numeric comparison
    try:
        pn = float(p.replace(",", ""))
        gn = float(g.replace(",", ""))
        return abs(pn - gn) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass

    # Try fraction comparison
    try:
        return Fraction(p) == Fraction(g)
    except (ValueError, ZeroDivisionError):
        pass

    # Try evaluating simple expressions (2^10, etc.)
    try:
        pe = p.replace("^", "**")
        ge = g.replace("^", "**")
        if eval(pe) == eval(ge):
            return True
    except Exception:
        pass

    return False
