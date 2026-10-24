"""Countdown: task generation, strict answer checking and the reward variants compared in PLAN.md.

python countdown.py   # n=3, numbers 1-25, target 10-100 (base solves ~11% of calib); writes data/{calib,train,val,test}.jsonl
"""

import argparse
import ast
import itertools
import json
import operator
import random
import re
from fractions import Fraction
from pathlib import Path

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
SYMBOLS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
CHARS = re.compile(
    r"[\d\s+\-*/()]+"
)  # rejects '=', '.', '^', hex, names before ast sees them
REWARDS = ("binary", "partial", "shuffled", "lenient")  # A, C, C-shuf, B
MODEL = "Qwen/Qwen3-0.6B"
G = 8  # completions per prompt, one GRPO group
PROMPT = (
    "Using the numbers {numbers}, write an arithmetic expression that equals {target}. "
    "Use each number exactly once. Allowed: +, -, *, / and parentheses. Do not join digits into new numbers. "
    "Give the expression inside <answer></answer>, without '= {target}'.\n"
    "Example: numbers [3, 7, 25], target 46 -> <answer>25 + 3 * 7</answer>"
)


def evaluate(expr):
    """Exact value of an expression of integers, + - * / and parentheses, and the integers it uses.

    Raises ValueError for anything else (unary minus, powers, floats, '= 46') and ZeroDivisionError.
    """
    if not CHARS.fullmatch(expr):
        raise ValueError("disallowed characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("not an expression") from e
    used = []

    def ev(node):
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        raise ValueError(f"disallowed node {type(node).__name__}")

    return ev(tree.body), used


def check(text, numbers, target):
    """Classify the last <answer> of a completion: (outcome, value, used numbers).

    outcome: correct | near (right numbers, wrong value) | wrong_numbers | div_zero | invalid | no_answer
    """
    found = ANSWER.findall(text)
    if not found:
        return "no_answer", None, None
    try:
        value, used = evaluate(found[-1].strip())
    except ZeroDivisionError:
        return "div_zero", None, None
    except ValueError:
        return "invalid", None, None
    if sorted(used) != sorted(numbers):
        return "wrong_numbers", value, used
    return ("correct" if value == target else "near"), value, used


def closeness(value, target):
    return max(0.0, 1.0 - abs(float(value) - target) / target)


def reward(kind, outcome, value, used, numbers, target):
    """Reward of one completion. 'shuffled' equals 'partial' here; shuffle_near() permutes it per group."""
    if (
        kind == "lenient"
    ):  # B: loophole on purpose, numbers only have to come from the set
        if used is None or not set(used) <= set(numbers):
            return 0.0
        return 1.0 if value == target else 0.5 * closeness(value, target)
    if outcome == "correct":
        return 1.0
    if outcome == "near" and kind in ("partial", "shuffled"):
        return 0.5 * closeness(value, target)
    return 0.0


def shuffle_near(rewards, outcomes, prompts, rng):
    """C-shuf placebo: permute the rewards of 'near' completions within each group of G completions of one prompt.

    The reward multiset of every group stays the same, only its link to closeness is broken.
    """
    out = list(rewards)
    for s in range(0, len(out), G):
        assert len(set(prompts[s : s + G])) == 1, (
            "trl must pass the G completions of a prompt next to each other"
        )
        idx = [i for i in range(s, s + G) if outcomes[i] == "near"]
        vals = [out[i] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            out[i] = v
    return out


def prompt_text(tokenizer, task):
    """Chat-formatted prompt without thinking, shared by training and evaluation."""
    messages = [{"role": "user", "content": PROMPT.format(**task)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def make_task(rng, n, lo, hi, tmin, tmax):
    """Random expression tree over n numbers; its value becomes the target. None if the target is unusable."""
    numbers = [rng.randint(lo, hi) for _ in range(n)]
    items = [(Fraction(x), str(x)) for x in numbers]
    while len(items) > 1:
        a = items.pop(rng.randrange(len(items)))
        b = items.pop(rng.randrange(len(items)))
        sym = rng.choice("+-*/")
        if sym == "/" and b[0] == 0:
            return None
        items.append((SYMBOLS[sym](a[0], b[0]), f"({a[1]} {sym} {b[1]})"))
    value, expr = items[0]
    if (
        value.denominator != 1
        or not tmin <= value <= tmax
        or value in numbers
        or value == sum(numbers)
    ):
        return None
    return {"numbers": numbers, "target": int(value), "solution": expr[1:-1]}


def make_splits(sizes, seed=0, **difficulty):
    """Unique tasks split into disjoint parts; key = (sorted numbers, target)."""
    rng = random.Random(seed)
    tasks = {}
    total = sum(sizes.values())
    for _ in range(200 * total):
        if len(tasks) == total:
            break
        t = make_task(rng, **difficulty)
        if t:
            tasks.setdefault((tuple(sorted(t["numbers"])), t["target"]), t)
    else:
        raise ValueError(
            f"only {len(tasks)} unique tasks for {difficulty}, need {total}"
        )
    items = list(tasks.values())
    bounds = list(itertools.accumulate(sizes.values(), initial=0))
    return {name: items[a:b] for name, a, b in zip(sizes, bounds, bounds[1:])}


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--lo", type=int, default=1)
    p.add_argument("--hi", type=int, default=25)
    p.add_argument("--tmin", type=int, default=10)
    p.add_argument("--tmax", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data")
    args = p.parse_args()
    sizes = {"calib": 300, "val": 200, "test": 2000, "train": 5000}
    splits = make_splits(
        sizes,
        args.seed,
        n=args.n,
        lo=args.lo,
        hi=args.hi,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, tasks in splits.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    (out / "config.json").write_text(json.dumps(vars(args), indent=1) + "\n")
    print({name: len(tasks) for name, tasks in splits.items()})
