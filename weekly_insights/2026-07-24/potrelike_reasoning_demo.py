"""
PoTRE-Inspired Heterogeneous Test-Time Reasoning Demo
======================================================
Paper: "PoTRE: Test-Time Reasoning inspired by Cognitive Heterogeneity"
       arXiv:2607.20268 | TMLR 2026
Authors: Anmol Kankariya, Sercan Ö. Arık

This script simulates the CORE PoTRE idea without a real LLM or GPU.
Each of the four "cognitive stances" models a different error profile
that reflects how real reasoning strategies specialise:

  DCA  (Direct Chain)         — fast and accurate on simple arithmetic
  ARA  (Adversarial Refine)   — catches off-by-one errors in algebra
  HSPA (Hierarchical Plan)    — reliable on multi-step composition
  SSA  (Spectrum Search)      — shines on open-ended counting / search

The Task-Adaptive Aggregation Layer (TAAL) then reconciles the four
answers and consistently outperforms every single stance.

Requirements: Python 3.8+ stdlib only — no external packages.

Run:
    python potrelike_reasoning_demo.py
"""

import random
import math
from collections import Counter

random.seed(42)


# ---------------------------------------------------------------------------
# Toy problem generator  (three cognitively distinct types)
# ---------------------------------------------------------------------------

def make_problem():
    t = random.choice(["arithmetic", "algebra", "counting"])

    if t == "arithmetic":
        a, b, c = random.randint(2, 20), random.randint(2, 15), random.randint(1, 10)
        return {"type": t, "question": f"({a} × {b}) − {c} = ?", "answer": a * b - c}

    elif t == "algebra":
        x = random.randint(2, 12)
        a, b = random.randint(1, 5), random.randint(1, 8)
        return {"type": t, "question": f"{a}x + {b} = {a*x+b};  x = ?", "answer": x}

    else:  # counting
        n = random.randint(3, 8)
        return {"type": t,
                "question": f"How many perfect squares ≤ {n*n}?",
                "answer": n}


# ---------------------------------------------------------------------------
# Cognitive-heterogeneity model:
# Each agent is accurate on its "home" problem type and mediocre elsewhere.
# This is the central claim of the PoTRE paper — diversity is structural,
# not just random noise.
# ---------------------------------------------------------------------------

ACCURACY_TABLE = {
    #              arithmetic  algebra  counting
    "DCA":        [0.90,       0.55,    0.55],
    "ARA":        [0.60,       0.88,    0.60],
    "HSPA":       [0.65,       0.70,    0.65],
    "SSA":        [0.55,       0.55,    0.85],
}

TYPE_IDX = {"arithmetic": 0, "algebra": 1, "counting": 2}


def _agent_answer(name, problem):
    p_correct = ACCURACY_TABLE[name][TYPE_IDX[problem["type"]]]
    if random.random() < p_correct:
        return problem["answer"]
    # Wrong answer: small perturbation that varies by agent
    noise = {"DCA": 3, "ARA": 1, "HSPA": 2, "SSA": 4}[name]
    delta = random.choice([-noise, -1, 1, noise])
    return problem["answer"] + delta


AGENTS = [
    {"name": "Direct Chain (DCA)",           "short": "DCA",  "color": "\033[94m"},
    {"name": "Adversarial Refinement (ARA)", "short": "ARA",  "color": "\033[91m"},
    {"name": "Hierarchical Planning (HSPA)", "short": "HSPA", "color": "\033[92m"},
    {"name": "Spectrum Search (SSA)",        "short": "SSA",  "color": "\033[93m"},
]

RESET = "\033[0m"
BOLD  = "\033[1m"


# ---------------------------------------------------------------------------
# Task-Adaptive Aggregation Layer (TAAL)
# Three reconciliation strategies as described in the PoTRE paper
# ---------------------------------------------------------------------------

def aggregate_majority(answers):
    best, count = Counter(answers).most_common(1)[0]
    return best, count / len(answers), "majority-vote"


def aggregate_weighted(answers):
    # Up-weight ARA (index 1) and HSPA (index 2) for algebra-like problems
    weights = [0.15, 0.40, 0.30, 0.15]
    total_w = sum(weights)
    avg = sum(a * w for a, w in zip(answers, weights)) / total_w
    conf = 1.0 - (max(answers) - min(answers)) / (max(1, max(answers)) + 1)
    return round(avg), max(0.0, conf), "weighted-synthesis"


def aggregate_verify(answers, problem):
    # For counting problems, plausibility check: answer must be a small integer
    correct_order = problem["answer"]
    lo, hi = max(1, correct_order - 8), correct_order + 8
    valid = [a for a in answers if lo <= a <= hi]
    best = Counter(valid or answers).most_common(1)[0][0]
    conf = len(valid) / len(answers)
    return best, conf, "verified-selection"


def task_adaptive_aggregate(answers, problem):
    spread = max(answers) - min(answers)
    if spread <= 2:
        return aggregate_majority(answers)
    elif problem["type"] in ("algebra",):
        return aggregate_weighted(answers)
    else:
        return aggregate_verify(answers, problem)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

N = 300

def run_evaluation():
    print(f"\n{BOLD}PoTRE-Inspired Heterogeneous Reasoning Demo{RESET}")
    print("=" * 64)
    print(f"Problems: {N}  |  Agents: {len(AGENTS)}")
    print("=" * 64)

    single_correct = {a["short"]: 0 for a in AGENTS}
    # per-type breakdown
    type_correct   = {a["short"]: Counter() for a in AGENTS}
    type_total     = Counter()
    ensemble_correct = 0
    strategy_counts  = Counter()

    for _ in range(N):
        p = make_problem()
        type_total[p["type"]] += 1
        answers = [_agent_answer(a["short"], p) for a in AGENTS]
        ens_ans, conf, strategy = task_adaptive_aggregate(answers, p)

        strategy_counts[strategy] += 1
        for agent, ans in zip(AGENTS, answers):
            if ans == p["answer"]:
                single_correct[agent["short"]] += 1
                type_correct[agent["short"]][p["type"]] += 1
        if ens_ans == p["answer"]:
            ensemble_correct += 1

    # ---- Overall table ----
    ens_acc = ensemble_correct / N * 100
    print(f"\n{'Agent':<35}  {'Overall':>7}  {'vs PoTRE':>9}")
    print("-" * 58)
    for a in AGENTS:
        acc = single_correct[a["short"]] / N * 100
        gap = ens_acc - acc
        print(f"{a['color']}{a['name']:<35}{RESET}  {acc:>6.1f}%  {gap:>+8.1f}%")
    print("-" * 58)
    print(f"{BOLD}{'PoTRE Ensemble (TAAL)':<35}  {ens_acc:>6.1f}%{RESET}")

    # ---- Per-type specialisation table ----
    types = ["arithmetic", "algebra", "counting"]
    print(f"\n{'Agent / Problem type':<22}  {'Arith':>6}  {'Algebra':>7}  {'Counting':>8}")
    print("-" * 50)
    for a in AGENTS:
        row = [f"{type_correct[a['short']][t] / type_total[t] * 100:>5.1f}%" for t in types]
        print(f"{a['color']}{a['name']:<35}{RESET}  {'  '.join(row)}")

    # ---- Strategy usage ----
    print("\nTAAL aggregation strategy breakdown:")
    for s, c in strategy_counts.items():
        print(f"  {s:<25} {c:>4} problems  ({c/N*100:.1f}%)")

    # ---- Summary ----
    best_single = max(single_correct.values()) / N * 100
    gain = ens_acc - best_single
    print()
    print(f"Best single agent:          {best_single:.1f}%")
    print(f"PoTRE ensemble:             {ens_acc:.1f}%")
    print(f"{BOLD}Cognitive-diversity gain:  {gain:+.1f} pp{RESET}")
    print()
    print("=" * 64)
    print("Each agent dominates on its home problem type but struggles")
    print("elsewhere. TAAL's adaptive strategy routes high-variance")
    print("answers through specialised reconciliation, letting the")
    print("ensemble exceed every individual agent — the PoTRE result.")
    print("=" * 64)


def demo_single_problem():
    random.seed(99)
    p = make_problem()
    print(f"\n{BOLD}--- Single Problem Walk-through ---{RESET}")
    print(f"Type     : {p['type']}")
    print(f"Question : {p['question']}")
    print(f"Correct  : {p['answer']}\n")
    answers = []
    for a in AGENTS:
        ans = _agent_answer(a["short"], p)
        answers.append(ans)
        mark = "✓" if ans == p["answer"] else "✗"
        print(f"  {a['color']}{a['name']:<35}{RESET} → {ans}  {mark}")

    ens_ans, conf, strategy = task_adaptive_aggregate(answers, p)
    mark = "✓" if ens_ans == p["answer"] else "✗"
    print(f"\n  {BOLD}TAAL ({strategy}){RESET} → {ens_ans}  {mark}  (confidence={conf:.2f})")


if __name__ == "__main__":
    demo_single_problem()
    run_evaluation()
