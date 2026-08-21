"""
Open-MOPD Proof-of-Concept: Token Budget Misallocation in Multi-Teacher Distillation

Based on: Open-MOPD: Diagnosing and Fixing Capability Imbalance in
Multi-Teacher On-Policy Distillation (arXiv:2608.19098, Aug 19 2026)

Demonstrates that naive multi-teacher on-policy distillation (M-OPD) allows
long-sequence tasks to crowd out short-sequence tasks in the optimization budget.
The fix: normalize each teacher's contribution by its average sequence length.

Requirements: numpy only (pip install numpy)
No GPU, no API key, no model download required.
"""

import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Synthetic "teachers" and task definitions
# ---------------------------------------------------------------------------
# Each teacher is modelled as an ideal distribution over token-level outputs.
# We track how well the student matches each teacher as a "skill score" in [0, 1].
# Oracle score = 1.0  (perfect specialist),  starting score = 0.30 (random baseline).

TASKS = {
    "math_reasoning": {"avg_tokens": 150, "std_tokens": 40},   # verbose, long-form
    "instruction_following": {"avg_tokens": 20, "std_tokens": 8},  # concise, short
}

N_STEPS = 100     # simulated gradient steps
BATCH_SIZE = 8    # examples per step
BASELINE = 0.30   # starting skill level (random-ish)
ORACLE = 1.00     # ceiling: what a dedicated specialist achieves


def sample_lengths(task_cfg: dict, n: int) -> np.ndarray:
    """Sample realistic token-sequence lengths for a task."""
    lengths = rng.normal(task_cfg["avg_tokens"], task_cfg["std_tokens"], n)
    return np.clip(lengths, 5, None).astype(int)


# ---------------------------------------------------------------------------
# 2. Training simulation
# ---------------------------------------------------------------------------

def simulate_training(rebalance: bool) -> dict:
    """
    Simulate one M-OPD training run.

    Each step:
      - Sample a batch for each teacher (varying sequence lengths)
      - Compute per-teacher gradient pressure (proportional to total tokens in batch)
      - If rebalance=True, divide by the task's expected token count
      - Distribute the optimization budget across tasks proportionally
      - Move each skill score toward the oracle by its budget share

    Returns:
        dict mapping task name -> final skill score
    """
    skills = {task: BASELINE for task in TASKS}
    lr = 0.015

    for _ in range(N_STEPS):
        raw_signals = {}
        for task, cfg in TASKS.items():
            lengths = sample_lengths(cfg, BATCH_SIZE)
            # Raw gradient pressure = total tokens this teacher generates
            raw_signals[task] = float(lengths.sum()) / BATCH_SIZE

        if rebalance:
            # Length-normalize: divide by expected sequence length so every task
            # gets equal weight regardless of verbosity
            signals = {
                task: raw_signals[task] / TASKS[task]["avg_tokens"]
                for task in TASKS
            }
        else:
            signals = raw_signals

        total_signal = sum(signals.values())

        for task in TASKS:
            budget_fraction = signals[task] / total_signal
            gap = ORACLE - skills[task]
            # Student moves toward oracle in proportion to budget it receives
            skills[task] += lr * budget_fraction * gap * 2.0

    return skills


# ---------------------------------------------------------------------------
# 3. Run both conditions
# ---------------------------------------------------------------------------

naive_skills      = simulate_training(rebalance=False)
rebalanced_skills = simulate_training(rebalance=True)
oracle_skills     = {task: ORACLE for task in TASKS}


def headroom_pct(skills: dict) -> float:
    """Fraction of available headroom (oracle - baseline) captured, as a percentage."""
    gains = [
        (skills[t] - BASELINE) / (ORACLE - BASELINE)
        for t in TASKS
    ]
    return float(np.mean(gains)) * 100


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------

SEP = "=" * 62
print(SEP)
print("Open-MOPD PoC: Token Budget Misallocation Simulation")
print(SEP)

print("\n[1] Task configuration:")
for task, cfg in TASKS.items():
    print(f"  {task:<25} avg_len={cfg['avg_tokens']:>4} tokens")

print(f"\n[2] Results after {N_STEPS} training steps:")
print(f"\n  {'Task':<25} {'Oracle':>8} {'Naive':>8} {'Rebalanced':>12}")
print("  " + "-" * 57)
for task in TASKS:
    print(
        f"  {task:<25} "
        f"{oracle_skills[task]:>8.2f} "
        f"{naive_skills[task]:>8.2f} "
        f"{rebalanced_skills[task]:>12.2f}"
    )

naive_pct = headroom_pct(naive_skills)
fixed_pct = headroom_pct(rebalanced_skills)

print(f"\n[3] Average headroom captured:")
print(f"  Naive M-OPD   : {naive_pct:.1f}%  (paper reports ~35.6% on real models)")
print(f"  Rebalanced    : {fixed_pct:.1f}%")

print("\n[4] Per-task budget allocation (one representative step):")
rep_raw, rep_norm = {}, {}
for task, cfg in TASKS.items():
    lengths = sample_lengths(cfg, BATCH_SIZE)
    raw = float(lengths.sum()) / BATCH_SIZE
    rep_raw[task] = raw
    rep_norm[task] = raw / cfg["avg_tokens"]

total_raw  = sum(rep_raw.values())
total_norm = sum(rep_norm.values())
print(f"\n  {'Task':<25} {'Naive %':>10} {'Rebalanced %':>14}")
print("  " + "-" * 51)
for task in TASKS:
    pn = rep_raw[task]  / total_raw  * 100
    pf = rep_norm[task] / total_norm * 100
    print(f"  {task:<25} {pn:>9.1f}% {pf:>13.1f}%")

ratio = TASKS["math_reasoning"]["avg_tokens"] / TASKS["instruction_following"]["avg_tokens"]
print(f"\n[5] Key takeaway:")
print(f"  math_reasoning ({TASKS['math_reasoning']['avg_tokens']} tokens avg) gets")
print(f"  ~{ratio:.0f}x more gradient budget than instruction_following"
      f" ({TASKS['instruction_following']['avg_tokens']} tokens avg)")
print(f"  in naive M-OPD even though both tasks matter equally.")
print(f"  Length-normalization closes the gap:")
print(f"  {naive_pct:.0f}% headroom captured → {fixed_pct:.0f}% headroom captured.")
print()
print("  One-line fix in any training loop:")
print("    BEFORE: loss = (per_token_kl * mask).sum() / mask.sum()")
print("    AFTER:  loss = (per_token_kl * mask).sum(dim=-1)")
print("                       / mask.sum(dim=-1)        # mean per example")
print("            loss = loss.mean()                   # then mean over batch")
print("\n" + SEP)
