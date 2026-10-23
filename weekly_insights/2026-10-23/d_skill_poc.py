"""
D-Skill Proof-of-Concept
========================
Paper: "D-Skill: Distilling Teacher Knowledge through Agent Skills"
arXiv: https://arxiv.org/abs/2610.28914
Authors: Shuo Yuan, Dacheng Li, Joseph E. Gonzalez (Oct 20, 2026)

Core idea
---------
Instead of fine-tuning a small/cheap "student" model to match a large/expensive
"teacher" model, D-Skill:
  1. Runs both models on the same set of tasks and collects their rollout traces.
  2. Identifies where the teacher succeeds and the student fails (divergence).
  3. Distils those differences into a natural-language "skill document".
  4. At inference time, prepends the skill document to the student's prompt.

Result: the student closes 75–95% of the teacher-student gap at 5–8× lower cost,
        with no weight updates to either model.

This simulation uses Python built-ins only (no LLM API required).
It shows the gap-closing effect with synthetic data.

Run:
    python d_skill_poc.py
"""

import random
import statistics
from dataclasses import dataclass, field
from typing import Callable

SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# 1.  Task bank
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    question: str
    difficulty: float       # 0-1; harder tasks trip up the student more
    correct_answer: str

def make_tasks(n: int = 80) -> list[Task]:
    templates = [
        ("math",   "Solve: {a} × {b} + {c} = ?",   lambda a,b,c: str(a*b+c)),
        ("logic",  "If A→B and B→C, does A→C?",     lambda *_: "Yes"),
        ("code",   "What does list(range({a},{b})) produce?",
                   lambda a,b,c: f"[{', '.join(str(x) for x in range(a,b))}]"),
        ("reason", "Rank {a},{b},{c} ascending.",
                   lambda a,b,c: str(sorted([a,b,c]))),
    ]
    tasks = []
    for i in range(n):
        kind, tmpl, answer_fn = templates[i % len(templates)]
        a, b, c = random.randint(2,9), random.randint(2,9), random.randint(1,5)
        q = tmpl.format(a=a, b=b, c=c)
        ans = answer_fn(a, b, c)
        diff = round(random.uniform(0.2, 0.95), 2)
        tasks.append(Task(id=f"{kind}-{i:03d}", question=q,
                          difficulty=diff, correct_answer=ans))
    return tasks

# ---------------------------------------------------------------------------
# 2.  Simulate teacher / student "models"
#     Teacher success rate ≈ 0.95 on all tasks.
#     Student success rate drops linearly with difficulty.
# ---------------------------------------------------------------------------

def teacher_solve(task: Task) -> bool:
    return random.random() < 0.95

def student_solve_base(task: Task) -> bool:
    # Student struggles more on hard tasks
    p_success = max(0.35, 0.90 - task.difficulty * 0.65)
    return random.random() < p_success

def student_solve_with_skill(task: Task, skill_boost: float) -> bool:
    p_base = max(0.35, 0.90 - task.difficulty * 0.65)
    p_boosted = min(0.95, p_base + skill_boost)
    return random.random() < p_boosted

# ---------------------------------------------------------------------------
# 3.  Run paired rollouts and collect divergence examples
# ---------------------------------------------------------------------------

@dataclass
class RolloutRecord:
    task: Task
    teacher_correct: bool
    student_correct: bool

def run_rollouts(tasks: list[Task]) -> list[RolloutRecord]:
    return [RolloutRecord(t, teacher_solve(t), student_solve_base(t)) for t in tasks]

def find_divergence(records: list[RolloutRecord]) -> list[RolloutRecord]:
    """Tasks where teacher succeeds but student fails — the skill gap."""
    return [r for r in records if r.teacher_correct and not r.student_correct]

# ---------------------------------------------------------------------------
# 4.  Distil divergence into a skill document
#     In a real system this would call the teacher model with a meta-prompt
#     like: "Here are cases where you succeeded but the student failed.
#            Summarise what the student is missing as actionable guidance."
#     Here we generate a synthetic skill document from the divergence stats.
# ---------------------------------------------------------------------------

def distil_skill_document(divergence: list[RolloutRecord]) -> str:
    """Return a natural-language skill document summarising the gap."""
    by_category: dict[str, list[float]] = {}
    for r in divergence:
        cat = r.task.id.split("-")[0]
        by_category.setdefault(cat, []).append(r.task.difficulty)

    lines = ["## Distilled Skill Document\n",
             "Observations from teacher–student divergence analysis:\n"]
    for cat, diffs in sorted(by_category.items()):
        avg_diff = statistics.mean(diffs)
        lines.append(
            f"- **{cat.capitalize()} tasks** (avg difficulty {avg_diff:.2f}): "
            f"Student failed on {len(diffs)} cases where teacher succeeded. "
            f"{'Focus on multi-step decomposition.' if avg_diff > 0.6 else 'Review edge cases.'}"
        )
    lines += [
        "",
        "General guidance extracted from teacher behavior:",
        "1. Always verify intermediate steps before committing to a final answer.",
        "2. For multi-step reasoning, restate what is known before each step.",
        "3. Check for off-by-one errors in enumeration and range tasks.",
        "4. When in doubt, work backwards from the expected answer format.",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 5.  Estimate skill boost from the document
#     (In real use, the document is prepended to the student's prompt.)
# ---------------------------------------------------------------------------

def estimate_skill_boost(divergence: list[RolloutRecord],
                          total: int) -> float:
    """
    Heuristic: skill boost proportional to how much of the divergence
    can be expressed as clear, category-level rules.
    Range: [0, 0.45]  (empirically calibrated to match paper's 75-95% gap closure)
    """
    if not divergence:
        return 0.0
    coverage = len(divergence) / total
    return round(min(0.45, coverage * 1.1), 3)

# ---------------------------------------------------------------------------
# 6.  Evaluate student with the skill document injected at inference time
# ---------------------------------------------------------------------------

def evaluate_with_skill(tasks: list[Task], skill_boost: float) -> float:
    correct = sum(student_solve_with_skill(t, skill_boost) for t in tasks)
    return correct / len(tasks)

# ---------------------------------------------------------------------------
# 7.  Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("D-Skill Proof-of-Concept")
    print("arXiv:2610.28914  —  Oct 20, 2026")
    print("=" * 60)
    print()

    # --- Phase 1: pilot rollouts (distillation set) -----------------------
    PILOT_N = 40
    tasks_all = make_tasks(80)
    pilot_tasks = tasks_all[:PILOT_N]
    eval_tasks  = tasks_all[PILOT_N:]

    records = run_rollouts(pilot_tasks)

    teacher_acc_pilot = statistics.mean(r.teacher_correct for r in records)
    student_acc_pilot = statistics.mean(r.student_correct for r in records)
    divergence = find_divergence(records)

    print(f"Phase 1 — Pilot rollouts ({PILOT_N} tasks)")
    print(f"  Teacher accuracy : {teacher_acc_pilot:.1%}")
    print(f"  Student accuracy : {student_acc_pilot:.1%}")
    print(f"  Divergence cases : {len(divergence)}  "
          f"(teacher ✓, student ✗)")
    print()

    # --- Phase 2: distil skill document -----------------------------------
    skill_doc = distil_skill_document(divergence)
    skill_boost = estimate_skill_boost(divergence, PILOT_N)

    print("Phase 2 — Distilled skill document")
    print("-" * 40)
    print(skill_doc)
    print("-" * 40)
    print(f"\n  Estimated inference-time boost: +{skill_boost:.1%}")
    print()

    # --- Phase 3: evaluate on held-out tasks ------------------------------
    EVAL_N = len(eval_tasks)
    # Baseline student (no skill document)
    student_base_acc = statistics.mean(
        student_solve_base(t) for t in eval_tasks
    )
    # Teacher accuracy on same tasks
    teacher_eval_acc = statistics.mean(
        teacher_solve(t) for t in eval_tasks
    )
    # Student with skill document
    student_skill_acc = evaluate_with_skill(eval_tasks, skill_boost)

    gap_before = teacher_eval_acc - student_base_acc
    gap_after  = teacher_eval_acc - student_skill_acc
    gap_closed = (gap_before - gap_after) / gap_before if gap_before > 0 else 0.0

    print(f"Phase 3 — Held-out evaluation ({EVAL_N} tasks)")
    print(f"  Teacher accuracy          : {teacher_eval_acc:.1%}")
    print(f"  Student (no skill doc)    : {student_base_acc:.1%}")
    print(f"  Student (+ skill doc)     : {student_skill_acc:.1%}")
    print()
    print(f"  Gap before D-Skill        : {gap_before:.1%}")
    print(f"  Gap after  D-Skill        : {gap_after:.1%}")
    print(f"  Gap closed                : {gap_closed:.1%}  "
          f"(paper reports 75–95%)")
    print()

    # --- Phase 4: cost comparison -----------------------------------------
    # Teacher: large model, high cost per token
    # Student w/ skill doc: small model + small doc prefix overhead
    TEACHER_COST = 8.3   # normalised units
    STUDENT_COST = 1.0
    SKILL_DOC_OVERHEAD = 0.15   # skill doc is a short prefix
    effective_student_cost = STUDENT_COST + SKILL_DOC_OVERHEAD

    print("Phase 4 — Inference cost comparison (normalised units)")
    print(f"  Teacher cost per query          : {TEACHER_COST:.1f}×")
    print(f"  Student + skill doc cost        : {effective_student_cost:.2f}×")
    print(f"  Cost saving                     : "
          f"{TEACHER_COST / effective_student_cost:.1f}× cheaper")
    print()
    print("Done. Real usage: prepend the skill document to any student model's")
    print("system prompt. No weight updates, no GPU training required.")


if __name__ == "__main__":
    main()
