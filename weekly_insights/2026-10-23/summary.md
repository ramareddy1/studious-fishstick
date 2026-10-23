# Weekly AI Insight — 2026-10-23

## Title
**D-Skill: Distilling Teacher Knowledge through Agent Skills**

## Source
- **arXiv:** https://arxiv.org/abs/2610.28914
- **Authors:** Shuo Yuan, Dacheng Li, Joseph E. Gonzalez
- **Submitted:** October 20, 2026
- **GitHub:** No official repository found as of 2026-10-23 (see proof-of-concept below)

---

## Why It Matters

Running a large, expensive "teacher" model for every query is often unnecessary — but fine-tuning a small "student" model to match it requires GPU training, labeled data, and weeks of work. **D-Skill sidesteps both problems entirely.**

The core insight: a model's behavioral advantage over a cheaper competitor can be extracted as a *natural-language document* by comparing their rollouts side-by-side. Wherever the teacher succeeds and the student fails, D-Skill logs the behavioral difference, distils the pattern into a readable "skill document", and injects it into the student's prompt at inference time — **no weight updates needed on either model**.

### Key Results (paper)
| Setup | Teacher | Student baseline | Student + D-Skill | Gap closed |
|---|---|---|---|---|
| SpreadsheetBench-Verified | GPT-5.6 Terra | 47% | 84% | **84%** |
| ALFWorld | GPT-5.6 Terra | 61% | 94% | **87%** |
| Chess puzzles | GPT-5.6 Terra | 52% | 89% | **77%** |
- Inference cost with D-Skill: **5.7–8.3× cheaper** than running the teacher on every query
- Works across any two models — the student doesn't need to share a family with the teacher

### Why now?
D-Skill generalises a pattern that has emerged independently in several recent works (SFT-as-Context, Socratic Skill Learning), but it is the first to make the skill document the *primary artifact* — readable, editable, and reusable across tasks. A student who is "too cheap" for a hard problem can receive a skill document from a one-time teacher consultation and handle the same category of problem indefinitely.

This matters because:
1. **No training infrastructure** — works with any hosted API or local model
2. **Interpretable** — the skill document is plain text; you can read, audit, and edit it
3. **Composable** — a library of skill documents can be mixed and matched at runtime
4. **Model-agnostic** — tested across three teacher-student pairs and five task domains

---

## Implementation Details

### Method Summary
1. **Collect paired rollouts**: run teacher and student on a set of representative tasks
2. **Find divergence**: keep only cases where teacher succeeds and student fails
3. **Distil skill document**: ask the teacher (or generate heuristically) "what behavioral pattern explains why you succeeded where the student failed?"
4. **Deploy**: at inference time, prepend the skill document to the student's system prompt

### Existing GitHub Repository
No official repository was found at time of writing (2026-10-23).

### Proof-of-Concept Script
See `d_skill_poc.py` in this folder. Runs with **Python built-ins only** — no GPU, no API key, no dependencies.

**Quick start:**
```bash
python d_skill_poc.py
```

**What the script demonstrates:**
1. Simulates teacher and student agents on 80 synthetic tasks across 4 categories
2. Collects pilot rollouts (40 tasks) and identifies the divergence cases
3. Distils the divergence into a natural-language skill document
4. Evaluates the student on 40 held-out tasks — once without, once with the skill document
5. Reports gap closure percentage and inference cost comparison

**Sample output:**
```
Phase 3 — Held-out evaluation (40 tasks)
  Teacher accuracy          : 100.0%
  Student (no skill doc)    : 47.5%
  Student (+ skill doc)     : 90.0%

  Gap before D-Skill        : 52.5%
  Gap after  D-Skill        : 10.0%
  Gap closed                : 81.0%  (paper reports 75–95%)

Phase 4 — Inference cost comparison (normalised units)
  Teacher cost per query          : 8.3×
  Student + skill doc cost        : 1.15×
  Cost saving                     : 7.2× cheaper
```

---

## Practical Notes for Real Usage

**Step 1 — Collect rollouts (one-time, ~50 tasks is sufficient):**
```python
# Run teacher and student on the same prompts, record outcomes
teacher_traces = [teacher_model(q) for q in tasks]
student_traces = [student_model(q) for q in tasks]
```

**Step 2 — Identify divergence and distil the skill document:**
```python
divergence = [
    (q, t, s) for q, t, s in zip(tasks, teacher_traces, student_traces)
    if is_correct(t) and not is_correct(s)
]

# Call the teacher once with a meta-prompt:
meta_prompt = f"""
Here are {len(divergence)} tasks where you succeeded but a smaller model failed.
Summarise in plain language what behavioral patterns explain your success,
as actionable guidance for the smaller model.
Divergence examples:
{format_divergence(divergence[:10])}
"""
skill_document = teacher_model(meta_prompt)
```

**Step 3 — Deploy cheaply:**
```python
system_prompt = f"""
You are a helpful assistant. Apply the following skill guidance:
{skill_document}
"""
# All future queries use the cheap student model + skill prefix
response = student_model(system_prompt + user_query)
```

**Tips:**
- One skill document per task category (math, code, reasoning) works better than one universal document
- Refresh the skill document every few weeks as the student model updates
- The skill document is human-readable — review it before deploying to catch hallucinations
