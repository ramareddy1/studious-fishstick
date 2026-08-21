# Weekly AI Insight — 2026-08-21

## Title
**Open-MOPD: Diagnosing and Fixing Capability Imbalance in Multi-Teacher On-Policy Distillation**

## Source
- **Paper:** [arXiv:2608.19098](https://arxiv.org/abs/2608.19098)
- **Submitted:** August 19, 2026
- **Related prior work:** [MOPD (arXiv:2606.30406)](https://arxiv.org/abs/2606.30406)

---

## Why It Matters

Multi-teacher on-policy distillation (M-OPD) is the training paradigm behind the next generation of generalist AI models: run separate expert models for math, coding, reasoning, and instruction-following, then distill them all into a single student. In theory, the student inherits every expert's ability. In practice, something silently goes wrong.

**Open-MOPD** (arXiv:2608.19098) is the first rigorous open diagnosis of this failure. The paper's central finding is alarming: standard M-OPD captures only **35.6% of available headroom** relative to a perfectly-routed specialist ensemble — and the root cause is not gradient conflict between teachers (the community's prior assumption), but a structural bias baked into the training objective itself.

### The Token Budget Problem

In on-policy distillation the optimization signal is a per-token KL divergence: the model gets credit for matching each teacher's output, one token at a time. Tasks that produce long outputs — step-by-step math reasoning (hundreds of tokens) — naturally contribute far more gradient signal than tasks with short, concise outputs — instruction following (tens of tokens). The math teacher shouts; the instruction-following teacher whispers. Over thousands of steps, the student learns to shout back and forgets to listen.

### The Fix

The solution the paper proposes is conceptually simple: **normalize each example's training signal by its sequence length** before averaging. This reallocates the optimization budget equitably across task types regardless of verbosity. The controlled benchmark on SmolLM3-3B-Base (a model small enough for academic use) shows the fix substantially closes the capability gap.

### Why This Matters Beyond LLM Labs

This finding generalizes to any multi-task fine-tuning scenario, not just industrial-scale M-OPD:

| Scenario | Implication |
|---|---|
| Fine-tuning on mixed-length datasets | Normalize loss by sequence length to prevent verbose tasks from drowning out concise ones |
| LoRA on a mix of summarization + classification | Short classification examples need their own budget protection |
| RL with mixed-length rollouts | Per-rollout reward should be length-normalized |
| Instruction tuning on multi-domain corpora | Verbose instruction types inflate their own gradient share |

---

## Implementation Details

### Existing GitHub Repositories

| Resource | Description |
|---|---|
| [chrisliu298/awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) | Curated collection of M-OPD papers, tools and frameworks |
| [MOPD original (arXiv:2606.30406)](https://arxiv.org/abs/2606.30406) | The foundational M-OPD paper from June 2026 |

No official Open-MOPD code release was available at time of writing (August 21, 2026).

### Proof-of-Concept Script

See [`open_mopd_poc.py`](./open_mopd_poc.py) in this folder.

**Quick start (no GPU, no API key):**
```bash
pip install numpy   # only dependency
python open_mopd_poc.py
```

**What the script does:**

1. Defines two synthetic teachers — one long-sequence (math, avg 150 tokens) and one short-sequence (instruction following, avg 20 tokens)
2. Simulates 100 training steps of naive M-OPD, tracking how skill in each task evolves
3. Simulates the same training with length-normalized budget allocation
4. Prints per-task skill scores, headroom percentages, and the budget split each method gives to each task

**Sample output:**
```
==============================================================
Open-MOPD PoC: Token Budget Misallocation Simulation
==============================================================

[1] Task configuration:
  math_reasoning            avg_len= 150 tokens
  instruction_following     avg_len=  20 tokens

[2] Results after training simulation:
  Task                       Oracle    Naive   Rebalanced
  ---------------------------------------------------------
  math_reasoning               1.00     0.89         0.95
  instruction_following        1.00     0.47         0.94

[3] Average headroom captured:
  Naive M-OPD   : 36.2%  (paper reports ~35.6% on real models)
  Rebalanced    : 74.8%

[4] Per-task budget allocation (naive vs rebalanced):
  Task                       Naive %  Rebalanced %
  ---------------------------------------------------
  math_reasoning               88.2%          50.3%
  instruction_following        11.8%          49.7%

[5] Key takeaway:
  In naive M-OPD the math_reasoning task (150 tokens avg) gets
  ~7.5x more gradient budget than instruction_following (20 tokens)
  even though both tasks are equally important.
  Length-normalization restores balanced budget allocation and
  improves capability capture from 36% → 75%.

  Apply this fix to any multi-task fine-tuning pipeline by
  dividing each example's loss by its sequence length before
  averaging across the batch.
==============================================================
```

### Applying the Fix to a Real Training Loop

The key change is a single line in your loss computation:

```python
# BEFORE (naive — verbose tasks dominate)
loss = (per_token_kl * mask).sum() / mask.sum()

# AFTER (length-normalized — equal budget per example)
loss = (per_token_kl * mask).sum(dim=-1) / mask.sum(dim=-1)  # per-example mean
loss = loss.mean()                                             # then batch mean
```

This change is compatible with any standard training framework (Hugging Face Trainer, PyTorch Lightning, etc.) and adds zero overhead.
