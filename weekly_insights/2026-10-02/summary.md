# Weekly AI Insight — 2026-10-02

## Title
**Diversity Combining for Multi-Path LLM Reasoning**

## Source
- **Paper:** [arXiv:2609.38829](https://arxiv.org/abs/2609.38829)
- **Venue:** NeurIPS 2026 (accepted)
- **Submitted:** September 30, 2026
- **Authors:** Guangsheng Yu, Litianyi Zhang, Qin Wang, Xu Wang, Mingyuan Li, Shaoxiong Ji, Ren Ping Liu, Massimo Piccardi

---

## Why It Matters

Self-consistency (SC) is today's go-to trick for getting better answers out of large language models: sample the same question K times, take the majority vote, and accuracy goes up. It's simple, it works — until it doesn't.

The problem is that SC silently hits a ceiling. Running K=32 paths instead of K=8 almost never helps as much as going from K=1 to K=4. The gains plateau fast. Most practitioners assume this is just Bayesian diminishing returns. **This paper proves it is something different and fixable.**

### The Root Cause: Path Correlation

The authors borrow a concept from wireless communications called **diversity combining**. In signal processing, if you have multiple noisy copies of the same transmission, they only add information if they are *independent* (uncorrelated). If all copies share the same noise pattern, you gain nothing from having more of them.

The same applies to LLM reasoning paths. When you use identical prompt templates, the model's internal reasoning trajectory is heavily correlated across samples — it makes the same kinds of errors on the same kinds of questions. The effective sample size is:

```
K_eff = K / (1 + (K-1) × ρ)
```

where ρ is the pairwise correlation among path outcomes. If ρ = 0.6 (typical for same-template SC), K_eff saturates near **1.67 regardless of K**. You are running 32 LLM calls and getting the statistical benefit of 1.67 independent answers.

### The Fix: Diverse Prompt Templates

The solution is straightforward: change the prompt template for each path. "Step by step, Q" vs "Think carefully: Q" vs "Using logical reasoning: Q" forces the model down different internal reasoning trajectories. Empirically, this reduces ρ from ~0.6 to ~0.1 across 5 models and 12 benchmarks — increasing K_eff from 1.67 to effectively 10 with just K=10 diverse paths.

### The Adaptive-K Rule

Even better, the paper provides a formula (Eq. 12) to compute the optimal K before doing full inference:

1. Run exactly **4 pilot paths** with diverse templates.
2. Estimate the pairwise correlation ρ from pilot outcomes.
3. Set K* = smallest K where K_eff ≥ 95% of its theoretical ceiling (1/ρ).

This selects K* with **no held-out data and no per-instance scorer**, retaining 96–103% of MV@K=32 accuracy across all benchmarks. The implication: a student on a free API tier can match an expensive K=32 run by choosing templates wisely and stopping at K*.

---

## Implementation Details

### Existing GitHub Repositories

No official implementation was available at time of writing (2026-10-02). The paper mentions a repository (`OniReimu/DiversityCombining`) but it had not been made public yet.

### Proof-of-Concept Script

See [`diversity_combining_poc.py`](./diversity_combining_poc.py) in this folder.

**Quick start — no dependencies, no API key:**
```bash
python3 diversity_combining_poc.py
```

**What the script demonstrates:**

1. Simulates K paths from a same-template strategy vs. diverse-template strategy over 200 questions.
2. Shows that same-template paths are near-perfectly correlated (ρ≈1.0) — more paths add zero accuracy.
3. Shows that diverse-template paths are uncorrelated (ρ≈0.0) — each path adds independent signal.
4. Implements the Adaptive-K rule: compute K* from a 4-path pilot via the K_eff formula.
5. Prints a table of accuracy vs K for both strategies.

**Sample output:**
```
  K    Same Template   Diverse Templates
  ----------------------------------------
  1           67.0%              67.0%
  4           67.0%              72.0%
  8           67.0%              80.5%
  32          67.0%              80.5%

  Gain K=2→K=32 (same template):    +0.0%
  Gain K=2→K=32 (diverse templates): +13.5%
```

### Applying This to Real LLM Inference

Only two changes are needed in any existing self-consistency pipeline:

**Step 1 — Swap repeated templates for diverse ones:**
```python
DIVERSE_TEMPLATES = [
    "Step by step, {q}",
    "Think carefully before answering: {q}",
    "Let's work through this systematically. {q}",
    "First, identify the key facts. Then answer: {q}",
    "Approach this from first principles: {q}",
    "Let's reason through this: {q}",
    "Break down the problem: {q}",
    "Using logical reasoning: {q}",
]

answers = []
for k, template in enumerate(DIVERSE_TEMPLATES[:K_star]):
    prompt = template.format(q=question)
    answers.append(llm(prompt))
final_answer = majority_vote(answers)
```

**Step 2 — Compute K* before full inference (save 60–80% of API calls):**
```python
import statistics

def adaptive_k(pilot_results: list[list[int]], max_k: int = 32) -> int:
    """pilot_results: list of [0/1] correctness lists, one per pilot path."""
    pairs = []
    for i in range(len(pilot_results)):
        for j in range(i+1, len(pilot_results)):
            xi, xj = pilot_results[i], pilot_results[j]
            mx, my = statistics.mean(xi), statistics.mean(xj)
            sx = statistics.stdev(xi) or 1e-9
            sy = statistics.stdev(xj) or 1e-9
            cov = sum((a-mx)*(b-my) for a,b in zip(xi,xj)) / len(xi)
            pairs.append(cov / (sx*sy))
    rho = statistics.mean(pairs)
    ceiling = 1.0 / max(rho, 1e-6)
    for k in range(1, max_k+1):
        k_eff = k / (1 + (k-1)*rho)
        if k_eff >= 0.95 * ceiling:
            return k
    return max_k
```
