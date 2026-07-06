"""
CausalMix Demo — Data Mixture as Causal Inference for LLM Training
Based on: arXiv:2607.01104 (July 1, 2026)

Core idea: Treat domain mixing ratios as a "treatment" and data-pool
features as "covariates", then estimate the Conditional Average Treatment
Effect (CATE) so that optimal mix weights can be PREDICTED for any unseen
data pool without running new proxy experiments.

The key comparison:
  - RegMix: learns T -> Y on a fixed training pool, then applies that
            fixed mixture to new pools (ignores pool context X).
  - CausalMix: learns [X, T] -> Y across many pools, then optimises T
               given the specific features X of a new pool.

Dependencies: numpy, scikit-learn  (no GPU required)
Run:          pip install numpy scikit-learn
              python causal_mix_demo.py
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1.  Problem setup — 4 data domains, each described by 2 pool features
# ---------------------------------------------------------------------------
DOMAINS = ["code", "math", "web", "reasoning"]
N_DOMAINS = len(DOMAINS)
# Pool features per domain: [avg_quality, relative_size]
FEAT_DIM = N_DOMAINS * 2   # 8

def make_pool(quality: list, sizes: list) -> np.ndarray:
    """Build a normalised feature vector for one pool state."""
    sizes = np.array(sizes, dtype=float)
    sizes /= sizes.sum()
    feats = []
    for q, s in zip(quality, sizes):
        feats.extend([q, s])
    return np.array(feats)

def sample_mixtures(n: int) -> np.ndarray:
    return RNG.dirichlet(np.ones(N_DOMAINS), size=n)

# ---------------------------------------------------------------------------
# 2.  Ground-truth performance function (hidden from the learner)
#
#   perf = Σ_d  quality[d] * size[d] * mix[d]        (alignment bonus)
#          - 0.25 * mix[code] * (1 - quality[code])   (skill conflict w/ math)
#          - 0.15 * Σ_d |mix[d] - size[d]|            (mismatch penalty)
#          + noise
# ---------------------------------------------------------------------------
def true_perf(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    quality  = X[:, 0::2]   # (n, 4)
    dom_size = X[:, 1::2]   # (n, 4)
    alignment = (quality * dom_size * T).sum(axis=1)
    conflict  = 0.25 * T[:, 0] * (1.0 - quality[:, 0])
    mismatch  = 0.15 * np.abs(T - dom_size).sum(axis=1)
    noise     = RNG.normal(0, 0.015, size=X.shape[0])
    return alignment - conflict - mismatch + noise

# ---------------------------------------------------------------------------
# 3.  Simulate proxy experiments across MANY different training pools
#     (this is what makes CausalMix able to generalise — it sees varied X)
# ---------------------------------------------------------------------------
N_PROXY = 512

# Sample diverse pool states for the proxy phase
pool_qualities = RNG.uniform(0.3, 0.9, size=(N_PROXY, N_DOMAINS))
pool_sizes_raw = RNG.dirichlet(np.ones(N_DOMAINS) * 2, size=N_PROXY)
X_proxy = np.hstack([pool_qualities, pool_sizes_raw])
# Interleave quality/size so each domain's features are adjacent
X_proxy_interleaved = np.empty((N_PROXY, FEAT_DIM))
for d in range(N_DOMAINS):
    X_proxy_interleaved[:, 2*d]   = pool_qualities[:, d]
    X_proxy_interleaved[:, 2*d+1] = pool_sizes_raw[:, d]

T_proxy = sample_mixtures(N_PROXY)
Y_proxy = true_perf(X_proxy_interleaved, T_proxy)

print("=" * 62)
print("  CAUSALMIX DEMO  —  arXiv:2607.01104  (July 2026)")
print("=" * 62)
print(f"\n[Step 1] Proxy runs: {N_PROXY} experiments across diverse pools")
print(f"  Performance range: [{Y_proxy.min():.3f}, {Y_proxy.max():.3f}]")

# ---------------------------------------------------------------------------
# 4.  Train the models
#
#   CausalMix (S-learner): fit GBM on [X | T] -> Y
#   RegMix baseline:       fit GBM on T alone -> Y  (ignores pool features)
# ---------------------------------------------------------------------------
# --- CausalMix ---
XT = np.hstack([X_proxy_interleaved, T_proxy])
scaler = StandardScaler()
XT_scaled = scaler.fit_transform(XT)

causal_model = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.04,
    subsample=0.8, random_state=0
)
causal_model.fit(XT_scaled, Y_proxy)

# --- RegMix --- (ignores X; only learns what T worked on average)
regmix_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    random_state=1
)
regmix_model.fit(T_proxy, Y_proxy)

print(f"\n[Step 2] Models trained")
print(f"  CausalMix train R² = {causal_model.score(XT_scaled, Y_proxy):.3f}")
print(f"  RegMix    train R² = {regmix_model.score(T_proxy, Y_proxy):.3f}")

# ---------------------------------------------------------------------------
# 5.  Evaluate on TWO new pools never seen during training
# ---------------------------------------------------------------------------
N_CANDS = 30_000
T_cands = sample_mixtures(N_CANDS)

def best_mix_causal(X_vec: np.ndarray) -> tuple:
    """Find best T for pool X using CausalMix (pool-aware)."""
    X_rep = np.tile(X_vec, (N_CANDS, 1))
    XT_c = scaler.transform(np.hstack([X_rep, T_cands]))
    pred  = causal_model.predict(XT_c)
    best  = pred.argmax()
    return T_cands[best], pred[best]

def best_mix_regmix() -> tuple:
    """RegMix: pick whatever T it predicts as best (pool-unaware)."""
    pred = regmix_model.predict(T_cands)
    best = pred.argmax()
    return T_cands[best]

def oracle_mix(X_vec: np.ndarray) -> tuple:
    X_rep = np.tile(X_vec, (N_CANDS, 1))
    scores = true_perf(X_rep, T_cands)
    best   = scores.argmax()
    return T_cands[best], scores[best]

T_regmix = best_mix_regmix()

# New pool A: high-math-quality pool with math dominating the data
pool_A = make_pool(
    quality=[0.55, 0.92, 0.40, 0.70],   # math very high quality
    sizes  =[0.10, 0.55, 0.15, 0.20],   # math dominates
)

# New pool B: balanced high-quality pool skewed toward web+reasoning
pool_B = make_pool(
    quality=[0.80, 0.60, 0.85, 0.88],
    sizes  =[0.20, 0.15, 0.40, 0.25],
)

print("\n[Step 3] Optimal mixture search on shifted pools")
print("         (true performance evaluated via ground-truth function)\n")

header = f"  {'Method':<18} {'Perf':>8}   {'code':>6} {'math':>6} {'web':>6} {'reas':>6}"
sep    = "  " + "-" * 56

for pool_label, X_vec in [("Pool A (math-heavy)", pool_A), ("Pool B (web+reas)", pool_B)]:
    print(f"  ── {pool_label} ──")
    print(header)
    print(sep)

    T_uni = np.full(N_DOMAINS, 0.25)

    for label, T_mix in [
        ("Uniform",   T_uni),
        ("RegMix",    T_regmix),
    ]:
        perf = true_perf(X_vec.reshape(1,-1), T_mix.reshape(1,-1))[0]
        mix_s = " ".join(f"{v:6.2f}" for v in T_mix)
        print(f"  {label:<18} {perf:8.4f}   {mix_s}")

    T_cm, _  = best_mix_causal(X_vec.reshape(1,-1))
    perf_cm  = true_perf(X_vec.reshape(1,-1), T_cm.reshape(1,-1))[0]
    mix_s = " ".join(f"{v:6.2f}" for v in T_cm)
    print(f"  {'CausalMix':<18} {perf_cm:8.4f}   {mix_s}")

    T_or, perf_or = oracle_mix(X_vec.reshape(1,-1))
    mix_s = " ".join(f"{v:6.2f}" for v in T_or)
    print(f"  {'Oracle (true)':<18} {perf_or:8.4f}   {mix_s}")

    # Gap analysis
    perf_uni = true_perf(X_vec.reshape(1,-1), T_uni.reshape(1,-1))[0]
    perf_rm  = true_perf(X_vec.reshape(1,-1), T_regmix.reshape(1,-1))[0]
    total_gap = perf_or - perf_uni
    causal_gain = perf_cm - perf_rm
    print(f"\n  CausalMix vs RegMix:  +{causal_gain:.4f} performance gain")
    if total_gap > 0:
        print(f"  CausalMix recovers {100*causal_gain/total_gap:.1f}% of the gap from uniform to oracle")
    print()

# ---------------------------------------------------------------------------
# 6.  CATE interpretation — what the causal model learned
# ---------------------------------------------------------------------------
print("[Step 4] CATE interpretation — marginal effect of +0.10 allocation")
print("         to each domain, evaluated on Pool A (math-heavy pool)\n")

X_ref = pool_A.reshape(1, -1)
T_base = np.full(N_DOMAINS, 0.25).reshape(1, -1)
base = causal_model.predict(scaler.transform(np.hstack([np.tile(X_ref, (1,1)), T_base])))[0]

print(f"  {'Domain':<12}  {'CATE (+0.10)':>14}  Direction")
print("  " + "-" * 38)
for i, domain in enumerate(DOMAINS):
    T_delta = np.full(N_DOMAINS, 0.25)
    T_delta[i] += 0.10
    T_delta /= T_delta.sum()
    pred = causal_model.predict(
        scaler.transform(np.hstack([X_ref, T_delta.reshape(1,-1)]))
    )[0]
    effect = pred - base
    bar = ("▲ " if effect >= 0 else "▼ ") + "#" * int(abs(effect) * 200)
    print(f"  {domain:<12}  {effect:>+14.4f}  {bar}")

print("\n[Done]")
print("  CausalMix learns WHICH mixture is best for EACH pool state,")
print("  not just on average — enabling generalisation without re-running")
print("  expensive small-model proxy sweeps on the new data pool.")
