"""
DRPO — Divergence Regularized Policy Optimization (toy demo)
Paper: "Rethinking the Divergence Regularization in LLM RL"
arXiv: 2606.09821  |  Tencent Hunyuan / UIUC / NUS  |  June 2026

Core idea of the paper:
  Modern LLM RL post-training (PPO / GRPO) uses a HARD MASK: when the
  probability ratio pi_new(a)/pi_old(a) drifts outside a trust region
  [1-eps, 1+eps], the gradient for that token is simply zeroed out.
  This throws away the learning signal for exactly the tokens that are
  changing the most.

  DRPO replaces that hard mask with a SMOOTH, ADVANTAGE-WEIGHTED
  QUADRATIC REGULARIZER on the policy shift (ratio - 1). It keeps the
  same trust-region geometry (small, well-behaved updates near ratio=1)
  but produces a bounded, continuous, non-zero gradient everywhere —
  including outside the old trust region, where it acts as a corrective
  pull back toward the old policy instead of a dead zone.

This script is a *pedagogical, simplified* re-implementation of that idea
on a toy K-armed bandit (stand-in for a single-token decision in an LLM).
It is NOT the paper's exact formula (the full paper was behind a 403 at
fetch time) but captures the structural difference that matters:

  PPO-clip :  grad = 0                              outside the trust region
  DRPO     :  grad = ratio * (A - 2*lambda*|A|*(ratio-1))   everywhere

Dependency: numpy only. Runs in well under a second on a laptop CPU.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Toy environment: a K-armed bandit. Each "arm" stands in for one token in a
# vocabulary; TRUE_REWARDS stands in for the reward model's preference.
# ---------------------------------------------------------------------------
K = 5
TRUE_REWARDS = np.array([0.2, 0.9, 0.5, 0.1, 0.4])

EPS = 0.1          # PPO trust region half-width (tight -> clipping kicks in fast)
LAMBDA = 2.0       # DRPO regularization strength
LR = 1.0           # step size (aggressive, to push ratios outside the trust region)
BATCH = 64         # samples per rollout
INNER_EPOCHS = 20  # gradient reuse steps per rollout (off-policy degree)
OUTER_ITERS = 60   # number of rollouts
SEEDS = 5          # average over multiple random seeds


def softmax(logits):
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def sample_rollout(logits, rng):
    """Sample a batch of actions from the *old* (frozen) policy."""
    probs = softmax(logits)
    actions = rng.choice(K, size=BATCH, p=probs)
    rewards = TRUE_REWARDS[actions] + rng.normal(0, 0.05, BATCH)
    advantages = rewards - rewards.mean()
    old_probs = probs[actions]
    return actions, advantages, old_probs


def grad_logp(logits, actions):
    """d log pi(a) / d logits_j = 1[j==a] - pi(j), for a softmax policy."""
    probs = softmax(logits)
    onehots = np.eye(K)[actions]
    return onehots - probs, probs


def ppo_step(logits, actions, advantages, old_probs):
    """PPO-clip surrogate gradient — the 'hard mask' baseline."""
    grad_lp, probs = grad_logp(logits, actions)
    ratio = probs[actions] / old_probs
    clipped = np.clip(ratio, 1 - EPS, 1 + EPS)

    surr_unclipped = ratio * advantages
    surr_clipped = clipped * advantages

    # Take the pessimistic (min) surrogate. When the clipped term wins,
    # the ratio is treated as a constant -> zero gradient from that sample.
    use_unclipped = surr_unclipped <= surr_clipped
    coeff = np.where(use_unclipped, ratio * advantages, 0.0)
    dead_fraction = 1.0 - use_unclipped.mean()

    total_grad = (grad_lp * coeff[:, None]).mean(axis=0)
    return total_grad, dead_fraction


def drpo_step(logits, actions, advantages, old_probs):
    """DRPO: smooth advantage-weighted quadratic regularizer, no hard mask."""
    grad_lp, probs = grad_logp(logits, actions)
    ratio = probs[actions] / old_probs

    # Continuous correction term: pulls ratio back toward 1, scaled by how
    # large the advantage is, instead of clipping it to zero.
    reg = -2 * LAMBDA * np.abs(advantages) * (ratio - 1)
    coeff = ratio * (advantages + reg)

    total_grad = (grad_lp * coeff[:, None]).mean(axis=0)
    return total_grad, 0.0  # never "dead" — always a gradient


def run(method, rng):
    logits = np.zeros(K)
    reward_history = []
    dead_fractions = []

    for _ in range(OUTER_ITERS):
        actions, advantages, old_probs = sample_rollout(logits, rng)

        for _ in range(INNER_EPOCHS):
            if method == "ppo":
                grad, dead = ppo_step(logits, actions, advantages, old_probs)
            else:
                grad, dead = drpo_step(logits, actions, advantages, old_probs)
            logits = logits + LR * grad
            dead_fractions.append(dead)

        reward_history.append(float(softmax(logits) @ TRUE_REWARDS))

    return np.array(reward_history), np.mean(dead_fractions)


def main():
    optimal_reward = TRUE_REWARDS.max()
    print("=" * 70)
    print("  DRPO vs PPO-clip — toy policy-gradient comparison")
    print("  arXiv:2606.09821 (June 2026): smooth regularizer vs hard mask")
    print("=" * 70)
    print(f"  {K}-armed bandit, true rewards = {TRUE_REWARDS.tolist()}")
    print(f"  optimal achievable expected reward = {optimal_reward:.3f}")
    print(f"  trust region eps = {EPS}, DRPO lambda = {LAMBDA}, "
          f"{INNER_EPOCHS} reused gradient steps per rollout")
    print()

    results = {}
    for method in ("ppo", "drpo"):
        all_histories = []
        dead_fracs = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            history, dead = run(method, rng)
            all_histories.append(history)
            dead_fracs.append(dead)
        results[method] = (np.mean(all_histories, axis=0), np.mean(dead_fracs))

    for method, (history, dead) in results.items():
        label = "PPO-clip (hard mask)" if method == "ppo" else "DRPO (smooth reg.)"
        print(f"  {label:24s} final reward = {history[-1]:.3f}   "
              f"(gap to optimal: {optimal_reward - history[-1]:.3f})   "
              f"zero-gradient steps: {dead*100:5.1f}%")

    print()
    print("  Reward over training (every 10th iter, averaged over "
          f"{SEEDS} seeds):")
    for method, (history, _) in results.items():
        label = "PPO " if method == "ppo" else "DRPO"
        sampled = " ".join(f"{r:.2f}" for r in history[::10])
        print(f"    {label}: {sampled}")

    print()
    print("  TAKEAWAY")
    print("  --------")
    print("  With many reused gradient steps per rollout (INNER_EPOCHS="
          f"{INNER_EPOCHS}), PPO-clip zeroes out a large fraction of")
    print("  per-step gradients once the policy drifts outside the trust")
    print("  region, stalling learning on those samples. DRPO's smooth")
    print("  quadratic regularizer keeps every sample's gradient alive,")
    print("  pulling divergent updates back continuously -> faster, more")
    print("  stable convergence to the optimal arm under heavy gradient reuse.")


if __name__ == "__main__":
    main()
