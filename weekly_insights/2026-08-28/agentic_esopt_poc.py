"""
Proof-of-Concept: Evolution Strategies (ES) for LLM Agent Fine-Tuning
Based on: Agentic ESOpt (arXiv:2608.17310)

Demonstrates the core antithetic ES update loop used to fine-tune LLM
agents without backpropagation — no GPU required.

Install:  pip install numpy
Run:      python agentic_esopt_poc.py

Full implementation: https://github.com/zz1358m/Agentic-ESOpt
"""

import numpy as np

rng = np.random.default_rng(42)

# ── Fixed task definition ─────────────────────────────────────────────────────
# The "environment" has a learnable ground-truth: action = argmax of W_task @ x.
# A perfect agent would learn W_task. We hold W_task fixed throughout training.
# This gives a stable reward landscape that ES can actually ascend.

STATE_DIM  = 16
N_ACTIONS  = 4
HORIZON    = 20   # steps per episode

W_task = rng.standard_normal((N_ACTIONS, STATE_DIM))   # fixed, unknown to agent
STATES = rng.standard_normal((200, STATE_DIM))          # fixed episode state pool


# ── Toy "agent" model ─────────────────────────────────────────────────────────
# Two linear layers + ReLU. In real ESOpt this is a 4B–27B LLM.

class ToyAgent:
    def __init__(self):
        self.W1 = rng.standard_normal((32, STATE_DIM)) * 0.05
        self.b1 = np.zeros(32)
        self.W2 = rng.standard_normal((N_ACTIONS, 32)) * 0.05
        self.b2 = np.zeros(N_ACTIONS)

    def flat_params(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in
                               (self.W1, self.b1, self.W2, self.b2)])

    def load_flat_params(self, theta: np.ndarray):
        shapes = [("W1", (32, STATE_DIM)), ("b1", (32,)),
                  ("W2", (N_ACTIONS, 32)), ("b2", (N_ACTIONS,))]
        i = 0
        for attr, shape in shapes:
            n = int(np.prod(shape))
            setattr(self, attr, theta[i: i + n].reshape(shape).copy())
            i += n

    def forward(self, x: np.ndarray) -> int:
        h = np.maximum(0, self.W1 @ x + self.b1)
        return int(np.argmax(self.W2 @ h + self.b2))


# ── Environment ───────────────────────────────────────────────────────────────

def correct_action(state: np.ndarray) -> int:
    """Ground-truth action the agent should learn to predict."""
    return int(np.argmax(W_task @ state))

def run_episode(agent: ToyAgent, episode_id: int) -> float:
    """Fixed episode (deterministic state sequence) → scalar reward ∈ [0,1]."""
    idx    = np.arange(episode_id * HORIZON, episode_id * HORIZON + HORIZON) % len(STATES)
    states = STATES[idx]
    hits   = sum(agent.forward(s) == correct_action(s) for s in states)
    return hits / HORIZON

def evaluate(agent: ToyAgent, n: int = 10) -> float:
    return float(np.mean([run_episode(agent, i) for i in range(n)]))


# ── Antithetic Evolution Strategies optimizer ─────────────────────────────────
# Mirrors Agentic ESOpt Algorithm 1:
#
#   For each step:
#     1. Sample k noise vectors ε_i ∈ ℝ^d
#     2. R⁺_i = evaluate( θ + σ·ε_i )    (positive perturbation)
#     3. R⁻_i = evaluate( θ − σ·ε_i )    (antithetic)
#     4. ĝ = 1/(2kσ) · Σ_i (R⁺_i − R⁻_i)·ε_i
#     5. θ ← θ + lr · ĝ
#
# All evaluations are FORWARD PASSES ONLY.
# No autograd · no .backward() · no gradient tape · no activation storage.

def es_step(agent: ToyAgent, sigma: float, lr: float,
            k: int, n_ep: int, step: int) -> float:
    theta = agent.flat_params()
    d     = theta.size
    grad  = np.zeros(d)
    rewards = []

    for j in range(k):
        eps = rng.standard_normal(d)

        agent.load_flat_params(theta + sigma * eps)
        r_pos = evaluate(agent, n=n_ep)

        agent.load_flat_params(theta - sigma * eps)
        r_neg = evaluate(agent, n=n_ep)

        grad    += (r_pos - r_neg) * eps
        rewards += [r_pos, r_neg]

    agent.load_flat_params(theta + lr * grad / (2 * k * sigma))
    return float(np.mean(rewards))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    agent    = ToyAgent()
    n_params = agent.flat_params().size

    SIGMA = 0.05
    LR    = 0.30
    K     = 8
    N_EP  = 8
    STEPS = 40

    print("=" * 60)
    print("Agentic ESOpt — Proof-of-Concept Demo")
    print("Paper: arXiv:2608.17310")
    print("Repo:  https://github.com/zz1358m/Agentic-ESOpt")
    print("=" * 60)
    print(f"Parameters : {n_params}")
    print(f"ES config  : sigma={SIGMA}, lr={LR}, k={K}, steps={STEPS}")
    print()

    baseline = evaluate(agent, n=20)
    print(f"Baseline reward (random init): {baseline:.4f}  (≈0.25 = random chance)")
    print()
    print(f"{'Step':>5}  {'Batch Reward':>14}  {'Eval Reward':>12}")
    print("-" * 38)

    for step in range(1, STEPS + 1):
        batch_r = es_step(agent, SIGMA, LR, K, N_EP, step)
        if step % 8 == 0:
            eval_r = evaluate(agent, n=20)
            print(f"{step:>5}  {batch_r:>14.4f}  {eval_r:>12.4f}")

    final = evaluate(agent, n=50)
    print("-" * 38)
    print(f"\nFinal reward after {STEPS} ES steps : {final:.4f}")
    print(f"Improvement over random init        : {final - baseline:+.4f}")
    print()
    print("Key insight demonstrated:")
    print("  The agent improved using ONLY forward passes — no autograd,")
    print("  no .backward(), no gradient tape, no GPU required.")
    print()
    print("Scaling to real LLMs:")
    print("  ┌─ Replace ToyAgent.forward()  → vLLM/transformers inference call")
    print("  ├─ Replace run_episode()       → your agentic task runner")
    print("  └─ es_step() logic stays identical regardless of model size")


if __name__ == "__main__":
    main()
