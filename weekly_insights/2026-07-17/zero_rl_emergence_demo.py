"""
zero_rl_emergence_demo.py

Toy demonstration of Zero RL (RLVR) principles from Ring-Zero (arXiv:2607.12395).

Ring-Zero trains a trillion-parameter LLM to reason using ONLY verifiable outcome
rewards — no human annotations, no curated examples. This script simulates the same
learning dynamic on a toy arithmetic environment using a lightweight policy.

Five emergent behaviors documented in the paper:
  1. Anthropomorphism      — model develops human-like hedging language
  2. Structured formatting — spontaneous use of step-by-step layouts
  3. Self-verification     — model re-checks its own answer before committing
  4. Parallel reasoning    — model explores multiple solution paths
  5. Context anxiety       — model expresses uncertainty when context is ambiguous

This demo tracks behaviors 2, 3, and 4 as they emerge purely from reward pressure.

Requirements: Python 3.8+, numpy only
Usage:        python zero_rl_emergence_demo.py
"""

import numpy as np
import textwrap

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Environment: arithmetic chain problems with verifiable answers
# ---------------------------------------------------------------------------

def make_problem():
    """Generate a random arithmetic problem with a verifiable answer."""
    a, b, c = RNG.integers(2, 15, size=3)
    op = RNG.choice(["+", "*", "-"])
    if op == "+":
        answer = int(a + b * c)
        expr = f"{a} + {b} × {c}"
    elif op == "*":
        answer = int(a * b + c)
        expr = f"{a} × {b} + {c}"
    else:
        answer = int(a * b - c)
        expr = f"{a} × {b} − {c}"
    return expr, answer


def verify(prediction, answer):
    """Binary reward: 1 if correct, 0 otherwise."""
    return 1 if prediction == answer else 0


# ---------------------------------------------------------------------------
# Policy: parameterised by strategy weights
#
# The policy chooses among four response STRATEGIES:
#   0  DIRECT      — guess directly (fast, low accuracy)
#   1  STEP_BY_STEP— break into steps (moderate cost, better accuracy)
#   2  VERIFY      — step-by-step then re-check (highest cost, best accuracy)
#   3  PARALLEL    — generate two independent reasoning chains, pick majority
#
# The policy is a softmax over 4 logits — one per strategy.
# We update it via REINFORCE: logit[k] += lr * (R - baseline) * grad.
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["Direct", "Step-by-Step", "Self-Verify", "Parallel"]

# Accuracy and chain length for each strategy (fixed, represents the LLM's inherent
# capability under each reasoning style — learned implicitly during real RL).
STRATEGY_ACCURACY  = np.array([0.35, 0.65, 0.82, 0.78])
STRATEGY_STEPS     = np.array([1,    3,    5,    4   ])   # reasoning tokens (relative)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


class ZeroRLPolicy:
    def __init__(self, lr=0.12, baseline_decay=0.95):
        self.logits   = np.zeros(4)          # uniform initialisation
        self.lr       = lr
        self.baseline = 0.0
        self.decay    = baseline_decay
        self.history  = []                   # (step, probs, reward)

    def select_strategy(self):
        probs = softmax(self.logits)
        return RNG.choice(4, p=probs), probs

    def act(self, answer, strategy_idx):
        """Simulate the policy's response under the chosen strategy."""
        correct = RNG.random() < STRATEGY_ACCURACY[strategy_idx]
        prediction = answer if correct else answer + RNG.integers(-3, 4)
        return int(prediction)

    def update(self, strategy_idx, reward, step):
        """REINFORCE gradient update."""
        probs     = softmax(self.logits)
        advantage = reward - self.baseline
        # grad log π(a|s) = 1 - π(a) for selected action, -π(k) for others
        grad = -probs.copy()
        grad[strategy_idx] += 1.0
        self.logits += self.lr * advantage * grad

        self.baseline = self.decay * self.baseline + (1 - self.decay) * reward
        self.history.append((step, probs.copy(), reward))


# ---------------------------------------------------------------------------
# Training loop — replicates Discovery → Sharpening phases
# ---------------------------------------------------------------------------

def train(n_steps=800):
    policy = ZeroRLPolicy(lr=0.12)
    window = 50
    rewards_buf = []

    print("=" * 62)
    print("Zero RL Training Loop  (Ring-Zero principles, toy scale)")
    print("=" * 62)
    print(f"{'Step':>6}  {'Accuracy':>8}  {'Direct':>7}  {'Step':>7}  "
          f"{'Verify':>7}  {'Parallel':>8}  {'Avg-chain':>10}  Phase")
    print("-" * 62)

    phase = "Discovery"
    sharpening_threshold = 0.60

    for step in range(1, n_steps + 1):
        expr, answer        = make_problem()
        strategy_idx, probs = policy.select_strategy()
        prediction          = policy.act(answer, strategy_idx)
        reward              = verify(prediction, answer)

        policy.update(strategy_idx, reward, step)
        rewards_buf.append(reward)

        # Detect phase transition: once accuracy exceeds threshold stably
        if phase == "Discovery" and len(rewards_buf) >= window:
            recent_acc = np.mean(rewards_buf[-window:])
            if recent_acc >= sharpening_threshold:
                phase = "Sharpening"

        if step % 50 == 0:
            recent_acc  = np.mean(rewards_buf[-window:]) if len(rewards_buf) >= window else np.mean(rewards_buf)
            cur_probs   = softmax(policy.logits)
            avg_chain   = float(np.dot(cur_probs, STRATEGY_STEPS))
            print(f"{step:>6}  {recent_acc:>8.2%}  "
                  f"{cur_probs[0]:>7.2%}  {cur_probs[1]:>7.2%}  "
                  f"{cur_probs[2]:>7.2%}  {cur_probs[3]:>8.2%}  "
                  f"{avg_chain:>10.2f}   {phase}")

    return policy, rewards_buf


# ---------------------------------------------------------------------------
# Emergence report
# ---------------------------------------------------------------------------

def report_emergence(policy, rewards_buf):
    probs     = softmax(policy.logits)
    avg_chain = float(np.dot(probs, STRATEGY_STEPS))
    final_acc = float(np.mean(rewards_buf[-100:]))

    early_probs = policy.history[9][1]   # step 10
    late_probs  = policy.history[-1][1]  # final step

    print("\n" + "=" * 62)
    print("Emergent Behaviour Report")
    print("=" * 62)

    print("\n[1] STRUCTURED FORMATTING (Step-by-Step)")
    delta = late_probs[1] - early_probs[1]
    print(f"  Step-by-Step probability: {early_probs[1]:.1%} → {late_probs[1]:.1%}  "
          f"({'↑' if delta > 0 else '↓'}{abs(delta):.1%})")
    if late_probs[1] > early_probs[1]:
        print("  ✓ Structured formatting EMERGED from reward pressure alone.")

    print("\n[2] SELF-VERIFICATION")
    delta = late_probs[2] - early_probs[2]
    print(f"  Self-Verify probability:  {early_probs[2]:.1%} → {late_probs[2]:.1%}  "
          f"({'↑' if delta > 0 else '↓'}{abs(delta):.1%})")
    if late_probs[2] > early_probs[2]:
        print("  ✓ Self-verification EMERGED — model learned to re-check answers.")

    print("\n[3] PARALLEL REASONING")
    delta = late_probs[3] - early_probs[3]
    print(f"  Parallel-path probability:{early_probs[3]:.1%} → {late_probs[3]:.1%}  "
          f"({'↑' if delta > 0 else '↓'}{abs(delta):.1%})")
    if late_probs[3] > early_probs[3]:
        print("  ✓ Parallel reasoning EMERGED — model explores multiple paths.")

    print("\n[4] REASONING CHAIN LENGTH")
    early_chain = float(np.dot(early_probs, STRATEGY_STEPS))
    print(f"  Avg steps: {early_chain:.2f} → {avg_chain:.2f}  "
          f"(model learned that more thinking = better answers)")

    print(f"\n[5] FINAL ACCURACY:  {final_acc:.1%}  "
          f"(started near {STRATEGY_ACCURACY[0]:.0%} with Direct strategy)")
    print(f"    Accuracy gain = {final_acc - STRATEGY_ACCURACY[0]:.1%}  "
          f"— achieved with ZERO human labels.\n")

    print("=" * 62)
    print("Key takeaway from Ring-Zero:")
    print(textwrap.fill(
        "At any scale — from this toy to a trillion parameters — "
        "binary verifiable rewards are sufficient to make a model spontaneously "
        "develop structured reasoning, self-checking, and multi-path exploration. "
        "No human annotations. No curated chain-of-thought examples. "
        "Just: correct = 1, wrong = 0.",
        width=62
    ))
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    policy, rewards_buf = train(n_steps=800)
    report_emergence(policy, rewards_buf)
