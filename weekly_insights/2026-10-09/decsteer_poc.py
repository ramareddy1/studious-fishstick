"""
DecSteer Proof-of-Concept
Paper: "Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators
        via Low-Rank Activation Steering"
arXiv: https://arxiv.org/abs/2610.06950
Authors: Ran Li, Lei Chen (Oct 3, 2026)
GitHub:  https://github.com/rlisml/decisionsteer

Core idea: instead of letting an LLM reason for thousands of tokens,
train a tiny low-rank module (~23K params) that steers the model's hidden
activations at the final layer directly toward the right answer.
No RL required — pure behavior cloning from (question, answer) pairs.

Simulation design:
  A frozen LLM produces hidden states where:
  - Signal lives in a small "decision subspace" (first SIGNAL_DIM dims).
  - The remaining D_MODEL-SIGNAL_DIM dims are pure noise.
  This represents how real LLMs already encode the answer in a subspace but
  the full representation is noisy.  A rank-4 DecSteer operator learns to
  project out the noise and amplify the decision subspace.

Requires only numpy.  Run: python decsteer_poc.py
"""

import numpy as np
import random

# ── reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── hyper-parameters ──────────────────────────────────────────────────────────
D_MODEL    = 256   # total hidden dimension
SIGNAL_DIM = 16    # dims that carry the actual answer signal
N_LABELS   = 4     # answer choices (A/B/C/D)
RANK       = 4     # DecSteer bottleneck rank
N_TRAIN    = 1000
N_TEST     = 400
LR         = 0.08
EPOCHS     = 150
NOISE_STD  = 0.9   # noise added to non-signal dims; makes baseline ~50-60%


# ── ground-truth "decision subspace" ─────────────────────────────────────────
# The true answer lives in the first SIGNAL_DIM dimensions.
# The LLM head is a *random* full-D projection (frozen), so it can't cleanly
# separate signal from noise — it treats all D_MODEL dims equally.
SIGNAL_CENTROIDS = np.eye(SIGNAL_DIM, N_LABELS).T  # one-hot in signal subspace

# Frozen LLM head: a random D_MODEL→N_LABELS projection (not perfectly aligned)
HEAD_W = np.random.randn(N_LABELS, D_MODEL) * 0.1
HEAD_W[:, :SIGNAL_DIM] += SIGNAL_CENTROIDS   # partial alignment to signal dims


def frozen_lm_hidden(label: int) -> np.ndarray:
    """Simulate the frozen LLM's final hidden state.
    Signal in first SIGNAL_DIM dims; remaining dims are pure noise that
    overwhelms the head's ability to classify correctly."""
    h = np.zeros(D_MODEL)
    h[:SIGNAL_DIM] = SIGNAL_CENTROIDS[label] * 1.0   # signal
    h[SIGNAL_DIM:] = np.random.randn(D_MODEL - SIGNAL_DIM) * NOISE_STD
    return h


def head_predict(h: np.ndarray) -> int:
    """Frozen LLM head: argmax over a random linear projection of h."""
    return int(np.argmax(HEAD_W @ h))


# ── DecSteer: low-rank activation steering module ────────────────────────────
# Adds Δh = B @ (A @ h) to the frozen hidden state.
# A ∈ R^{rank × D},  B ∈ R^{D × rank}  →  2·rank·D trainable params.
# B is zero-initialized (from the paper: ensures identity behavior at step 0,
# which keeps gradients flowing from the very first update).

class DecSteer:
    def __init__(self, d: int = D_MODEL, rank: int = RANK):
        self.A  = np.random.randn(rank, d) * 0.02
        self.B  = np.zeros((d, rank))       # zero-init per paper
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

    @property
    def n_params(self) -> int:
        return self.A.size + self.B.size

    def forward(self, h: np.ndarray) -> np.ndarray:
        self._z = self.A @ h          # bottleneck activation (rank-vector)
        delta   = self.B @ self._z    # full-dim steering delta
        return h + delta

    def backward(self, h: np.ndarray, grad_out: np.ndarray) -> None:
        """Accumulate gradients for A and B."""
        self.dB += np.outer(grad_out, self._z)
        self.dA += np.outer(self.B.T @ grad_out, h)

    def step(self, lr: float) -> None:
        self.A -= lr * self.dA / N_TRAIN
        self.B -= lr * self.dB / N_TRAIN
        self.dA[:] = 0
        self.dB[:] = 0


# ── loss helpers ──────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def loss_and_grad(h_steered: np.ndarray, label: int):
    """Cross-entropy loss + gradient w.r.t. steered hidden state."""
    scores   = HEAD_W @ h_steered
    probs    = softmax(scores)
    loss     = -np.log(probs[label] + 1e-9)
    delta_p  = probs.copy()
    delta_p[label] -= 1.0
    grad_h   = HEAD_W.T @ delta_p   # dL/dh̃
    return loss, grad_h


# ── dataset ───────────────────────────────────────────────────────────────────
def make_dataset(n: int):
    labels  = [random.randrange(N_LABELS) for _ in range(n)]
    hiddens = [frozen_lm_hidden(l) for l in labels]
    return hiddens, labels


train_h, train_y = make_dataset(N_TRAIN)
test_h,  test_y  = make_dataset(N_TEST)


# ── baseline: frozen LLM, no steering ────────────────────────────────────────
baseline_acc = np.mean([head_predict(h) == y for h, y in zip(test_h, test_y)])
print("=" * 60)
print("  DecSteer Proof-of-Concept")
print("  arXiv: https://arxiv.org/abs/2610.06950")
print("=" * 60)
print(f"\nD_MODEL={D_MODEL}, SIGNAL_DIM={SIGNAL_DIM}, RANK={RANK}")
print(f"DecSteer parameter count : {2*RANK*D_MODEL:,}")
print(f"Baseline accuracy        : {baseline_acc:.1%}  (frozen LLM, no steering)\n")


# ── training ──────────────────────────────────────────────────────────────────
steer = DecSteer()

for epoch in range(EPOCHS):
    idxs = list(range(N_TRAIN))
    random.shuffle(idxs)
    epoch_loss = 0.0

    for i in idxs:
        h, y     = train_h[i], train_y[i]
        h_s      = steer.forward(h)
        loss, gh = loss_and_grad(h_s, y)
        epoch_loss += loss
        steer.backward(h, gh)

    steer.step(LR)

    if (epoch + 1) % 30 == 0:
        preds = [head_predict(steer.forward(h)) for h in test_h]
        acc   = np.mean([p == y for p, y in zip(preds, test_y)])
        print(f"  Epoch {epoch+1:3d} | avg loss={epoch_loss/N_TRAIN:.4f} "
              f"| steered acc={acc:.1%}")


# ── final evaluation ──────────────────────────────────────────────────────────
preds_steered = [head_predict(steer.forward(h)) for h in test_h]
steered_acc   = np.mean([p == y for p, y in zip(preds_steered, test_y)])
random_acc    = 1.0 / N_LABELS

print("\n" + "=" * 60)
print("  Final Results")
print("=" * 60)
print(f"  Random chance               : {random_acc:.1%}")
print(f"  Frozen LLM (no steering)    : {baseline_acc:.1%}")
print(f"  DecSteer rank={RANK} ({steer.n_params:,} params) : {steered_acc:.1%}")
print(f"  Improvement over baseline   : +{steered_acc - baseline_acc:.1%}")

print("\n" + "=" * 60)
print("  Key takeaways from the paper")
print("=" * 60)
print("  1. A frozen LLM's activations already contain the answer signal")
print("     in a low-dimensional subspace — but noise from other dims")
print("     causes the head to misfire on ~35-45% of examples.")
print("  2. A rank-4 DecSteer operator (just 2·4·D trainable params)")
print("     learns to amplify that subspace and suppress the noise.")
print("  3. Training is pure behavior cloning — no RL, no reward model.")
print("  4. At inference time: the 3,000+ token reasoning chain is replaced")
print("     by a single matrix multiply, cutting latency by ~600×.")
print("  5. Skill hot-swap: load a different (A, B) pair to give the model")
print("     a completely new decision skill with no weight updates.")
