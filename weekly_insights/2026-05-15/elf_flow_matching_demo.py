"""
ELF: Embedded Language Flows — Proof-of-Concept Demo
Paper : https://arxiv.org/abs/2605.10938  (MIT, May 11 2026)
Authors: Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao,
         Tianhong Li, Yoon Kim, Jacob Andreas, Kaiming He

Demonstrates the three core ELF ideas:
  1. Token -> continuous embedding space
  2. Flow Matching: learn a straight-line velocity field noise -> data
  3. Nearest-neighbor discretisation back to tokens only at t=1
  + Classifier-Free Guidance (CFG) in embedding space

Dependencies: numpy only
Run: python elf_flow_matching_demo.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# Toy vocabulary and embedding table
# ---------------------------------------------------------------------------
VOCAB = ["[PAD]", "the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "slow"]
V = len(VOCAB)
D = 16  # embedding dimension

rng = np.random.default_rng(0)
E = rng.standard_normal((V, D)).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)  # unit-norm rows

token2id = {t: i for i, t in enumerate(VOCAB)}
id2token = {i: t for t, i in token2id.items()}


def embed(tokens):
    """Discrete tokens -> continuous embeddings (encoder)."""
    return np.stack([E[token2id[t]] for t in tokens])


def discretize(x):
    """
    Continuous embeddings -> nearest-neighbour token.
    Uses the *same* embedding matrix E (shared-weight decode), exactly as ELF does.
    """
    sims = x @ E.T          # (seq_len, V)
    return [id2token[i] for i in sims.argmax(axis=1)]


# ---------------------------------------------------------------------------
# Velocity network (toy linear model)
# ---------------------------------------------------------------------------
def velocity(x_t, t, x1):
    """
    Conditional Flow Matching velocity field.
    For linear paths x_t = (1-t)*x0 + t*x1 the exact velocity is:
        v*(x_t, t | x1, x0) = x1 - x0
    Because x1 and x0 are related via x_t we use the equivalent:
        v*(x_t, t | x1) = (x1 - x_t) / (1 - t + eps)
    Both formulations converge to x1 as t -> 1.
    """
    return (x1 - x_t) / (1.0 - t + 1e-5)


# ---------------------------------------------------------------------------
# Flow Matching: demonstrate the training objective
# ---------------------------------------------------------------------------
def flow_matching_loss_demo(x1, steps=200):
    """
    Show the CFM loss at random time steps.
    In real ELF a Transformer predicts the velocity; here we use the closed form.
    """
    losses = []
    for _ in range(steps):
        t = rng.uniform(0.01, 0.99)
        x0 = rng.standard_normal(x1.shape).astype(np.float32)  # Gaussian noise
        x_t = (1.0 - t) * x0 + t * x1                          # linear interpolation
        v_gt   = x1 - x0                                         # ground-truth velocity
        v_pred = velocity(x_t, t, x1)                           # predicted velocity
        losses.append(np.mean((v_pred - v_gt) ** 2))
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Inference: Euler ODE integration from noise (t=0) to data (t=1)
# ---------------------------------------------------------------------------
def generate(x1_cond, seq_len, ode_steps=20):
    """Integrate the ODE from pure noise to a generated embedding sequence."""
    x = rng.standard_normal((seq_len, D)).astype(np.float32)
    dt = 1.0 / ode_steps
    trajectory = [x.copy()]
    for step in range(ode_steps):
        t = step * dt
        x = x + velocity(x, t, x1_cond) * dt
        trajectory.append(x.copy())
    return x, trajectory


# ---------------------------------------------------------------------------
# Classifier-Free Guidance in embedding space
# ---------------------------------------------------------------------------
def generate_cfg(x1_cond, seq_len, guidance_scale=2.0, ode_steps=20):
    """
    CFG steers generation toward the conditioning embedding.
    null_cond = zero vector (unconditional = no target signal).
    v_guided = v_uncond + scale * (v_cond - v_uncond)
    """
    x = rng.standard_normal((seq_len, D)).astype(np.float32)
    null_cond = np.zeros_like(x1_cond)  # unconditioned target
    dt = 1.0 / ode_steps
    for step in range(ode_steps):
        t = step * dt
        v_cond   = velocity(x, t, x1_cond)
        v_uncond = velocity(x, t, null_cond)
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + v_guided * dt
    return x


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 62
    print(sep)
    print("ELF: Embedded Language Flows — Concept Demo")
    print("arXiv 2605.10938 | MIT | Published May 11, 2026")
    print(sep)

    src = ["the", "cat", "sat", "on", "mat"]
    print(f"\nSource tokens  : {src}")

    x1 = embed(src)
    print(f"Embedding shape: {x1.shape}  (seq={len(src)}, dim={D})")

    # Sanity: round-trip through embed -> discretize
    recovered = discretize(x1)
    print(f"Round-trip test: {recovered}")
    assert recovered == src

    # Training objective demo
    loss = flow_matching_loss_demo(x1)
    print(f"\nCFM training loss (200 random-t samples): {loss:.6f}")
    print("(Near zero because we use the exact velocity field — a trained")
    print(" Transformer would approximate this from data alone.)")

    # ODE generation
    print("\nODE integration: noise (t=0) -> data (t=1), 20 Euler steps")
    x_gen, traj = generate(x1, seq_len=len(src), ode_steps=20)
    print(f"\n{'Step':>5}  {'MSE to x1':>12}  Decoded tokens")
    print("-" * 50)
    for i in [0, 5, 10, 15, 19, 20]:
        mse = np.mean((traj[i] - x1) ** 2)
        tokens = discretize(traj[i])
        print(f"{i:>5}  {mse:>12.6f}  {tokens}")

    print(f"\nFinal output: {discretize(x_gen)}")

    # CFG demo
    print("\n--- Classifier-Free Guidance (guidance_scale=2.0) ---")
    x_cfg = generate_cfg(x1, seq_len=len(src), guidance_scale=2.0)
    print(f"CFG output   : {discretize(x_cfg)}")
    print("(CFG amplifies the conditioned direction -> faster convergence)")

    print(f"\n{sep}")
    print("Core ELF innovations shown:")
    print("  [1] Continuous embedding space — no discrete token arithmetic")
    print("  [2] Linear ODE (Flow Matching) — simpler than score diffusion")
    print("  [3] Shared-weight discretise at t=1 — no extra decoder")
    print("  [4] CFG works natively — same as image diffusion models")
    print(f"{sep}")
    print("Official repo: https://github.com/lillian039/ELF")
    print(sep)


if __name__ == "__main__":
    main()
