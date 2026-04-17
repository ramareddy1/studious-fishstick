"""
TurboQuant KV Cache Compression - Conceptual Proof of Concept
Based on: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
arXiv: https://arxiv.org/abs/2504.19874  (ICLR 2026, Google Research)

Community implementations (plug into HuggingFace / vLLM today):
  Python / HuggingFace : https://github.com/OnlyTerp/turboquant
  Triton + vLLM        : https://github.com/0xSero/turboquant
  Rust                 : https://github.com/RecursiveIntell/turbo-quant

── The problem TurboQuant solves ────────────────────────────────────────────
Long-context LLM inference is bottlenecked by the KV cache — the memory that
stores every previous token's attention keys and values. A 70B-param model with
a 128K-token context window needs ~140 GB just for the KV cache in FP16.

TurboQuant compresses the KV cache 5-6× with near-zero quality loss by using:

  Stage 1 — PolarQuant (3 bits/channel):
    Apply a fixed random rotation to each vector, then scalar-quantize.
    The rotation makes variance uniform across dimensions so the scalar
    quantizer achieves near-optimal MSE — no training or calibration needed.

  Stage 2 — QJL residual correction (1 bit/channel, for keys only):
    The quantization introduces a bias in inner products (attention logits).
    A 1-bit Johnson-Lindenstrauss sketch of the quantization residual
    corrects this bias, giving unbiased attention scores at negligible cost.

Run with:  python turboquant_demo.py
Requires:  pip install numpy  (nothing else)
"""

import numpy as np


# ── Quantization helpers ──────────────────────────────────────────────────────

def haar_rotation(d: int, seed: int = 42) -> np.ndarray:
    """Haar-distributed random orthogonal matrix (d×d)."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(H)
    return Q.astype(np.float32)


def quantize_per_vector(X: np.ndarray, bits: int):
    """Per-vector min-max scalar quantization. Returns reconstructed float array."""
    lo  = X.min(axis=1, keepdims=True)          # (N, 1)
    hi  = X.max(axis=1, keepdims=True)          # (N, 1)
    levels = 2 ** bits - 1
    scale = np.where(hi != lo, (hi - lo) / levels, np.ones_like(hi))
    q = np.round((X - lo) / scale).clip(0, levels).astype(np.uint8)
    return q.astype(np.float32) * scale + lo    # dequantized (N, D)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── Compression pipelines ─────────────────────────────────────────────────────

def compress_fp16(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float16).astype(np.float32)


def compress_naive(X: np.ndarray, bits: int = 3) -> np.ndarray:
    """Scalar quantize each vector independently — no rotation."""
    return quantize_per_vector(X, bits)


def compress_polarquant(X: np.ndarray, R: np.ndarray, bits: int = 3) -> np.ndarray:
    """PolarQuant: rotate → per-vector quantize → un-rotate."""
    rotated = X @ R.T
    recon   = quantize_per_vector(rotated, bits)
    return recon @ R


# ── Quality metrics ───────────────────────────────────────────────────────────

def attn_weights(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    return softmax((keys @ query) / np.sqrt(keys.shape[1]))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def tv_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def top_k_recall(true: np.ndarray, approx: np.ndarray, k: int = 5) -> float:
    return len(set(np.argsort(true)[-k:]) & set(np.argsort(approx)[-k:])) / k


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# ── Memory accounting ─────────────────────────────────────────────────────────

def kv_bytes(N: int, D: int, bits: int, qjl_dims: int = 0) -> int:
    """Estimate bytes for keys + values at `bits` bits/element + optional QJL sketch."""
    kv = 2 * ((N * D * bits + 7) // 8 + 8)   # packed bits + 2 floats metadata
    qjl = (N * qjl_dims + 7) // 8             # 1-bit JL sketch of keys
    return kv + qjl


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  TurboQuant KV Cache Compression — PoC Demo")
    print("=" * 64)

    rng  = np.random.default_rng(42)
    N, D = 512, 128   # 512 cached tokens, 128-dim attention head
    BITS = 3

    # Simulate transformer KV vectors with realistic structure:
    # - A few dominant directions carry most signal (like an LLM's learned subspaces)
    # - Add outlier dimensions (common in real transformers, makes naive quant hard)
    low_rank = rng.standard_normal((N, 8)).astype(np.float32) @ rng.standard_normal((8, D)).astype(np.float32)
    outliers = np.zeros((N, D), dtype=np.float32)
    outlier_dims = rng.choice(D, size=12, replace=False)
    outliers[:, outlier_dims] = rng.standard_normal((N, 12)).astype(np.float32) * 5.0  # 5× larger
    keys   = low_rank + outliers + rng.standard_normal((N, D)).astype(np.float32) * 0.1
    values = rng.standard_normal((N, D)).astype(np.float32)
    query  = rng.standard_normal(D).astype(np.float32)

    # Ground truth
    true_attn = attn_weights(query, keys)

    # Compress
    R          = haar_rotation(D)
    keys_fp16  = compress_fp16(keys)
    keys_naive = compress_naive(keys, BITS)
    keys_pq    = compress_polarquant(keys, R, BITS)

    attn_fp16  = attn_weights(query, keys_fp16)
    attn_naive = attn_weights(query, keys_naive)
    attn_pq    = attn_weights(query, keys_pq)

    # ── Section 1: Vector reconstruction quality ─────────────────────────────
    print(f"\nTest setup  : {N} tokens × {D} dims/head | {BITS}-bit quantization")
    print(f"Outlier dims: {len(outlier_dims)} dimensions at 5× larger magnitude")

    print(f"\nKey vector reconstruction MSE (lower = better):")
    print(f"  {'Method':<22}  {'MSE':>12}")
    print(f"  {'-'*36}")
    for label, keys_hat in [
        ("FP16",               keys_fp16),
        (f"Naive {BITS}-bit",         keys_naive),
        (f"PolarQuant {BITS}-bit",    keys_pq),
    ]:
        print(f"  {label:<22}  {mse(keys, keys_hat):>12.6f}")

    # ── Section 2: Attention quality ─────────────────────────────────────────
    print(f"\nAttention weight quality vs FP32 (lower KL = better):")
    print(f"  {'Method':<22}  {'KL-div':>10}  {'TV-dist':>10}  Top-5")
    print(f"  {'-'*55}")
    for label, attn in [
        ("FP16",               attn_fp16),
        (f"Naive {BITS}-bit",         attn_naive),
        (f"PolarQuant {BITS}-bit",    attn_pq),
    ]:
        print(
            f"  {label:<22}  "
            f"{kl_div(true_attn, attn):>10.6f}  "
            f"{tv_dist(true_attn, attn):>10.6f}  "
            f"{top_k_recall(true_attn, attn)*100:.0f}%"
        )

    # ── Section 3: Memory footprint ──────────────────────────────────────────
    b_fp32  = N * D * 4 * 2          # keys + values in FP32
    b_fp16  = N * D * 2 * 2
    b_naive = kv_bytes(N, D, BITS)
    b_pq    = kv_bytes(N, D, BITS)
    b_turbo = kv_bytes(N, D, BITS, qjl_dims=D // 4)  # + 1-bit QJL sketch for keys

    print(f"\nKV cache memory  (keys + values, {N} tokens):")
    print(f"  {'Method':<22}  {'Bytes':>10}  {'vs FP32':>10}")
    print(f"  {'-'*46}")
    for label, nb in [
        ("FP32",                b_fp32),
        ("FP16",                b_fp16),
        (f"Naive {BITS}-bit",          b_naive),
        (f"PolarQuant {BITS}-bit",     b_pq),
        ("TurboQuant (~3.5b)", b_turbo),
    ]:
        print(f"  {label:<22}  {nb:>10,}  {b_fp32/nb:>8.1f}×")

    print(f"""
── How to use today ────────────────────────────────────────────────
  pip install turboquant            # community Python package
  # Then in your HuggingFace inference loop, replace the attention
  # KV cache with a TurboQuant-compressed cache — same API, 5-6× smaller.

  # For vLLM (production serving):
  # https://github.com/0xSero/turboquant provides a Triton kernel patch.

── Why this matters ────────────────────────────────────────────────
  On a 70B model at 128K context:
    FP16 KV cache ≈ 140 GB  → needs 2× A100 80GB just for cache
    TurboQuant    ≈  24 GB  → fits on a single A100 with room for weights

  For students / on consumer hardware (RTX 4090 24 GB):
    FP16 at 4K context ≈ 4 GB   → fits, barely
    TurboQuant at 24K  ≈ 4 GB   → 6× longer context, same VRAM

── Paper ───────────────────────────────────────────────────────────
  https://arxiv.org/abs/2504.19874  (ICLR 2026, Google Research)
""")


if __name__ == "__main__":
    main()
