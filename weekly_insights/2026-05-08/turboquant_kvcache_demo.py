"""
turboquant_kvcache_demo.py

Educational demonstration of the TurboQuant KV cache compression concept.
Based on: "TurboQuant: Near-Optimal KV Cache Quantization" (ICLR 2026, arXiv:2504.19874)

Demonstrates:
  1. FP16 vs quantized KV cache memory usage
  2. Walsh-Hadamard rotation (spreads outliers before quantization)
  3. Lloyd-Max scalar quantization (3-bit keys, 2-bit values)
  4. Attention output fidelity vs FP16 baseline

Requirements: numpy only
Run: pip install numpy && python turboquant_kvcache_demo.py
"""

import numpy as np
import time

np.random.seed(42)

SEQ_LEN  = 512
HEAD_DIM = 64
N_HEADS  = 8
N_LAYERS = 12
K_BITS   = 3
V_BITS   = 2


# ── Walsh-Hadamard Transform ──────────────────────────────────────────────────

def hadamard_matrix(n: int) -> np.ndarray:
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2)
    return H


def wht_rotate(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Rotate vectors to spread outliers evenly across dimensions."""
    return x @ H.T


# ── Lloyd-Max Quantization ────────────────────────────────────────────────────

def lloyd_max_codebook(data: np.ndarray, n_levels: int, n_iter: int = 60) -> np.ndarray:
    """Build a 1-D Lloyd-Max optimal codebook from calibration data."""
    lo, hi = data.min(), data.max()
    centroids = np.linspace(lo, hi, n_levels)
    for _ in range(n_iter):
        assignments = np.argmin(np.abs(data[:, None] - centroids[None, :]), axis=1)
        new_centroids = np.array([
            data[assignments == k].mean() if (assignments == k).any() else centroids[k]
            for k in range(n_levels)
        ])
        if np.allclose(new_centroids, centroids, atol=1e-7):
            break
        centroids = new_centroids
    return centroids


def quantize(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    flat = x.ravel()
    indices = np.argmin(np.abs(flat[:, None] - centroids[None, :]), axis=1)
    return centroids[indices].reshape(x.shape)


# ── Memory helpers ────────────────────────────────────────────────────────────

def fp16_kvcache_bytes(seq, heads, dim, layers):
    return 2 * seq * heads * dim * layers * 2  # K and V, 2 bytes per FP16


def quant_kvcache_bytes(seq, heads, dim, layers, k_bits, v_bits):
    k = (seq * heads * dim * k_bits) / 8
    v = (seq * heads * dim * v_bits) / 8
    return (k + v) * layers


# ── Attention ─────────────────────────────────────────────────────────────────

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  TurboQuant KV Cache Compression Demo")
    print("  Paper: arXiv:2504.19874  (ICLR 2026, Google DeepMind)")
    print("=" * 65)
    print(f"\nConfig: seq_len={SEQ_LEN}  n_heads={N_HEADS}  head_dim={HEAD_DIM}  n_layers={N_LAYERS}")
    print(f"        K={K_BITS}-bit  V={V_BITS}-bit\n")

    # Memory comparison
    fp16_mb  = fp16_kvcache_bytes(SEQ_LEN, N_HEADS, HEAD_DIM, N_LAYERS) / 1e6
    quant_mb = quant_kvcache_bytes(SEQ_LEN, N_HEADS, HEAD_DIM, N_LAYERS, K_BITS, V_BITS) / 1e6
    ratio    = fp16_mb / quant_mb
    print("Memory:")
    print(f"  FP16 KV cache   : {fp16_mb:.2f} MB")
    print(f"  TurboQuant cache: {quant_mb:.2f} MB  ({K_BITS}-bit K, {V_BITS}-bit V)")
    print(f"  Compression     : {ratio:.1f}x\n")

    # Build rotation matrix
    H = hadamard_matrix(HEAD_DIM)

    # Simulate one head's KV cache
    K_fp16 = np.random.randn(SEQ_LEN, HEAD_DIM).astype(np.float32)
    V_fp16 = np.random.randn(SEQ_LEN, HEAD_DIM).astype(np.float32)
    Q      = np.random.randn(1, HEAD_DIM).astype(np.float32)

    # Build codebooks from calibration samples
    calib  = np.random.randn(2000, HEAD_DIM).astype(np.float32)
    k_lvls = 2 ** K_BITS
    v_lvls = 2 ** V_BITS
    print(f"Building Lloyd-Max codebooks ({k_lvls} levels K, {v_lvls} levels V)...", end="", flush=True)
    t0 = time.time()
    cb_K = lloyd_max_codebook(wht_rotate(calib, H).ravel(), k_lvls)
    cb_V = lloyd_max_codebook(wht_rotate(calib, H).ravel(), v_lvls)
    print(f" done in {time.time()-t0:.2f}s\n")

    # Compress: rotate then quantize
    K_quant = quantize(wht_rotate(K_fp16, H), cb_K)
    V_quant = quantize(wht_rotate(V_fp16, H), cb_V)
    Q_rot   = wht_rotate(Q, H)

    # Attention outputs
    out_fp16  = attention(Q,     K_fp16, V_fp16)
    out_quant = attention(Q_rot, K_quant, V_quant)

    # Fidelity
    mse     = float(np.mean((out_fp16 - out_quant) ** 2))
    rel_err = float(np.linalg.norm(out_fp16 - out_quant) / (np.linalg.norm(out_fp16) + 1e-8))
    cosine  = float(
        (out_fp16.ravel() @ out_quant.ravel()) /
        (np.linalg.norm(out_fp16) * np.linalg.norm(out_quant) + 1e-8)
    )

    print("Attention Fidelity (FP16 baseline vs TurboQuant):")
    print(f"  MSE              : {mse:.6f}")
    print(f"  Relative error   : {rel_err*100:.3f}%")
    print(f"  Cosine similarity: {cosine:.6f}")
    print()

    print("=" * 65)
    print(f"  {ratio:.1f}x memory reduction")
    print(f"  {100 - rel_err*100:.2f}% attention fidelity retained")
    print()
    print("  Paper results:  5-6x memory | <0.5% accuracy drop | 8x faster")
    print("=" * 65)
    print()
    print("Production implementations:")
    print("  https://github.com/hackimov/turboquant-kv        (PyTorch)")
    print("  https://github.com/tonbistudio/turboquant-pytorch (from-scratch)")
    print("  https://github.com/0xSero/turboquant             (vLLM)")
    print("  https://github.com/AmesianX/TurboQuant           (llama.cpp)")


if __name__ == "__main__":
    main()
