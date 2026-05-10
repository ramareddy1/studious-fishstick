# Weekly AI Insight — 2026-05-08

## Title
**TurboQuant: Near-Optimal KV Cache Quantization for Long-Context LLM Inference**

## Source
- **Paper (arXiv):** https://arxiv.org/abs/2504.19874
- **Conference:** ICLR 2026
- **Authors:** Amir Zandieh et al. (Google DeepMind)

---

## Why It Matters

The **KV cache** is the memory buffer every transformer uses to store all prior tokens' key and value attention states. It grows linearly with sequence length and becomes the single biggest memory bottleneck for long-context inference — making 100K+ token contexts on consumer hardware nearly impossible.

TurboQuant compresses the KV cache to **~3 bits per value** using two mathematically grounded steps, achieving:

| Metric | Value |
|---|---|
| Memory reduction | **5–6x** |
| Inference speedup | **up to 8x** at long contexts |
| Accuracy drop | **< 0.5%** on standard benchmarks |
| Bits for K | **3-bit** (8 centroids) |
| Bits for V | **2-bit** (4 centroids) |

For students: this is a **drop-in inference optimization** — no retraining required. You can apply it to open models (Llama, Mistral, Gemma) today using any of the existing PyTorch or llama.cpp implementations below.

---

## How It Works

```
Standard Attention (per cached token):
  K, V stored as FP16  →  ~32 bytes / token / layer

TurboQuant Attention:
  Step 1 — Rotate K with Walsh-Hadamard Transform (WHT)
           → spreads outliers evenly across dimensions
  Step 2 — Quantize rotated K to 3-bit (Lloyd-Max codebook)
           Quantize V to 2-bit  (Lloyd-Max codebook)
           → ~3 bytes / token / layer  (≈10x fewer bytes)

  At query time:
  → Rotate query Q with same WHT
  → Compute attention with quantized K, V
  → Output ≈ identical to FP16 (Johnson-Lindenstrauss bound)
```

The **Walsh-Hadamard Transform** is the key insight: it costs O(d log d) to apply and eliminates the large per-channel outliers that normally destroy low-bit quantization quality. The **Lloyd-Max codebook** is then optimal in the mean-squared error sense for the resulting smooth distribution.

---

## GitHub Implementations

Multiple production-grade implementations already exist:

| Repository | Description |
|---|---|
| [hackimov/turboquant-kv](https://github.com/hackimov/turboquant-kv) | PyTorch; 6x memory reduction, 8x faster inference |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | From-scratch PyTorch; 5x compression, 99.5% attention fidelity |
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | Triton kernels + vLLM integration |
| [AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant) | llama.cpp integration; 5.2x memory reduction |
| [scos-lab/turboquant](https://github.com/scos-lab/turboquant) | Reference implementation (paper reproduction) |

Integration issues / discussions:
- [SGLang #21618](https://github.com/sgl-project/sglang/issues/21618) — core quantization + Triton kernels in progress
- [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) — community discussion

---

## Generated Educational Script

See [`turboquant_kvcache_demo.py`](./turboquant_kvcache_demo.py) in this folder.

The script demonstrates:
1. **FP16 vs quantized memory usage** — concrete byte counts
2. **Walsh-Hadamard rotation** — the outlier-spreading preprocessing step
3. **Lloyd-Max codebook construction** — from calibration data
4. **Attention fidelity check** — MSE, relative error, and cosine similarity vs FP16

**Run it (no GPU needed):**
```bash
pip install numpy
python turboquant_kvcache_demo.py
```

---

## Practical Notes

- **No GPU required** for the demo — runs on CPU with numpy only.
- **For real inference**, `hackimov/turboquant-kv` has the simplest PyTorch API; `AmesianX/TurboQuant` drops into llama.cpp.
- **Savings are additive**: combining TurboQuant KV compression with 4-bit weight quantization (GGUF/AWQ) cuts total model memory by up to **90%**.
- **Works for RAG pipelines**: any application caching large embedding stores benefits from the same rotation + low-bit quantization approach.
- **The math generalises**: the WHT + Lloyd-Max combination is applicable to any high-dimensional vector cache, not just transformer KV states.
