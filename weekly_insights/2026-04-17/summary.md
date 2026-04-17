# Weekly AI Insight — 2026-04-17

## Title
**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**

## Source
- **Paper (arXiv):** https://arxiv.org/abs/2504.19874
- **OpenReview (ICLR 2026):** https://openreview.net/pdf/6593f484501e295cdbe7efcbc46d7f20fc7e741f.pdf
- **Google Research Blog:** https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- **TechCrunch coverage:** https://techcrunch.com/2026/03/25/google-turboquant-ai-memory-compression-silicon-valley-pied-piper/

---

## Why It Matters

Running large language models is bottlenecked not by compute but by **memory bandwidth** — specifically the Key-Value (KV) cache that holds every token's attention state. On a 70-billion-parameter model with a 128K-token context window, the KV cache alone requires ~140 GB in FP16 — far beyond a single GPU.

**TurboQuant** (Google Research, ICLR 2026) achieves **5–6× compression of the KV cache** with near-zero quality degradation, matching FP16 accuracy on all tested benchmarks (LongBench, RULER, Needle-in-a-Haystack). It requires **no training, no calibration, and no model-specific tuning**, making it a drop-in upgrade for any transformer architecture.

Why this is a big deal:
- Enables **6× longer context windows** on the same GPU hardware
- Makes 70B+ model inference viable on **consumer GPUs** (24 GB VRAM)
- Is architecture-agnostic — applies to any transformer, including BERT, Llama, Mistral, Gemma
- Reduces inference cost proportionally to memory bandwidth reduction

---

## How It Works

TurboQuant is a two-stage, training-free pipeline applied to every KV vector:

### Stage 1 — PolarQuant (3 bits/channel)
1. Multiply each key/value vector by a fixed random rotation matrix **R** (computed once at load time).
2. The rotation redistributes variance uniformly across all dimensions — instead of a few "hot" dimensions dominating, every dimension carries roughly equal information.
3. Apply an analytical per-vector scalar quantizer to 3 bits.
4. At read time, un-rotate to reconstruct approximate vectors.

**Result:** Significant MSE reduction vs. naive 3-bit quantization because the rotation prevents the outlier dimensions from dominating the quantizer range.

### Stage 2 — QJL Residual Correction (1 bit/channel, keys only)
1. Compute the quantization residual (original key − PolarQuant reconstruction).
2. Apply a 1-bit Johnson-Lindenstrauss (JL) sketch: `sign(residual @ Phi)` for a fixed random matrix Phi.
3. At attention-score time, use this 1-bit sketch to provide an **unbiased inner-product estimator**, correcting the systematic bias that PolarQuant introduces in attention logits.

**Why QJL matters:** Naive compression biases attention logits directionally (not just adds noise). QJL removes this bias analytically using random projections, recovering FP16-level attention quality at only ~D/4 additional bits per key.

---

## Existing GitHub Implementations

| Repo | Language | Notes |
|------|----------|-------|
| [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) | Python | HuggingFace-compatible, ~5× compression |
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | Python | Triton kernels + vLLM integration, 3-bit keys / 2-bit values |
| [RecursiveIntell/turbo-quant](https://github.com/RecursiveIntell/turbo-quant) | Rust | Zero-overhead, PolarQuant + QJL in Rust |

---

## Proof-of-Concept Script

See [`turboquant_demo.py`](./turboquant_demo.py) in this folder.

**Requirements:** `pip install numpy`

**What it demonstrates:**
- Simulating transformer KV vectors with realistic outlier dimensions (common in real LLMs)
- Per-vector scalar quantization (naive baseline)
- PolarQuant: Haar rotation + per-vector quantization
- Memory footprint comparison across FP32 / FP16 / Naive 3-bit / PolarQuant / TurboQuant
- Attention quality metrics: KL divergence, TV distance, Top-5 recall

**Run it:**
```bash
pip install numpy
python turboquant_demo.py
```

**Sample output (512 tokens × 128 dims, 3-bit):**
```
Key vector reconstruction MSE:
  FP16                   0.000000
  Naive 3-bit            0.723754   ← more error
  PolarQuant 3-bit       0.482108   ← 33% less error, same bits

KV cache memory (keys + values):
  FP32                  524,288 bytes   (1.0×)
  FP16                  262,144 bytes   (2.0×)
  Naive 3-bit            49,168 bytes  (10.7×)
  PolarQuant 3-bit       49,168 bytes  (10.7×)
  TurboQuant (~3.5b)     51,216 bytes  (10.2×)
```

---

## Practical Notes

- **Quick start:** `pip install turboquant` and follow the HuggingFace integration guide at [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant).
- **Production serving:** Use the [vLLM Triton integration](https://github.com/0xSero/turboquant) for high-throughput inference.
- The rotation matrix **R** is computed once per model head — its storage overhead is negligible (D² floats).
- Best gains are on long-context tasks (>4K tokens); short-context tasks show smaller KV cache benefits.
- Works best with modern architectures using grouped-query attention (GQA) like Llama 3, Mistral, Gemma 2.
- On an RTX 4090 (24 GB VRAM), TurboQuant enables running a 70B model at 8K context where FP16 would require ~140 GB.
