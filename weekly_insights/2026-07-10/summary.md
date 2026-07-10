# Weekly AI Insight — 2026-07-10

## Title
**Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding**

## Source
- **Paper:** [arXiv:2607.05722](https://arxiv.org/abs/2607.05722)
- **Authors:** Yonggan Fu et al. (NVIDIA)
- **Published:** July 7, 2026
- **Coverage:** [NVIDIA's New LLM Decodes 6x More Tokens Without an Auxiliary Draft Model — TechTimes, July 9, 2026](https://www.techtimes.com/articles/319976/20260709/nvidias-new-llm-decodes-6x-more-tokens-without-auxiliary-draft-model.htm)

---

## Why It Matters

Every deployed LLM faces the same bottleneck: autoregressive (AR) generation is inherently sequential. Each output token requires a full forward pass, which limits throughput. The dominant fix — speculative decoding — requires maintaining a *second*, smaller "draft" model that proposes tokens cheaply, with the big model verifying them in parallel. Maintaining two separate models adds operational complexity, memory overhead, and alignment hassle.

**Nemotron-Labs-Diffusion eliminates the draft model entirely.**

The insight is elegant: train the model to be *both* an autoregressive model and a masked diffusion model simultaneously using a joint objective. At inference time, switching the attention pattern transforms the same set of weights into three distinct modes:

| Mode | Description | Throughput (TPF) |
|------|-------------|-----------------|
| **AR** | Standard left-to-right generation | 1× (baseline) |
| **Diffusion** | Parallel masked token prediction | ~2.6× |
| **Self-Speculation** | Diffusion drafts, AR verifies — shared KV cache | **6.4×** |

In self-speculation mode, the diffusion "head" of the model proposes a block of tokens in parallel. The AR "head" of the same model then verifies the block. Because they share weights and a KV cache — no round-trip to a separate model, no memory overhead for a second model — the verification is extremely cheap.

### Key Results (from the paper)
- The 8B instruct model achieves **6.82 accepted tokens per step** vs. Eagle3's 2.75 (the previous SOTA for speculative decoding with a separate draft model).
- **4× end-to-end throughput** on SGLang + NVIDIA GB200 vs. Qwen3-8B (same parameter count).
- Available in **3B, 8B, and 14B** parameter sizes; all open-weight with commercial-use license.
- A 3B base and instruct model on HuggingFace is accessible on consumer GPUs (≥16 GB VRAM).

### Why this is important for AI

Inference cost is the single biggest operational expense for AI products. Any technique that cuts tokens-per-second cost by 4× without additional hardware is immediately impactful. Nemotron-Labs-Diffusion does this with a fundamentally new training paradigm — not a better draft model, but a model that is *architecturally* both drafter and verifier. This could reset the baseline for what efficient LLM serving looks like in 2026 and beyond.

---

## Implementation Details

### Existing Implementations

Official GitHub and HuggingFace resources are available:

| Resource | Link |
|----------|------|
| GitHub repository | [NVlabs/Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion) |
| HuggingFace 3B Instruct | [nvidia/Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B) |
| HuggingFace 3B Base | [nvidia/Nemotron-Labs-Diffusion-3B-Base](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B-Base) |
| HuggingFace 8B Instruct | [nvidia/Nemotron-Labs-Diffusion-8B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B) |
| HuggingFace 14B Base | [nvidia/Nemotron-Labs-Diffusion-14B-Base](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B-Base) |
| NVIDIA Blog post | [Towards Speed-of-Light Text Generation](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion) |

### Quick Start (from the official repo)

```bash
pip install torch transformers
# Then load and run in any of the three modes:
python nemotron_self_speculation_demo.py
```

See [`nemotron_self_speculation_demo.py`](./nemotron_self_speculation_demo.py) in this folder for a self-contained conceptual walkthrough that:
1. Illustrates the three decoding modes (AR, Diffusion, Self-Speculation) with a toy transformer
2. Measures and compares NFE (Number of Forward Equivalents) — the paper's primary efficiency metric
3. Shows why self-speculation accepts more tokens than a separate small draft model
4. Runs entirely on CPU with no model downloads required

**Quick start for the demo:**
```bash
pip install torch
python nemotron_self_speculation_demo.py
```

### Related Repositories
- [NVlabs/Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion) — Official NVIDIA implementation
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — Production LLM serving; Nemotron diffusion support added in June 2026
- [sgl-project/sglang](https://github.com/sgl-project/sglang) — SGLang (used for official benchmarks)
- [huggingface/diffusion-pipe](https://github.com/huggingface/diffusion-pipe) — HuggingFace diffusion LM training recipes
- [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) — Apple MLX community (Nemotron-3B GGUF available for Mac)
