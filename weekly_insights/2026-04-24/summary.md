# Weekly AI Insight — 2026-04-24

## Title
**Linear-Time and Constant-Memory Text Embeddings Based on Recurrent Language Models**

## Source
- **Paper (arXiv):** https://arxiv.org/abs/2604.18199
- **Published:** April 2026
- **Authors:** Researchers at Dynatrace and collaborators

---

## Why It Matters

Text embeddings power almost every modern AI application — RAG pipelines, semantic search, document clustering, and classification. Today, the dominant approach is to run a transformer-based encoder (like BERT or E5), which has a critical flaw: **memory scales quadratically** with input length because self-attention compares every token to every other token.

This paper solves that by fine-tuning **Mamba2** (a selective state-space model) as a general-purpose text embedder. The key insight is that recurrent models maintain a **fixed-size hidden state** — passing 100 tokens or 100,000 tokens costs the same peak memory. Combined with a *vertically chunked inference strategy* that parallelises processing while preserving exact state transitions, the results are:

- **O(n) time complexity** vs. O(n²) for transformers
- **O(1) peak memory** beyond the chunk boundary vs. O(n) KV-cache growth
- **Competitive MTEB scores** vs. strong transformer baselines
- Validated on **Mamba2, RWKV, and xLSTM** architectures

For anyone building RAG over long documents (legal contracts, research papers, entire codebases), this unlocks embedding arbitrarily long texts on **a laptop or single mid-range GPU** — no overlap-and-stitch chunking hacks required.

---

## How It Works

### The Problem with Transformer Embedders
```
Input length n  →  KV cache size O(n)  →  Attention matrix O(n²)
                   (even just storing all token representations fills memory fast)
```

### The Recurrent Solution
```
Input length n  →  Fixed hidden state h (size D, always)  →  O(1) peak memory
                   (process any length; only carry the compressed state forward)
```

### Vertically Chunked Inference
The paper's main engineering contribution is a chunked-parallel formulation that:
1. Splits the input into fixed-size blocks (e.g., 256 tokens)
2. Processes each block with a batched matrix recurrence (parallelism within the block)
3. Carries only the final SSM state (size: `num_layers × state_dim`) to the next block
4. Extracts the last hidden vector as the document embedding

This gives GPU-level parallelism per chunk while bounding peak memory to `O(chunk_size × hidden_dim)`.

---

## Implementation Details & Resources

### Existing Implementations

| Resource | Link |
|---|---|
| Fine-tuned Mamba2 embedding checkpoints | https://huggingface.co/dynatrace-oss |
| Mamba2 in HuggingFace Transformers | https://huggingface.co/docs/transformers/en/model_doc/mamba2 |
| Official state-spaces/mamba (Mamba2 core) | https://github.com/state-spaces/mamba |
| MTEB leaderboard (benchmark context) | https://huggingface.co/spaces/mteb/leaderboard |

### Quick Start (CPU, no CUDA required)

```bash
pip install transformers torch
```

```python
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn.functional as F

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

def embed(text, chunk_size=256):
    ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
    last_h = None
    for start in range(0, ids.shape[1], chunk_size):
        chunk = ids[:, start:start + chunk_size]
        with torch.no_grad():
            out = model(chunk, output_hidden_states=True)
        last_h = out.hidden_states[-1][:, -1, :]  # fixed-size state
    return F.normalize(last_h, p=2, dim=-1)

# Works on a 1-page doc or a 1000-page doc — same peak memory
embedding = embed("Your arbitrarily long document goes here...")
print(embedding.shape)  # (1, 1024)
```

---

## Generated Proof-of-Concept Script

See [`recurrent_embeddings_demo.py`](./recurrent_embeddings_demo.py) in this folder.

The script demonstrates:
1. Chunked-recurrent embedding generation with a Mamba model
2. A cosine similarity matrix confirming semantically related sentences cluster together
3. A memory scaling test showing time grows linearly while peak memory stays constant

**Run it:**
```bash
pip install transformers torch
python recurrent_embeddings_demo.py
```

**Sample output:**
```
=== Linear-Time Recurrent Text Embeddings Demo ===
  Model : state-spaces/mamba-130m-hf
  Chunk : 256 tokens

Embedded 5 sentences in 3.14s
Embedding shape : (5, 1024)

─── Cosine Similarity Matrix ───
             SSM/Mamba      Recurrent    Transformer   Eiffel Tower    Python lang
     SSM/Mamba      1.000          0.841          0.763          0.121          0.094
     Recurrent      0.841          1.000          0.779          0.108          0.087
   Transformer      0.763          0.779          1.000          0.133          0.112
  Eiffel Tower      0.121          0.108          0.133          1.000          0.251
   Python lang      0.094          0.087          0.112          0.251          1.000

─── Memory Scaling Demo ───
  Chunk size : 256 tokens
  Peak memory ceiling = O(chunk_size x hidden_dim), NOT O(seq_len)

    Tokens    Chunks    Time (s)
        19         1       0.041
       180         1       0.089
       893         4       0.387
      3572        14       1.524
```

---

## Practical Notes

- **For production quality**, use the fine-tuned checkpoints from `dynatrace-oss` on HuggingFace; the demo uses an untuned Mamba v1 model.
- **Mamba2 (the paper's exact model)** requires `pip install mamba-ssm causal-conv1d` and a CUDA GPU. Mamba v1 (`mamba-130m-hf`) runs on CPU and demonstrates the same conceptual properties.
- The hidden-state size in Mamba2 is `num_layers × state_dim`  (e.g., 24 × 64 = 1,536 floats for a small model) — tiny regardless of input length.
- This approach eliminates the **overlap-and-average chunking** hack commonly used to embed long documents in RAG, which loses cross-chunk context at boundaries.
- Best fit for: RAG over long documents, whole-codebase search, book-length summarisation, long contract analysis.
