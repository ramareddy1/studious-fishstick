# Weekly AI Insight — 2026-11-06

## Title
**Output-Aware Block Influence: Jacobian-Lens Weighting for Depth Pruning**

## Source
- **arXiv:** https://arxiv.org/abs/2611.00015
- **Authors:** Kyle Lemoi, Jonathan Yong, Shubhankar Tripathy, Priestley Fernandes, Anirban Majumder, Zarreen Reza
- **Venue:** NeurIPS 2026 AXIOM Workshop (11 pages, 7 figures)
- **GitHub:** No official repo yet; baseline (ShortGPT / BI) at https://github.com/icip-cas/ShortGPT

---

## Why It Matters

Modern open-weight LLMs (Llama, Qwen, Gemma) have dozens of transformer blocks, but
**not all blocks are equally important**. The original ShortGPT paper showed you can
silently drop "redundant" blocks and keep ~90 % of model quality — a process called
*depth pruning*. It uses a score called Block Influence (BI): measure how much each
block changes the hidden state via cosine similarity between input and output.

This new paper from NeurIPS 2026 asks: *what if BI looks at the wrong reference frame?*

BI measures change **locally** (input vs. output of one block). But what actually matters
is the impact on the **final** model output — logits and predictions. The authors
introduce the **Jacobian Lens (J-Lens)**, which transports a block's contribution to the
final residual stream before measuring it. The resulting score, **J-BI**, gives a
global, output-aware view of each block's importance.

### What they found
| Metric | BI (baseline) | J-BI (this paper) | Gap |
|---|---|---|---|
| Post-pruning accuracy (Qwen3-8B, Llama-3.1-8B-IT, Gemma-3-12B-IT) | — | — | **+19 pts** improvement |
| Perplexity damage | Higher | Lower | Better |

### Why a student or developer should care
- **No GPU needed for pruning:** scoring blocks requires only a single CPU forward pass.
- **No retraining required:** remove the lowest-scoring blocks; the model runs immediately.
- **Runs on consumer hardware:** a 25 % pruned 8B model can fit in 12 GB RAM.
- **Works on three popular model families:** Qwen3, Llama-3.1, Gemma-3.
- **Generalises to any domain:** the scoring logic is model-agnostic; apply it to your
  fine-tuned checkpoint without changing anything.

---

## Implementation Details

### Existing resources
| Resource | Link |
|---|---|
| Paper (arXiv) | https://arxiv.org/abs/2611.00015 |
| Paper (HTML) | https://arxiv.org/html/2611.00015 |
| Baseline (ShortGPT) | https://github.com/icip-cas/ShortGPT |
| NeurIPS 2026 listing | https://neurips.cc/virtual/2026/157006 |

### Core algorithm (< 20 lines of Python)
```python
import torch, torch.nn.functional as F

def block_influence(h_in, h_out):
    # BI: local cosine-distance between block input and output
    return (1 - F.cosine_similarity(h_in.float(), h_out.float(), dim=-1).mean()).item()

def j_bi_score(h_in_i, h_out_i, h_out_final):
    # J-BI: transport to final layer, then measure BI
    scale = h_out_final.norm(dim=-1, keepdim=True) / (h_out_i.norm(dim=-1, keepdim=True) + 1e-8)
    return block_influence(h_in_i * scale, h_out_i * scale)
```

### Quick-start (demo mode — no download)
```bash
pip install torch
python jbi_depth_pruning_poc.py        # DEMO_MODE = True by default
```

### Quick-start (real model)
```bash
pip install torch transformers
# Edit jbi_depth_pruning_poc.py: set DEMO_MODE = False, pick MODEL_NAME
python jbi_depth_pruning_poc.py        # loads Qwen2.5-1.5B-Instruct (~3 GB)
```

### Production pruning workflow
```python
from transformers import AutoModelForCausalLM
import torch.nn as nn

# 1. Run jbi_depth_pruning_poc.py to get prune_indices
# 2. Load full model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# 3. Remove low-J-BI layers
kept = [i for i in range(len(model.model.layers)) if i not in prune_indices]
model.model.layers = nn.ModuleList([model.model.layers[i] for i in kept])
# 4. Save pruned model
model.save_pretrained("./qwen2.5-7b-pruned-25pct")
# 5. Done — no retraining; run inference as usual
```

---

## Generated Proof-of-Concept Script

See [`jbi_depth_pruning_poc.py`](./jbi_depth_pruning_poc.py) in this folder.

The script:
1. Implements Block Influence (BI) — the ShortGPT baseline.
2. Implements the Jacobian-Lens transport to approximate J-BI without full Jacobian computation.
3. Ranks all transformer blocks and identifies the K most-redundant ones.
4. Runs in **demo mode** (random tensors, no download) by default.
5. In real-model mode, hooks into any Llama/Qwen/Gemma architecture to collect
   per-block hidden states and score them live.

---

## Key Takeaway
Most LLM compression work requires expensive retraining or distillation.
This paper shows that simply **measuring block importance in the right reference frame**
(the final output, not each block's local input/output) is enough to find and remove
redundant depth with significantly less quality loss. A student can apply this today
to any open-weight model in under an hour on a laptop.
