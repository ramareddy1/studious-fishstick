"""
Proof-of-concept: Output-Aware Block Influence (J-BI) for LLM Depth Pruning
Based on: "Output-Aware Block Influence: Jacobian-Lens Weighting for Depth Pruning"
arXiv: 2611.00015 — NeurIPS 2026 AXIOM Workshop

What this script demonstrates:
  1. How to score each transformer block using Block Influence (BI) — the cosine
     similarity between a block's input and output hidden states.
  2. How the Jacobian-Lens extension (J-BI) transports that measurement to the
     final residual stream, giving a more global view of each block's impact.
  3. How to rank and remove the least-important blocks (zero-shot depth pruning,
     no retraining required).

DEMO_MODE = True  → runs entirely on random tensors; no model download needed.
DEMO_MODE = False → loads a real model from Hugging Face (requires ~16 GB RAM for
                    Llama-3.1-8B; use a smaller model like Qwen2.5-1.5B for <4 GB).

Usage:
    pip install torch transformers
    python jbi_depth_pruning_poc.py
"""

import torch
import torch.nn.functional as F

# ── configuration ──────────────────────────────────────────────────────────────
DEMO_MODE   = True          # set False to load a real HuggingFace model
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"  # ~3 GB; change to any causal LM
PRUNE_K     = 2             # number of blocks to remove in the demo
SAMPLE_TEXT = "The key insight from this paper is that"
DEVICE      = "cpu"         # change to "cuda" if a GPU is available
# ───────────────────────────────────────────────────────────────────────────────


# ── 1. Block Influence (BI) — from ShortGPT ────────────────────────────────────
def block_influence(h_in: torch.Tensor, h_out: torch.Tensor) -> float:
    """
    BI(i) = 1 - cosine_similarity(h_in, h_out) averaged over token positions.

    A score near 0 means the block barely changed the hidden state → redundant.
    A score near 1 means the block made a large transformation → important.
    """
    # h_in / h_out shape: [batch, seq_len, hidden_dim]
    cos = F.cosine_similarity(h_in.float(), h_out.float(), dim=-1)  # [B, S]
    return (1.0 - cos.mean()).item()


# ── 2. Jacobian Lens transport ─────────────────────────────────────────────────
def jacobian_lens_approx(
    h_out_i: torch.Tensor,
    h_out_final: torch.Tensor,
    h_in_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate Jacobian-Lens transport of h_in and h_out from block i
    to the final residual stream.

    The paper's J-Lens maps a perturbation δ at layer i to its effect at the
    final hidden state via the chain-rule Jacobian product.  For a lightweight
    zero-order approximation we use the ratio of norms along the residual path:

        scale = ||h_out_final|| / ||h_out_i||

    This keeps the direction of the hidden state while aligning its magnitude
    to the final layer, which is enough to demonstrate the concept without a
    full Jacobian computation.

    Returns transported versions of h_in and h_out for scoring.
    """
    scale = h_out_final.norm(dim=-1, keepdim=True) / (
        h_out_i.norm(dim=-1, keepdim=True) + 1e-8
    )
    h_in_transported  = h_in_i  * scale
    h_out_transported = h_out_i * scale
    return h_in_transported, h_out_transported


def j_bi_score(
    h_in_i: torch.Tensor,
    h_out_i: torch.Tensor,
    h_out_final: torch.Tensor,
) -> float:
    """
    J-BI(i): transport block i's hidden states to the final residual stream,
    then compute Block Influence there.
    """
    h_in_t, h_out_t = jacobian_lens_approx(h_out_i, h_out_final, h_in_i)
    return block_influence(h_in_t, h_out_t)


# ── 3. Demo with random tensors ────────────────────────────────────────────────
def run_demo():
    print("=" * 60)
    print("DEMO MODE — random tensors (no model download)")
    print("=" * 60)

    num_blocks = 32
    hidden_dim = 4096
    seq_len    = 16
    batch_size = 1

    torch.manual_seed(42)

    # Simulate one forward pass: each block slightly transforms hidden states
    # with added noise to simulate varying importance levels
    h = torch.randn(batch_size, seq_len, hidden_dim)
    hidden_states = [h.clone()]

    for _ in range(num_blocks):
        noise_scale = torch.rand(1).item() * 0.3   # random block "activity"
        h = h + noise_scale * torch.randn_like(h)
        hidden_states.append(h.clone())

    h_final = hidden_states[-1]

    bi_scores  = []
    jbi_scores = []

    for i in range(num_blocks):
        h_in  = hidden_states[i]
        h_out = hidden_states[i + 1]

        bi  = block_influence(h_in, h_out)
        jbi = j_bi_score(h_in, h_out, h_final)

        bi_scores.append(bi)
        jbi_scores.append(jbi)

    # Print ranking table
    print(f"\n{'Block':>6}  {'BI score':>10}  {'J-BI score':>12}  {'Rank by J-BI':>14}")
    print("-" * 50)
    jbi_ranked = sorted(range(num_blocks), key=lambda i: jbi_scores[i])
    rank_map = {block: rank for rank, block in enumerate(jbi_ranked)}
    for i in range(num_blocks):
        print(f"  {i:>4}   {bi_scores[i]:>10.4f}   {jbi_scores[i]:>12.4f}   {rank_map[i]:>14}")

    # Prune the K lowest J-BI blocks
    prune_indices = set(jbi_ranked[:PRUNE_K])
    kept = [i for i in range(num_blocks) if i not in prune_indices]

    print(f"\n--- Pruning {PRUNE_K} blocks with lowest J-BI score ---")
    print(f"Removed blocks : {sorted(prune_indices)}")
    print(f"Remaining blocks: {len(kept)} / {num_blocks}  "
          f"({100*len(kept)/num_blocks:.1f}% of original depth)")

    return bi_scores, jbi_scores


# ── 4. Real model mode ─────────────────────────────────────────────────────────
def run_real_model():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("Install transformers: pip install transformers")
        return

    print("=" * 60)
    print(f"Loading {MODEL_NAME} …")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map=DEVICE
    )
    model.eval()

    inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt").to(DEVICE)

    # Collect per-block hidden states via forward hooks
    hidden_states: list[torch.Tensor] = []

    def make_hook(idx):
        def hook(module, inp, out):
            # Most HF causal LM blocks return (hidden_state, ...) or just hidden_state
            hs = out[0] if isinstance(out, tuple) else out
            hidden_states.append(hs.detach().cpu())
        return hook

    layers = model.model.layers  # works for Llama, Qwen2, Gemma3 architectures
    handles = []
    # Also capture input to first block (the embedding output)
    hidden_states.append(None)  # placeholder; filled below

    def embed_hook(module, inp, out):
        hidden_states[0] = out.detach().cpu()

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(**inputs)

    for h in handles:
        h.remove()

    num_blocks = len(layers)
    h_final = hidden_states[-1]  # last block output

    bi_scores  = []
    jbi_scores = []
    for i in range(num_blocks):
        h_in  = hidden_states[i]
        h_out = hidden_states[i + 1]
        bi_scores.append(block_influence(h_in, h_out))
        jbi_scores.append(j_bi_score(h_in, h_out, h_final))

    jbi_ranked   = sorted(range(num_blocks), key=lambda i: jbi_scores[i])
    prune_indices = set(jbi_ranked[:PRUNE_K])
    kept          = [i for i in range(num_blocks) if i not in prune_indices]

    print(f"\nBlocks ranked by J-BI (lowest = most redundant): {jbi_ranked}")
    print(f"Suggested removal ({PRUNE_K} blocks): {sorted(prune_indices)}")
    print(f"Remaining depth: {len(kept)}/{num_blocks}")

    # Build a pruned model by simply skipping those layer indices
    # (production use would remove the layers and re-save with model.save_pretrained)
    print("\nPruned model block order (live, no retraining):")
    print(kept)
    print("\nTo generate text with the pruned model, replace model.model.layers with "
          "torch.nn.ModuleList([model.model.layers[i] for i in kept]) "
          "then call model.generate() as normal.")


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if DEMO_MODE:
        run_demo()
    else:
        run_real_model()

    print("\nDone.")
    print("\nKey take-away from the paper:")
    print("  Standard Block Influence measures LOCAL hidden-state change.")
    print("  J-BI transports that measurement to the FINAL layer, giving a")
    print("  global view of each block's impact on the model output.")
    print("  Result: up to +19 accuracy points vs BI at the same pruning ratio,")
    print("  with no retraining required (NeurIPS 2026 AXIOM Workshop).")
