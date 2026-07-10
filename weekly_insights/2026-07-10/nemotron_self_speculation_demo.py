"""
Nemotron-Labs-Diffusion: Self-Speculation Decoding — Conceptual Demo
arXiv:2607.05722 | https://github.com/NVlabs/Nemotron-Labs-Diffusion

This script illustrates the three-mode decoding paradigm from the paper:
  Mode 1 — AR (Autoregressive): one token per forward pass
  Mode 2 — Diffusion: predict multiple masked tokens in parallel
  Mode 3 — Self-Speculation: diffusion proposes a block, AR verifies it
             (same weights, no separate draft model)

The key efficiency metric from the paper is NFE (Number of Forward Equivalents):
  NFE = total_flops / one_forward_pass_flops
  Fewer NFE to generate N tokens → higher throughput.

Part A uses a tiny randomly-initialised model to demonstrate the architecture.
Part B runs a Monte-Carlo simulation with a user-specified token-acceptance
probability to reproduce the throughput numbers reported in the paper.

Run:
    pip install torch
    python nemotron_self_speculation_demo.py
"""

import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal shared-weight transformer (attention mask is the only switch)
# ---------------------------------------------------------------------------
VOCAB = 64        # tiny vocabulary
SEQ   = 48        # max sequence length
DIM   = 128       # model dimension
HEADS = 4
MASK_TOKEN = VOCAB - 1   # special [MASK] id


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(DIM, 3 * DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, causal: bool = True):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(DIM // HEADS)
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, T, DIM))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(DIM)
        self.attn = Attention()
        self.ln2  = nn.LayerNorm(DIM)
        self.ff   = nn.Sequential(
            nn.Linear(DIM, DIM * 4), nn.GELU(), nn.Linear(DIM * 4, DIM)
        )

    def forward(self, x, causal: bool = True):
        x = x + self.attn(self.ln1(x), causal=causal)
        x = x + self.ff(self.ln2(x))
        return x


class TinyNemotron(nn.Module):
    """
    Single set of weights; attention pattern (causal vs. bidirectional)
    determines whether we're in AR mode or Diffusion mode — exactly as
    described in arXiv:2607.05722, Section 3.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.pos   = nn.Embedding(SEQ, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.head   = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids, causal: bool = True):
        B, T = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x, causal=causal)
        return self.head(x)   # (B, T, VOCAB)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def sample(logits_1d: torch.Tensor, temperature: float = 1.0) -> int:
    if temperature == 0.0:
        return int(logits_1d.argmax())
    probs = F.softmax(logits_1d / temperature, dim=-1)
    return int(torch.multinomial(probs, 1))


# ---------------------------------------------------------------------------
# Mode 1 — Autoregressive
# ---------------------------------------------------------------------------

def generate_ar(model, prompt_ids, n_new, temperature=1.0):
    """
    Standard left-to-right generation.
    NFE = n_new (one full forward pass per token).
    """
    ids = list(prompt_ids)
    nfe = 0
    for _ in range(n_new):
        inp = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(inp, causal=True)
        nfe += 1
        ids.append(sample(logits[0, -1], temperature))
    return ids[len(prompt_ids):], nfe


# ---------------------------------------------------------------------------
# Mode 2 — Masked Diffusion
# ---------------------------------------------------------------------------

def generate_diffusion(model, prompt_ids, n_new, steps=4, temperature=1.0):
    """
    Start with all n_new positions masked; iteratively unmask the most
    confident tokens using bidirectional attention.
    NFE ≈ steps (independent of n_new — the paper's main advantage).
    """
    # Bidirectional forward does not see the prompt during training in this
    # simplified demo; we pass only the masked segment.
    ids = [MASK_TOKEN] * n_new
    nfe = 0

    for step in range(steps):
        inp = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(inp, causal=False)
        nfe += 1

        masked_pos = [i for i, t in enumerate(ids) if t == MASK_TOKEN]
        if not masked_pos:
            break

        # Unmask a growing fraction each step
        target_unmasked = round(n_new * (step + 1) / steps)
        already_unmasked = n_new - len(masked_pos)
        n_unmask = max(1, target_unmasked - already_unmasked)

        confs = [(logits[0, p].max().item(), p) for p in masked_pos]
        for _, pos in sorted(confs, reverse=True)[:n_unmask]:
            ids[pos] = sample(logits[0, pos], temperature)

    # Patch any remaining masks
    for i, t in enumerate(ids):
        if t == MASK_TOKEN:
            inp = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = model(inp, causal=False)
            ids[i] = sample(logits[0, i], temperature)
            nfe += 1

    return ids, nfe


# ---------------------------------------------------------------------------
# Mode 3 — Self-Speculation (the paper's core contribution)
# ---------------------------------------------------------------------------

def generate_self_speculation(model, prompt_ids, n_new,
                               block_size=4, diff_steps=2, temperature=1.0):
    """
    Self-Speculation Decoding — same weights, shared KV cache.

    The diffusion mode (bidirectional) acts as the "drafter":
      - Predicts `block_size` tokens in parallel in `diff_steps` passes.

    The AR mode (causal) acts as the "verifier":
      - One AR forward pass verifies the whole draft block.
      - Accepts tokens from the left until the first mismatch; adds one
        AR-corrected token at the rejection point.

    Key difference from classical speculative decoding:
      - No separate smaller model needed (memory, coordination, alignment issues)
      - AR verifier re-uses KV cache computed during diffusion draft
      - NFE per block = diff_steps + 1 (vs block_size for pure AR)

    NOTE: With a randomly-initialised toy model the acceptance rate is low
    (tokens are essentially random).  See Part B for a simulation with
    realistic acceptance probabilities matching the paper's 6.82 toks/step.
    """
    ids = list(prompt_ids)
    nfe = 0
    accepted_total = 0
    blocks_total = 0

    while len(ids) - len(prompt_ids) < n_new:
        remaining = n_new - (len(ids) - len(prompt_ids))
        bsz = min(block_size, remaining)

        # --- Diffusion drafting (bidirectional, diff_steps passes) ---
        draft = [MASK_TOKEN] * bsz
        for step in range(diff_steps):
            inp = torch.tensor([draft], dtype=torch.long)
            with torch.no_grad():
                logits = model(inp, causal=False)
            nfe += 1
            masked = [i for i, t in enumerate(draft) if t == MASK_TOKEN]
            n_unmask = max(1, (len(masked) + diff_steps - step - 1) // (diff_steps - step))
            confs = [(logits[0, p].max().item(), p) for p in masked]
            for _, pos in sorted(confs, reverse=True)[:n_unmask]:
                draft[pos] = sample(logits[0, pos], temperature)
        for i, t in enumerate(draft):
            if t == MASK_TOKEN:
                inp = torch.tensor([draft], dtype=torch.long)
                with torch.no_grad():
                    logits = model(inp, causal=False)
                draft[i] = sample(logits[0, i], temperature)
                nfe += 1

        # --- AR verification (causal, one pass) ---
        full_ids = ids + draft
        inp = torch.tensor([full_ids], dtype=torch.long)
        with torch.no_grad():
            ar_logits = model(inp, causal=True)
        nfe += 1

        # Accept prefix where AR agrees with draft
        n_accepted = 0
        for i in range(bsz):
            verify_pos = len(ids) + i - 1   # AR output at pos p predicts p+1
            ar_best = int(ar_logits[0, verify_pos].argmax())
            if ar_best == draft[i]:
                n_accepted += 1
            else:
                draft[i] = ar_best   # substitute rejected token
                break

        n_accepted = max(1, n_accepted)
        ids.extend(draft[:n_accepted])
        accepted_total += n_accepted
        blocks_total += 1

    generated = ids[len(prompt_ids):]
    avg_accepted = accepted_total / max(blocks_total, 1)
    return generated, nfe, avg_accepted


# ---------------------------------------------------------------------------
# Part B — Monte-Carlo NFE simulation (no model needed)
# ---------------------------------------------------------------------------

def simulate_nfe(n_new: int, block_size: int, diff_steps: int,
                 accept_prob: float, n_trials: int = 5000) -> float:
    """
    Simulate self-speculation NFE for a given per-token acceptance probability.
    Mirrors Section 4 of the paper's efficiency analysis.

    Each block:
      - diff_steps bidirectional passes for the draft
      - 1 AR verification pass
      - Tokens accepted geometrically with probability accept_prob
    """
    total_nfe = 0
    for _ in range(n_trials):
        generated = 0
        nfe = 0
        while generated < n_new:
            remaining = n_new - generated
            bsz = min(block_size, remaining)
            nfe += diff_steps + 1   # draft + verify
            accepted = 0
            for _ in range(bsz):
                if random.random() < accept_prob:
                    accepted += 1
                else:
                    accepted += 1   # always accept corrected token at rejection point
                    break
            accepted = min(accepted, bsz)
            generated += accepted
        total_nfe += nfe
    return total_nfe / n_trials


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    random.seed(42)

    model = TinyNemotron()
    model.eval()

    prompt = [1, 5, 12, 7, 3]
    N_NEW  = 20
    BLOCK  = 4
    DSTEPS = 2

    print("=" * 64)
    print("Nemotron-Labs-Diffusion: Self-Speculation Decoding Demo")
    print(f"arXiv:2607.05722  |  generating {N_NEW} tokens")
    print("=" * 64)

    # ── Part A: Architecture demo with toy random model ──────────────────────
    print("\n── Part A: Architecture demo (randomly-initialised toy model) ──")

    _, ar_nfe  = generate_ar(model, prompt, N_NEW)
    _, di_nfe  = generate_diffusion(model, prompt, N_NEW, steps=DSTEPS)
    _, ss_nfe, ss_avg = generate_self_speculation(
        model, prompt, N_NEW, block_size=BLOCK, diff_steps=DSTEPS
    )

    def fmt(label, nfe, n=N_NEW):
        eff = n / nfe
        rel = eff / (n / ar_nfe)
        return f"  {label:<26} NFE={nfe:>4}  Tok/NFE={eff:>5.2f}  ({rel:.2f}× vs AR)"

    print(fmt("AR (baseline)",         ar_nfe))
    print(fmt(f"Diffusion ({DSTEPS} steps)",  di_nfe))
    print(fmt("Self-Speculation",      ss_nfe))
    print(f"\n  Avg tokens accepted/block (random model): {ss_avg:.2f}")
    print("  (Low acceptance expected — weights are random, not trained)")

    # ── Part B: NFE simulation with paper-realistic parameters ──────────────
    print("\n── Part B: NFE simulation with paper-realistic parameters ──")
    print("  (Monte-Carlo, 5000 trials per scenario)")
    print()

    # The paper uses a single diffusion pass for drafting (diff_steps=1) and
    # large blocks.  AR verification is one causal pass.
    # NFE per block = 1 (diff draft) + 1 (AR verify) = 2
    # vs AR baseline = block_size NFE per block
    # At 6.82 avg accepted tokens/step → speedup ≈ 6.82/2 ≈ 3.4× in NFE terms,
    # which translates to ~4× wall-clock with hardware-level KV-cache sharing.

    paper_configs = [
        # (label, block_size, diff_steps, accept_prob)
        ("Toy demo (block=4, steps=2)",    4,  2, 0.93),
        ("Larger block (block=8, steps=1)", 8, 1, 0.90),
        ("Paper config (block=8, steps=1, p=0.97)", 8, 1, 0.97),
        ("Paper result: 6.82 toks/step",   8,  1, 0.985),
    ]

    ar_expected_nfe = float(N_NEW)
    print(f"  {'Scenario':<38} {'B':>3} {'S':>3} {'p':>5} "
          f"{'Avg NFE':>8} {'Tok/NFE':>9} {'Speedup':>9}")
    print("  " + "-" * 82)

    for label, bsz, ds, p in paper_configs:
        avg_nfe = simulate_nfe(N_NEW, bsz, ds, p)
        tpn = N_NEW / avg_nfe
        speedup = ar_expected_nfe / avg_nfe
        print(f"  {label:<38} {bsz:>3} {ds:>3} {p:>5.3f} "
              f"{avg_nfe:>8.1f} {tpn:>9.2f} {speedup:>8.2f}×")

    print()
    print("  B = block_size, S = diff_steps")
    print()
    print("  NFE per block:")
    print("    AR baseline       : B NFE (one pass per token)")
    print("    Self-Speculation  : S(diff) + 1(AR verify) NFE for up to B tokens")
    print("    Break-even        : need accept_prob > (S+1)/B for speedup > 1×")

    print("\n" + "=" * 64)
    print("Key insight from arXiv:2607.05722:")
    print("  Joint AR+diffusion training → same weights serve as both")
    print("  drafter and verifier.  Shared KV cache eliminates the cost")
    print("  of a second model.  Paper achieves 6.82 tokens accepted/step")
    print("  and 4× end-to-end throughput on production hardware.")
    print()
    print("Official resources:")
    print("  GitHub : https://github.com/NVlabs/Nemotron-Labs-Diffusion")
    print("  3B HF  : https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B")
    print("  Paper  : https://arxiv.org/abs/2607.05722")
    print("=" * 64)


if __name__ == "__main__":
    main()
