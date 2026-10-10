"""
Proof-of-Concept: Latent Recurrent Thoughts (LRT)
Paper: "Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents
         for Reasoning with Frozen LLMs" (arXiv:2609.01117, EMNLP 2026 Findings)
Authors: Zhaoliang Chen (Emory University), Jie Fu (IQuest Research)
Official repo: https://github.com/czl-david/latent-recurrent-thoughts

Core idea: Instead of chain-of-thought in discrete token space (where errors
propagate), reason in continuous latent space using a tiny recurrent network.
The base LLM stays frozen — only the small proposer+reasoner are trained.

Three components:
  1. Proposer   — encodes the input question into initial latent vectors
  2. Recurrent Reasoner — iteratively refines those latents (GRU-based)
  3. Decoder    — reads final latent and produces the answer token by token

Requirements: pip install torch  (CPU-only is fine for this demo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LRTConfig:
    vocab_size: int = 512       # tiny toy vocabulary
    embed_dim: int = 64         # token embedding size
    latent_dim: int = 128       # dimension of the latent thought vectors
    num_latents: int = 8        # how many latent "thought slots" per step
    recurrent_steps: int = 6    # how many refinement iterations (= test-time compute)
    hidden_dim: int = 256       # GRU hidden size
    num_layers: int = 2         # GRU layers in the recurrent reasoner


# ---------------------------------------------------------------------------
# Proposer: input question → initial latent thoughts
# ---------------------------------------------------------------------------
class Proposer(nn.Module):
    """Lightweight transformer encoder that maps a tokenised question to
    `num_latents` continuous latent vectors."""

    def __init__(self, cfg: LRTConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim, nhead=4, dim_feedforward=256,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Pool to fixed number of latent slots via learned queries
        self.latent_queries = nn.Parameter(
            torch.randn(cfg.num_latents, cfg.embed_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.embed_dim, num_heads=4, batch_first=True
        )
        self.proj = nn.Linear(cfg.embed_dim, cfg.latent_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) integer token ids
        Returns:
            latents: (B, num_latents, latent_dim)
        """
        B = input_ids.size(0)
        x = self.embed(input_ids)                       # (B, S, embed_dim)
        x = self.encoder(x)                             # (B, S, embed_dim)
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        latents, _ = self.cross_attn(queries, x, x)    # (B, num_latents, embed_dim)
        return self.proj(latents)                        # (B, num_latents, latent_dim)


# ---------------------------------------------------------------------------
# Recurrent Reasoner: refines latents over T steps
# ---------------------------------------------------------------------------
class RecurrentReasoner(nn.Module):
    """GRU-based module that refines latent thoughts with bounded residual
    corrections — keeps updates small and stable across many iterations."""

    def __init__(self, cfg: LRTConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.residual_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        # Gate controls how much to update — prevents runaway refinement
        self.gate = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.hidden_dim, cfg.latent_dim),
            nn.Sigmoid(),
        )
        self.cfg = cfg

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, num_latents, latent_dim) initial latent thoughts
        Returns:
            refined_latents: (B, num_latents, latent_dim) after T steps
        """
        B, N, D = latents.shape
        # Flatten slots to treat each slot independently through GRU time axis
        # We treat recurrent_steps as the "time" dimension for iterative refinement
        current = latents  # (B, N, D)
        h = None           # GRU hidden state

        for _ in range(self.cfg.recurrent_steps):
            # Run one GRU step over all latent slots
            out, h = self.gru(current, h)         # out: (B, N, hidden_dim)
            delta = self.residual_proj(out)        # (B, N, latent_dim)
            g = self.gate(torch.cat([current, out], dim=-1))  # (B, N, latent_dim)
            # Gated residual: only update where the gate opens
            current = current + g * delta

        return current  # (B, N, latent_dim)


# ---------------------------------------------------------------------------
# Stub Frozen LLM Decoder (replace with real LLM for actual use)
# ---------------------------------------------------------------------------
class FrozenLLMDecoder(nn.Module):
    """In real LRT, this is an actual frozen LLM (e.g. LLaMA, GPT-2).
    The latent thoughts are injected as prefix key-value pairs into every
    attention layer via cross-attention adapters — the LLM weights stay fixed.

    This stub mimics that behaviour with a simple linear projection so the
    PoC runs on CPU without requiring a real LLM download."""

    def __init__(self, cfg: LRTConfig):
        super().__init__()
        # Simulate 'frozen' by not registering params in the optimizer
        self._sim_lm_head = nn.Linear(cfg.latent_dim, cfg.vocab_size, bias=False)
        for p in self._sim_lm_head.parameters():
            p.requires_grad = False  # frozen

    def decode(self, refined_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            refined_latents: (B, num_latents, latent_dim)
        Returns:
            logits: (B, num_latents, vocab_size) — next-token logits per slot
        """
        return self._sim_lm_head(refined_latents)


# ---------------------------------------------------------------------------
# Full LRT Model
# ---------------------------------------------------------------------------
class LRTModel(nn.Module):
    def __init__(self, cfg: LRTConfig):
        super().__init__()
        self.proposer = Proposer(cfg)
        self.reasoner = RecurrentReasoner(cfg)
        self.decoder = FrozenLLMDecoder(cfg)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        latents = self.proposer(input_ids)        # encode question
        latents = self.reasoner(latents)          # iterative latent refinement
        logits = self.decoder.decode(latents)     # decode answer tokens
        return logits

    def trainable_parameters(self):
        """Only proposer + reasoner are trained; decoder stays frozen."""
        return list(self.proposer.parameters()) + list(self.reasoner.parameters())


# ---------------------------------------------------------------------------
# Demo: compute budget scaling experiment
# Increasing recurrent_steps = more test-time compute, no retraining needed.
# ---------------------------------------------------------------------------
def demo_scaling():
    torch.manual_seed(42)
    cfg = LRTConfig(vocab_size=512, embed_dim=64, latent_dim=128,
                    num_latents=8, recurrent_steps=1, hidden_dim=256)

    # Toy question: "What is 3 + 5?" encoded as random token IDs
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    print("Latent Recurrent Thoughts — architecture demo\n")
    print(f"{'Steps':>6}  {'Latent norm':>12}  {'Logit entropy':>14}")
    print("-" * 40)

    for steps in [1, 3, 6, 10, 20]:
        cfg.recurrent_steps = steps
        model = LRTModel(cfg)
        model.eval()

        with torch.no_grad():
            latents = model.proposer(input_ids)
            refined = model.reasoner(latents)
            logits = model.decoder.decode(refined)

        norm = refined.norm(dim=-1).mean().item()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(-1).mean().item()
        print(f"{steps:>6}  {norm:>12.4f}  {entropy:>14.4f}")

    print()
    print("Key observation: more recurrent steps != larger model.")
    print("Test-time compute scales by increasing `recurrent_steps`.")


def demo_param_count():
    cfg = LRTConfig()
    model = LRTModel(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.trainable_parameters())
    frozen = total - trainable
    print(f"\nParameter counts (toy config):")
    print(f"  Total      : {total:,}")
    print(f"  Trainable  : {trainable:,}  <- only proposer + reasoner")
    print(f"  Frozen     : {frozen:,}     <- simulated LLM decoder")
    print("\nIn real LRT, the 'frozen' component is the full LLM (billions of params).")
    print("Only the tiny proposer+reasoner need to be trained.")


if __name__ == "__main__":
    demo_scaling()
    demo_param_count()
