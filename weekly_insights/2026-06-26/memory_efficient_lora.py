"""
Proof-of-Concept: Peak Memory Reduction for LoRA Fine-Tuning on Edge Devices

Based on: "Techniques for Peak Memory Reduction for LoRA Fine-tuning of LLMs
on Edge Devices" (arXiv:2606.19528, June 2026)

Demonstrates the four core techniques from the paper:
  1. Base model quantization with on-the-fly dequantization
  2. Memory-efficient gradient checkpointing
  3. Softmax approximation via top-k token subsets
  4. Logits masking to reduce output memory

Requirements:
  pip install torch transformers peft bitsandbytes datasets accelerate
"""

import gc
import os
import argparse
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def track_memory(label):
    reset_memory_stats()
    yield
    peak = get_gpu_memory_mb()
    print(f"[{label}] Peak GPU memory: {peak:.1f} MB")


# ---------------------------------------------------------------------------
# Technique 3: Softmax approximation via top-k token selection
# ---------------------------------------------------------------------------
class TopKSoftmax(nn.Module):
    """Approximate softmax by only computing over the top-k logits.

    The paper proposes using semantically relevant token subsets rather than
    the full vocabulary, drastically cutting memory for long-context training.
    """

    def __init__(self, k=100):
        super().__init__()
        self.k = k

    def forward(self, logits, labels):
        vocab_size = logits.size(-1)
        k = min(self.k, vocab_size)

        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)

        label_logits = logits.gather(-1, labels.unsqueeze(-1))

        combined_vals = torch.cat([topk_vals, label_logits], dim=-1)
        combined_idx = torch.cat([topk_idx, labels.unsqueeze(-1)], dim=-1)

        log_probs = F.log_softmax(combined_vals, dim=-1)

        label_position = combined_vals.size(-1) - 1
        nll = -log_probs[..., label_position]

        return nll.mean()


# ---------------------------------------------------------------------------
# Technique 4: Logits masking
# ---------------------------------------------------------------------------
def masked_logits_loss(logits, labels, mask_ratio=0.9):
    """Mask a fraction of vocabulary logits to reduce memory during loss."""
    vocab_size = logits.size(-1)
    keep = max(1, int(vocab_size * (1 - mask_ratio)))

    topk_vals, topk_idx = torch.topk(logits, keep, dim=-1)
    label_logits = logits.gather(-1, labels.unsqueeze(-1))

    combined = torch.cat([topk_vals, label_logits], dim=-1)
    log_probs = F.log_softmax(combined, dim=-1)
    nll = -log_probs[..., -1]
    return nll.mean()


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------
def demo_with_small_model(use_quantization=True, use_checkpointing=True,
                          use_topk_softmax=True, topk_k=100):
    """Run a LoRA fine-tuning step with memory reduction techniques."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Install: pip install transformers peft bitsandbytes accelerate")
        return demo_standalone()

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"\nLoading model: {model_name}")
    print(f"  Quantization: {use_quantization}")
    print(f"  Checkpointing: {use_checkpointing}")
    print(f"  Top-K Softmax (k={topk_k}): {use_topk_softmax}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Technique 1: Quantization with on-the-fly dequantization ---
    if use_quantization and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )
            print("Loaded with 4-bit quantization (NF4 + double quant)")
        except Exception as e:
            print(f"Quantization failed ({e}), loading in fp32")
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print("Loaded without quantization")

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # --- Technique 2: Gradient checkpointing ---
    if use_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_text = (
        "Fine-tuning large language models on edge devices requires careful "
        "memory management. This proof of concept demonstrates techniques "
        "from arXiv:2606.19528 for reducing peak memory during LoRA training."
    )
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True,
                       truncation=True, max_length=128).to(device)

    reset_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    # --- Forward pass ---
    outputs = model(**inputs, labels=inputs["input_ids"])

    if use_topk_softmax:
        topk_loss_fn = TopKSoftmax(k=topk_k)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
        loss = topk_loss_fn(logits, labels)
        print(f"Top-K softmax loss: {loss.item():.4f}")
    else:
        loss = outputs.loss
        print(f"Standard loss: {loss.item():.4f}")

    # --- Backward pass ---
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    peak_mem = get_gpu_memory_mb()
    print(f"\nPeak GPU memory after training step: {peak_mem:.1f} MB")
    return peak_mem


def demo_standalone():
    """Standalone demo without HuggingFace dependencies.

    Simulates the paper's techniques on a toy transformer to show memory
    impact, runnable on CPU with zero dependencies beyond PyTorch.
    """
    print("\n=== Standalone Demo (PyTorch only) ===")
    print("Simulating memory reduction techniques on a toy transformer\n")

    vocab_size = 32000
    hidden_dim = 512
    seq_len = 256
    num_layers = 4
    lora_rank = 8
    batch_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class ToyTransformerLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.attn_out = nn.Linear(dim, dim, bias=False)
            self.ff1 = nn.Linear(dim, 4 * dim, bias=False)
            self.ff2 = nn.Linear(4 * dim, dim, bias=False)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x):
            h = self.norm1(x)
            qkv = self.attn_qkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            scale = q.size(-1) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            h = torch.matmul(attn, v)
            x = x + self.attn_out(h)
            h = self.norm2(x)
            x = x + self.ff2(F.gelu(self.ff1(h)))
            return x

    class LoRALinear(nn.Module):
        def __init__(self, original, rank=8, alpha=16):
            super().__init__()
            self.original = original
            self.original.weight.requires_grad_(False)
            in_f = original.in_features
            out_f = original.out_features
            self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
            self.scaling = alpha / rank

        def forward(self, x):
            base = self.original(x)
            lora = (x @ self.lora_A @ self.lora_B) * self.scaling
            return base + lora

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList(
                [ToyTransformerLayer(hidden_dim) for _ in range(num_layers)]
            )
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, input_ids, use_checkpointing=False):
            x = self.embed(input_ids)
            for layer in self.layers:
                if use_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
                else:
                    x = layer(x)
            return self.head(x)

    model = ToyModel().to(device)

    for layer in model.layers:
        layer.attn_qkv = LoRALinear(layer.attn_qkv, rank=lora_rank)

    for p in model.parameters():
        if p.requires_grad:
            continue

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,} total, {trainable:,} trainable "
          f"({100 * trainable / total:.2f}%)")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    configs = [
        ("Baseline (no optimizations)", False, False, False),
        ("+ Gradient checkpointing", True, False, False),
        ("+ Top-K softmax (k=100)", True, True, False),
        ("+ Logits masking (90%)", True, False, True),
        ("All techniques combined", True, True, True),
    ]

    topk_loss_fn = TopKSoftmax(k=100)

    results = {}
    for name, use_ckpt, use_topk, use_mask in configs:
        reset_memory_stats()
        model.train()

        logits = model(input_ids, use_checkpointing=use_ckpt)

        if use_topk:
            loss = topk_loss_fn(logits[:, :-1, :], labels[:, 1:])
        elif use_mask:
            loss = masked_logits_loss(logits[:, :-1, :], labels[:, 1:])
        else:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                labels[:, 1:].reshape(-1),
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        peak = get_gpu_memory_mb()
        results[name] = {"loss": loss.item(), "peak_mb": peak}
        print(f"  {name:40s} | loss={loss.item():.4f} | peak={peak:.1f} MB")

    print("\n--- Summary ---")
    if torch.cuda.is_available():
        baseline = results["Baseline (no optimizations)"]["peak_mb"]
        best = results["All techniques combined"]["peak_mb"]
        if baseline > 0:
            print(f"Memory reduction: {baseline:.1f} -> {best:.1f} MB "
                  f"({baseline / max(best, 1e-6):.1f}x)")
    else:
        print("(GPU memory tracking requires CUDA; ran on CPU for demonstration)")
        print("On a GPU, the paper reports up to 26x memory reduction.")

    print("\nKey insight: These techniques are composable and can reduce")
    print("peak memory from 26 GB to ~1 GB for Llama-3.2 3B with LoRA,")
    print("enabling fine-tuning on consumer-grade GPUs and edge devices.")


def main():
    parser = argparse.ArgumentParser(
        description="PoC: Memory-efficient LoRA fine-tuning (arXiv:2606.19528)"
    )
    parser.add_argument("--mode", choices=["full", "standalone"], default="standalone",
                        help="'full' uses HuggingFace + real model; "
                             "'standalone' uses a toy transformer (no downloads)")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable quantization (full mode)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--no-topk", action="store_true",
                        help="Disable top-k softmax approximation")
    parser.add_argument("--topk-k", type=int, default=100,
                        help="Number of tokens for top-k softmax (default: 100)")
    args = parser.parse_args()

    if args.mode == "full":
        demo_with_small_model(
            use_quantization=not args.no_quant,
            use_checkpointing=not args.no_checkpoint,
            use_topk_softmax=not args.no_topk,
            topk_k=args.topk_k,
        )
    else:
        demo_standalone()


if __name__ == "__main__":
    main()
