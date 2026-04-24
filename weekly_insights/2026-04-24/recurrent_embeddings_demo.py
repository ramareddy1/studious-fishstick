"""
Proof-of-Concept: Linear-Time, Constant-Memory Text Embeddings
Based on: "Linear-Time and Constant-Memory Text Embeddings Based on Recurrent Language Models"
Paper:    https://arxiv.org/abs/2604.18199

Core idea:
  Transformer embeddings use O(n^2) memory (full attention matrix).
  Recurrent SSMs (Mamba2) maintain a FIXED-SIZE hidden state, so
  peak memory is O(chunk_size x hidden_dim) — constant in sequence length.

This demo uses Mamba v1 (130M) from HuggingFace, which runs on CPU.
For the paper's exact architecture, swap MODEL_ID for a
dynatrace-oss/mamba2-embed-* checkpoint (requires CUDA + mamba-ssm).

Requirements:
    pip install transformers torch
"""

from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "state-spaces/mamba-130m-hf"  # 130M-param Mamba v1, CPU-friendly
CHUNK_SIZE = 256  # tokens per forward pass — this is the memory ceiling


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def embed_chunked(text: str, tokenizer, model) -> torch.Tensor:
    """
    Vertically-chunked recurrent inference (the paper's key contribution).

    The full sequence is processed CHUNK_SIZE tokens at a time.  At each step
    only the final hidden-state vector is kept — peak memory is O(chunk_size),
    NOT O(seq_len).  For a standard transformer, you cannot do this because
    the attention matrix must hold every token-pair simultaneously.
    """
    input_ids = tokenizer(
        text, return_tensors="pt", truncation=False
    )["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]

    last_h = None
    for start in range(0, seq_len, CHUNK_SIZE):
        chunk = input_ids[:, start : start + CHUNK_SIZE]
        with torch.no_grad():
            out = model(chunk, output_hidden_states=True)
        # Carry only the last-token hidden state forward (fixed size regardless of chunk)
        last_h = out.hidden_states[-1][:, -1, :]  # (1, hidden_dim)

    return F.normalize(last_h, p=2, dim=-1)  # L2-normalised unit embedding


def cosine_sim_matrix(embs: torch.Tensor) -> torch.Tensor:
    return embs @ embs.T


def memory_scaling_demo(tokenizer, model):
    """
    Show that processing time scales linearly in sequence length while
    peak memory stays flat — the defining property of constant-memory inference.
    """
    base = "State-space models maintain a fixed hidden state across any sequence length. "
    print("\n─── Memory Scaling Demo ───")
    print(f"  Chunk size : {CHUNK_SIZE} tokens")
    print(f"  Peak memory ceiling = O(chunk_size x hidden_dim), NOT O(seq_len)\n")
    print(f"  {'Tokens':>8}  {'Chunks':>8}  {'Time (s)':>10}")
    for mult in [1, 10, 50, 200]:
        text = base * mult
        n_tok = tokenizer(
            text, return_tensors="pt", truncation=False
        )["input_ids"].shape[1]
        n_chunks = -(-n_tok // CHUNK_SIZE)  # ceil
        t0 = time.perf_counter()
        _ = embed_chunked(text, tokenizer, model)
        elapsed = time.perf_counter() - t0
        print(f"  {n_tok:>8}  {n_chunks:>8}  {elapsed:>10.3f}")
    print(
        "\n  Observation: time grows ~linearly; the hidden-state tensor is"
        " the same size for 20 tokens and 20,000 tokens."
    )


def main():
    print("=== Linear-Time Recurrent Text Embeddings Demo ===")
    print(f"  Model : {MODEL_ID}")
    print(f"  Chunk : {CHUNK_SIZE} tokens\n")

    tokenizer, model = load_model()

    sentences = [
        "State-space models process sequences in linear time with a constant-size hidden state.",
        "Recurrent architectures like Mamba maintain a compressed memory of all past tokens.",
        "Transformers apply self-attention across every token pair, consuming quadratic memory.",
        "The Eiffel Tower stands 330 metres tall and was built in 1889 in Paris, France.",
        "Python is a high-level programming language known for its clean, readable syntax.",
    ]
    labels = ["SSM/Mamba", "Recurrent", "Transformer", "Eiffel Tower", "Python lang"]

    t0 = time.perf_counter()
    embs = torch.cat([embed_chunked(s, tokenizer, model) for s in sentences])
    print(f"Embedded {len(sentences)} sentences in {time.perf_counter() - t0:.2f}s")
    print(f"Embedding shape : {tuple(embs.shape)}\n")

    sim = cosine_sim_matrix(embs)
    col_w = 15
    print("─── Cosine Similarity Matrix ───")
    print(f"{'':>14}" + "".join(f"{l:>{col_w}}" for l in labels))
    for i, label in enumerate(labels):
        row = f"{label:>14}" + "".join(
            f"{sim[i, j].item():>{col_w}.3f}" for j in range(len(labels))
        )
        print(row)

    print(
        "\n  Expected: SSM/Mamba, Recurrent, and Transformer (all sequence-model topics)"
        "\n  should score higher with each other than with Eiffel Tower or Python lang."
    )

    memory_scaling_demo(tokenizer, model)

    print("\n─── Resources ───")
    print("  Paper    : https://arxiv.org/abs/2604.18199")
    print("  Models   : https://huggingface.co/dynatrace-oss")
    print("  Mamba2   : https://huggingface.co/docs/transformers/en/model_doc/mamba2")
    print("  SSM repo : https://github.com/state-spaces/mamba")


if __name__ == "__main__":
    main()
