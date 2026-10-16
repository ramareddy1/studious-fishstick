"""
SFT-as-Context: Mitigating Catastrophic Forgetting Without Retraining
Based on: arXiv:2610.11132 (Kenan Tang et al., October 8, 2026)

Core idea: Instead of relying solely on a fine-tuned model (which forgets general skills),
keep the parent model and pass the SFT model's output as *context* before the parent answers.
The parent then benefits from the SFT specialization via in-context learning while retaining
its broad capabilities.

Requirements: pip install transformers torch openai
For the OpenAI-compatible demo, only `openai` is needed.
"""

import os

# ─── Option A: OpenAI-compatible API (minimal deps) ───────────────────────────
# Swap base_url for any OpenAI-compatible local server (e.g. Ollama, LM Studio).

from openai import OpenAI

PARENT_MODEL = os.getenv("PARENT_MODEL", "gpt-4o-mini")
SFT_MODEL    = os.getenv("SFT_MODEL",    "gpt-4o-mini")   # replace with your fine-tuned model id
API_KEY      = os.getenv("OPENAI_API_KEY", "your-key-here")

client = OpenAI(api_key=API_KEY)


def query_model(model: str, messages: list[dict]) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return resp.choices[0].message.content.strip()


def sft_as_context(user_query: str, system_prompt: str = "") -> dict:
    """
    Two-step SFT-as-Context inference:
      Step 1 – Ask the SFT (specialist) model.
      Step 2 – Feed its answer as context to the parent (general) model and re-ask.
    Returns both responses for comparison.
    """
    base_msgs = [{"role": "user", "content": user_query}]
    if system_prompt:
        base_msgs = [{"role": "system", "content": system_prompt}] + base_msgs

    # Step 1: specialist response
    sft_response = query_model(SFT_MODEL, base_msgs)

    # Step 2: inject specialist response as context for parent model
    augmented_msgs = [
        {
            "role": "system",
            "content": (
                (system_prompt + "\n\n" if system_prompt else "") +
                "A specialist model has already answered this question. "
                "Use its response as a helpful reference while drawing on your own broad knowledge:\n\n"
                f"<specialist_response>\n{sft_response}\n</specialist_response>"
            ),
        },
        {"role": "user", "content": user_query},
    ]
    parent_response = query_model(PARENT_MODEL, augmented_msgs)

    return {
        "query":           user_query,
        "sft_response":    sft_response,
        "parent_response": parent_response,
    }


# ─── Option B: Local HuggingFace models (no API key needed) ───────────────────
# Uncomment and adapt if you want a fully offline demo.

# from transformers import pipeline
#
# parent_pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct",
#                         device_map="auto", max_new_tokens=256)
# sft_pipe    = pipeline("text-generation", model="your-local-sft-checkpoint",
#                         device_map="auto", max_new_tokens=256)
#
# def query_local(pipe, messages):
#     return pipe(messages)[0]["generated_text"][-1]["content"]
#
# def sft_as_context_local(user_query):
#     sft_out = query_local(sft_pipe, [{"role":"user","content":user_query}])
#     augmented = [
#         {"role":"system","content":f"Reference from specialist:\n{sft_out}\n\nNow answer:"},
#         {"role":"user","content":user_query},
#     ]
#     parent_out = query_local(parent_pipe, augmented)
#     return {"sft_response": sft_out, "parent_response": parent_out}


# ─── Demo ─────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    # Specialist task (where SFT model should shine)
    "Explain the backpropagation algorithm step by step for a neural network with one hidden layer.",
    # General task (where parent model should retain capability)
    "Write a haiku about the ocean.",
]

if __name__ == "__main__":
    print("=== SFT-as-Context Demo ===\n")
    print("NOTE: Set OPENAI_API_KEY, PARENT_MODEL, and SFT_MODEL env vars before running.\n")
    print("With identical models this shows the plumbing; swap SFT_MODEL for a real fine-tune.\n")

    for q in DEMO_QUERIES:
        print(f"Query: {q}\n{'─'*60}")
        results = sft_as_context(q)
        print(f"SFT response:    {results['sft_response'][:300]}...")
        print(f"Parent response: {results['parent_response'][:300]}...")
        print()
