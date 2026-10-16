# Weekly AI Insight — 2026-10-16

## Title
**SFT-as-Context: A Training-Free Cure for Catastrophic Forgetting in Fine-Tuned LLMs**

## Source
- **arXiv:** https://arxiv.org/abs/2610.11132
- **Authors:** Kenan Tang, Andong Hua, Chengxuan Qian, Saket Tiwari, Yao Qin
- **Submitted:** October 8, 2026
- **GitHub:** No official implementation found as of 2026-10-16 (see proof-of-concept below)

---

## Why It Matters

Every time you fine-tune a large language model (LLM) — to make it better at coding, math, or a domain task — it *forgets* how to do everything else. This "catastrophic forgetting" is one of the biggest headaches in applied AI. Existing fixes (LoRA, elastic weight consolidation, continual learning) add complexity and still trade off between specialization and generality.

**SFT-as-Context sidesteps the problem entirely — without touching the weights of either model.**

The key insight: a fine-tuned model's output on a given query is itself a high-quality "example" of the specialized skill. If you prepend that output to the original query and feed both to the *parent* (unfine-tuned) model, the parent picks up the specialization via in-context learning — while retaining its own broad knowledge.

### Key Results
- Tested on **19 parent-SFT model pairs** across **11 benchmarks**
- Stays within **2.2 pp** of the SFT model on fine-tuned tasks (e.g., AIME 2024, LiveCodeBench)
- Stays within **2.2 pp** of the parent model on general-capability benchmarks
- A small open-source SFT model can boost a large closed-source parent **above either model alone**
- **Zero additional training cost** — runs at inference time with any two models

---

## Implementation Details

No official code repository exists yet. The approach is simple enough to implement from scratch:

1. **Query the SFT model** with the user's question → get `sft_response`
2. **Inject `sft_response` as context** into the parent model's system prompt
3. **Query the parent model** — it now benefits from the specialist knowledge

Works with any OpenAI-compatible API, local Ollama instance, or HuggingFace pipeline. The `sft_as_context_demo.py` script in this folder demonstrates the full two-step inference loop.

---

## Generated Script

See [`sft_as_context_demo.py`](./sft_as_context_demo.py) in this folder.

```python
# Minimal sketch of the core idea:
sft_response = sft_model.complete(user_query)

augmented_prompt = f"""
A specialist model answered this query. Use it as context:
<specialist_response>
{sft_response}
</specialist_response>
"""
final_response = parent_model.complete(augmented_prompt + user_query)
```

**To run the demo:**
```bash
pip install openai
export OPENAI_API_KEY="your-key"
export SFT_MODEL="your-fine-tuned-model-id"   # or any chat model for testing
export PARENT_MODEL="gpt-4o-mini"
python sft_as_context_demo.py
```

For a fully local setup, swap in HuggingFace pipelines (see comments in the script).
