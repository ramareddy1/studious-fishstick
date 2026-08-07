# Weekly AI Insight — 2026-08-07

## Title
**SIRIN: A Unified Toolkit for Detecting Contextual Hallucinations in Retrieval-Augmented and Memory-Grounded LLM Systems**

## Source
- **Paper:** [arXiv:2608.00033](https://arxiv.org/abs/2608.00033)
- **Authors:** Julia Belikova, Rauf Parchiev, Mikhail Filimonov, Konstantin Polev, Andrey Savchenko, Maksim Makarenko (Sber AI Lab)
- **Venue:** Preprint (submitted August 2026)
- **Live demo:** [hf.co/spaces/parchiev/SIRIN](https://hf.co/spaces/parchiev/SIRIN)

---

## Why It Matters

Retrieval-Augmented Generation (RAG) is now the dominant pattern for deploying LLMs that need to cite facts — from enterprise search to personal AI assistants. Yet the failure mode everyone quietly accepts is the **contextual hallucination**: a model that returns a fluent, confident answer that simply isn't supported by the retrieved documents it was given.

SIRIN (arXiv:2608.00033) is the first unified toolkit to systematically tackle this. Rather than picking one detection strategy, it brings three complementary paradigms under a single interface, configuration system, and evaluation pipeline:

| Paradigm | How it works | Requires |
|---|---|---|
| **Representation probing** | Reads the model's internal activation vectors to find spans the model is "unsure about" | White-box (open-weight model) |
| **Uncertainty estimation** | Samples multiple outputs and measures disagreement across runs | Black-box (API access) |
| **Judge-style verification** | Feeds the context–query–answer triple to a strong LLM and asks for a faithfulness verdict | Black-box (API access) |

SIRIN also adds a **pre-generation answerability check**: before the LLM even generates a response, it estimates whether the retrieved context is capable of answering the query. Flagging unanswerable queries early avoids wasted inference and the hallucinations that stem from an LLM being forced to answer with no supporting evidence.

### Why this architecture is significant

1. **No single detector is enough.** NLI-based checks miss semantic hallucinations that are lexically plausible. Uncertainty sampling misses confident-but-wrong answers. Judge models are slow and expensive for every query. SIRIN's ensemble is more robust than any single method at a fraction of the cost of running all three independently.

2. **Plug-in design.** Adding a new detector is a single Python class — no changes to the evaluation harness, the web UI, or the metrics logging. This dramatically lowers the barrier for researchers and practitioners to benchmark new approaches.

3. **Span-level inspection.** Most hallucination detectors give a single score per response. SIRIN also highlights the specific spans in the answer that lack contextual support, making it actionable: you can show users exactly which sentence is unverified.

4. **Works on memory-grounded agents.** The paper demonstrates SIRIN as a faithfulness gate within long-term memory systems — the same architecture shown in last week's insight. If an agent's memory retrieval returns a misleading snippet, SIRIN can catch the downstream hallucination before it reaches the user.

### Who should care

Anyone building a RAG pipeline today — whether that's a student adding document Q&A to a personal project, an engineer shipping a customer-facing chatbot, or a researcher evaluating LLMs — is currently flying blind on hallucinations unless they instrument their system. SIRIN provides that instrument.

---

## Implementation Details

### Existing Resources

| Resource | Link |
|---|---|
| Live demo (HuggingFace Space) | [hf.co/spaces/parchiev/SIRIN](https://hf.co/spaces/parchiev/SIRIN) |
| Related: Hallucination detection survey | [EdinburghNLP/awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) |
| Related: CORTEX token-level detection | [arXiv:2606.31033](https://arxiv.org/abs/2606.31033) |

No official GitHub repo for arXiv:2608.00033 was found at time of writing. The proof-of-concept below replicates the three-paradigm approach with minimal dependencies.

### Proof-of-Concept Script

See [`sirin_hallucination_detector.py`](./sirin_hallucination_detector.py) in this folder.

The script implements all three SIRIN paradigms in lightweight form — no GPU required:

| Module | SIRIN paradigm it maps to | Dependencies |
|---|---|---|
| `answerability_score()` | Pre-generation answerability check | None (stdlib only) |
| `nli_entailment_score()` | Judge-style verification via cross-encoder NLI | `sentence-transformers` (optional) |
| `find_unsupported_spans()` | Span-level unsupported claim detection | None (stdlib only) |
| `detect_hallucination()` | Ensemble scorer | Combines the above |

**Quick start:**
```bash
# Full mode (NLI model ~300 MB, runs on CPU)
pip install sentence-transformers torch
python sirin_hallucination_detector.py

# Zero-dependency mode (keyword overlap only)
python sirin_hallucination_detector.py   # works without pip install too
```

**Sample output:**
```
SIRIN-inspired Contextual Hallucination Detector
Based on arXiv:2608.00033

Running detection on 4 context–query–answer triples …

────────────────────────────────────────────────────────────
Q: When was the Eiffel Tower built and how tall is it?
A: The Eiffel Tower was constructed between 1887 and 1889 by Gustave Eiffel.…
  answerability  : 0.800
  nli entailment : 0.921
  span support   : 1.000
  ensemble risk  : 0.069  → SAFE

────────────────────────────────────────────────────────────
Q: What new features did Python 3.12 add?
A: Python 3.12 added type parameter syntax and f-string improvements. It als…
  answerability  : 0.667
  nli entailment : 0.311
  span support   : 0.500
  flagged spans  : ['It also introduced a new garbage collector that reduced memory usage by 40%.']
  ensemble risk  : 0.604  → HALLUCINATED

────────────────────────────────────────────────────────────
Q: What is the annual revenue of this company?
A: The company generates approximately €20 million in annual revenue.
  answerability  : 0.000
  nli entailment : 0.048
  span support   : 0.000
  flagged spans  : ['The company generates approximately €20 million in annual revenue.']
  ensemble risk  : 0.974  → HALLUCINATED
```

### Extending to a Production RAG Pipeline

```python
from sirin_hallucination_detector import detect_hallucination, RAGTriple

def rag_with_hallucination_gate(query: str, retrieved_docs: list[str], llm_answer: str) -> dict:
    """Wrap your existing RAG output with a hallucination gate."""
    context = "\n\n".join(retrieved_docs)
    report = detect_hallucination(RAGTriple(context=context, query=query, answer=llm_answer))

    return {
        "answer": llm_answer,
        "verdict": report.verdict,
        "risk": report.ensemble_risk,
        "flagged_spans": report.unsupported_spans,
        "show_to_user": report.verdict != "HALLUCINATED",
    }
```

The gate can be swapped in front of any existing RAG pipeline — it is provider-agnostic and adds no latency to the generation step itself (it runs after the LLM responds).

### Swapping in a Stronger Judge

To use an LLM judge instead of the NLI cross-encoder (at higher cost but better accuracy), replace `nli_entailment_score` with:

```python
from openai import OpenAI

def llm_judge_score(context: str, answer: str) -> float:
    """0.0 = hallucinated, 1.0 = fully grounded."""
    client = OpenAI()
    prompt = (
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Is every claim in the Answer fully supported by the Context? "
        "Reply with a single number from 0 (not supported) to 1 (fully supported)."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.5
```
