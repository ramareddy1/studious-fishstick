"""
SIRIN-inspired Contextual Hallucination Detector for RAG Systems
=================================================================
Inspired by: arXiv:2608.00033
  "SIRIN: A Unified Toolkit for Detecting Contextual Hallucinations in
   Retrieval-Augmented and Memory-Grounded LLM Systems"
  Belikova et al., Sber AI Lab, August 2026

This proof-of-concept captures SIRIN's three-paradigm approach:
  1. Keyword-overlap answerability check (zero-dependency)
  2. NLI-based entailment scoring    (requires sentence-transformers)
  3. Span-level unsupported claim detection (heuristic, zero-dependency)

Quick start:
  pip install sentence-transformers torch          # for full NLI mode
  python sirin_hallucination_detector.py           # fallback works without pip
"""

import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RAGTriple:
    """A context–query–answer triple — the unit SIRIN evaluates."""
    context: str
    query: str
    answer: str


@dataclass
class HallucinationReport:
    context: str
    query: str
    answer: str
    answerability_score: float     # 0=unanswerable  1=clearly answerable
    nli_entailment_score: float    # 0=contradiction  1=entailment
    span_support_score: float      # 0=all unsupported  1=all supported
    ensemble_risk: float           # final hallucination risk  0=safe  1=hallucinated
    unsupported_spans: List[str] = field(default_factory=list)
    verdict: str = ""


# ---------------------------------------------------------------------------
# Detector 1 — Keyword-overlap answerability (no dependencies)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "what", "who", "when", "where", "how", "why", "is", "are", "was",
    "were", "a", "an", "the", "of", "in", "to", "and", "or", "do",
    "did", "does", "that", "this", "it", "be", "has", "have", "had",
}


def _tokenize(text: str) -> set:
    return set(re.findall(r"\b\w+\b", text.lower()))


def answerability_score(context: str, query: str) -> float:
    """Return fraction of non-stopword query terms present in the context."""
    q_tokens = _tokenize(query) - _STOPWORDS
    if not q_tokens:
        return 0.5
    c_tokens = _tokenize(context)
    return len(q_tokens & c_tokens) / len(q_tokens)


# ---------------------------------------------------------------------------
# Detector 2 — NLI-based entailment (requires sentence-transformers)
# ---------------------------------------------------------------------------

_nli_model = None


def _load_nli():
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            print("  [NLI] Loading cross-encoder/nli-deberta-v3-small …")
            _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
        except ImportError:
            print("  [NLI] sentence-transformers not installed; using fallback 0.5")
            _nli_model = "unavailable"
    return _nli_model


def nli_entailment_score(context: str, answer: str) -> float:
    """
    Probability that the context *entails* the answer.
    Label ordering for cross-encoder/nli-deberta-v3-small: [contradiction, entailment, neutral]
    Falls back to 0.5 if the library is absent.
    """
    model = _load_nli()
    if model == "unavailable":
        return 0.5
    scores = model.predict([(context, answer)], apply_softmax=True)[0]
    return float(scores[1])  # entailment probability


# ---------------------------------------------------------------------------
# Detector 3 — Span-level unsupported claim detection (no dependencies)
# ---------------------------------------------------------------------------

def find_unsupported_spans(context: str, answer: str) -> List[str]:
    """
    Split answer into sentences; flag those whose vocabulary overlaps with
    the context by less than 20 %.  A rough but surprisingly effective filter.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    c_tokens = _tokenize(context)
    flagged = []
    for sent in sentences:
        s_tokens = _tokenize(sent) - _STOPWORDS
        if not s_tokens:
            continue
        overlap = len(s_tokens & c_tokens) / len(s_tokens)
        if overlap < 0.20:
            flagged.append(sent)
    return flagged


def span_support_score(context: str, answer: str) -> float:
    """Fraction of answer sentences that ARE supported by the context."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if not sentences:
        return 1.0
    unsupported = find_unsupported_spans(context, answer)
    return 1.0 - len(unsupported) / len(sentences)


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------

def detect_hallucination(triple: RAGTriple) -> HallucinationReport:
    """
    Run all three detectors and combine into a single hallucination-risk score.

    Weights mirror SIRIN's recommendation: NLI carries the most signal when
    available; answerability and span support serve as fast pre/post filters.
    """
    print(f"\n{'─'*60}")
    print(f"Q: {triple.query[:80]}")
    print(f"A: {triple.answer[:80]}{'…' if len(triple.answer) > 80 else ''}")

    a = answerability_score(triple.context, triple.query)
    e = nli_entailment_score(triple.context, triple.answer)
    s = span_support_score(triple.context, triple.answer)
    unsupported = find_unsupported_spans(triple.context, triple.answer)

    print(f"  answerability  : {a:.3f}")
    print(f"  nli entailment : {e:.3f}")
    print(f"  span support   : {s:.3f}")
    if unsupported:
        print(f"  flagged spans  : {unsupported}")

    # Higher a/e/s → lower hallucination risk
    risk = 1.0 - (0.20 * a + 0.55 * e + 0.25 * s)
    risk = max(0.0, min(1.0, risk))

    verdict = (
        "SAFE"         if risk < 0.35 else
        "UNCERTAIN"    if risk < 0.60 else
        "HALLUCINATED"
    )
    print(f"  ensemble risk  : {risk:.3f}  → {verdict}")

    return HallucinationReport(
        context=triple.context,
        query=triple.query,
        answer=triple.answer,
        answerability_score=a,
        nli_entailment_score=e,
        span_support_score=s,
        ensemble_risk=risk,
        unsupported_spans=unsupported,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Demo examples
# ---------------------------------------------------------------------------

EXAMPLES = [
    RAGTriple(
        context=(
            "The Eiffel Tower is located in Paris, France. "
            "It was built between 1887 and 1889 by Gustave Eiffel "
            "and stands 330 metres tall including its broadcast antenna."
        ),
        query="When was the Eiffel Tower built and how tall is it?",
        answer=(
            "The Eiffel Tower was constructed between 1887 and 1889 by Gustave Eiffel. "
            "It stands 330 metres tall."
        ),
    ),
    RAGTriple(
        context=(
            "Python 3.12 was released in October 2023. "
            "It introduced type parameter syntax and f-string improvements."
        ),
        query="What new features did Python 3.12 add?",
        answer=(
            "Python 3.12 added type parameter syntax and f-string improvements. "
            "It also introduced a new garbage collector that reduced memory usage by 40%."
        ),
    ),
    RAGTriple(
        context=(
            "The company was founded in 2005 and employs around 500 people. "
            "Its headquarters are in Berlin."
        ),
        query="What is the annual revenue of this company?",
        answer="The company generates approximately €20 million in annual revenue.",
    ),
    RAGTriple(
        context=(
            "AlphaFold 3, released by Google DeepMind in 2024, can predict "
            "the structure of proteins, DNA, RNA, and small molecules jointly."
        ),
        query="What can AlphaFold 3 predict?",
        answer=(
            "AlphaFold 3 can predict the structure of proteins, DNA, RNA, "
            "and small molecules jointly."
        ),
    ),
]


if __name__ == "__main__":
    print("SIRIN-inspired Contextual Hallucination Detector")
    print("Based on arXiv:2608.00033\n")
    print("Running detection on 4 context–query–answer triples …")

    reports = [detect_hallucination(t) for t in EXAMPLES]

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    labels = ["Grounded answer (safe)", "Hallucinated span", "Unanswerable context", "Grounded answer (safe)"]
    for i, (r, label) in enumerate(zip(reports, labels), 1):
        status = "✓" if r.verdict == "SAFE" else "⚠" if r.verdict == "UNCERTAIN" else "✗"
        print(f"  [{status}] Example {i} ({label}): {r.verdict}  risk={r.ensemble_risk:.2f}")

    print("""
Key insight from arXiv:2608.00033 (SIRIN):
  Combining NLI entailment + answerability + span-level checks into an
  ensemble catches hallucinations that any single detector misses.
  The plug-in design means you can swap in stronger detectors (e.g. an
  LLM judge) without changing the evaluation pipeline.
""")
