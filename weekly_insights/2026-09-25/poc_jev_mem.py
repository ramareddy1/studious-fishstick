"""
Jev-Mem: System-One-Controlled Agentic Memory — Proof of Concept
arXiv: 2609.23986  |  https://github.com/libingzheren/Jev-Mem

Captures the core idea: split memory management into two layers.
  - System-1 (lightweight, rule/heuristic-based): governs EVERY memory op
      * classifies new observations into memory types
      * builds relational edges (semantic, temporal, causal, entity)
      * routes queries, allocates retrieval budgets, scores candidates
      * decides when to stop (adaptive early exit)
  - System-2 (LLM-based, only for synthesis): answers questions from
    the evidence that System-1 already surfaced.

Why it matters: standard agentic memory calls an LLM for every bookkeeping
decision, making construction 6× slower and queries 37% more expensive than
needed.  System-1 handles the high-frequency decisions cheaply; System-2 only
runs once per user query.

Runs on CPU, no GPU required.  Set OPENAI_API_KEY or ANTHROPIC_API_KEY for
real System-2 synthesis; otherwise falls back to a mock that returns the top
retrieved memory verbatim.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal

MemoryType = Literal["episodic", "semantic", "procedural", "preference"]
RelType = Literal["semantic", "temporal", "causal", "entity"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    text: str
    timestamp: float = field(default_factory=time.time)
    mem_type: MemoryType = "episodic"
    entity_tags: list[str] = field(default_factory=list)
    # edges: {memory_id: relation_type}
    edges: dict[int, RelType] = field(default_factory=dict)
    # type scores assigned by System-1 on construction
    episodic_score: float = 0.0
    semantic_score: float = 0.0
    procedural_score: float = 0.0
    preference_score: float = 0.0


# ---------------------------------------------------------------------------
# System-1 Controller (fast, LLM-free)
# ---------------------------------------------------------------------------

class System1Controller:
    """
    Cheap heuristic controller — no LLM calls.
    In the paper this is a small learned model; here we use rules that
    capture the same semantics well enough to demonstrate the architecture.
    """

    # -- Type classification keywords -----------------------------------------

    _PROCEDURAL_SIGNALS = {
        "how to", "steps", "procedure", "process", "algorithm", "run", "execute",
        "install", "configure", "command", "recipe", "workflow", "tutorial",
    }
    _PREFERENCE_SIGNALS = {
        "like", "dislike", "prefer", "favorite", "hate", "love", "want",
        "don't want", "always", "never", "style", "tone",
    }
    _SEMANTIC_SIGNALS = {
        "is", "are", "was", "were", "means", "definition", "concept",
        "theory", "fact", "generally", "typically", "usually",
    }

    # -- Query routing keywords -----------------------------------------------

    _CAUSAL_SIGNALS   = {"why", "because", "cause", "reason", "led to", "result"}
    _TEMPORAL_SIGNALS = {"when", "before", "after", "during", "since", "ago", "date"}
    _ENTITY_SIGNALS   = {"who", "which", "whose", "name", "person", "company", "place"}

    # -------------------------------------------------------------------------

    def classify(self, text: str) -> tuple[MemoryType, dict[str, float]]:
        """Return dominant type and all four scores for a piece of text."""
        lower = text.lower()
        tokens = set(re.findall(r"\w+", lower))
        bigrams = {" ".join(p) for p in zip(lower.split(), lower.split()[1:])}
        bag = tokens | bigrams

        scores = {
            "procedural": len(bag & self._PROCEDURAL_SIGNALS) / max(1, len(self._PROCEDURAL_SIGNALS)),
            "preference": len(bag & self._PREFERENCE_SIGNALS) / max(1, len(self._PREFERENCE_SIGNALS)),
            "semantic":   len(bag & self._SEMANTIC_SIGNALS)   / max(1, len(self._SEMANTIC_SIGNALS)),
            "episodic":   0.05,  # base probability — anything is episodic by default
        }
        # Boost episodic if it contains a past-tense verb (rough proxy)
        if re.search(r"\b(was|were|did|had|went|saw|made|told|found)\b", lower):
            scores["episodic"] += 0.15

        dominant = max(scores, key=scores.get)
        return dominant, scores  # type: ignore[return-value]

    def extract_entities(self, text: str) -> list[str]:
        """Rough entity extraction: capitalised runs of words."""
        return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)

    def route_query(self, query: str) -> list[RelType]:
        """Return the relation types most relevant to this query."""
        lower = query.lower()
        rels: list[RelType] = []
        if any(s in lower for s in self._CAUSAL_SIGNALS):
            rels.append("causal")
        if any(s in lower for s in self._TEMPORAL_SIGNALS):
            rels.append("temporal")
        if any(s in lower for s in self._ENTITY_SIGNALS):
            rels.append("entity")
        if not rels:
            rels.append("semantic")  # default
        return rels

    def score_candidate(self, query: str, memory: Memory) -> float:
        """
        Fast relevance score: token overlap (BM25-lite) + recency bonus.
        Real Jev-Mem also uses vector cosine; we omit embeddings for portability.
        """
        q_tokens = set(re.findall(r"\w+", query.lower()))
        m_tokens = set(re.findall(r"\w+", memory.text.lower()))
        overlap = len(q_tokens & m_tokens)
        idf_weight = math.log(1 + overlap)  # rough IDF proxy

        # Recency: exponential decay with half-life of 1 hour
        age_hours = (time.time() - memory.timestamp) / 3600
        recency = math.exp(-0.693 * age_hours)  # 0.693 ≈ ln(2)

        return idf_weight + 0.2 * recency

    def should_stop(self, scores: list[float], budget: int) -> bool:
        """
        Adaptive stopping: halt traversal when the marginal score gain
        falls below a threshold OR the budget is exhausted.
        """
        if len(scores) >= budget:
            return True
        if len(scores) >= 2 and (scores[-2] - scores[-1]) < 0.05:
            return True
        return False


# ---------------------------------------------------------------------------
# Memory Store (multi-relational graph)
# ---------------------------------------------------------------------------

class MemoryStore:
    def __init__(self):
        self._memories: list[Memory] = []
        self._ctrl = System1Controller()

    def add(self, text: str) -> Memory:
        mem_type, scores = self._ctrl.classify(text)
        entities = self._ctrl.extract_entities(text)
        mem = Memory(
            text=text,
            mem_type=mem_type,
            entity_tags=entities,
            episodic_score=scores["episodic"],
            semantic_score=scores["semantic"],
            procedural_score=scores["procedural"],
            preference_score=scores["preference"],
        )
        # Build edges to existing memories
        mem_id = len(self._memories)
        for existing_id, existing in enumerate(self._memories):
            rel = self._infer_relation(mem, existing)
            if rel:
                mem.edges[existing_id] = rel
                existing.edges[mem_id] = rel
        self._memories.append(mem)
        return mem

    def _infer_relation(self, a: Memory, b: Memory) -> RelType | None:
        shared_entities = set(a.entity_tags) & set(b.entity_tags)
        if shared_entities:
            return "entity"
        # Temporal: if both are episodic and close in time
        if a.mem_type == "episodic" and b.mem_type == "episodic":
            if abs(a.timestamp - b.timestamp) < 60:
                return "temporal"
        # Semantic: similar type scores
        sim = 1 - abs(a.semantic_score - b.semantic_score)
        if sim > 0.8 and a.mem_type == b.mem_type == "semantic":
            return "semantic"
        return None

    def retrieve(self, query: str, budget: int = 5) -> list[Memory]:
        """System-1-guided retrieval."""
        ctrl = self._ctrl
        rel_types = ctrl.route_query(query)

        # Filter candidates by relation types
        candidates = [
            m for m in self._memories
            if m.mem_type in _TYPES_FOR_RELS[rel_types[0]]
        ]
        if not candidates:
            candidates = list(self._memories)

        # Score + adaptive stop
        scored = sorted(
            candidates,
            key=lambda m: ctrl.score_candidate(query, m),
            reverse=True,
        )
        selected, scores_so_far = [], []
        for mem in scored:
            score = ctrl.score_candidate(query, mem)
            scores_so_far.append(score)
            selected.append(mem)
            if ctrl.should_stop(scores_so_far, budget):
                break
        return selected


# Map a primary relation type to which memory types are most relevant
_TYPES_FOR_RELS: dict[RelType, tuple[MemoryType, ...]] = {
    "causal":   ("episodic", "semantic"),
    "temporal": ("episodic",),
    "entity":   ("episodic", "semantic", "preference"),
    "semantic": ("semantic", "procedural"),
}


# ---------------------------------------------------------------------------
# System-2 Synthesiser (LLM-based, called once per query)
# ---------------------------------------------------------------------------

def synthesize(query: str, evidence: list[Memory]) -> str:
    """
    Pass query + retrieved evidence to an LLM for final answer synthesis.
    Uses openai if OPENAI_API_KEY is set, anthropic if ANTHROPIC_API_KEY is
    set, otherwise returns a deterministic mock.
    """
    context = "\n".join(f"- {m.text}" for m in evidence)

    # -- Mock (zero-dependency) -------------------------------------------
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        if not evidence:
            return "(no relevant memory found)"
        return f"[MOCK] Based on memory: {evidence[0].text}"

    # -- Real API mode ---------------------------------------------------
    system_prompt = (
        "You are a helpful assistant with access to a memory store. "
        "Answer the user's question using ONLY the evidence provided. "
        "Be concise."
    )
    user_message = f"Evidence:\n{context}\n\nQuestion: {query}"

    if os.getenv("OPENAI_API_KEY"):
        import openai  # type: ignore
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()

    # Anthropic fallback
    import anthropic  # type: ignore
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    store = MemoryStore()

    observations = [
        "Alice prefers concise, bullet-point responses over long paragraphs.",
        "Python's GIL prevents true multi-threading for CPU-bound tasks.",
        "The project deadline was moved to October 3rd during the team meeting.",
        "To deploy the service: run `docker build -t myapp . && docker push myapp`.",
        "Alice mentioned she wants the UI redesign done before the client demo.",
        "The client demo is scheduled for September 30th.",
        "Async I/O with asyncio bypasses the GIL by yielding control between tasks.",
    ]

    print("=== Jev-Mem PoC — Building Memory ===\n")
    for obs in observations:
        mem = store.add(obs)
        print(f"[{mem.mem_type:>12}]  {obs[:70]}")
        if mem.edges:
            print(f"              ^ linked to memories: {list(mem.edges.keys())}")

    queries = [
        "When is the client demo?",
        "Why does Python have threading limitations?",
        "How do I deploy the service?",
        "What does Alice prefer?",
    ]

    print("\n=== Queries ===\n")
    for q in queries:
        t0 = time.perf_counter()
        evidence = store.retrieve(q)
        answer   = synthesize(q, evidence)
        elapsed  = time.perf_counter() - t0
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"   (retrieved {len(evidence)} memories in {elapsed*1000:.1f} ms)\n")


if __name__ == "__main__":
    run_demo()
