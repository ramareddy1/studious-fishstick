# Weekly AI Insight — 2026-09-25

## Title
**Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents**

## Source
- **Paper:** [arXiv:2609.23986](https://arxiv.org/abs/2609.23986)
- **Submitted:** September 21, 2026
- **Authors:** Dongming Jiang, Yi Li, Bingzhe Li — University of Texas at Dallas
- **Official GitHub:** [libingzheren/Jev-Mem](https://github.com/libingzheren/Jev-Mem)
- **Open fork:** [furkan-bayram132/Jev-Mem-Opencode](https://github.com/furkan-bayram132/Jev-Mem-Opencode)

---

## Why It Matters

Every AI agent that needs to remember things faces the same hidden bottleneck: the memory system calls an LLM for every little bookkeeping decision — *"is this fact worth storing?", "which existing memories does this relate to?", "how many candidates do I need before I can stop searching?"*. Those calls are expensive, sequential, and largely unnecessary.

**Jev-Mem** solves this by importing an idea from cognitive science: human memory is managed by fast, automatic *System-1* processes, not by slow deliberative *System-2* reasoning. System-2 (the LLM) is reserved for the one task it's actually needed for: synthesising a final answer from evidence that System-1 has already surfaced.

### The Three-Plane Architecture

| Plane | What it does | Cost |
|-------|-------------|------|
| **System-1 controller** | Classify observations into memory types (episodic / semantic / procedural / preference), build relational graph (semantic, temporal, causal, entity edges), route queries, allocate retrieval budgets, score candidates, stop early | Microseconds — pure heuristics / tiny classifier |
| **Multi-relational memory store** | Structured graph linking memories across four relation types | O(n) |
| **System-2 LLM** | Read the curated evidence list and compose a grounded answer | One LLM call per user query |

### Benchmark Results (LoCoMo, GPT-4o-mini backbone)

| Metric | Best prior baseline | **Jev-Mem** | Delta |
|--------|-------------------|------------|-------|
| LLM-as-a-Judge score | 0.700 | **0.777** | +11.0% |
| Memory construction time | 1,043 s | **158 s** | 6.6× faster |
| Average query latency | 1.47 s | **0.93 s** | −36.7% |

The quality goes *up* while the compute cost goes *down* — the right tool for each subtask turns out to be better at it than a general-purpose LLM.

### Why This Generalises Beyond Memory

The pattern — offload high-frequency, low-complexity decisions to a fast non-LLM controller; reserve the LLM for rare, high-value synthesis — applies broadly:

- Tool-selection routing in agentic pipelines
- Deciding when to call a web search vs. use cached knowledge
- Real-time dialogue state tracking
- Code review triage before deep analysis

Any system that repeatedly asks an LLM "what should I do next?" is a candidate for a System-1 control layer.

---

## Implementation Details & GitHub Links

The official implementation is available at:
- **https://github.com/libingzheren/Jev-Mem**

The repo includes:
- `memory/` — memory construction, relation graph, retrieval engine
- `tests/` — offline regression tests against the LoCoMo benchmark
- `examples/` — synthetic inputs for quick local validation

**Minimum viable setup:** Python 3.10+, any OpenAI-compatible LLM API. The System-1 controller runs entirely without a GPU.

---

## Proof-of-Concept Script

See [`poc_jev_mem.py`](poc_jev_mem.py) in this folder for a fully self-contained demo.

The script implements all three planes:
- A `System1Controller` that classifies memories, infers relational edges, routes queries by type, and applies adaptive early-exit stopping.
- A `MemoryStore` that maintains the multi-relational graph.
- A `synthesize()` function that calls GPT-4o-mini or Claude Haiku for final-answer synthesis (falls back to a zero-dependency mock if no API key is set).

**To run (zero dependencies, mock mode):**
```bash
python poc_jev_mem.py
```

**With OpenAI (real synthesis):**
```bash
OPENAI_API_KEY=sk-... python poc_jev_mem.py
```

**With Anthropic:**
```bash
ANTHROPIC_API_KEY=sk-ant-... python poc_jev_mem.py
```

Expected output shows each observation classified into a memory type (episodic / semantic / procedural / preference), any inferred relational edges, and then grounded answers to four queries — all with sub-millisecond System-1 retrieval.
