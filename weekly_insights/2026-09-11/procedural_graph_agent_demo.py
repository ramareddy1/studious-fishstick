"""
Procedural Graph Agent — Proof-of-Concept
Based on: "Procedural Graphs: Self-Evolving Execution Structures for LLM Agents"
          arXiv:2609.09153 (September 8, 2026)

Concept: organise procedural knowledge as (procedure, relation, procedure) triplets,
similar to a knowledge graph but for *what-to-do* reasoning. An agent navigating
this graph gets step-level guidance at each node, reducing goal-drift over long
task horizons.

No external dependencies — runs with the Python standard library only.
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProcNode:
    name: str
    description: str
    success_count: int = 0
    fail_count: int = 0


@dataclass
class ProcEdge:
    relation: str       # e.g. "leads_to", "retries_with", "escalates_to"
    weight: float = 1.0  # higher = more preferred path


class ProcedureGraph:
    """Lightweight directed graph of (procedure → relation → procedure) triplets."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ProcNode] = {}
        # adjacency: src_name → [(dst_name, edge)]
        self.edges: Dict[str, List[Tuple[str, ProcEdge]]] = {}

    # ------------------------------------------------------------------ build

    def add_node(self, name: str, description: str) -> None:
        self.nodes[name] = ProcNode(name=name, description=description)
        self.edges.setdefault(name, [])

    def add_edge(self, src: str, relation: str, dst: str, weight: float = 1.0) -> None:
        self.edges.setdefault(src, []).append(
            (dst, ProcEdge(relation=relation, weight=weight))
        )

    # ---------------------------------------------------------------- guidance

    def guidance(self, current: str) -> str:
        """
        Translate the local subgraph around `current` into a short natural-language
        guidance string — stand-in for the paper's LLM-based guidance model.
        """
        node = self.nodes.get(current)
        if not node:
            return "Unknown node — cannot provide guidance."

        neighbours = self.edges.get(current, [])
        if not neighbours:
            return f"[{current}] Terminal step: {node.description}. No further steps."

        options = "; ".join(
            f"→ {dst} ({e.relation}, w={e.weight:.2f})" for dst, e in neighbours
        )
        return (
            f"[{current}] {node.description}\n"
            f"  Next options: {options}"
        )

    # ------------------------------------------------------------------ navigate

    def next_step(self, current: str) -> Optional[str]:
        """
        Choose the next node by weighted random selection (mimics the solver
        using guidance to pick a specific action).
        """
        choices = self.edges.get(current, [])
        if not choices:
            return None
        names = [dst for dst, _ in choices]
        weights = [e.weight for _, e in choices]
        return random.choices(names, weights=weights, k=1)[0]

    # ----------------------------------------------------------------- evolve

    def update(self, path: List[str], succeeded: bool) -> None:
        """
        Self-evolve: reinforce edges on successful paths, penalise failed ones.
        This is the lightweight stand-in for the graph's learning/evolution mechanism.
        """
        delta = +0.2 if succeeded else -0.1
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            for d, edge in self.edges.get(src, []):
                if d == dst:
                    edge.weight = max(0.05, edge.weight + delta)
        if succeeded:
            self.nodes[path[-1]].success_count += 1
        else:
            self.nodes[path[-1]].fail_count += 1


# ---------------------------------------------------------------------------
# Build a sample task graph: "answer a user question with web research"
# ---------------------------------------------------------------------------

def build_research_agent_graph() -> ProcedureGraph:
    g = ProcedureGraph()

    g.add_node("start",         "Receive user question and decompose into sub-queries.")
    g.add_node("search_web",    "Issue web search for each sub-query.")
    g.add_node("filter_results","Filter search results for relevance and freshness.")
    g.add_node("extract_facts", "Extract key facts from filtered results.")
    g.add_node("synthesise",    "Synthesise facts into a coherent draft answer.")
    g.add_node("verify",        "Cross-check draft claims against source documents.")
    g.add_node("respond",       "Deliver final answer to user.")
    g.add_node("clarify",       "Ask user for clarification if query is ambiguous.")
    g.add_node("retry_search",  "Reformulate query and re-issue search.")

    # Happy path
    g.add_edge("start",         "leads_to",     "search_web",     weight=1.5)
    g.add_edge("start",         "escalates_to", "clarify",        weight=0.5)
    g.add_edge("search_web",    "leads_to",     "filter_results", weight=1.5)
    g.add_edge("search_web",    "retries_with", "retry_search",   weight=0.5)
    g.add_edge("filter_results","leads_to",     "extract_facts",  weight=1.0)
    g.add_edge("extract_facts", "leads_to",     "synthesise",     weight=1.0)
    g.add_edge("synthesise",    "leads_to",     "verify",         weight=1.0)
    g.add_edge("verify",        "leads_to",     "respond",        weight=1.2)
    g.add_edge("verify",        "retries_with", "extract_facts",  weight=0.4)
    g.add_edge("clarify",       "leads_to",     "search_web",     weight=1.0)
    g.add_edge("retry_search",  "leads_to",     "filter_results", weight=1.0)

    return g


# ---------------------------------------------------------------------------
# Simulate an agent episode
# ---------------------------------------------------------------------------

def run_episode(g: ProcedureGraph, verbose: bool = True) -> Tuple[List[str], bool]:
    path: List[str] = ["start"]
    current = "start"

    if verbose:
        print("\n  Guidance at each step:")

    for _ in range(15):          # cap at 15 steps to avoid infinite loops
        if verbose:
            indent = "    "
            print(textwrap.indent(g.guidance(current), indent))

        nxt = g.next_step(current)
        if nxt is None:
            break
        path.append(nxt)
        current = nxt
        if current == "respond":
            break

    succeeded = (current == "respond")
    return path, succeeded


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)

    print("=" * 60)
    print("Procedural Graph Agent Demo")
    print("Based on arXiv:2609.09153 — Yuxing Lu et al., Sep 2026")
    print("=" * 60)

    g = build_research_agent_graph()

    # ---- single annotated episode ----------------------------------------
    print("\n[1] Single annotated episode (step-by-step guidance)\n")
    path, ok = run_episode(g, verbose=True)
    print(f"\n  Path taken : {' → '.join(path)}")
    print(f"  Outcome    : {'SUCCESS' if ok else 'INCOMPLETE'}")
    g.update(path, ok)

    # ---- 50 self-evolution rounds -----------------------------------------
    print("\n[2] Running 50 self-evolution rounds (success outcome injected)\n")
    for episode in range(50):
        # Randomly inject 70% success rate to let the graph learn
        p, _ = run_episode(g, verbose=False)
        g.update(p, succeeded=(random.random() < 0.7))

    # ---- show evolved edge weights ----------------------------------------
    print("[3] Evolved edge weights after 51 episodes\n")
    print(f"  {'Source':20s} {'→ Dest':20s} {'Relation':20s} {'Weight':>7}")
    print("  " + "-" * 72)
    for src, adj in g.edges.items():
        for dst, edge in sorted(adj, key=lambda x: -x[1].weight):
            print(f"  {src:20s} {'→ '+dst:20s} {edge.relation:20s} {edge.weight:7.3f}")

    # ---- key takeaway -------------------------------------------------------
    print("\n[4] Key takeaway\n")
    takeaway = """\
    Procedural Graphs inject explicit structural knowledge ('what to do
    next') into an LLM agent without any fine-tuning.  Each node provides
    concise guidance drawn from the local subgraph, reducing goal-drift on
    long-horizon tasks.  After each episode the graph self-evolves: edges on
    successful paths grow stronger, nudging the solver toward proven routes
    on the next run.

    Apply this to your own agents by:
      1. Mapping your task's logical steps as ProcNodes.
      2. Connecting them with labelled edges (relation names are free-form).
      3. At each step, pass g.guidance(current_step) as a system-prompt
         prefix to your LLM — it gets concise next-step context.
      4. After each run, call g.update(path, succeeded) to keep the graph
         improving.
    """
    print(textwrap.dedent(takeaway))
    print("=" * 60)


if __name__ == "__main__":
    main()
