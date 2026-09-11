# Weekly AI Insight — 2026-09-11

## Title
**Procedural Graphs: Self-Evolving Execution Structures for LLM Agents**

## Source
- **Paper:** [arXiv:2609.09153](https://arxiv.org/abs/2609.09153)
- **Submitted:** September 8, 2026
- **Authors:** Yuxing Lu, Yicheng Chen, Shanchan Wu, Sercan Ö. Arık
- **Topics:** cs.AI · cs.CL · cs.MA

---

## Why It Matters

LLM agents are increasingly deployed to tackle multi-step, tool-using tasks — web research, code generation, data pipelines. The dominant approach hands the agent a plain text history and asks it to decide each action from scratch. This works fine for short tasks, but over long horizons the agent loses its thread: it repeats steps it already tried, skips mandatory checks, or calls the wrong tool at the wrong time.

**Procedural Graphs** (PG) is a structural solution. The core analogy:

| Knowledge Graph | Procedural Graph |
|---|---|
| Stores *what is* as `(entity, relation, entity)` | Stores *what to do* as `(procedure, relation, procedure)` |
| Answers: "What is the capital of France?" | Answers: "What step comes after filtering search results?" |

A Procedural Graph imposes an explicit skeleton of procedural knowledge on top of the LLM. At every decision step, a lightweight **guidance model** inspects the agent's current node in the graph, reads the local neighbourhood (which steps are adjacent, what relation links them), and emits a short natural-language guidance string. This guidance is passed into the solver LLM as extra context — it biases the next action without hard-coding it.

The graph **self-evolves**: after each episode the edge weights are updated based on outcome. Successful paths get reinforced; failed paths are penalised. Over many runs the graph organically builds a learned prior of which procedural routes actually work for a given task.

### Why This Is Accessible

- Works with any LLM (API or local) — no fine-tuning required
- The graph is a plain Python data structure (nodes + weighted edges)
- Guidance injection is a single system-prompt prefix addition
- Self-evolution is three lines of weight arithmetic

---

## Implementation Details

### Existing GitHub Repositories

No official code release was available at time of writing (September 11, 2026). No community re-implementations were found.

### Proof-of-Concept Script

See [`procedural_graph_agent_demo.py`](./procedural_graph_agent_demo.py) in this folder.

**Quick start (zero dependencies):**
```bash
python procedural_graph_agent_demo.py
```

**What the script does:**

1. Models the task *"answer a user question with web research"* as a Procedural Graph with 9 nodes and 11 directed weighted edges.
2. Runs one annotated episode showing the step-by-step guidance output at each node.
3. Runs 50 self-evolution rounds and prints the evolved edge weights — the happy-path edges (`start → search_web`, `extract_facts → synthesise`) grow to ≈5–7× their starting weight as the graph learns.

**Sample output (abbreviated):**
```
[1] Single annotated episode
    [start] Receive user question and decompose into sub-queries.
      Next options: → search_web (leads_to, w=1.50); → clarify (escalates_to, w=0.50)
    ...
  Path taken : start → search_web → filter_results → ... → respond
  Outcome    : SUCCESS

[3] Evolved edge weights after 51 episodes
  filter_results  → extract_facts  leads_to   6.400
  extract_facts   → synthesise     leads_to   6.900
  ...
```

### Applying This to Your Own Agent

```python
# 1. Build your task graph
g = ProcedureGraph()
g.add_node("plan",    "Break the goal into sub-tasks.")
g.add_node("execute", "Run the chosen sub-task.")
g.add_node("review",  "Verify the result meets the acceptance criteria.")
g.add_node("done",    "Return result to caller.")
g.add_edge("plan",    "leads_to",     "execute")
g.add_edge("execute", "leads_to",     "review")
g.add_edge("review",  "leads_to",     "done")
g.add_edge("review",  "retries_with", "execute")  # retry loop

# 2. Inject guidance at each step
current = "plan"
while current:
    guidance = g.guidance(current)        # <-- feed into your LLM
    response = your_llm(system=guidance)  # <-- LLM picks the action
    current  = g.next_step(current)

# 3. Evolve after each episode
g.update(path_taken, succeeded=True)
```
