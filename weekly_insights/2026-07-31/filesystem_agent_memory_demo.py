"""
Filesystem-Based Memory for LLM Agents — Proof-of-Concept
Based on: arxiv:2607.26637 (Zhou et al., July 29 2026)

The core idea: an LLM agent stores long-term memory as a directory tree of
markdown files.  The agent reads, writes, and reorganises these files itself.
No vector database, no special infrastructure — just plain files.

Run with no dependencies:
    python filesystem_agent_memory_demo.py
"""

import os
import json
import time
import shutil
import textwrap
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# FilesystemMemory: the core abstraction
# ---------------------------------------------------------------------------

class FilesystemMemory:
    """
    Agent memory backed by a directory tree of Markdown files.

    Directory layout (inspired by the paper's taxonomy):
        memory_root/
            facts/          # atomic, stable knowledge
            episodic/       # timestamped interaction logs
            reflections/    # higher-order summaries the agent writes itself
            index.md        # table of contents the agent maintains
    """

    def __init__(self, root: str = "./agent_memory"):
        self.root = Path(root)
        for sub in ("facts", "episodic", "reflections"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._refresh_index()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_fact(self, key: str, content: str, tags: list[str] | None = None) -> Path:
        """Store an atomic fact as key.md under facts/."""
        safe_key = key.replace(" ", "_").lower()
        path = self.root / "facts" / f"{safe_key}.md"
        tags_line = ", ".join(tags) if tags else ""
        path.write_text(
            f"# {key}\n\n"
            f"**tags:** {tags_line}\n"
            f"**updated:** {_now()}\n\n"
            f"{content}\n",
            encoding="utf-8",
        )
        self._refresh_index()
        return path

    def log_episode(self, summary: str, detail: str = "") -> Path:
        """Append a timestamped episode entry under episodic/."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self.root / "episodic" / f"{ts}.md"
        path.write_text(
            f"# Episode — {_now()}\n\n"
            f"## Summary\n{summary}\n\n"
            f"## Detail\n{detail or '(none)'}\n",
            encoding="utf-8",
        )
        self._refresh_index()
        return path

    def write_reflection(self, title: str, content: str) -> Path:
        """Store a higher-order reflection (agent's own synthesis)."""
        safe = title.replace(" ", "_").lower()
        path = self.root / "reflections" / f"{safe}.md"
        path.write_text(
            f"# Reflection: {title}\n\n"
            f"**written:** {_now()}\n\n"
            f"{content}\n",
            encoding="utf-8",
        )
        self._refresh_index()
        return path

    # ------------------------------------------------------------------
    # Read / query operations
    # ------------------------------------------------------------------

    def read(self, path: Path | str) -> str:
        return Path(path).read_text(encoding="utf-8")

    def search(self, query: str) -> list[tuple[float, Path]]:
        """
        Simple keyword search across all memory files.
        Returns (score, path) pairs sorted by descending score.
        In a real system the LLM would do semantic retrieval.
        """
        q_words = set(query.lower().split())
        results: list[tuple[float, Path]] = []
        for md in self.root.rglob("*.md"):
            text = md.read_text(encoding="utf-8").lower()
            hits = sum(1 for w in q_words if w in text)
            if hits:
                results.append((hits / len(q_words), md))
        return sorted(results, key=lambda x: -x[0])

    # ------------------------------------------------------------------
    # Evolution / maintenance operations (the paper's key contribution)
    # ------------------------------------------------------------------

    def consolidate_episodes(self, max_episodes: int = 5) -> Path | None:
        """
        When episodic files exceed max_episodes, roll them into a reflection.
        This mimics the 'sustainability' mechanism from the paper: the agent
        periodically rewrites old memories into compact reflections and prunes
        the episode log.
        """
        episodes = sorted((self.root / "episodic").glob("*.md"))
        if len(episodes) <= max_episodes:
            return None

        to_merge = episodes[:-max_episodes]
        merged_text = "\n\n---\n\n".join(
            f.read_text(encoding="utf-8") for f in to_merge
        )
        reflection_path = self.write_reflection(
            f"consolidated_{len(to_merge)}_episodes",
            f"Auto-consolidated {len(to_merge)} old episodes:\n\n{merged_text}",
        )
        for f in to_merge:
            f.unlink()
        self._refresh_index()
        return reflection_path

    def reorganise(self, moves: dict[str, str]) -> list[str]:
        """
        Rename / move files, updating the index.
        Represents the agent re-structuring its own memory tree.
        moves = {old_relative_path: new_relative_path}
        """
        done = []
        for old, new in moves.items():
            src = self.root / old
            dst = self.root / new
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                done.append(f"{old} -> {new}")
        self._refresh_index()
        return done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_index(self):
        """Rewrite index.md to reflect the current file tree."""
        lines = ["# Agent Memory Index\n", f"_updated: {_now()}_\n\n"]
        for sub in ("facts", "episodic", "reflections"):
            files = sorted((self.root / sub).glob("*.md"))
            if files:
                lines.append(f"## {sub.title()}\n")
                for f in files:
                    lines.append(f"- [{f.stem}]({f.relative_to(self.root)})\n")
                lines.append("\n")
        (self.root / "index.md").write_text("".join(lines), encoding="utf-8")

    def stats(self) -> dict:
        return {
            sub: len(list((self.root / sub).glob("*.md")))
            for sub in ("facts", "episodic", "reflections")
        }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Demo: simulate a simple agent using filesystem memory
# ---------------------------------------------------------------------------

def run_demo(mem_root: str = "/tmp/demo_agent_memory"):
    print("=" * 60)
    print("Filesystem-Based LLM Agent Memory — Demo")
    print("=" * 60)

    mem = FilesystemMemory(mem_root)

    # --- Store some facts ---
    print("\n[1] Saving atomic facts...")
    mem.save_fact("user_preference_language", "The user prefers Python.", tags=["user", "prefs"])
    mem.save_fact("user_preference_style", "The user likes concise, well-commented code.", tags=["user", "prefs"])
    mem.save_fact("project_name", "The project is called Aria, a personal assistant.", tags=["project"])
    print(f"    Facts saved. Stats: {mem.stats()}")

    # --- Log some episodes ---
    print("\n[2] Logging interaction episodes...")
    for i in range(1, 8):
        mem.log_episode(
            summary=f"Session {i}: user asked a coding question.",
            detail=f"User requested help with topic-{i}. Agent responded with a code snippet.",
        )
        time.sleep(0.01)  # ensure unique timestamps
    print(f"    Episodes logged. Stats: {mem.stats()}")

    # --- Demonstrate search ---
    print("\n[3] Searching memory for 'coding'...")
    hits = mem.search("coding Python")
    for score, path in hits[:3]:
        print(f"    score={score:.2f}  {path.relative_to(mem.root)}")

    # --- Demonstrate sustainability / consolidation ---
    print("\n[4] Consolidating old episodes (sustainability)...")
    reflection = mem.consolidate_episodes(max_episodes=3)
    if reflection:
        print(f"    Merged old episodes into: {reflection.relative_to(mem.root)}")
    print(f"    Stats after consolidation: {mem.stats()}")

    # --- Demonstrate reorganisation ---
    print("\n[5] Agent re-organises its own memory tree...")
    # Move the project fact into a subfolder
    (mem.root / "facts" / "project").mkdir(exist_ok=True)
    moved = mem.reorganise({"facts/project_name.md": "facts/project/project_name.md"})
    print(f"    Moved: {moved}")

    # --- Show index ---
    print("\n[6] Current memory index (index.md):")
    print(textwrap.indent(mem.read(mem.root / "index.md"), "    "))

    # --- Read a specific memory ---
    print("\n[7] Reading a specific fact:")
    fact_path = mem.root / "facts" / "user_preference_language.md"
    print(textwrap.indent(mem.read(fact_path), "    "))

    # Clean up
    shutil.rmtree(mem_root, ignore_errors=True)
    print("\n[Done] Temporary memory directory cleaned up.")
    print("\nKey insight from arxiv:2607.26637:")
    print("  Filesystem-backed memory is human-readable, debuggable, and")
    print("  self-organising — an agent can maintain growing knowledge without")
    print("  any vector database or special infrastructure.")


if __name__ == "__main__":
    run_demo()
