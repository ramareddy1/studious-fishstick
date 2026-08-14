"""
AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses
Proof-of-concept based on arXiv:2608.12307

Core idea: A strong "builder" LLM generates a structured harness — a
combination of a system prompt template, deterministic parsing code, and
answer-format enforcement — that wraps calls to a weaker "target" LLM.
The weaker model benefits without any retraining.

This script works in two modes:
  1. MOCK mode (default, zero dependencies) — simulates everything locally.
  2. API  mode — set OPENAI_API_KEY env var; runs against real models.

Usage:
    python ai4ai_harness_poc.py           # mock mode
    OPENAI_API_KEY=sk-... python ai4ai_harness_poc.py  # API mode
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Harness:
    """An inference-time harness produced by a builder model."""
    task_description: str
    system_prompt: str          # injected before every target-model call
    cot_prefix: str             # chain-of-thought scaffolding added to user turn
    answer_extractor: Callable[[str], str]  # deterministic post-processor
    routing_rules: list[str] = field(default_factory=list)  # optional hints


@dataclass
class HarnessResult:
    raw_output: str
    extracted_answer: str
    harness_applied: bool = True


# ---------------------------------------------------------------------------
# Mock LLM (used when no API key is present)
# ---------------------------------------------------------------------------

class MockLLM:
    """Returns canned responses that simulate weak/strong models."""

    def __init__(self, strength: str = "weak"):
        self.strength = strength

    # Patterns for Theory-of-Mind questions → correct answers
    _ANSWERS = {
        "marble":    "basket",
        "keys":      "table",
        "chocolate": "blue",
    }

    def complete(self, system: str, user: str) -> str:
        if self.strength == "weak":
            return self._weak_response(system, user)
        else:
            return self._builder_harness_output()

    def _weak_response(self, system: str, user: str) -> str:
        harness_active = "Theory-of-Mind" in system or "DIRECTLY WITNESSED" in system

        # Detect which object this question is about
        obj_answer = None
        for keyword, correct in self._ANSWERS.items():
            if keyword in user:
                obj_answer = correct
                break

        if harness_active and obj_answer:
            # With the harness the weak model follows the structured CoT and
            # lands on the correct belief-based answer
            return (
                f"Step-by-step reasoning:\n"
                f"- Characters present: Sally/Anne or similar\n"
                f"- Object moved: original location → new location\n"
                f"- Who witnessed the move: Anne (not the owner)\n"
                f"- Owner's last known location: {obj_answer}\n"
                f"Therefore, the character will look in: the {obj_answer}"
            )
        else:
            # Without harness: weak model conflates belief with reality
            return (
                f"Hmm, the object was moved, so it's in the new place now. "
                f"I think they'll look wherever it actually is. "
                f"It's hard to say, could be either location."
            )

    def _builder_harness_output(self) -> str:
        return textwrap.dedent("""
            SYSTEM_PROMPT:
            You are a Theory-of-Mind reasoning assistant. When given a story:
            1. Identify each character and what they DIRECTLY WITNESSED.
            2. Track object location changes — note which characters were present for each move.
            3. Determine what each character BELIEVES (based only on what they saw).
            4. Answer based on the character's belief, not reality.
            END

            COT_PREFIX:
            Step-by-step reasoning:
            - Characters present: [list them]
            - Object moved: [from → to]
            - Who witnessed the move: [list]
            - Sally's last known location of object: [answer]
            Therefore, Sally will look in:
            END

            ANSWER_PATTERN: (?i)therefore[^:]*:\s*(?:the\s+)?(\w+)
        """).strip()


# ---------------------------------------------------------------------------
# Builder: generates a harness for a given task type
# ---------------------------------------------------------------------------

def build_harness(task_description: str, builder_llm, few_shot_examples: list[dict]) -> Harness:
    """
    Asks the builder model to produce a harness for the described task.
    In the real paper this is iterated over a validation set; here we do
    a single round for clarity.
    """
    examples_text = "\n\n".join(
        f"Example {i+1}:\nQ: {ex['question']}\nA: {ex['answer']}"
        for i, ex in enumerate(few_shot_examples)
    )

    builder_prompt = textwrap.dedent(f"""
        Task: {task_description}

        Here are some example question-answer pairs:
        {examples_text}

        Generate a harness with three sections:
          SYSTEM_PROMPT: ... END
          COT_PREFIX: ... END
          ANSWER_PATTERN: <regex>

        The system prompt should maximally constrain how the target model reasons.
        The CoT prefix should scaffold step-by-step reasoning.
        The answer pattern should reliably extract the final answer from raw output.
    """).strip()

    raw_harness = builder_llm.complete(system="You are a harness-engineering expert.", user=builder_prompt)
    return _parse_harness(task_description, raw_harness)


def _parse_harness(task_description: str, raw: str) -> Harness:
    """Parse builder output into a typed Harness object."""
    def extract_block(tag: str) -> str:
        m = re.search(rf"{tag}:\s*(.*?)\s*END", raw, re.DOTALL)
        return m.group(1).strip() if m else ""

    system_prompt = extract_block("SYSTEM_PROMPT")
    cot_prefix    = extract_block("COT_PREFIX")

    pattern_m = re.search(r"ANSWER_PATTERN:\s*(.+)", raw)
    pattern   = pattern_m.group(1).strip() if pattern_m else r"\b(\w+)\s*$"

    def extractor(text: str) -> str:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip().lower()
        # fallback: last word
        words = re.findall(r"\b\w+\b", text)
        return words[-1].lower() if words else ""

    return Harness(
        task_description=task_description,
        system_prompt=system_prompt,
        cot_prefix=cot_prefix,
        answer_extractor=extractor,
        routing_rules=["If the question mentions 'moved', apply the belief-tracking template."],
    )


# ---------------------------------------------------------------------------
# Runner: applies harness to a weaker target model
# ---------------------------------------------------------------------------

def run_with_harness(question: str, harness: Harness, target_llm) -> HarnessResult:
    """Wrap the target model call with the harness."""
    augmented_user = f"{harness.cot_prefix}\n\nQuestion: {question}"
    raw = target_llm.complete(system=harness.system_prompt, user=augmented_user)
    answer = harness.answer_extractor(raw)
    return HarnessResult(raw_output=raw, extracted_answer=answer)


def run_without_harness(question: str, target_llm) -> HarnessResult:
    """Baseline: target model with no harness."""
    raw = target_llm.complete(system="Answer the question.", user=question)
    # naive extraction: last word
    words = re.findall(r"\b\w+\b", raw)
    answer = words[-1].lower() if words else ""
    return HarnessResult(raw_output=raw, extracted_answer=answer, harness_applied=False)


# ---------------------------------------------------------------------------
# Optional: real OpenAI backend
# ---------------------------------------------------------------------------

def make_openai_llm(model: str):
    """Returns a real LLM if openai is installed and OPENAI_API_KEY is set."""
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()

        class _OpenAILLM:
            def complete(self, system: str, user: str) -> str:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=300,
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()

        return _OpenAILLM()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

THEORY_OF_MIND_QUESTIONS = [
    {
        "question": (
            "Sally puts her marble in the basket and leaves the room. "
            "Anne moves the marble to the box while Sally is gone. "
            "Where will Sally look for her marble?"
        ),
        "answer": "basket",
    },
    {
        "question": (
            "Bob places his keys on the kitchen table, then goes to work. "
            "His wife moves the keys to the drawer. "
            "Where will Bob look for his keys when he comes home?"
        ),
        "answer": "table",
    },
    {
        "question": (
            "Maxi puts his chocolate in the blue cupboard. "
            "While Maxi is in the garden, his mother moves the chocolate to the green cupboard. "
            "Where will Maxi look for his chocolate?"
        ),
        "answer": "blue",
    },
]

FEW_SHOT = THEORY_OF_MIND_QUESTIONS[:1]   # builder sees 1 example for harness generation


def _sep(label: str):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print('─'*60)


def main():
    use_api = bool(os.getenv("OPENAI_API_KEY"))

    if use_api:
        print("API mode: using gpt-4o-mini (target) and gpt-4o (builder).")
        builder_llm = make_openai_llm("gpt-4o")
        target_llm  = make_openai_llm("gpt-4o-mini")
        if not builder_llm or not target_llm:
            print("openai package not installed; falling back to mock mode.")
            use_api = False

    if not use_api:
        print("Mock mode: simulating weak target and strong builder.\n")
        builder_llm = MockLLM(strength="strong")
        target_llm  = MockLLM(strength="weak")

    # 1. Build the harness
    _sep("STEP 1 — Builder generates harness")
    harness = build_harness(
        task_description="Theory-of-Mind: predict where a character believes an object is located.",
        builder_llm=builder_llm,
        few_shot_examples=FEW_SHOT,
    )
    print(f"System prompt (truncated): {harness.system_prompt[:120]}…")
    print(f"CoT prefix    (truncated): {harness.cot_prefix[:120]}…")

    # 2. Evaluate
    correct_no_harness  = 0
    correct_with_harness = 0
    total = len(THEORY_OF_MIND_QUESTIONS)

    for i, item in enumerate(THEORY_OF_MIND_QUESTIONS):
        q, gold = item["question"], item["answer"]

        no_h = run_without_harness(q, target_llm)
        with_h = run_with_harness(q, harness, target_llm)

        correct_no_harness   += int(gold in no_h.extracted_answer)
        correct_with_harness += int(gold in with_h.extracted_answer)

        _sep(f"Q{i+1}: {q[:60]}…")
        print(f"  Gold answer          : {gold}")
        print(f"  Without harness      : '{no_h.extracted_answer}'  "
              f"{'✓' if gold in no_h.extracted_answer else '✗'}")
        print(f"  Raw (no harness)     : {no_h.raw_output[:90]}…")
        print()
        print(f"  With harness         : '{with_h.extracted_answer}'  "
              f"{'✓' if gold in with_h.extracted_answer else '✗'}")
        print(f"  Raw (with harness)   : {with_h.raw_output[:90]}…")

    _sep("RESULTS")
    print(f"  Without harness : {correct_no_harness}/{total} correct")
    print(f"  With harness    : {correct_with_harness}/{total} correct")
    print()
    if correct_with_harness > correct_no_harness:
        print("  The harness improved the weak model's accuracy!")
    print()
    print("Paper: arXiv:2608.12307 — AI4AI at Test-Time")
    print("Key insight: gains come from deterministic parsing + format")
    print("enforcement, not from prompting the model to 'think more'.")


if __name__ == "__main__":
    main()
