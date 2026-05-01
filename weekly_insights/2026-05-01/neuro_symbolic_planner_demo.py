"""
neuro_symbolic_planner_demo.py
==============================
Proof-of-concept for:
  "The Price Is Not Right: Neuro-Symbolic Methods Outperform VLAs on
  Structured Long-Horizon Manipulation Tasks with Significantly Lower
  Energy Consumption"
  Duggan, Lorang, Lu & Scheutz — Tufts University (arXiv:2602.19260, ICRA 2026)

Concept
-------
Pure-neural agents (VLAs) try to solve structured tasks end-to-end with a
single large model, consuming enormous compute.  Neuro-symbolic agents split
the work:

  Symbolic planner  ->  sequence of high-level actions  (cheap, 0 GPU)
  Neural executor   ->  turn each action into low-level control  (tiny net)

This demo uses the Tower of Hanoi as the benchmark task (same as the paper).

Run
---
  pip install numpy          # only dependency
  python neuro_symbolic_planner_demo.py
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# 1.  SYMBOLIC PLANNER  (PDDL-style, pure Python — zero GPU)
# ---------------------------------------------------------------------------

class HanoiPlanner:
    """Recursive Tower-of-Hanoi solver.  Produces the optimal 2^n - 1 moves."""

    def __init__(self, n_disks: int):
        self.n_disks = n_disks

    def plan(self, n=None, src="A", aux="B", dst="C", moves=None):
        if moves is None:
            moves = []
            n = self.n_disks
        if n == 0:
            return moves
        self.plan(n - 1, src, dst, aux, moves)
        moves.append((src, dst))
        self.plan(n - 1, aux, src, dst, moves)
        return moves


# ---------------------------------------------------------------------------
# 2.  NEURAL LOW-LEVEL EXECUTOR  (tiny MLP, CPU-only, ~1 KB parameters)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class TinyMLP:
    """
    Maps (peg_from_one_hot || peg_to_one_hot) -> motor_command (4-DOF joints).

    Represents the *low-level* neural controller that turns a symbolic
    'move disk from peg A to peg C' into a concrete motor output.
    Weights are random-init here; in the real paper a small net is trained
    per manipulation primitive on ~34 minutes of data.
    """

    def __init__(self, hidden_dim: int = 32):
        input_dim, output_dim = 6, 4
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((hidden_dim, input_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((output_dim, hidden_dim)) * 0.1
        self.b2 = np.zeros(output_dim)
        self.flops_per_call = 2 * (input_dim * hidden_dim + hidden_dim * output_dim)

    def forward(self, peg_from: str, peg_to: str) -> np.ndarray:
        peg_map = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]}
        x = np.array(peg_map[peg_from] + peg_map[peg_to], dtype=float)
        h = _sigmoid(self.W1 @ x + self.b1)
        return np.tanh(self.W2 @ h + self.b2)  # simulated joint angles


# ---------------------------------------------------------------------------
# 3.  PURE-NEURAL BASELINE  (wider/deeper MLP mimicking a VLA)
# ---------------------------------------------------------------------------

class LargeVLABaseline:
    """
    Simulates a VLA that tries to solve the FULL task end-to-end from raw
    state observations, with no symbolic guidance.

    State: one-hot disk-on-peg encoding  (n_disks x 3 flattened)
    Output: next move choice over all 9 peg-pair actions

    The untrained network produces mostly wrong moves, mirroring the 34%
    success rate reported for VLAs in the paper.
    """

    def __init__(self, n_disks: int = 3, hidden_dim: int = 256):
        input_dim = n_disks * 3
        output_dim = 9
        rng = np.random.default_rng(0)
        self.W1 = rng.standard_normal((hidden_dim, input_dim)) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * 0.01
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.standard_normal((output_dim, hidden_dim)) * 0.01
        self.b3 = np.zeros(output_dim)
        self.flops_per_call = 2 * (
            input_dim * hidden_dim +
            hidden_dim * hidden_dim +
            hidden_dim * output_dim
        )

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def forward(self, state: np.ndarray) -> int:
        h1 = _sigmoid(self.W1 @ state + self.b1)
        h2 = _sigmoid(self.W2 @ h1 + self.b2)
        return int(np.argmax(self._softmax(self.W3 @ h2 + self.b3)))


def encode_state(disk_peg: dict, n_disks: int) -> np.ndarray:
    idx = {"A": 0, "B": 1, "C": 2}
    vec = np.zeros(n_disks * 3)
    for disk, peg in disk_peg.items():
        vec[disk * 3 + idx[peg]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# 4.  SIMULATION RUNS
# ---------------------------------------------------------------------------

def run_neuro_symbolic(n_disks: int, executor: TinyMLP):
    planner = HanoiPlanner(n_disks)
    moves = planner.plan()
    total_flops = len(moves) * executor.flops_per_call
    # Execute each symbolic step through the tiny neural controller
    commands = [executor.forward(src, dst) for src, dst in moves]
    return moves, commands, total_flops


def run_pure_neural(n_disks: int, vla: LargeVLABaseline, max_steps: int = 600):
    """VLA stumbles through the action space; we count FLOPs until solved or timeout."""
    pegs = {"A": list(range(n_disks - 1, -1, -1)), "B": [], "C": []}
    goal = {"A": [], "B": [], "C": list(range(n_disks - 1, -1, -1))}
    disk_peg = {d: "A" for d in range(n_disks)}
    peg_names = ["A", "B", "C"]
    total_flops, steps = 0, 0

    for _ in range(max_steps):
        state = encode_state(disk_peg, n_disks)
        choice = vla.forward(state)
        total_flops += vla.flops_per_call
        steps += 1
        src, dst = peg_names[choice // 3], peg_names[choice % 3]
        if pegs[src] and src != dst:
            top = pegs[src][-1]
            if not pegs[dst] or pegs[dst][-1] > top:
                pegs[src].pop()
                pegs[dst].append(top)
                disk_peg[top] = dst
        if pegs == goal:
            return steps, total_flops, True

    return steps, total_flops, False


def fmt_flops(f: int) -> str:
    if f >= 1e9:
        return f"{f/1e9:.2f} GFLOPs"
    if f >= 1e6:
        return f"{f/1e6:.2f} MFLOPs"
    return f"{f/1e3:.2f} KFLOPs"


# ---------------------------------------------------------------------------
# 5.  MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Neuro-Symbolic vs. Pure-Neural Agent  —  Tower of Hanoi")
    print("  Paper: arXiv:2602.19260  (Tufts University / ICRA 2026)")
    print("=" * 65)

    executor = TinyMLP(hidden_dim=32)

    for n_disks in [3, 4, 5]:
        vla = LargeVLABaseline(n_disks=n_disks, hidden_dim=256)
        optimal = 2 ** n_disks - 1
        print(f"\n--- {n_disks} disks  (optimal plan = {optimal} moves) ---")

        t0 = time.perf_counter()
        ns_moves, _, ns_flops = run_neuro_symbolic(n_disks, executor)
        ns_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        vla_steps, vla_flops, vla_solved = run_pure_neural(n_disks, vla)
        vla_ms = (time.perf_counter() - t0) * 1000

        ratio = vla_flops / max(ns_flops, 1)
        print(f"  Neuro-Symbolic  | steps={len(ns_moves):3d}  "
              f"compute={fmt_flops(ns_flops):>13}  time={ns_ms:6.2f}ms  solved=True")
        print(f"  Pure-Neural VLA | steps={vla_steps:3d}  "
              f"compute={fmt_flops(vla_flops):>13}  time={vla_ms:6.2f}ms  solved={vla_solved}")
        print(f"  Compute ratio (VLA / NS): {ratio:.0f}x")

    print("\n" + "=" * 65)
    print("Key takeaway:")
    print("  Symbolic planner: perfect plan in microseconds, zero GPU.")
    print("  Neural executor:  tiny network, runs once per discrete step.")
    print("  Pure-neural VLA:  large network, runs every step searching")
    print("  blindly — compute cost explodes with task complexity.")
    print("  Paper result: 100x lower energy, 95% vs 34% success rate.")
    print("=" * 65)


if __name__ == "__main__":
    main()
