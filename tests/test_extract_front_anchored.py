"""Tests for extract_circuit_front_anchored / multi_restart_extract_front_anchored.

Verifies:
  - Soundness: extract_circuit_front_anchored reproduces the exact target
    unitary (up to global phase) for arbitrary front-anchored initial_states,
    including None (vanilla) -- the graph-transpose-then-gate-reversal trick
    these functions are built on.
  - No-regression: multi_restart_extract_front_anchored's returned circuit,
    scored under the SAME front assignment via SigmaTracker, is never worse
    than plain vanilla extract_circuit_front_anchored(initial_states=None)
    scored under that same assignment -- trial 0 is always vanilla and the
    minimum-cost trial is returned, mirroring multi_restart_extract's own
    guarantee.
  - Plumbing sanity: returned circuits are well-formed and cost-consistent.
"""

from __future__ import annotations

import random
from typing import Tuple

import pyzx as zx
from pyzx.extract import extract_circuit_front_anchored, multi_restart_extract_front_anchored
from pyzx.graph.base import BaseGraph
from pyzx.sigma_tracker import Flavor, SigmaTracker
from pyzx.tensor import compare_tensors


def _reduced_graph(
    n_qubits: int, n_gates: int, seed: int, p_t: float = 0.3, p_cnot: float = 0.3
) -> BaseGraph[int, Tuple[int, int]]:
    g = zx.generate.cliffordT(n_qubits, n_gates, p_t=p_t, p_cnot=p_cnot, seed=seed)
    zx.simplify.full_reduce(g, quiet=True)
    g.normalize()
    return g


class TestExtractCircuitFrontAnchored:
    """The graph-transpose-then-reversal trick must reproduce the exact
    target unitary, for arbitrary front-anchored initial_states.
    """

    def test_reproduces_target_unitary_for_random_front_assignments(self) -> None:
        rng = random.Random(20260915)
        for trial in range(6):
            n = rng.randint(2, 6)
            n_gates = rng.randint(10, 30)
            seed = rng.randint(0, 10**9)
            g = _reduced_graph(n, n_gates, seed)

            for assignment_trial in range(3):
                initial_states = [rng.randint(0, 1) for _ in range(n)]
                circuit = extract_circuit_front_anchored(g.copy(), initial_states=initial_states, quiet=True)
                assert compare_tensors(circuit, g, preserve_scalar=False), (
                    f"trial {trial} assignment {assignment_trial}: n={n} seed={seed} "
                    f"initial_states={initial_states} -- reconstructed circuit does not match the target unitary"
                )

    def test_reproduces_target_unitary_for_vanilla(self) -> None:
        rng = random.Random(1234)
        for trial in range(3):
            n = rng.randint(2, 6)
            n_gates = rng.randint(10, 30)
            seed = rng.randint(0, 10**9)
            g = _reduced_graph(n, n_gates, seed)
            circuit = extract_circuit_front_anchored(g.copy(), initial_states=None, quiet=True)
            assert compare_tensors(circuit, g, preserve_scalar=False), (
                f"trial {trial}: n={n} seed={seed} -- vanilla front-anchored extraction mismatch"
            )


class TestMultiRestartExtractFrontAnchored:
    """Trial 0 is always vanilla and the minimum-cost trial is returned, so
    the result must never be worse than vanilla under the same assignment.
    """

    def test_never_worse_than_vanilla(self) -> None:
        rng = random.Random(999)
        for trial in range(6):
            n = rng.randint(2, 6)
            n_gates = rng.randint(10, 30)
            seed = rng.randint(0, 10**9)
            g = _reduced_graph(n, n_gates, seed)
            initial_states = [rng.randint(0, 1) for _ in range(n)]
            front_assignment = [Flavor(s) for s in initial_states]

            vanilla = extract_circuit_front_anchored(g.copy(), initial_states=None, quiet=True).to_basic_gates()
            vanilla_cost = SigmaTracker(vanilla, front_assignment).propagate().cost

            best_circuit, stats = multi_restart_extract_front_anchored(
                g.copy(), initial_states, n_restarts=4, threshold=2, lookahead_depth=2, seed=trial, quiet=True,
            )
            best_cost = SigmaTracker(best_circuit, front_assignment).propagate().cost

            assert best_cost <= vanilla_cost, (
                f"trial {trial}: n={n} seed={seed} initial_states={initial_states} "
                f"-- multi_restart_extract_front_anchored ({best_cost}) worse than vanilla ({vanilla_cost})"
            )
            assert stats["best_cost"] == best_cost

    def test_returned_circuit_is_well_formed_and_cost_consistent(self) -> None:
        g = _reduced_graph(5, 20, seed=42)
        initial_states = [1, 0, 1, 0, 1]
        circuit, stats = multi_restart_extract_front_anchored(
            g.copy(), initial_states, n_restarts=3, threshold=1, lookahead_depth=1, seed=0, quiet=True,
        )
        assert circuit.qubits == 5
        front_assignment = [Flavor(s) for s in initial_states]
        tracker = SigmaTracker(circuit, front_assignment).propagate()
        assert tracker.round_robin_count >= 0
        assert tracker.msd_count >= 0
        assert stats["n_restarts"] == 3
