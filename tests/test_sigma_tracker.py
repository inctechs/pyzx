"""Tests for the σ-state tracker.

Validates each cost rule from the R/R' tetrahedral code framework
independently, then tests their interaction.
"""

from __future__ import annotations

import random
import unittest

import pyzx as zx
from pyzx.circuit.gates import S as SGate
from pyzx.circuit.gates import T as TGate
from pyzx.sigma_tracker import (
    CostType,
    Flavor,
    SigmaTracker,
)

R = Flavor.R
Rp = Flavor.R_PRIME


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_circuit(n_qubits: int, gate_list: list) -> zx.Circuit:
    """Build a PyZX circuit from a list of (gate_name, *args) tuples.

    Supported formats:
        ('H', qubit)
        ('T', qubit)          ('T*', qubit) for T-dagger
        ('S', qubit)          ('S*', qubit) for S-dagger
        ('CNOT', control, target)
        ('CZ', q1, q2)
        ('Z', qubit)          ('X', qubit)
    """
    c = zx.Circuit(n_qubits)
    for spec in gate_list:
        name = spec[0]
        if name == "H":
            c.add_gate("HAD", spec[1])
        elif name == "T":
            c.add_gate("T", spec[1])
        elif name == "T*":
            c.gates.append(TGate(spec[1], adjoint=True))
        elif name == "S":
            c.add_gate("S", spec[1])
        elif name == "S*":
            c.gates.append(SGate(spec[1], adjoint=True))
        elif name == "CNOT":
            c.add_gate("CNOT", spec[1], spec[2])
        elif name == "CZ":
            c.add_gate("CZ", spec[1], spec[2])
        elif name == "Z":
            c.add_gate("Z", spec[1])
        elif name == "X":
            c.add_gate("NOT", spec[1])
        else:
            msg = f"Unknown gate: {name}"
            raise ValueError(msg)
    return c


def make_tracker(n_qubits, gate_list, assignment=None, w_msd=1.0, w_rr=1.0) -> SigmaTracker:
    """Build a propagated SigmaTracker for the given gate list."""
    c = make_circuit(n_qubits, gate_list)
    t = SigmaTracker(c, assignment, w_msd=w_msd, w_rr=w_rr)
    t.propagate()
    return t


# ===================================================================
# Single-qubit Z-diagonal gate cost rules
# ===================================================================


class TestSingleQubitZDiagonal(unittest.TestCase):
    """T, T†, S, S† are free on R, expensive (MSD) on R'."""

    def test_free_on_r(self):
        for gate in ["T", "T*", "S", "S*"]:
            with self.subTest(gate=gate):
                t = make_tracker(1, [(gate, 0)], [R])
                self.assertEqual(t.cost, 0)
                self.assertEqual(t.msd_count, 0)

    def test_expensive_on_r_prime(self):
        for gate in ["T", "T*", "S", "S*"]:
            with self.subTest(gate=gate):
                t = make_tracker(1, [(gate, 0)], [Rp])
                self.assertEqual(t.cost, 1)
                self.assertEqual(t.msd_count, 1)
                self.assertEqual(t.round_robin_count, 0)
                self.assertEqual(t.expensive_ops[0].cost_type, CostType.MSD)


# ===================================================================
# Hadamard flavor toggling
# ===================================================================


class TestHadamardToggle(unittest.TestCase):
    def test_h_flips_r_to_r_prime(self):
        """Start R, apply H -> R', then T -> expensive."""
        t = make_tracker(1, [("H", 0), ("T", 0)], [R])
        self.assertEqual(t.msd_count, 1)

    def test_h_flips_r_prime_to_r(self):
        """Start R', apply H -> R, then T -> free."""
        t = make_tracker(1, [("H", 0), ("T", 0)], [Rp])
        self.assertEqual(t.msd_count, 0)

    def test_double_h_cancels(self):
        """HH = I, so flavor returns to initial."""
        t = make_tracker(1, [("H", 0), ("H", 0), ("T", 0)], [R])
        self.assertEqual(t.cost, 0)

    def test_triple_h_is_single_flip(self):
        """Three H's net to one flip."""
        t = make_tracker(1, [("H", 0), ("H", 0), ("H", 0), ("T", 0)], [R])
        self.assertEqual(t.msd_count, 1)


# ===================================================================
# CNOT cost rules (all 4 flavor combinations)
# ===================================================================


class TestCNOTCost(unittest.TestCase):
    def test_cnot_cost_table(self):
        cases = [
            ("R-R", R, R, 0),
            ("R-R'", R, Rp, 0),
            ("R'-R'", Rp, Rp, 0),
            ("R'-R_expensive", Rp, R, 1),
        ]
        for case_id, ctrl, tgt, expected_cost in cases:
            with self.subTest(case_id):
                t = make_tracker(2, [("CNOT", 0, 1)], [ctrl, tgt])
                self.assertEqual(t.cost, expected_cost)

    def test_expensive_cnot_is_round_robin(self):
        t = make_tracker(2, [("CNOT", 0, 1)], [Rp, R])
        self.assertEqual(t.round_robin_count, 1)
        self.assertEqual(t.expensive_ops[0].cost_type, CostType.ROUND_ROBIN)


# ===================================================================
# CZ cost rules (all 4 flavor combinations)
# ===================================================================


class TestCZCost(unittest.TestCase):
    def test_cz_cost_table(self):
        cases = [
            ("R-R", R, R, 0),
            ("R-R'", R, Rp, 0),
            ("R'-R", Rp, R, 0),
            ("R'-R'_expensive", Rp, Rp, 1),
        ]
        for case_id, fa, fb, expected_cost in cases:
            with self.subTest(case_id):
                t = make_tracker(2, [("CZ", 0, 1)], [fa, fb])
                self.assertEqual(t.cost, expected_cost)

    def test_expensive_cz_is_round_robin(self):
        t = make_tracker(2, [("CZ", 0, 1)], [Rp, Rp])
        self.assertEqual(t.round_robin_count, 1)
        self.assertEqual(t.expensive_ops[0].cost_type, CostType.ROUND_ROBIN)


# ===================================================================
# No-arbitrage: CNOT <-> CZ conversion doesn't help
# ===================================================================


class TestNoArbitrage(unittest.TestCase):
    def test_cnot_cz_equivalence(self):
        """CNOT(a,b) = (I x H) CZ(a,b) (I x H).

        The expensive CNOT case (R', R) maps to the expensive CZ case (R', R')
        because the H on qubit b flips its flavor.
        """
        t_cnot = make_tracker(2, [("CNOT", 0, 1)], [Rp, R])
        t_equiv = make_tracker(2, [("H", 1), ("CZ", 0, 1), ("H", 1)], [Rp, R])
        self.assertEqual(t_cnot.cost, 1)
        self.assertEqual(t_equiv.cost, 1)


# ===================================================================
# Paulis are always free
# ===================================================================


class TestPaulis(unittest.TestCase):
    def test_z_free(self):
        for flavor in [R, Rp]:
            with self.subTest(flavor=flavor):
                t = make_tracker(1, [("Z", 0)], [flavor])
                self.assertEqual(t.cost, 0)

    def test_x_free(self):
        for flavor in [R, Rp]:
            with self.subTest(flavor=flavor):
                t = make_tracker(1, [("X", 0)], [flavor])
                self.assertEqual(t.cost, 0)


# ===================================================================
# Complex flavor propagation
# ===================================================================


class TestFlavorPropagation(unittest.TestCase):
    def test_multi_gate_tracking(self):
        """q0: R -> H -> R' -> T(expensive) -> H -> R -> T(free)
        q1: R' (unchanged, no H gates) -> T(expensive).
        """
        t = make_tracker(
            2,
            [("H", 0), ("T", 0), ("H", 0), ("T", 0), ("T", 1)],
            [R, Rp],
        )
        self.assertEqual(t.msd_count, 2)
        self.assertEqual(t.round_robin_count, 0)

    def test_flavor_at_specific_positions(self):
        c = make_circuit(
            2, [("H", 0), ("T", 0), ("H", 0), ("T", 0), ("T", 1)]
        )
        t = SigmaTracker(c, [R, Rp]).propagate()
        self.assertEqual(t.flavor_at(1, 0), Rp)  # after first H
        self.assertEqual(t.flavor_at(3, 0), R)   # after second H
        self.assertEqual(t.flavor_at(4, 1), Rp)  # q1 never gets H


# ===================================================================
# Assignment optimization
# ===================================================================


class TestOptimization(unittest.TestCase):
    def test_single_t_assigns_r(self):
        c = make_circuit(2, [("T", 0)])
        t = SigmaTracker(c)
        best, cost = t.optimize_assignment()
        self.assertEqual(cost, 0)
        self.assertEqual(best[0], R)

    def test_unavoidable_conflict(self):
        """T, H, T on same qubit: one T is always expensive regardless of assignment."""
        c = make_circuit(1, [("T", 0), ("H", 0), ("T", 0)])
        t = SigmaTracker(c)
        _, cost = t.optimize_assignment()
        self.assertEqual(cost, 1)

    def test_multi_qubit_joint(self):
        """T on q0 + CNOT(0,1): optimal q0=R avoids both costs."""
        c = make_circuit(2, [("T", 0), ("CNOT", 0, 1)])
        t = SigmaTracker(c)
        best, cost = t.optimize_assignment()
        self.assertEqual(cost, 0)
        self.assertEqual(best[0], R)

    def test_optimizer_updates_tracker_state(self):
        """After optimize, tracker should reflect the best assignment."""
        c = make_circuit(2, [("T", 0), ("CNOT", 0, 1)])
        t = SigmaTracker(c)
        best, cost = t.optimize_assignment()
        self.assertEqual(t.assignment, best)
        self.assertEqual(t.cost, cost)

    def test_brute_force_rejects_large_n(self):
        c = zx.Circuit(25)
        t = SigmaTracker(c)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            t.optimize_assignment()


# ===================================================================
# Cost weights
# ===================================================================


class TestWeightedCost(unittest.TestCase):
    def test_custom_weights(self):
        t = make_tracker(
            2,
            [("T", 0), ("CNOT", 0, 1)],
            [Rp, R],
            w_msd=3.0,
            w_rr=5.0,
        )
        self.assertEqual(t.msd_count, 1)
        self.assertEqual(t.round_robin_count, 1)
        self.assertAlmostEqual(t.cost, 8.0)

    def test_zero_weight_ignores_type(self):
        t = make_tracker(
            2,
            [("T", 0), ("CNOT", 0, 1)],
            [Rp, R],
            w_msd=0.0,
            w_rr=5.0,
        )
        self.assertAlmostEqual(t.cost, 5.0)
        self.assertEqual(t.msd_count, 1)  # still counted, just zero-weighted


# ===================================================================
# Reporting and partitioning helpers
# ===================================================================


class TestReporting(unittest.TestCase):
    def test_expensive_gate_indices(self):
        t = make_tracker(
            2,
            [("T", 0), ("H", 0), ("T", 0), ("CNOT", 0, 1)],
            [R, R],
        )
        self.assertEqual(t.expensive_gate_indices(), [2, 3])

    def test_free_segments(self):
        t = make_tracker(
            2,
            [("T", 0), ("H", 0), ("T", 0), ("T", 1), ("CNOT", 1, 0)],
            [R, R],
        )
        self.assertEqual(t.free_segments(), [(0, 1), (3, 4)])

    def test_cost_breakdown_by_qubit(self):
        t = make_tracker(
            3,
            [("T", 0), ("T", 1), ("CNOT", 0, 2)],
            [Rp, Rp, R],
        )
        bd = t.cost_breakdown_by_qubit()
        self.assertEqual(bd[0]["msd"], 1)
        self.assertEqual(bd[0]["round_robin"], 1)
        self.assertEqual(bd[1]["msd"], 1)
        self.assertEqual(bd[1]["round_robin"], 0)
        self.assertEqual(bd[2]["msd"], 0)
        self.assertEqual(bd[2]["round_robin"], 1)

    def test_final_flavors(self):
        t = make_tracker(
            2,
            [("H", 0), ("H", 0), ("H", 1)],
            [R, R],
        )
        finals = t.final_flavors()
        self.assertEqual(finals[0], R)   # two H's cancel
        self.assertEqual(finals[1], Rp)  # one H flips


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases(unittest.TestCase):
    def test_empty_circuit(self):
        c = zx.Circuit(3)
        t = SigmaTracker(c).propagate()
        self.assertEqual(t.cost, 0)
        self.assertEqual(len(t.expensive_ops), 0)

    def test_h_only_circuit(self):
        t = make_tracker(1, [("H", 0), ("H", 0), ("H", 0)], [R])
        self.assertEqual(t.cost, 0)
        self.assertEqual(t.final_flavors(), [Rp])  # odd number of H's

    def test_assignment_length_mismatch(self):
        c = zx.Circuit(3)
        with self.assertRaisesRegex(ValueError, "Assignment length"):
            SigmaTracker(c, [R, R])

    def test_set_assignment_invalidates_cache(self):
        c = make_circuit(1, [("T", 0)])
        t = SigmaTracker(c, [Rp]).propagate()
        self.assertEqual(t.cost, 1)
        t.set_assignment([R]).propagate()
        self.assertEqual(t.cost, 0)

    def test_summary_contains_key_info(self):
        t = make_tracker(2, [("T", 0)], [Rp, R])
        s = t.summary()
        self.assertIn("MSD", s)
        self.assertIn("R'", s)


# ===================================================================
# Pareto frontier (weight-free) over initial assignments
# ===================================================================


_RANDOM_GATE_TYPES_1Q = ["H", "T", "T*", "S", "S*", "Z", "X"]
_RANDOM_GATE_TYPES_2Q = ["CNOT", "CZ"]


def _random_gate_list(
    n_qubits: int, n_gates: int, rng: random.Random
) -> list[tuple[str, int] | tuple[str, int, int]]:
    """Random mix of 1- and 2-qubit gates, using make_circuit's vocabulary."""
    gate_list: list[tuple[str, int] | tuple[str, int, int]] = []
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.4:
            kind = rng.choice(_RANDOM_GATE_TYPES_2Q)
            a, b = rng.sample(range(n_qubits), 2)
            gate_list.append((kind, a, b))
        else:
            kind = rng.choice(_RANDOM_GATE_TYPES_1Q)
            q = rng.randrange(n_qubits)
            gate_list.append((kind, q))
    return gate_list


def _all_assignment_points(tracker: SigmaTracker) -> list[tuple[int, int]]:
    """Oracle: (rr, msd) for every one of the 2^n assignments, computed via
    the already-tested single-assignment path (set_assignment + propagate),
    independent of pareto_assignments()'s internal fast loop.
    """
    n = tracker.n_qubits
    points = []
    for bits in range(1 << n):
        assignment = [Flavor((bits >> i) & 1) for i in range(n)]
        tracker.set_assignment(assignment).propagate()
        points.append((tracker.round_robin_count, tracker.msd_count))
    return points


def _dominates(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True iff point a dominates point b: a <= b componentwise, strictly in
    at least one component.
    """
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def _naive_pareto_corners(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Naive O(k^2) pairwise-dominance oracle over distinct (rr, msd) points."""
    distinct = set(points)
    return {p for p in distinct if not any(_dominates(q, p) for q in distinct if q != p)}


class TestParetoAssignments(unittest.TestCase):
    """Exact, weight-free Pareto frontier over the 2^n initial assignments."""

    def test_matches_naive_dominance_filter(self) -> None:
        rng = random.Random(1234)
        for _trial in range(15):
            n = rng.randint(1, 10)
            n_gates = rng.randint(0, 20)
            c = make_circuit(n, _random_gate_list(n, n_gates, rng))
            t = SigmaTracker(c)

            frontier = t.pareto_assignments()
            frontier_corners = {(rr, msd) for _, rr, msd in frontier}

            naive_corners = _naive_pareto_corners(_all_assignment_points(t))

            self.assertEqual(frontier_corners, naive_corners)

    def test_frontier_points_are_achievable(self) -> None:
        rng = random.Random(5678)
        for _trial in range(10):
            n = rng.randint(1, 8)
            n_gates = rng.randint(0, 20)
            c = make_circuit(n, _random_gate_list(n, n_gates, rng))
            t = SigmaTracker(c)

            for assignment, rr, msd in t.pareto_assignments():
                t.set_assignment(assignment).propagate()
                self.assertEqual(t.round_robin_count, rr)
                self.assertEqual(t.msd_count, msd)

    def test_weighted_argmin_lies_on_frontier(self) -> None:
        # Continuous, strictly-positive weights avoid degenerate ties: any
        # global minimizer of w_rr*rr + w_msd*msd is Pareto-optimal (if it
        # were dominated by another achievable point, that point would have
        # strictly lower weighted cost, contradicting minimality), so its
        # (rr, msd) must not be dominated by any frontier point.
        rng = random.Random(91)
        for _trial in range(8):
            n = rng.randint(1, 8)
            n_gates = rng.randint(0, 20)
            c = make_circuit(n, _random_gate_list(n, n_gates, rng))

            frontier = SigmaTracker(c).pareto_assignments()

            w_rr = rng.uniform(0.1, 5.0)
            w_msd = rng.uniform(0.1, 5.0)
            t = SigmaTracker(c, w_msd=w_msd, w_rr=w_rr)
            t.optimize_assignment()
            point = (t.round_robin_count, t.msd_count)

            self.assertFalse(any(_dominates((frr, fmsd), point) for _, frr, fmsd in frontier))

    def test_frontier_is_a_proper_staircase(self) -> None:
        rng = random.Random(2026)
        for _trial in range(10):
            n = rng.randint(1, 8)
            n_gates = rng.randint(0, 20)
            c = make_circuit(n, _random_gate_list(n, n_gates, rng))
            t = SigmaTracker(c)

            frontier = t.pareto_assignments()
            rrs = [rr for _, rr, _ in frontier]
            msds = [msd for _, _, msd in frontier]

            self.assertEqual(rrs, sorted(rrs))
            self.assertEqual(len(set(rrs)), len(rrs))
            self.assertEqual(len(set(msds)), len(msds))
            self.assertTrue(all(msds[i] > msds[i + 1] for i in range(len(msds) - 1)))

    def test_rejects_large_n(self) -> None:
        c = zx.Circuit(25)
        t = SigmaTracker(c)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            t.pareto_assignments()


if __name__ == '__main__':
    unittest.main()
