"""Tests for the σ-state tracker.

Validates each cost rule from the R/R' tetrahedral code framework
independently, then tests their interaction.
"""

from __future__ import annotations

import pytest
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


@pytest.fixture
def tracker_factory():
    """Factory fixture: returns a function that builds a propagated SigmaTracker."""

    def _make(n_qubits, gate_list, assignment=None, w_msd=1.0, w_rr=1.0):
        c = make_circuit(n_qubits, gate_list)
        t = SigmaTracker(c, assignment, w_msd=w_msd, w_rr=w_rr)
        t.propagate()
        return t

    return _make


# ===================================================================
# Single-qubit Z-diagonal gate cost rules
# ===================================================================


class TestSingleQubitZDiagonal:
    """T, T†, S, S† are free on R, expensive (MSD) on R'."""

    @pytest.mark.parametrize("gate", ["T", "T*", "S", "S*"])
    def test_free_on_r(self, tracker_factory, gate):
        t = tracker_factory(1, [(gate, 0)], [R])
        assert t.cost == 0
        assert t.msd_count == 0

    @pytest.mark.parametrize("gate", ["T", "T*", "S", "S*"])
    def test_expensive_on_r_prime(self, tracker_factory, gate):
        t = tracker_factory(1, [(gate, 0)], [Rp])
        assert t.cost == 1
        assert t.msd_count == 1
        assert t.round_robin_count == 0
        assert t.expensive_ops[0].cost_type == CostType.MSD


# ===================================================================
# Hadamard flavor toggling
# ===================================================================


class TestHadamardToggle:
    def test_h_flips_r_to_r_prime(self, tracker_factory):
        """Start R, apply H -> R', then T -> expensive."""
        t = tracker_factory(1, [("H", 0), ("T", 0)], [R])
        assert t.msd_count == 1

    def test_h_flips_r_prime_to_r(self, tracker_factory):
        """Start R', apply H -> R, then T -> free."""
        t = tracker_factory(1, [("H", 0), ("T", 0)], [Rp])
        assert t.msd_count == 0

    def test_double_h_cancels(self, tracker_factory):
        """HH = I, so flavor returns to initial."""
        t = tracker_factory(1, [("H", 0), ("H", 0), ("T", 0)], [R])
        assert t.cost == 0

    def test_triple_h_is_single_flip(self, tracker_factory):
        """Three H's net to one flip."""
        t = tracker_factory(1, [("H", 0), ("H", 0), ("H", 0), ("T", 0)], [R])
        assert t.msd_count == 1


# ===================================================================
# CNOT cost rules (all 4 flavor combinations)
# ===================================================================


class TestCNOTCost:
    @pytest.mark.parametrize(
        ("ctrl", "tgt", "expected_cost"),
        [
            (R, R, 0),
            (R, Rp, 0),
            (Rp, Rp, 0),
            (Rp, R, 1),
        ],
        ids=["R-R", "R-R'", "R'-R'", "R'-R_expensive"],
    )
    def test_cnot_cost_table(self, tracker_factory, ctrl, tgt, expected_cost):
        t = tracker_factory(2, [("CNOT", 0, 1)], [ctrl, tgt])
        assert t.cost == expected_cost

    def test_expensive_cnot_is_round_robin(self, tracker_factory):
        t = tracker_factory(2, [("CNOT", 0, 1)], [Rp, R])
        assert t.round_robin_count == 1
        assert t.expensive_ops[0].cost_type == CostType.ROUND_ROBIN


# ===================================================================
# CZ cost rules (all 4 flavor combinations)
# ===================================================================


class TestCZCost:
    @pytest.mark.parametrize(
        ("fa", "fb", "expected_cost"),
        [
            (R, R, 0),
            (R, Rp, 0),
            (Rp, R, 0),
            (Rp, Rp, 1),
        ],
        ids=["R-R", "R-R'", "R'-R", "R'-R'_expensive"],
    )
    def test_cz_cost_table(self, tracker_factory, fa, fb, expected_cost):
        t = tracker_factory(2, [("CZ", 0, 1)], [fa, fb])
        assert t.cost == expected_cost

    def test_expensive_cz_is_round_robin(self, tracker_factory):
        t = tracker_factory(2, [("CZ", 0, 1)], [Rp, Rp])
        assert t.round_robin_count == 1
        assert t.expensive_ops[0].cost_type == CostType.ROUND_ROBIN


# ===================================================================
# No-arbitrage: CNOT <-> CZ conversion doesn't help
# ===================================================================


class TestNoArbitrage:
    def test_cnot_cz_equivalence(self, tracker_factory):
        """CNOT(a,b) = (I x H) CZ(a,b) (I x H).

        The expensive CNOT case (R', R) maps to the expensive CZ case (R', R')
        because the H on qubit b flips its flavor.
        """
        t_cnot = tracker_factory(2, [("CNOT", 0, 1)], [Rp, R])
        t_equiv = tracker_factory(2, [("H", 1), ("CZ", 0, 1), ("H", 1)], [Rp, R])
        assert t_cnot.cost == t_equiv.cost == 1


# ===================================================================
# Paulis are always free
# ===================================================================


class TestPaulis:
    @pytest.mark.parametrize("flavor", [R, Rp], ids=["R", "R'"])
    def test_z_free(self, tracker_factory, flavor):
        t = tracker_factory(1, [("Z", 0)], [flavor])
        assert t.cost == 0

    @pytest.mark.parametrize("flavor", [R, Rp], ids=["R", "R'"])
    def test_x_free(self, tracker_factory, flavor):
        t = tracker_factory(1, [("X", 0)], [flavor])
        assert t.cost == 0


# ===================================================================
# Complex flavor propagation
# ===================================================================


class TestFlavorPropagation:
    def test_multi_gate_tracking(self, tracker_factory):
        """q0: R -> H -> R' -> T(expensive) -> H -> R -> T(free)
        q1: R' (unchanged, no H gates) -> T(expensive).
        """
        t = tracker_factory(
            2,
            [("H", 0), ("T", 0), ("H", 0), ("T", 0), ("T", 1)],
            [R, Rp],
        )
        assert t.msd_count == 2
        assert t.round_robin_count == 0

    def test_flavor_at_specific_positions(self):
        c = make_circuit(
            2, [("H", 0), ("T", 0), ("H", 0), ("T", 0), ("T", 1)]
        )
        t = SigmaTracker(c, [R, Rp]).propagate()
        assert t.flavor_at(1, 0) == Rp  # after first H
        assert t.flavor_at(3, 0) == R   # after second H
        assert t.flavor_at(4, 1) == Rp  # q1 never gets H


# ===================================================================
# Assignment optimization
# ===================================================================


class TestOptimization:
    def test_single_t_assigns_r(self):
        c = make_circuit(2, [("T", 0)])
        t = SigmaTracker(c)
        best, cost = t.optimize_assignment()
        assert cost == 0
        assert best[0] == R

    def test_unavoidable_conflict(self):
        """T, H, T on same qubit: one T is always expensive regardless of assignment."""
        c = make_circuit(1, [("T", 0), ("H", 0), ("T", 0)])
        t = SigmaTracker(c)
        _, cost = t.optimize_assignment()
        assert cost == 1

    def test_multi_qubit_joint(self):
        """T on q0 + CNOT(0,1): optimal q0=R avoids both costs."""
        c = make_circuit(2, [("T", 0), ("CNOT", 0, 1)])
        t = SigmaTracker(c)
        best, cost = t.optimize_assignment()
        assert cost == 0
        assert best[0] == R

    def test_optimizer_updates_tracker_state(self):
        """After optimize, tracker should reflect the best assignment."""
        c = make_circuit(2, [("T", 0), ("CNOT", 0, 1)])
        t = SigmaTracker(c)
        best, cost = t.optimize_assignment()
        assert t.assignment == best
        assert t.cost == cost

    def test_brute_force_rejects_large_n(self):
        c = zx.Circuit(25)
        t = SigmaTracker(c)
        with pytest.raises(ValueError, match="infeasible"):
            t.optimize_assignment()


# ===================================================================
# Cost weights
# ===================================================================


class TestWeightedCost:
    def test_custom_weights(self, tracker_factory):
        t = tracker_factory(
            2,
            [("T", 0), ("CNOT", 0, 1)],
            [Rp, R],
            w_msd=3.0,
            w_rr=5.0,
        )
        assert t.msd_count == 1
        assert t.round_robin_count == 1
        assert t.cost == pytest.approx(8.0)

    def test_zero_weight_ignores_type(self, tracker_factory):
        t = tracker_factory(
            2,
            [("T", 0), ("CNOT", 0, 1)],
            [Rp, R],
            w_msd=0.0,
            w_rr=5.0,
        )
        assert t.cost == pytest.approx(5.0)
        assert t.msd_count == 1  # still counted, just zero-weighted


# ===================================================================
# Reporting and partitioning helpers
# ===================================================================


class TestReporting:
    def test_expensive_gate_indices(self, tracker_factory):
        t = tracker_factory(
            2,
            [("T", 0), ("H", 0), ("T", 0), ("CNOT", 0, 1)],
            [R, R],
        )
        assert t.expensive_gate_indices() == [2, 3]

    def test_free_segments(self, tracker_factory):
        t = tracker_factory(
            2,
            [("T", 0), ("H", 0), ("T", 0), ("T", 1), ("CNOT", 1, 0)],
            [R, R],
        )
        assert t.free_segments() == [(0, 1), (3, 4)]

    def test_cost_breakdown_by_qubit(self, tracker_factory):
        t = tracker_factory(
            3,
            [("T", 0), ("T", 1), ("CNOT", 0, 2)],
            [Rp, Rp, R],
        )
        bd = t.cost_breakdown_by_qubit()
        assert bd[0]["msd"] == 1
        assert bd[0]["round_robin"] == 1
        assert bd[1]["msd"] == 1
        assert bd[1]["round_robin"] == 0
        assert bd[2]["msd"] == 0
        assert bd[2]["round_robin"] == 1

    def test_final_flavors(self, tracker_factory):
        t = tracker_factory(
            2,
            [("H", 0), ("H", 0), ("H", 1)],
            [R, R],
        )
        finals = t.final_flavors()
        assert finals[0] == R   # two H's cancel
        assert finals[1] == Rp  # one H flips


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_circuit(self):
        c = zx.Circuit(3)
        t = SigmaTracker(c).propagate()
        assert t.cost == 0
        assert len(t.expensive_ops) == 0

    def test_h_only_circuit(self, tracker_factory):
        t = tracker_factory(1, [("H", 0), ("H", 0), ("H", 0)], [R])
        assert t.cost == 0
        assert t.final_flavors() == [Rp]  # odd number of H's

    def test_assignment_length_mismatch(self):
        c = zx.Circuit(3)
        with pytest.raises(ValueError, match="Assignment length"):
            SigmaTracker(c, [R, R])

    def test_set_assignment_invalidates_cache(self):
        c = make_circuit(1, [("T", 0)])
        t = SigmaTracker(c, [Rp]).propagate()
        assert t.cost == 1
        t.set_assignment([R]).propagate()
        assert t.cost == 0

    def test_summary_contains_key_info(self, tracker_factory):
        t = tracker_factory(2, [("T", 0)], [Rp, R])
        s = t.summary()
        assert "MSD" in s
        assert "R'" in s
