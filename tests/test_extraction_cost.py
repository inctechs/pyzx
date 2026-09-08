# PyZX - Python library for quantum circuit rewriting
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import random
from fractions import Fraction
from typing import Any, List, Tuple

from pyzx.circuit import Circuit
from pyzx.generate import cliffordT
from pyzx.simplify import full_reduce
from pyzx.extract import (
    CostAccumulator,
    count_cnot_faults,
    count_cz_faults,
    count_phase_faults,
    is_expensive_misplaced_phase,
    multi_restart_extract,
)

SEED = 1337


def _reference_cost_forward(circuit: Circuit, initial_assignment: List[int]) -> Tuple[int, int]:
    """Independent forward-walk reference, following the *documented* R/R'
    semantics of pyzx.sigma_tracker (its module
    docstring / CircuitStructure._analyze comments): CNOT bad iff
    control=R'(0) & target=R(1); CZ bad iff both R'(0); phases classified via
    the shared is_expensive_misplaced_phase predicate. Walks forward from
    initial_assignment (state before circuit.gates[0]), toggling on HAD.

    NOTE: as of this writing, pyzx.sigma_tracker.SigmaTracker's
    CircuitStructure._analyze has a confirmed bug that inverts all four of
    these classifications relative to its own docstring (verified with
    trivial one/two-qubit circuits: e.g. a lone CNOT(0,1) with
    initial_assignment=[R_PRIME, R] costs 0, but [R, R_PRIME] costs 1 -- the
    opposite of "control=R', target=R is expensive"). So this reference
    intentionally does NOT call the real SigmaTracker; it exists to cross
    check extraction's backward-pass counters (count_cnot_faults/
    count_cz_faults/count_phase_faults) against an independently written
    forward pass, using the documented semantics both sides agree should
    hold. The real SigmaTracker bug should be fixed in pyzx.sigma_tracker
    separately.

    Returns (round_robin_count, msd_count).
    """
    state = list(initial_assignment)
    rr = 0
    msd = 0
    for gate in circuit.gates:
        name = gate.name
        if name == 'HAD':
            state[gate.target] ^= 1  # type: ignore[attr-defined]
        elif name == 'CNOT':
            c, t = gate.control, gate.target  # type: ignore[attr-defined]
            if state[c] == 0 and state[t] == 1:
                rr += 1
        elif name == 'CZ':
            a, b = gate.control, gate.target  # type: ignore[attr-defined]
            if state[a] == 0 and state[b] == 0:
                rr += 1
        elif name in ('ZPhase', 'XPhase', 'T', 'S'):
            if is_expensive_misplaced_phase(name, gate.phase, state[gate.target]):  # type: ignore[attr-defined]
                msd += 1
    return rr, msd


class TestRecordPhase(unittest.TestCase):
    """Unit tests for CostAccumulator.record_phase."""

    def test_misplaced_t_is_bad(self) -> None:
        acc = CostAccumulator()
        acc.record_phase(0, Fraction(1, 4))  # T on R' (0) -> stranded, bad
        self.assertEqual(acc.bad_phases, 1)

    def test_correctly_placed_t_is_free(self) -> None:
        acc = CostAccumulator()
        acc.record_phase(1, Fraction(1, 4))  # T on R (1) -> free
        self.assertEqual(acc.bad_phases, 0)

    def test_pauli_z_is_always_free(self) -> None:
        acc = CostAccumulator()
        acc.record_phase(0, Fraction(1, 1))  # Z (Pauli, integer phase) -> free even on R'
        self.assertEqual(acc.bad_phases, 0)

    def test_cost_property_weighs_bad_phases_by_w_msd(self) -> None:
        acc = CostAccumulator(w_cnot=1.0, w_cz=1.0, w_msd=3.0)
        acc.record_phase(0, Fraction(1, 4))
        acc.record_phase(0, Fraction(1, 4))
        acc.record_cnot(0, 1)
        self.assertEqual(acc.bad_phases, 2)
        self.assertEqual(acc.cost, 1.0 * 1 + 3.0 * 2)


class TestCountPhaseFaults(unittest.TestCase):
    """Unit tests for the backward-pass count_phase_faults scorer."""

    def test_misplaced_t(self) -> None:
        c = Circuit(1)
        c.add_gate("ZPhase", 0, Fraction(1, 4))  # T
        faults = count_phase_faults(c, [0])  # qubit ends the circuit as R' -> bad
        self.assertEqual(faults, {'bad': 1, 'safe': 0, 'total': 1})

    def test_correctly_placed_t(self) -> None:
        c = Circuit(1)
        c.add_gate("ZPhase", 0, Fraction(1, 4))  # T
        faults = count_phase_faults(c, [1])  # qubit ends the circuit as R -> free
        self.assertEqual(faults, {'bad': 0, 'safe': 1, 'total': 1})

    def test_pauli_z_ignored(self) -> None:
        c = Circuit(1)
        c.add_gate("ZPhase", 0, Fraction(1, 1))  # Z, a Pauli
        faults = count_phase_faults(c, [0])
        self.assertEqual(faults, {'bad': 0, 'safe': 1, 'total': 1})

    def test_had_toggle_is_exercised_in_backward_pass(self) -> None:
        # Gates in forward order: T(q0) ; HAD(q0). final_states=[1] is the state
        # AFTER the HAD. The backward pass must see the HAD first, toggle the
        # state to 0, and only then evaluate T at state 0 (bad).
        c = Circuit(1)
        c.add_gate("ZPhase", 0, Fraction(1, 4))  # T
        c.add_gate("HAD", 0)
        faults = count_phase_faults(c, [1])
        self.assertEqual(faults, {'bad': 1, 'safe': 0, 'total': 1})


class TestForwardBackwardReconciliation(unittest.TestCase):
    """The extraction-side weighted cost (computed backward from the output
    boundary via count_cnot_faults/count_cz_faults/count_phase_faults) must equal
    an independently-computed forward-walk cost using the *documented* R/R'
    semantics (_reference_cost_forward), for the same physical circuit.

    This is intentionally not a reconciliation against the real, installed
    SigmaTracker: see _reference_cost_forward's docstring for the confirmed bug
    in pyzx.sigma_tracker.CircuitStructure._analyze
    that inverts its CNOT/CZ/phase classifications relative to its own
    docstring, which currently makes such a comparison meaningless.
    """

    @staticmethod
    def _boundary_states(circuit: Circuit, states: List[int]) -> List[int]:
        """Given the state after circuit.gates[-1] (the convention used by
        count_cnot_faults/count_cz_faults/count_phase_faults), walk HAD toggles
        backward to recover the state before circuit.gates[0] (the convention
        _reference_cost_forward's initial_assignment uses)."""
        states = list(states)
        for gate in reversed(circuit.gates):
            if gate.name == 'HAD':
                states[gate.target] ^= 1  # type: ignore[attr-defined]
        return states

    def _assert_reconciles(self, g: Any, initial_states: List[int], w: float, m: float, seed: int) -> None:
        circuit, stats = multi_restart_extract(
            g, initial_states, n_restarts=6, seed=seed,
            w_cnot=w, w_cz=w, w_msd=m,
        )
        chosen = circuit.to_basic_gates()

        cnot_faults = count_cnot_faults(chosen, list(initial_states))
        cz_faults = count_cz_faults(chosen, list(initial_states))
        phase_faults = count_phase_faults(chosen, list(initial_states))
        extraction_cost = w * cnot_faults['bad'] + w * cz_faults['bad'] + m * phase_faults['bad']

        forward_initial = self._boundary_states(chosen, initial_states)
        rr_count, msd_count = _reference_cost_forward(chosen, forward_initial)
        reference_cost = w * rr_count + m * msd_count

        self.assertAlmostEqual(extraction_cost, reference_cost)
        self.assertEqual(cnot_faults['bad'] + cz_faults['bad'], rr_count)
        self.assertEqual(phase_faults['bad'], msd_count)

    def test_reconciles_on_small_clifford_t_circuits(self) -> None:
        random.seed(SEED)
        for i in range(5):
            g = cliffordT(4, 30, p_t=0.3, p_cnot=0.3, seed=SEED + i)
            full_reduce(g, quiet=True)
            initial_states = [random.randint(0, 1) for _ in range(g.qubit_count())]
            with self.subTest(i=i):
                self._assert_reconciles(g.clone(), initial_states, w=1.0, m=2.0, seed=i)


class TestMultiRestartNoRegression(unittest.TestCase):
    """With w_msd > 0, multi_restart_extract's chosen circuit must never be more
    expensive (by the weighted cost) than the deterministic vanilla trial (trial 0)."""

    def test_chosen_cost_never_exceeds_vanilla_baseline(self) -> None:
        random.seed(SEED)
        for i in range(5):
            g = cliffordT(4, 30, p_t=0.3, p_cnot=0.3, seed=SEED + i)
            full_reduce(g, quiet=True)
            initial_states = [random.randint(0, 1) for _ in range(g.qubit_count())]
            with self.subTest(i=i):
                _, stats = multi_restart_extract(
                    g, initial_states, n_restarts=6, seed=i,
                    w_cnot=1.0, w_cz=1.0, w_msd=2.0,
                )
                vanilla_cost = stats['runs'][0]['cost']
                self.assertLessEqual(stats['best_cost'], vanilla_cost)


if __name__ == '__main__':
    unittest.main()
