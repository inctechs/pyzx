"""Equivalence test between the two independent counting paths of the R/R'
tetrahedral cost model:

  - SigmaTracker (pyzx.sigma_tracker): forward propagation of flavor from an
    initial assignment, counting via CircuitStructure._analyze /
    potential_costs.
  - pyzx.extract's backward-pass fault counters (count_cnot_faults,
    count_cz_faults, count_phase_faults): walk a finished circuit backward
    from known final qubit states.

Both classify phase-gate expensiveness via the single shared predicate in
pyzx.tetrahedral_cost (see Task A of this prompt series). This test is the
ground truth that the two counting paths built on top of that shared
classifier actually agree, for the same circuit under the same assignment,
on both round_robin (CNOT+CZ) and msd (phase) counts.
"""

from __future__ import annotations

import random

import pyzx as zx
from pyzx.circuit import Circuit
from pyzx.extract import count_cnot_faults, count_cz_faults, count_phase_faults
from pyzx.sigma_tracker import Flavor, SigmaTracker

_RANDOM_GATE_TYPES_1Q = ["H", "T", "T*", "S", "S*", "Z", "X"]
_RANDOM_GATE_TYPES_2Q = ["CNOT", "CZ"]


def _make_random_circuit(n_qubits: int, n_gates: int, rng: random.Random) -> Circuit:
    """Self-contained random Clifford+T circuit builder (H, T/T*, S/S*, CNOT,
    CZ, Z, X), independent of pyzx.generate/extract_circuit so this test
    exercises the classifiers directly rather than extraction's own output.
    """
    c = Circuit(n_qubits)
    for _ in range(n_gates):
        if n_qubits >= 2 and rng.random() < 0.4:
            kind = rng.choice(_RANDOM_GATE_TYPES_2Q)
            a, b = rng.sample(range(n_qubits), 2)
            c.add_gate(kind, a, b)
        else:
            kind = rng.choice(_RANDOM_GATE_TYPES_1Q)
            q = rng.randrange(n_qubits)
            if kind == "H":
                c.add_gate("HAD", q)
            elif kind == "T":
                c.add_gate("T", q)
            elif kind == "T*":
                c.add_gate("T", q, adjoint=True)
            elif kind == "S":
                c.add_gate("S", q)
            elif kind == "S*":
                c.add_gate("S", q, adjoint=True)
            elif kind == "Z":
                c.add_gate("Z", q)
            elif kind == "X":
                c.add_gate("NOT", q)
    return c


def _assert_counts_agree(circuit: Circuit, assignment: list[Flavor], trial_label: str) -> None:
    tracker = SigmaTracker(circuit, assignment).propagate()
    final_states = [int(f) for f in tracker.final_flavors()]

    cn = count_cnot_faults(circuit, final_states)
    cz = count_cz_faults(circuit, final_states)
    ph = count_phase_faults(circuit, final_states)

    backward_rr = cn['bad'] + cz['bad']
    assert backward_rr == tracker.round_robin_count, (
        f"{trial_label}: round_robin mismatch -- extraction backward pass says "
        f"{backward_rr} (cnot={cn['bad']}, cz={cz['bad']}), SigmaTracker forward "
        f"pass says {tracker.round_robin_count}. assignment={assignment}"
    )
    assert ph['bad'] == tracker.msd_count, (
        f"{trial_label}: msd mismatch -- extraction backward pass says {ph['bad']}, "
        f"SigmaTracker forward pass says {tracker.msd_count}. assignment={assignment}"
    )


class TestCostModelEquivalence:
    """SigmaTracker's forward counts must equal extraction's backward counts,
    for the same circuit under the same assignment, derived via
    SigmaTracker.final_flavors().
    """

    def test_agrees_on_random_synthetic_circuits(self) -> None:
        rng = random.Random(20260908)
        for trial in range(40):
            n = rng.randint(1, 10)
            n_gates = rng.randint(0, 40)
            circuit = _make_random_circuit(n, n_gates, rng)

            for assignment_trial in range(5):
                assignment = [rng.choice([Flavor.R, Flavor.R_PRIME]) for _ in range(n)]
                _assert_counts_agree(
                    circuit, assignment,
                    f"synthetic trial {trial} (n={n}, gates={n_gates}) assignment {assignment_trial}",
                )

    def test_agrees_on_real_extracted_circuits(self) -> None:
        rng = random.Random(1337)
        for trial in range(15):
            n = rng.randint(2, 8)
            n_gates = rng.randint(10, 60)
            seed = rng.randint(0, 10**9)
            g = zx.generate.cliffordT(n, n_gates, p_t=0.3, p_cnot=0.3, seed=seed)
            zx.simplify.full_reduce(g, quiet=True)
            circuit = zx.extract_circuit(g.copy(), quiet=True).to_basic_gates()

            for assignment_trial in range(5):
                assignment = [rng.choice([Flavor.R, Flavor.R_PRIME]) for _ in range(circuit.qubits)]
                _assert_counts_agree(
                    circuit, assignment,
                    f"extracted trial {trial} (n={n}, gates={n_gates}, seed={seed}) "
                    f"assignment {assignment_trial}",
                )
