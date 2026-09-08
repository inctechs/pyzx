"""Shared, dependency-free primitives for the R/R' tetrahedral-code cost model.

Convention used throughout this module and its callers: qubit state/flavor
`0 = R'`, `1 = R`.

Two independent parts of this fork compute costs under this model:
  - `pyzx.extract` (`CostAccumulator`, `count_cnot_faults`, `count_cz_faults`,
    `count_phase_faults`) -- forward accumulation during extraction, and a
    backward pass over a finished circuit from known final qubit states.
  - `pyzx.sigma_tracker` (`SigmaTracker`, `CircuitStructure`) -- forward
    propagation of flavor from an initial assignment.

Both classify "is this phase gate expensive given the qubit's current
flavor" identically. This module is the single implementation of that
classification, imported by both, so the two cost models cannot silently
diverge. It imports nothing from `.extract` or `.sigma_tracker` (no import
cycles).
"""

from __future__ import annotations

from fractions import Fraction

from .utils import FractionLike

# Gate names that carry an inherently non-Pauli Z-diagonal phase (T/T-dagger, S/S-dagger).
# ZPhase is handled separately below since its Pauli-ness depends on the phase value.
_Z_DIAG_PHASE_NAMES = frozenset({'T', 'S'})

# Gate names relevant to phase-gate fault counting / classification in general
# (count_phase_faults, CircuitStructure._analyze): the full set of gates that
# is_expensive_misplaced_phase ever needs to be consulted for.
_PHASE_GATE_NAMES = frozenset({'ZPhase', 'XPhase', 'T', 'S'})


def _is_non_pauli_phase(phase: FractionLike) -> bool:
    """True iff `phase` (in units of pi) is not an integer multiple of pi, i.e. not a Pauli."""
    if isinstance(phase, Fraction):
        return phase.denominator != 1
    return float(phase) % 1.0 != 0.0  # type: ignore[arg-type]  # FractionLike includes symbolic Poly


def is_expensive_misplaced_phase(gate_name: str, phase: FractionLike, state: int) -> bool:
    """Is a phase gate expensive given the qubit's flavor `state`?

    state: 0 = R', 1 = R.

    Z-diagonal non-Pauli phases (T, S, T-dagger, S-dagger, or a ZPhase gate with a
    non-integer phase) are expensive on R' (state == 0); X-diagonal non-Pauli phases
    (XPhase with a non-integer phase) are expensive on R (state == 1). Paulis are
    always free. This is the single implementation shared by pyzx.extract and
    pyzx.sigma_tracker, so the two cost models cannot silently diverge.
    """
    if gate_name in _Z_DIAG_PHASE_NAMES or (gate_name == 'ZPhase' and _is_non_pauli_phase(phase)):
        return state == 0
    if gate_name == 'XPhase' and _is_non_pauli_phase(phase):
        return state == 1
    return False
