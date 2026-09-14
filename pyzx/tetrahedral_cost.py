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

This module also carries `expand_rr_weight`, the two-resource weight
contract shared by the cost-aware extraction pipeline: the paper's cost
model treats round-robin CZ and the forbidden-direction CNOT as a single
resource, so `w_cnot == w_cz` everywhere except standalone `extract_circuit`
(kept independent there for generality).
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


def expand_rr_weight(w_rr: float, w_msd: float) -> tuple[float, float, float]:
    """Expand the two-resource (w_rr, w_msd) weight pair into extraction's
    three-weight (w_cnot, w_cz, w_msd) signature, enforcing w_cnot == w_cz == w_rr.

    The paper's cost model treats round-robin CZ and the forbidden-direction
    CNOT as a single resource (one weight, w_rr). extract.py's standalone
    extract_circuit keeps independent w_cnot/w_cz parameters for generality,
    but every cost-aware pipeline entry point (multi_restart_extract, and any
    direct extract_circuit call made for the paper's cost-aware experiments)
    must not let them differ. Constructing weights through this one helper,
    rather than passing w_cnot/w_cz by hand at each call site, is what makes
    an accidental w_cnot != w_cz impossible to introduce.

    Returns (w_cnot, w_cz, w_msd) == (w_rr, w_rr, w_msd).
    """
    return w_rr, w_rr, w_msd


def combined_cost(rr_count: int, msd_count: int, *, w_msd: float = 1.0) -> float:
    """Single combined reporting cost for an extracted circuit.

    Reporting-only helper: an encoding-mismatched T falls back to a
    wrong-direction transversal operation, so it belongs in the same
    "expensive non-transversal operation" currency as a round-robin CZ or
    forbidden-direction CNOT. This collapses the two-resource
    (rr_count, msd_count) pair extraction/SigmaTracker track separately into
    one number for reporting, at exchange rate `w_msd` (mismatched-T cost in
    units of one expensive two-qubit op).

    Not used anywhere in the extraction objective or `run_cost_aware_pipeline`
    -- the optimizer must keep seeing both resources separately, since a run
    with few two-qubit ops but many mismatched T gates is still bad
    internally. `w_msd` defaults to 1.0 (no evidence in the code for a
    different exchange rate; see CostAccumulator/SigmaTracker's own default
    weights), but is a required-explicit parameter here so a caller reporting
    at a different ratio can't silently fall back to the wrong one.

    Parameters
    ----------
    rr_count : int
        Count of expensive two-qubit ops (bad CNOTs + bad CZs).
    msd_count : int
        Count of encoding-mismatched T/S phase gates.
    w_msd : float
        Cost of one mismatched T in units of one expensive two-qubit op.

    Returns
    -------
    float
        rr_count + w_msd * msd_count.
    """
    return rr_count + w_msd * msd_count
