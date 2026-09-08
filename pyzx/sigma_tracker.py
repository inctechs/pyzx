"""σ-state tracker for the R/R' tetrahedral code framework.

The [[15,1,3]] tetrahedral code (R) supports transversal {CNOT, T}.
Applying H transversally maps R to the "rotated" code R', which supports
transversal {CNOT (restricted), T_x}.  H itself is free — it just relabels
R ↔ R'.

This module tracks which code flavor (R or R') each logical qubit occupies
at each point in a Clifford+T circuit, identifies operations that require
expensive fault-tolerant gadgets, and can optimize the initial encoding
assignment to minimize cost.

Cost model
----------
FREE operations:
    T / T† / S / S† on an R qubit  (Z-diagonal, transversal on R)
    CNOT(c,t) unless c=R' and t=R
    CZ(a,b) unless both a=R' and b=R'
    H on any qubit  (just toggles R ↔ R')
    Paulis (X, Y, Z) on any qubit  (transversal on all CSS codes)

EXPENSIVE operations:
    T / T† / S / S† on an R' qubit  → MSD or code switch
    CNOT with control=R', target=R  → round-robin pieceable FT gadget
    CZ with both qubits in R'       → round-robin pieceable FT gadget

Key structural insight:
    σ_q(t) = σ_q(0) ⊕ h_parity_q(t)
    where h_parity_q(t) = (# of H gates on qubit q before step t) mod 2.
    The H-parity trajectory is a circuit property, independent of the
    initial assignment.  This allows separating circuit analysis from
    assignment optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .circuit import Circuit
    from .circuit.gates import Gate



# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class Flavor(IntEnum):
    """Code flavor for the R/R' framework."""
    R = 1
    R_PRIME = 0

    def flip(self) -> Flavor:
        return Flavor(1 - self.value)

    def __repr__(self) -> str:
        return "R" if self == Flavor.R else "R'"

    def __str__(self) -> str:
        return repr(self)


class CostType(IntEnum):
    """Classification of expensive operations."""
    MSD = 0          # Magic state distillation or code switch (T/S on R')
    ROUND_ROBIN = 1  # Round-robin pieceable FT gadget (bad CNOT/CZ)


@dataclass(frozen=True)
class ExpensiveOp:
    """Record of a single expensive gate instance in the circuit."""
    index: int                   # position in circuit.gates
    gate_name: str               # human-readable: 'T', 'S', 'CNOT', 'CZ', ...
    cost_type: CostType
    qubits: tuple[int, ...]      # involved qubits
    flavors: dict[int, Flavor]   # qubit → flavor at this gate position

    def __repr__(self) -> str:
        flav = ", ".join(f"q{q}={f}" for q, f in self.flavors.items())
        return (
            f"ExpensiveOp(idx={self.index}, {self.gate_name}, "
            f"{self.cost_type.name}, [{flav}])"
        )


# ---------------------------------------------------------------------------
# Gate classification helpers
# ---------------------------------------------------------------------------

# Gate names (from PyZX) that toggle the σ-flavor
_H_GATES = frozenset({"HAD"})

# Z-diagonal phase gates: free on R, expensive on R'
# These are NOT Paulis — Z itself is handled separately (it's free everywhere).
_Z_DIAG_PHASE_NAMES = frozenset({"T", "S"})

# Pauli names — always free on any CSS code
_PAULI_NAMES = frozenset({"Z", "NOT", "X", "Y"})


def _is_z_diagonal_expensive(gate: Gate) -> bool:
    """Check if a gate is a non-Pauli Z-diagonal gate (expensive on R').

    Handles T, S (and their adjoints), and also ZPhase gates with
    non-integer phase (ZPhase(1) = Z is Pauli, hence free).
    """
    if gate.name in _Z_DIAG_PHASE_NAMES:
        return True
    # ZPhase with non-integer phase (integer multiples of π are Paulis)
    if gate.name == "ZPhase":
        phase = gate.phase  # type: ignore[attr-defined]
        if isinstance(phase, Fraction):
            return phase.denominator != 1  # integer phase → Pauli → free
        return float(phase) % 1.0 != 0.0
    return False


# ---------------------------------------------------------------------------
# Precomputed circuit structure (assignment-independent)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PotentialCost:
    """A gate that *could* be expensive depending on the initial assignment.

    Stores the condition on the initial assignment bits that would make
    this gate expensive.  This allows fast cost evaluation without
    re-traversing the circuit.
    """
    gate_index: int
    gate_name: str
    cost_type: CostType
    qubits: tuple[int, ...]

    # For single-qubit gates (MSD): expensive iff σ_q(0) == expensive_if[q]
    # For two-qubit gates (ROUND_ROBIN): expensive iff all conditions hold
    # Dict maps qubit → the Flavor value of σ(0) that makes this gate expensive.
    expensive_if: dict[int, Flavor]


class CircuitStructure:
    """Assignment-independent analysis of a circuit's H-parity and gate structure.

    This is computed once per circuit and reused across all assignment evaluations.
    """

    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.n_qubits: int = circuit.qubits
        self.n_gates: int = len(circuit.gates)

        # h_parity[gate_idx][qubit] = cumulative H-parity BEFORE gate_idx executes
        # Shape: (n_gates + 1) × n_qubits  (last entry = parity after full circuit)
        self.h_parity: list[list[int]] = []

        # All potentially expensive gate positions
        self.potential_costs: list[_PotentialCost] = []

        self._analyze()

    def _analyze(self) -> None:
        """Single pass through the circuit to build h_parity and identify
        all potentially expensive gate positions.
        """
        parity = [0] * self.n_qubits  # running H-count mod 2 per qubit

        for gate_idx, gate in enumerate(self.circuit.gates):
            # Record parity state BEFORE this gate
            self.h_parity.append(list(parity))

            name = gate.name

            # --- Hadamard: toggle parity ---
            if name in _H_GATES:
                parity[gate.target] ^= 1  # type: ignore[attr-defined]

            # --- Z-diagonal non-Pauli (T, S, ZPhase) ---
            elif _is_z_diagonal_expensive(gate):
                q = gate.target  # type: ignore[attr-defined]
                # Expensive iff σ_q(0) ⊕ h_parity_q = R' (Flavor.R_PRIME = 0)
                # ⇔ σ_q(0) = 0 ⊕ h_parity_q
                bad_init = Flavor(0 ^ parity[q])
                self.potential_costs.append(_PotentialCost(
                    gate_index=gate_idx,
                    gate_name=name + ("†" if getattr(gate, "adjoint", False) else ""),
                    cost_type=CostType.MSD,
                    qubits=(q,),
                    expensive_if={q: bad_init},
                ))

            # --- CNOT: expensive iff control=R', target=R ---
            elif name == "CNOT":
                c, t = gate.control, gate.target  # type: ignore[attr-defined]
                # σ_c(0) ⊕ parity_c = R' (=0) ⇔ σ_c(0) = 0 ⊕ parity_c
                # σ_t(0) ⊕ parity_t = R  (=1) ⇔ σ_t(0) = 1 ⊕ parity_t
                bad_c = Flavor(0 ^ parity[c])
                bad_t = Flavor(1 ^ parity[t])
                self.potential_costs.append(_PotentialCost(
                    gate_index=gate_idx,
                    gate_name="CNOT",
                    cost_type=CostType.ROUND_ROBIN,
                    qubits=(c, t),
                    expensive_if={c: bad_c, t: bad_t},
                ))

            # --- CZ: expensive iff both R' ---
            elif name == "CZ":
                q1, q2 = gate.control, gate.target  # type: ignore[attr-defined]
                # σ_q1(0) ⊕ parity_q1 = R' (=0) ⇔ σ_q1(0) = 0 ⊕ parity_q1
                # σ_q2(0) ⊕ parity_q2 = R' (=0) ⇔ σ_q2(0) = 0 ⊕ parity_q2
                bad_q1 = Flavor(0 ^ parity[q1])
                bad_q2 = Flavor(0 ^ parity[q2])
                self.potential_costs.append(_PotentialCost(
                    gate_index=gate_idx,
                    gate_name="CZ",
                    cost_type=CostType.ROUND_ROBIN,
                    qubits=(q1, q2),
                    expensive_if={q1: bad_q1, q2: bad_q2},
                ))

            # --- XPhase (non-Pauli, X-diagonal): expensive on R ---
            elif name == "XPhase":
                q = gate.target  # type: ignore[attr-defined]
                phase = gate.phase  # type: ignore[attr-defined]
                is_pauli = (isinstance(phase, Fraction) and phase.denominator == 1) or (
                    float(phase) % 1.0 == 0.0
                )
                if not is_pauli:
                    # Expensive iff σ_q(0) ⊕ parity_q = R (Flavor.R = 1)
                    # ⇔ σ_q(0) = 1 ⊕ parity_q
                    bad_init = Flavor(1 ^ parity[q])
                    self.potential_costs.append(_PotentialCost(
                        gate_index=gate_idx,
                        gate_name="XPhase",
                        cost_type=CostType.MSD,
                        qubits=(q,),
                        expensive_if={q: bad_init},
                    ))

            # Paulis, SWAP, identity, etc. → always free, no parity change

        # Final parity snapshot (after all gates)
        self.h_parity.append(list(parity))

    def flavor_at(self, gate_idx: int, qubit: int, assignment: Sequence[Flavor]) -> Flavor:
        """Compute the flavor of a qubit just before a given gate,
        given an initial assignment.
        """
        return Flavor(assignment[qubit] ^ self.h_parity[gate_idx][qubit])


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class SigmaTracker:
    """Full σ-state tracker: wraps CircuitStructure with a specific initial
    assignment, evaluates cost, and supports brute-force optimization.

    Usage
    -----
    >>> tracker = SigmaTracker(circuit)
    >>> tracker.propagate()
    >>> print(tracker.cost)
    >>> print(tracker.summary())
    >>> best_assignment, best_cost = tracker.optimize_assignment()
    """

    def __init__(
        self,
        circuit: Circuit,
        initial_assignment: Sequence[Flavor] | None = None,
        w_msd: float = 1.0,
        w_rr: float = 1.0,
    ) -> None:
        """Parameters
        ----------
        circuit : pyzx.Circuit
            The Clifford+T circuit to analyze.
        initial_assignment : sequence of Flavor, optional
            Initial R/R' assignment per qubit.  Defaults to all-R.
        w_msd : float
            Cost weight for MSD / code-switch operations.
        w_rr : float
            Cost weight for round-robin gadgets.
        """
        self.structure = CircuitStructure(circuit)
        self.n_qubits = self.structure.n_qubits
        self.w_msd = w_msd
        self.w_rr = w_rr

        if initial_assignment is None:
            self.assignment = [Flavor.R] * self.n_qubits
        else:
            if len(initial_assignment) != self.n_qubits:
                msg = (
                    f"Assignment length {len(initial_assignment)} "
                    f"!= circuit qubits {self.n_qubits}"
                )
                raise ValueError(msg)
            self.assignment = list(initial_assignment)

        # Results (populated by propagate)
        self._expensive_ops: list[ExpensiveOp] = []
        self._weighted_cost: float = 0.0
        self._msd_count: int = 0
        self._rr_count: int = 0
        self._propagated: bool = False

    def propagate(self) -> SigmaTracker:
        """Evaluate cost and identify expensive gates for the current assignment.

        This is fast: it iterates over precomputed potential-cost entries
        (not over all gates), checking whether each is triggered by the
        current assignment.
        """
        self._expensive_ops = []
        self._weighted_cost = 0.0
        self._msd_count = 0
        self._rr_count = 0

        for pc in self.structure.potential_costs:
            # Check whether ALL conditions for this gate being expensive are met
            is_expensive = all(
                self.assignment[q] == required_flavor
                for q, required_flavor in pc.expensive_if.items()
            )
            if is_expensive:
                # Build the full flavor dict for reporting
                flavors = {
                    q: self.structure.flavor_at(pc.gate_index, q, self.assignment)
                    for q in pc.qubits
                }
                op = ExpensiveOp(
                    index=pc.gate_index,
                    gate_name=pc.gate_name,
                    cost_type=pc.cost_type,
                    qubits=pc.qubits,
                    flavors=flavors,
                )
                self._expensive_ops.append(op)

                if pc.cost_type == CostType.MSD:
                    self._msd_count += 1
                    self._weighted_cost += self.w_msd
                else:
                    self._rr_count += 1
                    self._weighted_cost += self.w_rr

        self._propagated = True
        return self

    def _ensure_propagated(self) -> None:
        if not self._propagated:
            self.propagate()

    @property
    def cost(self) -> float:
        """Total weighted cost."""
        self._ensure_propagated()
        return self._weighted_cost

    @property
    def msd_count(self) -> int:
        """Number of MSD / code-switch operations needed."""
        self._ensure_propagated()
        return self._msd_count

    @property
    def round_robin_count(self) -> int:
        """Number of round-robin gadgets needed."""
        self._ensure_propagated()
        return self._rr_count

    @property
    def expensive_ops(self) -> list[ExpensiveOp]:
        """List of all expensive operations with positions and details."""
        self._ensure_propagated()
        return list(self._expensive_ops)

    def flavor_at(self, gate_idx: int, qubit: int) -> Flavor:
        """Get flavor of a qubit just before gate at gate_idx."""
        return self.structure.flavor_at(gate_idx, qubit, self.assignment)

    def final_flavors(self) -> list[Flavor]:
        """Flavor assignment after the entire circuit."""
        n = self.structure.n_gates
        return [
            self.structure.flavor_at(n, q, self.assignment)
            for q in range(self.n_qubits)
        ]

    def set_assignment(self, assignment: Sequence[Flavor]) -> SigmaTracker:
        """Set a new initial assignment (invalidates cached results)."""
        if len(assignment) != self.n_qubits:
            msg = f"Assignment length {len(assignment)} != {self.n_qubits}"
            raise ValueError(msg)
        self.assignment = list(assignment)
        self._propagated = False
        return self

    # -------------------------------------------------------------------
    # Assignment optimization
    # -------------------------------------------------------------------

    def optimize_assignment(self, method: str = "brute_force") -> tuple[list[Flavor], float]:
        """Find the initial assignment minimizing total cost.

        Parameters
        ----------
        method : str
            'brute_force' — exhaustive search over all 2^n assignments.
            Only feasible for n ≲ 20.

        Returns:
        -------
        (best_assignment, best_cost)
        """
        if method == "brute_force":
            return self._optimize_brute_force()
        msg = f"Unknown optimization method: {method!r}"
        raise ValueError(msg)

    def _optimize_brute_force(self) -> tuple[list[Flavor], float]:
        """Enumerate all 2^n assignments; return the cheapest."""
        n = self.n_qubits
        if n > 24:
            msg = (
                f"Brute-force over {n} qubits (2^{n} = {2**n:,} assignments) "
                f"is infeasible.  Consider a heuristic method."
            )
            raise ValueError(msg)

        best_cost = float("inf")
        best_bits = 0

        # Precompute the potential costs list for fast inner loop
        potential_costs = self.structure.potential_costs
        w_msd = self.w_msd
        w_rr = self.w_rr

        for bits in range(1 << n):
            cost = 0.0
            for pc in potential_costs:
                # Check all conditions
                triggered = True
                for q, required in pc.expensive_if.items():
                    # Flavor.R=0, Flavor.R_PRIME=1
                    # assignment[q] = (bits >> q) & 1
                    if ((bits >> q) & 1) != required:
                        triggered = False
                        break
                if triggered:
                    cost += w_msd if pc.cost_type == CostType.MSD else w_rr

            if cost < best_cost:
                best_cost = cost
                best_bits = bits

        best_assignment = [
            Flavor((best_bits >> i) & 1) for i in range(n)
        ]

        # Apply and propagate
        self.set_assignment(best_assignment).propagate()
        return best_assignment, best_cost

    # -------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the cost analysis."""
        self._ensure_propagated()

        gate_count = self.structure.n_gates
        lines = [
            "═══ σ-Tracker Summary ═══",
            f"Circuit: {gate_count} gates, {self.n_qubits} qubits",
            f"Initial assignment: [{', '.join(str(f) for f in self.assignment)}]",
            "",
            f"Weighted cost:  {self._weighted_cost:.1f}",
            f"  MSD / code-switch: {self._msd_count}  (w={self.w_msd})",
            f"  Round-robin:       {self._rr_count}  (w={self.w_rr})",
        ]

        if self._expensive_ops:
            lines.extend(("", "Expensive operations:"))
            lines.extend(f"  {op}" for op in self._expensive_ops)
        else:
            lines.extend(("", "No expensive operations — circuit is fully transversal!"))

        return "\n".join(lines)

    def cost_breakdown_by_qubit(self) -> dict[int, dict[str, int]]:
        """Per-qubit breakdown of expensive operations.

        Returns dict: qubit → {'msd': count, 'round_robin': count}
        Useful for identifying which qubits are most problematic.
        """
        self._ensure_propagated()
        breakdown: dict[int, dict[str, int]] = {
            q: {"msd": 0, "round_robin": 0} for q in range(self.n_qubits)
        }
        for op in self._expensive_ops:
            key = "msd" if op.cost_type == CostType.MSD else "round_robin"
            for q in op.qubits:
                breakdown[q][key] += 1
        return breakdown

    def expensive_gate_indices(self) -> list[int]:
        """Sorted list of circuit gate indices that are expensive.

        Useful for downstream partitioning: these are the positions
        where the circuit needs non-transversal gadgets.
        """
        self._ensure_propagated()
        return sorted(op.index for op in self._expensive_ops)

    def free_segments(self) -> list[tuple[int, int]]:
        """Identify maximal contiguous runs of gates that are all free.

        Returns list of (start_idx, end_idx) pairs (inclusive).
        These segments can be executed entirely within the R/R' framework
        without any expensive gadgets.
        """
        expensive_indices = set(self.expensive_gate_indices())
        n = self.structure.n_gates
        if n == 0:
            return []

        segments = []
        seg_start = None

        for i in range(n):
            if i not in expensive_indices:
                if seg_start is None:
                    seg_start = i
            elif seg_start is not None:
                segments.append((seg_start, i - 1))
                seg_start = None

        if seg_start is not None:
            segments.append((seg_start, n - 1))

        return segments
