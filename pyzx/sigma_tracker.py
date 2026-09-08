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
from typing import TYPE_CHECKING

from .tetrahedral_cost import _PHASE_GATE_NAMES, is_expensive_misplaced_phase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .circuit import Circuit



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

# Pauli names — always free on any CSS code
_PAULI_NAMES = frozenset({"Z", "NOT", "X", "Y"})

# Phase-gate classification (is a T/S/ZPhase/XPhase gate expensive given the
# qubit's flavor?) is implemented once, in pyzx.tetrahedral_cost, and shared
# with pyzx.extract's CostAccumulator/count_phase_faults -- see
# _PHASE_GATE_NAMES / is_expensive_misplaced_phase, imported above.


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

            # --- Phase gates (T, S, ZPhase, XPhase): expensive on R' or R ---
            # depending on gate/phase, per the shared is_expensive_misplaced_phase
            # classifier. Z-diagonal-type gates (T, S, non-Pauli ZPhase) are
            # expensive at state 0 (R'); XPhase-type (non-Pauli XPhase) at
            # state 1 (R); Paulis are never expensive at either state.
            elif name in _PHASE_GATE_NAMES:
                q = gate.target  # type: ignore[attr-defined]
                phase = gate.phase  # type: ignore[attr-defined]
                bad_init: Flavor | None
                if is_expensive_misplaced_phase(name, phase, 0):
                    # Expensive iff σ_q(0) ⊕ h_parity_q = R' (=0)
                    # ⇔ σ_q(0) = 0 ⊕ h_parity_q
                    bad_init = Flavor(0 ^ parity[q])
                elif is_expensive_misplaced_phase(name, phase, 1):
                    # Expensive iff σ_q(0) ⊕ h_parity_q = R (=1)
                    # ⇔ σ_q(0) = 1 ⊕ h_parity_q
                    bad_init = Flavor(1 ^ parity[q])
                else:
                    bad_init = None  # Pauli phase: never expensive, no potential cost.

                if bad_init is not None:
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

            # Paulis, SWAP, identity, etc. → always free, no parity change
            # (XPhase is handled above, in the _PHASE_GATE_NAMES branch.)

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

    def pareto_assignments(self) -> list[tuple[list[Flavor], int, int]]:
        """Compute the exact, weight-free Pareto frontier over all 2^n initial
        assignments in the (round_robin_count, msd_count) space.

        Assignment `a` dominates `b` iff `rr_a <= rr_b` and `msd_a <= msd_b`,
        with at least one strict inequality. Every one of the 2^n assignments
        is actually evaluated, so the returned frontier is exact -- dominance
        needs no weights.

        Uses the skyline algorithm below rather than a naive O(4^n) pairwise
        dominance filter:
          1. Enumerate all 2^n bit patterns, computing (rr, msd) via the same
             fast inner loop as _optimize_brute_force (iterate the
             precomputed structure.potential_costs, testing expensive_if).
          2. Dedupe per distinct rr, keeping only the minimum-msd
             representative (any tie-break on bits is fine -- every other
             point at that rr is dominated by it).
          3. Sort the survivors by rr ascending.
          4. Sweep once with running_min_msd = +inf, keeping a point iff its
             msd is strictly less than running_min_msd (a point whose msd
             equals the running min is dominated by an earlier, smaller-rr
             point), then updating running_min_msd.

        Complexity: O(2^n * |potential_costs| + 2^n log 2^n). Only feasible
        for n <= 24 (same guard as _optimize_brute_force).

        Returns:
        -------
        List of (assignment, rr_count, msd_count), sorted by rr_count
        ascending (equivalently msd_count descending) -- a proper staircase:
        no two entries share an rr_count or an msd_count.
        """
        n = self.n_qubits
        if n > 24:
            msg = (
                f"Brute-force over {n} qubits (2^{n} = {2**n:,} assignments) "
                f"is infeasible.  Consider a heuristic method."
            )
            raise ValueError(msg)

        potential_costs = self.structure.potential_costs

        # Step 1: enumerate all 2^n points as (rr, msd, bits).
        points: list[tuple[int, int, int]] = []
        for bits in range(1 << n):
            rr = 0
            msd = 0
            for pc in potential_costs:
                triggered = True
                for q, required in pc.expensive_if.items():
                    if ((bits >> q) & 1) != required:
                        triggered = False
                        break
                if triggered:
                    if pc.cost_type == CostType.MSD:
                        msd += 1
                    else:
                        rr += 1
            points.append((rr, msd, bits))

        # Step 2: dedupe per rr, keeping the minimum-msd representative.
        best_for_rr: dict[int, tuple[int, int]] = {}  # rr -> (msd, bits)
        for rr, msd, bits in points:
            current = best_for_rr.get(rr)
            if current is None or msd < current[0]:
                best_for_rr[rr] = (msd, bits)

        # Step 3: sort survivors by rr ascending.
        survivors = sorted(
            (rr, msd, bits) for rr, (msd, bits) in best_for_rr.items()
        )

        # Step 4: sweep, keeping only strictly-decreasing msd.
        frontier: list[tuple[int, int, int]] = []
        running_min_msd = float("inf")
        for rr, msd, bits in survivors:
            if msd < running_min_msd:
                frontier.append((rr, msd, bits))
                running_min_msd = msd

        # Step 5: decode bits into assignments.
        return [
            ([Flavor((bits >> i) & 1) for i in range(n)], rr, msd)
            for rr, msd, bits in frontier
        ]

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
