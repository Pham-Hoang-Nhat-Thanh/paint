from dataclasses import dataclass, field
from typing import List
from enum import Enum
import os
import traceback

class Phase(Enum):
    """Defines the phases of the evolutionary cycle"""
    EXPANDING = 0
    REFINEMENT = 1
    PRUNING = 2

@dataclass
class EvolutionaryCycle:
    """Tracks the state of an evolutionary cycle for architecture search"""
    cycle_id: int = 0
    node_values: List[float] = field(default_factory=list)
    stability_threshold: float = 0.001
    current_phase: Phase = Phase.EXPANDING
    # Number of add_evaluation() calls observed while in the current phase
    phase_iteration_count: int = 0
    # Max iterations the EXPANDING phase may last before forcing an advance
    max_expanding_iterations: int = 5
    def is_stable(self) -> bool:
        """Determine whether the cycle is stable.

        Stability is based on the range (max-min) of the last N evaluations.
        To avoid spurious stability detections from very small samples (e.g.
        a single value with delta==0), require at least `min_samples`
        evaluations before declaring stability.
        """
        min_samples = 15
        if len(self.node_values) < min_samples:
            return False

        recent_values = self.node_values[-min_samples:]
        delta = max(recent_values) - min(recent_values)
        return delta <= self.stability_threshold
    
    def add_evaluation(self, value: float):
        """Add a node value evaluation to the cycle"""
        self.node_values.append(value)
        # Track how many evaluations we've seen in this phase
        try:
            self.phase_iteration_count += 1
        except Exception:
            self.phase_iteration_count = 1

    def should_advance(self, num_neurons = None, num_connections = None) -> bool:
        """Decide whether the episode-level cycle should advance.

        Returns True when either stability has been reached, or when the
        EXPANDING phase has lasted for at least `max_expanding_iterations`.
        This prevents the expanding phase from persisting indefinitely.
        """
        # If stable by value-range criterion, advance
        if self.is_stable():
            return True

        # Force advance for EXPANDING phase after a limited number of iterations
        if self.current_phase == Phase.EXPANDING and self.phase_iteration_count >= self.max_expanding_iterations:
            return True
        
        # Force advance for PRUNING phase if the neurons/connections count is very low
        if self.current_phase == Phase.PRUNING:
            if (num_neurons is not None and num_neurons <= 5) or \
               (num_connections is not None and num_connections <= 10):
                return True

        return False

    def copy(self) -> 'EvolutionaryCycle':
        """Return a shallow copy of the cycle suitable for local/search use.

        The copy is independent (lists are shallow-copied) so searches can
        advance phases locally without mutating the episode-level cycle.
        """
        return EvolutionaryCycle(
            cycle_id=self.cycle_id,
            node_values=list(self.node_values),
            stability_threshold=self.stability_threshold,
            current_phase=self.current_phase
        )
    
    def advance_phase(self):
        """Advances to the next phase and resets the cycle's tracking data."""
        # Optional tracing: when enabled via environment variable, print
        # a stack trace and object id so callers that unexpectedly trigger
        # phase advances during simulations can be identified.
        try:
            if os.getenv('PAINT_TRACE_PHASE_ADVANCE') == '1':
                print(f"[PAINT TRACE] advance_phase called on EvolutionaryCycle id={id(self)} current_phase={self.current_phase}")
                for line in traceback.format_stack():
                    print(line.strip())
        except Exception:
            pass

        # Advance the phase and reset tracking
        self.cycle_id += 1
        self.current_phase = Phase((self.current_phase.value + 1) % len(Phase))
        self.node_values.clear()
        # Reset the per-phase iteration counter on phase advancement
        self.phase_iteration_count = 0
        return self.cycle_id
    
    def get_phase_value(self):
        """Return the semantic phase value (the one to pass to policy/mask APIs)."""
        # Keep calling code simple: always use this helper for the "phase value"
        return self.current_phase.value