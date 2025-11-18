from dataclasses import dataclass, field
from typing import List
from enum import Enum
import os
import traceback

class Phase(Enum):
    """Defines the phases of the evolutionary cycle.

    Attributes:
        EXPANDING: The phase for adding new neurons.
        REFINEMENT: The phase for adding connections and modifying activations.
        PRUNING: The phase for removing neurons and connections.
    """
    EXPANDING = 0
    REFINEMENT = 1
    PRUNING = 2

@dataclass
class EvolutionaryCycle:
    """Tracks the state of an evolutionary cycle for architecture search.

    Attributes:
        cycle_id (int): The ID of the current cycle.
        node_values (List[float]): A list of node value evaluations.
        stability_threshold (float): The threshold for stability.
        current_phase (Phase): The current phase of the cycle.
        phase_iteration_count (int): The number of iterations in the current phase.
        max_expanding_iterations (int): The maximum number of iterations for the
            expanding phase.
    """
    cycle_id: int = 0
    node_values: List[float] = field(default_factory=list)
    stability_threshold: float = 0.001
    current_phase: Phase = Phase.EXPANDING
    # Number of add_evaluation() calls observed while in the current phase
    phase_iteration_count: int = 0
    # Max iterations the EXPANDING phase may last before forcing an advance
    max_expanding_iterations: int = 5
    def is_stable(self) -> bool:
        """Determines whether the cycle is stable.

        Stability is based on the range (max-min) of the last N evaluations.

        Returns:
            bool: True if the cycle is stable, False otherwise.
        """
        min_samples = 15
        if len(self.node_values) < min_samples:
            return False

        recent_values = self.node_values[-min_samples:]
        delta = max(recent_values) - min(recent_values)
        return delta <= self.stability_threshold
    
    def add_evaluation(self, value: float):
        """Adds a node value evaluation to the cycle.

        Args:
            value (float): The value of the node evaluation.
        """
        self.node_values.append(value)
        # Track how many evaluations we've seen in this phase
        try:
            self.phase_iteration_count += 1
        except Exception:
            self.phase_iteration_count = 1

    def should_advance(self) -> bool:
        """Decides whether the episode-level cycle should advance.

        Returns True when either stability has been reached, or when the
        EXPANDING phase has lasted for at least `max_expanding_iterations`.

        Returns:
            bool: True if the cycle should advance, False otherwise.
        """
        # If stable by value-range criterion, advance
        if self.is_stable():
            return True

        # Force advance for EXPANDING phase after a limited number of iterations
        if self.current_phase == Phase.EXPANDING and self.phase_iteration_count >= self.max_expanding_iterations:
            return True

        return False

    def copy(self) -> 'EvolutionaryCycle':
        """Returns a shallow copy of the cycle.

        The copy is independent so searches can advance phases locally without
        mutating the episode-level cycle.

        Returns:
            EvolutionaryCycle: A shallow copy of the cycle.
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
        """Returns the semantic phase value.

        Returns:
            int: The value of the current phase.
        """
        # Keep calling code simple: always use this helper for the "phase value"
        return self.current_phase.value
