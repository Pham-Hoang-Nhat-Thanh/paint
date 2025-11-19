from dataclasses import dataclass, field
from typing import List
from enum import Enum
import os
import traceback

class Phase(Enum):
    """Defines the phases of the evolutionary search cycle.

    - EXPANDING: Focuses on adding new neurons to the architecture.
    - REFINEMENT: Focuses on adding connections and modifying activations.
    - PRUNING: Focuses on removing unnecessary neurons and connections.
    """
    EXPANDING = 0
    REFINEMENT = 1
    PRUNING = 2

@dataclass
class EvolutionaryCycle:
    """Manages the state of the evolutionary search cycle.

    This class tracks the current phase of the search, monitors for stability,
    and determines when to transition to the next phase.

    Attributes:
        cycle_id (int): A unique identifier for the current cycle.
        node_values (List[float]): A list of node values from recent
            evaluations, used to check for stability.
        stability_threshold (float): The threshold for value stability that
            can trigger a phase transition.
        current_phase (Phase): The current phase of the cycle.
        phase_iteration_count (int): The number of iterations spent in the
            current phase.
        max_expanding_iterations (int): The maximum number of iterations for
            the EXPANDING phase.
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
        """Determines if the search has reached a stable state.

        Stability is determined by checking if the range of the last N evaluation
        values is below a certain threshold.

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
        """Adds a new evaluation value to the cycle's history.

        Args:
            value (float): The evaluation value to add.
        """
        self.node_values.append(value)
        # Track how many evaluations we've seen in this phase
        try:
            self.phase_iteration_count += 1
        except Exception:
            self.phase_iteration_count = 1

    def should_advance(self, num_neurons = None, num_connections = None) -> bool:
        """Determines if the cycle should advance to the next phase.

        The cycle advances if the search is stable or if the current phase has
        exceeded its maximum allowed iterations.

        Args:
            num_neurons (int, optional): The current number of neurons.
            num_connections (int, optional): The current number of connections.

        Returns:
            bool: True if the cycle should advance, False otherwise.
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
        """Creates a copy of the evolutionary cycle.

        Returns:
            EvolutionaryCycle: A new instance with the same state.
        """
        return EvolutionaryCycle(
            cycle_id=self.cycle_id,
            node_values=list(self.node_values),
            stability_threshold=self.stability_threshold,
            current_phase=self.current_phase
        )
    
    def advance_phase(self):
        """Transitions the cycle to the next phase.

        This method updates the current phase and resets the cycle's tracking
        data.
        """
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
        """Returns the integer value of the current phase.

        Returns:
            int: The integer representation of the current phase.
        """
        # Keep calling code simple: always use this helper for the "phase value"
        return self.current_phase.value
