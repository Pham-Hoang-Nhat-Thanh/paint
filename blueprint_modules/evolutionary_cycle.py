from dataclasses import dataclass, field
from typing import List
from enum import Enum

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
    def is_stable(self) -> bool:
        if len(self.node_values) < 15:
            return False
        recent_values = self.node_values[-15:]
        delta = max(recent_values) - min(recent_values)
        return delta <= self.stability_threshold  # or another small value
    
    def add_evaluation(self, value: float):
        """Add a node value evaluation to the cycle"""
        self.node_values.append(value)

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
        self.cycle_id += 1
        self.current_phase = Phase((self.current_phase.value + 1) % len(Phase))
        self.node_values.clear()