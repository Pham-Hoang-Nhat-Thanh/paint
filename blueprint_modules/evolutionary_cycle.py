from dataclasses import dataclass, field
from typing import List
from .action import Action

@dataclass
class EvolutionaryCycle:
    """Tracks the state of an evolutionary cycle for architecture search"""
    cycle_id: int = 0
    node_values: List[float] = field(default_factory=list)
    actions_taken: List[Action] = field(default_factory=list)
    stability_threshold: float = 0.6
    
    def is_stable(self) -> bool:
        """Check if the cycle has reached stable performance based on recent node values"""
        if len(self.node_values) < 5:
            return False
        # Check if recent values are above threshold and stabilizing
        recent_values = self.node_values[-5:]
        return all(v >= self.stability_threshold for v in recent_values)
    
    def add_evaluation(self, value: float):
        """Add a node value evaluation to the cycle"""
        self.node_values.append(value)
    
    def add_action(self, action: Action):
        """Add an action taken in this cycle"""
        self.actions_taken.append(action)
    
    def reset(self):
        """Reset the cycle for a new evolutionary phase"""
        self.cycle_id += 1
        self.node_values.clear()
        self.actions_taken.clear()
    
    def should_prioritize_deisolation(self) -> bool:
        """Determine if we should prioritize de-isolation actions"""
        # Always prioritize de-isolation if cycle is not stable
        return not self.is_stable()
    
    def should_prioritize_exploration(self) -> bool:
        """Determine if we should prioritize exploration actions"""
        # Prioritize exploration if stable and enough actions taken
        return self.is_stable() and len(self.actions_taken) > 5
