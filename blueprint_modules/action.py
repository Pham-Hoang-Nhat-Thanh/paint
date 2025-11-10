from enum import Enum
from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
import traceback
from .network import NeuralArchitecture, NeuronType, ActivationType, Neuron

if TYPE_CHECKING:
    from .evolutionary_cycle import EvolutionaryCycle

class ActionType(Enum):
    ADD_NEURON = 0
    REMOVE_NEURON = 1
    MODIFY_ACTIVATION = 2
    ADD_CONNECTION = 3
    REMOVE_CONNECTION = 4

@dataclass
class Action:
    action_type: ActionType
    source_neuron: Optional[int] = None
    target_neuron: Optional[int] = None
    activation: Optional[ActivationType] = None
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    def __eq__(self, other):
        """Two actions are equal if all their fields match"""
        if not isinstance(other, Action):
            return False
        return (self.action_type == other.action_type and
                self.source_neuron == other.source_neuron and
                self.target_neuron == other.target_neuron and
                self.activation == other.activation)
    
    def __hash__(self):
        """Make Action hashable so it can be used in sets and as dict keys"""
        return hash((self.action_type, self.source_neuron, self.target_neuron, self.activation))

class ActionSpace:
    """Manages valid actions for a given neural architecture"""

    def __init__(self, max_neurons: int = 1000, max_connections: int = 10000,
                 max_steps_per_episode: int = 50, connection_candidate_multiplier: int = 3):
        self.max_neurons = max_neurons
        self.max_connections = max_connections
        self.max_steps_per_episode = max_steps_per_episode
        self.connection_candidate_multiplier = connection_candidate_multiplier
    
    def get_valid_actions(self, architecture: NeuralArchitecture,
                         evolutionary_cycle: Optional['EvolutionaryCycle'] = None) -> List[Action]:
        """Get all valid actions for the current architecture state with optimized constraints"""
        valid_actions = []

        # Determine prioritization strategy based on evolutionary cycle
        if evolutionary_cycle is not None:
            # Cycle-aware prioritization: evolutionary cycling behavior
            self._add_cycle_aware_actions(valid_actions, architecture, evolutionary_cycle)
        else:
            # Fallback: simple prioritization for policy network compatibility
            self._add_simple_prioritized_actions(valid_actions, architecture) 
        return valid_actions
    
    def apply_action(self, architecture: NeuralArchitecture, action: Action) -> bool:
        """Apply an action to the architecture, return success status"""
        try:
            if action.action_type == ActionType.ADD_NEURON:
                # New neurons are added as isolated (layer -1)
                # Layer will be recalculated on-demand when connections are added/removed
                # Use a placeholder layer_position (will be ignored for isolated neurons)
                layer_position = 0.5
                architecture.add_neuron(NeuronType.HIDDEN, action.activation, layer_position)
                return True
                
            elif action.action_type == ActionType.REMOVE_NEURON:
                return architecture.remove_neuron(action.source_neuron)
                
            elif action.action_type == ActionType.MODIFY_ACTIVATION:
                if action.source_neuron in architecture.neurons:
                    neuron = architecture.neurons[action.source_neuron]
                    if neuron.neuron_type == NeuronType.HIDDEN:
                        neuron.activation = action.activation
                        return True
                return False
                
            elif action.action_type == ActionType.ADD_CONNECTION:
                return architecture.add_connection(
                    action.source_neuron, 
                    action.target_neuron,
                    weight=np.random.normal(0, 0.1)
                )
                
            elif action.action_type == ActionType.REMOVE_CONNECTION:
                return architecture.remove_connection(
                    action.source_neuron, 
                    action.target_neuron
                )
                
        except Exception as e:
            print(f"Error applying action {action}: {e}")
            traceback.print_exc()
            return False

        return False

    def _add_cycle_aware_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture,
                                evolutionary_cycle: 'EvolutionaryCycle'):
        """Add actions with evolutionary cycle-aware prioritization"""
        neurons = architecture.neurons
        num_neurons = len(neurons)
        
        # Filter hidden neurons once
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]

        # Always allow structural changes (add/remove neurons)
        if num_neurons < self.max_neurons:
            # Propose adding neurons with any available activation type
            for activation in ActivationType:
                valid_actions.append(Action(
                    action_type=ActionType.ADD_NEURON,
                    activation=activation
                ))

        if hidden_neurons:
            valid_actions.append(Action(
                action_type=ActionType.REMOVE_NEURON,
                source_neuron=np.random.choice(hidden_neurons)
            ))

        # Determine phase based on cycle stability and isolated neurons
        isolated_hidden = self._get_isolated_hidden_neurons(architecture)
        prioritize_deisolation = len(isolated_hidden) > 0 and evolutionary_cycle.should_prioritize_deisolation()

        if prioritize_deisolation:
            # De-isolation phase: prioritize connections that fix isolated neurons
            self._add_deisolation_actions(valid_actions, architecture, isolated_hidden)
        else:
            # Refinement phase: allow activation changes and general connections
            self._add_refinement_actions(valid_actions, architecture, hidden_neurons)

        # Always allow connection removal if we have connections
        if architecture.connections:
            sample_conns = np.random.choice(architecture.connections,
                                          size=min(5, len(architecture.connections)), replace=False)
            for conn in sample_conns:
                valid_actions.append(Action(
                    action_type=ActionType.REMOVE_CONNECTION,
                    source_neuron=conn.source_id,
                    target_neuron=conn.target_id
                ))

    def _add_simple_prioritized_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture):
        """Add actions with simple prioritization for policy network compatibility"""
        neurons = architecture.neurons
        num_neurons = len(neurons)
        
        # Filter hidden neurons once
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]

        # Always allow structural changes
        if num_neurons < self.max_neurons:
            # Propose adding neurons with all activation types supported by the network
            for activation in ActivationType:
                valid_actions.append(Action(
                    action_type=ActionType.ADD_NEURON,
                    activation=activation
                ))

        if hidden_neurons:
            valid_actions.append(Action(
                action_type=ActionType.REMOVE_NEURON,
                source_neuron=np.random.choice(hidden_neurons)
            ))

        # Check for isolated neurons and prioritize de-isolation
        isolated_hidden = self._get_isolated_hidden_neurons(architecture)

        if isolated_hidden:
            # Prioritize de-isolation
            self._add_deisolation_actions(valid_actions, architecture, isolated_hidden)
        else:
            # Allow refinement actions
            self._add_refinement_actions(valid_actions, architecture, hidden_neurons)

        # Connection removal
        if architecture.connections:
            sample_conns = np.random.choice(architecture.connections,
                                          size=min(5, len(architecture.connections)), replace=False)
            for conn in sample_conns:
                valid_actions.append(Action(
                    action_type=ActionType.REMOVE_CONNECTION,
                    source_neuron=conn.source_id,
                    target_neuron=conn.target_id
                ))

    def _get_isolated_hidden_neurons(self, architecture: NeuralArchitecture) -> set:
        """Get set of isolated hidden neurons - optimized with set operations"""
        # Build connectivity sets in single pass through connections
        targets = set()
        sources = set()
        for conn in architecture.connections:
            targets.add(conn.target_id)
            sources.add(conn.source_id)
        
        # Find isolated hidden neurons (missing incoming OR outgoing connections)
        isolated = set()
        for neuron_id, neuron in architecture.neurons.items():
            if neuron.neuron_type == NeuronType.HIDDEN:
                if neuron_id not in targets or neuron_id not in sources:
                    isolated.add(neuron_id)
        return isolated

    def _add_deisolation_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture,
                                isolated_hidden: set):
        """Aggressively add actions that help de-isolate neurons - prioritize connections that fix isolation"""
        neurons = architecture.neurons
        existing_connections = {(conn.source_id, conn.target_id) for conn in architecture.connections}

        # Build connectivity sets efficiently in single pass
        targets = set()
        sources = set()
        for conn in architecture.connections:
            targets.add(conn.target_id)
            sources.add(conn.source_id)
        
        # Identify isolation types for each isolated neuron using sets (O(1) lookup)
        isolation_details = {
            neuron_id: {
                'has_in': neuron_id in targets,
                'has_out': neuron_id in sources
            }
            for neuron_id in isolated_hidden
        }

        # Separate neurons by type for smarter connection generation (single pass)
        input_neurons = []
        output_neurons = []
        hidden_neurons = []
        neuron_type_cache = {}  # Cache for faster type lookups
        
        for nid, n in neurons.items():
            neuron_type_cache[nid] = n.neuron_type
            if n.neuron_type == NeuronType.INPUT:
                input_neurons.append(nid)
            elif n.neuron_type == NeuronType.OUTPUT:
                output_neurons.append(nid)
            elif n.neuron_type == NeuronType.HIDDEN:
                hidden_neurons.append(nid)

        candidates = []
        added = set()

        # Priority 1: Fix neurons with NO connections at all (most isolated)
        completely_isolated = [nid for nid, details in isolation_details.items()
                              if not details['has_in'] and not details['has_out']]

        # Pre-slice neuron lists for better performance
        input_sample = input_neurons[:len(input_neurons)//2 + 1]
        output_sample = output_neurons[:len(output_neurons)//2 + 1]
        hidden_sample = hidden_neurons[:len(hidden_neurons)//2 + 1]
        
        for isolated_id in completely_isolated:
            # Try to connect to inputs and outputs first
            # Optimized: inputs->hidden and hidden->outputs are always valid, skip validation
            for input_id in input_sample:
                conn_key = (input_id, isolated_id)
                if conn_key not in existing_connections and conn_key not in added:
                    candidates.append(conn_key)  # input->hidden always valid
                    added.add(conn_key)

            # Connect from hidden to hidden
            for hidden_id in hidden_sample:
                if hidden_id != isolated_id:
                    conn_key = (hidden_id, isolated_id)
                    if conn_key not in existing_connections and conn_key not in added:
                        if self._is_valid_connection(neurons, hidden_id, isolated_id):
                            candidates.append(conn_key)
                            added.add(conn_key)

            for output_id in output_sample:
                conn_key = (isolated_id, output_id)
                if conn_key not in existing_connections and conn_key not in added:
                    candidates.append(conn_key)  # hidden->output always valid
                    added.add(conn_key)

        # Priority 2: Fix neurons missing inward connections
        missing_inward = [nid for nid, details in isolation_details.items() if not details['has_in']]
        for isolated_id in missing_inward:
            # Connect from inputs and other hidden neurons (pre-filter and slice)
            potential_sources = input_neurons + [h for h in hidden_neurons if h != isolated_id]
            for source_id in potential_sources[:10]:
                conn_key = (source_id, isolated_id)
                if conn_key not in existing_connections and conn_key not in added:
                    if self._is_valid_connection(neurons, source_id, isolated_id):
                        candidates.append(conn_key)
                        added.add(conn_key)

        # Priority 3: Fix neurons missing outward connections
        missing_outward = [nid for nid, details in isolation_details.items() if not details['has_out']]
        for isolated_id in missing_outward:
            # Connect to outputs and other hidden neurons (pre-filter and slice)
            potential_targets = output_neurons + [h for h in hidden_neurons if h != isolated_id]
            for target_id in potential_targets[:10]:
                conn_key = (isolated_id, target_id)
                if conn_key not in existing_connections and conn_key not in added:
                    if self._is_valid_connection(neurons, isolated_id, target_id):
                        candidates.append(conn_key)
                        added.add(conn_key)

        # Add all unique candidates (no arbitrary limit - let MCTS decide)
        for source_id, target_id in candidates:
            valid_actions.append(Action(
                action_type=ActionType.ADD_CONNECTION,
                source_neuron=source_id,
                target_neuron=target_id
            ))

    def _add_refinement_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture,
                               hidden_neurons: List[int]):
        """Add refinement actions: activation changes and general connections"""
        neurons = architecture.neurons

        # Activation modifications
        if hidden_neurons:
            sample_neurons = np.random.choice(hidden_neurons, size=min(3, len(hidden_neurons)), replace=False)
            for neuron_id in sample_neurons:
                current_activation = neurons[neuron_id].activation
                # Allow modifying activation to any supported ActivationType (except current)
                for new_activation in ActivationType:
                    if new_activation != current_activation:
                        valid_actions.append(Action(
                            action_type=ActionType.MODIFY_ACTIVATION,
                            source_neuron=neuron_id,
                            activation=new_activation
                        ))

        # General connections (not just de-isolation)
        if len(architecture.connections) < self.max_connections:
            all_neuron_ids = list(neurons.keys())
            existing_connections = {(conn.source_id, conn.target_id) for conn in architecture.connections}

            # Pre-compute number of attempts
            num_attempts = len(all_neuron_ids) ** 2 // 1000 * self.connection_candidate_multiplier # Adjusted for efficiency
            candidates = []
            added_in_loop = set()
            
            for _ in range(num_attempts):
                source_id = np.random.choice(all_neuron_ids)
                target_id = np.random.choice(all_neuron_ids)
                conn_key = (source_id, target_id)
                
                if source_id != target_id and conn_key not in existing_connections and conn_key not in added_in_loop:
                    if self._is_valid_connection(neurons, source_id, target_id):
                        candidates.append(conn_key)
                        added_in_loop.add(conn_key)

            for source_id, target_id in candidates: 
                valid_actions.append(Action(
                    action_type=ActionType.ADD_CONNECTION,
                    source_neuron=source_id,
                    target_neuron=target_id
                ))

    def _is_valid_connection(self, neurons: Dict[int, Neuron], source_id: int, target_id: int) -> bool:
        """Check if a connection between two neurons is valid based on neuron types - optimized"""
        source_type = neurons[source_id].neuron_type
        target_type = neurons[target_id].neuron_type

        # Fast path: Allow input->hidden, input->output, hidden->hidden, hidden->output
        # Reject everything else (fewer comparisons)
        
        # Output neurons can't be sources (except to themselves, which is already prevented)
        if source_type == NeuronType.OUTPUT:
            return False
        
        # Input neurons can't be targets
        if target_type == NeuronType.INPUT:
            return False
        
        # All other combinations are valid: input->hidden, input->output, hidden->hidden, hidden->output
        return True
