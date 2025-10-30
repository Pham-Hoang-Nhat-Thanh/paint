from enum import Enum
from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
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

class ActionSpace:
    """Manages valid actions for a given neural architecture"""

    def __init__(self, max_neurons: int = 1000, max_connections: int = 10000,
                 max_steps_per_episode: int = 50, connection_candidate_multiplier: int = 3):
        self.max_neurons = max_neurons
        self.max_connections = max_connections
        self.max_steps_per_episode = max_steps_per_episode
        self.connection_candidate_multiplier = connection_candidate_multiplier
    
    def get_valid_actions(self, architecture: NeuralArchitecture, current_step: int = 0,
                         evolutionary_cycle: Optional['EvolutionaryCycle'] = None) -> List[Action]:
        """Get all valid actions for the current architecture state with optimized constraints"""
        valid_actions = []

        # Check episode step limit
        if current_step >= self.max_steps_per_episode:
            return valid_actions  # No more actions allowed

        # Determine prioritization strategy based on evolutionary cycle
        if evolutionary_cycle is not None:
            # Cycle-aware prioritization: evolutionary cycling behavior
            self._add_cycle_aware_actions(valid_actions, architecture, current_step, evolutionary_cycle)
        else:
            # Fallback: simple prioritization for policy network compatibility
            self._add_simple_prioritized_actions(valid_actions, architecture, current_step)

        return valid_actions
    
    def apply_action(self, architecture: NeuralArchitecture, action: Action) -> bool:
        """Apply an action to the architecture, return success status"""
        try:
            if action.action_type == ActionType.ADD_NEURON:
                # Add hidden neuron at random layer position between input and output
                layer_pos = np.random.uniform(0.1, 0.9)
                architecture.add_neuron(NeuronType.HIDDEN, action.activation, layer_pos)
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
            return False

        return False

    def _add_cycle_aware_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture,
                                current_step: int, evolutionary_cycle: 'EvolutionaryCycle'):
        """Add actions with evolutionary cycle-aware prioritization"""
        neurons = architecture.neurons
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]
        num_neurons = len(neurons)
        num_connections = len(architecture.connections)

        # Always allow structural changes (add/remove neurons)
        if num_neurons < self.max_neurons:
            for activation in [ActivationType.RELU, ActivationType.TANH]:
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

    def _add_simple_prioritized_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture,
                                       current_step: int):
        """Add actions with simple prioritization for policy network compatibility"""
        neurons = architecture.neurons
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]
        num_neurons = len(neurons)
        num_connections = len(architecture.connections)

        # Always allow structural changes
        if num_neurons < self.max_neurons:
            for activation in [ActivationType.RELU, ActivationType.TANH]:
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
        """Get set of isolated hidden neurons"""
        isolated = set()
        for neuron_id, neuron in architecture.neurons.items():
            if neuron.neuron_type == NeuronType.HIDDEN:
                has_in = any(conn.target_id == neuron_id for conn in architecture.connections)
                has_out = any(conn.source_id == neuron_id for conn in architecture.connections)
                if not has_in or not has_out:
                    isolated.add(neuron_id)
        return isolated

    def _add_deisolation_actions(self, valid_actions: List[Action], architecture: NeuralArchitecture,
                                isolated_hidden: set):
        """Aggressively add actions that help de-isolate neurons - prioritize connections that fix isolation"""
        neurons = architecture.neurons
        all_neuron_ids = list(neurons.keys())
        existing_connections = {(conn.source_id, conn.target_id) for conn in architecture.connections}

        # Identify isolation types for each isolated neuron
        isolation_details = {}
        for neuron_id in isolated_hidden:
            has_in = any(conn.target_id == neuron_id for conn in architecture.connections)
            has_out = any(conn.source_id == neuron_id for conn in architecture.connections)
            isolation_details[neuron_id] = {'has_in': has_in, 'has_out': has_out}

        # Separate neurons by type for smarter connection generation
        input_neurons = [nid for nid, n in neurons.items() if n.neuron_type == NeuronType.INPUT]
        output_neurons = [nid for nid, n in neurons.items() if n.neuron_type == NeuronType.OUTPUT]
        hidden_neurons = [nid for nid, n in neurons.items() if n.neuron_type == NeuronType.HIDDEN]

        candidates = []
        added = set()

        # Priority 1: Fix neurons with NO connections at all (most isolated)
        completely_isolated = [nid for nid, details in isolation_details.items()
                              if not details['has_in'] and not details['has_out']]

        for isolated_id in completely_isolated:
            # Try to connect to inputs and outputs first
            for input_id in input_neurons[:5]:  # Limit to avoid explosion
                if (input_id, isolated_id) not in existing_connections and self._is_valid_connection(neurons, input_id, isolated_id):
                    candidates.append((input_id, isolated_id))
                    added.add((input_id, isolated_id))

            for output_id in output_neurons[:5]:
                if (isolated_id, output_id) not in existing_connections and self._is_valid_connection(neurons, isolated_id, output_id):
                    candidates.append((isolated_id, output_id))
                    added.add((isolated_id, output_id))

        # Priority 2: Fix neurons missing inward connections
        missing_inward = [nid for nid, details in isolation_details.items() if not details['has_in']]
        for isolated_id in missing_inward:
            # Connect from inputs and other hidden neurons
            potential_sources = input_neurons + [h for h in hidden_neurons if h != isolated_id]
            for source_id in potential_sources[:10]:  # More candidates for missing inward
                if (source_id, isolated_id) not in existing_connections and self._is_valid_connection(neurons, source_id, isolated_id):
                    candidates.append((source_id, isolated_id))
                    added.add((source_id, isolated_id))

        # Priority 3: Fix neurons missing outward connections
        missing_outward = [nid for nid, details in isolation_details.items() if not details['has_out']]
        for isolated_id in missing_outward:
            # Connect to outputs and other hidden neurons
            potential_targets = output_neurons + [h for h in hidden_neurons if h != isolated_id]
            for target_id in potential_targets[:10]:  # More candidates for missing outward
                if (isolated_id, target_id) not in existing_connections and self._is_valid_connection(neurons, isolated_id, target_id):
                    candidates.append((isolated_id, target_id))
                    added.add((isolated_id, target_id))

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
                for new_activation in [ActivationType.RELU, ActivationType.TANH, ActivationType.SIGMOID]:
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

            candidates = []
            for _ in range(min(15, len(all_neuron_ids) ** 2 // 10)):
                source_id = np.random.choice(all_neuron_ids)
                target_id = np.random.choice(all_neuron_ids)
                if source_id != target_id and (source_id, target_id) not in existing_connections:
                    if self._is_valid_connection(neurons, source_id, target_id):
                        candidates.append((source_id, target_id))

            for source_id, target_id in candidates[:5]:  # Limit for efficiency
                valid_actions.append(Action(
                    action_type=ActionType.ADD_CONNECTION,
                    source_neuron=source_id,
                    target_neuron=target_id
                ))

    def _is_valid_connection(self, neurons: Dict[int, Neuron], source_id: int, target_id: int) -> bool:
        """Check if a connection between two neurons is valid based on neuron types"""
        source_type = neurons[source_id].neuron_type
        target_type = neurons[target_id].neuron_type

        # Prevent input neurons from connecting to each other
        if source_type == NeuronType.INPUT and target_type == NeuronType.INPUT:
            return False

        # Prevent output neurons from connecting to input or hidden neurons
        if source_type == NeuronType.OUTPUT and target_type in [NeuronType.INPUT, NeuronType.HIDDEN]:
            return False

        # Prevent hidden neurons from connecting to input neurons (backward flow)
        if source_type == NeuronType.HIDDEN and target_type == NeuronType.INPUT:
            return False

        # Prevent output neurons from connecting to output neurons
        if source_type == NeuronType.OUTPUT and target_type == NeuronType.OUTPUT:
            return False

        # Allow all other connections: input->hidden, input->output, hidden->hidden, hidden->output
        return True
