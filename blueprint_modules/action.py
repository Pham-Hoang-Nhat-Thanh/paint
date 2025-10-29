from enum import Enum
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from .network import NeuralArchitecture, NeuronType, ActivationType

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
    
    def get_valid_actions(self, architecture: NeuralArchitecture, current_step: int = 0) -> List[Action]:
        """Get all valid actions for the current architecture state with optimized constraints"""
        valid_actions = []
        neurons = architecture.neurons
        hidden_neurons = [nid for nid, neuron in neurons.items()
                         if neuron.neuron_type == NeuronType.HIDDEN]

        # Check episode step limit
        if current_step >= self.max_steps_per_episode:
            return valid_actions  # No more actions allowed

        num_neurons = len(neurons)
        num_connections = len(architecture.connections)

        # Prioritize different action types based on current architecture state
        # Early stages: focus on adding structure
        if current_step < self.max_steps_per_episode // 3:
            # 1. ADD_NEURON action (if under max neurons) - build structure first
            if num_neurons < self.max_neurons:
                # Limit to 2 activation types for speed
                for activation in [ActivationType.RELU, ActivationType.TANH]:
                    valid_actions.append(Action(
                        action_type=ActionType.ADD_NEURON,
                        activation=activation
                    ))

            # 2. ADD_CONNECTION action - connect early structure
            if num_connections < self.max_connections and num_neurons > 10:
                # Smart connection sampling: include both input-to-hidden and hidden-to-output connections
                input_ids = [nid for nid, n in neurons.items() if n.neuron_type == NeuronType.INPUT]
                output_ids = [nid for nid, n in neurons.items() if n.neuron_type == NeuronType.OUTPUT]
                hidden_ids = [nid for nid, n in neurons.items() if n.neuron_type == NeuronType.HIDDEN]

                # Sample input-to-hidden connections (build feedforward structure)
                if input_ids and hidden_ids:
                    for _ in range(min(5, len(input_ids) * len(hidden_ids) // 10)):
                        source_id = np.random.choice(input_ids)
                        target_id = np.random.choice(hidden_ids)
                        if not any(c.source_id == source_id and c.target_id == target_id for c in architecture.connections):
                            valid_actions.append(Action(
                                action_type=ActionType.ADD_CONNECTION,
                                source_neuron=source_id,
                                target_neuron=target_id
                            ))

                # Sample hidden-to-output connections (connect to outputs)
                if hidden_ids and output_ids:
                    for _ in range(min(5, len(hidden_ids))):
                        source_id = np.random.choice(hidden_ids)
                        target_id = np.random.choice(output_ids)
                        if not any(c.source_id == source_id and c.target_id == target_id for c in architecture.connections):
                            valid_actions.append(Action(
                                action_type=ActionType.ADD_CONNECTION,
                                source_neuron=source_id,
                                target_neuron=target_id
                            ))

        # Middle stages: refine connections and activations
        elif current_step < 2 * self.max_steps_per_episode // 3:
            # 3. MODIFY_ACTIVATION action (only hidden neurons) - refine activations
            if hidden_neurons:
                # Sample subset for speed
                sample_neurons = np.random.choice(hidden_neurons, size=min(5, len(hidden_neurons)), replace=False)
                for neuron_id in sample_neurons:
                    current_activation = neurons[neuron_id].activation
                    for new_activation in [ActivationType.RELU, ActivationType.TANH, ActivationType.SIGMOID]:
                        if new_activation != current_activation:
                            valid_actions.append(Action(
                                action_type=ActionType.MODIFY_ACTIVATION,
                                source_neuron=neuron_id,
                                activation=new_activation
                            ))

            # 4. ADD_CONNECTION action - add more connections
            if num_connections < self.max_connections:
                all_neuron_ids = list(neurons.keys())
                existing_connections = {(conn.source_id, conn.target_id) for conn in architecture.connections}

                # Identify isolated hidden neurons for prioritization
                isolated_hidden = set()
                for neuron_id, neuron in neurons.items():
                    if neuron.neuron_type == NeuronType.HIDDEN:
                        has_in = any(conn.target_neuron == neuron_id for conn in architecture.connections.values())
                        has_out = any(conn.source_neuron == neuron_id for conn in architecture.connections.values())
                        if not has_in or not has_out:
                            isolated_hidden.add(neuron_id)

                # Sample fewer candidates for speed
                max_candidates = min(50, num_neurons * 2)
                candidate_pairs = set()
                for _ in range(max_candidates):
                    source_id = np.random.choice(all_neuron_ids)
                    target_id = np.random.choice(all_neuron_ids)
                    if source_id != target_id and (source_id, target_id) not in existing_connections:
                        candidate_pairs.add((source_id, target_id))

                # Prioritize de-isolating connections: add them first (up to 10 such actions)
                de_isolating_pairs = []
                other_pairs = []
                for source_id, target_id in candidate_pairs:
                    if source_id in isolated_hidden or target_id in isolated_hidden:
                        de_isolating_pairs.append((source_id, target_id))
                    else:
                        other_pairs.append((source_id, target_id))

                # Add de-isolating actions first (limit to 10 for balance)
                for source_id, target_id in de_isolating_pairs[:10]:
                    valid_actions.append(Action(
                        action_type=ActionType.ADD_CONNECTION,
                        source_neuron=source_id,
                        target_neuron=target_id
                    ))

                # Add remaining pairs
                for source_id, target_id in other_pairs + de_isolating_pairs[10:]:
                    valid_actions.append(Action(
                        action_type=ActionType.ADD_CONNECTION,
                        source_neuron=source_id,
                        target_neuron=target_id
                    ))

        # Late stages: pruning and final optimization
        else:
            # 5. REMOVE_CONNECTION action - prune unnecessary connections
            if architecture.connections:
                # Sample subset of connections to consider removing
                sample_conns = np.random.choice(architecture.connections,
                                              size=min(10, len(architecture.connections)), replace=False)
                for conn in sample_conns:
                    valid_actions.append(Action(
                        action_type=ActionType.REMOVE_CONNECTION,
                        source_neuron=conn.source_id,
                        target_neuron=conn.target_id
                    ))

            # 6. REMOVE_NEURON action (only hidden neurons) - prune if too many
            if num_neurons > 50 and hidden_neurons:  # Only if architecture is large
                sample_neurons = np.random.choice(hidden_neurons, size=min(3, len(hidden_neurons)), replace=False)
                for neuron_id in sample_neurons:
                    valid_actions.append(Action(
                        action_type=ActionType.REMOVE_NEURON,
                        source_neuron=neuron_id
                    ))

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