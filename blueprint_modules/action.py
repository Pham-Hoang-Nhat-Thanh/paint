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
    
    def __init__(self, max_neurons: int = 500):
        self.max_neurons = max_neurons
    
    def get_valid_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Get all valid actions for the current architecture state"""
        valid_actions = []
        neurons = architecture.neurons
        hidden_neurons = [nid for nid, neuron in neurons.items() 
                         if neuron.neuron_type == NeuronType.HIDDEN]
        
        # 1. ADD_NEURON action (if under max neurons)
        if len(neurons) < self.max_neurons:
            for activation in ActivationType:
                valid_actions.append(Action(
                    action_type=ActionType.ADD_NEURON,
                    activation=activation
                ))
        
        # 2. REMOVE_NEURON action (only hidden neurons)
        for neuron_id in hidden_neurons:
            valid_actions.append(Action(
                action_type=ActionType.REMOVE_NEURON,
                source_neuron=neuron_id
            ))
        
        # 3. MODIFY_ACTIVATION action (only hidden neurons)
        for neuron_id in hidden_neurons:
            current_activation = neurons[neuron_id].activation
            for new_activation in ActivationType:
                if new_activation != current_activation:
                    valid_actions.append(Action(
                        action_type=ActionType.MODIFY_ACTIVATION,
                        source_neuron=neuron_id,
                        activation=new_activation
                    ))
        
        # 4. ADD_CONNECTION action (any two different neurons)
        all_neuron_ids = list(neurons.keys())
        for source_id in all_neuron_ids:
            for target_id in all_neuron_ids:
                if source_id != target_id:
                    # Check if connection doesn't already exist
                    connection_exists = any(
                        conn.source_id == source_id and conn.target_id == target_id
                        for conn in architecture.connections
                    )
                    if not connection_exists:
                        valid_actions.append(Action(
                            action_type=ActionType.ADD_CONNECTION,
                            source_neuron=source_id,
                            target_neuron=target_id
                        ))
        
        # 5. REMOVE_CONNECTION action
        for conn in architecture.connections:
            valid_actions.append(Action(
                action_type=ActionType.REMOVE_CONNECTION,
                source_neuron=conn.source_id,
                target_neuron=conn.target_id
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