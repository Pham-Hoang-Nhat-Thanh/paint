import torch
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn

class NeuronType(Enum):
    INPUT = "input"
    HIDDEN = "hidden" 
    OUTPUT = "output"

class ActivationType(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LINEAR = "linear"

@dataclass
class Neuron:
    id: int
    neuron_type: NeuronType
    activation: ActivationType
    layer_position: float  # Rough positioning for visualization
    bias: float = 0.0
    
    def to_feature_vector(self):
        """Convert neuron to feature vector for graph representation"""
        type_encoding = {
            NeuronType.INPUT: [1, 0, 0],
            NeuronType.HIDDEN: [0, 1, 0], 
            NeuronType.OUTPUT: [0, 0, 1]
        }
        activation_encoding = {
            ActivationType.RELU: [1, 0, 0, 0],
            ActivationType.SIGMOID: [0, 1, 0, 0],
            ActivationType.TANH: [0, 0, 1, 0],
            ActivationType.LINEAR: [0, 0, 0, 1]
        }
        return type_encoding[self.neuron_type] + activation_encoding[self.activation] + [self.layer_position, self.bias]

@dataclass 
class Connection:
    source_id: int
    target_id: int
    weight: float
    enabled: bool = True

class NeuralArchitecture:
    """Graph-based representation of a neural network"""
    
    def __init__(self):
        self.neurons: Dict[int, Neuron] = {}
        self.connections: List[Connection] = []
        self.next_neuron_id = 0
        self.performance_metrics: Dict = {}
        
        # Initialize with MNIST base structure
        self._initialize_mnist_base()
    
    def _initialize_mnist_base(self):
        """Create initial 784 input + 10 output neurons for MNIST"""
        # Input neurons (784)
        for i in range(784):
            self.add_neuron(NeuronType.INPUT, ActivationType.LINEAR, layer_position=0.0)
        
        # Output neurons (10)  
        for i in range(10):
            self.add_neuron(NeuronType.OUTPUT, ActivationType.LINEAR, layer_position=1.0)
    
    def add_neuron(self, neuron_type: NeuronType, activation: ActivationType, layer_position: float) -> int:
        """Add a new neuron and return its ID"""
        neuron_id = self.next_neuron_id
        self.neurons[neuron_id] = Neuron(neuron_id, neuron_type, activation, layer_position)
        self.next_neuron_id += 1
        return neuron_id
    
    def add_connection(self, source_id: int, target_id: int, weight: float = 0.1) -> bool:
        """Add connection between neurons if valid"""
        if source_id not in self.neurons or target_id not in self.neurons:
            return False
        
        # Check if connection already exists
        for conn in self.connections:
            if conn.source_id == source_id and conn.target_id == target_id:
                return False
        
        self.connections.append(Connection(source_id, target_id, weight))
        return True
    
    def remove_neuron(self, neuron_id: int) -> bool:
        """Remove a neuron and all its connections"""
        if neuron_id not in self.neurons:
            return False
        
        neuron = self.neurons[neuron_id]
        if neuron.neuron_type in [NeuronType.INPUT, NeuronType.OUTPUT]:
            return False  # Cannot remove input/output neurons
        
        # Remove all connections involving this neuron
        self.connections = [
            conn for conn in self.connections 
            if conn.source_id != neuron_id and conn.target_id != neuron_id
        ]
        
        del self.neurons[neuron_id]
        return True
    
    def remove_connection(self, source_id: int, target_id: int) -> bool:
        """Remove a specific connection"""
        initial_count = len(self.connections)
        self.connections = [
            conn for conn in self.connections 
            if not (conn.source_id == source_id and conn.target_id == target_id)
        ]
        return len(self.connections) < initial_count
    
    def to_graph_representation(self) -> Dict:
        """Convert architecture to graph format for transformer"""
        node_features = []
        node_ids = sorted(self.neurons.keys())
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Node features
        for node_id in node_ids:
            node_features.append(self.neurons[node_id].to_feature_vector())
        
        # Edge indices (connections)
        edge_index = [[], []]
        edge_weights = []
        
        for conn in self.connections:
            if conn.enabled and conn.source_id in id_to_index and conn.target_id in id_to_index:
                edge_index[0].append(id_to_index[conn.source_id])
                edge_index[1].append(id_to_index[conn.target_id])
                edge_weights.append(conn.weight)
        
        return {
            'node_features': torch.FloatTensor(node_features),
            'edge_index': torch.LongTensor(edge_index),
            'edge_weights': torch.FloatTensor(edge_weights),
            'node_mapping': id_to_index,
            'performance': self.performance_metrics
        }
    
    def get_neuron_count(self) -> Dict[NeuronType, int]:
        """Count neurons by type"""
        counts = {neuron_type: 0 for neuron_type in NeuronType}
        for neuron in self.neurons.values():
            counts[neuron.neuron_type] += 1
        return counts
    
    def __str__(self):
        counts = self.get_neuron_count()
        return f"NeuralArchitecture(neurons={sum(counts.values())}, connections={len(self.connections)})"


class GraphNeuralNetwork(nn.Module):
    """
    A dynamic neural network that can be built from our graph architecture representation.
    This converts the graph structure into an executable PyTorch model.
    """
    
    def __init__(self, architecture: NeuralArchitecture):
        super().__init__()
        self.architecture = architecture
        self.layers = nn.ModuleDict()
        self.activation_functions = nn.ModuleDict()
        self.connection_matrices = {}
        
        self._build_network()
    
    def _build_network(self):
        """Convert graph architecture to executable layers"""
        graph_data = self.architecture.to_graph_representation()
        edge_index = graph_data['edge_index']
        edge_weights = graph_data['edge_weights']
        
        # Group neurons by approximate layer position
        neurons_by_layer = self._group_neurons_by_layer()
        
        # Build computational graph
        self._build_computational_graph(neurons_by_layer, edge_index, edge_weights)
    
    def _group_neurons_by_layer(self) -> Dict[float, List[Neuron]]:
        """Group neurons by their layer_position for sequential processing"""
        layers = {}
        tolerance = 0.1  # Group neurons within this position range
        
        for neuron in self.architecture.neurons.values():
            # Find existing layer group or create new one
            found_layer = False
            for layer_pos in layers.keys():
                if abs(neuron.layer_position - layer_pos) <= tolerance:
                    layers[layer_pos].append(neuron)
                    found_layer = True
                    break
            
            if not found_layer:
                layers[neuron.layer_position] = [neuron]
        
        # Sort layers by position
        return {pos: layers[pos] for pos in sorted(layers.keys())}
    
    def _build_computational_graph(self, neurons_by_layer, edge_index, edge_weights):
        """Build the actual computational graph from the architecture"""
        layer_positions = sorted(neurons_by_layer.keys())
        
        # Create mapping from neuron ID to layer index
        neuron_to_layer = {}
        for layer_idx, layer_pos in enumerate(layer_positions):
            for neuron in neurons_by_layer[layer_pos]:
                neuron_to_layer[neuron.id] = layer_idx
        
        # Build connection matrices between layers
        for i in range(len(layer_positions) - 1):
            current_layer_pos = layer_positions[i]
            next_layer_pos = layer_positions[i + 1]
            
            current_neurons = neurons_by_layer[current_layer_pos]
            next_neurons = neurons_by_layer[next_layer_pos]
            
            # Create weight matrix between current and next layer
            weight_matrix = self._build_weight_matrix(
                current_neurons, next_neurons, edge_index, edge_weights
            )
            
            layer_name = f"layer_{i}_{i+1}"
            self.layers[layer_name] = nn.Linear(len(current_neurons), len(next_neurons), bias=False)
            
            # Set the weights from our architecture
            with torch.no_grad():
                self.layers[layer_name].weight.data = weight_matrix
        
        # Store layer information for forward pass
        self.layer_neurons = [neurons_by_layer[pos] for pos in layer_positions]
        self.layer_activations = []
        
        # Set up activation functions for each layer
        for layer_idx, neurons in enumerate(self.layer_neurons):
            # For simplicity, use the most common activation in the layer
            activations = [neuron.activation for neuron in neurons]
            most_common_activation = max(set(activations), key=activations.count)
            
            activation_module = self._get_activation_module(most_common_activation)
            self.activation_functions[f"layer_{layer_idx}"] = activation_module
    
    def _build_weight_matrix(self, source_neurons, target_neurons, edge_index, edge_weights):
        """Build weight matrix between two layers based on connections"""
        source_ids = [neuron.id for neuron in source_neurons]
        target_ids = [neuron.id for neuron in target_neurons]
        
        # Create mapping from neuron ID to index
        source_id_to_idx = {id: idx for idx, id in enumerate(source_ids)}
        target_id_to_idx = {id: idx for idx, id in enumerate(target_ids)}
        
        # Initialize weight matrix
        weight_matrix = torch.zeros(len(target_neurons), len(source_neurons))
        
        # Fill weight matrix from connections
        for i in range(edge_index.shape[1]):
            source_id = edge_index[0, i].item()
            target_id = edge_index[1, i].item()
            weight = edge_weights[i].item()
            
            if source_id in source_id_to_idx and target_id in target_id_to_idx:
                source_idx = source_id_to_idx[source_id]
                target_idx = target_id_to_idx[target_id]
                weight_matrix[target_idx, source_idx] = weight
        
        return weight_matrix
    
    def _get_activation_module(self, activation_type: ActivationType) -> nn.Module:
        """Convert activation type to PyTorch module"""
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation_type == ActivationType.TANH:
            return nn.Tanh()
        elif activation_type == ActivationType.LINEAR:
            return nn.Identity()
        else:
            return nn.ReLU()  # Default
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the graph-based network"""
        # x shape: [batch_size, 784] for MNIST
        
        # Reshape if needed (for convolutional thinking)
        if x.dim() == 4:  # [batch, channels, height, width]
            x = x.view(x.size(0), -1)
        
        # Process through layers
        current_output = x
        
        for i in range(len(self.layer_neurons) - 1):
            layer_name = f"layer_{i}_{i+1}"
            activation_name = f"layer_{i+1}"  # Activation after the layer
            
            if layer_name in self.layers:
                # Linear transformation
                current_output = self.layers[layer_name](current_output)
                
                # Activation function
                if activation_name in self.activation_functions:
                    current_output = self.activation_functions[activation_name](current_output)
        
        # Final output should match MNIST classes (10)
        if current_output.shape[1] != 10:
            raise ValueError("Final output layer does not have 10 neurons for MNIST classification.")
        return current_output

    def get_parameter_count(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)