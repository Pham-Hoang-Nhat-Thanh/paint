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

        # Add initial random connections to enable basic functionality
        # Connect each output to a random subset of inputs
        import random
        random.seed(42)  # For reproducibility
        input_ids = list(range(784))
        for output_id in range(784, 794):  # Output neurons are 784-793
            # Connect to ~10% of inputs randomly
            num_connections = max(1, 784 // 10)  # At least 1 connection per output
            selected_inputs = random.sample(input_ids, num_connections)
            for input_id in selected_inputs:
                weight = random.uniform(-0.1, 0.1)  # Small random weights
                self.add_connection(input_id, output_id, weight)
    
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
            neuron_obj = self.neurons[node_id]

            # If the neuron already provides a feature conversion, use it.
            if hasattr(neuron_obj, 'to_feature_vector') and callable(getattr(neuron_obj, 'to_feature_vector')):
                node_features.append(neuron_obj.to_feature_vector())
                continue

            # Fallback: the neuron may be a plain dict (e.g. after deserialization).
            # Try to create a temporary Neuron instance to reuse the canonical conversion.
            try:
                # Extract fields with some tolerance for strings/enums
                n_type = neuron_obj.get('neuron_type') if isinstance(neuron_obj, dict) else None
                activation = neuron_obj.get('activation') if isinstance(neuron_obj, dict) else None
                layer_pos = neuron_obj.get('layer_position', 0.0) if isinstance(neuron_obj, dict) else getattr(neuron_obj, 'layer_position', 0.0)
                bias = neuron_obj.get('bias', 0.0) if isinstance(neuron_obj, dict) else getattr(neuron_obj, 'bias', 0.0)

                # Normalize enum inputs (accept enum members or strings)
                if isinstance(n_type, str):
                    try:
                        n_type_enum = NeuronType(n_type)
                    except Exception:
                        n_type_enum = NeuronType[n_type]
                elif isinstance(n_type, NeuronType):
                    n_type_enum = n_type
                else:
                    # Last resort: assume INPUT for unknown
                    n_type_enum = NeuronType.INPUT

                if isinstance(activation, str):
                    try:
                        activation_enum = ActivationType(activation)
                    except Exception:
                        activation_enum = ActivationType[activation]
                elif isinstance(activation, ActivationType):
                    activation_enum = activation
                else:
                    activation_enum = ActivationType.LINEAR

                temp_neuron = Neuron(node_id, n_type_enum, activation_enum, layer_pos, bias)
                node_features.append(temp_neuron.to_feature_vector())
            except Exception:
                # As a last fallback, create a minimal numeric vector [0]*node_feature_length-ish
                # Keep length 9 to match expected encoder (type(3)+activation(4)+pos+bias)
                node_features.append([0.0] * 9)
        
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
    This implements proper sparse message passing over the graph connections.
    """

    def __init__(self, architecture: NeuralArchitecture):
        super().__init__()
        self.architecture = architecture

        # Build sparse adjacency structure
        self._build_sparse_network()

    def _build_sparse_network(self):
        """Build sparse adjacency lists and layer structure"""
        # Group neurons by layer position
        self.neurons_by_layer = self._group_neurons_by_layer()
        self.layer_positions = sorted(self.neurons_by_layer.keys())

        # Create neuron ID to layer index mapping
        self.neuron_to_layer = {}
        for layer_idx, layer_pos in enumerate(self.layer_positions):
            for neuron in self.neurons_by_layer[layer_pos]:
                self.neuron_to_layer[neuron.id] = layer_idx

        # Build adjacency lists: target_id -> [(source_id, weight), ...]
        self.adjacency = {neuron_id: [] for neuron_id in self.architecture.neurons.keys()}
        for conn in self.architecture.connections:
            if conn.enabled:
                self.adjacency[conn.target_id].append((conn.source_id, conn.weight))

        # Set up activation functions per layer
        self.activation_functions = {}
        for layer_idx, neurons in enumerate(self.neurons_by_layer.values()):
            # Use most common activation in layer
            activations = [neuron.activation for neuron in neurons]
            most_common_activation = max(set(activations), key=activations.count)
            self.activation_functions[layer_idx] = self._get_activation_module(most_common_activation)

        # Create neuron index mappings within layers for output extraction
        self.layer_neuron_indices = {}
        for layer_pos, neurons in self.neurons_by_layer.items():
            layer_idx = self.layer_positions.index(layer_pos)
            neuron_ids = [n.id for n in neurons]
            self.layer_neuron_indices[layer_idx] = neuron_ids

        # Create trainable parameters for connection weights and neuron biases
        self.connection_weights = nn.ParameterDict()
        self.neuron_biases = nn.ParameterDict()

        # Initialize connection weights as trainable parameters
        for conn in self.architecture.connections:
            if conn.enabled:
                weight_param = nn.Parameter(torch.tensor(conn.weight, dtype=torch.float32))
                self.connection_weights[str(conn.source_id) + '_' + str(conn.target_id)] = weight_param

        # Initialize neuron biases as trainable parameters
        for neuron in self.architecture.neurons.values():
            bias_param = nn.Parameter(torch.tensor(neuron.bias, dtype=torch.float32))
            self.neuron_biases[str(neuron.id)] = bias_param

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
        """Forward pass with sparse message passing"""
        batch_size = x.size(0)

        # Initialize neuron outputs
        neuron_outputs = {}

        # Set input layer outputs
        input_layer_pos = self.layer_positions[0]
        input_neurons = self.neurons_by_layer[input_layer_pos]

        for i, neuron in enumerate(input_neurons):
            # Input neurons get flattened pixel values
            if x.dim() == 4:  # [batch, channels, height, width]
                x = x.view(batch_size, -1)
            neuron_outputs[neuron.id] = x[:, i:i+1]  # Shape: [batch_size, 1]

        # Process through layers sequentially
        for layer_idx in range(1, len(self.layer_positions)):
            current_layer_pos = self.layer_positions[layer_idx]
            current_neurons = self.neurons_by_layer[current_layer_pos]

            # Compute outputs for current layer neurons
            for neuron in current_neurons:
                # Aggregate weighted inputs from connected source neurons
                total_input = torch.zeros(batch_size, 1, device=x.device)

                for source_id, _ in self.adjacency[neuron.id]:
                    if source_id in neuron_outputs:
                        source_output = neuron_outputs[source_id]  # [batch_size, 1]
                        weight_key = str(source_id) + '_' + str(neuron.id)
                        weight = self.connection_weights[weight_key]
                        total_input += weight * source_output

                # Add neuron bias
                bias = self.neuron_biases[str(neuron.id)]
                total_input += bias

                # Apply activation
                activated_output = self.activation_functions[layer_idx](total_input)
                neuron_outputs[neuron.id] = activated_output

        # Collect output layer activations
        output_layer_pos = self.layer_positions[-1]
        output_neurons = self.neurons_by_layer[output_layer_pos]
        output_list = []

        for neuron in output_neurons:
            output_list.append(neuron_outputs[neuron.id])

        # Concatenate outputs: [batch_size, num_output_neurons]
        final_output = torch.cat(output_list, dim=1)

        # Ensure we have 10 outputs for MNIST
        if final_output.shape[1] != 10:
            raise ValueError(f"Output layer should have 10 neurons for MNIST, got {final_output.shape[1]}")

        return final_output

    def get_parameter_count(self) -> int:
        """Count trainable parameters"""
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params
