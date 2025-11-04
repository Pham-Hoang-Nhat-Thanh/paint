import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torch.amp import autocast

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
    layer_position: float
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
        # Initialize connection set for O(1) duplicate checking
        self._connection_set = set()
        
        # Input neurons (784)
        for i in range(784):
            self.add_neuron(NeuronType.INPUT, ActivationType.LINEAR, layer_position=0.0)

        # Output neurons (10)
        for i in range(10):
            self.add_neuron(NeuronType.OUTPUT, ActivationType.LINEAR, layer_position=1.0)

        # Add isolated hidden neuron to allow initial connections
        for i in range(100):
            layer_pos = torch.rand(1).item() * 0.8 + 0.1  # Random position between 0.1 and 0.9
            self.add_neuron(NeuronType.HIDDEN, ActivationType.RELU, layer_position=layer_pos)

        # Ensure each input has at least one outgoing connection (batch random generation)
        target_ids = torch.randint(784, 784 + 10 + 100, (784,))
        weights = torch.rand(784)
        for input_id in range(784):
            self.add_connection(input_id, target_ids[input_id].item(), weight=weights[input_id].item())
    
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
        
        # Use set for O(1) lookup instead of O(n) iteration
        if not hasattr(self, '_connection_set'):
            self._connection_set = {(conn.source_id, conn.target_id) for conn in self.connections}
        
        conn_key = (source_id, target_id)
        if conn_key in self._connection_set:
            return False
        
        self.connections.append(Connection(source_id, target_id, weight))
        self._connection_set.add(conn_key)
        return True
    
    def remove_neuron(self, neuron_id: int) -> bool:
        """Remove a neuron and all its connections"""
        if neuron_id not in self.neurons:
            return False
        
        neuron = self.neurons[neuron_id]
        if neuron.neuron_type in [NeuronType.INPUT, NeuronType.OUTPUT]:
            return False
        
        # Update connection set if it exists
        if hasattr(self, '_connection_set'):
            # Remove all connections involving this neuron from set
            self._connection_set = {
                (src, tgt) for src, tgt in self._connection_set 
                if src != neuron_id and tgt != neuron_id
            }
        
        # Remove all connections involving this neuron
        self.connections = [
            conn for conn in self.connections 
            if conn.source_id != neuron_id and conn.target_id != neuron_id
        ]
        
        del self.neurons[neuron_id]
        return True
    
    def remove_connection(self, source_id: int, target_id: int) -> bool:
        """Remove a specific connection"""
        conn_key = (source_id, target_id)
        
        # Update connection set if it exists
        if hasattr(self, '_connection_set') and conn_key in self._connection_set:
            self._connection_set.remove(conn_key)
        
        initial_count = len(self.connections)
        self.connections = [
            conn for conn in self.connections 
            if not (conn.source_id == source_id and conn.target_id == target_id)
        ]
        return len(self.connections) < initial_count
    
    def compute_signature(self) -> str:
        """Compute deterministic canonical signature of architecture for transposition detection.
        
        This signature is invariant to neuron ID renumbering but captures the graph structure,
        neuron types, activations, and layer positions. Two architectures with the same
        signature are considered identical states in MCTS.
        
        The signature format: <neuron_part>||<connection_part>
        - neuron_part: sorted list of (type, activation, layer_pos) tuples
        - connection_part: sorted list of (src_idx, tgt_idx) pairs where idx is position in sorted neuron list
        """
        # Sort neurons by (neuron_type, layer_position, activation) for canonical ordering
        # This creates a deterministic mapping independent of neuron IDs
        sorted_neurons = sorted(
            self.neurons.items(),
            key=lambda x: (x[1].neuron_type.value, x[1].layer_position, x[1].activation.value, x[0])
        )
        
        # Build neuron signature and ID mapping in single pass
        id_to_canonical_idx = {}
        neuron_sig_parts = [
            f"{neuron.neuron_type.name}:{neuron.activation.name}:{neuron.layer_position:.4f}"
            for canonical_idx, (neuron_id, neuron) in enumerate(sorted_neurons)
            if not (id_to_canonical_idx.__setitem__(neuron_id, canonical_idx))  # Side effect: populate mapping
        ]
        
        neuron_sig = "|".join(neuron_sig_parts)
        
        # Build connection signature using canonical indices - filter and map in one pass
        conn_sig_parts = [
            f"{id_to_canonical_idx[conn.source_id]}->{id_to_canonical_idx[conn.target_id]}"
            for conn in sorted(self.connections, key=lambda c: (c.source_id, c.target_id))
            if conn.enabled and conn.source_id in id_to_canonical_idx and conn.target_id in id_to_canonical_idx
        ]
        
        conn_sig = ",".join(conn_sig_parts)
        
        return f"{neuron_sig}||{conn_sig}"
    
    def to_graph_representation(self) -> Dict:
        """Convert architecture to graph format for transformer"""
        node_ids = sorted(self.neurons.keys())
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Vectorized node features using list comprehension
        node_features = [self.neurons[node_id].to_feature_vector() for node_id in node_ids]
        
        # Pre-filter connections and build edge data in single pass
        edge_data = [
            (id_to_index[conn.source_id], id_to_index[conn.target_id], conn.weight)
            for conn in self.connections
            if conn.enabled and conn.source_id in id_to_index and conn.target_id in id_to_index
        ]
        
        if edge_data:
            sources, targets, weights = zip(*edge_data)
            edge_index = torch.LongTensor([list(sources), list(targets)])
            edge_weights = torch.FloatTensor(weights)
        else:
            edge_index = torch.LongTensor([[], []])
            edge_weights = torch.FloatTensor([])
        
        return {
            'node_features': torch.FloatTensor(node_features),
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'node_mapping': id_to_index,
            'performance': self.performance_metrics
        }

    # --- Utility helpers for architecture statistics (used by MCTS reward shaping) ---
    def num_neurons(self) -> int:
        """Total number of neurons in the architecture."""
        return len(self.neurons)

    def num_connections(self) -> int:
        """Total number of connections (including disabled ones if present)."""
        return len(self.connections)

    def count_isolated_neurons(self, ignore_ids: set = None, only_hidden: bool = True) -> int:
        """Count neurons that have no incoming AND no outgoing enabled connections.

        ignore_ids: set of neuron ids to exclude from counting (useful for grace-period on newly
        added neurons).
        only_hidden: when True, only consider hidden neurons (don't penalize input/output neurons).
        """
        if ignore_ids is None:
            ignore_ids = set()

        # Initialize degree counters using defaultdict for cleaner code
        from collections import defaultdict
        incoming = defaultdict(int)
        outgoing = defaultdict(int)

        # Single pass through connections
        for conn in self.connections:
            if conn.enabled:
                outgoing[conn.source_id] += 1
                incoming[conn.target_id] += 1

        # Count isolated neurons
        isolated = 0
        for nid, neuron in self.neurons.items():
            if nid in ignore_ids:
                continue
            if only_hidden and neuron.neuron_type != NeuronType.HIDDEN:
                continue
            # Check if neuron has no connections (not in dicts or count is 0)
            if incoming[nid] == 0 or outgoing[nid] == 0:
                isolated += 1

        return isolated

    def connected_fraction(self, ignore_ids: set = None) -> float:
        """Return fraction of (hidden) neurons that are connected (1 - isolated_fraction).

        This normalizes by number of hidden neurons (so metric is stable across sizes).
        """
        if ignore_ids is None:
            ignore_ids = set()
        total_hidden = sum(1 for n in self.neurons.values() if n.neuron_type == NeuronType.HIDDEN and n.id not in ignore_ids)
        if total_hidden == 0:
            return 1.0
        isolated = self.count_isolated_neurons(ignore_ids=ignore_ids, only_hidden=True)
        return max(0.0, 1.0 - isolated / total_hidden)
    
    def get_neuron_count(self) -> Dict[NeuronType, int]:
        """Count neurons by type"""
        from collections import Counter
        counts = Counter(neuron.neuron_type for neuron in self.neurons.values())
        # Ensure all neuron types are present in result
        return {neuron_type: counts.get(neuron_type, 0) for neuron_type in NeuronType}
    
    def to_serializable_dict(self) -> dict:
        """Convert architecture to serializable dict for multiprocessing"""
        return {
            'next_neuron_id': self.next_neuron_id,
            'neurons': [
                {
                    'id': neuron.id,
                    'neuron_type': neuron.neuron_type.value,  # Convert enum to string
                    'activation': neuron.activation.value,    # Convert enum to string
                    'layer_position': neuron.layer_position,
                    'bias': neuron.bias
                }
                for neuron in self.neurons.values()
            ],
            'connections': [
                {
                    'source_id': conn.source_id,
                    'target_id': conn.target_id,
                    'weight': conn.weight,
                    'enabled': conn.enabled
                }
                for conn in self.connections
            ]
        }

    @classmethod
    def from_serializable_dict(cls, data: dict) -> 'NeuralArchitecture':
        """Reconstruct architecture from serializable dict"""
        arch = cls()
        arch.neurons = {}
        arch.connections = []
        arch.next_neuron_id = data['next_neuron_id']

        # Reconstruct neurons
        for neuron_data in data['neurons']:
            neuron = Neuron(
                id=neuron_data['id'],
                neuron_type=NeuronType(neuron_data['neuron_type']),
                activation=ActivationType(neuron_data['activation']),
                layer_position=neuron_data['layer_position'],
                bias=neuron_data['bias']
            )
            arch.neurons[neuron.id] = neuron

        # Reconstruct connections
        for conn_data in data['connections']:
            conn = Connection(
                source_id=conn_data['source_id'],
                target_id=conn_data['target_id'],
                weight=conn_data['weight'],
                enabled=conn_data['enabled']
            )
            arch.connections.append(conn)

        return arch

    def __str__(self):
        counts = self.get_neuron_count()
        return f"NeuralArchitecture(neurons={sum(counts.values())}, connections={len(self.connections)})"


class GraphNeuralNetwork(nn.Module):
    """
    Optimized Graph Neural Network that maintains the original interface while being much faster.
    Key improvements without breaking gradients:
    1. Batched weight matrices instead of individual connections
    2. Pre-computed layer connectivity for efficient forward pass
    3. Vectorized operations without inplace modifications
    4. Proper gradient flow through all operations
    """
    
    def __init__(self, architecture: NeuralArchitecture, device: Optional[str] = None):
        super().__init__()
        self.architecture = architecture
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Enable GPU optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # Use new API for TF32 precision
                torch.set_float32_matmul_precision('high')
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
                torch.backends.cuda.matmul.fp32_precision = 'tf32'

        # Build optimized network structure
        self._build_optimized_network()
    
    def _build_optimized_network(self):
        """Build optimized network structure with batched operations"""
        # Clear old parameter dictionaries to ensure clean rebuild
        self.weight_matrices = nn.ParameterDict()
        self.bias_vectors = nn.ParameterDict()
        self.weight_masks = {}  # Store masks as regular dict (not parameters)
        self.activation_modules = nn.ModuleDict()

        # Group neurons by layer position with tolerance
        self.layer_groups = self._group_neurons_by_layer()
        self.layer_positions = sorted(self.layer_groups.keys())
        
        # Precompute layer mappings and connectivity
        self._precompute_layer_mappings()
        
        # Build efficient weight matrices and biases
        self._build_efficient_parameters()
        
        # Precompute activation functions (without inplace operations)
        self._precompute_activations()
    
    def _group_neurons_by_layer(self) -> Dict[float, List[Neuron]]:
        """Optimized neuron grouping"""
        layers = {}
        tolerance = 0.1
        
        # Pre-sort neurons by position for better grouping
        sorted_neurons = sorted(self.architecture.neurons.values(), 
                              key=lambda n: n.layer_position)
        
        for neuron in sorted_neurons:
            found_layer = False
            for layer_pos in list(layers.keys()):
                if abs(neuron.layer_position - layer_pos) <= tolerance:
                    layers[layer_pos].append(neuron)
                    found_layer = True
                    break
            
            if not found_layer:
                layers[neuron.layer_position] = [neuron]
        
        return {pos: layers[pos] for pos in sorted(layers.keys())}
    
    def _precompute_layer_mappings(self):
        """Precompute efficient layer-to-layer mappings"""
        self.layer_indices = {}
        self.neuron_to_layer_idx = {}
        
        # Create mapping from neuron ID to layer index
        for layer_idx, layer_pos in enumerate(self.layer_positions):
            neurons = self.layer_groups[layer_pos]
            neuron_ids = [n.id for n in neurons]
            self.layer_indices[layer_idx] = neuron_ids
            
            for neuron_id in neuron_ids:
                self.neuron_to_layer_idx[neuron_id] = layer_idx
        
        # Precompute layer connectivity matrix
        self._build_connectivity_matrix()
    
    def _build_connectivity_matrix(self):
        """Build efficient connectivity representation between layers"""
        self.layer_connectivity = {}
        num_layers = len(self.layer_positions)
        
        # Initialize connectivity dictionary
        for i in range(num_layers):
            for j in range(i + 1, num_layers):  # Only forward connections
                self.layer_connectivity[(i, j)] = []
        
        # Populate connectivity
        for conn in self.architecture.connections:
            if conn.enabled:
                source_layer = self.neuron_to_layer_idx.get(conn.source_id)
                target_layer = self.neuron_to_layer_idx.get(conn.target_id)
                
                if source_layer is not None and target_layer is not None and source_layer < target_layer:
                    self.layer_connectivity[(source_layer, target_layer)].append(
                        (conn.source_id, conn.target_id, conn.weight)
                    )
    
    def _build_efficient_parameters(self):
        """Build efficient parameter tensors without breaking gradients"""
        # Build weight matrices for connected layers
        for (src_layer, tgt_layer), connections in self.layer_connectivity.items():
            if not connections:
                continue
                
            src_neurons = self.layer_indices[src_layer]
            tgt_neurons = self.layer_indices[tgt_layer]
            
            # Create mapping from neuron ID to index within layer
            src_id_to_idx = {nid: idx for idx, nid in enumerate(src_neurons)}
            tgt_id_to_idx = {nid: idx for idx, nid in enumerate(tgt_neurons)}

            # Initialize weight matrix with zeros (truly sparse)
            weight_matrix = torch.zeros(len(tgt_neurons), len(src_neurons))
            # Initialize mask matrix with zeros
            mask_matrix = torch.zeros(len(tgt_neurons), len(src_neurons), dtype=torch.bool)

            # Fill weight matrix and mask with actual connection weights
            for src_id, tgt_id, weight in connections:
                if src_id in src_id_to_idx and tgt_id in tgt_id_to_idx:
                    src_idx = src_id_to_idx[src_id]
                    tgt_idx = tgt_id_to_idx[tgt_id]
                    weight_matrix[tgt_idx, src_idx] = weight
                    mask_matrix[tgt_idx, src_idx] = True

            # Store as parameter and mask
            key = f"weights_{src_layer}_{tgt_layer}"
            param = nn.Parameter(weight_matrix.to(self.device))
            # Register hook to freeze irrelevant weights (zero gradients for non-connected positions)
            def mask_hook(grad, mask=mask_matrix.to(self.device)):
                return grad * mask.float()
            param.register_hook(mask_hook)
            self.weight_matrices[key] = param
            self.weight_masks[key] = mask_matrix.to(self.device)

        # Build bias vectors for each layer
        for layer_idx, neuron_ids in self.layer_indices.items():
            biases = torch.zeros(len(neuron_ids))
            for i, neuron_id in enumerate(neuron_ids):
                neuron = self.architecture.neurons[neuron_id]
                biases[i] = neuron.bias
            
            key = f"biases_{layer_idx}"
            self.bias_vectors[key] = nn.Parameter(biases.to(self.device))
    
    def _precompute_activations(self):
        """Precompute activation functions without inplace operations"""
        self.activation_modules = nn.ModuleDict()
        
        for layer_idx, neuron_ids in self.layer_indices.items():
            if not neuron_ids:
                continue
                
            # Use most common activation in layer
            activations = [self.architecture.neurons[nid].activation for nid in neuron_ids]
            most_common = max(set(activations), key=activations.count)
            
            # NO inplace operations to avoid gradient issues
            self.activation_modules[str(layer_idx)] = self._get_activation_module(most_common)
    
    def _get_activation_module(self, activation_type: ActivationType) -> nn.Module:
        """Convert activation type to PyTorch module - NO inplace operations"""
        if activation_type == ActivationType.RELU:
            return nn.ReLU()  # Removed inplace=True
        elif activation_type == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation_type == ActivationType.TANH:
            return nn.Tanh()
        elif activation_type == ActivationType.LINEAR:
            return nn.Identity()
        else:
            return nn.ReLU()  # Removed inplace=True
    
    def forward(self, x: torch.Tensor, use_mixed_precision: bool = False) -> torch.Tensor:
        """
        Optimized forward pass with batched operations and proper gradient flow
        x: [batch_size, 784] for MNIST
        use_mixed_precision: Enable automatic mixed precision for faster computation
        """
        # Ensure input is on correct device
        x = x.to(self.device)

        batch_size = x.size(0)
        
        # Reshape input if needed
        if x.dim() == 4:
            x = x.view(batch_size, -1)
        
        # Initialize layer outputs dictionary
        layer_outputs = {}
        
        # Process input layer
        input_layer_idx = 0
        input_neuron_ids = self.layer_indices[input_layer_idx]
        
        if len(input_neuron_ids) != 784:
            # Handle case where input layer doesn't match expected size
            # This can happen during architecture search
            if len(input_neuron_ids) > 784:
                # Too many input neurons, take first 784
                input_data = x
            else:
                # Too few input neurons, pad with zeros
                padding = torch.zeros(batch_size, 784 - len(input_neuron_ids), device=self.device)
                input_data = torch.cat([x[:, :len(input_neuron_ids)], padding], dim=1)
        else:
            input_data = x
        
        layer_outputs[input_layer_idx] = input_data
        
        # Process subsequent layers in order
        for current_layer_idx in range(1, len(self.layer_positions)):
            current_output = None
            
            # Aggregate inputs from all previous layers
            for prev_layer_idx in range(current_layer_idx):
                weight_key = f"weights_{prev_layer_idx}_{current_layer_idx}"
                
                if weight_key in self.weight_matrices:
                    prev_output = layer_outputs[prev_layer_idx]
                    weight_matrix = self.weight_matrices[weight_key]
                    mask = self.weight_masks[weight_key]

                    # Apply mask to weight matrix (zero out non-connected weights, freezing them during training)
                    masked_weight_matrix = weight_matrix * mask.float()

                    # Use autocast for mixed precision if enabled
                    if use_mixed_precision:
                        with autocast('cuda'):
                            # Batched matrix multiplication (gradient-safe)
                            layer_input = torch.matmul(prev_output, masked_weight_matrix.t())
                    else:
                        # Batched matrix multiplication (gradient-safe)
                        layer_input = torch.matmul(prev_output, masked_weight_matrix.t())

                    if current_output is None:
                        current_output = layer_input
                    else:
                        current_output = current_output + layer_input  # Explicit operation
            
            # Add bias if we have any inputs
            if current_output is not None:
                bias_key = f"biases_{current_layer_idx}"
                if bias_key in self.bias_vectors:
                    current_output = current_output + self.bias_vectors[bias_key].unsqueeze(0)
                
                # Apply activation (gradient-safe, no inplace)
                activation_key = str(current_layer_idx)
                if activation_key in self.activation_modules:
                    if use_mixed_precision:
                        with autocast('cuda'):
                            current_output = self.activation_modules[activation_key](current_output)
                    else:
                        current_output = self.activation_modules[activation_key](current_output)

                layer_outputs[current_layer_idx] = current_output
            else:
                # If no connections to this layer, initialize with zeros
                num_neurons = len(self.layer_indices[current_layer_idx])
                layer_outputs[current_layer_idx] = torch.zeros(batch_size, num_neurons,
                                                             device=self.device)

        # Extract output from last layer
        output_layer_idx = len(self.layer_positions) - 1
        final_output = layer_outputs[output_layer_idx]
        
        # Ensure we have exactly 10 outputs for MNIST
        if final_output.shape[1] > 10:
            final_output = final_output[:, :10]  # Take first 10 neurons
        elif final_output.shape[1] < 10:
            # Pad with zeros if needed (gradient-safe)
            padding = torch.zeros(batch_size, 10 - final_output.shape[1],
                                device=self.device)
            final_output = torch.cat([final_output, padding], dim=1)
        
        return final_output
    
    def get_parameter_count(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.jit.export
    def jit_forward(self, x: torch.Tensor) -> torch.Tensor:
        """JIT-compiled forward pass for maximum performance"""
        return self.forward(x, use_mixed_precision=False)

    def enable_jit_compilation(self):
        """Compile the forward method with TorchScript for optimization"""
        if not hasattr(self, '_jit_compiled'):
            try:
                self._jit_compiled = torch.jit.script(self.jit_forward)
                print("JIT compilation successful")
            except Exception as e:
                print(f"JIT compilation failed: {e}")
                self._jit_compiled = None

    def forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        """Use JIT-compiled forward pass if available"""
        if hasattr(self, '_jit_compiled') and self._jit_compiled is not None:
            return self._jit_compiled(x)
        else:
            return self.forward(x)

    def update_architecture(self, new_architecture: NeuralArchitecture):
        """Rebuild the network structure for a modified architecture"""
        # Save current learned parameters to preserve training progress
        old_weight_matrices = dict(self.weight_matrices)
        old_bias_vectors = dict(self.bias_vectors)
        old_weight_masks = dict(self.weight_masks)

        self.architecture = new_architecture
        # Clear any cached JIT compilation to prevent OOM from accumulated compiled models
        if hasattr(self, '_jit_compiled'):
            self._jit_compiled = None
            # Force garbage collection to free memory immediately
            import gc
            gc.collect()

        # Rebuild the entire network structure
        self._build_optimized_network()

        # Copy back learned weights for existing parameter keys to preserve training
        for key, new_matrix in self.weight_matrices.items():
            if key in old_weight_matrices:
                old_matrix = old_weight_matrices[key]
                old_mask = old_weight_masks[key]
                new_mask = self.weight_masks[key]

                # Only copy weights if shapes match (architecture changes may alter matrix sizes)
                if old_matrix.shape == new_matrix.shape:
                    # Only copy weights where both old and new masks allow connections
                    combined_mask = old_mask & new_mask
                    new_matrix.data[combined_mask] = old_matrix.data[combined_mask]

        for key, new_bias in self.bias_vectors.items():
            if key in old_bias_vectors:
                old_bias = old_bias_vectors[key]
                # Copy overlapping elements from old learned biases
                min_len = min(len(old_bias), len(new_bias))
                new_bias.data[:min_len] = old_bias.data[:min_len]
