import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torch.amp import autocast
import random

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
        
        # Cache for optimization - invalidate when structure changes
        self._sorted_neuron_ids = None
        self._connectivity_cache = None  # {'has_incoming': set, 'has_outgoing': set}
        self._topological_layers_cache = None  # {neuron_id: layer}
        
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

        # Add 10 hidden neurons and place them at the center (0.5)
        # Using 0.5 makes them match the default isolated position and group
        # them into the middle layer for the GNN without adding connections yet.
        num_hidden = 10
        for _ in range(num_hidden):
            # Default activation for initial hidden neurons: RELU
            self.add_neuron(NeuronType.HIDDEN, ActivationType.RELU)

        # Ensure all input neurons are connected to at least one hidden neuron
        input_ids = [nid for nid, neuron in self.neurons.items() if neuron.neuron_type == NeuronType.INPUT]
        hidden_ids = [nid for nid, neuron in self.neurons.items() if neuron.neuron_type == NeuronType.HIDDEN]
        for input_id in input_ids:
            # Connect each input neuron to a random hidden neuron
            target_hidden_id = random.choice(hidden_ids)
            self.add_connection(input_id, target_hidden_id, weight=0.1)
        
    def add_neuron(self, neuron_type: NeuronType, activation: ActivationType, layer_position: Optional[float] = None) -> int:
        """Add a new neuron and return its ID.

        If `layer_position` is None the neuron is inserted with a safe placeholder
        position and the topology is recomputed to derive a proper normalized
        float position for all neurons (via `sync_layer_positions_from_topology`).

        This ensures callers can create neurons without explicitly choosing a
        float position and still obtain consistent positional encodings.
        """
        neuron_id = self.next_neuron_id
        # Use a sensible placeholder while the neuron exists in the graph so
        # topology computations can include it. For isolates this matches the
        # default used by sync_layer_positions_from_topology().
        placeholder_pos = 0.5 if layer_position is None else layer_position
        self.neurons[neuron_id] = Neuron(neuron_id, neuron_type, activation, placeholder_pos)
        self.next_neuron_id += 1

        # Invalidate caches (topology changed)
        self._sorted_neuron_ids = None
        self._connectivity_cache = None
        # Invalidate topological layers cache so next computation is fresh
        self._topological_layers_cache = None

        # If caller didn't provide a float position, compute it from topology
        # so integer topological layers are mapped to normalized floats.
        if layer_position is None:
            try:
                # This will call compute_topological_layers() internally and
                # rewrite all neuron.layer_position values accordingly.
                self.sync_layer_positions_from_topology()
            except Exception:
                # Keep placeholder if sync fails for any reason; caller can
                # call sync_layer_positions_from_topology() later.
                pass

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
        # Invalidate connectivity cache (but keep topological for local update)
        self._connectivity_cache = None
        # Locally update both source and target: 
        # - target may move from isolated to connected (gained incoming)
        # - source may move from isolated to connected (has outgoing now)
        if hasattr(self, '_topological_layers_cache') and self._topological_layers_cache is not None:
            self.recalculate_affected_layers({source_id, target_id})
        # Synchronize layer_position floats with updated topology
        self.sync_layer_positions_from_topology()
        return True
    
    def remove_neuron(self, neuron_id: int) -> bool:
        """Remove a neuron and all its connections"""
        if neuron_id not in self.neurons:
            return False
        
        neuron = self.neurons[neuron_id]
        if neuron.neuron_type in [NeuronType.INPUT, NeuronType.OUTPUT]:
            return False
        
        # Collect neurons affected by this removal (both sources and targets)
        affected_neurons = set()
        for conn in self.connections:
            if conn.source_id == neuron_id:
                affected_neurons.add(conn.target_id)  # Targets lose incoming
            elif conn.target_id == neuron_id:
                affected_neurons.add(conn.source_id)  # Sources lose outgoing
        
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
        # Invalidate caches
        self._sorted_neuron_ids = None
        self._connectivity_cache = None
        # Invalidate topological layers cache (structure changed)
        self._topological_layers_cache = None
        # Synchronize layer_position floats with updated topology
        self.sync_layer_positions_from_topology()
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
        # Invalidate caches when connection removed
        if len(self.connections) < initial_count:
            self._connectivity_cache = None
            # Invalidate topological layers cache - connection removal affects layer assignments
            # Safer to force full recomputation than risk stale local updates
            if hasattr(self, '_topological_layers_cache'):
                self._topological_layers_cache = None
            # Synchronize layer_position floats with updated topology
            self.sync_layer_positions_from_topology()
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
        node_ids = self._get_sorted_neuron_ids()
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        # Local references for speed
        neurons = self.neurons
        connections = self.connections

        # Build node features as a single tensor using as_tensor (avoids an extra copy when possible)
        if node_ids:
            # to_feature_vector returns a small python list per neuron; as_tensor will create one FloatTensor
            node_features = torch.as_tensor(
                [neurons[nid].to_feature_vector() for nid in node_ids],
                dtype=torch.float
            )
        else:
            # Feature length: 3 (type) + 4 (activation) + 2 (layer_pos,bias) = 9
            node_features = torch.empty((0, 9), dtype=torch.float)

        # Efficiently build edge index arrays without creating an intermediate big list of tuples
        src_list = []
        tgt_list = []
        weight_list = []
        id_to_idx_get = id_to_index.get
        for conn in connections:
            if not conn.enabled:
                continue
            s = id_to_idx_get(conn.source_id)
            t = id_to_idx_get(conn.target_id)
            if s is None or t is None:
                continue
            src_list.append(s)
            tgt_list.append(t)
            weight_list.append(conn.weight)

        if src_list:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            edge_weights = torch.as_tensor(weight_list, dtype=torch.float)
        else:
            # Use empty tensors with correct shapes to avoid resizing later
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty((0,), dtype=torch.float)

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'node_mapping': id_to_index,
            'performance': self.performance_metrics,
            'sorted_neuron_ids': node_ids  # Return sorted IDs for reuse
        }
    
    def _get_sorted_neuron_ids(self) -> List[int]:
        """Get sorted neuron IDs (cached) - O(1) after first call, O(n log n) on first call"""
        if self._sorted_neuron_ids is None:
            self._sorted_neuron_ids = sorted(self.neurons.keys())
        return self._sorted_neuron_ids
    
    def _get_connectivity_sets(self) -> Dict[str, set]:
        """Get connectivity sets (has_incoming, has_outgoing) - cached and reused"""
        if self._connectivity_cache is None:
            has_incoming = set()
            has_outgoing = set()
            for conn in self.connections:
                has_incoming.add(conn.target_id)
                has_outgoing.add(conn.source_id)
            self._connectivity_cache = {'has_incoming': has_incoming, 'has_outgoing': has_outgoing}
        return self._connectivity_cache
    
    def compute_topological_layers(self) -> Dict[int, int]:
        """Compute deterministic topological layers based on graph connectivity.
        
        Layer assignment:
        - Input neurons: layer 0 (no incoming connections)
        - Layer k (k > 0): neurons where max(layer_indices of incoming neurons) == k-1
        - Isolated HIDDEN neurons: layer -1 (no incoming OR no outgoing connections, but only for hidden neurons)
        - Output neurons: automatically assigned highest layer among all neurons
        
        Returns:
            Dict[neuron_id, layer] where layer is int (-1 for isolated hidden neurons, 0+ for connected)
        """
        # Initialize cache attribute if it doesn't exist (for backward compatibility with deserialized objects)
        if not hasattr(self, '_topological_layers_cache'):
            self._topological_layers_cache = None
        
        # Return cached result if available
        if self._topological_layers_cache is not None:
            return self._topological_layers_cache
        
        # Build connectivity tracking
        incoming_neighbors = {nid: [] for nid in self.neurons}
        outgoing_neighbors = {nid: [] for nid in self.neurons}
        
        for conn in self.connections:
            if conn.enabled:
                incoming_neighbors[conn.target_id].append(conn.source_id)
                outgoing_neighbors[conn.source_id].append(conn.target_id)
        
        # Find isolated HIDDEN neurons (missing incoming OR outgoing, but only hidden neurons)
        isolated_neurons = set()
        for nid, neuron in self.neurons.items():
            if neuron.neuron_type == NeuronType.HIDDEN:
                if len(incoming_neighbors[nid]) == 0 or len(outgoing_neighbors[nid]) == 0:
                    isolated_neurons.add(nid)
        
        # Assign layer -1 to isolated hidden neurons
        layers = {nid: -1 for nid in isolated_neurons}
        
        # For non-isolated neurons, compute layers iteratively
        non_isolated = {nid: neuron for nid, neuron in self.neurons.items() if nid not in isolated_neurons}
        
        # Input neurons (no incoming connections)
        for nid in non_isolated:
            if len(incoming_neighbors[nid]) == 0:
                layers[nid] = 0
        
        # Iteratively assign layers based on max layer of incoming neighbors
        max_iterations = len(non_isolated) + 1  # Prevent infinite loops
        for iteration in range(max_iterations):
            prev_assigned = len([nid for nid in non_isolated if nid in layers and layers[nid] >= 0])
            
            for nid in non_isolated:
                if nid not in layers or layers[nid] < 0:
                    # Check if all incoming neighbors have assigned layers (including isolated neurons with layer -1)
                    if len(incoming_neighbors[nid]) == 0:
                        # No incoming; should not happen in non_isolated, but handle gracefully
                        layers[nid] = 0
                    else:
                        # All incoming neighbors must have assigned layers (may include -1 for isolated)
                        incoming_layers = [layers[src] for src in incoming_neighbors[nid] if src in layers]
                        
                        if len(incoming_layers) == len(incoming_neighbors[nid]):
                            # All incoming neighbors assigned; compute max layer + 1
                            # Note: if all incomings are isolated (layer -1), max is -1, so layer becomes 0
                            max_incoming_layer = max(incoming_layers)
                            layers[nid] = max_incoming_layer + 1
            
            curr_assigned = len([nid for nid in non_isolated if nid in layers and layers[nid] >= 0])
            if curr_assigned == prev_assigned:
                # No progress; stop
                break
        
        # Cache the result
        self._topological_layers_cache = layers
        return layers
    
    def recalculate_affected_layers(self, affected_neuron_ids: set) -> Dict[int, int]:
        """Efficiently recalculate layers for affected neurons only (local update).
        
        This is more efficient than full recalculation when only a few neurons are changed.
        Used after add_connection, remove_connection, or remove_neuron to update affected neurons.
        
        Args:
            affected_neuron_ids: set of neuron IDs whose layers may have changed
            
        Returns:
            Updated layers dict (full state)
        """
        # Initialize cache attribute if it doesn't exist (for backward compatibility)
        if not hasattr(self, '_topological_layers_cache'):
            self._topological_layers_cache = None
        
        # Start with current cached layers (don't call compute_topological_layers to avoid full recomputation)
        if self._topological_layers_cache is None:
            # If not cached, do full computation
            return self.compute_topological_layers()
        
        layers = self._topological_layers_cache
        
        # Build connectivity tracking
        incoming_neighbors = {nid: [] for nid in self.neurons}
        outgoing_neighbors = {nid: [] for nid in self.neurons}
        
        for conn in self.connections:
            if conn.enabled:
                incoming_neighbors[conn.target_id].append(conn.source_id)
                outgoing_neighbors[conn.source_id].append(conn.target_id)
        
        # Recompute layers for affected neurons and their downstream dependents
        # Propagate changes forward: if a neuron's layer changes, its targets may need recalculation
        to_process = set(affected_neuron_ids)
        processed = set()
        
        max_iterations = len(self.neurons) + 1
        for iteration in range(max_iterations):
            if not to_process:
                break
            
            next_to_process = set()
            
            for nid in to_process:
                if nid in processed:
                    continue
                
                old_layer = layers.get(nid, -1)
                neuron = self.neurons[nid]
                
                # Recompute layer for this neuron
                # Only HIDDEN neurons can be isolated (layer -1)
                if neuron.neuron_type == NeuronType.HIDDEN and (len(incoming_neighbors[nid]) == 0 or len(outgoing_neighbors[nid]) == 0):
                    # Isolated hidden neuron
                    new_layer = -1
                elif len(incoming_neighbors[nid]) == 0:
                    # No incoming connections: this is an input neuron
                    new_layer = 0
                else:
                    # Non-isolated: compute max of incoming layers + 1
                    incoming_layers = [layers.get(src, -1) for src in incoming_neighbors[nid]]
                    max_in = max(incoming_layers) if incoming_layers else -1
                    
                    if max_in >= 0:
                        new_layer = max_in + 1
                    else:
                        # Unresolved dependencies; keep old layer for now
                        new_layer = old_layer
                
                layers[nid] = new_layer
                
                # If layer changed, mark downstream neurons for recomputation
                if new_layer != old_layer:
                    next_to_process.update(outgoing_neighbors[nid])
                
                processed.add(nid)
            
            to_process = next_to_process - processed
        
        # Update cache with new layers
        self._topological_layers_cache = layers
        return layers
    
    def sync_layer_positions_from_topology(self, center_for_isolated: float = 0.5) -> None:
        """Synchronize neuron layer_position floats (0..1) with computed topological integer layers.
        
        This maps topological layers to normalized float positions:
        - INPUT neurons: layer 0 -> position 0.0 (FIXED, never changed)
        - OUTPUT neurons: highest layer -> position 1.0 (FIXED, never changed)
        - HIDDEN neurons: intermediate layers mapped proportionally to (0, 1)
        - Isolated HIDDEN neurons (layer -1): fixed at center_for_isolated (default 0.5)
        
        Call this after structural changes (add/remove connections, remove neurons) to keep
        layer_position consistent with the graph topology. Without this, layer_position can
        drift out of sync with the actual computed layers, breaking positional encodings and
        layout computations.
        
        Args:
            center_for_isolated: float in [0, 1], position to assign isolated hidden neurons (default 0.5)
        """
        # Get current topological layer assignments
        layers = self.compute_topological_layers()

        # Quickly determine min/max layer for HIDDEN neurons only (excluding isolates == -1 and fixed I/O neurons)
        min_hidden_layer = None
        max_hidden_layer = None
        for neuron_id, layer in layers.items():
            neuron = self.neurons[neuron_id]
            # Skip INPUT (fixed at 0.0) and OUTPUT (fixed at 1.0) neurons
            if neuron.neuron_type in (NeuronType.INPUT, NeuronType.OUTPUT):
                continue
            # Skip isolated hidden neurons (layer -1)
            if layer == -1:
                continue
            if min_hidden_layer is None or layer < min_hidden_layer:
                min_hidden_layer = layer
            if max_hidden_layer is None or layer > max_hidden_layer:
                max_hidden_layer = layer

        # Assign positions in a single pass over neurons
        for neuron_id, neuron in self.neurons.items():
            # INPUT neurons: always position 0.0
            if neuron.neuron_type == NeuronType.INPUT:
                neuron.layer_position = 0.0
            # OUTPUT neurons: always position 1.0
            elif neuron.neuron_type == NeuronType.OUTPUT:
                neuron.layer_position = 1.0
            # HIDDEN neurons: map topological layer to (0, 1)
            else:
                layer = layers.get(neuron_id, -1)
                if layer == -1:
                    # Isolated hidden neuron
                    neuron.layer_position = center_for_isolated
                elif min_hidden_layer is None or max_hidden_layer is None:
                    # No connected hidden neurons (edge case); use center
                    neuron.layer_position = center_for_isolated
                else:
                    # Map hidden layer to (0, 1) proportionally
                    hidden_range = max(1, max_hidden_layer - min_hidden_layer)
                    neuron.layer_position = float((layer - min_hidden_layer) / hidden_range)
    
    def get_layer_for_neuron(self, neuron_id: int) -> int:
        """Get topological layer for a single neuron. Returns -1 if isolated or not found."""
        layers = self.compute_topological_layers()
        return layers.get(neuron_id, -1)

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
        arch = cls.__new__(cls)  # Create instance without calling __init__
        arch.neurons = {}
        arch.connections = []
        arch.next_neuron_id = data['next_neuron_id']
        arch.performance_metrics = {}
        
        # Initialize caches
        arch._sorted_neuron_ids = None
        arch._connectivity_cache = None
        arch._topological_layers_cache = None
        arch._connection_set = set()

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
            arch._connection_set.add((conn.source_id, conn.target_id))

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

        # Build bias vectors for each layer (including disconnected layers)
        for layer_idx, neuron_ids in self.layer_indices.items():
            biases = torch.zeros(len(neuron_ids))
            for i, neuron_id in enumerate(neuron_ids):
                neuron = self.architecture.neurons[neuron_id]
                biases[i] = neuron.bias
            
            key = f"biases_{layer_idx}"
            self.bias_vectors[key] = nn.Parameter(biases.to(self.device))
    
    def _precompute_activations(self):
        """Precompute activation functions without inplace operations"""
        # Build a cache of activation modules (ActivationType -> nn.Module)
        self.activation_module_cache = {}
        for act in ActivationType:
            self.activation_module_cache[act.name] = self._get_activation_module(act).to(self.device)

        # For each layer, build boolean masks that indicate which neurons use which activation
        # This allows elementwise (per-neuron) activation application in the forward pass.
        self.activation_masks = {}
        for layer_idx, neuron_ids in self.layer_indices.items():
            if not neuron_ids:
                continue

            activations = [self.architecture.neurons[nid].activation for nid in neuron_ids]
            layer_masks = {}
            for act in set(activations):
                # boolean mask over columns in this layer's output
                mask = torch.tensor([1 if a == act else 0 for a in activations], dtype=torch.bool, device=self.device)
                layer_masks[act.name] = mask

            self.activation_masks[str(layer_idx)] = layer_masks
    
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
        input_data = x[:, :len(input_neuron_ids)]  # Assume input layer matches first N neurons
        
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
                # Apply per-neuron activations by using precomputed masks and cached modules.
                if hasattr(self, 'activation_masks') and activation_key in self.activation_masks:
                    layer_act_masks = self.activation_masks[activation_key]
                    # Create a new tensor for activated output to avoid in-place ops
                    activated_output = current_output.new_zeros(current_output.shape)
                    for act_name, mask in layer_act_masks.items():
                        if mask.any():
                            act_module = self.activation_module_cache[act_name]
                            if use_mixed_precision:
                                with autocast('cuda'):
                                    activated_output[:, mask] = act_module(current_output[:, mask])
                            else:
                                activated_output[:, mask] = act_module(current_output[:, mask])
                    current_output = activated_output
                else:
                    # Fallback: no per-neuron masks available (shouldn't happen) — apply identity
                    current_output = current_output

                layer_outputs[current_layer_idx] = current_output
            else:
                # If no connections to this layer, use learnable bias vector to maintain gradient flow
                # CRITICAL: Must use actual parameters to keep computation graph connected
                bias_key = f"biases_{current_layer_idx}"
                if bias_key in self.bias_vectors:
                    num_neurons = len(self.layer_indices[current_layer_idx])
                    zeros = torch.zeros(batch_size, num_neurons, device=self.device)
                    layer_outputs[current_layer_idx] = zeros + self.bias_vectors[bias_key].unsqueeze(0)
                else:
                    # Fallback: create zeros with requires_grad from a dummy parameter
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
            # Pad with learnable parameters to maintain gradient flow
            # CRITICAL: Use bias vectors as learnable padding to keep computation graph connected
            padding_size = 10 - final_output.shape[1]
            # Create a learnable padding from output layer's bias vector (padded portion)
            output_layer_idx = len(self.layer_positions) - 1
            bias_key = f"biases_{output_layer_idx}"
            if bias_key in self.bias_vectors and self.bias_vectors[bias_key].shape[0] >= 10:
                # Use the last padding_size elements from bias vector
                padding = self.bias_vectors[bias_key][-padding_size:].unsqueeze(0).expand(batch_size, -1)
            else:
                # Fallback: create zeros with batch dimension
                padding = torch.zeros(batch_size, padding_size, device=self.device)
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
