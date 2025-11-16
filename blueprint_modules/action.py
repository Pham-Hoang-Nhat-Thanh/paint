from enum import Enum
from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass
import traceback
from .network import NeuralArchitecture, NeuronType, ActivationType
import gc
from .evolutionary_cycle import Phase

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
                 max_steps_per_episode: int = 50, connection_candidate_multiplier: int = 3,
                 model_max_neurons: int = None):
        # max_neurons: search-time soft limit (number of neurons allowed in architecture)
        # model_max_neurons: hard limit imposed by model heads (logit vector sizes)
        self.max_neurons = max_neurons
        self.model_max_neurons = model_max_neurons if model_max_neurons is not None else max_neurons
        self.max_connections = max_connections
        self.max_steps_per_episode = max_steps_per_episode
        self.connection_candidate_multiplier = connection_candidate_multiplier
        # Cache for full action space primitives keyed by architecture signature
        # Stored format: signature -> {'actions': List[Action], 'types': np.ndarray, 'sources': np.ndarray,
        #                              'targets': np.ndarray, 'activations': np.ndarray}
        self._full_action_primitives = {}

    def get_valid_actions(self, architecture: NeuralArchitecture,
                         evolutionary_cycle: Optional['EvolutionaryCycle'] = None) -> List[Action]:
        """Get valid actions for the current architecture state based on the evolutionary phase."""
        if evolutionary_cycle is None:
            return self._get_full_action_space(architecture)

        phase = evolutionary_cycle.current_phase
        if phase == Phase.EXPANDING:
            return self._get_expanding_actions(architecture)
        elif phase == Phase.REFINEMENT:
            return self._get_refinement_actions(architecture)
        elif phase == Phase.PRUNING:
            return self._get_pruning_actions(architecture)
        else:
            return []

    def _get_full_action_space(self, architecture: NeuralArchitecture) -> List[Action]:
        """Return the full set of valid actions for the architecture.

        This enumerates all sensible actions while applying validation rules
        (type-based and topological). The returned list is deterministically
        ordered by neuron ids and activation enumeration to make behaviour stable
        across runs.
        """
        valid_actions: List[Action] = []
        neurons = architecture.neurons

        # ADD_NEURON: propose for each activation type if we can add neurons
        num_neurons = len(neurons)
        can_add_neuron = num_neurons < self.max_neurons and architecture.next_neuron_id < self.model_max_neurons
        if can_add_neuron:
            # Deterministically propose one ADD_NEURON per ActivationType (policy heads can map them)
            for activation in ActivationType:
                valid_actions.append(Action(action_type=ActionType.ADD_NEURON, activation=activation))

        # REMOVE_NEURON: propose removal for isolated hidden neurons (deterministic order)
        isolated_hidden = self._get_isolated_hidden_neurons(architecture)
        removable = sorted([nid for nid, n in neurons.items() if n.neuron_type == NeuronType.HIDDEN and nid in isolated_hidden])
        for nid in removable:
            valid_actions.append(Action(action_type=ActionType.REMOVE_NEURON, source_neuron=nid))

        # MODIFY_ACTIVATION: every hidden neuron may change to any other activation
        hidden_ids = sorted([nid for nid, n in neurons.items() if n.neuron_type == NeuronType.HIDDEN])
        for nid in hidden_ids:
            current_act = neurons[nid].activation
            for act in ActivationType:
                if act != current_act:
                    valid_actions.append(Action(action_type=ActionType.MODIFY_ACTIVATION, source_neuron=nid, activation=act))

        # ADD_CONNECTION: vectorized enumeration of ordered pairs (src != tgt) that are valid and not present
        all_ids = np.array(sorted(neurons.keys()), dtype=int)
        n = len(all_ids)
        if n > 0:
            # Build index mapping and arrays for neuron properties
            id_to_idx = {int(nid): i for i, nid in enumerate(all_ids)}
            # Layers (may be cached inside architecture)
            layers = np.array([architecture.get_layer_for_neuron(int(nid)) for nid in all_ids], dtype=int)
            # Types
            is_output = np.array([neurons[int(nid)].neuron_type == NeuronType.OUTPUT for nid in all_ids], dtype=bool)
            is_input = np.array([neurons[int(nid)].neuron_type == NeuronType.INPUT for nid in all_ids], dtype=bool)

            # Adjacency matrix for existing connections (small loop over connections)
            adjacency = np.zeros((n, n), dtype=bool)
            for conn in architecture.connections:
                s = conn.source_id
                t = conn.target_id
                if s in id_to_idx and t in id_to_idx:
                    adjacency[id_to_idx[s], id_to_idx[t]] = True

            # Create grid of all pairs via repeated indices
            src_idx = np.repeat(np.arange(n, dtype=int), n)
            tgt_idx = np.tile(np.arange(n, dtype=int), n)

            # Exclude self-connections
            neq_mask = src_idx != tgt_idx

            # Exclude already existing connections
            adj_flat = adjacency.reshape(-1)
            not_existing_mask = ~adj_flat

            # Type-based rules: source not OUTPUT, target not INPUT
            src_not_output = ~is_output[src_idx]
            tgt_not_input = ~is_input[tgt_idx]

            # Layer rule: allow if either isolated (-1) or src_layer <= tgt_layer
            # Additionally explicitly allow pairs involving isolated hidden neurons
            src_layers = layers[src_idx]
            tgt_layers = layers[tgt_idx]
            layer_ok = (src_layers == -1) | (tgt_layers == -1) | (src_layers <= tgt_layers)

            # Explicitly include isolated hidden neurons (topo layer == -1)
            try:
                if isolated_hidden:
                    # Build boolean masks for sampled flattened indices
                    src_isolated = np.isin(all_ids[src_idx], list(isolated_hidden))
                    tgt_isolated = np.isin(all_ids[tgt_idx], list(isolated_hidden))
                    layer_ok = layer_ok | src_isolated | tgt_isolated
            except Exception:
                # If isolation detection fails for any reason, fall back to previous rule
                pass

            valid_pair_mask = neq_mask & not_existing_mask & src_not_output & tgt_not_input & layer_ok

            valid_positions = np.nonzero(valid_pair_mask)[0]
            # Append Actions in deterministic order implied by all_ids sorting and index flattening
            for pos in valid_positions:
                s_idx = int(src_idx[pos])
                t_idx = int(tgt_idx[pos])
                s_id = int(all_ids[s_idx])
                t_id = int(all_ids[t_idx])
                valid_actions.append(Action(action_type=ActionType.ADD_CONNECTION, source_neuron=s_id, target_neuron=t_id))

        # REMOVE_CONNECTION: every existing connection can be removed (preserve input order)
        for conn in architecture.connections:
            valid_actions.append(Action(action_type=ActionType.REMOVE_CONNECTION, source_neuron=conn.source_id, target_neuron=conn.target_id))

        # Build primitive numpy arrays for fast downstream scoring and membership tests
        try:
            sig = architecture.compute_signature()
        except Exception:
            sig = None

        # Map ActivationType to small integer indices for compact encoding
        act_to_idx = {act: i for i, act in enumerate(ActivationType)}

        types = np.array([int(a.action_type.value) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
        sources = np.array([(-1 if a.source_neuron is None else int(a.source_neuron)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
        targets = np.array([(-1 if a.target_neuron is None else int(a.target_neuron)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
        activations = np.array([(-1 if a.activation is None else act_to_idx.get(a.activation, -1)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)

        if sig is not None:
            try:
                self._full_action_primitives[sig] = {
                    'actions': valid_actions,
                    'types': types,
                    'sources': sources,
                    'targets': targets,
                    'activations': activations
                }
            except Exception:
                # Best-effort caching: if any issue occurs, continue without failing
                pass

        return valid_actions
    
    def _get_expanding_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Actions for the expanding phase: only adding new neurons."""
        valid_actions: List[Action] = []
        num_neurons = len(architecture.neurons)
        can_add_neuron = num_neurons < self.max_neurons and architecture.next_neuron_id < self.model_max_neurons
        if can_add_neuron:
            for activation in ActivationType:
                valid_actions.append(Action(action_type=ActionType.ADD_NEURON, activation=activation))
        return valid_actions
    
    def _get_refinement_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Actions for the refinement phase: adding connections and modifying activations."""
        valid_actions: List[Action] = []
        neurons = architecture.neurons
        hidden_ids = sorted([nid for nid, n in neurons.items() if n.neuron_type == NeuronType.HIDDEN])

        # MODIFY_ACTIVATION: every hidden neuron may change to any other activation
        for nid in hidden_ids:
            current_act = neurons[nid].activation
            for act in ActivationType:
                if act != current_act:
                    valid_actions.append(Action(action_type=ActionType.MODIFY_ACTIVATION, source_neuron=nid, activation=act))
        
        all_ids = np.array(sorted(neurons.keys()), dtype=int)
        n = len(all_ids)
        if n > 0:
            id_to_idx = {int(nid): i for i, nid in enumerate(all_ids)}
            layers = np.array([architecture.get_layer_for_neuron(int(nid)) for nid in all_ids], dtype=int)
            is_output = np.array([neurons[int(nid)].neuron_type == NeuronType.OUTPUT for nid in all_ids], dtype=bool)
            is_input = np.array([neurons[int(nid)].neuron_type == NeuronType.INPUT for nid in all_ids], dtype=bool)
            adjacency = np.zeros((n, n), dtype=bool)
            for conn in architecture.connections:
                s = conn.source_id
                t = conn.target_id
                if s in id_to_idx and t in id_to_idx:
                    adjacency[id_to_idx[s], id_to_idx[t]] = True

            # Create grid of all pairs via repeated indices
            src_idx = np.repeat(np.arange(n, dtype=int), n)
            tgt_idx = np.tile(np.arange(n, dtype=int), n)

            # Exclude self-connections
            neq_mask = src_idx != tgt_idx

            # Exclude already existing connections
            adj_flat = adjacency.reshape(-1)
            not_existing_mask = ~adj_flat

            # Type-based rules: source not OUTPUT, target not INPUT
            src_not_output = ~is_output[src_idx]
            tgt_not_input = ~is_input[tgt_idx]

            # Layer rule: allow if either isolated (-1) or src_layer <= tgt_layer
            # Additionally explicitly allow pairs involving isolated hidden neurons
            src_layers = layers[src_idx]
            tgt_layers = layers[tgt_idx]
            layer_ok = (src_layers == -1) | (tgt_layers == -1) | (src_layers <= tgt_layers)

            valid_pair_mask = neq_mask & not_existing_mask & src_not_output & tgt_not_input & layer_ok

            valid_positions = np.nonzero(valid_pair_mask)[0]
            # Append Actions in deterministic order implied by all_ids sorting and index flattening
            for pos in valid_positions:
                s_idx = int(src_idx[pos])
                t_idx = int(tgt_idx[pos])
                s_id = int(all_ids[s_idx])
                t_id = int(all_ids[t_idx])
                valid_actions.append(Action(action_type=ActionType.ADD_CONNECTION, source_neuron=s_id, target_neuron=t_id))

        # Build primitive numpy arrays for fast downstream scoring and membership tests
        try:
            sig = architecture.compute_signature()
        except Exception:
            sig = None

        # Map ActivationType to small integer indices for compact encoding
        act_to_idx = {act: i for i, act in enumerate(ActivationType)}

        types = np.array([int(a.action_type.value) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
        sources = np.array([(-1 if a.source_neuron is None else int(a.source_neuron)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
        targets = np.array([(-1 if a.target_neuron is None else int(a.target_neuron)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
        activations = np.array([(-1 if a.activation is None else act_to_idx.get(a.activation, -1)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)

        if sig is not None:
            try:
                self._full_action_primitives[sig] = {
                    'actions': valid_actions,
                    'types': types,
                    'sources': sources,
                    'targets': targets,
                    'activations': activations
                }
            except Exception:
                # Best-effort caching: if any issue occurs, continue without failing
                pass

        return valid_actions
    
    def _get_pruning_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Actions for the pruning phase: removing neurons and connections."""
        valid_actions: List[Action] = []
        isolated_hidden = self._get_isolated_hidden_neurons(architecture)
        removable = sorted([nid for nid, n in architecture.neurons.items() if n.neuron_type == NeuronType.HIDDEN and nid in isolated_hidden])
        for nid in removable:
            valid_actions.append(Action(action_type=ActionType.REMOVE_NEURON, source_neuron=nid))
        
        for conn in architecture.connections:
            valid_actions.append(Action(action_type=ActionType.REMOVE_CONNECTION, source_neuron=conn.source_id, target_neuron=conn.target_id))
        
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

    def _get_isolated_hidden_neurons(self, architecture: NeuralArchitecture) -> set:
        """Get set of isolated hidden neurons - O(n) where n is number of neurons.
        
        Uses cached connectivity from architecture to avoid re-scanning connections.
        Cache is automatically invalidated when architecture is modified.
        """
        # Get cached connectivity (O(1) after first call, O(n+m) only on first call or after modification)
        connectivity = architecture._get_connectivity_sets()
        has_incoming = connectivity['has_incoming']
        has_outgoing = connectivity['has_outgoing']
        
        # Single pass over neurons only - O(n)
        isolated = set()
        for neuron_id, neuron in architecture.neurons.items():
            if neuron.neuron_type == NeuronType.HIDDEN:
                # Isolated if missing either incoming or outgoing connections
                if neuron_id not in has_incoming or neuron_id not in has_outgoing:
                    isolated.add(neuron_id)
        
        return isolated

    def _is_valid_connection(self, architecture: NeuralArchitecture, source_id: int, target_id: int) -> bool:
        """Check if a connection between two neurons is valid.

        Rules enforced:
        - Output neurons cannot be sources.
        - Input neurons cannot be targets.
        - Disallow connections where the source is in a strictly higher topological layer
          than the target (this prevents backward edges / cycles).
        - Allow connections involving isolated neurons (layer -1) because these actions
          are used to de-/re-isolate neurons (we want to allow de-isolation).

        This reduces the need for an expensive cycle-check while still permitting
        de-isolation actions to proceed.
        """
        neurons = architecture.neurons
        source_type = neurons[source_id].neuron_type
        target_type = neurons[target_id].neuron_type

        # Output neurons can't be sources
        if source_type == NeuronType.OUTPUT:
            return False
        
        # Input neurons can't be targets
        if target_type == NeuronType.INPUT:
            return False

        # Consult topological layers to forbid backward edges. Use architecture helper.
        try:
            src_layer = architecture.get_layer_for_neuron(source_id)
            tgt_layer = architecture.get_layer_for_neuron(target_id)
        except Exception:
            # If we can't obtain layers for any reason, fall back to permissive boolean
            # based on types only (safer than raising here).
            return True

        # Allow connections that involve isolated neurons (layer == -1) so that
        # de-isolation/reconnection actions remain possible.
        if src_layer == -1 or tgt_layer == -1:
            return True

        # Disallow connections where source is in a strictly higher/topologically later
        # layer than the target (would create a backward edge).
        if src_layer > tgt_layer:
            return False

        return True

    def get_action_primitives(self, architecture: NeuralArchitecture):
        """Get cached action primitives for fast scoring and membership tests"""
        try:
            sig = architecture.compute_signature()
        except Exception:
            sig = None

        if sig in self._full_action_primitives:
            return self._full_action_primitives[sig]
        else:
            # Compute full action space to populate cache
            self._get_full_action_space(architecture)
            return self._full_action_primitives.get(sig, None)

    def clear_cache(self):
        """Clear cached action primitives and release memory held by numpy arrays/lists."""
        try:
            self._full_action_primitives.clear()
        except Exception:
            self._full_action_primitives = {}
        # Force GC to release references to numpy arrays
        gc.collect()
        