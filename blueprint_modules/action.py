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
        """Get valid actions for the current architecture state.

        By default (`full=True`) this returns the full validated action space (ADD_NEURON,
        REMOVE_NEURON, MODIFY_ACTIVATION, ADD_CONNECTION, REMOVE_CONNECTION) in a
        deterministic ordering. If `full=False` the previous prioritized behavior
        (cycle-aware or simple) is preserved for backward compatibility.
        """
        return self._get_full_action_space(architecture)

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
