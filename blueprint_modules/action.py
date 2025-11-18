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
    """Enumeration of possible actions on the neural architecture."""
    ADD_NEURON = 0
    REMOVE_NEURON = 1
    MODIFY_ACTIVATION = 2
    ADD_CONNECTION = 3
    REMOVE_CONNECTION = 4

@dataclass
class Action:
    """Represents an action to be applied to a neural architecture.

    Attributes:
        action_type (ActionType): The type of action to perform.
        source_neuron (Optional[int]): The ID of the source neuron.
        target_neuron (Optional[int]): The ID of the target neuron.
        activation (Optional[ActivationType]): The activation function.
        parameters (Dict): Additional parameters for the action.
    """
    action_type: ActionType
    source_neuron: Optional[int] = None
    target_neuron: Optional[int] = None
    activation: Optional[ActivationType] = None
    parameters: Dict = None
    
    def __post_init__(self):
        """Initializes the parameters dictionary if it is None."""
        if self.parameters is None:
            self.parameters = {}
    
    def __eq__(self, other):
        """Checks if two actions are equal.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the actions are equal, False otherwise.
        """
        if not isinstance(other, Action):
            return False
        return (self.action_type == other.action_type and
                self.source_neuron == other.source_neuron and
                self.target_neuron == other.target_neuron and
                self.activation == other.activation)
    
    def __hash__(self):
        """Computes the hash of the action to allow it to be used in sets.

        Returns:
            int: The hash of the action.
        """
        return hash((self.action_type, self.source_neuron, self.target_neuron, self.activation))

class ActionSpace:
    """Manages the valid actions for a given neural architecture.

    This class is responsible for generating the set of all possible valid
    actions that can be applied to a neural architecture, taking into account
    various constraints and the current evolutionary phase.

    Attributes:
        max_neurons (int): The maximum number of neurons allowed.
        model_max_neurons (int): The hard limit of neurons imposed by the model.
        max_connections (int): The maximum number of connections allowed.
        max_steps_per_episode (int): The maximum number of steps per episode.
        connection_candidate_multiplier (int): The multiplier for connection
            candidates.
    """

    def __init__(self, max_neurons: int = 1000, max_connections: int = 10000,
                 max_steps_per_episode: int = 50, connection_candidate_multiplier: int = 3,
                 model_max_neurons: int = None):
        """Initializes the ActionSpace.

        Args:
            max_neurons (int): The maximum number of neurons allowed.
            max_connections (int): The maximum number of connections allowed.
            max_steps_per_episode (int): The maximum number of steps per episode.
            connection_candidate_multiplier (int): The multiplier for
                connection candidates.
            model_max_neurons (int, optional): The hard limit of neurons imposed
                by the model. Defaults to None.
        """
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
        """Gets the valid actions for the current architecture state.

        Args:
            architecture (NeuralArchitecture): The neural architecture.
            evolutionary_cycle (Optional['EvolutionaryCycle']): The current
                evolutionary cycle. Defaults to None.

        Returns:
            List[Action]: A list of valid actions.
        """
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
        """Returns the full set of valid actions for the architecture.

        This enumerates all sensible actions while applying validation rules
        (type-based and topological). The returned list is deterministically
        ordered by neuron ids and activation enumeration to make behaviour stable
        across runs.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            List[Action]: A list of all valid actions.
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
        """Gets the actions for the expanding phase (only adding new neurons).

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            List[Action]: A list of valid expanding actions.
        """
        valid_actions: List[Action] = []
        num_neurons = len(architecture.neurons)
        can_add_neuron = num_neurons < self.max_neurons and architecture.next_neuron_id < self.model_max_neurons
        if can_add_neuron:
            for activation in ActivationType:
                valid_actions.append(Action(action_type=ActionType.ADD_NEURON, activation=activation))
        return valid_actions
    
    def _get_refinement_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Gets actions for the refinement phase.

        This includes adding connections and modifying activations.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            List[Action]: A list of valid refinement actions.
        """
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
        """Gets actions for the pruning phase.

        This includes removing neurons and connections.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            List[Action]: A list of valid pruning actions.
        """
        valid_actions: List[Action] = []
        isolated_hidden = self._get_isolated_hidden_neurons(architecture)
        removable = sorted([nid for nid, n in architecture.neurons.items() if n.neuron_type == NeuronType.HIDDEN and nid in isolated_hidden])
        for nid in removable:
            valid_actions.append(Action(action_type=ActionType.REMOVE_NEURON, source_neuron=nid))
        
        for conn in architecture.connections:
            valid_actions.append(Action(action_type=ActionType.REMOVE_CONNECTION, source_neuron=conn.source_id, target_neuron=conn.target_id))
        
        return valid_actions

    def apply_action(self, architecture: NeuralArchitecture, action: Action) -> bool:
        """Applies an action to the architecture.

        Args:
            architecture (NeuralArchitecture): The neural architecture.
            action (Action): The action to apply.

        Returns:
            bool: True if the action was applied successfully, False otherwise.
        """
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
        """Gets the set of isolated hidden neurons.

        This method uses cached connectivity from the architecture to avoid
        re-scanning connections.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            set: A set of isolated hidden neuron IDs.
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
        """Checks if a connection between two neurons is valid.

        Args:
            architecture (NeuralArchitecture): The neural architecture.
            source_id (int): The ID of the source neuron.
            target_id (int): The ID of the target neuron.

        Returns:
            bool: True if the connection is valid, False otherwise.
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
        """Gets cached action primitives for fast scoring and membership tests.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            Dict: The cached action primitives.
        """
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
        """Clears the cached action primitives."""
        try:
            self._full_action_primitives.clear()
        except Exception:
            self._full_action_primitives = {}
        # Force GC to release references to numpy arrays
        gc.collect()
        