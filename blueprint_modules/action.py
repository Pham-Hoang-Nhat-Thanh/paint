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
    """Defines the types of actions that can modify a neural architecture."""
    ADD_NEURON = 0
    REMOVE_NEURON = 1
    MODIFY_ACTIVATION = 2
    ADD_CONNECTION = 3
    REMOVE_CONNECTION = 4

@dataclass
class Action:
    """Represents a single modification to a neural architecture.

    Attributes:
        action_type (ActionType): The type of action to be performed.
        source_neuron (Optional[int]): The ID of the source neuron, if
            applicable.
        target_neuron (Optional[int]): The ID of the target neuron, if
            applicable.
        activation (Optional[ActivationType]): The activation function, if
            applicable.
        parameters (Dict): Additional parameters for the action.
    """
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
    """Defines and manages the set of valid actions for a neural architecture.

    This class is responsible for generating all possible valid actions that can
    be taken from a given architectural state, subject to various constraints.

    Attributes:
        max_neurons (int): The maximum number of neurons allowed in an
            architecture.
        max_connections (int): The maximum number of connections allowed.
        model_max_neurons (int): The hard limit of neurons supported by the
            policy network model.
    """

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
        """Returns a list of valid actions for the given architecture.

        The set of valid actions can be filtered based on the current
        evolutionary phase to guide the search process.

        Args:
            architecture (NeuralArchitecture): The current architecture.
            evolutionary_cycle (Optional['EvolutionaryCycle']): The current
                evolutionary cycle, which determines the phase.

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

    def _cache_actions(self, architecture: NeuralArchitecture, valid_actions: List[Action]) -> None:
        """Cache action primitives for fast lookup."""
        try:
            sig = architecture.compute_signature()
            if sig is None:
                return
            
            act_to_idx = {act: i for i, act in enumerate(ActivationType)}
            
            types = np.array([int(a.action_type.value) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
            sources = np.array([(-1 if a.source_neuron is None else int(a.source_neuron)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
            targets = np.array([(-1 if a.target_neuron is None else int(a.target_neuron)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
            activations = np.array([(-1 if a.activation is None else act_to_idx.get(a.activation, -1)) for a in valid_actions], dtype=int) if valid_actions else np.empty((0,), dtype=int)
            
            self._full_action_primitives[sig] = {
                'actions': valid_actions,
                'types': types,
                'sources': sources,
                'targets': targets,
                'activations': activations
            }
        except Exception:
            pass  # Best-effort caching

    def _try_get_cached_actions(self, architecture: NeuralArchitecture) -> Optional[List[Action]]:
        """Try to retrieve cached actions."""
        try:
            sig = architecture.compute_signature()
            if sig in self._full_action_primitives:
                cached_data = self._full_action_primitives[sig]
                if 'actions' in cached_data:
                    return cached_data['actions']
        except Exception:
            pass
        return None

    def _get_add_neuron_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Generate ADD_NEURON actions if allowed."""
        valid_actions = []
        num_neurons = len(architecture.neurons)
        can_add_neuron = num_neurons < self.max_neurons and architecture.next_neuron_id < self.model_max_neurons
        if can_add_neuron:
            for activation in ActivationType:
                valid_actions.append(Action(action_type=ActionType.ADD_NEURON, activation=activation))
        return valid_actions

    def _get_remove_neuron_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Generate REMOVE_NEURON actions for isolated hidden neurons."""
        isolated_hidden = self._get_isolated_hidden_neurons(architecture)
        removable = sorted([nid for nid, n in architecture.neurons.items() 
                        if n.neuron_type == NeuronType.HIDDEN and nid in isolated_hidden])
        return [Action(action_type=ActionType.REMOVE_NEURON, source_neuron=nid) for nid in removable]

    def _get_modify_activation_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Generate MODIFY_ACTIVATION actions for all hidden neurons."""
        valid_actions = []
        neurons = architecture.neurons
        hidden_ids = sorted([nid for nid, n in neurons.items() if n.neuron_type == NeuronType.HIDDEN])
        
        for nid in hidden_ids:
            current_act = neurons[nid].activation
            for act in ActivationType:
                if act != current_act:
                    valid_actions.append(Action(action_type=ActionType.MODIFY_ACTIVATION, 
                                            source_neuron=nid, activation=act))
        return valid_actions

    def _get_add_connection_actions_vectorized(self, architecture: NeuralArchitecture) -> List[Action]:
        """Generate ADD_CONNECTION actions using optimized vectorized approach."""
        valid_actions = []
        neurons = architecture.neurons
        
        # Pre-filter valid sources and targets
        valid_sources = np.array(sorted([
            nid for nid, n in neurons.items() if n.neuron_type != NeuronType.OUTPUT
        ]), dtype=int)
        
        valid_targets = np.array(sorted([
            nid for nid, n in neurons.items() if n.neuron_type != NeuronType.INPUT
        ]), dtype=int)
        
        if len(valid_sources) == 0 or len(valid_targets) == 0:
            return valid_actions
        
        # Create grid of valid pairs
        src_ids = np.repeat(valid_sources, len(valid_targets))
        tgt_ids = np.tile(valid_targets, len(valid_sources))
        
        # Apply filters
        not_self = src_ids != tgt_ids
        
        existing_connections = frozenset(
            (c.source_id, c.target_id) for c in architecture.connections
        )
        is_new = np.array([
            (s, t) not in existing_connections for s, t in zip(src_ids, tgt_ids)
        ], dtype=bool)
        
        # Layer validation with isolation support
        layers = {nid: architecture.get_layer_for_neuron(nid) 
                for nid in np.union1d(valid_sources, valid_targets)}
        src_layers = np.array([layers[s] for s in src_ids], dtype=int)
        tgt_layers = np.array([layers[t] for t in tgt_ids], dtype=int)
        layer_ok = (src_layers <= tgt_layers) | (src_layers == -1) | (tgt_layers == -1)
        
        # Combine all filters
        final_mask = not_self & is_new & layer_ok
        valid_pairs = np.stack((src_ids[final_mask], tgt_ids[final_mask]), axis=1)
        
        for s_id, t_id in valid_pairs:
            valid_actions.append(Action(
                action_type=ActionType.ADD_CONNECTION,
                source_neuron=int(s_id),
                target_neuron=int(t_id)
            ))
        
        return valid_actions

    def _get_remove_connection_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Generate REMOVE_CONNECTION actions for all existing connections."""
        return [Action(action_type=ActionType.REMOVE_CONNECTION, 
                    source_neuron=conn.source_id, 
                    target_neuron=conn.target_id) 
                for conn in architecture.connections]

    def _get_expanding_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Actions for the expanding phase: only adding new neurons."""
        # Check cache first
        cached = self._try_get_cached_actions(architecture)
        if cached is not None:
            return cached
        
        valid_actions = self._get_add_neuron_actions(architecture)
        self._cache_actions(architecture, valid_actions)
        return valid_actions

    def _get_refinement_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Actions for the refinement phase: adding connections and modifying activations."""
        # Check cache first
        cached = self._try_get_cached_actions(architecture)
        if cached is not None:
            return cached
        
        valid_actions = []
        valid_actions.extend(self._get_modify_activation_actions(architecture))
        valid_actions.extend(self._get_add_connection_actions_vectorized(architecture))
        
        self._cache_actions(architecture, valid_actions)
        return valid_actions

    def _get_pruning_actions(self, architecture: NeuralArchitecture) -> List[Action]:
        """Actions for the pruning phase: removing neurons and connections."""
        # Check cache first
        cached = self._try_get_cached_actions(architecture)
        if cached is not None:
            return cached
        
        valid_actions = []
        valid_actions.extend(self._get_remove_neuron_actions(architecture))
        valid_actions.extend(self._get_remove_connection_actions(architecture))
        
        self._cache_actions(architecture, valid_actions)
        return valid_actions

    def _get_full_action_space(self, architecture: NeuralArchitecture) -> List[Action]:
        """Return the full set of valid actions for the architecture.

        This enumerates all sensible actions while applying validation rules
        (type-based and topological). The returned list is deterministically
        ordered by neuron ids and activation enumeration to make behaviour stable
        across runs.
        """
        # Check cache first
        cached = self._try_get_cached_actions(architecture)
        if cached is not None:
            return cached
        
        valid_actions = []
        
        # Collect all action types
        valid_actions.extend(self._get_add_neuron_actions(architecture))
        valid_actions.extend(self._get_remove_neuron_actions(architecture))
        valid_actions.extend(self._get_modify_activation_actions(architecture))
        valid_actions.extend(self._get_add_connection_actions_vectorized(architecture))
        valid_actions.extend(self._get_remove_connection_actions(architecture))
        
        self._cache_actions(architecture, valid_actions)
        return valid_actions
    
    def apply_action(self, architecture: NeuralArchitecture, action: Action) -> bool:
        """Applies a given action to the architecture.

        This method modifies the architecture in-place based on the provided
        action.

        Args:
            architecture (NeuralArchitecture): The architecture to modify.
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
        """Clears the action cache to free memory.

        This is useful to call periodically to prevent the cache from growing
        indefinitely.
        """
        try:
            self._full_action_primitives.clear()
        except Exception:
            self._full_action_primitives = {}
        # Force GC to release references to numpy arrays
        gc.collect()
        