import traceback
from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture
from blueprint_modules.action import Action, ActionSpace
from blueprint_modules.evolutionary_cycle import EvolutionaryCycle
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict, List
import torch
import math
import torch.nn.functional as F
from collections import deque, defaultdict
import time

class NeuralMCTSNode(MCTSNode):
    """Represents a node in the MCTS tree, enhanced with neural network data.

    This class extends the base `MCTSNode` to store policy and value
    predictions from the neural network, which are used to guide the search
    process.

    Attributes:
        policy_value (Dict): A dictionary containing the raw output from the
            policy-value network for this node's architecture.
        prior_prob (float): The prior probability of selecting the action that
            led to this node, as predicted by the policy network.
    """

    def __init__(self, architecture: NeuralArchitecture, policy_value: Dict = None,
                 parent=None, action: Action = None, curriculum=None):
        # MCTSNode stores architecture, parent, action
        super().__init__(architecture, parent, action)
        # Use a dictionary for children for O(1) action lookup
        self.children: Dict[Action, 'NeuralMCTSNode'] = {}
        # Neural network predictions for this node's state
        self.policy_value = policy_value
        # Prior probability of the action that led to this node
        self.prior_prob = 0.0
        # Curriculum state
        self.curriculum = curriculum
        # --- Caches for lazy instantiation and performance ---
        # Cache for valid actions and their corresponding priors from the policy network
        # This is populated once when the node is first expanded.
        self._unexpanded_priors: Dict[Action, float] = None
        # Cache action masks to avoid recomputation
        self._cached_masks = None
        # Track which phase the masks/priors were computed for
        self._cache_phase_token = None
    
    def is_fully_expanded(self) -> bool:
        """Checks if all valid actions from this node have been explored.

        A node is considered fully expanded if there are no more unexpanded
        actions with non-zero prior probability.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        # Fully expanded if the unexpanded actions cache has been populated and is empty
        return self._unexpanded_priors is not None and len(self._unexpanded_priors) == 0
    
    def best_child(self, exploration_weight: float = 1.0) -> 'NeuralMCTSNode':
        """Selects the best child node using the PUCT algorithm.

        The PUCT (Polynomial Upper Confidence Trees) formula balances
        exploitation (choosing nodes with high value) and exploration (choosing
        nodes with high prior probability and low visit counts).

        Args:
            exploration_weight (float): The constant controlling the level of
                exploration.

        Returns:
            NeuralMCTSNode: The child node with the highest PUCT score.
        """
        if not self.children:
            return None

        children_list = list(self.children.values())

        # Vectorized selection: if any child has zero visits, prefer first such child
        # (keeps original behavior which returns +inf for zero-visit children).
        try:
            visits = torch.tensor([float(child.visits) for child in children_list], dtype=torch.float32)
            zero_mask = visits == 0.0
            if zero_mask.any():
                zero_idxs = torch.nonzero(zero_mask).squeeze(1).tolist()
                # choose among zero-visit children the one with highest prior
                priors = [children_list[i].prior_prob for i in zero_idxs]
                best = zero_idxs[int(torch.argmax(torch.tensor(priors)).item())]
                return children_list[best]

            # Q: average value per child
            values = torch.tensor([float(child.value) for child in children_list], dtype=torch.float32)
            q_values = values / visits

            # U: exploration term (PUCT)
            priors = torch.tensor([float(child.prior_prob) for child in children_list], dtype=torch.float32)
            parent_visits = float(self.visits) if self.visits > 0 else 1.0
            u_values = exploration_weight * priors * (math.sqrt(parent_visits) / (1.0 + visits))

            scores = q_values + u_values
            best_idx = int(torch.argmax(scores).item())
            return children_list[best_idx]
        except Exception:
            # Fallback to original Python implementation on any failure
            def puct_score(child: 'NeuralMCTSNode') -> float:
                if child.visits == 0:
                    return float('inf')

                q_value = child.value / child.visits
                u_value = exploration_weight * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
                return q_value + u_value

            return max(children_list, key=puct_score)

class NeuralMCTS(MCTS):
    """Implements Monte Carlo Tree Search guided by a policy-value network.

    This class orchestrates the MCTS process, using a neural network to provide
    priors for action selection and to evaluate leaf nodes. It follows the
    AlphaZero methodology for search.

    Attributes:
        policy_value_net (UnifiedPolicyValueNetwork): The network used to guide
            the search.
        device (str): The device ('cpu' or 'cuda') to run the network on.
        exploration_weight (float): A constant controlling exploration in the
            PUCT formula.
    """

    def __init__(self, action_space: ActionSpace, policy_value_net: UnifiedPolicyValueNetwork,
                 device: str = 'cpu', exploration_weight: float = 1.0,
                 iso_weight: float = 0.01, comp_weight: float = 0.0, 
                 early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.001, max_children: int = 50, max_neurons: int = 1000):
        super().__init__(action_space, None, exploration_weight)
        self.policy_value_net = policy_value_net
        self.device = device
        self.iso_weight = iso_weight
        self.comp_weight = comp_weight
        # Early stopping settings
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        # Cache for evaluations to avoid redundant computations
        self.evaluation_cache = {}
        # Evolutionary cycle tracking
        self.current_cycle = EvolutionaryCycle(stability_threshold=0.001)
        # Reusable ActionManager instance
        self.action_manager = ActionManager(action_space=self.action_space, max_neurons=max_neurons)
        self.max_children = max_children
        # Profiling containers for expand timings
        # `_expand_profiler` accumulates totals per timing key across the search
        # `_expand_profiles` stores recent per-expand breakdowns for inspection
        self._expand_profiler = defaultdict(float)
        self._expand_profiles = deque(maxlen=2000)

    def cleanup(self):
        """Cleans up resources to prevent memory leaks.

        This method should be called after a search is complete to free up
        memory used by the evaluation cache and to trigger garbage collection.
        """
        # Clear evaluation cache to free memory
        self.evaluation_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()

    def _prepare_graph_data(self, architecture: NeuralArchitecture) -> Dict:
        """Optimized: Convert architecture to graph data for neural network using cached sorted IDs"""
        graph_data = architecture.to_graph_representation()
        
        # Add batch dimension but keep tensors on CPU. Move to device lazily
        graph_data['node_features'] = graph_data['node_features'].unsqueeze(0)
        graph_data['edge_index'] = graph_data['edge_index']
        # Use cached sorted neuron IDs from graph representation instead of sorting again
        sorted_neuron_ids = graph_data['sorted_neuron_ids']
        layer_positions = [float(architecture.neurons[neuron_id].layer_position) for neuron_id in sorted_neuron_ids]
        graph_data['layer_positions'] = torch.FloatTensor([layer_positions])
        
        return graph_data

    def _batch_graph_data_for_eval(self, parent_graph: Dict, child_graphs) -> Dict:
        """Batch graph data dicts for joint evaluation.
        
        Flexible batching: supports parent + single child (Dict) or parent + multiple children (List[Dict]).
        Follows PyG batching: concatenate graphs into single disconnected graph with batch tensor.
        Enables single policy_value_net forward pass for all graphs.
        
        Args:
            parent_graph: Dict with graph data for parent
            child_graphs: Dict (single child) or List[Dict] (multiple children)
        
        Returns:
            Batched graph dict with num_graphs indicator, batch tensor tracking graph membership
        """
        # Normalize child_graphs to list
        if isinstance(child_graphs, dict):
            child_graphs = [child_graphs]
        
        num_graphs = 1 + len(child_graphs)
        
        # Collect node features, layer positions
        parent_nodes = parent_graph['node_features'].squeeze(0)  # [num_nodes_parent, features]
        parent_pos = parent_graph['layer_positions'].squeeze(0)  # [num_nodes_parent]
        
        all_node_features = [parent_nodes]
        all_layer_positions = [parent_pos]
        node_offsets = [0, parent_nodes.shape[0]]  # Track cumulative node counts for edge offset
        
        for child_graph in child_graphs:
            child_nodes = child_graph['node_features'].squeeze(0)  # [num_nodes_child, features]
            child_pos = child_graph['layer_positions'].squeeze(0)  # [num_nodes_child]
            all_node_features.append(child_nodes)
            all_layer_positions.append(child_pos)
            node_offsets.append(node_offsets[-1] + child_nodes.shape[0])
        
        # Concatenate all node features and positions
        all_node_features_cat = torch.cat(all_node_features, dim=0)  # [total_nodes, features]
        all_layer_positions_cat = torch.cat(all_layer_positions, dim=0)  # [total_nodes]
        
        # Build batch tensor: graph_id for each node
        batch_list = []
        for graph_id in range(num_graphs):
            num_nodes = node_offsets[graph_id + 1] - node_offsets[graph_id]
            batch_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long, device=self.device))
        batch_tensor = torch.cat(batch_list, dim=0)  # [total_nodes]
        
        # Merge edge indices with offsets for each child
        all_edges = []
        
        parent_edges = parent_graph['edge_index']  # [2, num_edges_parent]
        if parent_edges.shape[1] > 0:
            all_edges.append(parent_edges)
        
        for i, child_graph in enumerate(child_graphs):
            offset = node_offsets[i + 1]
            child_edges = child_graph['edge_index']  # [2, num_edges_child]
            
            if child_edges.shape[1] > 0:
                # Add offset to child edge indices
                child_edges_offset = child_edges + offset
                all_edges.append(child_edges_offset)
        
        if all_edges:
            all_edge_indices = torch.cat(all_edges, dim=1)
        else:
            all_edge_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Return batched format
        return {
            'node_features': all_node_features_cat.unsqueeze(0),  # [1, total_nodes, features]
            'edge_index': all_edge_indices,                       # [2, total_edges]
            'layer_positions': all_layer_positions_cat.unsqueeze(0),  # [1, total_nodes]
            'batch': batch_tensor,  # [total_nodes] - graph_id per node
            'node_offsets': node_offsets,  # Offsets for extracting per-graph outputs
            'num_graphs': num_graphs
        }
    
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
               temperature: float = 1.0, reuse_root: NeuralMCTSNode = None) -> NeuralMCTSNode:
        """Runs the MCTS search for a given number of iterations.

        This method performs the main MCTS loop: selection, expansion, and
        backpropagation. It uses the policy-value network to guide the search
        and can reuse a tree from a previous search to save computation.

        Args:
            initial_architecture (NeuralArchitecture): The starting point for
                the search.
            iterations (int): The number of MCTS iterations to perform.
            temperature (float): A parameter controlling the exploration in the
                final action selection.
            reuse_root (NeuralMCTSNode, optional): An existing node to start
                the search from, enabling tree reuse. Defaults to None.

        Returns:
            NeuralMCTSNode: The node corresponding to the best action found.
        """
        
        # Reuse existing tree root if provided, otherwise create new root
        if reuse_root is not None:
            root = reuse_root
            # Invalidate caches to force re-computation for the new search.
            # These attributes are part of the new lazy-instantiation design.
            root._unexpanded_priors = None
            root._cached_masks = None
            root._cache_phase_token = None
            # Force re-evaluation by the network in the first _expand call
            root.policy_value = None
        else:
            root = NeuralMCTSNode(initial_architecture)

        # Create a local copy of the episode-level cycle for this search so
        # the search can advance phases locally while starting from the
        # episode-level phase to avoid accidentally copying a
        # stale search-local cycle from a previous run.
        orig_cycle = self.current_cycle
        # Make a local search copy so the search can mutate phase/node history
        # without immediately affecting the episode-level tracker.
        search_cycle = orig_cycle.copy()
        # Point `self.current_cycle` to the local search copy so helper
        # methods naturally use the search-local phase during this call.
        self.current_cycle = search_cycle
        phase_value = self.current_cycle.get_phase_value()

        try:
            # Expand the root node to get initial policy and value. This populates
            # the root's policy_value and _unexpanded_priors attributes.
            self._expand(root)

            # Add Dirichlet noise to the root's priors to encourage exploration
            self._apply_dirichlet_noise_to_root(root)

            # Early stopping tracking
            best_values = deque(maxlen=self.early_stopping_patience)
            no_improvement_count = 0
            initial_visits = root.visits  # Track starting visits for reused trees
            
            print("Starting MCTS search...")
            select_duration = 0.0
            expand_duration = 0.0
            backup_duration = 0.0
            final_action_duration = 0.0
            for i in range(iterations):
                # ===== STEP 1: SELECT =====
                # Traverse the tree using PUCT, lazily creating a new node if an
                # unexpanded action is chosen. The returned node is the leaf
                # for this simulation.
                select_time = time.time()
                leaf_node = self._select(root)
                select_duration += time.time() - select_time

                # ===== STEP 2: EXPAND & EVALUATE =====
                # Evaluate the leaf node with the NN, get its value, and
                # compute priors for its children.
                expand_time = time.time()
                value = self._expand(leaf_node)
                expand_duration += time.time() - expand_time

                # Track evaluation for evolutionary cycle advancement
                self.current_cycle.add_evaluation(value)
                if self.current_cycle.should_advance(num_neurons=leaf_node.architecture.num_neurons(),
                                                     num_connections=leaf_node.architecture.num_connections()):
                    self.current_cycle.advance_phase()

                # ===== STEP 3: BACKUP =====
                # Propagate the value from the leaf node up to the root.
                backup_time = time.time()
                self._backpropagate(leaf_node, value)
                backup_duration += time.time() - backup_time

                # Early stopping check - only after sufficient new visits on reused trees
                new_visits = root.visits - initial_visits
                min_new_visits = min(20, iterations // 2)  # At least 20 new visits or half of iterations
                
                # Skip early stopping until we have enough new exploration on reused trees
                if new_visits < min_new_visits:
                    continue
                    
                current_best = root.value / root.visits if root.visits > 0 else 0.0
                best_values.append(current_best)

                if len(best_values) == self.early_stopping_patience and i > iterations // 2:
                    recent_best = max(best_values)
                    oldest_recent = min(best_values)
                    improvement = recent_best - oldest_recent

                    if improvement < self.early_stopping_min_delta:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0

                    if no_improvement_count >= self.early_stopping_patience:
                        print(f"Early stopping at iteration {i + 1}: no significant improvement in last {self.early_stopping_patience} iterations")
                        break

                if (i + 1) % max(1, iterations // 10) == 0:
                    print(f"MCTS iteration {i + 1}/{iterations} completed, best value: {current_best:.4f}")
                
            final_action_time = time.time()
            # ===== STEP 5: SELECT FINAL ACTION =====
            # Select final action using visit counts but ensure the selection
            # is legal under the episode-level cycle (`orig_cycle`). The
            # search used a local copy of the cycle; final action must be
            # permissible for the episode-level evolutionary phase.
            try:
                # Get episode-level valid actions for the root state
                episode_valid_actions = self.action_space.get_valid_actions(root.architecture,
                                                                           evolutionary_cycle=orig_cycle)
                episode_valid_set = set(episode_valid_actions)
                # Filter children whose actions are legal under the episode-level cycle

                episode_legal_children = {a: c for a, c in root.children.items() if a in episode_valid_set}
            except Exception:
                # Fallback to considering all children if mask retrieval fails
                episode_legal_children = root.children

            if episode_legal_children:
                # Use a lightweight candidate container so `_select_final_action`
                # picks from the filtered subset.
                candidate_root = NeuralMCTSNode(root.architecture)
                candidate_root.children = episode_legal_children
                final_node = self._select_final_action(candidate_root, temperature)
            else:
                final_node = self._select_final_action(root, temperature)

            if final_node and final_node.action:
                print("MCTS search completed successfully")
                return (final_node, root)
            else:
                print("MCTS search completed: no valid action found")
                return (root, root)  # Return tuple even on failure
        finally:
            self.current_cycle = orig_cycle
            final_action_duration += time.time() - final_action_time
            # Print timing summary for expand profiling
            try:
                self.dump_expand_profile()
            except Exception:
                pass
            #print(f"Timing Summary: Selection: {select_duration:.4f}s, Expansion: {expand_duration:.4f}s, "
            #    f"Backup: {backup_duration:.4f}s, Final Action: {final_action_duration:.4f}s")
    
    def _select_final_action(self, root: NeuralMCTSNode, temperature: float) -> NeuralMCTSNode:
        """Select final action based on visit counts (proportional selection)"""
        if not root.children:
            return root

        children_list = list(root.children.values())
        visit_counts = torch.tensor([child.visits for child in children_list])

        # Proportional selection based on visit counts (standard MCTS final selection)
        visit_probs = visit_counts.float() / visit_counts.sum()
        if temperature == 0.0:
            # Deterministic selection of most visited child
            selected_idx = int(torch.argmax(visit_probs).item())
        else:
            # Apply temperature scaling
            scaled_probs = visit_probs.pow(1.0 / temperature)
            scaled_probs = scaled_probs / scaled_probs.sum()
            selected_idx = Categorical(probs=scaled_probs).sample().item()
        return children_list[selected_idx]
    
    def _select(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Traverse the tree, creating a new node if an unexpanded action is selected.

        This method implements the selection phase of MCTS, using the PUCT
        algorithm to balance exploration and exploitation. It also handles the
        lazy instantiation of child nodes.

        Args:
            node: The starting node for the selection process.

        Returns:
            The leaf node to be evaluated and expanded.
        """
        current_node = node
        while True:
            # If the node has not been evaluated yet (i.e., it's a leaf from a
            # previous selection), return it for expansion.
            if current_node.policy_value is None:
                return current_node

            # If there are no possible actions from this node, it's a terminal state.
            if not current_node._unexpanded_priors and not current_node.children:
                return current_node

            # --- Select the best action using PUCT score ---
            # This considers both existing children and unexpanded actions.
            best_action, best_score = None, -float('inf')

            # Combine instantiated children and unexpanded priors into one list for scoring
            # Children (already visited)
            all_actions = list(current_node.children.keys())
            # Unexpanded actions (not yet visited)
            if current_node._unexpanded_priors:
                all_actions.extend(list(current_node._unexpanded_priors.keys()))

            if not all_actions:
                 return current_node # Terminal node if no actions possible

            parent_visits_sqrt = math.sqrt(max(1, current_node.visits))

            # Vectorized PUCT calculation
            try:
                num_actions = len(all_actions)
                q_values = torch.zeros(num_actions, dtype=torch.float32)
                child_visits = torch.zeros(num_actions, dtype=torch.float32)
                priors = torch.zeros(num_actions, dtype=torch.float32)

                for i, action in enumerate(all_actions):
                    if action in current_node.children:
                        child = current_node.children[action]
                        q_values[i] = child.value / child.visits if child.visits > 0 else 0.0
                        child_visits[i] = child.visits
                        priors[i] = child.prior_prob
                    else: # Unexpanded action
                        q_values[i] = 0.0 # No value estimate yet
                        child_visits[i] = 0.0
                        priors[i] = current_node._unexpanded_priors[action]

                # PUCT formula: Q + c * P * sqrt(N_parent) / (1 + N_child)
                u_values = self.exploration_weight * priors * (parent_visits_sqrt / (1 + child_visits))
                scores = q_values + u_values

                best_idx = torch.argmax(scores).item()
                best_action = all_actions[best_idx]

            except Exception: # Fallback to iterative calculation
                for action in all_actions:
                    if action in current_node.children:
                        child = current_node.children[action]
                        if child.visits == 0:
                            q_value = 0.0
                        else:
                            q_value = child.value / child.visits
                        prior = child.prior_prob
                        child_visit_count = child.visits
                    else: # Unexpanded action
                        q_value = 0.0  # No value estimate yet
                        prior = current_node._unexpanded_priors[action]
                        child_visit_count = 0

                    # PUCT formula
                    uct_score = q_value + self.exploration_weight * prior * (parent_visits_sqrt / (1 + child_visit_count))

                    if uct_score > best_score:
                        best_score = uct_score
                        best_action = action

            # --- Lazy Instantiation or Traversal ---
            if best_action in current_node.children:
                # Action has been explored before, descend to the child node
                current_node = current_node.children[best_action]
            else:
                # This is the first time this action is selected.
                # Create the child node (lazy instantiation).
                new_architecture = current_node._copy_architecture()
                success = self.action_space.apply_action(new_architecture, best_action)

                if success:
                    # Create and link the new child node
                    child_node = NeuralMCTSNode(
                        new_architecture,
                        parent=current_node,
                        action=best_action
                    )
                    # Set its prior probability from the parent's stored policy
                    child_node.prior_prob = current_node._unexpanded_priors[best_action]

                    # Add to parent's children and remove from unexpanded list
                    current_node.children[best_action] = child_node
                    del current_node._unexpanded_priors[best_action]

                    # This new node is the leaf for the current simulation
                    return child_node
                else:
                    # If action application fails, remove it from consideration and retry selection
                    del current_node._unexpanded_priors[best_action]
                    if not current_node._unexpanded_priors and not current_node.children:
                        return current_node # No more valid moves
                    continue # Retry selection without the failed action
        

    def _expand(self, node: NeuralMCTSNode) -> float:
        """Expand a leaf node by evaluating it with the policy-value network.

        This method performs the "expansion" step in the MCTS search. It does
        not create child nodes directly but instead computes the policy (prior
        probabilities) for all valid actions from the given node. These priors
        are stored in the node and used by the `_select` method to create child
        nodes lazily.

        Args:
            node: The leaf node to expand.

        Returns:
            The value of the node as predicted by the neural network.
        """
        # Profiling: per-call breakdown
        expand_profile = {
            'start': time.time(),
            'eval': 0.0,
            'valid_actions': 0.0,
            'mask_prep': 0.0,
            'policy_transfer': 0.0,
            'priors_compute': 0.0,
            'total': 0.0
        }

        # If node has policy value, it's already expanded. Return its value.
        if node.policy_value is not None:
            value_output = node.policy_value.get('value')
            return float(value_output.item()) if isinstance(value_output, torch.Tensor) else 0.0

        # --- 1. Evaluate the node with the neural network ---
        t0 = time.time()
        with torch.no_grad():
            graph_data = self._prepare_graph_data(node.architecture)
            graph_data['node_features'] = graph_data['node_features'].to(self.device)
            graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
            graph_data['layer_positions'] = graph_data['layer_positions'].to(self.device)
            policy_value = self.policy_value_net(graph_data, phase=self.current_cycle.get_phase_value())
            node.policy_value = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in policy_value.items()}
        expand_profile['eval'] = time.time() - t0

        # --- 2. Get valid actions and compute priors ---
        current_phase_token = self.current_cycle.cycle_id

        # Get (and cache on node) the action masks for the current phase
        t0 = time.time()
        if node._cached_masks is None or node._cache_phase_token != current_phase_token:
            node._cached_masks = self.action_manager.get_action_masks(
                node.architecture, phase=self.current_cycle.get_phase_value()
            )
            node._cache_phase_token = current_phase_token
        masks = node._cached_masks
        expand_profile['mask_prep'] = time.time() - t0

        # Get all valid actions for the current state and phase
        t0 = time.time()
        valid_actions = self.action_space.get_valid_actions(
            node.architecture, evolutionary_cycle=self.current_cycle
        )
        expand_profile['valid_actions'] = time.time() - t0

        if not valid_actions:
            node._unexpanded_priors = {}
            value_output = node.policy_value.get('value')
            return float(value_output.item()) if isinstance(value_output, torch.Tensor) else 0.0

        # Create a transient on-device view of policy outputs for prior computation
        policy_on_device = {}
        t0 = time.time()
        for k, v in node.policy_value.items():
            if torch.is_tensor(v):
                policy_on_device[k] = v.to(self.device)
        expand_profile['policy_transfer'] = time.time() - t0

        # Compute priors for all valid actions
        t0 = time.time()
        try:
            priors_tensor = self.action_manager._compute_priors_vectorized(
                policy_on_device, valid_actions, masks
            )
            priors_np = priors_tensor.detach().cpu().numpy()
            # Filter out actions with zero or near-zero probability
            priors_map = {
                action: float(prob)
                for action, prob in zip(valid_actions, priors_np)
                if float(prob) > 1e-8
            }
        except Exception:
            traceback.print_exc()
            priors_map = {} # Fallback to empty priors on failure
        expand_profile['priors_compute'] = time.time() - t0

        # Cache the computed priors on the node for lazy instantiation in `_select`
        node._unexpanded_priors = priors_map

        # Finalize profiling and return the node's value
        expand_profile['total'] = time.time() - expand_profile['start']
        try:
            self._expand_profiles.append(expand_profile)
            for k, v in expand_profile.items():
                if k != 'start':
                    self._expand_profiler[k] += float(v or 0.0)
        except Exception:
            pass

        value_output = node.policy_value.get('value')
        return float(value_output.item()) if isinstance(value_output, torch.Tensor) else 0.0

    def _apply_dirichlet_noise_to_root(self, root: NeuralMCTSNode, epsilon: float = 0.25, alpha: float = 0.3):
        """Apply Dirichlet noise to the root node's action priors for exploration.

        This method modifies the `_unexpanded_priors` of the root node in-place.
        It mixes the original priors (from the policy network) with random noise
        sampled from a Dirichlet distribution.

        Args:
            root: The root node of the MCTS tree.
            epsilon: The weight of the noise (e.g., 0.25 means 25% noise).
            alpha: The concentration parameter for the Dirichlet distribution.
        """
        if not root._unexpanded_priors:
            return

        actions = list(root._unexpanded_priors.keys())
        original_priors = torch.tensor([root._unexpanded_priors[a] for a in actions], dtype=torch.float32)

        # Sample Dirichlet noise
        noise = torch.distributions.Dirichlet(torch.full((len(actions),), alpha)).sample()

        # Mix the priors with the noise
        noisy_priors = (1 - epsilon) * original_priors + epsilon * noise

        # Update the node's priors with the noisy version
        for i, action in enumerate(actions):
            root._unexpanded_priors[action] = float(noisy_priors[i])

    def get_visit_distribution(self, node: 'NeuralMCTSNode', temperature: float = 1.0) -> torch.Tensor:
        """Extracts the visit count distribution from a node's children.

        This distribution, often referred to as the improved policy, is used as
        a target for training the policy network. It reflects the knowledge
        gained during the MCTS search.

        Args:
            node (NeuralMCTSNode): The node from which to extract the
                distribution.
            temperature (float): A parameter to control the softness of the
                distribution.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over
            actions.
        """
        if not node.children or len(node.children) == 0:
            # No children explored, return empty tensor
            return torch.tensor([], dtype=torch.float32)
        
        # Extract visit counts for each child
        visit_counts = torch.tensor([child.visits for child in node.children], 
                                   dtype=torch.float32)
        
        # Apply temperature scaling
        if temperature > 0:
            # Normalize visits: π(a) = visits^(1/T) / sum(visits^(1/T))
            scaled_visits = torch.pow(visit_counts, 1.0 / temperature)
            visit_probs = scaled_visits / scaled_visits.sum()
        else:
            # Zero temperature: select highest visit count (greedy)
            visit_probs = torch.zeros_like(visit_counts)
            visit_probs[visit_counts.argmax()] = 1.0
        
        return visit_probs

    def dump_expand_profile(self, last_n: int = 10, show_cumulative: bool = True, reset: bool = False):
        """Print a human-readable summary of expand timing profiling.

        - last_n: show the last N per-expand breakdowns (most recent first)
        - show_cumulative: print cumulative totals and per-call averages
        - reset: clear collected profiling data after printing
        """
        # Defensive: ensure attributes exist
        if not hasattr(self, '_expand_profiles') or not hasattr(self, '_expand_profiler'):
            print("No expand profiling data available.")
            return

        profiles = list(self._expand_profiles)
        total_calls = len(profiles)

        print("\n=== NeuralMCTS Expand Profiling Summary ===")
        print(f"Total tracked expand calls: {total_calls}")

        if total_calls == 0:
            print("(no expand calls recorded)")
        else:
            # Print last N calls
            n = min(last_n, total_calls)
            print(f"\nLast {n} expand calls (most recent first):")
            for i, prof in enumerate(reversed(profiles[-n:])):
                idx = total_calls - i
                total = prof.get('total', 0.0)
                eval_t = prof.get('eval', 0.0)
                priors_t = prof.get('priors_compute', 0.0)
                sel_t = prof.get('selection', 0.0)
                child_t = prof.get('child_eval', 0.0)
                apply_t = prof.get('apply_action', 0.0)
                mask_t = prof.get('mask_prep', 0.0)
                policy_t = prof.get('policy_transfer', 0.0)
                update_t = prof.get('update_parent', 0.0)
                valid_t = prof.get('valid_actions', 0.0)
                print(f"#{idx}: total={total:.6f}s eval={eval_t:.6f}s priors={priors_t:.6f}s sel={sel_t:.6f}s child={child_t:.6f}s apply={apply_t:.6f}s mask={mask_t:.6f}s policy={policy_t:.6f}s update={update_t:.6f}s valid_actions={valid_t:.6f}s")

        if show_cumulative:
            print("\nCumulative totals and averages:")
            cumul = self._expand_profiler
            calls = total_calls if total_calls > 0 else 1
            # Keys to display in priority order
            keys = ['eval', 'valid_actions', 'mask_prep', 'policy_transfer', 'priors_compute', 'selection', 'apply_action', 'child_eval', 'update_parent', 'total']
            for k in keys:
                v = float(cumul.get(k, 0.0))
                print(f"{k}: total={v:.6f}s avg={(v / calls):.6f}s")

        print("=== End Expand Profiling ===\n")

        if reset:
            try:
                self._expand_profiles.clear()
                for k in list(self._expand_profiler.keys()):
                    self._expand_profiler[k] = 0.0
            except Exception:
                pass

# Helper to safe-index logits (handle both 1-D and 2-D)
def safe_squeeze(tensor):
    return tensor.squeeze(0) if tensor.dim() > 1 and tensor.shape[0] == 1 else tensor

