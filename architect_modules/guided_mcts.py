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
        super().__init__(architecture, parent, action)
        self.policy_value = policy_value  # Neural network predictions
        self.prior_prob = 0.0  # Prior probability from policy network
        self.curriculum = curriculum
        # Cache valid actions for this node to check if fully expanded
        self._valid_actions_phase_token = None
        # Incremental dict of expanded actions for O(1) filtering (vectorized action filtering)
        self._unexpanded_actions_dict = {}
        # Cache action masks to avoid recomputation across _compute_action_prior calls
        self._priors_cache = None # torch.Tensor of priors
        self._priors_phase_token = None
        # Track whether Dirichlet noise has been applied for a given phase (root-only)
        self._dirichlet_applied_phase_token = None
    
    def is_fully_expanded(self) -> bool:
        """Checks if all valid actions from this node have been explored.

        A node is considered fully expanded if all possible actions from its
        state have been added as child nodes.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return (self._unexpanded_actions_dict is not None and
                len(self._unexpanded_actions_dict) == 0)
    
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
        # Vectorized selection: if any child has zero visits, prefer first such child
        # (keeps original behavior which returns +inf for zero-visit children).
        try:
            visits = torch.tensor([float(child.visits) for child in self.children], dtype=torch.float32)
            zero_mask = visits == 0.0
            if zero_mask.any():
                zero_idxs = torch.nonzero(zero_mask).squeeze(1).tolist()
                # choose among zero-visit children the one with highest prior
                priors = [self.children[i].prior_prob for i in zero_idxs]
                best = zero_idxs[int(torch.argmax(torch.tensor(priors)).item())]
                return self.children[best]

            # Q: average value per child
            values = torch.tensor([float(child.value) for child in self.children], dtype=torch.float32)
            q_values = values / visits

            # U: exploration term (PUCT)
            priors = torch.tensor([float(child.prior_prob) for child in self.children], dtype=torch.float32)
            parent_visits = float(self.visits) if self.visits > 0 else 1.0
            u_values = exploration_weight * priors * torch.sqrt(torch.tensor(parent_visits)) / (1.0 + visits)

            scores = q_values + u_values
            best_idx = int(torch.argmax(scores).item())
            return self.children[best_idx]
        except Exception:
            # Fallback to original Python implementation on any failure
            def puct_score(child: 'NeuralMCTSNode') -> float:
                if child.visits == 0:
                    return float('inf')

                q_value = child.value / child.visits
                u_value = exploration_weight * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
                return q_value + u_value

            return max(self.children, key=puct_score)

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
            # invalidate caches keyed by token so new search_token forces recompute
            root._valid_actions_cache = None
            root._valid_actions_phase_token = None
            root._priors_cache = None
            root._priors_phase_token = None
            root._dirichlet_applied_phase_token = None
            if hasattr(root, '_cached_masks'):
                root._cached_masks = None
            # optionally reset policy so network is re-evaluated for the new semantic phase
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
            # Get neural network predictions for root
            with torch.no_grad():
                graph_data = self._prepare_graph_data(initial_architecture)
                graph_data['node_features'] = graph_data['node_features'].to(self.device)
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
                graph_data['layer_positions'] = graph_data['layer_positions'].to(self.device)
                policy_value = self.policy_value_net(graph_data, phase=phase_value)
                root.policy_value = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in policy_value.items()}

            # Early stopping tracking
            best_values = deque(maxlen=self.early_stopping_patience)
            no_improvement_count = 0
            initial_visits = root.visits  # Track starting visits for reused trees
            
            # Add Dirichlet noise to root for exploration (AlphaZero-style)
            # This ensures we explore even when policy network has strong (but potentially wrong) priors
            self._apply_dirichlet_noise_to_root(root.policy_value, num_actions=
                                                len(self.action_space.get_valid_actions(root.architecture,
                                                                    evolutionary_cycle=self.current_cycle)))

            print("Starting MCTS search...")
            select_duration = 0.0
            expand_duration = 0.0
            backup_duration = 0.0
            final_action_duration = 0.0
            for i in range(iterations):
                select_time = time.time()
                # ===== STEP 1: SELECT =====
                # Traverse tree using PUCT until reaching a leaf node
                node = self._select(root)
                select_duration += time.time() - select_time

                expand_time = time.time()
                # ===== STEP 2: EXPAND & EVALUATE =====
                # Expand one new child and get its value in one step (no redundant re-evaluation)
                expanded_node, value = self._expand(node)
                expand_duration += time.time() - expand_time

                # If expansion produced a child, record its evaluation in the local cycle
                if expanded_node is not None:
                    self.current_cycle.add_evaluation(value)
                    if self.current_cycle.should_advance(num_neurons=expanded_node.architecture.num_neurons(),
                                                        num_connections=expanded_node.architecture.num_connections()):
                        self.current_cycle.advance_phase()

                backup_time = time.time()
                # ===== STEP 4: BACKUP =====
                # Propagate value up the tree
                self._backpropagate(expanded_node, value)
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
                episode_legal_children = [c for c in root.children if c.action in episode_valid_set]
            except Exception:
                # Fallback to considering all children if mask retrieval fails
                episode_legal_children = list(root.children)

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

        visit_counts = torch.tensor([child.visits for child in root.children])

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
        return root.children[selected_idx]
    
    def _select(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Traverse the tree from the given node to a leaf using PUCT selection."""
        current_node = node
        # descend while the current node is fully expanded AND has children to descend into
        while current_node.children and current_node.is_fully_expanded():
            current_node = current_node.best_child(self.exploration_weight)
            if current_node is None:
                break
        return current_node
        

    def _expand(self, node: NeuralMCTSNode) -> tuple:
        """Expand the given node by adding one new child using neural-guided action selection.
        """
        # Profiling: per-call breakdown
        expand_profile = {
            'start': time.time(),
            'eval': 0.0,
            'valid_actions': 0.0,
            'mask_prep': 0.0,
            'policy_transfer': 0.0,
            'priors_compute': 0.0,
            'selection': 0.0,
            'apply_action': 0.0,
            'child_eval': 0.0,
            'update_parent': 0.0,
            'total': 0.0
        }
        # Cache valid actions for this node. Invalidate and recompute when the
        # search phase changes because valid action sets (masks/rules) may
        # differ between phases.
        current_phase_token = self.current_cycle.cycle_id
        t0 = time.time()
        if (node._unexpanded_actions_dict is None or
            getattr(node, '_valid_actions_phase_token', None) != current_phase_token):
            all_valid_actions = self.action_space.get_valid_actions(
                node.architecture, evolutionary_cycle=self.current_cycle
            )
            node._unexpanded_actions_dict = {a: i for i, a in enumerate(all_valid_actions)}
            # Track which phase the valid-actions cache belongs to separately
            # from `_priors_phase` which is used for priors caching.
            node._valid_actions_phase_token = current_phase_token
            # Invalidate priors cache when valid actions change
            node._priors_cache = None
            node._priors_phase_token = None
        expand_profile['valid_actions'] = time.time() - t0

        # Vectorized action filtering: use incremental set to filter in O(n) instead of O(n²)
        valid_actions = list(node._unexpanded_actions_dict.keys())
        if len(valid_actions) == 0:
            # Extract value from new network output signature
            value_output = node.policy_value.get('value')
            if isinstance(value_output, torch.Tensor):
                value = value_output.item()
            else:
                value = 0.0  # Fallback
            return (node, value)  # No unexpanded actions available

        masks = self.action_manager.get_action_masks(node.architecture, phase=self.current_cycle.get_phase_value())

        # Ensure masks and transient policy tensors are on the model/device
        t0 = time.time()
        for k in list(masks.keys()):
            if k != 'tensor_size' and torch.is_tensor(masks[k]):
                masks[k] = masks[k].to(self.device)
        expand_profile['mask_prep'] = time.time() - t0

        # Create a transient on-device view of the cached policy outputs so
        # policy-conditioned heads (which live on the model device) receive
        # inputs on the same device during priors computation.
        policy_on_device = {}
        t0 = time.time()
        for k, v in node.policy_value.items():
            if torch.is_tensor(v):
                try:
                    policy_on_device[k] = v.to(self.device)
                except Exception:
                    policy_on_device[k] = v
            else:
                policy_on_device[k] = v
        expand_profile['policy_transfer'] = time.time() - t0

        # Compose priors for all valid actions
        if (node._priors_cache is not None and
            node._priors_phase_token == current_phase_token):
            # Reuse cached priors if phase hasn't changed
            t0 = time.time()
            P_tensor = node._priors_cache.to(self.device)
            expand_profile['priors_compute'] = time.time() - t0
        else:
            # Prefer the vectorized prior computation for speed and correctness
            try:
                # Compute full priors for all valid actions
                t0 = time.time()
                priors_tensor = self.action_manager._compute_priors_vectorized(policy_on_device, valid_actions, masks)
                P_tensor = priors_tensor.detach().clamp(min=1e-12)  # [num_valid_actions]
                expand_profile['priors_compute'] = time.time() - t0
            except Exception:
                traceback.print_exc()
                raise RuntimeError("Vectorized prior computation failed during expansion.")        

            # cache priors and phase version
            node._priors_cache = P_tensor.cpu() # store CPU-detached copy
            node._priors_phase_token = current_phase_token

        # ----- Select the unexpanded action with highest UCT value -----
        # UCT: Q + c * P * sqrt(N) / (1 + n)
        selected_action = None
        c = self.exploration_weight  # exploration coefficient

        # AlphaZero PUCT: U = c_puct * P * sqrt(sum_parent_visits) / (1 + N_child)
        # For unexpanded actions N_child == 0 and Q is unknown (treated as 0)
        parent_visits = max(1, node.visits)
        try:
            t0 = time.time()   
            # For unexpanded actions, child visit counts are zero -> denominator = 1
            u_tensor = c * P_tensor * torch.sqrt(torch.tensor(float(parent_visits), device=self.device))  # Use vectorized U computation on device

            # Select index of maximum U value
            sel_idx = int(torch.argmax(u_tensor).item())
            selected_action = valid_actions[sel_idx]
            expand_profile['selection'] = time.time() - t0
        except Exception:
            traceback.print_exc()
            raise RuntimeError("Vectorized UCT selection failed during expansion.")

        # ----- Apply selected action (lazy expansion) -----
        new_architecture = node._copy_architecture()
        t0 = time.time()
        success = self.action_space.apply_action(new_architecture, selected_action)
        expand_profile['apply_action'] = time.time() - t0
        if not success:
            del node._unexpanded_actions_dict[selected_action]
            # Extract value from new network output signature
            value_output = node.policy_value.get('value')
            if isinstance(value_output, torch.Tensor):
                value = value_output.item()
            else:
                raise RuntimeError("Node policy_value missing 'value' tensor during failed expansion.")
            return node, value

        # Create child node and evaluate it, caching CPU-detached policy outputs
        child_node = NeuralMCTSNode(new_architecture, parent=node, action=selected_action)
        t0 = time.time()
        with torch.no_grad():
            child_graph = self._prepare_graph_data(new_architecture)
            child_graph['node_features'] = child_graph['node_features'].to(self.device)
            child_graph['edge_index'] = child_graph['edge_index'].to(self.device)
            child_graph['layer_positions'] = child_graph['layer_positions'].to(self.device)
            child_policy = self.policy_value_net(child_graph, phase=self.current_cycle.get_phase_value())
        
        # Store CPU-detached copy to avoid retaining GPU memory
        child_node.policy_value = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in child_policy.items()}
        child_node.prior_prob = node._priors_cache[node._unexpanded_actions_dict[selected_action]]
        expand_profile['child_eval'] = time.time() - t0

        # Update parent
        t0 = time.time()
        del node._unexpanded_actions_dict[selected_action]
        node.children.append(child_node)
        expand_profile['update_parent'] = time.time() - t0

        # Extract value from new network output signature for return
        value_output = child_node.policy_value.get('value')
        if isinstance(value_output, torch.Tensor):
            value = value_output.item()
        else:
            raise RuntimeError("Node policy_value missing 'value' tensor during expansion.")
        # Finalize profiling for this expand call
        expand_profile['total'] = time.time() - expand_profile['start']
        # Store per-call breakdown and update cumulative totals
        try:
            self._expand_profiles.append(expand_profile)
            for k, v in expand_profile.items():
                if k == 'start':
                    continue
                self._expand_profiler[k] += float(v or 0.0)
        except Exception:
            pass

        return child_node, value

    def _apply_dirichlet_noise_to_root(self, policy_output: Dict, num_actions: int, 
                                         epsilon: float = 0.25, alpha: float = 0.3):
        """Apply Dirichlet noise to root policy logits for exploration (AlphaZero-style).
        
        This mixes the prior policy with random noise:
            mixed_prior(a) = (1 - epsilon) * prior(a) + epsilon * noise(a)
        
        Args:
            policy_output: Dict with action_type logits to modify in-place
            num_actions: Number of valid actions for noise scaling
            epsilon: Weight of noise (0.25 = 75% prior, 25% noise)
            alpha: Dirichlet concentration parameter (lower = more dispersed)
        """
        # Handle new network output signature - extract action_type logits
        action_type_logits = policy_output['action_type']
        # Ensure 1-D probabilities vector
        probs = F.softmax(action_type_logits, dim=-1)
        if probs.dim() > 1 and probs.shape[0] == 1:
            probs = probs.squeeze(0)

        device = probs.device
        num = probs.shape[-1]

        # Sample Dirichlet noise on the same device
        if num_actions <= 0:
            num_actions = num
        concentration = torch.full((num,), float(alpha), device=device)
        noise_dist = torch.distributions.Dirichlet(concentration)
        noise = noise_dist.sample()

        # Mix probabilities with noise and convert back to logit space
        mixed = (1.0 - epsilon) * probs + epsilon * noise
        mixed = torch.clamp(mixed, 1e-8, 1.0)
        mixed_logits = torch.log(mixed)

        # Restore batch dim if needed
        if action_type_logits.dim() > 1 and action_type_logits.shape[0] == 1:
            policy_output['action_type'] = mixed_logits.unsqueeze(0)
        else:
            policy_output['action_type'] = mixed_logits

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

    def dump_expand_profile(self, last_n: int = 10, show_cumulative: bool = True, reset: bool = True):
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

