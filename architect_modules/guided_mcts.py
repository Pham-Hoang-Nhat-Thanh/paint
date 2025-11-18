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
from collections import deque

class NeuralMCTSNode(MCTSNode):
    """MCTS node enhanced with neural network predictions.

    This class extends the basic MCTS node with attributes to store policy-value
    predictions, prior probabilities, and other data for neural-guided search.

    Attributes:
        policy_value (Dict): Predictions from the policy-value network.
        prior_prob (float): Prior probability from the policy network.
        curriculum: The curriculum used for the search.
        _valid_actions_cache (List[Action]): A cache for valid actions.
        _valid_actions_phase_token: Token for the valid actions cache phase.
        _unexpanded_actions_set (set): A set of unexpanded actions.
        _priors_cache (Dict): A cache for action priors.
        _priors_phase_token: Token for the priors cache phase.
        _dirichlet_applied_phase_token: Token for applied Dirichlet noise.
    """

    def __init__(self, architecture: NeuralArchitecture, policy_value: Dict = None,
                 parent=None, action: Action = None, curriculum=None):
        """Initializes the NeuralMCTSNode.

        Args:
            architecture (NeuralArchitecture): The neural architecture for this node.
            policy_value (Dict, optional): Predictions from the policy-value
                network. Defaults to None.
            parent (NeuralMCTSNode, optional): The parent node. Defaults to None.
            action (Action, optional): The action that led to this node.
                Defaults to None.
            curriculum (optional): The curriculum used for the search.
                Defaults to None.
        """
        super().__init__(architecture, parent, action)
        self.policy_value = policy_value  # Neural network predictions
        self.prior_prob = 0.0  # Prior probability from policy network
        self.curriculum = curriculum
        # Cache valid actions for this node to check if fully expanded
        self._valid_actions_cache = None
        self._valid_actions_phase_token = None
        # Incremental set of expanded actions for O(1) filtering (vectorized action filtering)
        self._unexpanded_actions_set = set()
        # Cache action masks to avoid recomputation across _compute_action_prior calls
        self._priors_cache = None
        self._priors_phase_token = None
        # Track whether Dirichlet noise has been applied for a given phase (root-only)
        self._dirichlet_applied_phase_token = None
    
    def is_fully_expanded(self) -> bool:
        """Checks if all valid actions have been expanded as children.
        
        Since NeuralMCTS doesn't use an untried_actions list, this compares
        the expanded actions (children) with all valid actions for this state.

        Returns:
            bool: True if all valid actions have been expanded, False otherwise.
        """
        return (self._valid_actions_cache is not None and self._unexpanded_actions_set is not None and
                len(self._unexpanded_actions_set) == 0)
    
    def best_child(self, exploration_weight: float = 1.0) -> 'NeuralMCTSNode':
        """Selects the best child using the PUCT formula (AlphaZero style).

        Args:
            exploration_weight (float): The exploration weight.

        Returns:
            NeuralMCTSNode: The best child node.
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
            u_values = exploration_weight * priors * (math.sqrt(parent_visits) / (1.0 + visits))

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
    """MCTS enhanced with neural network guidance.

    This class implements the Monte Carlo Tree Search algorithm guided by a
    policy-value network, in the style of AlphaZero.

    Attributes:
        policy_value_net (UnifiedPolicyValueNetwork): The policy-value network.
        device (str): The device to run on ('cpu' or 'cuda').
        iso_weight (float): The weight for isolation penalty.
        comp_weight (float): The weight for complexity penalty.
        early_stopping_patience (int): Patience for early stopping.
        early_stopping_min_delta (float): Minimum delta for early stopping.
        evaluation_cache (Dict): A cache for evaluations.
        current_cycle (EvolutionaryCycle): The current evolutionary cycle.
        action_manager (ActionManager): The action manager.
        max_children (int): The maximum number of children per node.
    """

    def __init__(self, action_space: ActionSpace, policy_value_net: UnifiedPolicyValueNetwork,
                 device: str = 'cpu', exploration_weight: float = 1.0,
                 iso_weight: float = 0.01, comp_weight: float = 0.0, 
                 early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.001, max_children: int = 50, max_neurons: int = 1000):
        """Initializes the NeuralMCTS.

        Args:
            action_space (ActionSpace): The action space.
            policy_value_net (UnifiedPolicyValueNetwork): The policy-value network.
            device (str): The device to run on. Defaults to 'cpu'.
            exploration_weight (float): The exploration weight. Defaults to 1.0.
            iso_weight (float): The weight for isolation penalty. Defaults to 0.01.
            comp_weight (float): The weight for complexity penalty. Defaults to 0.0.
            early_stopping_patience (int): Patience for early stopping.
                Defaults to 5.
            early_stopping_min_delta (float): Minimum delta for early stopping.
                Defaults to 0.001.
            max_children (int): Maximum number of children per node. Defaults to 50.
            max_neurons (int): Maximum number of neurons. Defaults to 1000.
        """
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

    def cleanup(self):
        """Cleans up resources used by NeuralMCTS."""
        # Clear evaluation cache to free memory
        self.evaluation_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()

    def _prepare_graph_data(self, architecture: NeuralArchitecture) -> Dict:
        """Converts architecture to graph data for the neural network.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            Dict: The graph data.
        """
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
        """Batches graph data dicts for joint evaluation.
        
        Follows PyG batching: concatenate graphs into a single disconnected
        graph with a batch tensor. Enables a single policy_value_net forward
        pass for all graphs.
        
        Args:
            parent_graph (Dict): The graph data for the parent.
            child_graphs (Union[Dict, List[Dict]]): The graph data for the
                child or children.
        
        Returns:
            Dict: The batched graph data.
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
        """Runs a standard neural-guided MCTS search (AlphaZero style).
        
        Each iteration consists of:
        1. SELECT: Traverse the tree to a leaf using the PUCT formula.
        2. EXPAND: Add one new child to the leaf and evaluate it.
        3. BACKUP: Propagate the result up the tree.
        
        Args:
            initial_architecture (NeuralArchitecture): The starting architecture.
            iterations (int): The number of MCTS iterations to run.
            temperature (float): The temperature for action selection.
            reuse_root (NeuralMCTSNode, optional): An existing tree root to
                continue the search from. Defaults to None.

        Returns:
            NeuralMCTSNode: The best node found during the search.
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
            for i in range(iterations):
                # ===== STEP 1: SELECT =====
                # Traverse tree using PUCT until reaching a leaf node
                node = self._select(root)

                # ===== STEP 2: EXPAND & EVALUATE =====
                # Expand one new child and get its value in one step (no redundant re-evaluation)
                expanded_node, value = self._expand(node)

                # If expansion produced a child, record its evaluation in the local cycle
                if expanded_node is not None:
                    self.current_cycle.add_evaluation(value)
                    if self.current_cycle.should_advance():
                        self.current_cycle.advance_phase()

                # ===== STEP 4: BACKUP =====
                # Propagate value up the tree
                self._backpropagate(expanded_node, value)

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
    
    def _select_final_action(self, root: NeuralMCTSNode, temperature: float) -> NeuralMCTSNode:
        """Selects the final action based on visit counts.

        Args:
            root (NeuralMCTSNode): The root node of the search tree.
            temperature (float): The temperature for proportional selection.

        Returns:
            NeuralMCTSNode: The selected final node.
        """
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
        """Traverses the tree from the given node to a leaf using PUCT selection.

        Args:
            node (NeuralMCTSNode): The node to start the selection from.

        Returns:
            NeuralMCTSNode: The selected leaf node.
        """
        current_node = node
        # descend while the current node is fully expanded AND has children to descend into
        while current_node.children and current_node.is_fully_expanded():
            current_node = current_node.best_child(self.exploration_weight)
            if current_node is None:
                break
        return current_node
        

    def _expand(self, node: NeuralMCTSNode) -> tuple:
        """Expands the given node by adding one new child.

        Args:
            node (NeuralMCTSNode): The node to expand.

        Returns:
            tuple: A tuple containing the expanded node and its value.
        """
        import time
        expand_start_time = time.time()
        _mcts_policy_eval_time = None
        _mcts_priors_time = None
        _mcts_child_eval_time = None
        # Ensure parent policy_value is available
        if node.policy_value is None:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                # Move inputs to device for forward, then cache CPU-detached outputs
                graph_data['node_features'] = graph_data['node_features'].to(self.device)
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
                graph_data['layer_positions'] = graph_data['layer_positions'].to(self.device)
                t_policy = time.time()
                policy_value = self.policy_value_net(graph_data, phase=self.current_cycle.get_phase_value())
                _mcts_policy_eval_time = time.time() - t_policy
                # Handle new network output signature - store all components
                node.policy_value = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in policy_value.items()}
                #print(f"[MCTS::_expand] policy_eval_time={_mcts_policy_eval_time:.4f}s node_id={id(node)}")

        # Cache valid actions for this node. Invalidate and recompute when the
        # search phase changes because valid action sets (masks/rules) may
        # differ between phases.
        current_phase_token = self.current_cycle.cycle_id
        if (node._valid_actions_cache is None or
            getattr(node, '_valid_actions_phase_token', None) != current_phase_token):
            all_valid_actions = self.action_space.get_valid_actions(
                node.architecture, evolutionary_cycle=self.current_cycle
            )
            node._valid_actions_cache = all_valid_actions
            node._unexpanded_actions_set = set(all_valid_actions)
            # Track which phase the valid-actions cache belongs to separately
            # from `_priors_phase` which is used for priors caching.
            node._valid_actions_phase_token = current_phase_token
        all_valid_actions = node._valid_actions_cache

        # Vectorized action filtering: use incremental set to filter in O(n) instead of O(n²)
        valid_actions = list(node._unexpanded_actions_set)
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
        model_device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        for k in list(masks.keys()):
            if k != 'tensor_size' and torch.is_tensor(masks[k]):
                masks[k] = masks[k].to(model_device)

        # Create a transient on-device view of the cached policy outputs so
        # policy-conditioned heads (which live on the model device) receive
        # inputs on the same device during priors computation.
        policy_on_device = {}
        for k, v in node.policy_value.items():
            if torch.is_tensor(v):
                try:
                    policy_on_device[k] = v.to(model_device)
                except Exception:
                    policy_on_device[k] = v
            else:
                policy_on_device[k] = v

        # Compose priors for all top-K valid actions
        if (node._priors_cache is not None and
            node._priors_phase_token == current_phase_token):
            # Reuse cached priors if phase hasn't changed
            priors_map = node._priors_cache
        else:
            # Prefer the vectorized prior computation for speed and correctness
            try:
                # Compute full priors for all valid actions if action space is small
                t_priors = time.time()
                priors_tensor = self.action_manager._compute_priors_vectorized(policy_on_device, valid_actions, masks)
                # Move to CPU numpy for mapping; clamp minimum probability for safety
                priors_np = priors_tensor.detach().cpu().numpy()
                priors_map = {a: float(max(float(p), 1e-12)) for a, p in zip(valid_actions, priors_np)}
                _mcts_priors_time = time.time() - t_priors
                #print(f"[MCTS::_expand] priors_compute_time={_mcts_priors_time:.4f}s node_id={id(node)} num_actions={len(valid_actions)}")
            except Exception:
                traceback.print_exc()
                raise RuntimeError("Vectorized prior computation failed during expansion.")        

            # cache priors and phase version
            node._priors_cache = priors_map
            node._priors_phase_token = current_phase_token

        # ----- Select the unexpanded action with highest UCT value -----
        # UCT: Q + c * P * sqrt(N) / (1 + n)
        best_uct = -float('inf')
        selected_action = None
        c = self.exploration_weight  # exploration coefficient

        # AlphaZero PUCT: U = c_puct * P * sqrt(sum_parent_visits) / (1 + N_child)
        # For unexpanded actions N_child == 0 and Q is unknown (treated as 0)
        parent_visits = max(1, node.visits)
        try:
            # Build prior tensor in the same order as `valid_actions`
            priors_list = [priors_map.get(a, node._priors_cache.get(a) if node._priors_cache is not None else 1e-12) for a in valid_actions]
            P_tensor = torch.tensor(priors_list, dtype=torch.float32)

            # For unexpanded actions, child visit counts are zero -> denominator = 1
            denom = 1.0
            u_tensor = c * P_tensor * (math.sqrt(parent_visits) / denom)

            # Select index of maximum U value
            sel_idx = int(torch.argmax(u_tensor).item())
            selected_action = valid_actions[sel_idx]
            best_uct = float(u_tensor[sel_idx].item())
        except Exception:
            # Fallback to safe Python loop if vectorized path fails
            selected_action = None
            best_uct = -float('inf')
            for a in valid_actions:
                Q = 0.0  # exploitation term unknown for unexpanded children
                N_child = 0  # unexpanded -> zero visits
                P = priors_map.get(a, node._priors_cache.get(a) if node._priors_cache is not None else 0.0)
                uct_score = Q + c * P * (math.sqrt(parent_visits) / (1 + N_child))
                if uct_score > best_uct:
                    best_uct = uct_score
                    selected_action = a

        # ----- Apply selected action (lazy expansion) -----
        new_architecture = node._copy_architecture()
        success = self.action_space.apply_action(new_architecture, selected_action)
        if not success:
            node._unexpanded_actions_set.remove(selected_action)
            # Extract value from new network output signature
            value_output = node.policy_value.get('value')
            if isinstance(value_output, torch.Tensor):
                value = value_output.item()
            else:
                raise RuntimeError("Node policy_value missing 'value' tensor during failed expansion.")
            return node, value

        # Create child node and evaluate it, caching CPU-detached policy outputs
        child_node = NeuralMCTSNode(new_architecture, parent=node, action=selected_action)
        with torch.no_grad():
            child_graph = self._prepare_graph_data(new_architecture)
            child_graph['node_features'] = child_graph['node_features'].to(self.device)
            child_graph['edge_index'] = child_graph['edge_index'].to(self.device)
            child_graph['layer_positions'] = child_graph['layer_positions'].to(self.device)
            t_child = time.time()
            child_policy = self.policy_value_net(child_graph, phase=self.current_cycle.get_phase_value())
            _mcts_child_eval_time = time.time() - t_child
        # Store CPU-detached copy to avoid retaining GPU memory
        child_node.policy_value = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in child_policy.items()}
        child_node.prior_prob = node._priors_cache[selected_action]

        # Update parent
        node._unexpanded_actions_set.remove(selected_action)
        node.children.append(child_node)

        # Extract value from new network output signature for return
        value_output = child_node.policy_value.get('value')
        if isinstance(value_output, torch.Tensor):
            value = value_output.item()
        else:
            raise RuntimeError("Node policy_value missing 'value' tensor during expansion.")

        total_expand_time = time.time() - expand_start_time
        #print(f"[MCTS::_expand] total_time={total_expand_time:.4f}s node_id={id(node)} policy_eval={_mcts_policy_eval_time} priors_time={_mcts_priors_time} child_eval={_mcts_child_eval_time}\n")

        return child_node, value

    def _apply_dirichlet_noise_to_root(self, policy_output: Dict, num_actions: int, 
                                         epsilon: float = 0.25, alpha: float = 0.3):
        """Applies Dirichlet noise to the root policy logits for exploration.
        
        This mixes the prior policy with random noise:
            mixed_prior(a) = (1 - epsilon) * prior(a) + epsilon * noise(a)
        
        Args:
            policy_output (Dict): The policy output dictionary with action_type
                logits to modify in-place.
            num_actions (int): The number of valid actions for noise scaling.
            epsilon (float): The weight of the noise (0.25 = 75% prior, 25% noise).
            alpha (float): The Dirichlet concentration parameter.
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

    def _select_top_k_actions_by_prior(self, policy_output: Dict, valid_actions: List[Action],
                                       k: int, masks: Dict) -> List[Action]:
        """Selects the top-K actions by prior probability.
        
        This is optimized by using vectorized action type prior scoring instead
        of per-action computation.
        
        Args:
            policy_output (Dict): The policy network predictions.
            valid_actions (List[Action]): A list of all valid Action objects.
            k (int): The number of top actions to select.
            masks (Dict): The action masks.
        
        Returns:
            List[Action]: A list of the top-K actions sorted by prior.
        """
        if len(valid_actions) <= k:
            # Fewer actions than K, return all
            return valid_actions

        # Get masks for action types
        device = policy_output['action_type'].device
        masks['action_type'] = masks['action_type'].to(device)
        
        # Extract action_type logits and apply mask (vectorized)
        action_type_logits = policy_output['action_type']  # [batch=1, num_action_types]
        if action_type_logits.dim() > 1 and action_type_logits.shape[0] == 1:
            action_type_logits = action_type_logits.squeeze(0)  # [num_action_types]
        
        # Apply mask and compute softmax (vectorized for all actions at once)
        masked_logits = action_type_logits + masks['action_type']  # [num_action_types]
        action_type_probs = F.softmax(masked_logits, dim=-1)  # [num_action_types]
        
        # Torch-based vectorized scoring: compute primary scores by indexing the
        # action_type probability tensor and apply small secondary penalties.
        # Use torch.topk on-device to avoid CPU roundtrips.
        num_actions = len(valid_actions)
        device = action_type_probs.device

        # Ensure action_type_probs is 1-D on correct device
        if action_type_probs.dim() > 1 and action_type_probs.shape[0] == 1:
            action_type_probs = action_type_probs.squeeze(0)

        # Build tensor of action_type indices for all valid actions
        action_type_idx = torch.tensor([a.action_type.value for a in valid_actions], dtype=torch.long, device=device)

        # Primary scores: gather probabilities for each action's type
        primary_scores = action_type_probs[action_type_idx]  # [num_actions]

        # Secondary heuristic: vectorized via boolean masks converted to a tensor
        has_source = torch.tensor([1 if a.source_neuron is not None else 0 for a in valid_actions],
                                  dtype=primary_scores.dtype, device=device)
        has_target = torch.tensor([1 if a.target_neuron is not None else 0 for a in valid_actions],
                                  dtype=primary_scores.dtype, device=device)
        has_activation = torch.tensor([1 if a.activation is not None else 0 for a in valid_actions],
                                      dtype=primary_scores.dtype, device=device)

        secondary_scores = 0.1 * has_source + 0.1 * has_target + 0.05 * has_activation

        combined_scores = primary_scores - secondary_scores

        if k < num_actions:
            topk = torch.topk(combined_scores, k=k, largest=True)
            topk_idx = topk.indices  # already sorted by score desc
        else:
            # k >= num_actions: sort all
            topk_idx = torch.argsort(combined_scores, descending=True)

        # Convert indices to Python ints and select actions
        top_k_actions = [valid_actions[int(i.item())] for i in topk_idx]
        
        return top_k_actions

    def get_visit_distribution(self, node: 'NeuralMCTSNode', temperature: float = 1.0) -> torch.Tensor:
        """Extracts the visit count distribution from an MCTS node.
        
        Converts visit counts to a probability distribution, which is the
        MCTS-improved policy that AlphaZero trains the network to match.
        
        Args:
            node (NeuralMCTSNode): The MCTS node (root of the search tree).
            temperature (float): The temperature for softening the distribution.
        
        Returns:
            torch.Tensor: A tensor of action probabilities.
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

# Helper to safe-index logits (handle both 1-D and 2-D)
def safe_squeeze(tensor):
    """Safely squeezes a tensor.

    Args:
        tensor (torch.Tensor): The tensor to squeeze.

    Returns:
        torch.Tensor: The squeezed tensor.
    """
    return tensor.squeeze(0) if tensor.dim() > 1 and tensor.shape[0] == 1 else tensor
