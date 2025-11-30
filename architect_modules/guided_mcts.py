from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture, ActivationType
from blueprint_modules.action import Action, ActionSpace, ActionType
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict, List
import torch
import math
import numpy as np
import torch.nn.functional as F
from collections import deque
import time
import logging

class NeuralMCTSNode(MCTSNode):
    """MCTS node enhanced with neural network predictions"""

    def __init__(self, architecture: NeuralArchitecture, policy_value: Dict = None,
                 parent=None, action: Action = None, curriculum=None):
        super().__init__(architecture, parent, action)
        self.policy_value = policy_value  # Neural network predictions
        self.prior_prob = 0.0  # Prior probability from policy network
        self.curriculum = curriculum
        # Cache valid actions for this node to check if fully expanded
        self._valid_actions_cache = None
        # Flag to track if Dirichlet noise has been applied (for root only)
        self._dirichlet_applied = False
        # Cache action masks to avoid recomputation across _compute_action_prior calls
        self._cached_masks = None

    def _copy_architecture(self) -> NeuralArchitecture:
        """Create a deep copy of the node's architecture."""
        return self.architecture.copy()
    
    def is_fully_expanded(self, max_children: int) -> bool:
        """Check if all valid actions have been expanded as children, respecting max_children.

        A node is considered fully expanded if either of these conditions is met:
        1. The number of children has reached the `max_children` limit (progressive widening).
        2. All possible valid actions for this state have been expanded as children.
        
        Args:
            max_children: The maximum number of children a node is allowed to have.

        Returns:
            True if the node is fully expanded, False otherwise.
        """
        # Condition 1: Reached the maximum allowed children
        if len(self.children) >= max_children:
            return True

        # Condition 2: All possible valid actions have been expanded
        if self._valid_actions_cache is None:
            # If the cache of valid actions hasn't been populated yet, the node cannot be
            # considered fully expanded based on this criterion.
            return False
        
        # Check if all valid actions have corresponding children
        expanded_actions = {child.action for child in self.children if child.action is not None}
        return len(expanded_actions) >= len(self._valid_actions_cache)
    
    def best_child(self, exploration_weight: float = 1.0) -> 'NeuralMCTSNode':
        """Select best child using PUCT formula (AlphaZero style)"""
        if not self.children:
            return None

        def puct_score(child: 'NeuralMCTSNode') -> float:
            if child.visits == 0:
                return float('inf')

            # PUCT formula: Q + U
            # Q: exploitation term (average value)
            q_value = child.value / child.visits

            # U: exploration term
            u_value = exploration_weight * child.prior_prob * \
                     math.sqrt(self.visits) / (1 + child.visits)

            return q_value + u_value

        return max(self.children, key=puct_score)

class NeuralMCTS(MCTS):
    """MCTS enhanced with neural network guidance"""

    def __init__(self, action_space: ActionSpace, policy_value_net: UnifiedPolicyValueNetwork,
                 device: str = 'cpu', exploration_weight: float = 1.0,
                 early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.001, max_children: int = 50, max_neurons: int = 1000,
                 mcts_batch_size: int = 8, logger: logging.Logger = None):
        super().__init__(action_space, None, exploration_weight)
        self.policy_value_net = policy_value_net
        self.device = device
        # Early stopping settings
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        # Reusable ActionManager instance
        self.action_manager = ActionManager(action_space=self.action_space, max_neurons=max_neurons)
        self.max_children = max_children
        self.mcts_batch_size = mcts_batch_size
        self.logger = logger

    def cleanup(self):
        """Clean up resources used by NeuralMCTS"""
        # Force garbage collection
        import gc
        gc.collect()

    def _dismantle_tree(self, node: 'NeuralMCTSNode'):
        """Recursively dismantle the MCTS tree to break circular references."""
        if not node:
            return
        # Use a list copy for safe iteration while modifying children
        for child in list(node.children):
            self._dismantle_tree(child)
        node.parent = None
        node.children.clear()

    def clear_episode_caches(self, roots: List['NeuralMCTSNode'] = None, preserve_roots: bool = False):
        """Clear caches and dismantle the MCTS tree to prevent memory leaks."""
        if roots:
            for root in roots if isinstance(roots, list) else [roots]:
                if not preserve_roots:
                    self._dismantle_tree(root)
        # Also clear the action manager's cache if it exists
        if hasattr(self.action_manager, 'clear_cache'):
            self.action_manager.clear_cache()

    def _prepare_graph_data(self, architecture: NeuralArchitecture) -> Dict:
        """Optimized: Convert architecture to graph data for neural network using cached sorted IDs"""
        graph_data = architecture.to_graph_representation()
        
        # Add batch dimension and move to device
        graph_data['node_features'] = graph_data['node_features'].unsqueeze(0).to(self.device)
        graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
        # Use cached sorted neuron IDs from graph representation instead of sorting again
        sorted_neuron_ids = graph_data['sorted_neuron_ids']
        layer_positions = [float(architecture.neurons[neuron_id].layer_position) for neuron_id in sorted_neuron_ids]
        graph_data['layer_positions'] = torch.FloatTensor([layer_positions]).to(self.device)
        
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


    def _evaluate_node(self, node: NeuralMCTSNode, is_simulation: bool = False) -> float:
        """Evaluate a node using the policy-value network (AlphaZero style).
        
        No supervised stage - the policy-value network is the sole evaluator from the start.
        It learns from MCTS-generated experience and improves over time.
        """
        # Ensure we have a policy_value for this node
        if node.policy_value is None:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                node.policy_value = self.policy_value_net(graph_data)
        
        return node.policy_value['value'].item()
    
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
               temperature: float = 1.0, reuse_root: NeuralMCTSNode = None, mcts_batch_size: int = 8) -> NeuralMCTSNode:
        """Run standard neural-guided MCTS search (AlphaZero style).
        
        Terminology:
        - iteration: One complete MCTS cycle (SELECT → EXPAND → BACKUP)
        - step: One action within a rollout simulation (during _simulate, max_depth steps)
        - episode: One full architecture design session (in architecture_trainer.py)
        
        Each iteration (i):
        1. SELECT: Traverse tree to leaf using PUCT formula
        2. EXPAND: Add one new child to leaf and evaluate it (if not terminal)
        3. BACKUP: Propagate result up the tree
        
        Args:
            initial_architecture: Starting architecture for search
            iterations: Number of MCTS iterations to run
            temperature: Temperature for action selection
            reuse_root: Optional existing tree root to continue search (enables tree reuse)
        """
        return self.search_batched(initial_architecture, iterations, temperature, reuse_root, mcts_batch_size)

    def search_batched(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
                       temperature: float = 1.0, reuse_root: NeuralMCTSNode = None, batch_size: int = None) -> NeuralMCTSNode:
        """Run batched neural-guided MCTS search for higher throughput."""
        batch_size = batch_size if batch_size is not None else self.mcts_batch_size
        if reuse_root is not None:
            root = reuse_root
            root._valid_actions_cache = None
        else:
            root = NeuralMCTSNode(initial_architecture)

        if root.policy_value is None:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(initial_architecture)
                root.policy_value = self.policy_value_net(graph_data)
        
        if not root.children:
            self._expand(root)

        best_values = deque(maxlen=self.early_stopping_patience)
        no_improvement_count = 0
        
        num_batches = (iterations + batch_size - 1) // batch_size
        search_start_time = time.time()
        total_simulations = 0

        for i in range(num_batches):
            current_batch_size = min(batch_size, iterations - (i * batch_size))
            if current_batch_size <= 0:
                break

            leaves = [self._select(root) for _ in range(current_batch_size)]
            
            self._evaluate_and_expand_leaves(leaves)
            total_simulations += current_batch_size

            # Logging and Early stopping check
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - search_start_time
                simulations_per_second = total_simulations / elapsed_time if elapsed_time > 0 else 0
                current_best = root.value / root.visits if root.visits > 0 else 0.0
                self.logger.info(f"Batch {i+1}/{num_batches}, Best Value: {current_best:.4f}, "
                                 f"Throughput: {simulations_per_second:.2f} sims/sec")

                if i > num_batches // 2:
                    best_values.append(current_best)
                    if len(best_values) == self.early_stopping_patience:
                        if max(best_values) - min(best_values) < self.early_stopping_min_delta:
                            no_improvement_count += 1
                        else:
                            no_improvement_count = 0
                        
                        if no_improvement_count >= self.early_stopping_patience:
                            self.logger.info("Early stopping due to convergence.")
                            break

        final_node = self._select_final_action(root, temperature)
        if final_node and final_node.action:
            return (final_node, root)
        else:
            self.logger.warning("MCTS search completed: no valid action found")
            return (root, root)

    def _select(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Select a leaf node for expansion using the PUCT algorithm.

        This method traverses the tree from the given node downwards. At each step,
        it selects the child with the highest PUCT score. The traversal continues
        until it reaches a node that is not fully expanded or a leaf node (a node
        with no children). This ensures that the search explores promising paths while
        also maintaining exploration.

        Args:
            node: The starting node for the selection process (usually the root).

        Returns:
            The selected leaf node that is ready for expansion.
        """
        while node.is_fully_expanded(self.max_children) and node.children:
            node = node.best_child(self.exploration_weight)
        return node
        
    def _add_dirichlet_noise_to_root(self, root: NeuralMCTSNode, epsilon: float = 0.25, alpha: float = 0.3):
        """Add Dirichlet noise to root node priors for exploration (AlphaZero-style).
        
        This ensures exploration even when the policy network has strong but potentially
        incorrect priors. The noise is only added at the root of each search.
        
        Args:
            root: Root node of MCTS search
            epsilon: Weight of noise (0.25 = 75% prior, 25% noise)
            alpha: Dirichlet concentration parameter (lower = more dispersed)
        """
        if not root.children or len(root.children) == 0:
            return  # No children yet, noise will be applied during expansion
        
        # Generate Dirichlet noise
        num_children = len(root.children)
        noise = np.random.dirichlet([alpha] * num_children)
        
        # Mix noise with existing priors
        for i, child in enumerate(root.children):
            child.prior_prob = (1 - epsilon) * child.prior_prob + epsilon * noise[i]
    
    def _select_final_action(self, root: NeuralMCTSNode, temperature: float) -> NeuralMCTSNode:
        """Select final action based on visit counts, respecting temperature."""
        if not root.children:
            return root

        visit_counts = torch.tensor([child.visits for child in root.children], dtype=torch.float32)

        if temperature == 0:
            # Greedy selection: choose the action with the most visits
            selected_idx = torch.argmax(visit_counts).item()
        else:
            # Sample from the distribution visit_counts^(1/T)
            visit_distribution = torch.pow(visit_counts, 1.0 / temperature)
            visit_distribution /= torch.sum(visit_distribution)
            selected_idx = Categorical(probs=visit_distribution).sample().item()

        return root.children[selected_idx]

    def _expand_node(self, node: NeuralMCTSNode) -> List[NeuralMCTSNode]:
        """
        Expands a single node by generating its children.

        Args:
            node: The node to expand.

        Returns:
            A tuple containing:
            - A list of the newly created child nodes.
            - A tensor of the priors for the new children.
        """
        # Ensure parent policy_value is available
        if node.policy_value is None:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                node.policy_value = self.policy_value_net(graph_data)

        if node._valid_actions_cache is None:
            node._valid_actions_cache = self.action_space.get_valid_actions(node.architecture)
        
        expanded_actions = {child.action for child in node.children}
        valid_actions = [a for a in node._valid_actions_cache if a not in expanded_actions]
        if not valid_actions:
            return []

        if node._cached_masks is None:
            node._cached_masks = self.action_manager.get_action_masks(node.architecture)

        top_k_actions = self._select_top_k_actions_by_prior(
            node.policy_value, valid_actions, node.architecture, k=self.max_children, masks=node._cached_masks
        )
        
        if not top_k_actions:
            return []

        priors = self.action_manager._compute_priors_vectorized(
            node.policy_value, top_k_actions, node._cached_masks
        )
        
        child_prior_pairs = []
        for action, prior_prob in zip(top_k_actions, priors):
            new_architecture = node._copy_architecture()
            if self.action_space.apply_action(new_architecture, action):
                child = NeuralMCTSNode(new_architecture, parent=node, action=action)
                child.prior_prob = prior_prob.item()
                node.children.append(child)
                child_prior_pairs.append((child, prior_prob))
        return child_prior_pairs

    def _expand(self, node: NeuralMCTSNode) -> None:
        """Expand a single node, applying Dirichlet noise if it's the root."""
        child_prior_pairs = self._expand_node(node)

        is_root = node.parent is None
        if is_root and not node._dirichlet_applied and child_prior_pairs:
            priors = torch.tensor([p for _, p in child_prior_pairs], device=self.device)
            noise = torch.from_numpy(np.random.dirichlet([0.3] * len(priors))).to(priors.device, dtype=priors.dtype)
            final_priors = 0.75 * priors + 0.25 * noise
            for i, (child, _) in enumerate(child_prior_pairs):
                child.prior_prob = final_priors[i].item()
            node._dirichlet_applied = True

    def _evaluate_and_expand_leaves(self, leaves: List[NeuralMCTSNode]):
        """Evaluate leaf nodes in a batch, then backpropagate and expand."""
        
        unique_unevaluated_leaves = list(dict.fromkeys([
            leaf for leaf in leaves if leaf.policy_value is None and not leaf.children
        ]))

        if unique_unevaluated_leaves:
            leaf_graphs = [self._prepare_graph_data(leaf.architecture) for leaf in unique_unevaluated_leaves]
            batched_graph_data = self._batch_graph_data_for_eval_children(leaf_graphs)
            with torch.no_grad():
                batched_policy_value = self.policy_value_net(batched_graph_data)
            
            values = batched_policy_value['value']
            for i, leaf in enumerate(unique_unevaluated_leaves):
                leaf.policy_value = {
                    'value': values[i:i+1],
                    'action_type': batched_policy_value['action_type'][i:i+1],
                    'source_logits_dict': {k: v[i:i+1] for k, v in batched_policy_value['source_logits_dict'].items()},
                    'target_heads': batched_policy_value['target_heads'],
                    'activation_heads': batched_policy_value['activation_heads'],
                    'shared_features': batched_policy_value['shared_features'][i:i+1],
                    'source_encoder': batched_policy_value['source_encoder']
                }

        # 4. Backpropagate and expand for each simulation path
        for leaf in leaves:
            value = leaf.policy_value['value'].item()
            self._backpropagate(leaf, value)
            
            # Expand the leaf to create new children for future selections
            if not leaf.children:
                self._expand(leaf)

    def _batch_graph_data_for_eval_children(self, child_graphs: List[Dict]) -> Dict:
        """
        Creates a single batched graph from a list of individual graph data dictionaries.

        Args:
            child_graphs: A list of graph data dictionaries, as created by `_prepare_graph_data`.

        Returns:
            A single graph data dictionary representing the batched graph.
        """
        if not child_graphs:
            return {}

        num_graphs = len(child_graphs)
        all_node_features = []
        all_layer_positions = []
        node_offsets = [0]
        
        for child_graph in child_graphs:
            child_nodes = child_graph['node_features'].squeeze(0)
            child_pos = child_graph['layer_positions'].squeeze(0)
            all_node_features.append(child_nodes)
            all_layer_positions.append(child_pos)
            node_offsets.append(node_offsets[-1] + child_nodes.shape[0])

        all_node_features_cat = torch.cat(all_node_features, dim=0)
        all_layer_positions_cat = torch.cat(all_layer_positions, dim=0)

        batch_list = []
        for graph_id in range(num_graphs):
            num_nodes = node_offsets[graph_id + 1] - node_offsets[graph_id]
            batch_list.append(torch.full((num_nodes,), graph_id, dtype=torch.long, device=self.device))
        batch_tensor = torch.cat(batch_list, dim=0)

        all_edges = []
        for i, child_graph in enumerate(child_graphs):
            offset = node_offsets[i]
            child_edges = child_graph['edge_index']
            if child_edges.shape[1] > 0:
                all_edges.append(child_edges + offset)

        if all_edges:
            all_edge_indices = torch.cat(all_edges, dim=1)
        else:
            all_edge_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return {
            'node_features': all_node_features_cat.unsqueeze(0),
            'edge_index': all_edge_indices,
            'layer_positions': all_layer_positions_cat.unsqueeze(0),
            'batch': batch_tensor,
            'num_graphs': num_graphs
        }

    def _backpropagate(self, node: NeuralMCTSNode, value: float):
        """Backpropagate the value up the tree from a single node."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _select_top_k_actions_by_prior(self, policy_output: Dict, valid_actions: List[Action],
                                       architecture: NeuralArchitecture, k: int, masks: Dict = None) -> List[Action]:
        """Select top-K actions using a fast, heuristic-based approximation of prior probability.

        This method serves as a high-speed filter for large action spaces. It calculates a composite
        score for each action by combining the log-probabilities from all available *unconditional*
        policy heads (action_type, source_neuron, and add_neuron's activation). This avoids the
        expensive computation of conditional heads (e.g., target_neuron) while still providing a
        much more accurate heuristic than using action_type alone.

        Args:
            policy_output: Dictionary with policy network predictions.
            valid_actions: List of all valid Action objects.
            architecture: The current architecture (for obtaining masks).
            k: The number of top actions to select.
            masks: Optional pre-computed action masks to avoid re-computation.

        Returns:
            A list of the top-K actions, sorted by their heuristic score.
        """
        if len(valid_actions) <= k:
            # Fewer actions than K, return all
            return valid_actions

        num_actions = len(valid_actions)
        device = policy_output['action_type'].device

        # Get action masks if not provided, and make a copy to avoid modifying the cache
        if masks is None:
            masks = self.action_manager.get_action_masks(architecture)
        
        local_masks = masks.copy()
        tensor_size = local_masks.pop('tensor_size', self.action_manager.max_neurons)
        
        # Ensure masks are on the correct device
        for key in local_masks:
            if isinstance(local_masks[key], torch.Tensor):
                local_masks[key] = local_masks[key].to(device)

        # --- 1. Calculate Action Type Log Probs (Primary Score) ---
        action_type_logits = policy_output['action_type'].squeeze(0)
        masked_action_logits = action_type_logits + local_masks['action_type']
        log_action_type_probs = F.log_softmax(masked_action_logits, dim=-1)

        # --- 2. Vectorized Action Metadata Setup ---
        action_type_idx = torch.tensor([a.action_type.value for a in valid_actions], dtype=torch.long, device=device)
        source_neurons = torch.tensor([a.source_neuron if a.source_neuron is not None else -1 for a in valid_actions], dtype=torch.long, device=device)
        
        activation_types = list(ActivationType)
        activations = torch.tensor([activation_types.index(a.activation) if a.activation is not None else -1 for a in valid_actions], dtype=torch.long, device=device)

        # Initialize scores with the log probability of each action's type
        scores = log_action_type_probs[action_type_idx]

        # --- 3. Add Log Probs from Unconditional Heads ---
        # ADD_NEURON's activation is unconditional
        add_neuron_mask = (action_type_idx == ActionType.ADD_NEURON.value)
        if add_neuron_mask.any():
            add_neuron_act_logits = policy_output['activation_heads']['add_neuron'](policy_output['shared_features']).squeeze(0)
            log_act_probs = F.log_softmax(add_neuron_act_logits, dim=-1)
            action_activations = activations[add_neuron_mask]
            valid_activation_mask = action_activations >= 0
            if valid_activation_mask.any():
                scores[add_neuron_mask][valid_activation_mask] += log_act_probs[action_activations[valid_activation_mask]]

        # Source neurons for various actions are unconditional
        source_logits_dict = policy_output['source_logits_dict']
        action_name_map = {
            'remove_neuron': (ActionType.REMOVE_NEURON, 'remove_source_neurons'),
            'add_connection': (ActionType.ADD_CONNECTION, 'source_neurons'),
            'remove_connection': (ActionType.REMOVE_CONNECTION, 'source_neurons'),
            'modify_activation': (ActionType.MODIFY_ACTIVATION, 'modify_source_neurons')
        }

        for name, (action_enum, mask_key) in action_name_map.items():
            action_mask = (action_type_idx == action_enum.value)
            if action_mask.any():
                source_logits = source_logits_dict.get(name)
                if source_logits is not None:
                    source_mask = local_masks.get(mask_key, local_masks['source_neurons'])
                    prepared_logits = self.action_manager._prepare_logits(source_logits, tensor_size, source_mask)
                    log_source_probs = F.log_softmax(prepared_logits.squeeze(0), dim=-1)
                    
                    action_sources = source_neurons[action_mask]
                    valid_source_mask = action_sources >= 0
                    if valid_source_mask.any():
                        scores[action_mask][valid_source_mask] += log_source_probs[action_sources[valid_source_mask]]

        # --- 4. Select Top-K Actions Based on Combined Scores ---
        if k < num_actions:
            _, topk_idx = torch.topk(scores, k=k, largest=True)
        else:
            topk_idx = torch.argsort(scores, descending=True)

        return [valid_actions[i] for i in topk_idx]

    def get_visit_distribution(self, node: 'NeuralMCTSNode', temperature: float = 1.0) -> torch.Tensor:
        """Extract visit count distribution from MCTS node.
        
        Converts visit counts to probability distribution:
            π(a|s) = visit_count(a)^(1/temperature) / sum(visit_count^(1/temperature))
        
        This is the MCTS-improved policy that AlphaZero trains the network to match.
        
        Args:
            node: MCTS node (root of search tree)
            temperature: Temperature for softening distribution (1.0 = visit counts only)
                        Higher temperature = more uniform; 0 = greedy (argmax visits)
        
        Returns:
            torch.Tensor of shape [num_children] with action probabilities
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
