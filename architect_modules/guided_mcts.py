from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture, ActivationType, NeuronType, Neuron, Connection
from blueprint_modules.action import Action, ActionSpace, ActionType
from blueprint_modules.evolutionary_cycle import EvolutionaryCycle
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict, List
import time
import torch
import math
import numpy as np
import torch.nn.functional as F
from collections import deque
import gc

class NeuralMCTSNode(MCTSNode):
    """MCTS node enhanced with neural network predictions"""

    def __init__(self, architecture: NeuralArchitecture, policy_value: Dict = None,
                 parent=None, action: Action = None, curriculum=None, mcts=None):
        super().__init__(architecture, parent, action)
        self.policy_value = policy_value  # Neural network predictions
        self.prior_prob = 0.0  # Prior probability from policy network
        self.curriculum = curriculum
        # Cache valid actions for this node to check if fully expanded
        self._valid_actions_cache = None
        # Flag to track if Dirichlet noise has been applied (for root only)
        self._dirichlet_applied = False
        # Incremental set of expanded actions for O(1) filtering (vectorized action filtering)
        self._expanded_actions_set = set()
        # Back-reference to owning MCTS instance (used for progressive widening parameters)
        self.mcts = mcts
    
    
    def is_fully_expanded(self) -> bool:
        """Check if node should be traversed deeper or returned for expansion.
        
        With unlimited child expansion (progressive widening via top-K selection):
        - Return True only if ALL valid actions have been expanded AND all valid actions exhausted
        - Return False if there are unexpanded actions (allows tree to grow deeper)
        
        This ensures _select doesn't keep returning fully-expanded leaf nodes that
        have no new children to expand.
        """
        if self._valid_actions_cache is None:
            # Can't determine; assume not fully expanded
            return False
        
        # Check how many valid actions remain unexpanded
        unexpanded_actions = [a for a in self._valid_actions_cache if a not in self._expanded_actions_set]

        # If there are no unexpanded actions and we have children, we're fully expanded
        if len(unexpanded_actions) == 0 and len(self.children) > 0:
            return True

        return False
    
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
                 iso_weight: float = 0.01, comp_weight: float = 0.0, 
                 early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.001,
                 max_children: int = 50):
        super().__init__(action_space, None, exploration_weight)
        self.policy_value_net = policy_value_net
        self.device = device
        self.iso_weight = iso_weight
        self.comp_weight = comp_weight
        # Early stopping settings
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        # Evolutionary cycle tracking
        self.current_cycle = EvolutionaryCycle()
        # Reusable ActionManager instance
        self.action_manager = ActionManager(action_space=self.action_space)
        self.max_children = max_children

    def cleanup(self):
        """Clean up resources used by NeuralMCTS"""
        # Reset evolutionary cycle
        self.current_cycle.reset() if hasattr(self.current_cycle, 'reset') else None
        
        # Force garbage collection
        gc.collect()

    def clear_episode_caches(self, roots=None):
        """Clear caches that should not persist across episodes.

        Args:
            roots: Optional NeuralMCTSNode or iterable of nodes whose node-level
                   policy tensors should be offloaded/cleared. If None, only
                   the evaluation cache is cleared.

        This is safe to call at episode boundaries (not during inner MCTS sims)
        to free CPU/GPU memory that shouldn't persist across episodes.
        """

        # If caller provided node roots, offload their policy tensors to CPU
        if roots is None:
            # Best-effort device cache clearing
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return

        # Normalize roots to iterable
        if isinstance(roots, NeuralMCTSNode):
            roots = [roots]

        # Final best-effort device cache clear
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
            'edge_index': all_edge_indices,                       # [2, total_edges]                     # [total_edges]
            'layer_positions': all_layer_positions_cat.unsqueeze(0),  # [1, total_nodes]
            'batch': batch_tensor,  # [total_nodes] - graph_id per node
            'node_offsets': node_offsets,  # Offsets for extracting per-graph outputs
            'num_graphs': num_graphs
        }

    
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
               temperature: float = 1.0, reuse_root: NeuralMCTSNode = None) -> NeuralMCTSNode:
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
        
        # Reuse existing tree root if provided, otherwise create new root
        if reuse_root is not None:
            root = reuse_root
            root._valid_actions_cache = None
            # Ensure root has back-reference to this MCTS instance for progressive widening
            try:
                root.mcts = self
            except Exception:
                pass
        else:
            root = NeuralMCTSNode(initial_architecture, mcts=self)

        # Get neural network predictions for root
        with torch.no_grad():
            graph_data = self._prepare_graph_data(initial_architecture)
            policy_value = self.policy_value_net(graph_data)
            root.policy_value = policy_value

        # Early stopping tracking
        best_values = deque(maxlen=self.early_stopping_patience)
        no_improvement_count = 0
        initial_visits = root.visits  # Track starting visits for reused trees
        
        # Add Dirichlet noise to root for exploration (AlphaZero-style)
        # This ensures we explore even when policy network has strong (but potentially wrong) priors
        self._add_dirichlet_noise_to_root(root)

        print("Starting MCTS search...")
        for i in range(iterations):
            # ===== STEP 1: SELECT =====
            # Traverse tree using PUCT until reaching a leaf node
            node = self._select(root)

            # ===== STEP 2: EXPAND & EVALUATE =====
            # Expand one new child and get its value in one step (no redundant re-evaluation)
            expanded_node, value = self._expand(node)

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

        # ===== STEP 5: SELECT FINAL ACTION =====
        # Use visit counts to select final action (most visited = most promising)
        final_node = self._select_final_action(root, temperature)
        if final_node and final_node.action:
            # Update evolutionary cycle with final node's value
            final_value = final_node.value / final_node.visits
            self.current_cycle.add_evaluation(final_value)

            # Reset cycle if structural change
            if final_node.action.action_type in [ActionType.ADD_NEURON, ActionType.REMOVE_NEURON]:
                self.current_cycle.reset()

            print("MCTS search completed successfully")
            
            # Return tuple: (selected_child, search_root)
            # selected_child becomes new root for next search (tree reuse)
            # search_root provides visit distribution for training
            return (final_node, root)
        else:
            print("MCTS search completed: no valid action found")
            return (root, root)  # Return tuple even on failure
        
    def _select(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Select a node to expand using neural-guided PUCT selection.

        Differences from the base implementation:
        - Ensure node._valid_actions_cache is populated so is_fully_expanded() is meaningful
        - Use NeuralMCTSNode.best_child (PUCT using prior_prob) to traverse
        - Stop and return the first node that is not fully expanded or that has no policy_value
        """
        # Defensive: ensure we operate on NeuralMCTSNode-like nodes
        cur = node

        while True:
            # Initialize valid actions cache if not present so is_fully_expanded() works
            if getattr(cur, '_valid_actions_cache', None) is None:
                try:
                    cur._valid_actions_cache = self.action_space.get_valid_actions(
                        cur.architecture, evolutionary_cycle=self.current_cycle
                    )
                except Exception:
                    # If action generation fails, treat node as non-expanded (expand it)
                    return cur

            # If node has no children or is not fully expanded, return it for expansion
            if not cur.children or not cur.is_fully_expanded():
                return cur

            # Select best child using PUCT via NeuralMCTSNode.best_child
            next_child = cur.best_child(self.exploration_weight)
            if next_child is None:
                return cur

            cur = next_child

            # If the selected node hasn't been evaluated yet (no policy_value), stop here
            if getattr(cur, 'policy_value', None) is None:
                return cur
    
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
        """Select final action based on visit counts (proportional selection)"""
        if not root.children:
            return root

        visit_counts = torch.tensor([child.visits for child in root.children])

        # Proportional selection based on visit counts (standard MCTS final selection)
        visit_probs = visit_counts.float() / visit_counts.sum()
        selected_idx = Categorical(probs=visit_probs).sample().item()

        return root.children[selected_idx]
    
    def _expand(self, node: NeuralMCTSNode) -> tuple:
        """Expand node with evaluation caching.
        
        Evaluate only the selected child. Cache result for future encounters.
        """
        # --- Profiling instrumentation ---
        t_start = time.time()
        timings = {}

        # Ensure parent policy_value is available
        t = time.time()
        if node.policy_value is None:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                node.policy_value = self.policy_value_net(graph_data)
        timings['parent_eval'] = time.time() - t

        # Get valid actions (cache on node)
        t = time.time()
        if node._valid_actions_cache is None:
            all_valid_actions = self.action_space.get_valid_actions(
                node.architecture, evolutionary_cycle=self.current_cycle
            )
            node._valid_actions_cache = all_valid_actions
        all_valid_actions = node._valid_actions_cache
        timings['get_valid_actions'] = time.time() - t

        # Filter unexpanded actions.
        # Optimized: build a lightweight tuple-key set for expanded actions and
        # perform tuple membership tests which are faster than repeated Action
        # hashing/equality checks when the action list is large.
        t = time.time()
        if not node._expanded_actions_set:
            valid_actions = all_valid_actions
        else:
            expanded_keys = set()
            for ea in node._expanded_actions_set:
                expanded_keys.add((int(ea.action_type.value),
                                   -1 if ea.source_neuron is None else int(ea.source_neuron),
                                   -1 if ea.target_neuron is None else int(ea.target_neuron),
                                   ea.activation.value if ea.activation is not None else ""))

            valid_actions = []
            for a in all_valid_actions:
                key = (int(a.action_type.value),
                       -1 if a.source_neuron is None else int(a.source_neuron),
                       -1 if a.target_neuron is None else int(a.target_neuron),
                       a.activation.value if a.activation is not None else "")
                if key not in expanded_keys:
                    valid_actions.append(a)
        timings['filter_unexpanded'] = time.time() - t
        
        if not valid_actions:
            return (node, node.policy_value['value'].item())  # No unexpanded actions available
        
        # Build action masks for valid actions
        masks = self.action_manager.get_action_masks(node.architecture)
        # Move masks to device
        for key in masks:
            if key != 'tensor_size':
                masks[key] = masks[key].to(self.device)

        # Apply Dirichlet noise at root
        t = time.time()
        is_root = (node.parent is None)
        if is_root and not node._dirichlet_applied:
            self._apply_dirichlet_noise_to_policy(node.policy_value, len(valid_actions))
            node._dirichlet_applied = True
        timings['dirichlet'] = time.time() - t

        # Select top-K candidates (returns tensors/info kept on-device)
        t = time.time()
        topk_info = self._select_top_k_actions_by_prior(
            node.policy_value, valid_actions, k=self.max_children, masks=masks
        )
        timings['select_topk'] = time.time() - t

        # Sample ONE action from top-K for actual expansion (normal MCTS behavior)
        # topk_info contains device tensors; map selection logic to operate on tensors
        topk_indices = topk_info['indices']
        topk_count = self.max_children

        if topk_count == 1:
            # Single candidate: map device index back to Python action once
            selected_global_idx = int(topk_indices[0].item())
            selected_action = valid_actions[selected_global_idx]
        else:
            t = time.time()
            device = node.policy_value['action_type'].device
            masks['action_type'] = masks['action_type'].to(device)
            
            action_type_logits = node.policy_value['action_type']
            if action_type_logits.dim() > 1 and action_type_logits.shape[0] == 1:
                action_type_logits = action_type_logits.squeeze(0)
            
            masked_logits = action_type_logits + masks['action_type']
            log_probs = F.log_softmax(masked_logits, dim=-1)
            action_type_probs = torch.exp(log_probs)

            # Build per-topK action_type indices tensor by gathering from the full action_type_idx
            action_type_idx_full = topk_info['action_type_idx']  # device tensor for all valid_actions
            topk_action_type_idx = action_type_idx_full[topk_indices]  # indices for top-K (device)

            scores = action_type_probs[topk_action_type_idx]

            if torch.isnan(scores).any():
                probs = torch.ones(topk_count, device=device) / float(topk_count)
            else:
                probs = scores / (scores.sum() + 1e-8)
                probs = torch.clamp(probs, min=1e-8)
                probs = probs / probs.sum()

            selected_idx_in_topk = Categorical(probs=probs).sample().item()
            # Map selected top-K index back to global valid_actions index
            selected_global_idx = int(topk_indices[int(selected_idx_in_topk)].item())
            selected_action = valid_actions[selected_global_idx]
            timings['sample_action'] = time.time() - t

        # Apply selected action
        t = time.time()
        new_architecture = node._copy_architecture()
        success = self.action_space.apply_action(new_architecture, selected_action)
        timings['apply_action'] = time.time() - t
        
        if not success:
            return (node, node.policy_value['value'].item())

        t = time.time()
        timings['child_forward'] = 0.0
        timings['cache_hit'] = True
        with torch.no_grad():
            child_graph = self._prepare_graph_data(new_architecture)
            child_policy = self.policy_value_net(child_graph)
            child_value = child_policy['value'].item()
        timings['child_forward'] = time.time() - t
        timings['cache_hit'] = False
        
        # Create child node (attach mcts ref)
        child = NeuralMCTSNode(new_architecture, parent=node, action=selected_action, mcts=self)
        child.policy_value = child_policy
        
        t = time.time()
        try:
            child.prior_prob = float(self._compute_action_prior(
                node.policy_value, selected_action, masks=masks
            ))
        except Exception:
            child.prior_prob = 1.0 / topk_count  # Fallback uniform prior
        timings['compute_prior'] = time.time() - t
        
        node._expanded_actions_set.add(selected_action)
        node.children.append(child)

        timings['total'] = time.time() - t_start
        try:
            pass
            #print(f"[MCTS _expand timings] node_children={len(node.children)} valid_actions={len(all_valid_actions)} total={timings['total']:.4f}s breakdown={timings}")
        except Exception:
            pass

        return (child, child_value)
    
    def _apply_dirichlet_noise_to_policy(self, policy_output: Dict, num_actions: int, 
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
        # Generate Dirichlet noise for action type head
        noise = np.random.dirichlet([alpha] * max(1, num_actions))
        
        # Extract action type logits and convert noise to log scale for mixing with logits
        action_type_logits = policy_output['action_type']
        # Convert logits to probabilities via softmax
        action_type_probs = F.softmax(action_type_logits, dim=-1).detach().cpu().numpy()
        
        # Mix: noise affects only the action type head (main categorical decision)
        # Scale noise to match number of action types (5 in this case)
        if noise.shape[0] < action_type_probs.shape[-1]:
            # Pad noise if needed (though it shouldn't be)
            noise = np.pad(noise, (0, action_type_probs.shape[-1] - noise.shape[0]))
        else:
            noise = noise[:action_type_probs.shape[-1]]
        
        # Mix probabilities with noise
        mixed_probs = (1.0 - epsilon) * action_type_probs[0] + epsilon * noise
        
        # Convert mixed probabilities back to logits and update
        mixed_logits = np.log(np.clip(mixed_probs, 1e-8, 1.0))
        policy_output['action_type'] = torch.tensor(mixed_logits, dtype=action_type_logits.dtype, device=action_type_logits.device).unsqueeze(0)

    def _compute_action_prior(self, policy_output: Dict, action: Action, masks = None) -> float:
        """Vectorized prior computation using masked logits (consistent with ActionManager masking).
        
        Computes the prior probability as the softmax probability of the action under the policy,
        using the same masks that ActionManager applies when sampling.
        
        This ensures priors match the policy distribution over valid actions only.
        
        If node is provided and has cached masks, reuse them to avoid recomputation.
        """
        tensor_size = masks.pop('tensor_size')
        
        # Move masks to policy device
        device = policy_output['action_type'].device
        for key in masks:
            masks[key] = masks[key].to(device)
        
        # Helper to safe-index logits (handle both 1-D and 2-D)
        def safe_squeeze(tensor):
            return tensor.squeeze(0) if tensor.dim() > 1 and tensor.shape[0] == 1 else tensor
        
        prior = 1.0
        
        # 1. Action type probability (masked softmax over valid action types)
        action_type_logits = policy_output['action_type']  # [batch or scalar]
        action_type_logits = safe_squeeze(action_type_logits)  # [num_actions]
        # Apply mask
        masked_action_logits = action_type_logits + masks['action_type']
        action_type_probs = F.softmax(masked_action_logits, dim=-1)
        prior *= action_type_probs[action.action_type.value].item()
        
        # 2. Source neuron probability (if applicable, masked softmax)
        if action.source_neuron is not None:
            # Prepare logits with mask (padding + masking)
            prepared_source = self.action_manager._prepare_logits(
                policy_output['source_logits'], tensor_size, masks['source_neurons']
            )
            prepared_source = safe_squeeze(prepared_source)
            source_probs = F.softmax(prepared_source, dim=-1)
            if action.source_neuron < source_probs.shape[0]:
                prior *= source_probs[action.source_neuron].item()
        
        # 3. Target neuron probability (if applicable, masked softmax)
        if action.target_neuron is not None:
            prepared_target = self.action_manager._prepare_logits(
                policy_output['target_logits'], tensor_size, masks['target_neurons']
            )
            prepared_target = safe_squeeze(prepared_target)
            target_probs = F.softmax(prepared_target, dim=-1)
            if action.target_neuron < target_probs.shape[0]:
                prior *= target_probs[action.target_neuron].item()
        
        # 4. Activation probability (if applicable, masked softmax)
        if action.activation is not None:
            activation_logits = safe_squeeze(policy_output['activation_logits'])  # [num_activations]
            activation_probs = F.softmax(activation_logits, dim=-1)
            activation_idx = list(ActivationType).index(action.activation)
            if activation_idx < activation_probs.shape[0]:
                prior *= activation_probs[activation_idx].item()
        
        return prior

    def _select_top_k_actions_by_prior(self, policy_output: Dict, valid_actions: List[Action],
                                       k: int, masks) -> Dict:
        """Select top-K actions by prior probability with numerical stability.
        """
        num_actions = len(valid_actions)
        if num_actions <= k:
            return valid_actions

        # Acquire masks and logits once, move to device
        device = policy_output['action_type'].device

        action_type_logits = policy_output['action_type'].to(device)
        if action_type_logits.dim() > 1 and action_type_logits.shape[0] == 1:
            action_type_logits = action_type_logits.squeeze(0)

        # Use log_softmax for numerical stability (keep tensor ops on device)
        masked_logits = action_type_logits + masks['action_type']
        action_type_probs = F.softmax(masked_logits, dim=-1)  # [num_action_types]

        # Build compact numpy arrays (fast C loop) and convert to device tensors
        action_type_np = np.fromiter((a.action_type.value for a in valid_actions), dtype=np.int64, count=num_actions)
        # Move arrays to tensors on device to keep subsequent ops on-device
        action_type_idx = torch.from_numpy(action_type_np).to(device=device, dtype=torch.long)

        # Compute combined log-scores on device
        primary_scores = action_type_probs[action_type_idx]  # [num_actions]

        # Use torch.topk on device for O(N) expected selection of top-K
        topk = torch.topk(primary_scores, k=k, largest=True)

        # Keep indices and supporting tensors on-device for subsequent processing
        topk_indices = topk.indices  # tensor on device

        # Return a small struct containing tensors that the caller can use
        # to continue on-device processing. Caller can map a single selected
        # final index back to the Python Action only once when applying.
        return {
            'indices': topk_indices,                # Tensor[int] (device)
            'action_type_idx': action_type_idx,     # Tensor[int] (device), full length
            'scores': primary_scores  # Tensor[float] (device), full length
        }

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

