from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture, ActivationType, NeuronType, Neuron, Connection
from blueprint_modules.action import Action, ActionSpace, ActionType
from blueprint_modules.evolutionary_cycle import EvolutionaryCycle
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict
import torch
import math
import numpy as np

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
    
    def is_fully_expanded(self) -> bool:
        """Check if all valid actions have been expanded as children.
        
        Since NeuralMCTS doesn't use untried_actions list, we need to compare
        the expanded actions (children) with all valid actions for this state.
        """
        # Get all valid actions for this node (cache it to avoid recomputation)
        if self._valid_actions_cache is None:
            # Import here to avoid circular dependency
            from blueprint_modules.action import ActionSpace
            # We'll need the action_space from the parent MCTS instance
            # For now, return False to indicate not fully expanded if we can't check
            # The MCTS search method will set this properly
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
                 iso_weight: float = 0.01, comp_weight: float = 0.0, 
                 early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.001):
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
        self.current_cycle = EvolutionaryCycle()

    def cleanup(self):
        """Clean up resources used by NeuralMCTS"""
        # Clear evaluation cache to free memory
        self.evaluation_cache.clear()
        
        # Reset evolutionary cycle
        self.current_cycle.reset() if hasattr(self.current_cycle, 'reset') else None
        
        # Force garbage collection
        import gc
        gc.collect()

    def _prepare_graph_data(self, architecture: NeuralArchitecture) -> Dict:
        """Convert architecture to graph data for neural network"""
        graph_data = architecture.to_graph_representation()
        
        # Add batch dimension and move to device
        graph_data['node_features'] = graph_data['node_features'].unsqueeze(0).to(self.device)
        graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
        graph_data['edge_weights'] = graph_data['edge_weights'].to(self.device)
        
        # Create layer positions tensor
        layer_positions = []
        for neuron_id in sorted(architecture.neurons.keys()):
            layer_positions.append(float(architecture.neurons[neuron_id].layer_position))
        graph_data['layer_positions'] = torch.FloatTensor([layer_positions]).to(self.device)
        
        return graph_data


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
               temperature: float = 1.0, reuse_root: NeuralMCTSNode = None) -> NeuralMCTSNode:
        """Run standard neural-guided MCTS search (AlphaZero style).
        
        Terminology:
        - iteration: One complete MCTS cycle (SELECT → EXPAND → SIMULATE → BACKUP)
        - step: One action within a rollout simulation (during _simulate, max_depth steps)
        - episode: One full architecture design session (in architecture_trainer.py)
        
        Each iteration (i):
        1. SELECT: Traverse tree to leaf using PUCT formula
        2. EXPAND: Add one new child to leaf (if not terminal)
        3. SIMULATE: Run rollout from new/leaf node (up to max_depth steps)
        4. BACKUP: Propagate result up the tree
        
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
            # Verify root architecture matches initial_architecture
            root_neurons = sorted(root.architecture.neurons.keys())
            initial_neurons = sorted(initial_architecture.neurons.keys())
            if root_neurons != initial_neurons:
                print(f"ERROR: Root architecture mismatch in search()!")
                print(f"  reuse_root neurons: {root_neurons}")
                print(f"  initial_architecture neurons: {initial_neurons}")
                print(f"  Missing in root: {set(initial_neurons) - set(root_neurons)}")
                print(f"  Extra in root: {set(root_neurons) - set(initial_neurons)}")
            print(f"Reusing tree with {root.visits} visits from previous search")
        else:
            root = NeuralMCTSNode(initial_architecture)

        # Get neural network predictions for root
        with torch.no_grad():
            graph_data = self._prepare_graph_data(initial_architecture)
            policy_value = self.policy_value_net(graph_data)
            root.policy_value = policy_value

        # Early stopping tracking
        best_values = []
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

            # ===== STEP 2: EXPAND =====
            # Add one new child to leaf (returns expanded child or same node if no expansion possible)
            expanded_node = self._expand(node)

            # ===== STEP 3: SIMULATE/EVALUATE =====
            # Evaluate the expanded child (or leaf node if expansion failed)
            value = self._simulate(expanded_node)

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

            # Check for improvement over the last patience iterations
            if len(best_values) >= self.early_stopping_patience:
                recent_best = max(best_values[-self.early_stopping_patience:])
                oldest_recent = min(best_values[-self.early_stopping_patience:])
                improvement = recent_best - oldest_recent

                if improvement < self.early_stopping_min_delta:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0

                # Early stopping condition
                if no_improvement_count >= self.early_stopping_patience:
                    print(f"Early stopping at iteration {i + 1}: no significant improvement in last {self.early_stopping_patience} iterations")
                    break

            if (i + 1) % max(1, iterations // 10) == 0:
                print(f"MCTS iteration {i + 1}/{iterations} completed, best value: {current_best:.4f}")

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

            action_str = f"{final_node.action.action_type.name}"
            if final_node.action.source_neuron is not None:
                action_str += f"({final_node.action.source_neuron}"
                if final_node.action.target_neuron is not None:
                    action_str += f"->{final_node.action.target_neuron})"
                else:
                    action_str += ")"
            elif final_node.action.activation is not None:
                action_str += f"({final_node.action.activation.name})"
            print(f"MCTS FINAL ACTION: {action_str} (value: {final_node.value/final_node.visits:.3f}, visits: {final_node.visits})")
            print(f"MCTS tree stats: {root.visits} total visits, {len(root.children)} children at root")
            print("MCTS search completed successfully")
            
            # Return tuple: (selected_child, search_root)
            # selected_child becomes new root for next search (tree reuse)
            # search_root provides visit distribution for training
            return (final_node, root)
        else:
            print("MCTS search completed: no valid action found")
            return (root, root)  # Return tuple even on failure
        
    
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
    
    def _expand(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Expand node by creating ONE new child (standard MCTS expansion).
        
        Uses policy network guidance for action selection (AlphaZero style).
        Filters out already-expanded actions to avoid duplicates.
        """
        # Get neural network predictions if needed
        if not node.policy_value:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                node.policy_value = self.policy_value_net(graph_data)

        # Use action manager to get valid actions with evolutionary cycle context
        action_manager = ActionManager(action_space=self.action_space)

        # Get valid actions - filter out actions already expanded as children
        all_valid_actions = self.action_space.get_valid_actions(
            node.architecture, current_step=0, evolutionary_cycle=self.current_cycle
        )
        
        # Cache valid actions for is_fully_expanded check
        if node._valid_actions_cache is None:
            node._valid_actions_cache = all_valid_actions
        
        # Filter out actions that already have children
        expanded_actions = {child.action for child in node.children if child.action is not None}
        valid_actions = [a for a in all_valid_actions if a not in expanded_actions]

        if not valid_actions:
            return node  # No unexpanded actions available

        # Use policy network to select from valid actions
        action = action_manager.select_action(
            node.policy_value, node.architecture, exploration=True, use_policy=True
        )
        # Filter to only valid unexpanded actions
        if action not in valid_actions:
            action = np.random.choice(valid_actions)

        # Apply action to create new architecture
        new_architecture = node._copy_architecture()
        success = self.action_space.apply_action(new_architecture, action)

        if success:
            # Create child node
            child = NeuralMCTSNode(new_architecture, parent=node, action=action)

            # Compute prior probability from policy network
            child.prior_prob = self._compute_action_prior_prob(node.policy_value, action, node.architecture)

            # Get neural network predictions for child
            with torch.no_grad():
                graph_data = self._prepare_graph_data(new_architecture)
                child.policy_value = self.policy_value_net(graph_data)
            
            node.children.append(child)
            return child

        return node

    def _compute_action_prior_prob(self, policy_output: Dict, action: Action, architecture: NeuralArchitecture) -> float:
        """Compute the prior probability of an action from policy network output, boosting for de-isolating actions"""
        import torch.nn.functional as F

        prior_prob = 1.0

        # Action type probability
        action_type_logits = policy_output['action_type']
        action_type_probs = F.softmax(action_type_logits, dim=-1)
        prior_prob *= action_type_probs[0, action.action_type.value].item()

        # Source neuron probability (if applicable)
        if action.source_neuron is not None:
            source_logits = policy_output['source_logits']
            if source_logits.dim() == 1:
                source_probs = F.softmax(source_logits, dim=-1)
            else:
                source_probs = F.softmax(source_logits[0], dim=-1)
            if action.source_neuron < source_probs.shape[0]:
                prior_prob *= source_probs[action.source_neuron].item()

        # Target neuron probability (if applicable)
        if action.target_neuron is not None:
            target_logits = policy_output['target_logits']
            if target_logits.dim() == 1:
                target_probs = F.softmax(target_logits, dim=-1)
            else:
                target_probs = F.softmax(target_logits[0], dim=-1)
            if action.target_neuron < target_probs.shape[0]:
                prior_prob *= target_probs[action.target_neuron].item()

        # Activation probability (if applicable)
        if action.activation is not None:
            activation_logits = policy_output['activation_logits']
            activation_probs = F.softmax(activation_logits, dim=-1)
            activation_idx = list(ActivationType).index(action.activation)
            prior_prob *= activation_probs[0, activation_idx].item()

        # Boost prior for de-isolating actions (ADD_CONNECTION involving isolated hidden neurons)
        if action.action_type == ActionType.ADD_CONNECTION:
            # Identify isolated hidden neurons and their isolation types
            isolated_lacking_in = set()  # Neurons lacking inward connections
            isolated_lacking_out = set()  # Neurons lacking outward connections
            isolated_lacking_both = set()  # Neurons lacking both

            for neuron_id, neuron in architecture.neurons.items():
                if neuron.neuron_type == NeuronType.HIDDEN:
                    has_in = any(conn.target_id == neuron_id for conn in architecture.connections)
                    has_out = any(conn.source_id == neuron_id for conn in architecture.connections)
                    if not has_in and not has_out:
                        isolated_lacking_both.add(neuron_id)
                    elif not has_in:
                        isolated_lacking_in.add(neuron_id)
                    elif not has_out:
                        isolated_lacking_out.add(neuron_id)

            # Boost based on specific de-isolation
            boost = False
            if action.target_neuron in isolated_lacking_in or action.target_neuron in isolated_lacking_both:
                # Boost if target is a neuron lacking inward connections (or both)
                boost = True
            elif action.source_neuron in isolated_lacking_out:
                # Boost if source is a neuron lacking outward connections
                boost = True

            if boost:
                prior_prob *= 2.0  # Boost by factor of 2

        return prior_prob

    def _simulate(self, node: NeuralMCTSNode) -> float:
        """Evaluate a node using its value (direct evaluation).

        In NAS, random rollouts are not predictive because random future actions 
        don't necessarily lead to better architectures. Instead, we use direct 
        evaluation (quick_trainer in supervised, value network in policy stage).
        
        This is the AlphaZero approach: neural network evaluation replaces rollouts.
        """
        return self._evaluate_node(node, is_simulation=False)

