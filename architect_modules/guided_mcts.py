from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture, ActivationType
from blueprint_modules.action import Action, ActionSpace
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict, List
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

    def top_k_children(self, k: int, exploration_weight: float = 1.0) -> List['NeuralMCTSNode']:
        """Select top-k children using PUCT formula (AlphaZero style)"""
        if not self.children:
            return []

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

        # Sort children by PUCT score in descending order and return top-k
        sorted_children = sorted(self.children, key=puct_score, reverse=True)
        return sorted_children[:k]

class NeuralMCTS(MCTS):
    """MCTS enhanced with neural network guidance"""

    def __init__(self, action_space: ActionSpace, policy_value_net: UnifiedPolicyValueNetwork,
                 device: str = 'cpu', exploration_weight: float = 1.0, curriculum=None):
        super().__init__(action_space, None, exploration_weight)
        self.policy_value_net = policy_value_net
        self.device = device
        self.curriculum = curriculum
        
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
    
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
               temperature: float = 1.0) -> NeuralMCTSNode:
        """Run neural-guided MCTS search"""
        root = NeuralMCTSNode(initial_architecture)

        # Get neural network predictions for root
        with torch.no_grad():
            graph_data = self._prepare_graph_data(initial_architecture)
            policy_value = self.policy_value_net(graph_data)
            root.policy_value = policy_value

        for i in range(iterations):
            node = self._select(root)

            if not node.architecture.performance_metrics:
                # Evaluate leaf node using value network
                value = node.policy_value['value'].item()
                node.architecture.performance_metrics = {'estimated_accuracy': value}
                node.value = value
                node.visits = 1
            else:
                node = self._expand(node)
                if node != self._select(root):  # If we expanded
                    value = node.policy_value['value'].item()
                else:
                    value = self._simulate(node)

            self._backpropagate(node, value)

            if i % 20 == 0:
                best_child = root.best_child(0)  # Greedy best
                if best_child:
                    print(f"MCTS iteration {i}, best value: {best_child.value/best_child.visits:.3f}")

        return self._select_final_action(root, temperature)

    def search_with_beam(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
                         beam_width: int = 5, temperature: float = 1.0) -> List[NeuralMCTSNode]:
        """Run neural-guided MCTS search with beam search to maintain diversity"""
        root = NeuralMCTSNode(initial_architecture)

        # Get neural network predictions for root
        with torch.no_grad():
            graph_data = self._prepare_graph_data(initial_architecture)
            policy_value = self.policy_value_net(graph_data)
            root.policy_value = policy_value

        # Initialize beam with root
        beam = [root]

        for i in range(iterations):
            new_beam = []

            for node in beam:
                # Perform MCTS steps for each node in beam
                for _ in range(beam_width // len(beam) + 1):  # Distribute iterations
                    selected_node = self._select(node)

                    if not selected_node.architecture.performance_metrics:
                        # Evaluate leaf node using value network
                        value = selected_node.policy_value['value'].item()
                        selected_node.architecture.performance_metrics = {'estimated_accuracy': value}
                        selected_node.value = value
                        selected_node.visits = 1
                    else:
                        expanded_node = self._expand(selected_node)
                        if expanded_node != self._select(node):  # If we expanded
                            value = expanded_node.policy_value['value'].item()
                        else:
                            value = self._simulate(expanded_node)

                    self._backpropagate(expanded_node, value)

                # Add top-k children to new beam
                top_children = node.top_k_children(beam_width)
                new_beam.extend(top_children)

            # Select top beam_width nodes for next iteration
            if new_beam:
                # Sort by value/visits (greedy)
                new_beam.sort(key=lambda x: x.value / x.visits if x.visits > 0 else 0, reverse=True)
                beam = new_beam[:beam_width]

            if i % 20 == 0:
                best_node = max(beam, key=lambda x: x.value / x.visits if x.visits > 0 else 0)
                print(f"MCTS beam iteration {i}, best value: {best_node.value/best_node.visits:.3f}")

        return beam
    
    def _select_final_action(self, root: NeuralMCTSNode, temperature: float) -> NeuralMCTSNode:
        """Select final action based on visit counts and temperature"""
        if not root.children:
            return root
        
        visit_counts = torch.tensor([child.visits for child in root.children])
        
        if temperature == 0:
            # Greedy selection
            selected_idx = visit_counts.argmax().item()
        else:
            # Temperature-based selection
            visit_probs = visit_counts.float() ** (1 / temperature)
            visit_probs = visit_probs / visit_probs.sum()
            selected_idx = Categorical(probs=visit_probs).sample().item()
        
        return root.children[selected_idx]
    
    def _expand(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Expand node using policy network priors"""
        if not node.policy_value:
            # Get neural network predictions if not available
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                node.policy_value = self.policy_value_net(graph_data)

        # Use action manager to get valid actions
        action_manager = ActionManager(action_space=self.action_space)

        # Determine whether to use policy or random based on curriculum
        if self.curriculum:
            train_params = self.curriculum.get_training_parameters()
            use_policy = train_params['policy_mix_ratio'] > 0.0
        else:
            use_policy = False  # Default to random if no curriculum

        action = action_manager.select_action(
            node.policy_value, node.architecture, exploration=True, use_policy=use_policy
        )

        # Apply action to create new architecture
        new_architecture = node._copy_architecture()
        success = self.action_space.apply_action(new_architecture, action)

        if success:
            # Create child node
            child = NeuralMCTSNode(new_architecture, parent=node, action=action)

            # Compute prior probability for this action from policy network
            child.prior_prob = self._compute_action_prior_prob(node.policy_value, action, node.architecture)

            # Get neural network predictions for child
            with torch.no_grad():
                graph_data = self._prepare_graph_data(new_architecture)
                child.policy_value = self.policy_value_net(graph_data)

            node.children.append(child)
            return child

        return node

    def _compute_action_prior_prob(self, policy_output: Dict, action: Action, architecture: NeuralArchitecture) -> float:
        """Compute the prior probability of an action from policy network output"""
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

        return prior_prob

    def _simulate(self, node: NeuralMCTSNode, max_depth: int = 5) -> float:
        """Simulate from a node using random actions and evaluate with the policy-value net.

        Overrides the base MCTS._simulate which expects an external evaluation function.
        This version performs random rollouts on the architecture and uses the
        neural network to estimate the final state's value.
        """
        # Copy architecture for simulation
        current_arch = node._copy_architecture()

        for _ in range(max_depth):
            valid_actions = self.action_space.get_valid_actions(current_arch)
            if not valid_actions:
                break

            # Randomly choose an action and apply
            action = np.random.choice(valid_actions)
            self.action_space.apply_action(current_arch, action)

        # Evaluate the simulated architecture using the policy-value network
        with torch.no_grad():
            graph_data = self._prepare_graph_data(current_arch)
            policy_value = self.policy_value_net(graph_data)
            value = policy_value['value'].item()

        return value