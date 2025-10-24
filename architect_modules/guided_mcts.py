from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture
from blueprint_modules.action import Action, ActionSpace
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict
import torch
import math
import numpy as np


class NeuralMCTSNode(MCTSNode):
    """MCTS node enhanced with neural network predictions"""
    
    def __init__(self, architecture: NeuralArchitecture, policy_value: Dict = None, 
                 parent=None, action: Action = None):
        super().__init__(architecture, parent, action)
        self.policy_value = policy_value  # Neural network predictions
        self.prior_prob = 0.0  # Prior probability from policy network
    
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
                 device: str = 'cpu', exploration_weight: float = 1.0):
        super().__init__(action_space, None, exploration_weight)
        self.policy_value_net = policy_value_net
        self.device = device
        
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
            layer_positions.append(architecture.neurons[neuron_id].layer_position)
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
        action_manager = ActionManager()
        
        # Sample action using policy network
        action = action_manager.select_action(
            node.policy_value, node.architecture, exploration=True
        )
        
        # Apply action to create new architecture
        new_architecture = node._copy_architecture()
        success = self.action_space.apply_action(new_architecture, action)
        
        if success:
            # Create child node
            child = NeuralMCTSNode(new_architecture, parent=node, action=action)
            
            # Get prior probability for this action
            # This is simplified - in practice, you'd compute the exact probability
            child.prior_prob = 0.1  # Placeholder
            
            # Get neural network predictions for child
            with torch.no_grad():
                graph_data = self._prepare_graph_data(new_architecture)
                child.policy_value = self.policy_value_net(graph_data)
            
            node.children.append(child)
            return child
        
        return node

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