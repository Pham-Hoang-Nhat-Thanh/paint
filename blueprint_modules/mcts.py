import math
from typing import List
from .network import NeuralArchitecture, Neuron, Connection
from .action import Action, ActionSpace
import numpy as np

class MCTSNode:
    """Node in the Monte Carlo Tree Search"""
    
    def __init__(self, architecture: NeuralArchitecture, parent=None, action: Action = None):
        self.architecture = architecture
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0  # Estimated value (accuracy)
        self.untried_actions: List[Action] = []
        
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.0):
        """Select best child using UCB1 formula"""
        if not self.children:
            return None
            
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(
                2 * math.log(self.visits) / child.visits
            )
            return exploitation + exploration
        
        return max(self.children, key=ucb_score)
    
    def expand(self, action_space: ActionSpace):
        """Expand this node by adding a child for an untried action"""
        if not self.untried_actions:
            # Initialize untried actions if needed
            self.untried_actions = action_space.get_valid_actions(self.architecture)
            
        if self.untried_actions:
            # Take one untried action
            action = self.untried_actions.pop()
            
            # Create new architecture by applying action
            new_architecture = self._copy_architecture()
            success = action_space.apply_action(new_architecture, action)
            
            if success:
                child_node = MCTSNode(new_architecture, parent=self, action=action)
                self.children.append(child_node)
                return child_node
        
        return None
    
    def _copy_architecture(self) -> NeuralArchitecture:
        """Create a deep copy of the architecture"""
        # Simple deep copy implementation
        new_arch = NeuralArchitecture()
        new_arch.neurons = {nid: Neuron(
            neuron.id, neuron.neuron_type, neuron.activation, 
            neuron.layer_position, neuron.bias
        ) for nid, neuron in self.architecture.neurons.items()}
        
        new_arch.connections = [Connection(
            conn.source_id, conn.target_id, conn.weight, conn.enabled
        ) for conn in self.architecture.connections]
        
        new_arch.next_neuron_id = self.architecture.next_neuron_id
        new_arch.performance_metrics = self.architecture.performance_metrics.copy()
        
        return new_arch

class MCTS:
    """Monte Carlo Tree Search for architecture exploration"""
    
    def __init__(self, action_space: ActionSpace, evaluation_fn, exploration_weight=1.0):
        self.action_space = action_space
        self.evaluation_fn = evaluation_fn  # Function to evaluate architecture performance
        self.exploration_weight = exploration_weight
        
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100):
        """Run MCTS from initial architecture"""
        root = MCTSNode(initial_architecture)
        
        for i in range(iterations):
            node = self._select(root)
            if not node.architecture.performance_metrics:
                # Evaluate leaf node
                value = self.evaluation_fn(node.architecture)
                node.architecture.performance_metrics['accuracy'] = value
                node.value = value
                node.visits = 1
            else:
                node = self._expand(node)
                value = self._simulate(node)
            
            self._backpropagate(node, value)
            
            if i % 10 == 0:
                print(f"MCTS iteration {i}, best value: {root.best_child(0).value if root.children else 0:.3f}")
        
        return root.best_child(0)  # Return best child (exploitation only)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand"""
        while node.children:
            node = node.best_child(self.exploration_weight)
            
            # If node hasn't been evaluated, stop here
            if not node.architecture.performance_metrics:
                return node
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node"""
        if not node.architecture.performance_metrics:
            return node  # Not evaluated yet
            
        child = node.expand(self.action_space)
        if child:
            return child
        return node

    def _simulate(self, node: MCTSNode, max_depth: int = 5) -> float:
        """Simulate from a node using random actions"""
        current_arch = node._copy_architecture()
        
        for depth in range(max_depth):
            valid_actions = self.action_space.get_valid_actions(current_arch)
            if not valid_actions:
                break
                
            # Random action selection for simulation
            action = np.random.choice(valid_actions)
            self.action_space.apply_action(current_arch, action)
        
        return self.evaluation_fn(current_arch)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent