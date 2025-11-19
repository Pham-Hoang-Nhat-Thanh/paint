import math
from typing import List
from .network import NeuralArchitecture, Neuron, Connection
from .action import Action, ActionSpace
import numpy as np
import copy

class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search tree.

    Each node corresponds to a specific neural architecture and stores
    information about the search process, such as visit counts, value
    estimates, and child nodes.

    Attributes:
        architecture (NeuralArchitecture): The neural architecture represented
            by this node.
        parent (MCTSNode): The parent node in the search tree.
        action (Action): The action that led from the parent to this node.
        children (List[MCTSNode]): A list of child nodes.
        visits (int): The number of times this node has been visited during
            the search.
        value (float): The estimated value of this node's architecture.
        untried_actions (List[Action]): A list of actions that have not yet
            been explored from this node.
    """
    
    def __init__(self, architecture: NeuralArchitecture, parent=None, action: Action = None):
        self.architecture = architecture
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0  # Estimated value (accuracy)
        self.untried_actions: List[Action] = []
        
    def is_fully_expanded(self) -> bool:
        """Checks if all possible actions from this node have been explored.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.0):
        """Selects the best child node using the UCB1 formula.

        The UCB1 (Upper Confidence Bound 1) formula balances exploitation
        (choosing nodes with high average value) and exploration (choosing nodes
        that have been visited less frequently).

        Args:
            exploration_weight (float): A constant that controls the trade-off
                between exploitation and exploration.

        Returns:
            MCTSNode: The child node with the highest UCB1 score.
        """
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
        """Expands the current node by creating a new child node.

        A new child is created by applying one of the untried actions to the
        current node's architecture.

        Args:
            action_space (ActionSpace): The action space, used to get valid
                actions.

        Returns:
            MCTSNode: The new child node, or None if expansion fails.
        """
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
        """Create a fast copy of the architecture (no MNIST base init, no deepcopy)."""
        # Avoid calling NeuralArchitecture.__init__ to skip MNIST base init
        arch = self.architecture
        new_arch = object.__new__(NeuralArchitecture)
        # Copy neurons (shallow copy of dict, but new Neuron objects)
        new_arch.neurons = {nid: Neuron(
            neuron.id, neuron.neuron_type, neuron.activation,
            neuron.layer_position, neuron.bias
        ) for nid, neuron in arch.neurons.items()}
        # Copy connections (new Connection objects)
        new_arch.connections = [Connection(
            conn.source_id, conn.target_id, conn.weight, conn.enabled
        ) for conn in arch.connections]
        # Copy next_neuron_id
        new_arch.next_neuron_id = arch.next_neuron_id
        # Copy performance metrics (shallow copy)
        try:
            new_arch.performance_metrics = arch.performance_metrics.copy()
        except Exception:
            new_arch.performance_metrics = {}
        # Copy _connection_set if present
        if hasattr(arch, '_connection_set'):
            new_arch._connection_set = set(arch._connection_set)
        # IMPORTANT: Copy cached attributes when structure is identical (before applying action)
        # Caches will be invalidated when action is applied (add/remove neurons/connections)
        if hasattr(arch, '_sorted_neuron_ids') and arch._sorted_neuron_ids is not None:
            new_arch._sorted_neuron_ids = list(arch._sorted_neuron_ids)  # Copy the list
        else:
            new_arch._sorted_neuron_ids = None
        
        if hasattr(arch, '_connectivity_cache') and arch._connectivity_cache is not None:
            new_arch._connectivity_cache = {
                'has_incoming': set(arch._connectivity_cache['has_incoming']),
                'has_outgoing': set(arch._connectivity_cache['has_outgoing'])
            }
        else:
            new_arch._connectivity_cache = None
        # Copy any other custom attributes if needed
        return new_arch

class MCTS:
    """Implements a standard Monte Carlo Tree Search algorithm.

    This class orchestrates the MCTS process, including the selection,
    expansion, simulation, and backpropagation phases.

    Attributes:
        action_space (ActionSpace): The action space for the search.
        evaluation_fn (callable): A function that evaluates the performance of
            a given architecture.
        exploration_weight (float): A constant controlling the level of
            exploration.
    """
    
    def __init__(self, action_space: ActionSpace, evaluation_fn, exploration_weight=1.0):
        self.action_space = action_space
        self.evaluation_fn = evaluation_fn  # Function to evaluate architecture performance
        self.exploration_weight = exploration_weight
        
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100):
        """Runs the MCTS search for a specified number of iterations.

        Args:
            initial_architecture (NeuralArchitecture): The starting
                architecture for the search.
            iterations (int): The number of iterations to run the search.

        Returns:
            MCTSNode: The best child node found after the search.
        """
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
                if node:
                    value = self._simulate(node)
                    self._backpropagate(node, value)

            if i % 20 == 0:  # Reduced logging frequency
                print(f"MCTS iteration {i}, best value: {root.best_child(0).value if root.children else 0:.3f}")

        return root.best_child(0)  # Return best child (exploitation only)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCT.
        
        Traverse down the tree by selecting best children until we reach:
        1. A node that is not fully expanded (still has untried actions), OR
        2. A leaf node (no children)
        """
        while node.is_fully_expanded() and node.children:
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