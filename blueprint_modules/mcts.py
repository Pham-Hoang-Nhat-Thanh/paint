import math
from typing import List
from .network import NeuralArchitecture, Neuron, Connection
from .action import Action, ActionSpace
import numpy as np
import copy

class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search tree.

    Attributes:
        architecture (NeuralArchitecture): The neural architecture of this node.
        parent (MCTSNode): The parent node.
        action (Action): The action that led to this node.
        children (List[MCTSNode]): A list of child nodes.
        visits (int): The number of times this node has been visited.
        value (float): The estimated value (e.g., accuracy) of this node.
        untried_actions (List[Action]): A list of actions that have not yet
            been tried from this node.
    """
    
    def __init__(self, architecture: NeuralArchitecture, parent=None, action: Action = None):
        """Initializes the MCTSNode.

        Args:
            architecture (NeuralArchitecture): The neural architecture of this node.
            parent (MCTSNode, optional): The parent node. Defaults to None.
            action (Action, optional): The action that led to this node.
                Defaults to None.
        """
        self.architecture = architecture
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0  # Estimated value (accuracy)
        self.untried_actions: List[Action] = []
        
    def is_fully_expanded(self) -> bool:
        """Checks if the node is fully expanded.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.0):
        """Selects the best child using the UCB1 formula.

        Args:
            exploration_weight (float): The exploration weight.

        Returns:
            MCTSNode: The best child node.
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
        """Expands this node by adding a child for an untried action.

        Args:
            action_space (ActionSpace): The action space.

        Returns:
            MCTSNode: The new child node, or None if no new child was created.
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
        """Creates a fast copy of the architecture.

        This method avoids the overhead of deepcopy and the MNIST base
        initialization.

        Returns:
            NeuralArchitecture: A copy of the architecture.
        """
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
    """Monte Carlo Tree Search for architecture exploration.

    Attributes:
        action_space (ActionSpace): The action space.
        evaluation_fn (callable): A function to evaluate the performance of an
            architecture.
        exploration_weight (float): The exploration weight.
    """
    
    def __init__(self, action_space: ActionSpace, evaluation_fn, exploration_weight=1.0):
        """Initializes the MCTS.

        Args:
            action_space (ActionSpace): The action space.
            evaluation_fn (callable): A function to evaluate the performance
                of an architecture.
            exploration_weight (float): The exploration weight.
        """
        self.action_space = action_space
        self.evaluation_fn = evaluation_fn  # Function to evaluate architecture performance
        self.exploration_weight = exploration_weight
        
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100):
        """Runs the MCTS search.

        Args:
            initial_architecture (NeuralArchitecture): The initial architecture.
            iterations (int): The number of iterations to run.

        Returns:
            MCTSNode: The best child node found.
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
        """Selects a node to expand using the UCT formula.

        Args:
            node (MCTSNode): The node to start the selection from.

        Returns:
            MCTSNode: The selected node.
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_weight)
            
            # If node hasn't been evaluated, stop here
            if not node.architecture.performance_metrics:
                return node
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expands a node.

        Args:
            node (MCTSNode): The node to expand.

        Returns:
            MCTSNode: The new child node, or the original node if no new child
                was created.
        """
        if not node.architecture.performance_metrics:
            return node  # Not evaluated yet
            
        child = node.expand(self.action_space)
        if child:
            return child
        return node

    def _simulate(self, node: MCTSNode, max_depth: int = 5) -> float:
        """Simulates from a node using random actions.

        Args:
            node (MCTSNode): The node to simulate from.
            max_depth (int): The maximum depth of the simulation.

        Returns:
            float: The evaluation of the final architecture.
        """
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
        """Backpropagates the value up the tree.

        Args:
            node (MCTSNode): The node to start backpropagation from.
            value (float): The value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
