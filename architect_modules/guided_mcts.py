from blueprint_modules.mcts import MCTS, MCTSNode
from blueprint_modules.network import NeuralArchitecture, ActivationType, NeuronType, Neuron, Connection
from blueprint_modules.action import Action, ActionSpace, ActionType
from blueprint_modules.evolutionary_cycle import EvolutionaryCycle
from .policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from torch.distributions import Categorical
from typing import Dict, List
import torch
import math
import numpy as np
import traceback
from concurrent.futures import ProcessPoolExecutor


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
                 device: str = 'cpu', exploration_weight: float = 1.0, curriculum=None, quick_trainer=None, reward_loss_weight=0.01,
                 iso_weight: float = 0.01, comp_weight: float = 0.0, parallel_workers: int = 4,
                 early_stopping_patience: int = 5, early_stopping_min_delta: float = 0.001):
        super().__init__(action_space, None, exploration_weight)
        self.policy_value_net = policy_value_net
        self.device = device
        self.curriculum = curriculum
        self.quick_trainer = quick_trainer
        self.reward_loss_weight = reward_loss_weight
        # Reward shaping weights: penalize isolated neurons and optionally penalize connection count
        self.iso_weight = iso_weight
        self.comp_weight = comp_weight
        # Parallel processing settings
        self.parallel_workers = parallel_workers
        # Early stopping settings
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        # Cache for evaluations to avoid redundant computations on architectures with identical isolated hidden neurons
        self.evaluation_cache = {}
        # Evolutionary cycle tracking
        self.current_cycle = EvolutionaryCycle()

  
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

    def _value_from_quick_trainer(self, architecture: NeuralArchitecture, node_action: Action = None, is_simulation: bool = False) -> float:
        """Compute shaped value from quick_trainer (accuracy/loss) plus isolation/complexity terms.

        node_action: optional Action that produced the architecture (used to give a one-step grace to newly added neurons).
        is_simulation: when True, be conservative about granting grace (simulations often can't identify new neuron ids reliably).
        """
        if not self.quick_trainer:
            return np.random.uniform(0.0, 0.5)

        try:
            # Update the quick trainer's model with the new architecture
            self.quick_trainer.update_architecture(architecture)
            accuracy, loss = self.quick_trainer.train_and_evaluate(architecture)
        except Exception:
            return np.random.uniform(0.0, 0.5)

        base_reward = accuracy - self.reward_loss_weight * loss

        # Determine ignore ids for grace period: only when we have a node_action that added a neuron
        ignore_ids = set()
        if not is_simulation and node_action is not None:
            try:
                if node_action.action_type == ActionType.ADD_NEURON and architecture.next_neuron_id > 0:
                    # newly added neuron id should be last allocated id
                    ignore_ids.add(architecture.next_neuron_id - 1)
            except Exception:
                pass

        isolated = architecture.count_isolated_neurons(ignore_ids=ignore_ids, only_hidden=True)
        total_hidden = sum(1 for n in architecture.neurons.values() if n.neuron_type == NeuronType.HIDDEN and n.id not in ignore_ids)
        total_hidden = max(1, total_hidden)
        iso_term = isolated / total_hidden

        num_conn = architecture.num_connections()
        num_neurons = max(1, architecture.num_neurons())
        comp_term = num_conn / num_neurons

        return base_reward - self.iso_weight * iso_term - self.comp_weight * comp_term

    def _evaluate_node(self, node: NeuralMCTSNode, is_simulation: bool = False) -> float:
        """Evaluate a node using quick_trainer in supervised stage or the policy-value net.

        This is the shared evaluator used by both `search` and `search_with_beam` so logic is
        consistent and centralized.
        """

        # Determine stage
        if self.curriculum:
            train_params = self.curriculum.get_training_parameters()
            stage = train_params.get('stage', 'supervised')
        else:
            stage = 'supervised' if self.quick_trainer else 'policy'

        if stage == 'supervised' and self.quick_trainer:
            value = self._value_from_quick_trainer(node.architecture, node_action=node.action, is_simulation=is_simulation)
        else:
            # Ensure we have a policy_value for this node
            if node.policy_value is None:
                with torch.no_grad():
                    graph_data = self._prepare_graph_data(node.architecture)
                    node.policy_value = self.policy_value_net(graph_data)
            value = node.policy_value['value'].item()

        return value
    
    def search(self, initial_architecture: NeuralArchitecture, iterations: int = 100,
               temperature: float = 1.0) -> NeuralMCTSNode:
        """Run neural-guided MCTS search with evolutionary cycle tracking and early stopping"""
        root = NeuralMCTSNode(initial_architecture)

        # Get neural network predictions for root (skip in supervised stage for speed)
        if self.curriculum:
            train_params = self.curriculum.get_training_parameters()
            stage = train_params.get('stage', 'supervised')
            if stage != 'supervised':
                with torch.no_grad():
                    graph_data = self._prepare_graph_data(initial_architecture)
                    policy_value = self.policy_value_net(graph_data)
                    root.policy_value = policy_value
        else:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(initial_architecture)
                policy_value = self.policy_value_net(graph_data)
                root.policy_value = policy_value

        # Early stopping tracking
        best_values = []
        no_improvement_count = 0

        # Use the centralized evaluator
        print("Starting MCTS search...")
        for i in range(iterations):
            node = self._select(root)

            if not node.architecture.performance_metrics:
                # Leaf evaluation
                value = self._evaluate_node(node, is_simulation=False)
                node.architecture.performance_metrics = {'estimated_accuracy': value}
                node.value = value
                node.visits = 1
            else:
                node = self._expand(node)
                if node != self._select(root):
                    # We expanded; evaluate the expanded node
                    value = self._evaluate_node(node, is_simulation=False)
                else:
                    # No expansion (selection returned same node) -> simulate rollout
                    value = self._simulate(node)

            self._backpropagate(node, value)

            # Early stopping check
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

        # Select final action using temperature (visit-count based) to allow stochasticity
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
            print(f"MCTS FINAL ACTION: {action_str} (value: {final_node.value/final_node.visits:.3f})")
            print("MCTS search completed successfully")
            return final_node
        else:
            print("MCTS search completed: no valid action found")
            return None
        
    
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
        """Expand node using policy network priors"""
        # Determine whether to use policy or random based on curriculum
        if self.curriculum:
            train_params = self.curriculum.get_training_parameters()
            stage = train_params.get('stage', 'supervised')
            # In supervised stage, use random actions to avoid slow policy network calls
            if stage == 'supervised':
                use_policy = False
                skip_nn_calls = True
            else:
                use_policy = train_params['policy_mix_ratio'] > 0.0
                skip_nn_calls = False
        else:
            use_policy = False  # Default to random if no curriculum
            skip_nn_calls = True

        # Get neural network predictions if needed and not skipping
        if not node.policy_value and not skip_nn_calls:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(node.architecture)
                node.policy_value = self.policy_value_net(graph_data)

        # Use action manager to get valid actions with evolutionary cycle context
        action_manager = ActionManager(action_space=self.action_space)

        # Get valid actions with cycle-aware prioritization
        valid_actions = self.action_space.get_valid_actions(
            node.architecture, current_step=0, evolutionary_cycle=self.current_cycle
        )

        if not valid_actions:
            return node  # No valid actions available

        if use_policy and node.policy_value is not None:
            # Use policy network to select from valid actions
            action = action_manager.select_action(
                node.policy_value, node.architecture, exploration=True, use_policy=True
            )
            # Filter to only valid actions if needed
            if action not in valid_actions:
                action = np.random.choice(valid_actions)
        else:
            # Random selection from valid actions
            action = np.random.choice(valid_actions)

        # Apply action to create new architecture
        new_architecture = node._copy_architecture()
        success = self.action_space.apply_action(new_architecture, action)

        if success:
            # Create child node
            child = NeuralMCTSNode(new_architecture, parent=node, action=action)

            # Compute prior probability for this action from policy network (skip in supervised)
            if skip_nn_calls:
                child.prior_prob = 1.0  # Default prior for supervised stage
            else:
                child.prior_prob = self._compute_action_prior_prob(node.policy_value, action, node.architecture)

            # Get neural network predictions for child (skip in supervised)
            if not skip_nn_calls:
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
                    has_in = any(conn.target_neuron == neuron_id for conn in architecture.connections.values())
                    has_out = any(conn.source_neuron == neuron_id for conn in architecture.connections.values())
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

    def _simulate(self, node: NeuralMCTSNode, max_depth: int = 5) -> float:
        """Simulate from a node using parallel random rollouts and evaluate with the policy-value net.

        Uses multiple processes to run parallel rollouts for better exploration.
        """
        if self.parallel_workers <= 1:
            # Fallback to single rollout if parallel disabled
            return self._single_rollout(node, max_depth)

        # Run multiple parallel rollouts
        rollout_values = []
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Convert architecture to serializable dict using the new method
            arch_data = node.architecture.to_serializable_dict()

            # Submit parallel rollout tasks - pass only serializable data
            futures = [executor.submit(self._single_rollout_process, arch_data, max_depth,
                                     self.reward_loss_weight, self.iso_weight, self.comp_weight)
                      for _ in range(self.parallel_workers)]

            # Collect results
            for future in futures:
                try:
                    value = future.result(timeout=30)  # 30 second timeout per rollout
                    rollout_values.append(value)
                except Exception as e:
                    print(f"Rollout failed: {e}")
                    traceback.print_exc()
                    rollout_values.append(0.0)  # Default value for failed rollouts

        # Return average of parallel rollouts
        return np.mean(rollout_values) if rollout_values else 0.0

    def _single_rollout(self, node: NeuralMCTSNode, max_depth: int = 5) -> float:
        """Single rollout simulation from a node using random actions.

        This is the core rollout logic that can be parallelized.
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
        # Skip neural network evaluation in supervised stage for speed
        if self.curriculum:
            train_params = self.curriculum.get_training_parameters()
            stage = train_params.get('stage', 'supervised')
            if stage == 'supervised':
                # Train the architecture to get real performance
                if self.quick_trainer:
                    accuracy, loss = self.quick_trainer.train_and_evaluate(current_arch)
                    base_reward = accuracy - self.reward_loss_weight * loss

                    # Apply connectivity / isolation penalties on simulated architecture
                    isolated = current_arch.count_isolated_neurons(ignore_ids=set(), only_hidden=True)
                    total_hidden = max(1, sum(1 for n in current_arch.neurons.values() if n.neuron_type == NeuronType.HIDDEN))
                    iso_term = isolated / total_hidden

                    num_conn = current_arch.num_connections()
                    num_neurons = max(1, current_arch.num_neurons())
                    comp_term = num_conn / num_neurons

                    value = base_reward - self.iso_weight * iso_term - self.comp_weight * comp_term
                else:
                    value = np.random.uniform(0.0, 0.5)  # Fallback if no trainer
            else:
                with torch.no_grad():
                    graph_data = self._prepare_graph_data(current_arch)
                    policy_value = self.policy_value_net(graph_data)
                    value = policy_value['value'].item()
        else:
            with torch.no_grad():
                graph_data = self._prepare_graph_data(current_arch)
                policy_value = self.policy_value_net(graph_data)
                value = policy_value['value'].item()

        return value

    @staticmethod
    def _single_rollout_process(architecture_data: dict, max_depth: int, reward_loss_weight: float,
                               iso_weight: float, comp_weight: float) -> float:
        """Static method for parallel rollout execution - uses serializable dict data"""
        try:
            # Import here to avoid circular imports in multiprocessing
            from blueprint_modules.action import ActionSpace
            from blueprint_modules.network import NeuralArchitecture, NeuronType, ActivationType

            # Create a fresh action space for this process
            action_space = ActionSpace()

            # Reconstruct architecture from serializable data
            current_arch = NeuralArchitecture()
            current_arch.neurons = {}
            current_arch.connections = []
            current_arch.next_neuron_id = architecture_data['next_neuron_id']

            # Reconstruct neurons
            for neuron_data in architecture_data['neurons']:
                neuron = Neuron(
                    id=neuron_data['id'],
                    neuron_type=NeuronType(neuron_data['neuron_type']),
                    activation=ActivationType(neuron_data['activation']),
                    layer_position=neuron_data['layer_position'],
                    bias=neuron_data['bias']
                )
                current_arch.neurons[neuron.id] = neuron

            # Reconstruct connections
            for conn_data in architecture_data['connections']:
                conn = Connection(
                    source_id=conn_data['source_id'],
                    target_id=conn_data['target_id'],
                    weight=conn_data['weight'],
                    enabled=conn_data['enabled']
                )
                current_arch.connections.append(conn)

            # Perform rollout
            for _ in range(max_depth):
                valid_actions = action_space.get_valid_actions(current_arch)
                if not valid_actions:
                    break

                # Randomly choose an action and apply
                action = np.random.choice(valid_actions)
                action_space.apply_action(current_arch, action)

            # Simple evaluation - use architecture properties as proxy
            # This avoids needing to pickle complex objects like trainers and networks
            connected_frac = current_arch.connected_fraction()
            num_neurons = current_arch.num_neurons()
            num_connections = current_arch.num_connections()

            # Size penalty (balance complexity)
            size_score = 1.0 / (1.0 + 0.01 * num_neurons)

            # Layer diversity bonus
            layer_positions = set(n.layer_position for n in current_arch.neurons.values())
            layer_bonus = min(1.0, len(layer_positions) / 5.0)

            # Combine scores
            proxy_score = 0.4 * connected_frac + 0.3 * size_score + 0.3 * layer_bonus

            # Add small random noise
            import random
            noise = random.uniform(-0.05, 0.05)

            return max(0.0, min(1.0, proxy_score + noise))

        except Exception as e:
            # Return default value on any error
            return 0.0
