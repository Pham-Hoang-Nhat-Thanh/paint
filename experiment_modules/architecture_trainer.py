import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from typing import Dict, List, Any
import torch.nn.functional as F
import json
import logging
import traceback
import warnings
import networkx as nx
import matplotlib
# Use Agg backend for headless/tmux environments (no display needed)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blueprint_modules.network import Neuron, Connection
from blueprint_modules.network import ActivationType

# Suppress TF32 deprecation warnings
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior")

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

from blueprint_modules.network import NeuralArchitecture
from blueprint_modules.action import ActionSpace, Action
from blueprint_modules.network_trainer import QuickTrainer
from architect_modules.guided_mcts import NeuralMCTS
from architect_modules.policy_value_net import UnifiedPolicyValueNetwork, ActionManager
from architect_modules.training_curriculum import TrainingCurriculum, PolicyValueLoss
from .experience_replay import ExperienceReplay
from .config import OverallConfig

class ArchitectureTrainer:
    """Main class that orchestrates the complete training process"""
    
    def __init__(self, config: OverallConfig, train_loader, test_loader):
        self.config = config
        self.device = torch.device(config.device)
        
        # Data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Core components
        self.policy_value_net = UnifiedPolicyValueNetwork(
            node_feature_dim=config.model.node_feature_dim,
            hidden_dim=config.model.hidden_dim,
            max_neurons=config.model.max_neurons,
            num_actions=config.model.num_actions,
            num_activations=config.model.num_activations
        ).to(self.device)
        
        self.action_space = ActionSpace(
            max_neurons=config.search.max_neurons,
            max_connections=config.search.max_connections,
            max_steps_per_episode=config.search.max_steps_per_episode,
            connection_candidate_multiplier=config.search.connection_candidate_multiplier
        )
        self.action_manager = ActionManager(
            max_neurons=config.model.max_neurons,
            exploration_boost=config.search.action_exploration_boost
        )
        self.quick_trainer = QuickTrainer(
            train_loader, test_loader,
            device=self.device,
            max_epochs=config.search.quick_train_epochs,
            use_mixed_precision=config.use_mixed_precision,
            eval_samples=2000  # Limit evaluation to 2000 samples for speed (80% faster eval)
        )

        # Training infrastructure
        self.curriculum = TrainingCurriculum(
            supervised_episodes=config.supervised.num_episodes,
            mixed_episodes=config.mixed.num_episodes,
            self_play_episodes=config.self_play.num_episodes
        )
        
        self.neural_mcts = NeuralMCTS(
            action_space=self.action_space,
            policy_value_net=self.policy_value_net,
            device=self.device,
            exploration_weight=config.mcts.exploration_weight,
            curriculum=self.curriculum,
            quick_trainer=self.quick_trainer,
            reward_loss_weight=config.search.reward_loss_weight,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
        )
        
        self.experience_buffer = ExperienceReplay(
            capacity=config.self_play.replay_buffer_size
        )
        
        self.loss_fn = PolicyValueLoss(
            value_weight=config.supervised.value_loss_weight,
            entropy_weight=config.supervised.entropy_weight
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=config.supervised.learning_rate,
            weight_decay=config.supervised.weight_decay
        )
        
        # Training state
        self.episode = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # Create directories
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        # File logger + JSONL metrics (disabled for speed)
        self.log_path = os.path.join(config.log_dir, "training.log")
        self.metrics_path = os.path.join(config.log_dir, "training_metrics.jsonl")

        # Configure logger (avoid duplicate handlers) - disabled for speed
        self.logger = logging.getLogger(f"ArchitectureTrainer")
        self.logger.setLevel(logging.WARNING)  # Only log warnings and errors
        # Add file handler if not present
        abs_log_path = os.path.abspath(self.log_path)
        if not any(isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, 'baseFilename', '')) == abs_log_path
                   for h in self.logger.handlers):
            fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(fh)

        print(f"ArchitectureTrainer initialized on {self.device}")
        print(f"Total episodes: {self.curriculum.total_episodes}")
    
    def cleanup(self):
        """Clean up trainer resources"""
        # Clean up neural MCTS
        if hasattr(self.neural_mcts, 'cleanup'):
            self.neural_mcts.cleanup()
        
        # Clear experience buffer
        if hasattr(self.experience_buffer, 'clear'):
            self.experience_buffer.clear()
        
        # Clear training history to free memory
        self.training_history.clear()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Trainer cleanup completed")
    
    def run_training_episode(self) -> Dict[str, Any]:
        """Run one complete training episode
        
        Note: MCTS tree (mcts_root) is maintained within this episode but NOT saved in checkpoints.
        Each new episode starts with a fresh tree (mcts_root=None). This is intentional because:
        - Each episode explores a different architecture design trajectory
        - Tree reuse is beneficial within an episode but not across episodes
        - Starting fresh each episode provides exploration diversity
        """
        print(f"Starting episode {self.episode} (stage: {self.curriculum.get_stage()})")
        # Initialize architecture
        current_arch = NeuralArchitecture()
        
        episode_experiences = []
        episode_rewards = []
        episode_steps = 0
        step_times = []  # Track time for each step
        
        # Get current training parameters
        train_params = self.curriculum.get_training_parameters()
        
        # Initialize MCTS tree reuse (for persistent tree across steps WITHIN this episode)
        # Not saved in checkpoints - each episode starts fresh for exploration diversity
        mcts_root = None  # Will store the selected child node for next iteration
        
        # Run search until convergence or max steps
        for step in range(self.config.search.max_steps_per_episode):
            print(f"Step {step + 1}/{self.config.search.max_steps_per_episode}:")
            step_start_time = time.time()
            # Decide whether to use policy network or MCTS
            use_policy = np.random.random() < train_params["policy_mix_ratio"]

            if use_policy and train_params["stage"] != "supervised":
                # Use policy network directly (faster than MCTS)
                next_action = self._get_policy_action(current_arch, train_params["temperature"])
                # Don't reuse tree when using policy network (tree is out of sync)
                mcts_root = None
                search_root = None
            else:  
                # Returns: (selected_child, search_root)
                best_node, search_root = self.neural_mcts.search(
                    current_arch,
                    iterations=self.config.mcts.num_simulations,
                    temperature=self.config.mcts.temperature,
                    reuse_root=mcts_root  # Reuse tree from previous step
                )
                
                next_action = best_node.action if best_node else None

            if next_action is None:
                print("No valid action found, terminating episode")
                break

            new_arch = best_node.architecture
            if new_arch is None:
                # Debug: Print detailed information about the failure
                print(f"ACTION FAILED: {next_action.action_type.name}")
                print(f"  Source neuron: {next_action.source_neuron}")
                print(f"  Target neuron: {next_action.target_neuron}")
                print(f"  Activation: {next_action.activation}")
                print(f"  Current arch neurons: {sorted(current_arch.neurons.keys())}")
                print(f"  Current arch connections: {len(current_arch.connections)}")
                if next_action.source_neuron is not None:
                    print(f"  Source exists: {next_action.source_neuron in current_arch.neurons}")
                if next_action.target_neuron is not None:
                    print(f"  Target exists: {next_action.target_neuron in current_arch.neurons}")
                
                mcts_root = None  # Reset tree on failure
                traceback.print_exc()
                raise RuntimeError("Action application failed")

            # Update the quick trainer's model with the new architecture
            self.quick_trainer.update_architecture(new_arch)

            # Draw architecture diagram after action (save every step for debugging)
            # In tmux, plt.savefig buffers may not flush immediately, so we force close/flush
            self._draw_architecture_diagram(new_arch, step)
            reward = self._evaluate_architecture(new_arch)

            # Calculate timing information
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_times.append(step_duration)

            print(f"Step completed: Reward = {reward:.4f} | Step time: {step_duration:.2f}s")

            # Store experience with MCTS visit distribution
            experience = self._create_experience(
                current_arch, next_action, reward, new_arch, train_params["stage"],
                search_root=search_root  # Contains visit distribution from MCTS
            )
            episode_experiences.append(experience)
            episode_rewards.append(reward)
            
            # Update current architecture
            current_arch = new_arch
            episode_steps += 1
            
            # Update tree reuse: best_node IS the child node we want to reuse as next root
            # Its architecture was created during expansion and matches current_arch
            if best_node is not None:
                mcts_root = best_node
                # Verify tree reuse correctness
                root_neurons = sorted(mcts_root.architecture.neurons.keys())
                current_neurons = sorted(current_arch.neurons.keys())
                if root_neurons != current_neurons:
                    print(f"WARNING: Tree reuse architecture mismatch detected!")
                    print(f"  mcts_root neurons: {root_neurons}")
                    print(f"  current_arch neurons: {current_neurons}")
                    print(f"  Difference: {set(current_neurons) - set(root_neurons)}")
                else:
                    print(f"  Tree reuse OK: {len(root_neurons)} neurons match (visits: {best_node.visits})")

            # Check termination conditions
            if self._should_terminate_episode(current_arch, step, reward):
                print(f"Episode termination condition met at step {step}")
                break
        # Log tree reuse statistics
        if mcts_root is not None and hasattr(mcts_root, 'visits'):
            print(f"Episode ended with MCTS tree containing {mcts_root.visits} total visits")
            print(f"Tree reuse benefit: accumulated {mcts_root.visits} visits across {episode_steps} steps")
        
        # Process episode results
        episode_metrics = self._process_episode_results(
            current_arch, episode_rewards, episode_experiences, episode_steps
        )
        
        # CRITICAL: Update all experiences with final episode reward (AlphaZero-style credit assignment)
        # This tells the value network: "If you reach state S in this trajectory, 
        # the final outcome was R_final" rather than just the immediate reward
        final_reward = episode_rewards[-1] if episode_rewards else 0.0
        for exp in episode_experiences:
            exp['value_target'] = final_reward  # Replace immediate reward with final outcome
        
        # Add experiences to replay buffer
        for exp in episode_experiences:
            priority = self._compute_experience_priority(exp, episode_metrics)
            self.experience_buffer.add(exp, priority)

        return episode_metrics
    
    def _get_policy_action(self, architecture: NeuralArchitecture, temperature: float) -> Action:
        """Get action from policy network"""
        with torch.no_grad():
            graph_data = self._prepare_graph_data(architecture)
            policy_output = self.policy_value_net(graph_data)
            
            # Use exploration based on temperature
            exploration = temperature > 0.1
            action = self.action_manager.select_action(
                policy_output, architecture, exploration=exploration
            )
            
            return action
    
    def _actions_match(self, action1: Action, action2: Action) -> bool:
        """Check if two actions are equivalent"""
        if action1 is None or action2 is None:
            return False
        
        # Must have same action type
        if action1.action_type != action2.action_type:
            return False
        
        # Check type-specific parameters
        from blueprint_modules.action import ActionType
        
        if action1.action_type in [ActionType.ADD_CONNECTION, ActionType.REMOVE_CONNECTION]:
            # Connection actions: must match source and target
            return (action1.source_neuron == action2.source_neuron and 
                    action1.target_neuron == action2.target_neuron)
        
        elif action1.action_type == ActionType.MODIFY_ACTIVATION:
            # Activation modification: must match neuron and new activation
            return (action1.source_neuron == action2.source_neuron and
                    action1.activation == action2.activation)
        
        elif action1.action_type == ActionType.REMOVE_NEURON:
            # Neuron removal: must match neuron ID
            return action1.source_neuron == action2.source_neuron
        
        elif action1.action_type == ActionType.ADD_NEURON:
            # Neuron addition: only activation type matters (ID assigned later)
            return action1.activation == action2.activation
        
        return False
    
    def _apply_action(self, architecture: NeuralArchitecture, action: Action) -> NeuralArchitecture:
        """Apply action to create new architecture"""
        # Create a copy of the architecture
        new_arch = NeuralArchitecture()
        # The default NeuralArchitecture() constructor initializes a MNIST base (inputs/outputs
        # and some initial connections). When creating a copy we don't want those default
        # neurons/connections to be present, so clear them before copying the real architecture
        # to avoid duplicating or accumulating connections.
        new_arch.neurons = {}
        new_arch.connections = []
        new_arch.next_neuron_id = 0
        
        # Copy neurons
        for neuron_id, neuron in architecture.neurons.items():
            new_arch.neurons[neuron_id] = Neuron(
                id=neuron.id,
                neuron_type=neuron.neuron_type,
                activation=neuron.activation,
                layer_position=neuron.layer_position,
                bias=neuron.bias
            )
            new_arch.next_neuron_id = max(new_arch.next_neuron_id, neuron_id + 1)

        # Copy connections
        for conn in architecture.connections:
            new_arch.connections.append(Connection(
                source_id=conn.source_id,
                target_id=conn.target_id,
                weight=conn.weight,
                enabled=conn.enabled
            ))
        
        new_arch.performance_metrics = architecture.performance_metrics.copy()
        
        # Apply the action
        success = self.action_space.apply_action(new_arch, action)
        return new_arch if success else None
    
    def _evaluate_architecture(self, architecture: NeuralArchitecture) -> float:
        """Quick evaluation of architecture"""
        try:
            # Use quick training for evaluation
            accuracy, loss = self.quick_trainer.train_and_evaluate(architecture)
            # Compute composite reward: accuracy - weight * loss
            reward = accuracy - self.config.search.reward_loss_weight * loss
            return reward
        except Exception as e:
            print(f"Evaluation error: {e}")
            traceback.print_exc()
            return 0.0
    
    def _prepare_graph_data(self, architecture: NeuralArchitecture) -> Dict:
        """Prepare graph data for neural network"""
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
    
    def _batch_graph_data(self, graph_data_list: List[Dict]) -> Dict:
        """Batch multiple graph data using PyTorch Geometric batching strategy
        
        PyG batching: Concatenate all graphs into a single disconnected graph
        - Node features: cat along node dimension [total_nodes, feature_dim]
        - Edge indices: offset by cumulative node counts
        - Add batch tensor to track which nodes belong to which graph
        - This avoids padding and is more memory efficient
        """
        if not graph_data_list:
            return {}
        
        if len(graph_data_list) == 1:
            # No batching needed for single graph
            gd = graph_data_list[0]
            return {
                'node_features': gd['node_features'],  # [1, num_nodes, features]
                'edge_index': gd['edge_index'],
                'edge_weights': gd['edge_weights'],
                'layer_positions': gd['layer_positions'],  # [1, num_nodes]
                'batch': torch.zeros(gd['node_features'].shape[1], dtype=torch.long, device=self.device),  # All nodes belong to graph 0
                'num_graphs': 1
            }
        
        batch_size = len(graph_data_list)
        
        # Collect all node features, edge indices, and create batch tensor
        all_node_features = []
        all_layer_positions = []
        all_edge_indices = []
        all_edge_weights = []
        batch_tensor = []
        
        node_offset = 0
        for graph_idx, gd in enumerate(graph_data_list):
            # Remove batch dimension [1, num_nodes, features] -> [num_nodes, features]
            node_feats = gd['node_features'].squeeze(0)
            layer_pos = gd['layer_positions'].squeeze(0)
            num_nodes = node_feats.shape[0]
            
            all_node_features.append(node_feats)
            all_layer_positions.append(layer_pos)
            
            # Offset edge indices
            if gd['edge_index'].shape[1] > 0:
                offset_edges = gd['edge_index'] + node_offset
                all_edge_indices.append(offset_edges)
                all_edge_weights.append(gd['edge_weights'])
            
            # Track which graph each node belongs to
            batch_tensor.append(torch.full((num_nodes,), graph_idx, dtype=torch.long, device=self.device))
            
            node_offset += num_nodes
        
        # Concatenate everything
        batched_node_features = torch.cat(all_node_features, dim=0)  # [total_nodes, features]
        batched_layer_positions = torch.cat(all_layer_positions, dim=0)  # [total_nodes]
        batch_tensor = torch.cat(batch_tensor, dim=0)  # [total_nodes]
        
        if all_edge_indices:
            batched_edge_index = torch.cat(all_edge_indices, dim=1)  # [2, total_edges]
            batched_edge_weights = torch.cat(all_edge_weights, dim=0)  # [total_edges]
        else:
            batched_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            batched_edge_weights = torch.zeros(0, device=self.device)
        
        # Add batch dimension back to match expected format [batch_size, num_nodes, features]
        # But now each "batch item" contains nodes from all graphs, separated by batch tensor
        # We need to reshape for the transformer: [1, total_nodes, features]
        batched_node_features = batched_node_features.unsqueeze(0)  # [1, total_nodes, features]
        batched_layer_positions = batched_layer_positions.unsqueeze(0)  # [1, total_nodes]
        
        return {
            'node_features': batched_node_features,
            'edge_index': batched_edge_index,
            'edge_weights': batched_edge_weights,
            'layer_positions': batched_layer_positions,
            'batch': batch_tensor,  # [total_nodes] - which graph each node belongs to
            'num_graphs': batch_size
        }
    
    def _create_experience(self, state: NeuralArchitecture, action: Action, 
                          reward: float, next_state: NeuralArchitecture, stage: str,
                          search_root=None) -> Dict:
        """Create experience tuple with MCTS visit distribution for AlphaZero-style training"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,  # Immediate reward (will be replaced with final episode reward)
            'next_state': next_state,
            'stage': stage,
            'timestamp': time.time()
        }
        
        # Store MCTS visit distribution if available (for policy training)
        if search_root is not None and hasattr(search_root, 'children'):
            # Extract visit distribution from MCTS search
            visit_counts = [child.visits for child in search_root.children]
            total_visits = sum(visit_counts)
            
            if total_visits > 0:
                # Normalize to get probability distribution
                visit_distribution = [v / total_visits for v in visit_counts]
                actions = [child.action for child in search_root.children]
                
                experience['mcts_policy'] = visit_distribution
                experience['mcts_actions'] = actions
        
        return experience
    
    def _should_terminate_episode(self, architecture: NeuralArchitecture, 
                                 step: int, reward: float) -> bool:
        """Check if episode should terminate"""
        # Check max steps
        if step >= self.config.search.max_steps_per_episode - 1:
            return True
        
        # Check neuron limit
        if len(architecture.neurons) >= self.config.search.max_neurons:
            return True
        
        # Check if target accuracy reached
        if reward >= self.config.search.target_accuracy:
            return True
        
        return False
    
    def _process_episode_results(self, final_arch: NeuralArchitecture, 
                                rewards: List[float], experiences: List[Dict], 
                                steps: int) -> Dict[str, Any]:
        """Process and return episode metrics"""
        final_reward = rewards[-1] if rewards else 0.0
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        metrics = {
            'episode': self.episode,
            'stage': self.curriculum.get_stage(),
            'steps': steps,
            'final_accuracy': final_reward,
            'average_reward': avg_reward,
            'total_neurons': len(final_arch.neurons),
            'total_connections': len(final_arch.connections),
            'experiences': len(experiences),
            'best_accuracy': max(rewards) if rewards else 0.0
        }
        
        # Update best accuracy
        if final_reward > self.best_accuracy:
            self.best_accuracy = final_reward
            metrics['is_best'] = True
        else:
            metrics['is_best'] = False
            
        return metrics
    
    def _compute_experience_priority(self, experience: Dict, episode_metrics: Dict) -> float:
        """Compute priority for experience replay"""
        reward = experience['reward']
        final_accuracy = episode_metrics['final_accuracy']
        
        # Higher priority for experiences that lead to good final accuracy
        priority = reward + final_accuracy
        
        # Bonus for novel architectures (simple heuristic)
        neuron_count = len(experience['state'].neurons)
        connection_count = len(experience['state'].connections)
        novelty_bonus = (neuron_count + connection_count) / 1000
        
        return priority + novelty_bonus
    
    def train_on_batch(self, batch_size: int = 32):
        """Train policy-value network on a batch of experiences with size-grouped batching
        
        Strategy: Group experiences by architecture size, then batch within each group.
        This allows GPU batching for same-sized architectures while handling variable sizes.
        """
        if len(self.experience_buffer) < batch_size:
            print(f"  [Training] Skipping: replay buffer has {len(self.experience_buffer)} experiences, need {batch_size}")
            return None
        
        print(f"  [Training] Starting with {len(self.experience_buffer)} total experiences in buffer")
            
        # Sample batch
        experiences, indices, weights = self.experience_buffer.sample(batch_size)
        
        if not experiences:
            return None
        
        # Group experiences by architecture size (number of neurons)
        size_groups = {}  # size -> [(exp, weight, index), ...]
        for exp, weight, idx in zip(experiences, weights, indices):
            arch_size = len(exp['state'].neurons)
            if arch_size not in size_groups:
                size_groups[arch_size] = []
            size_groups[arch_size].append((exp, weight, idx))
        
        # Accumulate gradients across all groups
        total_loss = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        total_experiences = 0
        
        # Process each experience individually
        # Note: Current graph transformer architecture doesn't support true batching
        # because it pools all nodes together expecting batch_size in dim 0
        # Future optimization: Modify transformer to use PyG batch tensor for per-graph pooling
        for arch_size, group_data in size_groups.items():
            print(f"  [Training] Processing {len(group_data)} experiences with {arch_size} neurons")
            for exp_idx, (exp, weight, idx) in enumerate(group_data):
                # Prepare graph data and targets
                graph_data = self._prepare_graph_data(exp['state'])
                targets = self._create_targets(exp)
                
                # Forward pass for single experience
                predictions = self.policy_value_net(graph_data)
                
                # Compute weighted loss
                try:
                    loss = self.loss_fn(predictions, targets) * weight
                    total_loss += loss
                    
                    # Log individual components (for monitoring)
                    with torch.no_grad():
                        policy_loss_accum += F.cross_entropy(predictions['action_type'], 
                                                      targets['action_type'])
                        value_loss_accum += F.mse_loss(predictions['value'].squeeze(),
                                                targets['value'].squeeze())
                except Exception as e:
                    print(f"    [ERROR] Loss computation failed for experience {exp_idx}/{len(group_data)}")
                    print(f"    Action type: {targets.get('action_type', 'N/A')}")
                    print(f"    Has mcts_policy: {'mcts_policy' in targets}")
                    print(f"    Has mcts_actions: {'mcts_actions' in targets}")
                    raise
            
            total_experiences += len(group_data)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_value_net.parameters(), 
            self.config.supervised.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Update priorities
        new_priorities = [abs(exp['reward']) for exp in experiences]
        self.experience_buffer.update_priorities(indices, new_priorities)
        
        out = {
            'total_loss': total_loss.item() / total_experiences,
            'policy_loss': policy_loss_accum.item() / total_experiences,
            'value_loss': value_loss_accum.item() / total_experiences
        }
        print(f"  [Training] Completed: total_loss={out['total_loss']:.6f}, policy={out['policy_loss']:.6f}, value={out['value_loss']:.6f}")
        return out
    
    def _create_targets(self, experience: Dict) -> Dict:
        """Create training targets from experience (AlphaZero-style)"""
        action = experience['action']

        # Use final episode reward (value_target) instead of immediate reward
        # This provides proper credit assignment for early actions
        value_target = experience.get('value_target', experience['reward'])
        
        targets = {
            'action_type': torch.tensor([action.action_type.value]).to(self.device),
            'value': torch.tensor([value_target]).to(self.device)  # Use final outcome, not immediate reward
        }

        # Add optional targets for hierarchical policy
        if action.source_neuron is not None:
            targets['source_neuron'] = torch.tensor([action.source_neuron]).to(self.device)
        if action.target_neuron is not None:
            targets['target_neuron'] = torch.tensor([action.target_neuron]).to(self.device)
        if action.activation is not None:
            activation_idx = list(ActivationType).index(action.activation)
            targets['activation'] = torch.tensor([activation_idx]).to(self.device)

        # Store MCTS policy distribution if available (for improved policy learning)
        if 'mcts_policy' in experience and 'mcts_actions' in experience:
            targets['mcts_policy'] = experience['mcts_policy']
            targets['mcts_actions'] = experience['mcts_actions']

        return targets
    
    def run_training(self):
        """Run complete training process"""
        print("Starting architecture search training...")
        
        while not self.curriculum.is_complete():
            # Run training episode
            episode_metrics = self.run_training_episode()
            
            # Train on batch if we have enough experiences
            # Use supervised.train_interval (per-stage) rather than a nonexistent top-level alias
            train_interval = getattr(self.config, 'train_interval', None)
            if train_interval is None:
                train_interval = self.config.supervised.train_interval

            if self.episode > 0 and self.episode % train_interval == 0:
                # Train on all accumulated experiences in the buffer
                remaining = len(self.experience_buffer)
                while remaining > 0:
                    batch_size = min(self.config.supervised.batch_size, remaining)
                    train_metrics = self.train_on_batch(batch_size)
                    if train_metrics:
                        episode_metrics.update(train_metrics)
                    remaining = len(self.experience_buffer)

            # Immediate logging and checkpoint on improvement so artifacts appear promptly
            try:
                # Log this episode right away (helps for debugging short runs)
                self._log_episode(episode_metrics)
                # If this episode produced a new best architecture, save a checkpoint immediately
                if episode_metrics.get('is_best'):
                    self._save_checkpoint()
            except Exception as e:
                print(f"Immediate log/checkpoint failed: {e}")
                traceback.print_exc()

            # Checkpoint if needed (before updating episode counter)
            if self.curriculum.should_checkpoint(self.config.checkpoint_interval):
                self._save_checkpoint()

            # Update curriculum
            self.curriculum.step()
            self.episode += 1

            # Update learning rate
            new_lr = self.curriculum.get_learning_rate()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # Log progress (less frequently to reduce I/O)
            self.training_history.append(episode_metrics)
            
            # Early stopping if we found a good architecture
            if episode_metrics['final_accuracy'] >= self.config.search.target_accuracy:
                print(f"Target accuracy reached! Stopping training.")
                break
        
        print("Training completed!")
        return self.training_history
    
    def _log_episode(self, metrics: Dict):
        """Log episode metrics"""
        if self.episode % self.config.log_interval == 0:
            stage = metrics['stage']
            accuracy = metrics['final_accuracy']
            neurons = metrics['total_neurons']
            connections = metrics['total_connections']
            
            print(f"Episode {self.episode} ({stage}): "
                  f"Accuracy={accuracy:.4f}, Neurons={neurons}, "
                  f"Connections={connections}, Best={self.best_accuracy:.4f}")

            # Structured file logging
            try:
                self.logger.info(f"Episode {self.episode} ({stage}) Accuracy={accuracy:.4f} "
                                 f"Neurons={neurons} Connections={connections} Best={self.best_accuracy:.4f}")
                # Append JSONL metrics
                with open(self.metrics_path, 'a', encoding='utf-8') as mf:
                    mf.write(json.dumps(metrics) + "\n")
            except Exception as e:
                # Fall back to console error if file write fails
                print(f"Failed to write logs: {e}")
                traceback.print_exc()
    
    def _save_checkpoint(self):
        """Save training checkpoint
        
        Note: MCTS tree state (mcts_root) is NOT saved because:
        - Tree is episode-scoped (resets between episodes)
        - Each episode should start with fresh tree for exploration diversity
        - Tree reuse is only beneficial within a single episode
        - Saving trees would be memory-intensive and not beneficial
        """
        checkpoint = {
            'episode': self.episode,
            'policy_value_net_state': self.policy_value_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'curriculum_state': self.curriculum.get_state(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'experience_buffer': self.experience_buffer,
            'config': self.config,
            # Add parallelization-related state (configuration, not tree)
            'neural_mcts_state': {
                'evaluation_cache': self.neural_mcts.evaluation_cache,
                'current_cycle': self.neural_mcts.current_cycle,
                'exploration_weight': self.neural_mcts.exploration_weight,
                'reward_loss_weight': self.neural_mcts.reward_loss_weight,
                'iso_weight': self.neural_mcts.iso_weight,
                'comp_weight': self.neural_mcts.comp_weight
                # Note: mcts_root tree is NOT saved (episode-scoped, intentionally reset)
            }
        }

        filename = f"checkpoint_ep{self.episode}.pth"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy_value_net.load_state_dict(checkpoint['policy_value_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.curriculum.set_state(checkpoint['curriculum_state'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
        self.experience_buffer = checkpoint['experience_buffer']
        self.episode = checkpoint['episode'] + 1  # Start next episode

        # Restore parallelization-related state
        if 'neural_mcts_state' in checkpoint:
            mcts_state = checkpoint['neural_mcts_state']
            self.neural_mcts.evaluation_cache = mcts_state.get('evaluation_cache', {})
            self.neural_mcts.current_cycle = mcts_state.get('current_cycle', self.neural_mcts.current_cycle)
            self.neural_mcts.exploration_weight = mcts_state.get('exploration_weight', self.neural_mcts.exploration_weight)
            self.neural_mcts.reward_loss_weight = mcts_state.get('reward_loss_weight', self.neural_mcts.reward_loss_weight)
            self.neural_mcts.iso_weight = mcts_state.get('iso_weight', self.neural_mcts.iso_weight)
            self.neural_mcts.comp_weight = mcts_state.get('comp_weight', self.neural_mcts.comp_weight)

        # Ensure curriculum episode_count matches the resumed episode
        self.curriculum.episode_count = self.episode

        print(f"Checkpoint loaded from episode {checkpoint['episode']}, resuming from episode {self.episode}")

    def _draw_architecture_diagram(self, architecture: NeuralArchitecture, step: int):
        """Draw a diagram of the neural architecture after an action
        
        Note: In tmux/headless environments, matplotlib may buffer output.
        We explicitly close figures and flush to ensure files are written.
        """
        try:
            # Create a directed graph
            G = nx.DiGraph()

            # Add nodes with attributes
            for neuron_id, neuron in architecture.neurons.items():
                node_label = f"N{neuron_id}\n{neuron.neuron_type.value[:3]}\n{neuron.activation.value[:3]}"
                node_color = {
                    'input': 'lightblue',
                    'hidden': 'lightgreen',
                    'output': 'lightcoral'
                }.get(neuron.neuron_type.value, 'gray')

                G.add_node(neuron_id,
                          label=node_label,
                          color=node_color,
                          layer=neuron.layer_position,
                          type=neuron.neuron_type.value)

            # Add edges
            for conn in architecture.connections:
                if conn.enabled:
                    G.add_edge(conn.source_id, conn.target_id,
                              weight=abs(conn.weight),
                              color='red' if conn.weight < 0 else 'blue')

            # Create positions based on layer positions
            pos = {}
            layers = {}
            for neuron_id, neuron in architecture.neurons.items():
                layer_pos = neuron.layer_position
                if layer_pos not in layers:
                    layers[layer_pos] = []
                layers[layer_pos].append(neuron_id)

            # Sort layers
            sorted_layers = sorted(layers.keys())
            for layer_idx, layer_pos in enumerate(sorted_layers):
                neurons_in_layer = layers[layer_pos]
                # Sort neurons within layer by ID for consistent layout
                neurons_in_layer.sort()
                y_positions = np.linspace(1, -1, len(neurons_in_layer))
                for i, neuron_id in enumerate(neurons_in_layer):
                    pos[neuron_id] = (layer_pos, y_positions[i])

            # Draw the graph
            plt.figure(figsize=(12, 8))

            # Draw nodes
            node_colors = [G.nodes[n]['color'] for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                 node_size=800, alpha=0.8)

            # Draw edges
            edges = G.edges()
            edge_colors = [G[u][v]['color'] for u, v in edges]
            edge_weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                 width=[w*2 for w in edge_weights],
                                 alpha=0.6, arrows=True, arrowsize=20)

            # Draw labels
            labels = {n: G.nodes[n]['label'] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

            # Add title and info
            plt.title(f'Neural Architecture - Episode {self.episode}, Step {step}\n'
                     f'Neurons: {len(architecture.neurons)}, Connections: {len(architecture.connections)}')
            plt.axis('off')
            plt.tight_layout()

            # Save the diagram
            os.makedirs(f'architecture_diagrams/ep{self.episode:03d}', exist_ok=True)
            filename = f'architecture_diagrams/ep{self.episode:03d}/step{step:02d}.jpg'
            
            # Save and explicitly flush to disk (important for tmux/headless)
            plt.savefig(filename, dpi=100, bbox_inches='tight', format='jpg')
            plt.close('all')  # Close all figures to free memory
            
            # Force filesystem sync (ensure file is written to disk)
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            print(f"      Architecture diagram saved: {filename}", flush=True)

        except Exception as e:
            print(f"Failed to draw architecture diagram: {e}")
            traceback.print_exc()
