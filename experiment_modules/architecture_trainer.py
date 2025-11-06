import torch
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
from PIL import Image, ImageDraw
from blueprint_modules.network import Neuron, Connection
from blueprint_modules.action import ActionType

# Suppress TF32 deprecation warnings
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior")

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

from blueprint_modules.network import NeuralArchitecture, ActivationType
from blueprint_modules.action import ActionSpace, Action
from blueprint_modules.network_trainer import QuickTrainer
from architect_modules.guided_mcts import NeuralMCTS
from architect_modules.policy_value_net import UnifiedPolicyValueNetwork, ActionManager
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
        
        self.neural_mcts = NeuralMCTS(
            action_space=self.action_space,
            policy_value_net=self.policy_value_net,
            device=self.device,
            exploration_weight=config.mcts.exploration_weight,
            iso_weight=config.search.iso_weight if hasattr(config.search, 'iso_weight') else 0.01,
            comp_weight=config.search.comp_weight if hasattr(config.search, 'comp_weight') else 0.0,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            max_children=config.mcts.max_children if hasattr(config.mcts, 'max_children') else 50
        )
        
        # QuickTrainer for final episode evaluation
        self.quick_trainer = QuickTrainer(
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            max_epochs=self.config.search.final_train_epochs,
        )
        
        self.experience_buffer = ExperienceReplay(
            capacity=10000  # Replay buffer size
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=config.model.learning_rate if hasattr(config.model, 'learning_rate') else 1e-3,
            weight_decay=1e-4
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

        # Configure logger for episode metrics
        self.logger = logging.getLogger(f"ArchitectureTrainer_{self.episode}_{time.time()}")
        self.logger.setLevel(logging.INFO)  # Log INFO and above (INFO, WARNING, ERROR)
        self.logger.propagate = False  # Don't propagate to root logger to avoid duplicate logs
        
        # Add file handler for training logs
        abs_log_path = os.path.abspath(self.log_path)
        if not any(isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, 'baseFilename', '')) == abs_log_path
                   for h in self.logger.handlers):
            fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(fh)

        print(f"ArchitectureTrainer initialized on {self.device}")
        print(f"Starting AlphaZero-style MCTS training (no curriculum, pure self-play)")
    
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
        """Run one complete training episode using AlphaZero-style MCTS."""
        print(f"Starting episode {self.episode}")
        # Initialize architecture
        current_arch = NeuralArchitecture()
        
        episode_experiences = []
        episode_rewards = []
        episode_steps = 0
        step_times = []  # Track time for each step
        
        # Initialize MCTS tree reuse (for persistent tree across steps WITHIN this episode)
        mcts_root = None  # Will store the selected child node for next iteration
        
        # Run MCTS-guided search until convergence or max steps
        for step in range(self.config.search.max_steps_per_episode):
            print(f"Step {step + 1}/{self.config.search.max_steps_per_episode}:")
            step_start_time = time.time()
            
            # MCTS search (always, no policy_mix_ratio since we're pure AlphaZero now)
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

            # Draw architecture diagram after action (save every step for debugging)
            self._draw_architecture_diagram(new_arch, step)
            reward = self._evaluate_architecture(new_arch)

            # Calculate timing information
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_times.append(step_duration)

            print(f"Step completed: Reward = {reward:.4f} | Step time: {step_duration:.2f}s")

            # Store experience with MCTS visit distribution
            experience = self._create_experience(
                current_arch, next_action, reward, new_arch,
                search_root=search_root  # Contains visit distribution from MCTS
            )
            episode_experiences.append(experience)
            episode_rewards.append(reward)
            
            # Update current architecture
            current_arch = new_arch
            episode_steps += 1
            
            # Update tree reuse: best_node IS the child node we want to reuse as next root
            if best_node is not None:
                mcts_root = best_node

            # Check termination conditions
            if self._should_terminate_episode(current_arch, step, reward):
                print(f"Episode termination condition met at step {step}")
                break
        
        # Process episode results
        episode_metrics = self._process_episode_results(
            current_arch, episode_rewards, episode_experiences, episode_steps
        )
        
        # Evaluate final architecture with QuickTrainer to show true progress
        final_eval = self._evaluate_final_architecture(current_arch)
        print(f"Final Architecture Evaluation: Accuracy={final_eval['final_accuracy']:.4f}, Loss={final_eval['final_loss']:.4f} ({final_eval['method']})")
        episode_metrics.update(final_eval)
        
        # =====CALCULATING FINAL REWARD=====
        # CRITICAL: Update all experiences with final episode reward (use true final evaluation)
        accuracy = float(final_eval.get('final_accuracy', episode_rewards[-1] if episode_rewards else 0.0))
        loss = float(final_eval.get('final_loss', 0.0))
        # normalize loss to (0,1)
        loss_norm = loss / (1.0 + loss)

        num_neurons = len(current_arch.neurons)
        num_connections = len(current_arch.connections)
        max_neurons = getattr(self.config.search, 'max_neurons', 0)
        max_connections = getattr(self.config.search, 'max_connections', 0)
        complexity_norm = ((num_neurons / max_neurons) + (num_connections / max_connections)) / 2.0 if (max_neurons > 0 and max_connections > 0) else 0.0

        raw_reward = self.config.search.reward_accuracy_weight * accuracy - self.config.search.reward_loss_weight * loss_norm - self.config.search.reward_complexity_weight * complexity_norm
        final_reward = max(0.0, raw_reward)  # Ensure non-negative reward

        for exp in episode_experiences:
            exp['value_target'] = final_reward
        
        # Add experiences to replay buffer
        for exp in episode_experiences:
            priority = self._compute_experience_priority(exp, episode_metrics)
            self.experience_buffer.add(exp, priority)

        # ===== LOG EPISODE RESULTS =====
        self._log_episode(episode_metrics)
        
        # ===== SAVE CHECKPOINT =====
        self._save_checkpoint()

        return episode_metrics
    
    
    def _actions_match(self, action1: Action, action2: Action) -> bool:
        """Check if two actions are equivalent"""
        if action1 is None or action2 is None:
            return False
        
        # Must have same action type
        if action1.action_type != action2.action_type:
            return False
        
        # Check type-specific parameters
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
        """Evaluate architecture using policy-value network (lightweight, per-step)"""
        try:
            # Use policy-value network for fast evaluation during episode
            with torch.no_grad():
                graph_data = self._prepare_graph_data(architecture)
                policy_output = self.policy_value_net(graph_data)
                # Value output is the estimated accuracy
                value = policy_output['value'].item()
                return value
        except Exception as e:
            print(f"Evaluation error: {e}")
            traceback.print_exc()
            return 0.0
    
    def _evaluate_final_architecture(self, architecture: NeuralArchitecture) -> Dict[str, float]:
        """Full evaluation of final episode architecture using actual training
        
        This is called at the end of each episode to show true progress.
        Uses QuickTrainer for actual training + evaluation on test set.
        """
        try:
            accuracy, loss = self.quick_trainer.train_and_evaluate(architecture)
            return {
                'final_accuracy': accuracy,
                'final_loss': loss,
                'method': 'quick_trainer'
            }
        except Exception as e:
            print(f"Final evaluation error: {e}")
            traceback.print_exc()
            # Fallback to policy-value estimate
            estimated_value = self._evaluate_architecture(architecture)
            return {
                'final_accuracy': estimated_value,
                'final_loss': 0.0,
                'method': 'fallback_policy_network'
            }

    def _prepare_graph_data(self, architecture: NeuralArchitecture) -> Dict:
        """Optimized: Prepare graph data for neural network using cached sorted IDs"""
        graph_data = architecture.to_graph_representation()
        
        # Add batch dimension and move to device
        graph_data['node_features'] = graph_data['node_features'].unsqueeze(0).to(self.device)
        graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
        graph_data['edge_weights'] = graph_data['edge_weights'].to(self.device)
        
        # Use cached sorted neuron IDs from graph representation instead of sorting again
        sorted_neuron_ids = graph_data['sorted_neuron_ids']
        layer_positions = [architecture.neurons[neuron_id].layer_position for neuron_id in sorted_neuron_ids]
        graph_data['layer_positions'] = torch.FloatTensor([layer_positions]).to(self.device)
        
        # For single graphs, num_graphs is always 1 (no 'batch' tensor needed)
        # The network.forward() will detect this and not attempt batching
        
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
    
    def _precompute_mcts_marginals(self, visit_distribution: torch.Tensor, 
                                   mcts_actions: List[Action]) -> Dict[str, List]:
        """Precompute per-component MCTS marginal distributions at experience creation time.
        
        This avoids repeated marginalization during training and ensures deterministic,
        cache-friendly computation. Stores as CPU lists for serialization and device safety.
        
        Returns Dict with keys:
        - 'mcts_policy_action_type': [num_action_types] prob distribution (CPU list)
        - 'mcts_policy_source': [max_neurons] prob distribution if source actions exist (CPU list)
        - 'mcts_policy_target': [max_neurons] prob distribution if target actions exist (CPU list)
        - 'mcts_policy_activation': [num_activations] prob distribution if activation actions exist (CPU list)
        """
        marginals = {}
        
        if visit_distribution is None or mcts_actions is None:
            return marginals
        
        try:
            # Cache expensive constants (avoid recomputation if called multiple times in episode)
            if not hasattr(self, '_mcts_marginal_constants'):
                self._mcts_marginal_constants = {
                    'num_action_types': len(ActionType),
                    'num_activations': len(ActivationType),
                    'activation_type_map': {act: idx for idx, act in enumerate(ActivationType)},
                    'max_neurons': self.config.model.max_neurons
                }
            
            const = self._mcts_marginal_constants
            
            # Convert visit_distribution to CPU if needed
            visit_dist_cpu = visit_distribution.cpu() if torch.is_tensor(visit_distribution) else visit_distribution
            
            # Initialize aggregators
            action_type_dist = torch.zeros(const['num_action_types'], dtype=torch.float32)
            source_dist = torch.zeros(const['max_neurons'], dtype=torch.float32)
            target_dist = torch.zeros(const['max_neurons'], dtype=torch.float32)
            activation_dist = torch.zeros(const['num_activations'], dtype=torch.float32)
            
            source_count = target_count = activation_count = 0
            
            # Single pass marginalization
            for prob, act in zip(visit_dist_cpu, mcts_actions):
                if act is None:
                    continue
                
                prob_val = float(prob)
                
                # Action type (always)
                action_type_dist[act.action_type.value] += prob_val
                
                # Source neuron
                if act.source_neuron is not None and 0 <= act.source_neuron < const['max_neurons']:
                    source_dist[act.source_neuron] += prob_val
                    source_count += 1
                
                # Target neuron
                if act.target_neuron is not None and 0 <= act.target_neuron < const['max_neurons']:
                    target_dist[act.target_neuron] += prob_val
                    target_count += 1
                
                # Activation
                if act.activation is not None:
                    act_idx = const['activation_type_map'].get(act.activation)
                    if act_idx is not None:
                        activation_dist[act_idx] += prob_val
                        activation_count += 1
            
            # Normalize and clamp with epsilon guard
            def _normalize_clamp(dist, count):
                if count == 0:
                    return None
                policy_sum = dist.sum()
                if policy_sum > 1e-8:
                    dist = dist / policy_sum
                dist = torch.clamp(dist, min=1e-8, max=1.0 - 1e-8)
                normalized = dist / dist.sum()
                # Convert to CPU list for serialization and storage safety
                return normalized.cpu().tolist()
            
            # Store normalized marginals as CPU lists
            normalized_action_dist = _normalize_clamp(action_type_dist, 1)  # Always has data
            if normalized_action_dist is not None:
                marginals['mcts_policy_action_type'] = normalized_action_dist
            
            if source_count > 0:
                normalized_source = _normalize_clamp(source_dist, source_count)
                if normalized_source is not None:
                    marginals['mcts_policy_source'] = normalized_source
            
            if target_count > 0:
                normalized_target = _normalize_clamp(target_dist, target_count)
                if normalized_target is not None:
                    marginals['mcts_policy_target'] = normalized_target
            
            if activation_count > 0:
                normalized_activation = _normalize_clamp(activation_dist, activation_count)
                if normalized_activation is not None:
                    marginals['mcts_policy_activation'] = normalized_activation
        
        except Exception as e:
            print(f"    [Warning] Failed to precompute MCTS marginals: {e}")
        
        return marginals
    
    def _create_experience(self, state: NeuralArchitecture, action: Action, 
                          reward: float, next_state: NeuralArchitecture,
                          search_root=None) -> Dict:
        """Create experience tuple with MCTS visit distribution for AlphaZero-style training
        
        AlphaZero Key Insight: Store MCTS-improved policy π(s) = visit_count(a) / total_visits
        The network learns to predict this π, not raw network priors. This leverages MCTS's 
        superior exploration and planning.
        
        Optimization: Precompute and store per-component marginal distributions as CPU lists
        to avoid repeated marginalization during training.
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,  # Immediate reward (will be replaced with final episode reward)
            'next_state': next_state,
            'timestamp': time.time(),
            'mcts_policy': None,  # Will be filled below
            'mcts_actions': None,   # For debugging
            # Per-component precomputed marginals (stored as CPU lists for safety/serialization)
            'mcts_policy_action_type': None,
            'mcts_policy_source': None,
            'mcts_policy_target': None,
            'mcts_policy_activation': None
        }
        
        # Extract MCTS visit distribution (CRITICAL for AlphaZero learning!)
        if search_root is not None and hasattr(search_root, 'children'):
            try:
                # Get visit distribution from neural_mcts helper
                visit_distribution = self.neural_mcts.get_visit_distribution(
                    search_root, temperature=1.0
                )
                
                if visit_distribution is not None and len(visit_distribution) > 0:
                    # Store raw distribution as CPU tensor for reference
                    experience['mcts_policy'] = visit_distribution.cpu() if torch.is_tensor(visit_distribution) else visit_distribution
                    experience['mcts_actions'] = [child.action for child in search_root.children]
                    
                    # OPTIMIZATION: Precompute per-component marginals at experience creation
                    # Returns Dict with keys 'mcts_policy_action_type', 'mcts_policy_source', etc. (CPU lists)
                    marginals = self._precompute_mcts_marginals(visit_distribution, experience['mcts_actions'])
                    experience.update(marginals)  # Merge marginals into experience
                    
                    if len(visit_distribution) <= 10:
                        print(f"    MCTS Policy stored: π(s) from {len(search_root.children)} children")
            except Exception as e:
                print(f"    [Warning] Failed to extract MCTS policy: {e}")
                # Continue without policy - will train only on value
        
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
        """Compute priority for experience replay using outcome + surprise (TD-error proxy).
        
        Priority reflects two signals:
        1. OUTCOME: experiences from high-value episodes are more important
        2. SURPRISE: |value_target - immediate_reward| = prediction error, indicates informativeness
        3. NOVELTY: bonus for larger/more complex architectures
        
        This ensures replay buffer learns from both good outcomes and surprising transitions.
        """
        # Get immediate reward (what PV net predicted during episode)
        immediate_reward = experience['reward']
        
        # Get final ground truth (QuickTrainer result or fallback)
        final_accuracy = episode_metrics['final_accuracy']
        
        # Compute surprise/TD-error: how wrong was the immediate prediction?
        # Clamp to avoid extreme priority from noise
        surprise = abs(final_accuracy - immediate_reward)
        surprise = min(surprise, 1.0)  # Cap at 1.0
        
        # Primary priority: outcome quality (good final results get higher priority)
        outcome_priority = final_accuracy
        
        # Secondary priority: surprise/informativeness (prediction errors are valuable)
        surprise_weight = getattr(self.config.search, 'priority_surprise_weight', 0.5)
        
        priority = outcome_priority + surprise_weight * surprise
        
        # Bonus for architectures of different sizes (encourage diversity)
        neuron_count = len(experience['state'].neurons)
        connection_count = len(experience['state'].connections)
        novelty_bonus = (neuron_count + connection_count) / 1000
        
        return priority + novelty_bonus
    
    def _get_legal_action_mask(self, state: NeuralArchitecture) -> Dict[str, torch.Tensor]:
        """Create mask for legal actions in this state (FIX #3: Action Legality Masking)
        
        Prevents training on impossible action combinations. AlphaZero reference:
        "We actually have to correct for this manually by masking out illegal moves, 
        and then re-normalizing the remaining scores"
        
        Returns:
            Dict with masks for action_type, source_neuron, target_neuron, activation
            (1.0 = legal, 0.0 = illegal)
        """
        legal_actions = self.action_space.get_valid_actions(state)
        
        # Initialize all-zero masks (everything starts as illegal)
        action_type_mask = torch.zeros(len(ActionType), dtype=torch.float32)
        source_mask = torch.zeros(self.config.model.max_neurons, dtype=torch.float32)
        target_mask = torch.zeros(self.config.model.max_neurons, dtype=torch.float32)
        activation_mask = torch.zeros(len(ActivationType), dtype=torch.float32)
        
        # Mark legal actions as 1.0
        for action in legal_actions:
            action_type_mask[action.action_type.value] = 1.0
            if action.source_neuron is not None:
                source_mask[action.source_neuron] = 1.0
            if action.target_neuron is not None:
                target_mask[action.target_neuron] = 1.0
            if action.activation is not None:
                try:
                    activation_idx = list(ActivationType).index(action.activation)
                    activation_mask[activation_idx] = 1.0
                except (ValueError, IndexError):
                    pass
        
        return {
            'action_type': action_type_mask.to(self.device),
            'source_neuron': source_mask.to(self.device),
            'target_neuron': target_mask.to(self.device),
            'activation': activation_mask.to(self.device),
        }

    def train_on_batch(self, batch_size: int = 32, sub_batch_size: int = 4):
        """Train with proper gradient accumulation across sub-batches.
        
        Optimizations:
        - Pre-compute annealed CE weight once (avoid per-graph calculations)
        - Pass pre-computed weights to _process_sub_batch to avoid redundant lookups
        - Cache config values (avoid repeated attribute lookups)
        - Reduce print overhead (batch-level only)
        
        MEMORY EFFICIENT FLOW:
        1. Sample batch_size experiences from replay buffer
        2. Split into sub-batches of size sub_batch_size
        3. For each sub-batch:
           - Process graphs through network
           - Compute loss for this sub-batch
           - Backward (accumulate gradients in network parameters)
           - Clear intermediate tensors from GPU
        4. After all sub-batches: gradient clipping and optimizer.step()
        5. Update priorities for all experiences
        
        Key benefit: Never keeps all batch_size graphs in memory simultaneously.
        Memory peak is O(sub_batch_size * avg_nodes) instead of O(batch_size * avg_nodes).
        
        Args:
            batch_size: Total number of experiences to sample
            sub_batch_size: Number of graphs per training step (default 4)
        
        Returns:
            Dictionary with accumulated losses
        """
        # Sample full batch from replay buffer
        experiences, indices, weights = self.experience_buffer.sample(batch_size)
        
        if not experiences:
            return None
        
        # Cache config values (avoid repeated attribute lookups per graph)
        ce_anneal_episodes = getattr(self.config.mcts, 'ce_anneal_episodes', 500)
        initial_ce_weight = getattr(self.config.mcts, 'component_ce_weight', 0.25)
        mcts_policy_weight = getattr(self.config.mcts, 'mcts_policy_weight', 1.0)
        final_ce_weight = 0.05
        
        # Pre-compute annealed CE weight once (all graphs use same schedule)
        if self.episode < ce_anneal_episodes:
            progress = self.episode / max(1, ce_anneal_episodes)
            annealed_ce_weight = initial_ce_weight * (1.0 - progress) + final_ce_weight * progress
        else:
            annealed_ce_weight = final_ce_weight
        
        # Initialize gradient accumulation across all sub-batches
        self.optimizer.zero_grad()
        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        mcts_policy_loss_accum = 0.0
        weighted_policy_loss_accum = 0.0
        weighted_mcts_policy_loss_accum = 0.0
        num_processed = 0
        valid_indices_list = []
        valid_rewards_list = []
        
        # Process batch in smaller sub-batches for true gradient accumulation
        
        for sub_batch_idx in range(0, len(experiences), sub_batch_size):
            sub_batch_end = min(sub_batch_idx + sub_batch_size, len(experiences))
            sub_experiences = experiences[sub_batch_idx:sub_batch_end]
            sub_weights = weights[sub_batch_idx:sub_batch_end]
            sub_indices = indices[sub_batch_idx:sub_batch_end]
            
            # Process this sub-batch with pre-computed weights (avoid per-graph recalculation)
            sub_result = self._process_sub_batch(sub_experiences, sub_weights, sub_indices, 
                                                 annealed_ce_weight, mcts_policy_weight)
            
            if sub_result is not None:
                # Backward on this sub-batch (accumulates gradients)
                sub_result['loss'].backward()
                
                # Accumulate metrics (raw and weighted)
                total_loss_accum += sub_result['total_loss']
                policy_loss_accum += sub_result['policy_loss']
                value_loss_accum += sub_result['value_loss']
                mcts_policy_loss_accum += sub_result['mcts_policy_loss']
                weighted_policy_loss_accum += sub_result['weighted_policy_loss']
                weighted_mcts_policy_loss_accum += sub_result['weighted_mcts_policy_loss']
                num_processed += sub_result['num_graphs']
                
                # Track for priority updates
                valid_indices_list.extend(sub_result['valid_indices'])
                valid_rewards_list.extend(sub_result['valid_rewards'])
            
            # Clear cache after each sub-batch to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        if num_processed == 0:
            print(f"  [Training] No legal actions in batch, skipping")
            return None
        
        # Single optimizer step after all sub-batches (where accumulated gradients apply)
        torch.nn.utils.clip_grad_norm_(
            self.policy_value_net.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Update priorities for all valid experiences
        if valid_indices_list:
            self.experience_buffer.update_priorities(valid_indices_list, valid_rewards_list)
        
        # Return averaged metrics
        metrics = {
            'total_loss': total_loss_accum / num_processed if num_processed > 0 else 0.0,
            'policy_loss': policy_loss_accum / num_processed if num_processed > 0 else 0.0,
            'value_loss': value_loss_accum / num_processed if num_processed > 0 else 0.0,
            'mcts_policy_loss': mcts_policy_loss_accum / num_processed if num_processed > 0 else 0.0,
            'weighted_policy_loss': weighted_policy_loss_accum / num_processed if num_processed > 0 else 0.0,
            'weighted_mcts_policy_loss': weighted_mcts_policy_loss_accum / num_processed if num_processed > 0 else 0.0,
            'num_graphs': num_processed
        }
        
        print(f"  [Training] Batch complete ({num_processed} graphs):")
        print(f"    total_loss={metrics['total_loss']:.6f}")
        print(f"    policy_loss (raw)={metrics['policy_loss']:.6f}, weighted={metrics['weighted_policy_loss']:.6f}")
        print(f"    mcts_policy_loss (raw)={metrics['mcts_policy_loss']:.6f}, weighted={metrics['weighted_mcts_policy_loss']:.6f}")
        print(f"    value_loss={metrics['value_loss']:.6f}")
        
        return metrics
    
    def _process_sub_batch(self, experiences, weights, indices, annealed_ce_weight, mcts_policy_weight):
        """Process a single sub-batch and compute loss for gradient accumulation.
        
        Optimizations:
        - Accept pre-computed annealed_ce_weight and mcts_policy_weight to avoid per-graph recalculation
        - Cache math.log values for loss normalization
        
        Args:
            experiences: List of experience dicts
            weights: List of sample weights for prioritized replay
            indices: List of buffer indices for priority updates
            annealed_ce_weight: Pre-computed CE weight for this batch (from train_on_batch)
            mcts_policy_weight: Pre-computed MCTS policy weight from config
        
        Returns:
            Dict with:
                - loss: scalar tensor (requires grad for backward)
                - total_loss: float value
                - policy_loss: float value
                - value_loss: float value
                - mcts_policy_loss: float value
                - weighted_policy_loss: float value
                - weighted_mcts_policy_loss: float value
                - num_graphs: int (number of valid graphs)
                - valid_indices: list of buffer indices for priority update
                - valid_rewards: list of rewards for priority update
        """
        import math
        
        # Cache normalization constants (used in multiple CE loss calculations)
        log_max_neurons = math.log(max(1.0, self.config.model.max_neurons))
        log_num_activations = math.log(max(1.0, len(ActivationType)))
        
        # Prepare graph data for this sub-batch
        graph_data_list = []
        targets_list = []
        weights_list = []
        action_masks_list = []
        valid_indices_list = []
        valid_experiences = []
        
        for exp_idx, (exp, weight, idx) in enumerate(zip(experiences, weights, indices)):
            # Prepare graph data
            graph_data = self._prepare_graph_data(exp['state'])
            
            # Check action legality
            action_mask = self._get_legal_action_mask(exp['state'])
            action = exp['action']
            
            # Validate action components
            is_action_type_legal = action_mask['action_type'][action.action_type.value] > 0
            if not is_action_type_legal:
                continue
            
            is_source_legal = (action.source_neuron is None or 
                             (action.source_neuron < len(action_mask['source_neuron']) and 
                              action_mask['source_neuron'][action.source_neuron] > 0))
            is_target_legal = (action.target_neuron is None or 
                             (action.target_neuron < len(action_mask['target_neuron']) and 
                              action_mask['target_neuron'][action.target_neuron] > 0))
            is_activation_legal = (action.activation is None or action_mask['activation'].max() > 0)
            
            if not (is_source_legal and is_target_legal and is_activation_legal):
                continue
            
            # Create targets
            targets = self._create_targets(exp)
            
            # Store valid experience
            valid_experiences.append(exp)
            graph_data_list.append(graph_data)
            targets_list.append(targets)
            weights_list.append(weight.item() if isinstance(weight, torch.Tensor) else weight)
            action_masks_list.append(action_mask)
            valid_indices_list.append(idx)
        
        if not graph_data_list:
            return None
        
        # Batch the graph data (only for this sub-batch)
        batched_graph_data = self._batch_graph_data(graph_data_list)
        num_graphs = len(graph_data_list)
        
        # Forward pass on sub-batch through network
        # No internal sub-batching needed since sub-batch is already small
        predictions = self.policy_value_net(batched_graph_data, sub_batch_size=num_graphs)
        
        # Compute losses for this sub-batch
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        mcts_policy_loss_sum = 0.0
        weighted_policy_loss_sum = 0.0
        weighted_mcts_policy_loss_sum = 0.0
        
        try:
            action_type_logits = predictions['action_type']  # [num_graphs, 5]
            source_logits = predictions['source_logits']      # [num_graphs, max_neurons]
            target_logits = predictions['target_logits']      # [num_graphs, max_neurons]
            activation_logits = predictions['activation_logits']  # [num_graphs, num_activations]
            values = predictions['value']                      # [num_graphs, 1]
            
            graph_losses = []
            
            for graph_idx in range(num_graphs):
                action_mask = action_masks_list[graph_idx]
                targets = targets_list[graph_idx]
                weight = weights_list[graph_idx]
                
                # Extract this graph's predictions
                action_logits = action_type_logits[graph_idx:graph_idx+1]  # [1, 5]
                source_pred = source_logits[graph_idx:graph_idx+1]  # [1, max_neurons]
                target_pred = target_logits[graph_idx:graph_idx+1]  # [1, max_neurons]
                activation_pred = activation_logits[graph_idx:graph_idx+1]  # [1, num_activations]
                value_pred = values[graph_idx:graph_idx+1]  # [1, 1]
                
                # Apply action legality masking
                masked_action_logits = action_logits.clone()
                action_mask_bool = (action_mask['action_type'].to(self.device) == 0).unsqueeze(0)
                masked_action_logits = masked_action_logits.masked_fill(action_mask_bool, -1e9)
                
                masked_source_logits = source_pred.clone()
                source_mask_bool = (action_mask['source_neuron'].to(self.device) == 0).unsqueeze(0)
                masked_source_logits = masked_source_logits.masked_fill(source_mask_bool, -1e9)
                
                masked_target_logits = target_pred.clone()
                target_mask_bool = (action_mask['target_neuron'].to(self.device) == 0).unsqueeze(0)
                masked_target_logits = masked_target_logits.masked_fill(target_mask_bool, -1e9)
                
                masked_activation_logits = activation_pred.clone()
                activation_mask_bool = (action_mask['activation'].to(self.device) == 0).unsqueeze(0)
                masked_activation_logits = masked_activation_logits.masked_fill(activation_mask_bool, -1e9)
                
                # Compute policy losses
                action_type_target = torch.tensor([targets['action_type']], dtype=torch.long, device=self.device)
                action_type_loss = F.cross_entropy(masked_action_logits, action_type_target)
                
                source_loss = torch.tensor(0.0, device=self.device)
                if 'source_neuron' in targets:
                    source_target = torch.tensor([targets['source_neuron']], dtype=torch.long, device=self.device)
                    source_loss = F.cross_entropy(masked_source_logits, source_target) / log_max_neurons
                
                target_loss = torch.tensor(0.0, device=self.device)
                if 'target_neuron' in targets:
                    target_target = torch.tensor([targets['target_neuron']], dtype=torch.long, device=self.device)
                    target_loss = F.cross_entropy(masked_target_logits, target_target) / log_max_neurons
                
                activation_loss = torch.tensor(0.0, device=self.device)
                if 'activation' in targets:
                    activation_target = torch.tensor([targets['activation']], dtype=torch.long, device=self.device)
                    activation_loss = F.cross_entropy(masked_activation_logits, activation_target) / log_num_activations
                
                loss_count = sum([1, 
                                 1 if 'source_neuron' in targets else 0,
                                 1 if 'target_neuron' in targets else 0,
                                 1 if 'activation' in targets else 0])
                policy_loss = (action_type_loss + source_loss + target_loss + activation_loss) / max(1, loss_count)
                
                # Compute value loss
                value_target = targets['value'].squeeze() if targets['value'].dim() > 1 else targets['value']
                value_pred_squeezed = value_pred.squeeze()
                if value_pred_squeezed.dim() == 0:
                    value_pred_squeezed = value_pred_squeezed.unsqueeze(0)
                if value_target.dim() == 0:
                    value_target = value_target.unsqueeze(0)
                value_loss = F.mse_loss(value_pred_squeezed, value_target)
                
                # Compute MCTS policy losses (KL divergence to per-component visit distributions)
                mcts_loss = torch.tensor(0.0, device=self.device)
                mcts_loss_item = 0.0
                num_mcts_targets = 0
                
                # 1. MCTS KL for action_type head
                if 'mcts_policy_action_type' in targets and targets['mcts_policy_action_type'] is not None:
                    try:
                        mcts_action_type_target = targets['mcts_policy_action_type'].unsqueeze(0).to(self.device)
                        kl_action_type = F.kl_div(
                            F.log_softmax(masked_action_logits, dim=1),
                            mcts_action_type_target,
                            reduction='mean'
                        )
                        mcts_loss = mcts_loss + kl_action_type
                        mcts_loss_item += kl_action_type.item()
                        num_mcts_targets += 1
                    except Exception as e:
                        print(f"    [ERROR] Graph {graph_idx}: MCTS action_type KL exception: {e}")
                
                # 2. MCTS KL for source_neuron head
                if 'mcts_policy_source' in targets and targets['mcts_policy_source'] is not None:
                    try:
                        mcts_source_target = targets['mcts_policy_source'].unsqueeze(0).to(self.device)
                        kl_source = F.kl_div(
                            F.log_softmax(masked_source_logits, dim=1),
                            mcts_source_target,
                            reduction='mean'
                        )
                        mcts_loss = mcts_loss + kl_source
                        mcts_loss_item += kl_source.item()
                        num_mcts_targets += 1
                    except Exception as e:
                        print(f"    [ERROR] Graph {graph_idx}: MCTS source KL exception: {e}")
                
                # 3. MCTS KL for target_neuron head
                if 'mcts_policy_target' in targets and targets['mcts_policy_target'] is not None:
                    try:
                        mcts_target_target = targets['mcts_policy_target'].unsqueeze(0).to(self.device)
                        kl_target = F.kl_div(
                            F.log_softmax(masked_target_logits, dim=1),
                            mcts_target_target,
                            reduction='mean'
                        )
                        mcts_loss = mcts_loss + kl_target
                        mcts_loss_item += kl_target.item()
                        num_mcts_targets += 1
                    except Exception as e:
                        print(f"    [ERROR] Graph {graph_idx}: MCTS target KL exception: {e}")
                
                # 4. MCTS KL for activation head
                if 'mcts_policy_activation' in targets and targets['mcts_policy_activation'] is not None:
                    try:
                        mcts_activation_target = targets['mcts_policy_activation'].unsqueeze(0).to(self.device)
                        kl_activation = F.kl_div(
                            F.log_softmax(masked_activation_logits, dim=1),
                            mcts_activation_target,
                            reduction='mean'
                        )
                        mcts_loss = mcts_loss + kl_activation
                        mcts_loss_item += kl_activation.item()
                        num_mcts_targets += 1
                    except Exception as e:
                        print(f"    [ERROR] Graph {graph_idx}: MCTS activation KL exception: {e}")
                
                # Average MCTS loss across components that have targets
                if num_mcts_targets > 0:
                    mcts_loss = mcts_loss / num_mcts_targets
                else:
                    mcts_loss = torch.tensor(0.0, device=self.device)
                
                # Safeguard: detect pathological losses
                mcts_loss_item = mcts_loss.item() if num_mcts_targets > 0 else 0.0
                if mcts_loss_item > 100.0:
                    # Zero out both the tensor loss used for gradients and the logged scalar
                    mcts_loss = torch.tensor(0.0, device=self.device)
                    mcts_loss_item = 0.0
                
                # Combine MCTS loss and CE loss with pre-computed weights
                weighted_mcts_loss = mcts_policy_weight * mcts_loss
                weighted_ce_loss = annealed_ce_weight * policy_loss
                
                # Combine losses for this graph
                graph_loss = (weighted_mcts_loss + weighted_ce_loss + value_loss) * weight
                graph_losses.append(graph_loss)
                
                # Track both raw and weighted losses for monitoring
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                mcts_policy_loss_sum += mcts_loss_item
                weighted_policy_loss_sum += weighted_ce_loss.item()
                weighted_mcts_policy_loss_sum += weighted_mcts_loss.item()
            
            # Sum all graph losses for this sub-batch
            if graph_losses:
                total_loss = sum(graph_losses)
        
        except Exception as e:
            print(f"    [ERROR] Loss computation failed in sub-batch: {e}")
            print(f"    Predictions shapes: {[(k, v.shape) for k, v in predictions.items()]}")
            raise
        
        # Priorities for replay buffer update: outcome + surprise (TD-error proxy)
        # Each experience used for training should have a priority reflecting:
        # 1. The final outcome (value_target) it led to
        # 2. The surprise/error: |value_target - immediate_reward|
        valid_rewards = []
        surprise_weight = getattr(self.config.search, 'priority_surprise_weight', 0.5)
        for exp in valid_experiences:
            # Get final target (ground truth for this experience)
            value_target = exp.get('value_target', exp['reward'])
            immediate_reward = exp['reward']
            
            # Surprise: how wrong was the network's prediction?
            surprise = abs(value_target - immediate_reward)
            surprise = min(surprise, 1.0)
            
            # Priority combines outcome quality + informativeness
            priority = value_target + surprise_weight * surprise
            valid_rewards.append(max(0.0, priority))
        
        return {
            'loss': total_loss,
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss_sum,                      # Raw (unweighted) supervised CE
            'value_loss': value_loss_sum,
            'mcts_policy_loss': mcts_policy_loss_sum,           # Raw (unweighted) MCTS KL
            'weighted_policy_loss': weighted_policy_loss_sum,   # After annealing weight
            'weighted_mcts_policy_loss': weighted_mcts_policy_loss_sum,  # After mcts_policy_weight
            'num_graphs': num_graphs,
            'valid_indices': valid_indices_list,
            'valid_rewards': valid_rewards
        }
    
    def _create_targets(self, experience: Dict) -> Dict:
        """Create training targets from experience (AlphaZero-style, complete factorized policy)
        
        Optimizations:
        - Cache ActionType enum and ActivationType mapping (avoid O(n) lookups)
        - Single pass over MCTS actions for all distributions (or use precomputed marginals)
        - Pre-allocate tensors on device to avoid redundant transfers
        - Use precomputed per-component marginals from experience if available (fast path)
        - Fallback to on-the-fly marginalization if precomputed not present (backward compat)
        
        AlphaZero trains on:
        1. Value target: final episode outcome z (credit assignment)
        2. Policy target: MCTS visit distribution π(s) aggregated by action type
        3. Action components (only those relevant to the action type)
        """
        
        # Cache expensive imports/lookups
        if not hasattr(self, '_action_type_set'):
            self._action_type_set = set([ActionType.ADD_CONNECTION, ActionType.REMOVE_CONNECTION, 
                                         ActionType.MODIFY_ACTIVATION, ActionType.REMOVE_NEURON])
            self._target_action_type_set = set([ActionType.ADD_CONNECTION, ActionType.REMOVE_CONNECTION])
            self._activation_action_type_set = set([ActionType.MODIFY_ACTIVATION, ActionType.ADD_NEURON])
            self._activation_type_list = list(ActivationType)
            self._activation_type_map = {act: idx for idx, act in enumerate(self._activation_type_list)}
            self._num_action_types = len(ActionType)
            self._num_activations = len(ActivationType)
        
        action = experience['action']
        value_target = experience.get('value_target', experience['reward'])
        
        # ===== VALUE TARGET =====
        targets = {
            'action_type': action.action_type.value,
            'value': torch.tensor([value_target], dtype=torch.float32, device=self.device)
        }

        # ===== COMPLETE FACTORIZED POLICY TARGETS (RELEVANT ONLY) =====
        if action.action_type in self._action_type_set and action.source_neuron is not None:
            targets['source_neuron'] = action.source_neuron
        
        if action.action_type in self._target_action_type_set and action.target_neuron is not None:
            targets['target_neuron'] = action.target_neuron
        
        if action.action_type in self._activation_action_type_set and action.activation is not None:
            targets['activation'] = self._activation_type_map.get(action.activation, -1)
            if targets['activation'] == -1:
                del targets['activation']

        # ===== MCTS POLICY TARGETS (ALL POLICY HEADS) =====
        # FAST PATH: Use precomputed marginals from experience if available (typical case)
        precomputed_marginals = {
            'mcts_policy_action_type': experience.get('mcts_policy_action_type'),
            'mcts_policy_source': experience.get('mcts_policy_source'),
            'mcts_policy_target': experience.get('mcts_policy_target'),
            'mcts_policy_activation': experience.get('mcts_policy_activation')
        }
        
        # Check if we have any precomputed marginals (new storage format)
        has_precomputed = any(m is not None for m in precomputed_marginals.values())
        
        if has_precomputed:
            # Use precomputed marginals (stored as CPU lists, convert to GPU tensors)
            if precomputed_marginals['mcts_policy_action_type'] is not None:
                targets['mcts_policy_action_type'] = torch.tensor(
                    precomputed_marginals['mcts_policy_action_type'], 
                    dtype=torch.float32, device=self.device
                )
            
            if precomputed_marginals['mcts_policy_source'] is not None:
                targets['mcts_policy_source'] = torch.tensor(
                    precomputed_marginals['mcts_policy_source'],
                    dtype=torch.float32, device=self.device
                )
            
            if precomputed_marginals['mcts_policy_target'] is not None:
                targets['mcts_policy_target'] = torch.tensor(
                    precomputed_marginals['mcts_policy_target'],
                    dtype=torch.float32, device=self.device
                )
            
            if precomputed_marginals['mcts_policy_activation'] is not None:
                targets['mcts_policy_activation'] = torch.tensor(
                    precomputed_marginals['mcts_policy_activation'],
                    dtype=torch.float32, device=self.device
                )
        
        elif 'mcts_policy' in experience and 'mcts_actions' in experience:
            # FALLBACK: On-the-fly marginalization (backward compat for old experiences)
            mcts_policy = experience['mcts_policy']
            mcts_actions = experience['mcts_actions']
            # Fix: avoid ambiguous boolean value for tensors/lists
            mcts_policy_len = mcts_policy.numel() if torch.is_tensor(mcts_policy) else len(mcts_policy)
            mcts_actions_len = mcts_actions.numel() if torch.is_tensor(mcts_actions) else len(mcts_actions)
            if mcts_policy is not None and mcts_actions is not None and mcts_policy_len > 0 and mcts_actions_len > 0:
                try:
                    max_neurons = self.config.model.max_neurons
                    
                    # Pre-allocate all distributions on device (avoid .to(device) transfers)
                    action_type_dist = torch.zeros(self._num_action_types, dtype=torch.float32, device=self.device)
                    source_dist = torch.zeros(max_neurons, dtype=torch.float32, device=self.device)
                    target_dist = torch.zeros(max_neurons, dtype=torch.float32, device=self.device)
                    activation_dist = torch.zeros(self._num_activations, dtype=torch.float32, device=self.device)
                    
                    source_count = target_count = activation_count = 0
                    
                    # Single pass over all MCTS actions (4x faster than 4 separate passes)
                    for prob, act in zip(mcts_policy, mcts_actions):
                        if act is None:
                            continue
                        
                        prob_val = float(prob)
                        
                        # Action type always included
                        action_type_dist[act.action_type.value] += prob_val
                        
                        # Source neuron (for relevant action types)
                        if act.source_neuron is not None and act.source_neuron < max_neurons:
                            source_dist[act.source_neuron] += prob_val
                            source_count += 1
                        
                        # Target neuron (for relevant action types)
                        if act.target_neuron is not None and act.target_neuron < max_neurons:
                            target_dist[act.target_neuron] += prob_val
                            target_count += 1
                        
                        # Activation (for relevant action types)
                        if act.activation is not None:
                            act_idx = self._activation_type_map.get(act.activation)
                            if act_idx is not None:
                                activation_dist[act_idx] += prob_val
                                activation_count += 1
                    
                    # Normalize and clamp all distributions (vectorized)
                    def _normalize_clamp(dist, count):
                        if count == 0:
                            return None
                        policy_sum = dist.sum()
                        if policy_sum > 1e-8:
                            dist = dist / policy_sum
                        dist = torch.clamp(dist, min=1e-8, max=1.0 - 1e-8)
                        return dist / dist.sum()
                    
                    # Add distributions to targets if count > 0
                    normalized_action_dist = _normalize_clamp(action_type_dist, 1)  # Always has data
                    if normalized_action_dist is not None:
                        targets['mcts_policy_action_type'] = normalized_action_dist
                    
                    if source_count > 0:
                        targets['mcts_policy_source'] = _normalize_clamp(source_dist, source_count)
                    
                    if target_count > 0:
                        targets['mcts_policy_target'] = _normalize_clamp(target_dist, target_count)
                    
                    if activation_count > 0:
                        targets['mcts_policy_activation'] = _normalize_clamp(activation_dist, activation_count)
                    
                except Exception as e:
                    print(f"    [Warning] Failed to create MCTS policy targets: {e}")

        return targets
    
    def run_training(self):
        """Run complete training process using AlphaZero-style MCTS + policy network"""
        print("Starting architecture search training...")
        
        while self.episode < self.config.max_episodes:
            # Run training episode
            episode_metrics = self.run_training_episode()
            
            # Train on batch if we have enough experiences
            train_interval = getattr(self.config, 'train_interval', 10)
            if self.episode > 0 and self.episode % train_interval == 0:
                # Train on all accumulated experiences in the buffer
                buffer_size = len(self.experience_buffer)
                print(f"  [Training] Starting with {buffer_size} total experiences in buffer")
                remaining = buffer_size
                while remaining > 0:
                    batch_size = min(self.config.batch_size, remaining)
                    train_metrics = self.train_on_batch(batch_size)
                    if train_metrics:
                        episode_metrics.update(train_metrics)
                    remaining -= batch_size
                # Clear buffer after training on all experiences
                self.experience_buffer.clear()

            # Checkpoint periodically
            if self.episode > 0 and self.episode % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Update episode counter
            self.episode += 1

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
        # Always log (removed interval check so every episode is logged)
        accuracy = metrics['final_accuracy']
        neurons = metrics['total_neurons']
        connections = metrics['total_connections']
        
        # Console output
        print(f"Episode {self.episode}: "
              f"Accuracy={accuracy:.4f}, Neurons={neurons}, "
              f"Connections={connections}, Best={self.best_accuracy:.4f}")

        # Structured file logging
        try:
            # Log to file handler
            log_message = (f"Episode {self.episode} Accuracy={accuracy:.4f} "
                          f"Neurons={neurons} Connections={connections} Best={self.best_accuracy:.4f}")
            self.logger.info(log_message)
            
            # Force flush handlers to ensure writes
            for handler in self.logger.handlers:
                handler.flush()
            
            # Append JSONL metrics
            with open(self.metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(json.dumps(metrics) + "\n")
                mf.flush()  # Ensure write
                
        except Exception as e:
            # Fall back to direct file write if logger fails
            print(f"Logger failed: {e}, attempting direct file write...")
            try:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{time.time()} INFO Episode {self.episode} Accuracy={accuracy:.4f} "
                           f"Neurons={neurons} Connections={connections} Best={self.best_accuracy:.4f}\n")
                    f.flush()
                with open(self.metrics_path, 'a', encoding='utf-8') as mf:
                    mf.write(json.dumps(metrics) + "\n")
                    mf.flush()
            except Exception as e2:
                print(f"Direct file write also failed: {e2}")
                traceback.print_exc()
    
    def _save_checkpoint(self):
        """Save training checkpoint (AlphaZero-style, no curriculum)
        
        Note: MCTS tree state (mcts_root) is NOT saved because:
        - Tree is episode-scoped (resets between episodes)
        - Each episode should start with fresh tree for exploration diversity
        - Tree reuse is only beneficial within a single episode
        """
        checkpoint = {
            'episode': self.episode,
            'policy_value_net_state': self.policy_value_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history,
            'config': self.config,
            'experience_buffer': self.experience_buffer.state_dict()
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
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
        self.episode = checkpoint['episode'] + 1  # Start next episode
        self.experience_buffer.load_state_dict(checkpoint['experience_buffer'])

        print(f"Checkpoint loaded from episode {checkpoint['episode']}, resuming from episode {self.episode}")

    def _draw_architecture_diagram(self, architecture: NeuralArchitecture, step: int):
        """Draw a diagram using PIL (much faster than matplotlib, 10-100x speedup)
        
        Optimizations:
        - Use PIL instead of matplotlib (PIL is 10-100x faster)
        - Layer-based hierarchical layout (O(n) instead of O(n²) spring layout)
        - Direct pixel manipulation instead of path rendering
        - Minimal overhead: no antialiasing, lightweight drawing
        """
        try:
            # Create directory
            os.makedirs(f'architecture_diagrams/ep{self.episode:03d}', exist_ok=True)

            # Constants for layout
            CANVAS_WIDTH = 1200
            CANVAS_HEIGHT = 900
            MARGIN = 50
            INPUT_OUTPUT_RADIUS = 3
            HIDDEN_RADIUS = 8  # Make hidden nodes bigger for visibility
            
            # Create image
            img = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), color='white')
            draw = ImageDraw.Draw(img)
            
            # Separate neurons by type (original layout scheme)
            input_neurons = []
            hidden_neurons = []
            output_neurons = []
            
            for neuron_id, neuron in architecture.neurons.items():
                if neuron.neuron_type.value == 'input':
                    input_neurons.append(neuron_id)
                elif neuron.neuron_type.value == 'hidden':
                    hidden_neurons.append(neuron_id)
                elif neuron.neuron_type.value == 'output':
                    output_neurons.append(neuron_id)
            
            # Compute pixel positions
            pos = {}  # neuron_id -> (x, y)
            
            # Position input neurons on the LEFT (x=MARGIN)
            input_neurons.sort()
            if input_neurons:
                y_step = (CANVAS_HEIGHT - 2 * MARGIN) / max(len(input_neurons) - 1, 1)
                for i, neuron_id in enumerate(input_neurons):
                    x = MARGIN
                    y = MARGIN + int(i * y_step)
                    pos[neuron_id] = (x, y)
            
            # Position output neurons on the RIGHT (x=CANVAS_WIDTH-MARGIN)
            output_neurons.sort()
            if output_neurons:
                y_step = (CANVAS_HEIGHT - 2 * MARGIN) / max(len(output_neurons) - 1, 1)
                for i, neuron_id in enumerate(output_neurons):
                    x = CANVAS_WIDTH - MARGIN
                    y = MARGIN + int(i * y_step)
                    pos[neuron_id] = (x, y)
            
            # Position hidden neurons in 2D space between inputs and outputs
            # Use layer_position for x-coordinate and neuron_id-based random for y-coordinate
            for neuron_id in hidden_neurons:
                neuron = architecture.neurons[neuron_id]
                # Map layer_position (0.0-1.0) to x-coordinate between input and output
                layer_pos = neuron.layer_position
                x = MARGIN + int(layer_pos * (CANVAS_WIDTH - 2 * MARGIN))
                
                # Use neuron_id as seed for reproducible y position
                rng = np.random.RandomState(neuron_id)
                y = MARGIN + int(rng.uniform(0, 1) * (CANVAS_HEIGHT - 2 * MARGIN))
                
                pos[neuron_id] = (x, y)
            
            # Draw edges first (so they appear behind nodes)
            for conn in architecture.connections:
                if conn.enabled and conn.source_id in pos and conn.target_id in pos:
                    x1, y1 = pos[conn.source_id]
                    x2, y2 = pos[conn.target_id]
                    edge_color = (200, 100, 100) if conn.weight < 0 else (100, 100, 200)
                    draw.line([(x1, y1), (x2, y2)], fill=edge_color, width=1)
            
            # Draw nodes colored by type
            color_map = {
                'input': (52, 152, 219),      # Blue
                'hidden': (46, 204, 113),     # Green
                'output': (231, 76, 60)       # Red
            }
            
            for neuron_id, neuron in architecture.neurons.items():
                if neuron_id in pos:
                    x, y = pos[neuron_id]
                    color = color_map.get(neuron.neuron_type.value, (128, 128, 128))
                    # Use bigger radius for hidden neurons
                    radius = HIDDEN_RADIUS if neuron.neuron_type.value == 'hidden' else INPUT_OUTPUT_RADIUS
                    # Draw filled circle
                    draw.ellipse(
                        [(x - radius, y - radius), 
                         (x + radius, y + radius)],
                        fill=color, outline=(0, 0, 0)
                    )
            
            # Add title text (very fast)
            title = f'Episode {self.episode}, Step {step}: {len(architecture.neurons)} neurons, {len(architecture.connections)} connections'
            draw.text((20, 10), title, fill=(0, 0, 0))
            
            # Save with minimal compression
            diagram_file = f'architecture_diagrams/ep{self.episode:03d}/step{step:03d}.jpg'
            img.save(diagram_file, quality=70, optimize=False)
            
            print(f"      Architecture diagram saved: {diagram_file}", flush=True)

        except ImportError:
            print("      PIL not available, skipping diagram")
        except Exception as e:
            print(f"Failed to draw architecture diagram: {e}")
            traceback.print_exc()
