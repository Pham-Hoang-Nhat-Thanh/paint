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
from PIL import Image, ImageDraw
from blueprint_modules.network import Neuron, Connection

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
        )
        
        # QuickTrainer for final episode evaluation
        self.quick_trainer = QuickTrainer(
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            max_epochs=10
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
                # Verify tree reuse correctness
                root_neurons = sorted(mcts_root.architecture.neurons.keys())
                current_neurons = sorted(current_arch.neurons.keys())
                if root_neurons != current_neurons:
                    print(f"WARNING: Tree reuse architecture mismatch detected!")
                    print(f"  mcts_root neurons: {root_neurons}")
                    print(f"  current_arch neurons: {current_neurons}")
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
        
        # Evaluate final architecture with QuickTrainer to show true progress
        final_eval = self._evaluate_final_architecture(current_arch)
        print(f"Final Architecture Evaluation: Accuracy={final_eval['final_accuracy']:.4f}, Loss={final_eval['final_loss']:.4f} ({final_eval['method']})")
        episode_metrics.update(final_eval)
        
        # CRITICAL: Update all experiences with final episode reward (AlphaZero-style credit assignment)
        final_reward = episode_rewards[-1] if episode_rewards else 0.0
        for exp in episode_experiences:
            exp['value_target'] = final_reward
        
        # Add experiences to replay buffer
        for exp in episode_experiences:
            priority = self._compute_experience_priority(exp, episode_metrics)
            self.experience_buffer.add(exp, priority)

        # ===== LOG EPISODE RESULTS =====
        self._log_episode(episode_metrics)
        
        # ===== SAVE CHECKPOINT IF BEST ACCURACY =====
        if episode_metrics.get('is_best', False):
            self._save_checkpoint()
            print(f"New best accuracy! Checkpoint saved.")

        return episode_metrics
    
    
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
                          reward: float, next_state: NeuralArchitecture,
                          search_root=None) -> Dict:
        """Create experience tuple with MCTS visit distribution for AlphaZero-style training
        
        AlphaZero Key Insight: Store MCTS-improved policy π(s) = visit_count(a) / total_visits
        The network learns to predict this π, not raw network priors. This leverages MCTS's 
        superior exploration and planning.
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,  # Immediate reward (will be replaced with final episode reward)
            'next_state': next_state,
            'timestamp': time.time(),
            'mcts_policy': None,  # Will be filled below
            'mcts_actions': None   # For debugging
        }
        
        # Extract MCTS visit distribution (CRITICAL for AlphaZero learning!)
        if search_root is not None and hasattr(search_root, 'children'):
            try:
                # Get visit distribution from neural_mcts helper
                visit_distribution = self.neural_mcts.get_visit_distribution(
                    search_root, temperature=1.0
                )
                
                if visit_distribution is not None and len(visit_distribution) > 0:
                    experience['mcts_policy'] = visit_distribution
                    experience['mcts_actions'] = [child.action for child in search_root.children]
                    
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
    
    def _get_legal_action_mask(self, state: NeuralArchitecture) -> Dict[str, torch.Tensor]:
        """Create mask for legal actions in this state (FIX #3: Action Legality Masking)
        
        Prevents training on impossible action combinations. AlphaZero reference:
        "We actually have to correct for this manually by masking out illegal moves, 
        and then re-normalizing the remaining scores"
        
        Returns:
            Dict with masks for action_type, source_neuron, target_neuron, activation
            (1.0 = legal, 0.0 = illegal)
        """
        from blueprint_modules.action import ActionType
        
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

    def train_on_batch(self, batch_size: int = 32):
        """Train policy-value network on a batch of experiences (AlphaZero-style)
        
        FIX #1: Uses MCTS visit distribution as policy target (π learning)
        FIX #2: Trains on complete factorized policy (source, target, activation)
        FIX #3: Masks illegal actions in loss computation
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
        mcts_policy_loss_accum = 0.0
        total_experiences = 0
        
        # Process each experience individually (graph transformer architecture doesn't support batching)
        for arch_size, group_data in size_groups.items():
            print(f"  [Training] Processing {len(group_data)} experiences with {arch_size} neurons")
            for exp_idx, (exp, weight, idx) in enumerate(group_data):
                # Prepare graph data and targets
                graph_data = self._prepare_graph_data(exp['state'])
                targets = self._create_targets(exp)
                
                # Forward pass for single experience
                predictions = self.policy_value_net(graph_data)
                
                # ===== FIX #3: Get legal action mask =====
                action_mask = self._get_legal_action_mask(exp['state'])
                
                # Compute losses: policy (complete) + value (MSE) + MCTS policy (KL divergence)
                try:
                    # ===== FIX #3: Apply masking to predictions =====
                    masked_action_logits = predictions['action_type'].clone()
                    masked_action_logits[:, action_mask['action_type'] == 0] = -1e9
                    
                    masked_source_logits = predictions['source_logits'].clone()
                    masked_source_logits[:, action_mask['source_neuron'] == 0] = -1e9
                    
                    masked_target_logits = predictions['target_logits'].clone()
                    masked_target_logits[:, action_mask['target_neuron'] == 0] = -1e9
                    
                    masked_activation_logits = predictions['activation_logits'].clone()
                    masked_activation_logits[:, action_mask['activation'] == 0] = -1e9
                    
                    # ===== FIX #2: Complete factorized policy loss =====
                    action_type_loss = F.cross_entropy(
                        masked_action_logits,
                        targets['action_type']
                    )
                    
                    source_loss = F.cross_entropy(
                        masked_source_logits,
                        targets['source_neuron']
                    ) if 'source_neuron' in targets else torch.tensor(0.0)
                    
                    target_loss = F.cross_entropy(
                        masked_target_logits,
                        targets['target_neuron']
                    ) if 'target_neuron' in targets else torch.tensor(0.0)
                    
                    activation_loss = F.cross_entropy(
                        masked_activation_logits,
                        targets['activation']
                    ) if 'activation' in targets else torch.tensor(0.0)
                    
                    # Combine all policy components
                    total_policy_loss = (action_type_loss + source_loss + target_loss + activation_loss) / 4.0
                    
                    # ===== FIX #1: MCTS policy learning (KL divergence) =====
                    # Network learns to predict MCTS-improved policy π(s)
                    mcts_policy_loss = torch.tensor(0.0, device=self.device)
                    if 'mcts_policy' in targets and targets['mcts_policy'] is not None:
                        try:
                            mcts_policy_tensor = targets['mcts_policy'].unsqueeze(0).to(self.device)
                            # KL divergence: D_KL(MCTS || network)
                            mcts_policy_loss = F.kl_div(
                                F.log_softmax(masked_action_logits, dim=1),
                                mcts_policy_tensor,
                                reduction='batchmean'
                            ) * 0.5  # Weight this loss component
                            mcts_policy_loss_accum += mcts_policy_loss.item()
                        except Exception as e:
                            print(f"    [Warning] MCTS policy loss skipped: {e}")
                    
                    # Value loss (existing)
                    value_loss = F.mse_loss(
                        predictions['value'].squeeze(),
                        targets['value'].squeeze()
                    )
                    
                    # Total loss: policy + value + MCTS policy (AlphaZero)
                    loss = (total_policy_loss + value_loss + mcts_policy_loss) * weight
                    total_loss += loss
                    
                    policy_loss_accum += total_policy_loss.item()
                    value_loss_accum += value_loss.item()
                    
                except Exception as e:
                    print(f"    [ERROR] Loss computation failed for experience {exp_idx}/{len(group_data)}")
                    print(f"    Predictions keys: {predictions.keys()}")
                    print(f"    Targets keys: {targets.keys()}")
                    raise
            
            total_experiences += len(group_data)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_value_net.parameters(), 
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        # Update priorities
        new_priorities = [abs(exp['reward']) for exp in experiences]
        self.experience_buffer.update_priorities(indices, new_priorities)
        
        out = {
            'total_loss': total_loss.item() / total_experiences,
            'policy_loss': policy_loss_accum / total_experiences,
            'value_loss': value_loss_accum / total_experiences,
            'mcts_policy_loss': mcts_policy_loss_accum / total_experiences if total_experiences > 0 else 0.0
        }
        print(f"  [Training] Completed:")
        print(f"    total_loss={out['total_loss']:.6f}, policy={out['policy_loss']:.6f}, value={out['value_loss']:.6f}, mcts_policy={out['mcts_policy_loss']:.6f}")
        return out
    
    def _create_targets(self, experience: Dict) -> Dict:
        """Create training targets from experience (AlphaZero-style, complete factorized policy)
        
        AlphaZero trains on:
        1. Value target: final episode outcome z (credit assignment)
        2. Policy target: MCTS visit distribution π(s) for each action component
        3. All action parameters (not just action_type)
        """
        action = experience['action']

        # Use final episode reward (value_target) instead of immediate reward
        # This provides proper credit assignment for early actions
        value_target = experience.get('value_target', experience['reward'])
        
        # ===== VALUE TARGET =====
        targets = {
            'action_type': torch.tensor([action.action_type.value]).to(self.device),
            'value': torch.tensor([value_target]).to(self.device)  # Use final outcome, not immediate reward
        }

        # ===== COMPLETE FACTORIZED POLICY TARGETS =====
        # Add targets for all action components (not just action_type)
        # These allow the network to predict the complete action, not just its category
        
        # Source neuron target
        if action.source_neuron is not None:
            targets['source_neuron'] = torch.tensor([action.source_neuron]).to(self.device)
        else:
            # Padding value for actions without source neuron
            targets['source_neuron'] = torch.tensor([0]).to(self.device)
        
        # Target neuron target
        if action.target_neuron is not None:
            targets['target_neuron'] = torch.tensor([action.target_neuron]).to(self.device)
        else:
            # Padding value for actions without target neuron
            targets['target_neuron'] = torch.tensor([0]).to(self.device)
        
        # Activation type target
        if action.activation is not None:
            try:
                activation_idx = list(ActivationType).index(action.activation)
                targets['activation'] = torch.tensor([activation_idx]).to(self.device)
            except (ValueError, IndexError):
                targets['activation'] = torch.tensor([0]).to(self.device)
        else:
            # Padding value for actions without activation
            targets['activation'] = torch.tensor([0]).to(self.device)

        # ===== MCTS POLICY TARGET =====
        # Store MCTS visit distribution if available (CRITICAL for AlphaZero learning!)
        # Network learns to predict MCTS-improved policy, not raw priors
        if 'mcts_policy' in experience and 'mcts_actions' in experience:
            targets['mcts_policy'] = experience['mcts_policy']
            targets['mcts_actions'] = experience['mcts_actions']

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
                remaining = len(self.experience_buffer)
                while remaining > 0:
                    batch_size = min(self.config.batch_size, remaining)
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
        if self.episode % self.config.log_interval == 0:
            accuracy = metrics['final_accuracy']
            neurons = metrics['total_neurons']
            connections = metrics['total_connections']
            
            print(f"Episode {self.episode}: "
                  f"Accuracy={accuracy:.4f}, Neurons={neurons}, "
                  f"Connections={connections}, Best={self.best_accuracy:.4f}")

            # Structured file logging
            try:
                self.logger.info(f"Episode {self.episode} Accuracy={accuracy:.4f} "
                                 f"Neurons={neurons} Connections={connections} Best={self.best_accuracy:.4f}")
                # Append JSONL metrics
                with open(self.metrics_path, 'a', encoding='utf-8') as mf:
                    mf.write(json.dumps(metrics) + "\n")
            except Exception as e:
                # Fall back to console error if file write fails
                print(f"Failed to write logs: {e}")
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
            'config': self.config
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
            
            # Quick exit if too many neurons (keep training fast)
            if len(architecture.neurons) > 2000:
                return
            
            # Constants for layout
            CANVAS_WIDTH = 1400
            CANVAS_HEIGHT = 1000
            MARGIN = 50
            INPUT_OUTPUT_RADIUS = 3
            HIDDEN_RADIUS = 6  # Make hidden nodes bigger for visibility
            
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
