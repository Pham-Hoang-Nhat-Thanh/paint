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
import networkx as nx
import matplotlib.pyplot as plt

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
            max_epochs=config.search.quick_train_epochs
        )

        # Training infrastructure
        self.curriculum = TrainingCurriculum(
            total_episodes=config.supervised.num_episodes + 
                          config.mixed.num_episodes + 
                          config.self_play.num_episodes
        )
        
        self.neural_mcts = NeuralMCTS(
            action_space=self.action_space,
            policy_value_net=self.policy_value_net,
            device=self.device,
            exploration_weight=config.mcts.exploration_weight,
            curriculum=self.curriculum,
            quick_trainer=self.quick_trainer,
            reward_loss_weight=config.search.reward_loss_weight
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
        # File logger + JSONL metrics
        self.log_path = os.path.join(config.log_dir, "training.log")
        self.metrics_path = os.path.join(config.log_dir, "training_metrics.jsonl")

        # Configure logger (avoid duplicate handlers)
        self.logger = logging.getLogger(f"ArchitectureTrainer")
        self.logger.setLevel(logging.INFO)
        # Add file handler if not present
        abs_log_path = os.path.abspath(self.log_path)
        if not any(isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, 'baseFilename', '')) == abs_log_path
                   for h in self.logger.handlers):
            fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8')
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(fh)

        print(f"ArchitectureTrainer initialized on {self.device}")
        print(f"Total episodes: {self.curriculum.total_episodes}")
    
    def run_training_episode(self) -> Dict[str, Any]:
        """Run one complete training episode"""
        print(f"Starting episode {self.episode} (stage: {self.curriculum.get_stage()})")
        # Initialize architecture
        current_arch = NeuralArchitecture()
        
        episode_experiences = []
        episode_rewards = []
        episode_steps = 0
        
        # Get current training parameters
        train_params = self.curriculum.get_training_parameters()
        
        # Run search until convergence or max steps
        for step in range(self.config.search.max_steps_per_episode):
            # Decide whether to use policy network or MCTS
            use_policy = np.random.random() < train_params["policy_mix_ratio"]
            print(f"  Step {step}: using {'policy' if use_policy and train_params['stage'] != 'supervised' else 'MCTS'}")

            if use_policy and train_params["stage"] != "supervised":
                # Use policy network directly
                next_action = self._get_policy_action(current_arch, train_params["temperature"])
            else:
                # Use full MCTS
                best_node = self.neural_mcts.search(
                    current_arch, 
                    iterations=self.config.mcts.num_simulations,
                    temperature=train_params["temperature"]
                )
                next_action = best_node.action if best_node else None
            
            if next_action is None:
                break
                
            # Apply action and evaluate
            new_arch = self._apply_action(current_arch, next_action)
            print(f"    Applied action {next_action.action_type} (neurons={len(new_arch.neurons)}, connections={len(new_arch.connections)})")
            if new_arch is None:
                break

            # Draw architecture diagram after action
            self._draw_architecture_diagram(new_arch, step)
            reward = self._evaluate_architecture(new_arch)

            print(f"      Received reward: {reward:.4f}")
            
            # Store experience
            experience = self._create_experience(
                current_arch, next_action, reward, new_arch, train_params["stage"]
            )
            episode_experiences.append(experience)
            episode_rewards.append(reward)
            
            # Update current architecture
            current_arch = new_arch
            episode_steps += 1
            
            # Check termination conditions
            if self._should_terminate_episode(current_arch, step, reward):
                break
        
        # Process episode results
        episode_metrics = self._process_episode_results(
            current_arch, episode_rewards, episode_experiences, episode_steps
        )
        
        # Add experiences to replay buffer
        for exp in episode_experiences:
            priority = self._compute_experience_priority(exp, episode_metrics)
            self.experience_buffer.add(exp, priority)
        print(f"Episode {self.episode} finished: steps={episode_steps}, experiences={len(episode_experiences)}, avg_reward={np.mean(episode_rewards) if episode_rewards else 0.0:.4f}")
        
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
            new_arch.neurons[neuron_id] = type('Neuron', (), {
                'id': neuron.id,
                'neuron_type': neuron.neuron_type,
                'activation': neuron.activation,
                'layer_position': neuron.layer_position,
                'bias': neuron.bias
            })()
            new_arch.next_neuron_id = max(new_arch.next_neuron_id, neuron_id + 1)
        
        # Copy connections
        for conn in architecture.connections:
            new_arch.connections.append(type('Connection', (), {
                'source_id': conn.source_id,
                'target_id': conn.target_id,
                'weight': conn.weight,
                'enabled': conn.enabled
            })())
        
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
            print(f"      Evaluation returned accuracy={accuracy:.4f}, loss={loss:.4f}, reward={reward:.4f}")
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
    
    def _create_experience(self, state: NeuralArchitecture, action: Action, 
                          reward: float, next_state: NeuralArchitecture, stage: str) -> Dict:
        """Create experience tuple"""
        return {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'stage': stage,
            'timestamp': time.time()
        }
    
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
        """Train policy-value network on a batch of experiences"""
        if len(self.experience_buffer) < batch_size:
            return None
            
        # Sample batch
        experiences, indices, weights = self.experience_buffer.sample(batch_size)
        
        if not experiences:
            return None
            
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        
        # Process each experience
        for i, exp in enumerate(experiences):
            # Prepare targets
            targets = self._create_targets(exp)
            
            # Forward pass
            graph_data = self._prepare_graph_data(exp['state'])
            predictions = self.policy_value_net(graph_data)
            
            # Compute loss
            loss = self.loss_fn(predictions, targets) * weights[i]
            total_loss += loss
            
            # Store individual losses for logging
            with torch.no_grad():
                # Compute individual losses for logging
                individual_targets = self._create_targets(exp)
                individual_predictions = self.policy_value_net(graph_data)
                policy_loss += F.cross_entropy(individual_predictions['action_type'], 
                                             individual_targets['action_type'])
                value_loss += F.mse_loss(individual_predictions['value'].squeeze(),
                                       individual_targets['value'])
        
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
            'total_loss': total_loss.item() / len(experiences),
            'policy_loss': policy_loss.item() / len(experiences),
            'value_loss': value_loss.item() / len(experiences)
        }
        print(f"Train on batch: total_loss={out['total_loss']:.6f}, policy_loss={out['policy_loss']:.6f}, value_loss={out['value_loss']:.6f}")
        return out
    
    def _create_targets(self, experience: Dict) -> Dict:
        """Create training targets from experience"""
        action = experience['action']
        
        targets = {
            'action_type': torch.tensor([action.action_type.value]),
            'value': torch.tensor([experience['reward']])
        }
        
        # Add optional targets
        if action.source_neuron is not None:
            targets['source_neuron'] = torch.tensor([action.source_neuron])
        if action.target_neuron is not None:
            targets['target_neuron'] = torch.tensor([action.target_neuron])
        if action.activation is not None:
            targets['activation'] = torch.tensor([action.activation.value])
        
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
                train_metrics = self.train_on_batch(self.config.supervised.batch_size)
                if train_metrics:
                    episode_metrics.update(train_metrics)

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
            
            # Update curriculum
            self.curriculum.step()
            self.episode += 1
            
            # Update learning rate
            new_lr = self.curriculum.get_learning_rate()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Log progress
            self.training_history.append(episode_metrics)
            self._log_episode(episode_metrics)
            
            # Checkpoint if needed
            if self.curriculum.should_checkpoint(self.config.checkpoint_interval):
                self._save_checkpoint()
            
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
        """Save training checkpoint"""
        checkpoint = {
            'episode': self.episode,
            'policy_value_net_state': self.policy_value_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'curriculum_state': self.curriculum.get_state(),
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_value_net.load_state_dict(checkpoint['policy_value_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.curriculum.set_state(checkpoint['curriculum_state'])
        self.best_accuracy = checkpoint['best_accuracy']
        self.training_history = checkpoint['training_history']
        self.episode = checkpoint['episode']
        
        print(f"Checkpoint loaded from episode {self.episode}")

    def _draw_architecture_diagram(self, architecture: NeuralArchitecture, step: int):
        """Draw a diagram of the neural architecture after an action"""
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
            os.makedirs('architecture_diagrams', exist_ok=True)
            filename = f'architecture_diagrams/ep{self.episode:03d}_step{step:02d}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"      Architecture diagram saved: {filename}")

        except Exception as e:
            print(f"Failed to draw architecture diagram: {e}")
            traceback.print_exc()
