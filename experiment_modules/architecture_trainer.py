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
import gc
from types import SimpleNamespace
from PIL import Image, ImageDraw
from blueprint_modules.network import Neuron, Connection
from blueprint_modules.action import ActionType
from blueprint_modules.evolutionary_cycle import EvolutionaryCycle, Phase

# Suppress TF32 deprecation warnings
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior")

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
            num_activations=config.model.num_activations,
            self_attention_heads=config.model.num_heads,
            transformer_layers=config.model.num_layers,
        ).to(self.device)

        
        self.action_space = ActionSpace(
            max_neurons=config.search.max_neurons,
            max_connections=config.search.max_connections,
            max_steps_per_episode=config.search.max_steps_per_episode,
            connection_candidate_multiplier=config.search.connection_candidate_multiplier,
            model_max_neurons=config.model.max_neurons
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
            max_children=config.mcts.max_children if hasattr(config.mcts, 'max_children') else 50,
            max_neurons=config.model.max_neurons if hasattr(config.model, 'max_neurons') else 1000
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

        # Evolutionary cycle tracker
        self.evolutionary_cycle = EvolutionaryCycle(
            stability_threshold=config.search.stability_threshold if hasattr(config.search, 'stability_threshold') else 0.001
        )
        
        # Training state
        self.episode = 0
        self.best_reward = 0.0
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
        self.policy_value_net.eval()  # Set to eval mode for MCTS
        
        # Initialize MCTS tree reuse (for persistent tree across steps WITHIN this episode)
        mcts_root = None  # Will store the selected child node for next iteration
        
        # Run MCTS-guided search until convergence or max steps
        for step in range(self.config.search.max_steps_per_episode):
            print(f"Step {step + 1}/{self.config.search.max_steps_per_episode}:")
            step_start_time = time.time()
            
            # Ensure MCTS uses the episode-level evolutionary cycle so searches
            # start from the current episode phase and merge progress back.
            # Instead of overwriting `current_cycle`, expose the episode-level
            # cycle as `episode_level_cycle` so `NeuralMCTS.search` can copy
            # it and run a local search-level copy safely.
            self.neural_mcts.current_cycle = self.evolutionary_cycle.copy()
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
            else:
                print(f"Selected action: {next_action.action_type.name} | "
                      f"Source: {next_action.source_neuron} | "
                      f"Target: {next_action.target_neuron} | "
                      f"Activation: {next_action.activation}")

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
            if (self.episode + 1) % self.config.diagram_save_interval == 0 or self.episode == 0:
                self._draw_architecture_diagram(new_arch, step)
            reward = best_node.value / best_node.visits # Use average value as reward

            ce_start = time.time()
            # Store experience with MCTS visit distribution
            experience = self._create_experience(
                current_arch, next_action, reward, new_arch,
                search_root=search_root  # Contains visit distribution from MCTS
            )
            ce_end = time.time()
            print(f"Experience creation took {ce_end - ce_start:.4f} seconds")

            episode_experiences.append(experience)
            episode_rewards.append(reward)
            
            # Update current architecture
            current_arch = new_arch
            episode_steps += 1
            
            # Update tree reuse: best_node IS the child node we want to reuse as next root
            if best_node is not None:
                mcts_root = best_node
                mcts_root.parent = None # MEMORY LEAK FIX: Prune parent to allow GC

            # Check for phase transition (use conservative advance rule)
            self.evolutionary_cycle.add_evaluation(reward)
            if self.evolutionary_cycle.should_advance():
                prev_phase = self.evolutionary_cycle.current_phase
                prev_iter = self.evolutionary_cycle.phase_iteration_count
                self.evolutionary_cycle.advance_phase()
                print(f"Advancing evolutionary phase: {prev_phase} -> {self.evolutionary_cycle.current_phase} after {prev_iter} evaluations")

            # Calculate timing information
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time

             # Check termination conditions
            terminate_check = self._should_terminate_episode(current_arch, step)
            if terminate_check == "max_steps" or terminate_check == "penalty":
                print(f"Episode termination condition met at step {step}")
                break

            print(f"Step completed: Reward = {reward:.4f} | Step time: {step_duration:.2f}s")

        # Process episode results
        episode_metrics = self._process_episode_results(
            current_arch, episode_rewards, episode_experiences, episode_steps
        )
        
        if terminate_check == "penalty":
            final_reward = -1.0  # Assign a penalty reward
        else:
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

            accuracy_deviation = self.config.search.target_accuracy - accuracy
            raw_reward = self.config.search.reward_accuracy_weight * np.exp(-accuracy_deviation) * accuracy\
                        - self.config.search.reward_loss_weight * loss_norm \
                        - self.config.search.reward_complexity_weight * complexity_norm
            final_reward = max(0.0, raw_reward)  # Ensure non-negative reward
            print(f"Final Reward Calculation: Raw={raw_reward:.4f}, Final={final_reward:.4f} (Accuracy Dev={accuracy_deviation:.4f}, Loss Norm={loss_norm:.4f}, Complexity Norm={complexity_norm:.4f})")
        for exp in episode_experiences:
            exp['value_target'] = final_reward
        
        # Add experiences to replay buffer
        for exp in episode_experiences:
            priority = self._compute_experience_priority(exp, episode_metrics)
            self.experience_buffer.add(exp, priority)

        # ===== LOG EPISODE RESULTS =====
        self._log_episode(episode_metrics)

        # ===== EPISODE-LEVEL CACHE CLEANUP =====
        # Centralized helper: clear MCTS episode-level caches (evaluation cache)
        # and offload node-level policy tensors for the final root so memory
        # doesn't accumulate across episodes. This preserves intra-episode
        # reuse while ensuring episode boundaries free unnecessary memory.
        try:
            if hasattr(self, 'neural_mcts'):
                try:
                    # Pass the last root (if any). At episode boundary we fully clear
                    # node-level policy tensors to free GPU memory. Experiences have
                    # already been materialized to CPU above, so full clear is safe.
                    roots = mcts_root if 'mcts_root' in locals() else None
                    self.neural_mcts.clear_episode_caches(roots=roots, preserve_roots=False)
                except Exception:
                    pass

            # Clear ActionManager caches too
            try:
                if hasattr(self, 'action_manager') and hasattr(self.action_manager, 'clear_cache'):
                    self.action_manager.clear_cache()
            except Exception:
                pass

            # Free device caches
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass
        
        gc.collect()

        return episode_metrics, current_arch.to_serializable_dict()
      
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
                policy_output = self.policy_value_net(graph_data, phase = self.evolutionary_cycle.current_phase.value)
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
        """Optimized: Prepare graph data for neural network using cached sorted IDs
        
        Returns GPU tensors for single-graph evaluation (used during MCTS).
        For batched training, use _batch_graph_data instead.
        """
        graph_data = architecture.to_graph_representation()
        
        # Move to GPU immediately for single-graph evaluation
        graph_data['node_features'] = graph_data['node_features'].unsqueeze(0).to(self.device)
        graph_data['edge_index'] = graph_data['edge_index'].to(self.device)

        # Use cached sorted neuron IDs from graph representation instead of sorting again
        sorted_neuron_ids = graph_data['sorted_neuron_ids']
        layer_positions = [architecture.neurons[neuron_id].layer_position for neuron_id in sorted_neuron_ids]
        graph_data['layer_positions'] = torch.FloatTensor([layer_positions]).to(self.device)
        
        return graph_data
    
    def _batch_graph_data(self, graph_data_list: List[Dict]) -> Dict:
        """Process graph dicts into batched format efficiently on GPU.
        
        Accepts EITHER:
        - CPU graph representations (from to_graph_representation()) for training batches
        - GPU graph dicts (from _prepare_graph_data()) for single graphs
        
        NOTE: PolicyValueNetwork requires CUDA inputs. We batch and transfer once.
        
        Optimizations:
        - Keep tensors on GPU (no CPU detour)
        - Use CUDA streams for parallel operations
        - Accepts CPU input to avoid serialized transfers
        """
        if not graph_data_list:
            return {}
            
        if len(graph_data_list) == 1:
            # Single graph - ensure on GPU and return
            gd = graph_data_list[0]
            
            # Handle both CPU (from to_graph_representation) and GPU inputs
            node_feats = gd['node_features']
            if node_feats.device.type == 'cpu':
                node_feats = node_feats.to(self.device)
            else:
                node_feats = node_feats.squeeze(0) if node_feats.dim() == 3 else node_feats
            
            edge_idx = gd['edge_index'].to(self.device) if gd['edge_index'].device.type == 'cpu' else gd['edge_index']
            layer_pos = gd['layer_positions']
            if layer_pos.device.type == 'cpu':
                layer_pos = layer_pos.to(self.device)
            else:
                layer_pos = layer_pos.squeeze(0) if layer_pos.dim() == 3 else layer_pos
            
            num_nodes = node_feats.shape[0]
            return {
                'node_features': node_feats.unsqueeze(0) if node_feats.dim() == 2 else node_feats,
                'edge_index': edge_idx,
                'layer_positions': layer_pos.unsqueeze(0) if layer_pos.dim() == 2 else layer_pos,
                'batch': torch.zeros(num_nodes, dtype=torch.long, device=self.device),
                'num_graphs': 1
            }
                
        batch_size = len(graph_data_list)
        streams = [torch.cuda.Stream() for _ in range(4)]  # One per tensor type
        
        # Collect data - handle both CPU and GPU inputs
        all_node_features = []
        all_layer_positions = []
        all_edge_indices = []
        batch_tensor = []
        
        node_offset = 0
        for graph_idx, gd in enumerate(graph_data_list):
            # Handle CPU tensors (from to_graph_representation) and GPU tensors
            node_feats = gd['node_features']
            if node_feats.dim() == 3:
                node_feats = node_feats.squeeze(0)  # Remove batch dim if present
            
            layer_pos = gd['layer_positions']
            if layer_pos.dim() == 3:
                layer_pos = layer_pos.squeeze(0)
            
            num_nodes = node_feats.shape[0]
            
            all_node_features.append(node_feats) 
            all_layer_positions.append(layer_pos)
            
            # Process edges
            edge_index = gd['edge_index']
            if edge_index.shape[1] > 0:
                offset_edges = edge_index + node_offset
                all_edge_indices.append(offset_edges)
            
            # Track graph assignment
            batch_tensor.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
            node_offset += num_nodes
            
        # Concatenate on CPU first, then batch transfer to GPU using streams
        with torch.cuda.stream(streams[0]):
            batched_features = torch.cat(all_node_features, dim=0).unsqueeze(0).to(self.device)
            
        with torch.cuda.stream(streams[1]):    
            batched_positions = torch.cat(all_layer_positions, dim=0).unsqueeze(0).to(self.device)
            
        with torch.cuda.stream(streams[2]):
            if all_edge_indices:
                batched_edges = torch.cat(all_edge_indices, dim=1).to(self.device)
            else:
                batched_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        with torch.cuda.stream(streams[3]):
            batch_tensor = torch.cat(batch_tensor, dim=0).to(self.device)
        
        # Synchronize all streams before returning
        torch.cuda.synchronize()
        
        return {
            'node_features': batched_features,
            'edge_index': batched_edges, 
            'layer_positions': batched_positions,
            'batch': batch_tensor,
            'num_graphs': batch_size
        }

    def _create_experience(self, state: NeuralArchitecture, action: Action, 
                      reward: float, next_state: NeuralArchitecture,
                      search_root=None) -> Dict:
        """Create experience with true AlphaZero MCTS policy targets"""
        exp_phase = int(self.evolutionary_cycle.current_phase.value)
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': time.time(),
            'mcts_policy': None,  # Raw visit distribution
            'mcts_actions': None,  # Action objects
            'evolutionary_phase': exp_phase,
            'legal_action_mask': self._get_legal_action_mask(state, phase=exp_phase)
        }

        # Extract MCTS visit distribution (AlphaZero policy improvement)
        if search_root is not None and hasattr(search_root, 'children'):
            try:
                visit_distribution = self.neural_mcts.get_visit_distribution(
                    search_root, temperature=1.0
                )
                
                if visit_distribution is not None and len(visit_distribution) > 0:
                    # Store raw distribution as CPU-native for safety
                    if torch.is_tensor(visit_distribution):
                        experience['mcts_policy'] = visit_distribution.detach().cpu().numpy().astype(np.float32)
                    else:
                        experience['mcts_policy'] = np.array(visit_distribution, dtype=np.float32)

                    # Store the actual action objects
                    experience['mcts_actions'] = [child.action for child in search_root.children]
                    
            except Exception as e:
                print(f"    [Warning] Failed to extract MCTS policy: {e}")
                
        return experience

    def _create_targets(self, experience: Dict) -> Dict:
        """Create true AlphaZero training targets - direct MCTS policy"""
        action = experience['action']
        value_target = experience.get('value_target', 0.0)
        
        targets = {
            'action_type': action.action_type.value,
            'value': torch.tensor([value_target], dtype=torch.float32, device=self.device)
        }

        # Store component targets for the taken action (optional supervised learning)
        if action.source_neuron is not None:
            targets['source_neuron'] = action.source_neuron
        
        if action.target_neuron is not None:
            targets['target_neuron'] = action.target_neuron
        
        if action.activation is not None:
            targets['activation'] = list(ActivationType).index(action.activation)

        # ===== ALPHAZERO POLICY TARGET =====
        # Simply store the MCTS visit distribution and corresponding actions
        if 'mcts_policy' in experience and 'mcts_actions' in experience:
            # Store as tensors on device
            if isinstance(experience['mcts_policy'], np.ndarray):
                targets['mcts_policy'] = torch.tensor(
                    experience['mcts_policy'], dtype=torch.float32, device=self.device
                )
            else:
                targets['mcts_policy'] = experience['mcts_policy'].to(self.device)
                
            targets['mcts_actions'] = experience['mcts_actions']
                
        return targets

    def _should_terminate_episode(self, architecture: NeuralArchitecture, 
                                 step: int) -> str:
        """Check if episode should terminate"""
        # Check max steps
        if step >= self.config.search.max_steps_per_episode - 1:
            return "max_steps"
        
        # Check if number of connections and hidden neurons is too low
        if (len(architecture.neurons) <= self.config.search.min_neurons and
            len(architecture.connections) <= self.config.search.min_connections):
            return "penalty"
        
        return ""
    
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
            'best_reward': max(rewards) if rewards else 0.0
        }
        
        # Update best reward
        if final_reward > self.best_reward:
            self.best_reward = final_reward
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
      
    def _get_legal_action_mask(self, state: NeuralArchitecture, phase: int = None) -> Dict[str, torch.Tensor]:
            """Create mask for legal actions on CPU (no GPU transfers).
            
            This is called during batch preparation to avoid per-graph GPU transfers.
            Masks are moved to GPU in batch later.
            
            Returns:
                Dict with masks on CPU device
            """

            # Determine phase key to use for per-phase caching
            try:
                if phase is None:
                    phase_key = int(self.evolutionary_cycle.current_phase.value) if hasattr(self, 'evolutionary_cycle') else 0
                else:
                    phase_key = int(phase)
            except Exception:
                phase_key = 0

            # Check cache on the architecture instance to avoid recomputation (per-phase)
            try:
                cache_by_phase = getattr(state, '_cached_legal_action_mask_cpu_by_phase', None)
                if cache_by_phase and isinstance(cache_by_phase, dict) and phase_key in cache_by_phase:
                    return cache_by_phase[phase_key]
            except Exception:
                pass

            # Initialize all-zero masks (everything starts as illegal)
            action_type_mask = torch.zeros(len(ActionType), dtype=torch.float32)
            source_mask = torch.zeros(self.config.model.max_neurons, dtype=torch.float32)
            target_mask = torch.zeros(self.config.model.max_neurons, dtype=torch.float32)
            activation_mask = torch.zeros(len(ActivationType), dtype=torch.float32)
            
            # Vectorized mask construction for improved performance
            try:
                # Select evolutionary cycle to use for computing legal actions
                if phase is not None:
                    try:
                        # Lightweight phase-only object to avoid constructing full EvolutionaryCycle
                        tmp_cycle = SimpleNamespace(current_phase=Phase(int(phase)))
                        legal_actions = self.action_space.get_valid_actions(state, evolutionary_cycle=tmp_cycle)
                    except Exception:
                        legal_actions = self.action_space.get_valid_actions(state, evolutionary_cycle=self.evolutionary_cycle)
                else:
                    # Use the episode-level evolutionary cycle when computing legal actions
                    # so the mask matches the search behavior (phase-dependent action sets).
                    legal_actions = self.action_space.get_valid_actions(state, evolutionary_cycle=self.evolutionary_cycle)
                if not legal_actions:
                    # Return empty masks if no legal actions are available
                    return {
                        'action_type': action_type_mask, 'source_neuron': source_mask,
                        'target_neuron': target_mask, 'activation': activation_mask
                    }

                # Vectorized creation of masks from the list of legal actions
                types = torch.tensor([a.action_type.value for a in legal_actions], dtype=torch.long)
                sources = torch.tensor([a.source_neuron for a in legal_actions if a.source_neuron is not None], dtype=torch.long)
                targets = torch.tensor([a.target_neuron for a in legal_actions if a.target_neuron is not None], dtype=torch.long)
                
                activation_indices = []
                for act in legal_actions:
                    if act.activation is not None:
                        try:
                            activation_indices.append(list(ActivationType).index(act.activation))
                        except ValueError:
                            pass # Should not happen with valid actions
                activations = torch.tensor(activation_indices, dtype=torch.long)

                action_type_mask.scatter_(0, types, 1.0)
                source_mask.scatter_(0, sources, 1.0)
                target_mask.scatter_(0, targets, 1.0)
                activation_mask.scatter_(0, activations, 1.0)
                
            except Exception:
                # Fallback for safety, though the vectorized approach should be robust
                for action in self.action_space.get_valid_actions(state):
                    action_type_mask[action.action_type.value] = 1.0
                    if action.source_neuron is not None: source_mask[action.source_neuron] = 1.0
                    if action.target_neuron is not None: target_mask[action.target_neuron] = 1.0
                    if action.activation is not None:
                        try:
                            activation_mask[list(ActivationType).index(action.activation)] = 1.0
                        except (ValueError, IndexError):
                            pass
            
            res = {
                'action_type': action_type_mask,
                'source_neuron': source_mask,
                'target_neuron': target_mask,
                'activation': activation_mask,
            }

            # Cache on the architecture instance for future calls during the same episode (per-phase)
            try:
                cache_by_phase = getattr(state, '_cached_legal_action_mask_cpu_by_phase', None)
                if cache_by_phase is None or not isinstance(cache_by_phase, dict):
                    cache_by_phase = {}
                cache_by_phase[phase_key] = res
                setattr(state, '_cached_legal_action_mask_cpu_by_phase', cache_by_phase)
            except Exception:
                pass

            return res

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
        
        
        # Initialize gradient accumulation across all sub-batches
        self.optimizer.zero_grad()
        total_loss_accum = 0.0
        value_loss_accum = 0.0
        mcts_policy_loss_accum = 0.0
        num_processed = 0
        valid_indices_list = []
        valid_rewards_list = []
        
        # === SUB-BATCH PROCESSING PHASE ===
        
        # Process batch in smaller sub-batches for true gradient accumulation
        
        for sub_batch_idx in range(0, len(experiences), sub_batch_size):
            sub_batch_end = min(sub_batch_idx + sub_batch_size, len(experiences))
            sub_experiences = experiences[sub_batch_idx:sub_batch_end]
            sub_weights = weights[sub_batch_idx:sub_batch_end]
            sub_indices = indices[sub_batch_idx:sub_batch_end]
            
            # Process this sub-batch with pre-computed weights (avoid per-graph recalculation)
            sub_result = self._process_sub_batch(sub_experiences, sub_weights, sub_indices)
            
            if sub_result is not None:
                # Backward on this sub-batch (accumulates gradients)
                loss_tensor = sub_result['loss']
                
                # DIAGNOSTIC: Verify loss is on GPU before backward
                if loss_tensor.device.type == 'cpu':
                    print(f"  [CRITICAL ERROR] Loss tensor on CPU before backward! Device: {loss_tensor.device}")
                    print(f"    Has gradients: {loss_tensor.requires_grad}")
                    
                loss_tensor.backward()
                
                # Verify gradients exist after backward
                has_gradients = False
                for param in self.policy_value_net.parameters():
                    if param.grad is not None:
                        has_gradients = True
                        break
                if not has_gradients:
                    print(f"  [WARNING] No gradients after backward pass")
                
                # Accumulate metrics (raw and weighted)
                total_loss_accum += sub_result['total_loss']
                value_loss_accum += sub_result['value_loss']
                mcts_policy_loss_accum += sub_result['mcts_policy_loss']
                num_processed += sub_result['num_graphs']
                
                # Track for priority updates
                valid_indices_list.extend(sub_result['valid_indices'])
                valid_rewards_list.extend(sub_result['valid_rewards'])
            
            # (removed per-sub-batch GPU cache clearing to avoid forcing frequent
            # GPU memory synchronization which reduces utilization)
        
        if num_processed == 0:
            print(f"  [Training] No legal actions in batch, skipping")
            return None
        
        # === OPTIMIZATION PHASE ===
        # Single optimizer step after all sub-batches (where accumulated gradients apply)
        torch.nn.utils.clip_grad_norm_(
            self.policy_value_net.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()


        # === PRIORITY UPDATE PHASE ===
        
        # Update priorities for all valid experiences
        if valid_indices_list:
            self.experience_buffer.update_priorities(valid_indices_list, valid_rewards_list)

        print(f"  [Training] Processed {num_processed} graphs in batch of {batch_size} with sub-batch size {sub_batch_size}.", end='')
        print(f"Total Loss: {total_loss_accum / num_processed:.4f}, Value Loss: {value_loss_accum / num_processed:.4f}, MCTS Policy Loss: {mcts_policy_loss_accum / num_processed:.4f}")
        
        # Return averaged metrics
        metrics = {
            'total_loss': total_loss_accum / num_processed if num_processed > 0 else 0.0,
            'value_loss': value_loss_accum / num_processed if num_processed > 0 else 0.0,
            'mcts_policy_loss': mcts_policy_loss_accum / num_processed if num_processed > 0 else 0.0,
            'num_graphs': num_processed
        }

        return metrics
    
    def _process_sub_batch(self, experiences, weights, indices):
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
                - value_loss: float value
                - mcts_policy_loss: float value
                - num_graphs: int (number of valid graphs)
                - valid_indices: list of buffer indices for priority update
                - valid_rewards: list of rewards for priority update
        """
        # === PHASE 1: BATCH CPU WORK (non-blocking, no GPU transfers yet) ===
        # Extract all graph representations, action masks, and targets on CPU first.
        # CRITICAL OPTIMIZATION: Use cached legal_action_mask from experience (52% speedup!)
        # Fallback: Recompute for old experiences without cache (backward compat)
        cpu_graph_reps = []
        action_masks_list_cpu = []  # Keep on CPU during batch prep
        targets_list = []
        valid_indices_list = []
        weights_list = []
        valid_experiences = []
        
        # Diagnostic counters for cache hit/miss
        cached_masks_used = 0
        computed_masks_fallback = 0
        
        for exp_idx, (exp, weight, idx) in enumerate(zip(experiences, weights, indices)):
            # OPTIMIZATION: Try cached mask first (new experiences from current episode)
            # Fallback: Compute on-the-fly for old experiences without cache (backward compat)
            if 'legal_action_mask' in exp and exp['legal_action_mask'] is not None:
                # Fast path: Use cached mask (O(1) dict lookup)
                action_mask_cpu = exp['legal_action_mask']
                cached_masks_used += 1
            else:
                # Fallback path: Recompute mask for old experiences (backward compat)
                # Prefer computing the mask for the phase the experience was created in.
                # This adds ~183ms per graph but only happens for old checkpoints
                phase_for_exp = exp.get('evolutionary_phase', None)
                action_mask_cpu = self._get_legal_action_mask(exp['state'], phase=phase_for_exp)
                # Cache it for future use (in case buffer is reused)
                exp['legal_action_mask'] = action_mask_cpu
                computed_masks_fallback += 1
            
            action = exp['action']
            
            # Validate action components (all CPU work)
            is_action_type_legal = action_mask_cpu['action_type'][action.action_type.value] > 0
            if not is_action_type_legal:
                continue
            
            is_source_legal = (action.source_neuron is None or
                             (action.source_neuron < len(action_mask_cpu['source_neuron']) and
                              action_mask_cpu['source_neuron'][action.source_neuron] > 0))
            if action.action_type in [ActionType.ADD_CONNECTION, ActionType.REMOVE_CONNECTION]:
                is_target_legal = (action.target_neuron is not None and
                                   action.target_neuron < len(action_mask_cpu['target_neuron']) and
                                   action_mask_cpu['target_neuron'][action.target_neuron] > 0)
            else:
                is_target_legal = action.target_neuron is None
            is_activation_legal = (action.activation is None or action_mask_cpu['activation'].max() > 0)
            
            if not (is_source_legal and is_target_legal and is_activation_legal):
                continue
            
            # CPU-side data only (no GPU transfers yet)
            graph_rep = exp['state'].to_graph_representation()
            cpu_graph_reps.append(graph_rep)
            action_masks_list_cpu.append(action_mask_cpu)
            targets_list.append(self._create_targets(exp))
            valid_indices_list.append(idx)
            weights_list.append(weight.item() if isinstance(weight, torch.Tensor) else weight)
            valid_experiences.append(exp)
        
        if not cpu_graph_reps:
            return None
        
        # Log cache diagnostics if we're using old experiences (for backward compat verification)
        if computed_masks_fallback > 0:
            print(f"  [Cache] Legal action masks: {cached_masks_used} cached, {computed_masks_fallback} computed (fallback for old experiences)")
        
        # === PHASE 2: BATCH GPU TRANSFERS (coordinated with CUDA streams) ===
        # Now batch all CPU graph representations for efficient parallel GPU transfer
        # _batch_graph_data uses CUDA streams to maximize throughput
        batched_graph_data = self._batch_graph_data(cpu_graph_reps)
        num_graphs = len(cpu_graph_reps)
        
        # DIAGNOSTIC: Verify batched data is on GPU
        if batched_graph_data['node_features'].device.type == 'cpu':
            print(f"  [CRITICAL ERROR] Batched node_features still on CPU! Device: {batched_graph_data['node_features'].device}")

        # Forward pass on sub-batch through network
        # No internal sub-batching needed since sub-batch is already small
        # Data is already on GPU from _batch_graph_data, no need for second transfer
        # Just ensure policy-value net is on device and in training mode
        self.policy_value_net.to(self.device)
        self.policy_value_net.train()  # CRITICAL: Ensure network is in training mode

        # Use the most common evolutionary phase among experiences in this sub-batch
        phases = [exp.get('evolutionary_phase', int(self.evolutionary_cycle.current_phase.value)) for exp in experiences]
        try:
            # pick modal phase to approximate per-graph conditioning
            batch_phase = int(max(set(phases), key=phases.count))
        except Exception:
            batch_phase = int(self.evolutionary_cycle.current_phase.value)

        predictions = self.policy_value_net(batched_graph_data, phase = batch_phase)
        
        # DIAGNOSTIC: Check if predictions are on GPU
        pred_device = predictions['action_type'].device
        if str(pred_device) == 'cpu':
            print(f"  [WARNING] Network predictions on CPU! Expected on {self.device}")
            print(f"    Batched data node_features device: {batched_graph_data['node_features'].device}")
            print(f"    Network device: {next(self.policy_value_net.parameters()).device}")
            # Force predictions to GPU immediately
            for k in predictions:
                if torch.is_tensor(predictions[k]):
                    predictions[k] = predictions[k].to(self.device)

        # Compute losses for this sub-batch
        # Keep total_loss as the autograd-carrying tensor; accumulate monitoring
        # scalars as tensors on device to avoid repeated GPU->CPU syncs (.item()).
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        value_loss_sum = torch.tensor(0.0, device=self.device)
        mcts_policy_loss_sum = torch.tensor(0.0, device=self.device)
        try:
            values = predictions['value']                      # [num_graphs, 1]
            
            # === VALUE LOSS (vectorized) ===
            value_targets = torch.stack([t['value'].squeeze() if t['value'].dim() > 0 else t['value'] 
                                         for t in targets_list], dim=0).to(self.device)
            values_squeezed = values.squeeze(-1)
            value_loss = F.mse_loss(values_squeezed, value_targets, reduction='none')  # [num_graphs]
            
            # === ALPHAZERO POLICY LOSS (vectorized across sub-batch) ===
            policy_loss = torch.zeros(num_graphs, device=self.device)

            # Flatten all MCTS actions across graphs in this sub-batch
            flat_graph_idx = []
            flat_action_types = []
            flat_sources = []
            flat_targets = []
            flat_activations = []
            flat_mcts_probs = []

            for g_idx, targets in enumerate(targets_list):
                if 'mcts_policy' not in targets or 'mcts_actions' not in targets:
                    continue
                for action, visit_prob in zip(targets['mcts_actions'], targets['mcts_policy']):
                    flat_graph_idx.append(g_idx)
                    flat_action_types.append(int(action.action_type.value))
                    flat_sources.append(-1 if action.source_neuron is None else int(action.source_neuron))
                    flat_targets.append(-1 if action.target_neuron is None else int(action.target_neuron))
                    flat_activations.append(-1 if action.activation is None else int(list(ActivationType).index(action.activation)))
                    flat_mcts_probs.append(float(visit_prob))

            if len(flat_graph_idx) > 0:
                device = self.device
                fg = torch.tensor(flat_graph_idx, dtype=torch.long, device=device)
                fa = torch.tensor(flat_action_types, dtype=torch.long, device=device)
                fs = torch.tensor(flat_sources, dtype=torch.long, device=device)
                ft = torch.tensor(flat_targets, dtype=torch.long, device=device)
                fact = torch.tensor(flat_activations, dtype=torch.long, device=device)
                fm = torch.tensor(flat_mcts_probs, dtype=torch.float32, device=device)

                # p(action_type) for each flat entry
                at_probs = F.softmax(predictions['action_type'], dim=-1)
                p_at = at_probs[fg, fa]

                # Initialize flat network probabilities
                p_flat = p_at.clone()

                # Group by (action_type, source) to compute conditional logits efficiently
                from collections import defaultdict
                groups = defaultdict(list)
                for i, (a, s) in enumerate(zip(flat_action_types, flat_sources)):
                    groups[(a, s)].append(i)

                shared_features = predictions.get('shared_features', None)
                source_encoder = predictions.get('source_encoder', None)

                for (a_val, s_val), indices in groups.items():
                    action_name = self.action_manager._get_action_type_name(a_val)
                    idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)

                    # SOURCE probability if available
                    if s_val >= 0 and 'source_logits_dict' in predictions and action_name in predictions['source_logits_dict']:
                        src_logits = predictions['source_logits_dict'][action_name]
                        src_probs = F.softmax(src_logits, dim=-1)
                        p_src_vals = src_probs[fg[idx_tensor], fs[idx_tensor]]
                        p_flat[idx_tensor] = p_flat[idx_tensor] * p_src_vals

                    # TARGET probability if applicable and head exists (requires valid source)
                    if s_val >= 0 and ft[idx_tensor].min().item() >= 0 and 'target_heads' in predictions and action_name in predictions['target_heads'] and shared_features is not None and source_encoder is not None:
                        # shared_features for this group's graphs
                        g_shared = shared_features[fg[idx_tensor]]  # [k, hidden]
                        # compute source feature once
                        src_one_hot = F.one_hot(torch.tensor([s_val], device=device), self.config.model.max_neurons).float()
                        src_feat = source_encoder(src_one_hot)  # [1, h']
                        src_feat_exp = src_feat.expand(g_shared.shape[0], -1)
                        conditioned = torch.cat([g_shared, src_feat_exp], dim=-1)
                        logits = predictions['target_heads'][action_name](conditioned)
                        probs = F.softmax(logits, dim=-1)
                        p_t_vals = probs[torch.arange(probs.shape[0], device=device), ft[idx_tensor]]
                        p_flat[idx_tensor] = p_flat[idx_tensor] * p_t_vals

                    # ACTIVATION probability if applicable
                    if fact[idx_tensor].min().item() >= 0 and 'activation_heads' in predictions and action_name in predictions['activation_heads'] and shared_features is not None:
                        g_shared = shared_features[fg[idx_tensor]]
                        # If source is valid and activation head expects conditioning, condition on source
                        if s_val >= 0 and source_encoder is not None and action_name == 'modify_activation':
                            src_one_hot = F.one_hot(torch.tensor([s_val], device=device), self.config.model.max_neurons).float()
                            src_feat = source_encoder(src_one_hot)
                            src_feat_exp = src_feat.expand(g_shared.shape[0], -1)
                            conditioned = torch.cat([g_shared, src_feat_exp], dim=-1)
                            logits = predictions['activation_heads'][action_name](conditioned)
                        else:
                            # Unconditional activation head (e.g., add_neuron)
                            logits = predictions['activation_heads'][action_name](g_shared)
                        probs = F.softmax(logits, dim=-1)
                        p_act_vals = probs[torch.arange(probs.shape[0], device=device), fact[idx_tensor]]
                        p_flat[idx_tensor] = p_flat[idx_tensor] * p_act_vals

                # Now compute per-graph AlphaZero cross-entropy loss using grouped flat probabilities
                eps = 1e-8
                for g in range(num_graphs):
                    mask = (fg == g)
                    if mask.sum() == 0:
                        continue
                    net_p = p_flat[mask]
                    mcts_p = fm[mask]
                    # normalize
                    net_p = net_p / (net_p.sum() + eps)
                    mcts_p = mcts_p / (mcts_p.sum() + eps)
                    policy_loss[g] = -torch.sum(mcts_p * torch.log(net_p + eps))
            
            # === COMBINE LOSSES ===
            weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=self.device)
            graph_losses = (policy_loss + value_loss) * weights_tensor
            total_loss = graph_losses.sum()

            # Accumulate metrics (no .item() to avoid GPU sync)
            value_loss_sum = value_loss.sum().detach()
            mcts_policy_loss_sum = policy_loss.sum().detach()
        except Exception as e:
            print(f"    [ERROR] Policy+Value loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Priorities for replay buffer update: outcome + surprise (TD-error proxy)
        # Each experience used for training should have a priority reflecting:
        # 1. The final outcome (value_target) it led to
        # 2. The surprise/error: |value_target - immediate_reward|
        valid_rewards = []
        surprise_weight = getattr(self.config.search, 'priority_surprise_weight', 0.5)
        for exp in valid_experiences:
            # Get final target (ground truth for this experience). Use only
            # the explicitly stored final episode reward; do NOT fall back to
            # the immediate per-step reward.
            value_target = exp.get('value_target', None)
            if value_target is None:
                value_target = 0.0
            immediate_reward = exp['reward']
            
            # Surprise: how wrong was the network's prediction?
            surprise = abs(value_target - immediate_reward)
            surprise = min(surprise, 1.0)
            
            # Priority combines outcome quality + informativeness
            priority = value_target + surprise_weight * surprise
            valid_rewards.append(max(0.0, priority))
        
        try:
            value_loss_val = float(value_loss_sum.detach().cpu().item())
        except Exception:
            value_loss_val = 0.0

        try:
            mcts_policy_loss_val = float(mcts_policy_loss_sum.detach().cpu().item())
        except Exception:
            mcts_policy_loss_val = 0.0
      
        return {
            'loss': total_loss,
            'total_loss': total_loss.item(),                  
            'value_loss': value_loss_val,
            'mcts_policy_loss': mcts_policy_loss_val,         
            'num_graphs': num_graphs,
            'valid_indices': valid_indices_list,
            'valid_rewards': valid_rewards,
        }
        
    def _compute_action_probability(self, predictions, graph_idx, action: Action):
        """Compute network's probability for a specific composite action"""
        try:
            # Get action type probability
            action_type_logits = predictions['action_type'][graph_idx]
            action_type_probs = F.softmax(action_type_logits, dim=-1)
            p_action_type = action_type_probs[action.action_type.value]
            
            total_prob = p_action_type
            
            # For conditional network, compute conditional probabilities
            action_type_name = self.action_manager._get_action_type_name(action.action_type.value)
            
            # Source probability (if applicable)
            if (action.source_neuron is not None and 
                'source_logits_dict' in predictions and
                action_type_name in predictions['source_logits_dict']):
                
                source_logits = predictions['source_logits_dict'][action_type_name][graph_idx]
                source_probs = F.softmax(source_logits, dim=-1)
                p_source = source_probs[action.source_neuron]
                total_prob *= p_source
                
                # Target probability (if applicable and conditioned on source)
                if (action.target_neuron is not None and
                    'target_heads' in predictions and
                    action_type_name in predictions['target_heads'] and
                    'shared_features' in predictions and
                    'source_encoder' in predictions):
                    
                    # Encode source for conditioning
                    source_one_hot = F.one_hot(
                        torch.tensor([action.source_neuron], device=self.device), 
                        self.config.model.max_neurons
                    ).float()
                    source_features = predictions['source_encoder'](source_one_hot)
                    graph_features = predictions['shared_features'][graph_idx:graph_idx+1]
                    conditioned_features = torch.cat([graph_features, source_features], dim=-1)
                    
                    # Get conditional target logits
                    target_logits = predictions['target_heads'][action_type_name](conditioned_features)
                    target_probs = F.softmax(target_logits, dim=-1)
                    p_target = target_probs[0, action.target_neuron]
                    total_prob *= p_target
                
                # Activation probability (if applicable)
                if (action.activation is not None and
                    'activation_heads' in predictions and
                    action_type_name in predictions['activation_heads']):
                    
                    if action_type_name == 'modify_activation' and action.source_neuron is not None:
                        # Condition on source
                        source_one_hot = F.one_hot(
                            torch.tensor([action.source_neuron], device=self.device), 
                            self.config.model.max_neurons
                        ).float()
                        source_features = predictions['source_encoder'](source_one_hot)
                        graph_features = predictions['shared_features'][graph_idx:graph_idx+1]
                        conditioned_features = torch.cat([graph_features, source_features], dim=-1)
                        activation_logits = predictions['activation_heads'][action_type_name](conditioned_features)
                    else:
                        # Unconditional
                        activation_logits = predictions['activation_heads'][action_type_name](
                            predictions['shared_features'][graph_idx:graph_idx+1]
                        )
                    
                    activation_probs = F.softmax(activation_logits, dim=-1)
                    activation_idx = list(ActivationType).index(action.activation)
                    p_activation = activation_probs[0, activation_idx]
                    total_prob *= p_activation
            
            return total_prob.item() if isinstance(total_prob, torch.Tensor) else total_prob
            
        except Exception as e:
            print(f"    [Warning] Failed to compute action probability: {e}")
            return None

    def run_training(self):
        """Run complete training process using AlphaZero-style MCTS + policy network"""
        print("Starting architecture search training...")
        
        while self.episode < self.config.max_episodes:
            # Run training episode
            episode_metrics, final_architecture = self.run_training_episode()
            
            # Train on batch(s) if we have enough experiences
            train_interval = getattr(self.config, 'train_interval', 10)
            if (self.episode + 1) % train_interval == 0 or self.episode == 0:
                
                # We will sample batches from the replay buffer. The buffer.sample()
                # method is prioritized and probabilistic, so multiple sample() calls
                # can return overlapping indices. Instead of attempting to track
                # unique indices (expensive), estimate expected coverage using the
                # per-item sampling probabilities and decrement an expected-remaining
                # measure. This preserves prioritized sampling while estimating when
                # we've 'likely' seen most of the buffer.
                buffer_size = len(self.experience_buffer)
                print(f"  [Training] Starting with {buffer_size} total experiences in buffer")

                if buffer_size > 0:
                    # Get current priority-based sampling probabilities
                    priorities = self.experience_buffer.priorities[:buffer_size].astype(np.float64)
                    # Protect against zero-sum
                    if priorities.sum() <= 0:
                        probs = np.ones(buffer_size, dtype=np.float64) / float(buffer_size)
                    else:
                        probs = priorities ** getattr(self.experience_buffer, 'alpha', 1.0)
                        probs = probs / probs.sum()

                    # Compute per-item probability of being sampled in one batch draw of size b:
                    # s_i = 1 - (1 - p_i)^b (approximate draws with replacement)
                    batch_size = min(self.config.batch_size, buffer_size)
                    b = float(batch_size)
                    s = 1.0 - np.power(1.0 - probs, b)

                    # Bound iterations to avoid pathological long loops; default: 3x full coverage
                    max_iters = max(1, int(np.ceil(buffer_size / max(1, batch_size) * 3)))

                    # Determine analytically how many iterations (t) are needed so that
                    # the expected number of unseen items sum((1 - s_i)^t) <= tol.
                    tol = 1e-3
                    t_required = max_iters
                    # Iterate t in Python but compute vectorized powers for speed; typically t is small
                    for t in range(1, max_iters + 1):
                        remaining = np.sum(np.power(1.0 - s, t))
                        if remaining <= tol:
                            t_required = t
                            break

                    # Execute the required number of training batches (no per-item bookkeeping loop)
                    for _ in range(t_required):
                        train_metrics = self.train_on_batch(batch_size, sub_batch_size=self.config.search.sub_batch_size)
                        if train_metrics:
                            episode_metrics.update(train_metrics)

                    # After loop, clear buffer to keep behavior consistent with previous code
                    self.experience_buffer.clear()


            # Checkpoint periodically
            if (self.episode + 1) % self.config.checkpoint_interval == 0 or self.episode == 0:
                self._save_checkpoint(final_architecture=final_architecture)

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
        # Sanitize and extract commonly used metrics for readable logs
        def _sanitize(v):
            # Convert torch tensors / numpy scalars / numpy arrays into python types
            try:
                import torch as _torch
            except Exception:
                _torch = None

            # Torch tensor
            if _torch is not None and isinstance(v, _torch.Tensor):
                try:
                    if v.numel() == 1:
                        return float(v.item())
                    return v.detach().cpu().tolist()
                except Exception:
                    return None

            # Numpy types
            try:
                import numpy as _np
                if isinstance(v, (_np.floating, _np.integer)):
                    return float(v)
                if isinstance(v, _np.ndarray):
                    return v.tolist()
            except Exception:
                pass

            # Common container types
            if isinstance(v, (list, tuple)):
                return [_sanitize(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _sanitize(val) for k, val in v.items()}

            # Fallback: primitive
            try:
                if isinstance(v, (int, float, str, bool)):
                    return v
            except Exception:
                pass

            # Last resort
            try:
                return float(v)
            except Exception:
                return str(v)

        accuracy = float(_sanitize(metrics.get('final_accuracy', 0.0)))
        avg_reward = _sanitize(metrics.get('average_reward', 0.0))
        steps = int(_sanitize(metrics.get('steps', 0)))
        neurons = int(_sanitize(metrics.get('total_neurons', 0)))
        connections = int(_sanitize(metrics.get('total_connections', 0)))
        experiences = int(_sanitize(metrics.get('experiences', 0)))
        is_best = bool(_sanitize(metrics.get('is_best', False)))

        # Console output: richer summary
        print(
            f"Episode {self.episode}: acc={accuracy:.4f}, avg_reward={avg_reward:.4f}, "
            f"steps={steps}, experiences={experiences}, neurons={neurons}, "
            f"conns={connections}, best={self.best_reward:.4f}, is_best={is_best}"
        )

        # Prepare structured log dict with selected fields (sanitize all values)
        log_dict = {
            'timestamp': time.time(),
            'episode': int(self.episode),
            'final_accuracy': accuracy,
            'average_reward': _sanitize(avg_reward),
            'steps': steps,
            'experiences': experiences,
            'total_neurons': neurons,
            'total_connections': connections,
            'best_reward': float(_sanitize(self.best_reward)),
            'is_best': is_best
        }

        # Include training metrics if present (losses, num_graphs, etc.)
        for k in ('total_loss', 'value_loss', 'mcts_policy_loss', 'num_graphs'):
            if k in metrics:
                log_dict[k] = _sanitize(metrics[k])

        # Structured file logging (JSONL) + logger info line
        try:
            # Log a compact one-line message to configured logger if available
            try:
                self.logger.info(f"Episode {self.episode}: final_acc={accuracy:.4f}, steps={steps}, experiences={experiences}")
            except Exception:
                # If logger isn't set up, ignore and proceed to JSONL
                pass

            # Force flush handlers to ensure writes (if logger exists)
            try:
                for handler in getattr(self, 'logger', type('X', (), {'handlers': []})) and getattr(self.logger, 'handlers', []):
                    try:
                        handler.flush()
                    except Exception:
                        pass
            except Exception:
                pass

            # Append sanitized JSONL metrics to metrics_path
            with open(self.metrics_path, 'a', encoding='utf-8') as mf:
                mf.write(json.dumps(log_dict) + "\n")
                mf.flush()

        except Exception as e:
            # Fall back to direct file write if logger/metrics file fails
            print(f"Logger/metrics write failed: {e}, attempting direct file write...")
            try:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{time.time()} INFO Episode {self.episode} final_acc={accuracy:.4f} "
                           f"steps={steps} experiences={experiences} neurons={neurons} conns={connections} best={self.best_reward:.4f}\n")
                    f.flush()

                # Also attempt metrics JSONL write as last resort
                with open(self.metrics_path, 'a', encoding='utf-8') as mf:
                    mf.write(json.dumps(log_dict) + "\n")
                    mf.flush()
            except Exception as e2:
                print(f"Direct file write also failed: {e2}")
                traceback.print_exc()
    
    def _save_checkpoint(self, final_architecture=None):
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
            'best_reward': self.best_reward,
            'training_history': self.training_history,
            'config': self.config,
            'experience_buffer': self.experience_buffer.state_dict(),
            'final_architecture': final_architecture
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
        self.best_reward = checkpoint['best_reward']
        self.training_history = checkpoint['training_history']
        self.episode = checkpoint['episode'] + 1  # Start next episode
        self.experience_buffer.load_state_dict(checkpoint['experience_buffer'])

        # Restore final architecture if present
        self.final_architecture = None
        if 'final_architecture' in checkpoint and checkpoint['final_architecture'] is not None:
            self.final_architecture = NeuralArchitecture.from_serializable_dict(checkpoint['final_architecture'])

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
            # Only draw 10 diagrams per episode to limit I/O
            if (step + 1) % max(self.config.search.max_steps_per_episode // 10, 1) != 0 and step != 0:
                return
            # Create directory
            os.makedirs(f'architecture_diagrams/ep{self.episode:03d}', exist_ok=True)

            # Constants for layout
            CANVAS_WIDTH = 1000
            CANVAS_HEIGHT = 800
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
            img.save(diagram_file, quality=70, optimize=True)

        except ImportError:
            print("      PIL not available, skipping diagram")
        except Exception as e:
            print(f"Failed to draw architecture diagram: {e}")
            traceback.print_exc()
