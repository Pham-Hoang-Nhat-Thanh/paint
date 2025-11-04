import math
import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class CurriculumState:
    """State of the training curriculum for checkpointing"""
    episode_count: int = 0
    stage: str = "supervised"
    policy_ratio: float = 0.0
    temperature: float = 1.0
    learning_rate: float = 1e-3
    best_accuracy: float = 0.0

class TrainingCurriculum:
    """Manages the training curriculum from supervised to RL"""

    def __init__(self, supervised_episodes: int, mixed_episodes: int, self_play_episodes: int):
        self.supervised_episodes = supervised_episodes
        self.mixed_episodes = supervised_episodes + mixed_episodes
        self.self_play_episodes = self.mixed_episodes + self_play_episodes
        self.total_episodes = self.self_play_episodes
        self.episode_count = 0

        # Current state
        self.state = CurriculumState()
        
    def get_stage(self) -> str:
        """Get current training stage"""
        if self.episode_count < self.supervised_episodes:
            return "supervised"
        elif self.episode_count < self.mixed_episodes:
            return "mixed"
        else:
            return "self_play"
    
    def get_policy_mix_ratio(self) -> float:
        """Get ratio of policy vs MCTS actions"""
        stage = self.get_stage()
        
        if stage == "supervised":
            return 0.0  # Pure MCTS (learning from expert)
        elif stage == "mixed":
            # Linear interpolation from 0 to 0.7
            progress = (self.episode_count - self.supervised_episodes) / \
                      (self.mixed_episodes - self.supervised_episodes)
            return 0.7 * progress
        else:  # self_play
            return 0.9  # Mostly policy
    
    def get_learning_rate(self, base_lr: float = 1e-3) -> float:
        """Get learning rate for current stage"""
        stage = self.get_stage()
        
        if stage == "supervised":
            return base_lr
        elif stage == "mixed":
            return base_lr * 0.5
        else:  # self_play
            # Cosine decay
            progress = (self.episode_count - self.mixed_episodes) / \
                      (self.self_play_episodes - self.mixed_episodes)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_temperature(self) -> float:
        """Get temperature for action selection"""
        stage = self.get_stage()
        
        if stage == "supervised":
            return 1.0  # High exploration
        elif stage == "mixed":
            return 0.5  # Medium exploration
        else:  # self_play
            return 0.1  # Low exploration
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get all training parameters for current stage"""
        return {
            "stage": self.get_stage(),
            "policy_mix_ratio": self.get_policy_mix_ratio(),
            "temperature": self.get_temperature(),
            "learning_rate": self.get_learning_rate(),
            "episode": self.episode_count
        }
    
    def step(self):
        """Advance curriculum by one episode"""
        self.episode_count += 1
        self.state.episode = self.episode_count
        self.state.stage = self.get_stage()
        self.state.policy_ratio = self.get_policy_mix_ratio()
        self.state.temperature = self.get_temperature()
        self.state.learning_rate = self.get_learning_rate()
    
    def get_state(self) -> CurriculumState:
        """Get current curriculum state for checkpointing"""
        return self.state
    
    def set_state(self, state: CurriculumState):
        """Set curriculum state (for loading checkpoints)"""
        self.episode_count = state.episode_count
        self.state = state
    
    def should_evaluate(self, eval_interval: int = 10) -> bool:
        """Check if it's time to evaluate"""
        return self.episode_count % eval_interval == 0
    
    def should_checkpoint(self, checkpoint_interval: int = 50) -> bool:
        """Check if it's time to save a checkpoint"""
        return self.episode_count % checkpoint_interval == 0
    
    def is_complete(self) -> bool:
        """Check if curriculum is complete"""
        return self.episode_count >= self.total_episodes

class PolicyValueLoss(nn.Module):
    """Combined loss for policy and value networks"""

    def __init__(self, value_weight: float = 1.0, entropy_weight: float = 0.01):
        super().__init__()
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        """
        predictions: output from policy_value_net
        targets: {
            'action_type': target action type (single action label for supervised),
            'source_neuron': target source neuron,
            'target_neuron': target target neuron,
            'activation': target activation,
            'value': target value,
            'mcts_policy': OPTIONAL - visit distribution from MCTS (AlphaZero-style),
            'mcts_actions': OPTIONAL - actions explored by MCTS
        }
        """
        total_loss = 0.0

        # Check if we have MCTS policy distribution (AlphaZero-style training)
        use_mcts_policy = 'mcts_policy' in targets and 'mcts_actions' in targets
        
        if use_mcts_policy:
            # AlphaZero-style: Train policy to match MCTS visit distribution
            # This is MORE informative than single action labels because it shows
            # the relative quality of ALL explored actions
            mcts_policy = targets['mcts_policy']  # List of probabilities
            mcts_actions = targets['mcts_actions']  # List of Action objects
            
            # Convert MCTS visit distribution to target logits for each policy head
            # We'll use KL divergence to match the distribution
            action_type_targets = torch.zeros_like(predictions['action_type'])
            
            for prob, action in zip(mcts_policy, mcts_actions):
                # Accumulate probability mass for each action component
                action_type_targets[0, action.action_type.value] += prob
            
            # KL divergence loss (or cross-entropy with soft targets)
            action_probs = F.log_softmax(predictions['action_type'], dim=-1)
            action_loss = -torch.sum(action_type_targets * action_probs)
            total_loss += action_loss
            
            # Note: For hierarchical policy (source/target/activation), we still use
            # hard labels from the selected action since MCTS doesn't provide
            # distributions over these sub-actions
            if targets.get('source_neuron') is not None:
                source_loss = F.cross_entropy(
                    predictions['source_logits'],
                    targets['source_neuron']
                )
                total_loss += source_loss

            if targets.get('target_neuron') is not None:
                target_loss = F.cross_entropy(
                    predictions['target_logits'],
                    targets['target_neuron']
                )
                total_loss += target_loss

            if targets.get('activation') is not None:
                activation_loss = F.cross_entropy(
                    predictions['activation_logits'],
                    targets['activation']
                )
                total_loss += activation_loss
        else:
            # Supervised learning: Train on single action labels
            action_loss = F.cross_entropy(
                predictions['action_type'],
                targets['action_type']
            )
            total_loss += action_loss

            # Source neuron loss (only if applicable)
            if targets.get('source_neuron') is not None:
                source_loss = F.cross_entropy(
                    predictions['source_logits'],
                    targets['source_neuron']
                )
                total_loss += source_loss

            # Target neuron loss (only if applicable)
            if targets.get('target_neuron') is not None:
                target_loss = F.cross_entropy(
                    predictions['target_logits'],
                    targets['target_neuron']
                )
                total_loss += target_loss

            # Activation loss (only if applicable)
            if targets.get('activation') is not None:
                activation_loss = F.cross_entropy(
                    predictions['activation_logits'],
                    targets['activation']
                )
                total_loss += activation_loss

        # Value loss (always used)
        value_loss = F.mse_loss(
            predictions['value'].squeeze(),
            targets['value'].squeeze()
        )
        total_loss += self.value_weight * value_loss

        # Entropy regularization
        entropy = self._compute_entropy(predictions)
        total_loss -= self.entropy_weight * entropy

        return total_loss

    def _compute_entropy(self, predictions: Dict) -> torch.Tensor:
        """Compute entropy of policy distribution for regularization"""
        entropy = 0.0

        # Action type entropy
        action_probs = F.softmax(predictions['action_type'], dim=-1)
        action_entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        entropy += action_entropy

        # Add entropy for other heads if needed

        return entropy
