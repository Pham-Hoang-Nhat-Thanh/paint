from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class ModelConfig:
    """Configuration for the Graph Transformer and Policy-Value Network"""
    # Graph Transformer
    node_feature_dim: int = 9  # Based on neuron feature vector (3 type + 4 activation + position + bias)
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    use_edge_features: bool = True
    
    # Policy-Value Network
    max_neurons: int = 1000  # Accommodate MNIST inputs
    num_actions: int = 5
    num_activations: int = 4
    
    # Shared Backbone
    backbone_hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.backbone_hidden_dims is None:
            self.backbone_hidden_dims = [256, 128]

@dataclass
class MCTSConfig:
    """Configuration for Neural MCTS"""
    # Search parameters
    num_simulations: int = 100
    exploration_weight: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Node expansion
    max_children: int = 50
    temperature: float = 1.0
    temperature_decay: float = 0.99
    
    # Rollout settings
    rollout_depth: int = 10
    use_value_network: bool = True

@dataclass
class TrainingStageConfig:
    """Base configuration for each training stage"""
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Loss weights
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    entropy_weight: float = 0.01
    
    # Update frequency
    train_interval: int = 1
    target_update_interval: int = 100

@dataclass
class SupervisedConfig(TrainingStageConfig):
    """Configuration for supervised pretraining stage"""
    num_episodes: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    # Expert data
    expert_data_path: str = "data/expert_architectures.pkl"
    min_expert_accuracy: float = 0.85
    
    # Curriculum
    warmup_epochs: int = 10
    use_teacher_forcing: bool = True

@dataclass
class MixedConfig(TrainingStageConfig):
    """Configuration for mixed exploration stage"""
    num_episodes: int = 100
    learning_rate: float = 5e-4
    
    # Policy vs MCTS mixing
    initial_policy_ratio: float = 0.0
    final_policy_ratio: float = 0.7
    mixing_schedule: str = "linear"
    
    # Exploration
    initial_temperature: float = 1.0
    final_temperature: float = 0.5
    
    # Experience replay
    replay_buffer_size: int = 5000
    initial_buffer_size: int = 1000

@dataclass
class SelfPlayConfig(TrainingStageConfig):
    """Configuration for self-play RL stage"""
    num_episodes: int = 300
    learning_rate: float = 1e-4
    
    # Policy guidance
    policy_ratio: float = 0.9
    temperature: float = 0.1
    
    # Experience replay
    replay_buffer_size: int = 10000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_epsilon: float = 1e-6
    
    # Training stability
    target_update_interval: int = 50
    gradient_clip: float = 1.0
    
    # Learning rate scheduling
    lr_schedule: str = "cosine"
    warmup_episodes: int = 10

@dataclass
class ArchitectureSearchConfig:
    """Configuration for architecture search process"""
    # Search constraints
    max_neurons: int = 1000
    max_connections: int = 10000
    max_steps_per_episode: int = 50
    
    # Evaluation
    quick_train_epochs: int = 3
    final_train_epochs: int = 10
    evaluation_batch_size: int = 64
    
    # Termination conditions
    target_accuracy: float = 0.97
    patience: int = 20
    
    # Action space
    allowed_actions: List = None
    
    def __post_init__(self):
        if self.allowed_actions is None:
            from blueprint_modules.action import ActionType
            self.allowed_actions = [
                ActionType.ADD_NEURON, ActionType.REMOVE_NEURON,
                ActionType.MODIFY_ACTIVATION, ActionType.ADD_CONNECTION,
                ActionType.REMOVE_CONNECTION
            ]

@dataclass
class OverallConfig:
    """Complete training configuration"""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # MCTS
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

    # Training stages
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    mixed: MixedConfig = field(default_factory=MixedConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    # Architecture search
    search: ArchitectureSearchConfig = field(default_factory=ArchitectureSearchConfig)
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"
    
    # Monitoring
    eval_interval: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 5
    # Global training interval (how often to run training steps across episodes)
    train_interval: int = 1