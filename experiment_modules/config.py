from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """Configuration for the Graph Transformer and Policy-Value Network"""
    # Graph Transformer
    node_feature_dim: int = 9  # Based on neuron feature vector (3 type + 4 activation + position + bias)
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 5
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
            self.backbone_hidden_dims = [128, 64]

@dataclass
class MCTSConfig:
    """Configuration for AlphaZero-style Neural MCTS (no rollouts)"""
    # Search parameters
    num_simulations: int = 1000
    exploration_weight: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Node expansion
    max_children: int = 50
    temperature: float = 1.0
    temperature_decay: float = 0.99
    
    # Loss weighting for MCTS KL vs supervised CE
    mcts_policy_weight: float = 1.0  # Weight for KL divergence to MCTS visit distribution
    component_ce_weight: float = 0.25  # Initial weight for supervised cross-entropy on policy components (source/target/activation)
    ce_anneal_episodes: int = 500  # Number of episodes over which to anneal CE weight from component_ce_weight to 0.05

@dataclass
class ArchitectureSearchConfig:
    """Configuration for architecture search process"""
    # Search constraints
    max_neurons: int = 1000
    max_connections: int = 10000
    max_steps_per_episode: int = 500  # Increased to allow more complex architectures

    # Evaluation
    quick_train_epochs: int = 1  # Reduced for faster evaluations
    final_train_epochs: int = 3  # Increased for more thorough final training
    evaluation_batch_size: int = 64  # Increased for faster evaluation with larger batches

    # Sub-batch size for training
    sub_batch_size: int = 8

    # Termination conditions
    target_accuracy: float = 0.97

    # Reward configuration
    reward_loss_weight: float = 0.1  # Weight for loss in composite reward
    reward_complexity_weight: float = 0.05  # Weight for complexity penalty in reward
    reward_accuracy_weight: float = 1.0  # Weight for accuracy in composite reward
    priority_surprise_weight = 0.8  # Default; increase for more emphasis on surprising/informative experiences

    # Action space balancing
    action_exploration_boost: float = 0.5  # Boost factor for underrepresented actions
    connection_candidate_multiplier: int = 3  # Multiplier for connection candidates (num_neurons * multiplier)

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
    """Complete training configuration for AlphaZero-style MCTS + policy network"""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # MCTS
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

    # Architecture search
    search: ArchitectureSearchConfig = field(default_factory=ArchitectureSearchConfig)
    
    # Training parameters 
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    max_episodes: int = 300
    
    # System
    device: str = "cuda:1"  # Options: "auto", "cpu", or "cuda:X"
    gpu_memory_fraction: float = 0.9
    enable_memory_monitoring: bool = True
    memory_check_threshold_mb: float = 5000
    seed: int = 42
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"
    
    # Monitoring
    eval_interval: int = 1
    checkpoint_interval: int = 1
    diagram_save_interval: int = 50
    log_interval: int = 1
    train_interval: int = 5  # Train every N episodes

    # GPU optimizations
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    enable_tf32: bool = True

    # Early stopping
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.001
