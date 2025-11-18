from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """Configuration for the Graph Transformer and Policy-Value Network.

    Attributes:
        node_feature_dim (int): The dimensionality of the node features.
        hidden_dim (int): The dimensionality of the hidden layers.
        num_heads (int): The number of attention heads in the transformer.
        num_layers (int): The number of layers in the transformer.
        dropout (float): The dropout rate.
        use_edge_features (bool): Whether to use edge features.
        max_neurons (int): The maximum number of neurons.
        num_actions (int): The number of possible actions.
        num_activations (int): The number of possible activation functions.
    """
    # Graph Transformer
    node_feature_dim: int = 9  # Based on neuron feature vector (3 type + 4 activation + position + bias)
    hidden_dim: int = 2048  # Increased for more capacity
    num_heads: int = 16
    num_layers: int = 6
    dropout: float = 0.2  # Increased dropout for better regularization
    use_edge_features: bool = True
    
    # Policy-Value Network
    max_neurons: int = 1000  # Accommodate MNIST inputs
    num_actions: int = 5
    num_activations: int = 4

@dataclass
class MCTSConfig:
    """Configuration for the AlphaZero-style Neural MCTS.

    Attributes:
        num_simulations (int): The number of simulations to run per move.
        exploration_weight (float): The exploration weight in the PUCT formula.
        dirichlet_alpha (float): The alpha parameter for the Dirichlet noise.
        dirichlet_epsilon (float): The epsilon parameter for the Dirichlet noise.
        max_children (int): The maximum number of children to expand per node.
        temperature (float): The temperature for the final move selection.
        temperature_decay (float): The decay rate for the temperature.
    """
    # Search parameters
    num_simulations: int = 500  # Increased for better search quality
    exploration_weight: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Node expansion
    max_children: int = 30
    temperature: float = 1.0
    temperature_decay: float = 0.99

@dataclass
class ArchitectureSearchConfig:
    """Configuration for the architecture search process.

    Attributes:
        max_neurons (int): The maximum number of neurons allowed in an architecture.
        max_connections (int): The maximum number of connections allowed.
        max_steps_per_episode (int): The maximum number of steps per episode.
        min_neurons (int): The minimum number of hidden neurons.
        min_connections (int): The minimum number of connections.
        quick_train_epochs (int): The number of epochs for quick training.
        final_train_epochs (int): The number of epochs for final training.
        evaluation_batch_size (int): The batch size for evaluation.
        sub_batch_size (int): The sub-batch size for training.
        stability_threshold (float): The stability threshold for the
            evolutionary cycle.
        target_accuracy (float): The target accuracy to stop the search.
        reward_loss_weight (float): The weight for the loss in the reward.
        reward_complexity_weight (float): The weight for the complexity penalty
            in the reward.
        reward_accuracy_weight (float): The weight for the accuracy in the reward.
        priority_surprise_weight (float): The weight for the surprise factor
            in the priority calculation.
        action_exploration_boost (float): The boost factor for underrepresented
            actions.
        connection_candidate_multiplier (int): The multiplier for connection
            candidates.
        allowed_actions (List): A list of allowed action types.
    """
    # Search constraints
    max_neurons: int = 1000
    max_connections: int = 10000
    max_steps_per_episode: int = 1000  # Increased to allow more complex architectures
    min_neurons: int = 25  # Minimum number of hidden neurons to prevent oversimplification
    min_connections: int = 250  # Minimum number of connections to prevent oversimplification

    # Evaluation
    quick_train_epochs: int = 1  # Reduced for faster evaluations
    final_train_epochs: int = 3  # Increased for more thorough final training
    evaluation_batch_size: int = 64  # Increased for faster evaluation with larger batches

    # Sub-batch size for training
    sub_batch_size: int = 8

    # Stability threshold for evolutionary cycle
    stability_threshold: float = 0.005  # Reduced to allow more exploration

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
        """Initializes the allowed actions if not provided."""
        if self.allowed_actions is None:
            from blueprint_modules.action import ActionType
            self.allowed_actions = [
                ActionType.ADD_NEURON, ActionType.REMOVE_NEURON,
                ActionType.MODIFY_ACTIVATION, ActionType.ADD_CONNECTION,
                ActionType.REMOVE_CONNECTION
            ]

@dataclass
class OverallConfig:
    """Complete training configuration.

    Attributes:
        model (ModelConfig): The model configuration.
        mcts (MCTSConfig): The MCTS configuration.
        search (ArchitectureSearchConfig): The architecture search configuration.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        max_grad_norm (float): The maximum gradient norm for clipping.
        max_episodes (int): The maximum number of episodes to run.
        device (str): The device to run on.
        gpu_memory_fraction (float): The fraction of GPU memory to use.
        enable_memory_monitoring (bool): Whether to enable memory monitoring.
        memory_check_threshold_mb (float): The memory check threshold in MB.
        seed (int): The random seed.
        log_dir (str): The directory for logs.
        checkpoint_dir (str): The directory for checkpoints.
        eval_interval (int): The evaluation interval.
        checkpoint_interval (int): The checkpoint interval.
        diagram_save_interval (int): The diagram save interval.
        log_interval (int): The log interval.
        train_interval (int): The training interval.
        use_mixed_precision (bool): Whether to use mixed precision.
        gradient_accumulation_steps (int): The number of gradient
            accumulation steps.
        enable_tf32 (bool): Whether to enable TF32.
        early_stopping_patience (int): The patience for early stopping.
        early_stopping_min_delta (float): The minimum delta for early stopping.
    """
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
    max_episodes: int = 500
    
    # System
    device: str = "cuda:2"  # Options: "auto", "cpu", or "cuda:X"
    gpu_memory_fraction: float = 0.9
    enable_memory_monitoring: bool = True
    memory_check_threshold_mb: float = 5000
    seed: int = 42
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"
    
    # Monitoring
    eval_interval: int = 1
    checkpoint_interval: int = 1
    diagram_save_interval: int = 1
    log_interval: int = 1
    train_interval: int = 10  # Train every N episodes

    # GPU optimizations
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    enable_tf32: bool = True

    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.001
