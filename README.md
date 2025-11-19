# PAINT — Playing Around In Network Topologies

PAINT is a research-oriented Neural Architecture Search (NAS) framework that treats architecture design as a planning problem. It combines Monte Carlo Tree Search (MCTS) with learned graph representations (Graph Transformers) to discover efficient, compact, and novel neuron-level topologies.

## Overview

PAINT is a framework for Neural Architecture Search (NAS) that uses a novel approach to discover new neural network architectures. Instead of relying on traditional evolutionary algorithms, PAINT treats architecture design as a planning problem. It uses a Monte Carlo Tree Search (MCTS) algorithm guided by a policy-value network to explore the vast search space of possible architectures.

The key idea behind PAINT is to represent neural networks as graphs, where neurons are nodes and connections are edges. The MCTS algorithm then iteratively builds and modifies these graphs, guided by a learned policy that suggests promising actions to take. The framework is designed to be highly extensible and customizable, making it a powerful tool for research and experimentation in the field of NAS.

## How it Works

The core of PAINT is the interplay between the MCTS algorithm and the policy-value network. The MCTS algorithm is responsible for exploring the search space of possible architectures, while the policy-value network provides the guidance for this exploration.

The search process starts with a simple initial architecture. The MCTS algorithm then performs a series of simulations, where each simulation consists of a sequence of actions that modify the architecture. The actions are chosen based on a combination of the policy network's predictions and the MCTS algorithm's own exploration strategy.

After each simulation, the resulting architecture is evaluated, and the results are used to update the MCTS tree and the policy-value network. This process is repeated for a fixed number of iterations, and the best architecture found during the search is returned.

## Key Features

- **Neural-guided MCTS**: AlphaZero-style MCTS, guided by a learned policy-value network, replaces classical evolutionary operators.
- **Graph Transformer Encoder**: Encodes neural architectures at the neuron level, supporting edge features and positional encodings.
- **Factorized Action Space**: Supports fine-grained actions (add/remove neuron, add/remove connection, modify activation) with masking and exploration incentives.
- **Transferable Priors**: Learns architectural priors that generalize across runs and tasks.
- **Efficient Graph Representation**: Fast, vectorized graph encoding and optimized PyTorch modules for architecture evaluation.

## Core Components

1.  **Graph-based Architecture Representation**:
    -   Neurons are represented as nodes and connections as edges in a graph structure. This is managed by the `NeuralArchitecture` class in `blueprint_modules/network.py`.
    -   Architectures are dynamic and can be modified by a set of discrete actions, such as adding or removing neurons and connections.

2.  **Neural-guided MCTS**:
    -   The MCTS algorithm is implemented in `blueprint_modules/mcts.py` and extended for neural guidance in `architect_modules/guided_mcts.py`.
    -   The policy-value network, defined in `architect_modules/policy_value_net.py`, uses a `GraphTransformer` encoder to process the graph representation of the architecture and output a policy and a value.

3.  **Training and Experimentation**:
    -   The training process is orchestrated by the `ArchitectureTrainer` class in `experiment_modules/architecture_trainer.py`.
    -   The main entry point for running experiments is `experiment_modules/main.py`.
    -   The framework uses MNIST as a sandbox for rapid prototyping and benchmarking.

## Project Structure

-   `architect_modules/`: Contains the implementations of the Graph Transformer, policy-value network, and other neural-guided MCTS utilities.
-   `blueprint_modules/`: Defines the core data structures and algorithms, such as the graph-based representation of neural networks, the MCTS algorithm, and the action space.
-   `experiment_modules/`: Includes the main scripts for running experiments, as well as configuration files and training orchestration.
-   `data/`: This directory is not tracked by git and is used to store the MNIST dataset.
-   `logs/`: This directory is used to store training logs and metrics.

## Quick Start

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main training script:**
    ```bash
    python experiment_modules/main.py
    ```

## Configuration

All the main hyperparameters and settings for the framework are located in `experiment_modules/config.py`. This includes configurations for the model, MCTS, and the architecture search process.

-   `ModelConfig`: Defines the parameters for the Graph Transformer and the policy-value network.
-   `MCTSConfig`: Contains the settings for the MCTS algorithm, such as the number of simulations and the exploration weight.
-   `ArchitectureSearchConfig`: Specifies the constraints for the architecture search, such as the maximum number of neurons and connections.
-   `OverallConfig`: A top-level configuration that aggregates all the other configurations.

## Example Usage

To run the architecture search, you can use the `main.py` script in the `experiment_modules` directory. This script will load the configuration, create an `ArchitectureTrainer` instance, and start the training process.

You can also use the framework programmatically by importing the `ArchitectureTrainer` and `OverallConfig` classes:

```python
from experiment_modules.architecture_trainer import ArchitectureTrainer
from experiment_modules.config import OverallConfig

config = OverallConfig()
trainer = ArchitectureTrainer(config)
history = trainer.run_training()
```

## Developer Notes

-   **Shape Mismatches**: If you encounter shape mismatches, it's likely due to a discrepancy between the `node_feature_dim` in `ModelConfig` and the feature vector length produced by `Neuron.to_feature_vector()` in `blueprint_modules/network.py`.
-   **Policy/Mask Size Mismatches**: These errors are often related to the `max_neurons` setting and the actual number of indexed neuron IDs. The `ActionManager` in `architect_modules/policy_value_net.py` is a good place to start debugging these issues.
-   **Prioritized Experience Replay**: The training process uses prioritized experience replay to improve sample efficiency.
-   **Checkpoints**: The training script automatically resumes from the latest checkpoint in the `checkpoints/` directory if one is available.

## Contributing

Contributions are welcome! Please fork the repository, create a new feature branch, and open a pull request with a clear description of your changes.

## License

This project is not licensed.
