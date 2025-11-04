# PAINT — Playing Around In Network Topologies

PAINT is a research-oriented Neural Architecture Search (NAS) framework that treats architecture design as a planning problem. It combines Monte Carlo Tree Search (MCTS) with learned graph representations (Graph Transformers) to discover efficient, compact, and novel neuron-level topologies.

## Overview

PAINT replaces traditional evolutionary search with a neural-guided planning approach. Architectures are represented as graphs (neurons = nodes, connections = edges) and search is performed using a policy-value network to guide MCTS. The framework is designed for experimentation and research: fast topology changes, quick evaluation of candidate networks, and an extensible curriculum for training the policy/value models.

## Key Features

- **Neural-guided MCTS**: AlphaZero-style MCTS, guided by a learned policy-value network, replaces classical evolutionary operators.
- **Graph Transformer Encoder**: Encodes neural architectures at the neuron level, supporting edge features and positional encodings.
- **Factorized Action Space**: Supports fine-grained actions (add/remove neuron, add/remove connection, modify activation) with masking and exploration incentives.
- **Transferable Priors**: Learns architectural priors that generalize across runs and tasks.
- **Efficient Graph Representation**: Fast, vectorized graph encoding and optimized PyTorch modules for architecture evaluation.

## Core Components

1. **Graph-based Architecture Representation**
	- Neurons are nodes and connections are edges (see `blueprint_modules/network.py`).
	- Architectures are dynamic and can be mutated by actions (add/remove neuron, modify activation, add/remove connection).
	- Supports serialization, signature computation, and efficient connectivity queries.

2. **Neural-guided MCTS**
	- MCTS base in `blueprint_modules/mcts.py`, extended by `architect_modules/guided_mcts.py` for neural guidance.
	- Policy-value network (`architect_modules/policy_value_net.py`) uses a `GraphTransformer` encoder (`architect_modules/graph_transformer.py`).
	- Action selection uses masking, Dirichlet noise, and visit-based selection for robust exploration.

3. **Training and Experimentation**
	- Training orchestration in `experiment_modules/architecture_trainer.py` and `main.py`.
	- Supports curriculum: supervised pretraining, mixed exploration, and self-play reinforcement learning.
	- MNIST is used as a sandbox for rapid prototyping and benchmarking.

## Repository Layout

- `blueprint_modules/` — Core data structures and algorithms (graph representation, MCTS base, action space)
- `architect_modules/` — Model implementations (Graph Transformer, policy/value net) and neural-guided MCTS utilities
- `experiment_modules/` — Experiments, configuration, and training orchestration (`main.py`, `architecture_trainer.py`, `config.py`)
- `data/` — (MNIST raw files, not tracked by git)
- `logs/` — Training logs and metrics
- `tests/` — Unit tests and examples

## Quick Start (Linux/Mac/Windows)

1. **Create & activate a virtual environment (recommended):**

	```bash
	python -m venv .venv
	source .venv/bin/activate  # or .venv\Scripts\activate on Windows
	```

2. **Install dependencies:**

	```bash
	pip install -r requirements.txt
	# Or, for minimal setup:
	pip install torch torchvision numpy
	```

3. **Run the entry script:**

	```bash
	python experiment_modules/main.py
	```

	- The code will auto-detect CUDA and use GPU if available.
	- Checkpoints and logs are saved in `paint/checkpoints/` and `paint/logs/`.

## Configuration

All main hyperparameters and defaults are in `experiment_modules/config.py` (model sizes, MCTS settings, training stages, and search constraints). Typical knobs you may tune:

- `ModelConfig.node_feature_dim`, `hidden_dim`, `num_heads`
- `MCTSConfig.num_simulations`, `exploration_weight`
- `ArchitectureSearchConfig.max_neurons`, `target_accuracy`

## Example Usage (Programmatic)

```python
from experiment_modules.architecture_trainer import ArchitectureTrainer
from experiment_modules.config import OverallConfig

config = OverallConfig()
trainer = ArchitectureTrainer(config)
history = trainer.run_training()
```

Adjust `config` before instantiating the trainer to change search limits, model sizes, or training schedule.

## Developer Notes

- **Shape mismatches** (e.g., `mat1 and mat2 shapes cannot be multiplied`) usually mean `ModelConfig.node_feature_dim` does not match the feature length produced in `blueprint_modules/network.py` (`Neuron.to_feature_vector()`).
- **Policy/mask size mismatches** often relate to `max_neurons` vs actual indexed neuron IDs. The `ActionManager` in `architect_modules/policy_value_net.py` builds masks; inspect it for related errors.
- **JIT and Mixed Precision**: The code supports TorchScript JIT and mixed precision for speed on modern GPUs.
- **Checkpoints**: Training auto-resumes from the latest checkpoint in `checkpoints/` if available.

## Results & Expected Behavior

PAINT is a research prototype. In experiments, it aims to:

- Discover compact architectures that reach competitive accuracy on small benchmarks (MNIST used as a sandbox)
- Learn transferable priors that speed up search in subsequent runs
- Produce novel connectivity patterns driven by planning rather than mutation operators

## Contributing

- Fork, add a feature branch, and open a PR with tests and a short description.
- Keep experiments small for CI and debugging; add larger reproductions in separate experiment scripts.

## Citation

If you use PAINT in research, please cite it:

```bibtex
@software{paint2024,
  title = {PAINT: Playing Around In Network Topologies},
  author = {Pham-Hoang-Nhat-Thanh},
  year = {2024},
  url = {https://github.com/Pham-Hoang-Nhat-Thanh/paint}
}
```

## License

This project does not include a LICENSE file in the repository root. If you plan to publish the code, add a `LICENSE` (MIT, Apache-2.0, etc.) and mention it here.
