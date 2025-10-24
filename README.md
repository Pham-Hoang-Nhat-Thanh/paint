# PAINT — Playing Around In Network Topologies

PAINT is a research-oriented Neural Architecture Search (NAS) framework that treats architecture design as a planning problem. It combines Monte Carlo Tree Search (MCTS) with learned graph representations (Graph Transformers) to discover efficient neuron-level topologies.

## Overview
PAINT replaces traditional evolutionary search with a neural-guided planning approach. Architectures are represented as graphs (neurons = nodes, connections = edges) and search is performed using a policy-value network to guide MCTS. The framework is designed for experimentation and research: fast topology changes, quick evaluation of candidate networks, and an extensible curriculum for training the policy/value models.

## Key innovation

- Use MCTS (AlphaZero-style) guided by a learned policy-value network instead of classical evolutionary operators.
- Use a Graph Transformer encoder to reason about neural architectures at the neuron level.
- Learn transferable architectural priors that help guide search across different runs and tasks.
- Prioritize neuron-level efficiency (compact topologies) in addition to performance.

## Core components

1. Graph-based architecture representation
	- Neurons are nodes and connections are edges (see `blueprint_modules/network.py`).
	- Architectures are dynamic and can be mutated by actions (add/remove neuron, modify activation, add/remove connection).

2. Neural-guided MCTS
	- MCTS implementation lives in `blueprint_modules/mcts.py` and is extended by `architect_modules/guided_mcts.py` to use network priors.
	- The policy-value network (see `architect_modules/policy_value_net.py`) uses a `GraphTransformer` encoder (`architect_modules/graph_transformer.py`).

3. Three-stage training curriculum
	- Supervised pretraining from expert constructions
	- Mixed exploration that balances policy and MCTS
	- Self-play reinforcement learning using the trained policy/value for rollouts

## Repository layout

Top-level (relevant files/folders):

- `blueprint_modules/` — core data structures and algorithms (graph representation, MCTS base, action space, quick trainer)
- `architect_modules/` — model implementations (Graph Transformer, policy/value net) and neural-guided MCTS utilities
- `experiment_modules/` — experiments, configuration and training orchestration (`main.py`, `architecture_trainer.py`, `config.py`)
- `data/` — (MNIST raw files kept out of the repo by `.gitignore`)
- `docs/` — design notes and phase reports
- `tests/` — unit tests and examples (excluded from repo via `.gitignore` by default in this workspace)

## Quick start (Windows / PowerShell)

1. Create & activate a virtualenv (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (PyTorch requires a compatible wheel for your CUDA / CPU setup):

```powershell
pip install torch torchvision numpy
```

3. Run the entry script (this project uses `experiment_modules/main.py`):

```powershell
python experiment_modules\main.py
```

Notes:
- The code will detect CUDA and prefer `cuda` when available; otherwise it uses `cpu`.
- There is no pinned `requirements.txt` in this repository — for reproducible experiments consider creating one from your environment (`pip freeze > requirements.txt`).

## Configuration

All main hyperparameters and defaults live in `experiment_modules/config.py` (model sizes, MCTS settings, training stages, and search constraints). Typical knobs you may tune:

- `ModelConfig.node_feature_dim`, `hidden_dim`, `num_heads`
- `MCTSConfig.num_simulations`, `exploration_weight`
- `ArchitectureSearchConfig.max_neurons`, `target_accuracy`

## Example usage (programmatic)

```python
from experiment_modules.architecture_trainer import ArchitectureTrainer
from experiment_modules.config import OverallConfig

config = OverallConfig()
trainer = ArchitectureTrainer(config)
history = trainer.run_training()
```

Adjust `config` before instantiating the trainer to change search limits, model sizes, or training schedule.

## Troubleshooting & tips

- Shape mismatches (e.g., `mat1 and mat2 shapes cannot be multiplied`) usually mean `ModelConfig.node_feature_dim` does not match the feature length produced in `blueprint_modules/network.py` (`Neuron.to_feature_vector()`); verify both sides match.
- Policy / mask size mismatches often relate to `max_neurons` vs actual indexed neuron IDs. The `ActionManager` in `architect_modules/policy_value_net.py` builds masks; inspect it when you see related errors.

## Results & expected behavior

PAINT is built as a research prototype. In experiments it aims to:

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
