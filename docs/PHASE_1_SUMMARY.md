# Phase 1 — Foundation

Goal
- Implement and validate the graph-based neural architecture representation and a quick training/evaluation pipeline for MNIST.
- Deliverables are runnable unit tests and a minimal MCTS-based search over architectures.

Core components & refs
- Architecture graph: [`core_modules.network.NeuralArchitecture`](core_modules/network.py) — implementation and helpers in [core_modules/network.py](core_modules/network.py).
- Executable model: [`core_modules.network.GraphNeuralNetwork`](core_modules/network.py) — conversion of a graph to PyTorch layers in [core_modules/network.py](core_modules/network.py).
- Search: [`core_modules.mcts.MCTS`](core_modules/mcts.py) and [`core_modules.mcts.MCTSNode`](core_modules/mcts.py) — tree search logic in [core_modules/mcts.py](core_modules/mcts.py).
- Actions & action space: [`core_modules.action.ActionSpace`](core_modules/action.py) and [`core_modules.action.Action`](core_modules/action.py) — valid moves and apply semantics in [core_modules/action.py](core_modules/action.py).
- Quick training & evaluation: [`core_modules.network_trainer.QuickTrainer`](core_modules/network_trainer.py) — simple training/eval pipeline in [core_modules/network_trainer.py](core_modules/network_trainer.py).

Milestones
1. NeuralArchitecture basics
   - Verify add/remove neuron and connection behavior.
   - Unit tests: [tests/phase1_test.py](tests/phase1_test.py) uses [`core_modules.network.NeuralArchitecture`](core_modules/network.py).

2. Convert graph -> PyTorch
   - Ensure [`core_modules.network.GraphNeuralNetwork`](core_modules/network.py) builds layers and forward works for MNIST-shaped inputs.
   - Confirm parameter counting via `get_parameter_count()`.

3. Quick training loop
   - Implement a short training run with [`core_modules.network_trainer.QuickTrainer`](core_modules/network_trainer.py).
   - Store `architecture.performance_metrics` after evaluation.

4. MCTS integration
   - Run a short MCTS (`[`core_modules.mcts.MCTS`](core_modules/mcts.py)`) over the action space and validate best candidate architecture.

How to run the Phase 1 checks
- Run the phase1 test script:
```sh
python tests/phase1_test.py
```
- Inspect test-driven outputs in [tests/phase1_test.py](tests/phase1_test.py).

Notes & expectations
- Tests use a small MNIST subset; keep `QuickTrainer.max_epochs` low to keep CI fast.
- If GraphNeuralNetwork cannot exactly match 10 outputs, the code raise a `ValueError` — see [`core_modules.network.GraphNeuralNetwork.forward`](core_modules/network.py).