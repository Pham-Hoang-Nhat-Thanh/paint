# Phase 2 — Core Model

Goal
- Implement graph transformer backbone and a unified policy-value network for architecture search.
- Integrate action masking and an ActionManager for valid policy-guided moves.

Core components & refs
- Graph Transformer backbone: [`architect_modules.graph_transformer.GraphTransformer`](architect_modules/graph_transformer.py) with modules:
  - [`architect_modules.graph_transformer.PositionalEncoding`](architect_modules/graph_transformer.py)
  - [`architect_modules.graph_transformer.EdgeAwareAttention`](architect_modules/graph_transformer.py)
  - [`architect_modules.graph_transformer.HierarchicalPooling`](architect_modules/graph_transformer.py)
  See implementation in [architect_modules/graph_transformer.py](architect_modules/graph_transformer.py).
- Policy/value head: [`architect_modules.policy_value_net.UnifiedPolicyValueNetwork`](architect_modules/policy_value_net.py) and helper [`architect_modules.policy_value_net.ActionManager`](architect_modules/policy_value_net.py) in [architect_modules/policy_value_net.py](architect_modules/policy_value_net.py).
- Guided MCTS: [`architect_modules.guided_mcts.NeuralMCTS`](architect_modules/guided_mcts.py) and [`architect_modules.guided_mcts.NeuralMCTSNode`](architect_modules/guided_mcts.py) in [architect_modules/guided_mcts.py](architect_modules/guided_mcts.py).
- Training curriculum and loss: [`architect_modules.training_curriculum.TrainingCurriculum`](architect_modules/training_curriculum.py) and [`architect_modules.training_curriculum.PolicyValueLoss`](architect_modules/training_curriculum.py) in [architect_modules/training_curriculum.py](architect_modules/training_curriculum.py).

Milestones
1. Graph Transformer
   - Verify positional encoding and multi-head edge-aware attention behavior.
   - Unit test scaffold: [tests/phase2_test.py](tests/phase2_test.py) calls [`architect_modules.graph_transformer.GraphTransformer`](architect_modules/graph_transformer.py).

2. Policy-value network
   - Implement shared backbone, policy heads (factorized outputs) and value head as in [`architect_modules.policy_value_net.UnifiedPolicyValueNetwork`](architect_modules/policy_value_net.py).
   - Confirm weight init via `_initialize_weights()`.

3. Action Manager & masking
   - Implement masks in [`architect_modules.policy_value_net.ActionManager.get_action_masks`](architect_modules/policy_value_net.py) and selection logic in `.select_action()`.
   - Ensure masks reference [`core_modules.network.NeuralArchitecture`](core_modules/network.py) layout and [`core_modules.action.ActionType`](core_modules/action.py).

4. Neural-guided MCTS
   - Integrate the policy-value network into [`architect_modules.guided_mcts.NeuralMCTS`](architect_modules/guided_mcts.py).
   - Prepare graph input via `.to_graph_representation()` and [`architect_modules.guided_mcts.NeuralMCTS._prepare_graph_data`](architect_modules/guided_mcts.py).

How to run Phase 2 checks
- Run the phase2 test script:
```sh
python tests/phase2_test.py
```
- Inspect shapes and keys in the policy-value output (test prints `action_type`, `value` shapes).

Notes & expectations
- Keep `max_neurons` consistent across `ActionManager` and `UnifiedPolicyValueNetwork`.
- Use the hierarchical pooling (`architect_modules.graph_transformer.HierarchicalPooling`) to derive a compact global embedding for the policy/value heads.
- Training end-to-end (policy + MCTS) is iterative; Phase 2 focuses on a stable forward pass, masking correctness, and integration tests in [tests/phase2_test.py](tests/phase2_test.py).

References
- Project roadmap: [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)
- Core modules: [core_modules/network.py](core_modules/network.py), [core_modules/mcts.py](core_modules/mcts.py), [core_modules/action.py](core_modules/action.py), [core_modules/network_trainer.py](core_modules/network_trainer.py)
- Architect modules: [architect_modules/graph_transformer.py](architect_modules/graph_transformer.py), [architect_modules/policy_value_net.py](architect_modules/policy_value_net.py), [architect_modules/guided_mcts.py](architect_modules/guided_mcts.py), [architect_modules/training_curriculum.py](architect_modules/training_curriculum.py)