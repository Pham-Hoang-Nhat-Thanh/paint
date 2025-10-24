import os
import sys

# This makes the test runnable from the repository root or the tests folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from blueprint_modules.network import NeuralArchitecture, NeuronType, ActivationType
from blueprint_modules.action import Action, ActionType, ActionSpace
from architect_modules.graph_transformer import GraphTransformer
from architect_modules.policy_value_net import UnifiedPolicyValueNetwork, ActionManager
import torch
import numpy as np


def test_graph_transformer():
    """Test the Graph Transformer implementation"""
    print("=== Testing Graph Transformer ===")
    
    # Create a simple test architecture
    arch = NeuralArchitecture()
    action_space = ActionSpace(max_neurons=1000)  # Increased
    
    # Add some structure
    for i in range(3):
        action = Action(ActionType.ADD_NEURON, activation=ActivationType.RELU)
        action_space.apply_action(arch, action)
    
    # Add some connections
    input_neurons = [nid for nid, neuron in arch.neurons.items() 
                    if neuron.neuron_type == NeuronType.INPUT]
    hidden_neurons = [nid for nid, neuron in arch.neurons.items() 
                     if neuron.neuron_type == NeuronType.HIDDEN]
    
    for i in range(5):
        if hidden_neurons:
            action = Action(
                ActionType.ADD_CONNECTION,
                source_neuron=np.random.choice(input_neurons[:10]),  # Only first 10 inputs
                target_neuron=np.random.choice(hidden_neurons)
            )
            action_space.apply_action(arch, action)
    
    print(f"Test architecture: {arch}")
    
    # Convert to graph data
    graph_data = arch.to_graph_representation()
    
    # Add batch dimension and layer positions
    graph_data['node_features'] = graph_data['node_features'].unsqueeze(0)
    graph_data['layer_positions'] = torch.FloatTensor([[n.layer_position for n in arch.neurons.values()]])
    
    print(f"Node features shape: {graph_data['node_features'].shape}")
    print(f"Edge index shape: {graph_data['edge_index'].shape}")
    
    # Test Graph Transformer
    node_feature_dim = graph_data['node_features'].shape[-1]
    transformer = GraphTransformer(node_feature_dim=node_feature_dim, hidden_dim=64)
    
    global_embed, node_embeds = transformer(graph_data)
    
    print(f"Global embedding shape: {global_embed.shape}")
    print(f"Node embeddings shape: {node_embeds.shape}")
    print("✓ Graph Transformer working correctly")
    
    # Test Policy-Value Network with larger max_neurons
    policy_net = UnifiedPolicyValueNetwork(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        max_neurons=1000  # Increased to handle MNIST input size
    )
    
    output = policy_net(graph_data)
    print(f"Policy-value output keys: {list(output.keys())}")
    print(f"Action type logits shape: {output['action_type'].shape}")
    print(f"Source logits shape: {output['source_logits'].shape}")
    print(f"Value output shape: {output['value'].shape}")
    print("✓ Policy-Value Network working correctly")
    
    # Test Action Manager with larger max_neurons
    action_manager = ActionManager(max_neurons=1000)  # Increased
    action = action_manager.select_action(output, arch, exploration=True)
    print(f"Selected action: {action}")
    print("✓ Action Manager working correctly")
    
    print("=== Graph Transformer Tests Complete ===")

if __name__ == "__main__":
    test_graph_transformer()