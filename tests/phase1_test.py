import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Ensure the project root is on sys.path so `core_modules` can be imported
# This makes the test runnable from the repository root or the tests folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from blueprint_modules.network import NeuralArchitecture, GraphNeuralNetwork, NeuronType, ActivationType
from blueprint_modules.action import Action, ActionType, ActivationType, ActionSpace
from blueprint_modules.mcts import MCTS
from blueprint_modules.network_trainer import QuickTrainer


def test_phase1_with_training():
    """Test Phase 1 components with proper forward/backward propagation"""
    print("=== Phase 1: Testing with Forward/Backward Propagation ===")
    
    # Load MNIST data (simplified)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use smaller subset for testing
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    # Small subsets for quick testing
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(range(1000))
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(range(500))
    )
    
    # 1. Test basic architecture
    print("1. Testing NeuralArchitecture...")
    arch = NeuralArchitecture()
    print(f"Initial architecture: {arch}")
    
    # 2. Add some structure
    print("2. Adding hidden neurons and connections...")
    action_space = ActionSpace(max_neurons=100)
    
    # Add a few hidden neurons
    for i in range(5):
        action = Action(ActionType.ADD_NEURON, activation=ActivationType.RELU)
        action_space.apply_action(arch, action)
    
    # Add some random connections
    input_neurons = [nid for nid, neuron in arch.neurons.items() 
                    if neuron.neuron_type == NeuronType.INPUT]
    output_neurons = [nid for nid, neuron in arch.neurons.items() 
                     if neuron.neuron_type == NeuronType.OUTPUT]
    hidden_neurons = [nid for nid, neuron in arch.neurons.items() 
                     if neuron.neuron_type == NeuronType.HIDDEN]
    
    # Connect some inputs to hiddens and hiddens to outputs
    for i in range(10):
        if hidden_neurons:
            action = Action(
                ActionType.ADD_CONNECTION,
                source_neuron=np.random.choice(input_neurons),
                target_neuron=np.random.choice(hidden_neurons)
            )
            action_space.apply_action(arch, action)
            
            action = Action(
                ActionType.ADD_CONNECTION,
                source_neuron=np.random.choice(hidden_neurons),
                target_neuron=np.random.choice(output_neurons)
            )
            action_space.apply_action(arch, action)
    
    print(f"After modifications: {arch}")
    
    # 3. Test GraphNeuralNetwork conversion
    print("3. Testing GraphNeuralNetwork conversion...")
    try:
        model = GraphNeuralNetwork(arch)
        print(f"Successfully created model with {model.get_parameter_count()} parameters")
        
        # Test forward pass
        test_input = torch.randn(2, 784)  # Batch of 2, MNIST flattened
        output = model(test_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        print(f"Output sample: {output}")
        
    except Exception as e:
        print(f"Model creation failed: {e}")
        return
    
    # 4. Test training
    print("4. Testing training pipeline...")
    trainer = QuickTrainer(train_loader, test_loader, device='cpu', max_epochs=2)
    
    # Mock evaluation for now (full training takes time)
    def quick_eval(architecture):
        # Simple heuristic based on architecture properties
        hidden_count = sum(1 for n in architecture.neurons.values() 
                          if n.neuron_type == NeuronType.HIDDEN)
        connection_count = len(architecture.connections)
        
        # Base accuracy + bonuses for reasonable structure
        base_acc = 0.1
        hidden_bonus = min(hidden_count * 0.02, 0.3)  # Up to 30% bonus for hidden neurons
        connection_bonus = min(connection_count * 0.001, 0.4)  # Up to 40% bonus for connections
        
        return base_acc + hidden_bonus + connection_bonus

    # 5. Test MCTS with quick evaluation
    print("5. Testing MCTS with quick evaluation...")
    mcts = MCTS(action_space, quick_eval, exploration_weight=1.0)
    
    # Run a very short MCTS
    best_node = mcts.search(arch, iterations=10)
    if best_node:
        print(f"MCTS completed. Best node value: {best_node.value:.3f}")
        print(f"Best architecture: {best_node.architecture}")
    
    # 6. Test actual training on a simple architecture
    print("6. Testing actual training on found architecture...")
    best_arch = best_node.architecture
    
    print(f"Simple test architecture: {best_arch}")
    
    # Quick training
    accuracy = trainer.train_and_evaluate(best_arch)
    print(f"Training completed. Accuracy: {accuracy:.4f}")
    
    print("=== Phase 1 Testing Complete ===")

if __name__ == "__main__":
    test_phase1_with_training()