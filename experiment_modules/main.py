import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys

# Ensure the project root is on sys.path so `modules` can be imported
# This makes the test runnable from the repository root or the tests folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiment_modules.config import OverallConfig
from experiment_modules.architecture_trainer import ArchitectureTrainer

def load_mnist_data(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader

def main():
    # Load configuration
    config = OverallConfig()
    
    # Load data
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data(
        batch_size=config.search.evaluation_batch_size
    )
    
    # Create trainer
    trainer = ArchitectureTrainer(config, train_loader, test_loader)
    
    # Run training
    print("Starting architecture search training...")
    history = trainer.run_training()
    
    # Print final results
    best_episode = max(history, key=lambda x: x['final_accuracy'])
    print(f"\nTraining completed!")
    print(f"Best accuracy: {best_episode['final_accuracy']:.4f}")
    print(f"Best architecture: {best_episode['total_neurons']} neurons, "
          f"{best_episode['total_connections']} connections")

if __name__ == "__main__":
    main()