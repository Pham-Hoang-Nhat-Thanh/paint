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

# Global cache for data loaders to avoid reloading
_data_cache = {}

def load_mnist_data(batch_size=64):
    """Load MNIST dataset with caching and optimized loading"""
    cache_key = f"mnist_{batch_size}"
    if cache_key in _data_cache:
        return _data_cache[cache_key]

    # Pre-check if data exists to provide feedback
    data_dir = './data'
    mnist_dir = os.path.join(data_dir, 'MNIST')
    if not os.path.exists(mnist_dir):
        print("Downloading MNIST dataset...")

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

    # Optimized data loading with parallel workers and memory pinning
    num_workers = min(4, os.cpu_count() or 2)  # Use up to 4 workers or available CPUs

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Cache the loaders
    _data_cache[cache_key] = (train_loader, test_loader)
    return train_loader, test_loader

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint file in the checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_ep') and f.endswith('.pth')]
    if not checkpoint_files:
        return None

    # Extract episode numbers and find the latest
    episodes = []
    for f in checkpoint_files:
        try:
            ep_str = f.split('checkpoint_ep')[1].split('.pth')[0]
            episodes.append((int(ep_str), f))
        except (ValueError, IndexError):
            continue

    if not episodes:
        return None

    latest_episode, latest_file = max(episodes, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest_file)

def main():
    # Load configuration
    config = OverallConfig()

    # Manual GPU selection - set to specific GPU (change this number as needed)
    gpu_id = 2  # Change this to select different GPU (0, 1, 2, etc.)
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        config.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        print(f"Manually selected GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        config.device = "cpu"
        print("CUDA not available or invalid GPU ID, using CPU")

    # Load data
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data(
        batch_size=config.search.evaluation_batch_size
    )

    # Create trainer
    trainer = ArchitectureTrainer(config, train_loader, test_loader)

    # Check for existing checkpoints and resume if available
    latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    if latest_checkpoint:
        print(f"Found existing checkpoint: {latest_checkpoint}")
        try:
            trainer.load_checkpoint(latest_checkpoint)
            print(f"Resumed training from episode {trainer.episode}")
        except Exception as e:
            print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            print("Starting training from scratch...")
    else:
        print("No existing checkpoints found. Starting training from scratch...")

    # Run training
    history = trainer.run_training()

    # Print final results
    best_episode = max(history, key=lambda x: x['final_accuracy'])
    print(f"\nTraining completed!")
    print(f"Best accuracy: {best_episode['final_accuracy']:.4f}")
    print(f"Best architecture: {best_episode['total_neurons']} neurons, "
          f"{best_episode['total_connections']} connections")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting...")
        exit(0)