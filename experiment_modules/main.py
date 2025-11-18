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
    """Loads the MNIST dataset with caching and optimized loading.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing the training and test data loaders.
    """
    global _data_cache
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

    # Use moderate number of workers, but disable persistent_workers to avoid multiprocessing issues
    # when loaders are passed to ProcessPoolExecutor workers
    num_workers = 4  # Reduced from 8 for better multiprocessing compatibility
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=False, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=False, pin_memory=True
    )

    # Cache the loaders
    _data_cache[cache_key] = (train_loader, test_loader)
    return train_loader, test_loader

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Finds the latest checkpoint file in the checkpoint directory.

    Args:
        checkpoint_dir (str): The directory where checkpoints are stored.

    Returns:
        str: The path to the latest checkpoint file, or None if no checkpoint
            is found.
    """
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
    """The main function for running the architecture search."""
    # Load configuration
    config = OverallConfig()

    # GPU selection - respect config.device unless "auto" or environment override
    if torch.cuda.is_available():
        # Check for manual GPU selection via environment variable (highest priority)
        manual_gpu = os.environ.get('GPU_ID')
        if manual_gpu is not None:
            try:
                gpu_id = int(manual_gpu)
                if gpu_id < torch.cuda.device_count():
                    config.device = f"cuda:{gpu_id}"
                    torch.cuda.set_device(gpu_id)
                    print(f"Using environment GPU_ID={gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                else:
                    print(f"Invalid GPU ID {gpu_id} (only {torch.cuda.device_count()} GPUs available)")
                    print(f"Falling back to config.device: {config.device}")
            except (ValueError, TypeError):
                print(f"Invalid GPU_ID environment variable: {manual_gpu}")
                print(f"Falling back to config.device: {config.device}")
        
        # If config.device is "auto", select GPU with most free memory
        if config.device == "auto":
            print("Auto-selecting GPU with most free memory...")
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory
                free_gb = free_memory / (1024**3)
                gpu_info.append((free_memory, i, free_gb))
                print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {free_gb:.2f} GB free")
            
            # Select GPU with MOST free memory (not least!)
            best_gpu_info = max(gpu_info, key=lambda x: x[0])
            best_gpu = best_gpu_info[1]
            best_free_gb = best_gpu_info[2]
            
            config.device = f"cuda:{best_gpu}"
            torch.cuda.set_device(best_gpu)
            print(f"Auto-selected GPU {best_gpu} with {best_free_gb:.2f} GB free: {torch.cuda.get_device_name(best_gpu)}")
        elif config.device.startswith("cuda:"):
            # Respect explicit cuda:X setting from config
            try:
                gpu_id = int(config.device.split(":")[1])
                if gpu_id < torch.cuda.device_count():
                    torch.cuda.set_device(gpu_id)
                    props = torch.cuda.get_device_properties(gpu_id)
                    free_memory = props.total_memory - torch.cuda.memory_allocated(gpu_id)
                    free_gb = free_memory / (1024**3)
                    print(f"Using config.device={config.device}: {torch.cuda.get_device_name(gpu_id)} ({free_gb:.2f} GB free)")
                else:
                    print(f"Config device {config.device} invalid (only {torch.cuda.device_count()} GPUs available)")
                    print("Falling back to GPU 0")
                    config.device = "cuda:0"
                    torch.cuda.set_device(0)
            except (ValueError, IndexError):
                print(f"Invalid device format: {config.device}")
                print("Falling back to GPU 0")
                config.device = "cuda:0"
                torch.cuda.set_device(0)
    else:
        config.device = "cpu"
        print("CUDA not available, using CPU")

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
    try:
        history = trainer.run_training()

        # Print final results
        best_episode = max(history, key=lambda x: x['final_accuracy'])
        print(f"\nTraining completed!")
        print(f"Best accuracy: {best_episode['final_accuracy']:.4f}")
        print(f"Best architecture: {best_episode['total_neurons']} neurons, "
              f"{best_episode['total_connections']} connections")
    finally:
        # Always clean up trainer resources
        trainer.cleanup()

def cleanup_and_exit():
    """Cleans up resources and exits gracefully."""
    print("\nCleaning up resources...")
    
    # Clear the data cache to help with cleanup
    global _data_cache
    _data_cache.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Ensure CUDA memory is cleared if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Cleanup completed. Exiting...")

if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(signum, frame):
        cleanup_and_exit()
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        cleanup_and_exit()
        exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_and_exit()
        exit(1)
