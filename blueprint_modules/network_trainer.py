from .network import NeuralArchitecture, GraphNeuralNetwork
import torch
import torch.nn as nn
from typing import Tuple
import traceback
from torch.amp import autocast
from torch.amp import GradScaler

class QuickTrainer:
    """Handles training and evaluation of graph-based neural networks.

    This class provides a streamlined process for training a `GraphNeuralNetwork`
    model based on a given `NeuralArchitecture`.

    Attributes:
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        device (str): The device to run on ('cpu' or 'cuda').
        max_epochs (int): The maximum number of epochs for training.
        use_mixed_precision (bool): Whether to use mixed precision training.
        eval_samples (int, optional): The number of samples to use for
            evaluation. Defaults to None.
        scaler (GradScaler, optional): The gradient scaler for mixed precision
            training. Defaults to None.
        model (GraphNeuralNetwork, optional): The neural network model.
            Defaults to None.
    """

    def __init__(self, train_loader, test_loader, device='cpu', max_epochs=5, use_mixed_precision=False, eval_samples=None):
        """Initializes the QuickTrainer.

        Args:
            train_loader: DataLoader for the training dataset.
            test_loader: DataLoader for the test dataset.
            device (str): The device to run on. Defaults to 'cpu'.
            max_epochs (int): The maximum number of epochs for training.
                Defaults to 5.
            use_mixed_precision (bool): Whether to use mixed precision training.
                Defaults to False.
            eval_samples (int, optional): The number of samples to use for
                evaluation. Defaults to None.
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.use_mixed_precision = use_mixed_precision
        self.eval_samples = eval_samples  # If set, limit evaluation to this many samples

        # Setup scaler for mixed precision
        if use_mixed_precision:
            # Construct GradScaler for mixed precision. Newer PyTorch versions accept
            # a `device` argument (e.g., 'cuda:0'); pass our configured device when
            # CUDA is available. Fall back to the no-arg constructor if the
            # installed GradScaler doesn't accept `device`.
            if torch.cuda.is_available():
                try:
                    self.scaler = GradScaler(device=str(self.device))
                except TypeError:
                    # Older PyTorch versions don't accept device argument
                    self.scaler = GradScaler()
            else:
                self.scaler = None
        else:
            self.scaler = None

        # Initialize model as None - will be set when architecture is provided
        self.model = None

    def _move_loader_dataset_to_device(self, loader) -> bool:
        """Attempts to move an entire DataLoader dataset to the specified device.

        Args:
            loader: The DataLoader whose dataset to move.

        Returns:
            bool: True if the dataset was moved, False otherwise.
        """
        dataset = getattr(loader, 'dataset', None)
        if dataset is None:
            return False

        # Do not move to CUDA if using multiple workers
        # `self.device` may be a string (e.g. 'cuda:0') or a torch.device object;
        # convert to string to perform a safe startswith check.
        device_str = str(self.device)
        if device_str.startswith('cuda') and hasattr(loader, 'num_workers') and loader.num_workers > 0:
            return False

        try:
            # TensorDataset: .tensors is a tuple of tensors
            if hasattr(dataset, 'tensors'):
                try:
                    dataset.tensors = tuple(
                        t.to(self.device) if isinstance(t, torch.Tensor) else t
                        for t in dataset.tensors
                    )
                    return True
                except Exception:
                    return False

            # Common torchvision datasets (MNIST, etc.) often have .data and .targets
            if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
                moved_any = False
                try:
                    if isinstance(dataset.data, torch.Tensor):
                        dataset.data = dataset.data.to(self.device)
                        moved_any = True
                except Exception:
                    pass
                try:
                    if isinstance(dataset.targets, torch.Tensor):
                        dataset.targets = dataset.targets.to(self.device)
                        moved_any = True
                except Exception:
                    pass
                return moved_any

        except Exception:
            return False

        return False
    
    def train_and_evaluate(self, architecture: NeuralArchitecture) -> Tuple[float, float]:
        """Trains and evaluates the graph-based network.

        This method supports fractional epochs.

        Args:
            architecture (NeuralArchitecture): The neural architecture to train.

        Returns:
            Tuple[float, float]: A tuple containing the final accuracy and the
                average loss of the last epoch.
        """
        try:
            # Convert architecture to executable model
            self.update_architecture(architecture)
            model = self.model

            # Ensure model resides on the configured device
            try:
                model.to(self.device)
            except Exception:
                pass

            # Setup training
            # Do NOT call `.to()` on the optimizer object — optimizers don't support `.to()`.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            # Attempt to move the entire training dataset to device once.
            train_dataset_moved = self._move_loader_dataset_to_device(self.train_loader)

            # Training loop with fractional epoch support
            model.train()
            num_full_epochs = int(self.max_epochs)
            fractional_part = self.max_epochs - num_full_epochs
            
            for epoch in range(num_full_epochs):
                epoch_loss = 0.0
                batch_count = 0
                correct = 0
                total = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    # If we didn't move the whole dataset up-front, move this batch.
                    if not train_dataset_moved:
                        data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()

                    # Forward pass with optimizations
                    # Use autocast when mixed precision is enabled and GPU is available
                    if self.use_mixed_precision and torch.cuda.is_available():
                        with autocast():
                            output = model(data, use_mixed_precision=self.use_mixed_precision)
                    else:
                        output = model(data, use_mixed_precision=self.use_mixed_precision)
                    loss = criterion(output, target)

                    # Backward pass with mixed precision support
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            # Handle fractional epoch if present
            if fractional_part > 0:
                num_batches_fractional = max(1, int(len(self.train_loader) * fractional_part))
                epoch_loss = 0.0
                batch_count = 0
                correct = 0
                total = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if batch_idx >= num_batches_fractional:
                        break

                    if not train_dataset_moved:
                        data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()

                    # Forward pass with optimizations
                    if self.use_mixed_precision and torch.cuda.is_available():
                        with autocast():
                            output = model(data, use_mixed_precision=self.use_mixed_precision)
                    else:
                        output = model(data, use_mixed_precision=self.use_mixed_precision)
                    loss = criterion(output, target)

                    # Backward pass with mixed precision support
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            # Final evaluation on test set
            final_accuracy, final_loss = self.evaluate_model(model)
            
            # Store performance metrics
            architecture.performance_metrics = {
                'accuracy': final_accuracy,
                'neuron_count': len(architecture.neurons),
                'connection_count': len(architecture.connections),
                'parameter_count': model.get_parameter_count(),
                'layers': len(model.layer_neurons) if hasattr(model, 'layer_neurons') else 1
            }
            
            return final_accuracy, final_loss
            
        except Exception as e:
            print(f"Training error: {e}")
            traceback.print_exc()
            return 0.0, 0.0
    
    def evaluate_model(self, model: GraphNeuralNetwork) -> float:
        """Evaluates the trained model on test data.

        If `self.eval_samples` is set, only evaluates on the first N samples.

        Args:
            model (GraphNeuralNetwork): The trained model.

        Returns:
            float: The accuracy of the model.
        """
        model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = nn.CrossEntropyLoss()

        # Try to move the whole test dataset to device once. If that succeeds,
        # we can skip per-batch `.to()` calls which reduces host->device transfers.
        test_dataset_moved = self._move_loader_dataset_to_device(self.test_loader)

        with torch.no_grad():
            batch_count = 0
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # Move inputs to the trainer device before any model call if needed
                if not test_dataset_moved:
                    data, target = data.to(self.device), target.to(self.device)

                output = model(data, use_mixed_precision=self.use_mixed_precision)
                batch_loss = criterion(output, target)
                loss += batch_loss.item()

                # Count batches for robust averaging
                batch_count += 1

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Early exit if we've evaluated enough samples
                if self.eval_samples is not None and total >= self.eval_samples:
                    break

        avg_loss = loss / max(1, batch_count)
        return (correct / total if total > 0 else 0.0), avg_loss

    def update_architecture(self, architecture: NeuralArchitecture):
        """Updates the model with a new architecture.

        Args:
            architecture (NeuralArchitecture): The new neural architecture.
        """
        if self.model is None:
            self.model = GraphNeuralNetwork(architecture, device=self.device)
        else:
            self.model.update_architecture(architecture)

    def architecture_to_pytorch(self, architecture: NeuralArchitecture) -> GraphNeuralNetwork:
        """Converts an architecture to an executable PyTorch model.

        Args:
            architecture (NeuralArchitecture): The neural architecture.

        Returns:
            GraphNeuralNetwork: The executable PyTorch model.
        """
        return GraphNeuralNetwork(architecture)
