from .network import NeuralArchitecture, GraphNeuralNetwork
import torch
import torch.nn as nn
from typing import Tuple
import traceback

class QuickTrainer:
    """Proper training and evaluation using the graph-based networks"""

    def __init__(self, train_loader, test_loader, device='cpu', max_epochs=5, use_mixed_precision=False, eval_samples=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.use_mixed_precision = use_mixed_precision
        self.eval_samples = eval_samples  # If set, limit evaluation to this many samples

        # Setup scaler for mixed precision
        if use_mixed_precision:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None

        # Initialize model as None - will be set when architecture is provided
        self.model = None
    
    def train_and_evaluate(self, architecture: NeuralArchitecture) -> Tuple[float, float]:
        """Train the graph-based network and return (final_accuracy, last_epoch_avg_loss)

        Supports fractional epochs (e.g., 2.5 epochs = 2 full epochs + half of the third)

        Returns:
            (accuracy, loss)
        """
        try:
            # Convert architecture to executable model
            self.update_architecture(architecture)
            model = self.model

            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
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
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()

                    # Forward pass with optimizations
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

                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()

                    # Forward pass with optimizations
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
        """Evaluate the trained model on test data
        
        If self.eval_samples is set, only evaluates on first N samples for speed.
        """
        model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                # Move inputs to the trainer device before any model call
                data, target = data.to(self.device), target.to(self.device)

                output = model(data, use_mixed_precision=self.use_mixed_precision)
                batch_loss = criterion(output, target)
                loss += batch_loss.item()

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Early exit if we've evaluated enough samples
                if self.eval_samples is not None and total >= self.eval_samples:
                    break

        avg_loss = loss / max(1, total // (self.test_loader.batch_size if hasattr(self.test_loader, 'batch_size') else 64))
        return (correct / total if total > 0 else 0.0), avg_loss

    def update_architecture(self, architecture: NeuralArchitecture):
        """Update the model with a new architecture"""
        if self.model is None:
            self.model = GraphNeuralNetwork(architecture, device=self.device)
        else:
            self.model.update_architecture(architecture)

    def architecture_to_pytorch(self, architecture: NeuralArchitecture) -> GraphNeuralNetwork:
        """Convert architecture to executable PyTorch model"""
        return GraphNeuralNetwork(architecture)