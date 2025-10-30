from .network import NeuralArchitecture, GraphNeuralNetwork
import torch
import torch.nn as nn
from typing import Tuple
import traceback

class QuickTrainer:
    """Proper training and evaluation using the graph-based networks"""
    
    def __init__(self, train_loader, test_loader, device='cpu', max_epochs=5, use_mixed_precision=False):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.use_mixed_precision = use_mixed_precision

        # Setup scaler for mixed precision
        if use_mixed_precision:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
    
    def train_and_evaluate(self, architecture: NeuralArchitecture) -> Tuple[float, float]:
        """Train the graph-based network and return (final_accuracy, last_epoch_avg_loss)

        Returns:
            (accuracy, loss)
        """
        try:
            # Convert architecture to executable model
            model = GraphNeuralNetwork(architecture)
            model.to(self.device)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            model.train()
            for epoch in range(self.max_epochs):
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
        """Evaluate the trained model on test data"""
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

        avg_loss = loss / len(self.test_loader) if len(self.test_loader) > 0 else 0.0
        return (correct / total if total > 0 else 0.0), avg_loss

    def architecture_to_pytorch(self, architecture: NeuralArchitecture) -> GraphNeuralNetwork:
        """Convert architecture to executable PyTorch model"""
        return GraphNeuralNetwork(architecture)