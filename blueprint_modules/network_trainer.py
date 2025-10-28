from .network import NeuralArchitecture, GraphNeuralNetwork
import torch
import torch.nn as nn

class QuickTrainer:
    """Proper training and evaluation using the graph-based networks"""
    
    def __init__(self, train_loader, test_loader, device='cpu', max_epochs=5):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
    
    def train_and_evaluate(self, architecture: NeuralArchitecture) -> float:
        """Train the graph-based network and return final accuracy"""
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
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                
                epoch_accuracy = correct / total if total > 0 else 0.0
                print(f"Epoch {epoch}: Loss = {epoch_loss/20:.4f}, Accuracy = {epoch_accuracy:.4f}")
                print(f"Sample outputs: {output[:5]}")
            
            # Final evaluation
            final_accuracy = self.evaluate_model(model)
            
            # Store performance metrics
            architecture.performance_metrics = {
                'accuracy': final_accuracy,
                'neuron_count': len(architecture.neurons),
                'connection_count': len(architecture.connections),
                'parameter_count': model.get_parameter_count(),
                'layers': len(model.layer_neurons) if hasattr(model, 'layer_neurons') else 1
            }
            
            return final_accuracy
            
        except Exception as e:
            print(f"Training error: {e}")
            return 0.0
    
    def evaluate_model(self, model: GraphNeuralNetwork) -> float:
        """Evaluate the trained model on test data"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0.0

    def architecture_to_pytorch(self, architecture: NeuralArchitecture) -> GraphNeuralNetwork:
        """Convert architecture to executable PyTorch model"""
        return GraphNeuralNetwork(architecture)