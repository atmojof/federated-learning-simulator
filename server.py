import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy

from models import SimpleMLP, get_weights, set_weights, add_dp_noise

class FLServer:
    """
    Simulates the central parameter server that orchestrates the
    Federated Learning training process, model updates, and evaluation.
    """
    def __init__(self, x_test, y_test, input_dim=784, hidden_dim=64, output_dim=10):
        # Global model initialization
        self.global_model = SimpleMLP(input_dim, hidden_dim, output_dim)
        
        # Test dataset
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        # Server momentum state (velocity vector)
        self.velocity = {k: np.zeros_like(v) for k, v in get_weights(self.global_model).items()}
        
    def get_global_weights(self):
        """
        Returns a copy of the current global model weights.
        """
        return get_weights(self.global_model)
        
    def set_global_weights(self, weights):
        """
        Updates the global model weights.
        """
        set_weights(self.global_model, weights)
        
    def aggregate(self, client_deltas, client_sizes, algorithm='FedAvg', momentum_beta=0.0, global_lr=1.0, dp_noise=0.0):
        """
        Aggregates weight update deltas from participating clients.
        Supports FedAvg, FedProx (same aggregation, proximal penalty is local), and FedAvgM.
        Applies Differential Privacy noise to the aggregated update if dp_noise > 0.
        """
        # Sum client samples
        total_samples = sum(client_sizes)
        
        # Initialize aggregated update delta
        aggregated_delta = {k: np.zeros_like(v) for k, v in self.get_global_weights().items()}
        
        # 1. Compute weighted average of client updates
        for delta, size in zip(client_deltas, client_sizes):
            weight = size / max(1, total_samples)
            for k in aggregated_delta.keys():
                aggregated_delta[k] += delta[k] * weight
                
        # 2. Add Differential Privacy Noise to the aggregated update (if enabled)
        # Sensitivity is scaled by the number of clients to reflect the impact
        if dp_noise > 0.0:
            # Inject noise with sensitivity scaled inversely by sqrt(clients)
            sensitivity = 0.05 / np.sqrt(len(client_sizes))
            aggregated_delta = add_dp_noise(aggregated_delta, dp_noise, sensitivity)
            
        # 3. Apply Server Momentum (FedAvgM)
        current_weights = self.get_global_weights()
        new_weights = {}
        
        for k in current_weights.keys():
            if momentum_beta > 0.0:
                # Update velocity: V_{t+1} = beta * V_t + Delta
                self.velocity[k] = momentum_beta * self.velocity[k] + aggregated_delta[k]
                # Update weights: W_{t+1} = W_t + global_lr * V_{t+1}
                update_step = global_lr * self.velocity[k]
            else:
                # Standard update: W_{t+1} = W_t + global_lr * Delta
                update_step = global_lr * aggregated_delta[k]
                
            new_weights[k] = current_weights[k] + update_step
            
        # Update server's global model
        self.set_global_weights(new_weights)
        
    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the global model on the central test dataset.
        Returns loss and accuracy.
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.test_loader:
            outputs = self.global_model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * len(batch_y)
            
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == batch_y).item()
            total += len(batch_y)
            
        mean_loss = test_loss / max(1, total)
        accuracy = correct / max(1, total)
        
        return mean_loss, accuracy
