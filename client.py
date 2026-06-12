import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import numpy as np

from models import get_weights, set_weights

class FLClient:
    """
    Simulates a local edge client in the federated network.
    Holds local private data and performs model training.
    """
    def __init__(self, client_id, x_data, y_targets, batch_size=32):
        self.client_id = client_id
        
        # Convert inputs to PyTorch Tensors
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_targets = torch.tensor(y_targets, dtype=torch.long)
        self.batch_size = min(batch_size, len(y_targets))
        
        # Create DataLoader for local training
        dataset = TensorDataset(self.x_data, self.y_targets)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, global_model, lr=0.01, local_epochs=3, mu=0.0):
        """
        Trains the global model on local client data.
        Supports FedProx proximal regularization if mu > 0.
        """
        # Create a local copy of the global model to update
        local_model = copy.deepcopy(global_model)
        local_model.train()
        
        # Capture global weights to compute proximal regularization term
        global_weights = {k: v.clone().detach() for k, v in global_model.state_dict().items()}
        
        optimizer = optim.SGD(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(local_epochs):
            batch_losses = []
            for batch_x, batch_y in self.dataloader:
                optimizer.zero_grad()
                outputs = local_model(batch_x)
                
                # Standard Cross-Entropy loss
                loss = criterion(outputs, batch_y)
                
                # Add FedProx Proximal Regularization Penalty: (mu / 2) * ||w - w_global||^2
                if mu > 0.0:
                    prox_penalty = 0.0
                    for name, param in local_model.named_parameters():
                        g_weight = global_weights[name]
                        prox_penalty += torch.sum((param - g_weight) ** 2)
                    loss = loss + (mu / 2.0) * prox_penalty
                    
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
                
                # Track training accuracy
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == batch_y).item()
                total_predictions += len(batch_y)
                
            epoch_losses.append(np.mean(batch_losses) if len(batch_losses) > 0 else 0.0)
            
        # Extract updated local weights
        local_weights = get_weights(local_model)
        
        # Extract baseline weights (global weights)
        base_weights = get_weights(global_model)
        
        # Compute weight update delta: Delta = W_local - W_global
        delta_weights = {k: local_weights[k] - base_weights[k] for k in base_weights.keys()}
        
        train_accuracy = correct_predictions / max(1, total_predictions)
        mean_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else 0.0
        
        return local_weights, delta_weights, mean_loss, train_accuracy
