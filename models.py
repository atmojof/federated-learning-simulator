import torch
import torch.nn as nn
import numpy as np

class SimpleMLP(nn.Module):
    """
    A lightweight Multi-Layer Perceptron (MLP) for digit/image classification.
    Runs extremely fast on CPU, ideal for interactive simulations.
    """
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten input: (Batch, Channels, Height, Width) -> (Batch, Input_Dim)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_weights(model):
    """
    Extracts model weights as a dictionary of NumPy arrays.
    """
    return {k: v.cpu().detach().numpy().copy() for k, v in model.state_dict().items()}


def set_weights(model, weights):
    """
    Loads model weights from a dictionary of NumPy arrays.
    """
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    model.load_state_dict(state_dict)


def add_dp_noise(delta_weights, noise_multiplier, sensitivity=0.1):
    """
    Applies Local Differential Privacy by adding calibrated Gaussian noise
    to the client's weight update (delta) before sending it to the server.
    
    Formula: Delta_noisy = Delta + N(0, (noise_multiplier * sensitivity)^2)
    """
    if noise_multiplier <= 0:
        return delta_weights
        
    noisy_delta = {}
    for k, v in delta_weights.items():
        # Standard deviation of noise
        std = noise_multiplier * sensitivity
        noise = np.random.normal(loc=0.0, scale=std, size=v.shape).astype(np.float32)
        noisy_delta[k] = v + noise
        
    return noisy_delta
