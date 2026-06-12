import unittest
import numpy as np
import torch
import copy

from models import SimpleMLP, get_weights, set_weights, add_dp_noise
from data_utils import partition_data
from client import FLClient
from server import FLServer

class TestFLSimulator(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock dataset
        self.num_samples = 100
        self.features_dim = 784
        self.num_classes = 10
        
        self.mock_x = np.random.normal(loc=0.0, scale=1.0, size=(self.num_samples, 1, 28, 28)).astype(np.float32)
        # Random labels 0 to 9
        self.mock_y = np.random.randint(0, 10, size=self.num_samples)

    def test_data_partition_iid(self):
        num_clients = 5
        partitions = partition_data(self.mock_x, self.mock_y, num_clients, partition_type='iid')
        
        self.assertEqual(len(partitions), num_clients)
        # Check that all samples are distributed evenly (100 / 5 = 20 samples each)
        for i in range(num_clients):
            self.assertEqual(len(partitions[i]['targets']), 20)
            self.assertEqual(partitions[i]['data'].shape, (20, 1, 28, 28))

    def test_data_partition_dirichlet(self):
        num_clients = 3
        # alpha=0.1 should create high imbalances
        partitions = partition_data(self.mock_x, self.mock_y, num_clients, partition_type='noniid', alpha=0.1)
        
        self.assertEqual(len(partitions), num_clients)
        total_samples = sum([len(p['targets']) for p in partitions])
        self.assertEqual(total_samples, self.num_samples)

    def test_client_training(self):
        # Setup one client with 20 samples
        c_x = self.mock_x[:20]
        c_y = self.mock_y[:20]
        client = FLClient(client_id=0, x_data=c_x, y_targets=c_y, batch_size=5)
        
        # Setup model
        model = SimpleMLP(input_dim=784, hidden_dim=32, output_dim=10)
        initial_weights = copy.deepcopy(get_weights(model))
        
        # Train GNN model locally (FedAvg)
        c_weights, c_delta, loss, acc = client.train(model, lr=0.01, local_epochs=1, mu=0.0)
        
        # Check weights updated
        for k in initial_weights.keys():
            self.assertFalse(np.array_equal(c_weights[k], initial_weights[k]))
            # Delta should match W_local - W_global
            np.testing.assert_allclose(c_delta[k], c_weights[k] - initial_weights[k], rtol=1e-5, atol=1e-5)

    def test_server_aggregation(self):
        # Create server
        server = FLServer(self.mock_x, self.mock_y, input_dim=784, hidden_dim=32, output_dim=10)
        initial_global = get_weights(server.global_model)
        
        # Create mock client updates (deltas)
        client_deltas = [
            {k: np.ones_like(v) * 0.1 for k, v in initial_global.items()},
            {k: np.ones_like(v) * 0.2 for k, v in initial_global.items()}
        ]
        client_sizes = [10, 30] # 10 samples for client 0, 30 samples for client 1
        
        # Aggregate using FedAvg
        server.aggregate(client_deltas, client_sizes, algorithm='FedAvg', momentum_beta=0.0)
        new_global = server.get_global_weights()
        
        # Expected weighted delta = (10/40)*0.1 + (30/40)*0.2 = 0.25*0.1 + 0.75*0.2 = 0.025 + 0.15 = 0.175
        for k in initial_global.keys():
            expected_weight = initial_global[k] + 0.175
            np.testing.assert_allclose(new_global[k], expected_weight, rtol=1e-5, atol=1e-5)

    def test_dp_noise(self):
        delta = {'layer1': np.zeros((10, 10), dtype=np.float32)}
        
        # If noise multiplier is 0, delta should not change
        c_delta = add_dp_noise(delta, noise_multiplier=0.0, sensitivity=0.1)
        np.testing.assert_equal(c_delta['layer1'], delta['layer1'])
        
        # If noise multiplier > 0, delta should become noisy (nonzero)
        noisy_delta = add_dp_noise(delta, noise_multiplier=1.0, sensitivity=0.1)
        self.assertFalse(np.array_equal(noisy_delta['layer1'], delta['layer1']))
        self.assertEqual(noisy_delta['layer1'].shape, (10, 10))

if __name__ == '__main__':
    unittest.main()
