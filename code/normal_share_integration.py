import flwr as fl
import random
import matplotlib.pyplot as plt
import pandas as pd

class NormalShareClientManager(fl.server.ClientManager):
    def __init__(self, total_clients, num_to_sample):
        super().__init__()
        self.total_clients = total_clients
        self.num_to_sample = num_to_sample

    def sample(self, num_clients: int, min_num_clients: int = None, criterion: fl.server.criterion.Criterion = None):
        all_clients = list(self.all().values())
        available_clients = [client for client in all_clients if client.is_available()]

        # ensure we have enough clients to sample from
        if len(available_clients) < num_clients:
            return []

        # override with normal_share function
        return normal_share(available_clients, self.total_clients, self.num_to_sample)

def normal_share(clients, total_clients, num_to_sample):
    # sort differences in ascending order
    client_diffs = [(client.cid, client.budgets - client.demands) for client in clients]

    # only consider positive differences
    client_diffs = [diff for diff in client_diffs if diff[1] > 0]

    # sort in ascending order
    client_diffs.sort(key=lambda x: x[1])

    # sample from the normal distribution
    median_index = len(client_diffs) // 2
    num_samples_each_side = num_to_sample // 2

    left_sample_indices = random.sample(range(median_index), num_samples_each_side)
    right_sample_indices = random.sample(range(median_index, len(client_diffs)), num_to_sample - num_samples_each_side)

    sampled_client_ids = [client_diffs[i][0] for i in left_sample_indices + right_sample_indices]

    return [client for client in clients if client.cid in sampled_client_ids]

# ensure clients have budgets and demands
class CustomClientProxy(fl.server.ClientProxy):
    def __init__(self, cid, budgets, demands):
        super().__init__(cid)
        self.budgets = budgets
        self.demands = demands