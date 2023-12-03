import flwr as fl
import numpy as np

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
    """
    This function takes in a list of client class instances, the total number of clients,
    and how many users to sample and then returns a list of the clients that will be used for
    the federated learning model based on sampling from the normal distribution of their differences.
    """
    # sort differences in ascending order
    client_diffs = [(client.id, client.budgets - client.demands) for client in clients]
    # only consider positive differences (budgets - demands > 0)
    client_diffs = [diff for diff in client_diffs if diff[1] > 0]
    # sort in ascending order
    client_diffs.sort(key=lambda x: x[1])

    # sample from the normal distribution
    median_index = len(client_diffs) // 2
    median_value = client_diffs[median_index][1]

    # get last value as sorted in ascending order
    largest_diff = client_diffs[-1][1]

    # mean = median, standard_dev = largest_diff - median / 3 (based on empirical rule)
    std_dev = (largest_diff - median_value) / 3
    sampled_values = np.random.normal(median_value, std_dev, num_to_sample)

    # select clients whose differences are closest to the sampled values
    sampled_clients = []
    for val in sampled_values:
        # abs difference of sampled value and actual
        closest_client = min(client_diffs, key=lambda x: abs(x[1] - val))
        sampled_clients.append(closest_client[0])
        # remove to avoid sampling duplicates
        client_diffs.remove(closest_client)  

    return [client for client in clients if client.id in sampled_clients]

# ensure clients have budgets and demands
class CustomClientProxy(fl.server.ClientProxy):
    def __init__(self, cid, budgets, demands):
        super().__init__(cid)
        self.budgets = budgets
        self.demands = demands