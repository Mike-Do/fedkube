import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import SimpleClientManager
import numpy as np

import argparse

class NormalShareClientManager(fl.server.SimpleClientManager):
    def __init__(self, total_clients, num_to_sample):
        super().__init__()
        self.total_clients = total_clients
        self.num_to_sample = num_to_sample

    def sample(self, num_clients: int, min_num_clients: int = None, criterion: fl.server.criterion.Criterion = None):
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = self.total_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        request_properties = {"tensor_type": "str"}
        ins = fl.common.GetPropertiesIns(
            config=request_properties
        )
        client_id_properties = {}
        for key, value in self.clients.items():
            properties = value.get_properties(ins=ins, timeout=None)
            if properties.status.message == "Success":
                client_id_properties[key] = properties.properties

        available_cids = list(self.clients)
        # AssertionError: (self.clients) {'ipv4:127.0.0.1:45898': <flwr.server.fleet.grpc_bidi.grpc_client_proxy.GrpcClientProxy object at 0x7fc4ed05fca0>, 'ipv4:127.0.0.1:45902': <flwr.server.fleet.grpc_bidi.grpc_client_proxy.GrpcClientProxy object at 0x7fc4eccb0a30>, 'ipv4:127.0.0.1:45908': <flwr.server.fleet.grpc_bidi.grpc_client_proxy.GrpcClientProxy object at 0x7fc4eccb87c0>}
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # normal share sampling
        sampled_cids = normal_share_sample(available_cids, client_id_properties, self.total_clients, self.num_to_sample)
        assert len(sampled_cids) >= self.num_to_sample, "Sampled clients less than num_to_sample"
        return [self.clients[cid] for cid in sampled_cids]
        # return sampled_cids

def normal_share_sample(available_cids, client_id_properties, total_clients, num_to_sample):
    """
    This function takes in a list of client class instances, the total number of clients,
    and how many users to sample and then returns a list of the clients that will be used for
    the federated learning model based on sampling from the normal distribution of their differences.
    """
    # sort differences in ascending order
    client_diffs = []
    # print("normal_share_sample >>>", client_id_properties)

    for cid in available_cids:
        client_diffs.append((cid, client_id_properties[cid]["privacy_budget"] - client_id_properties[cid]["demand"]))
    print("(normal_share_sample) client_diffs >>>", client_diffs)
    # client_diffs = [(client.id, client.budgets - client.demands) for client in clients]

    # only consider positive differences (budgets - demands > 0)
    client_diffs = [diff for diff in client_diffs if diff[1] > 0]
    print("(normal_share_sample) filtered client_diffs >>>", client_diffs)
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
    print("(normal_share_sample) sampled_values >>>", sampled_values)
    # select clients whose differences are closest to the sampled values
    sampled_clients = []
    for val in sampled_values:
        # abs difference of sampled value and actual
        if not client_diffs:  # Check if client_diffs is empty
            break
        closest_client = min(client_diffs, key=lambda x: abs(x[1] - val))
        sampled_clients.append(closest_client[0])
        # remove to avoid sampling duplicates
        client_diffs.remove(closest_client)  
    sampled_cids = [cid for cid in available_cids if cid in sampled_clients]
    print("WOOHOO >>>", sampled_cids)
    return sampled_cids

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    # Define strategy
    args = argparse.ArgumentParser()
    args.add_argument("--num_to_sample", type=int, default=4)
    args.add_argument("--total_clients", type=int, default=5)
    args.add_argument("--ipnip", type=str, default="0.0.0.0")
    args.add_argument("--ipnport", type=str, default="8000")

    args = args.parse_args()

    num_to_sample = args.num_to_sample
    total_clients = args.total_clients

    print("Total number of clients: ", total_clients)
    print("Number of clients to sample: ", num_to_sample)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Sample 100% of available clients for the next round
        fraction_evaluate=1.0, # Use 100% of available clients to evaluate
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=2,
        min_evaluate_clients=2,
    )

    strategy_dpfixed = fl.server.strategy.DPFedAvgFixed(
        strategy, num_to_sample, clip_norm = 0.5, noise_multiplier = 0.01, server_side_noising = True)

    # Start server
    ipnip = args.ipnip
    ipnport = args.ipnport
    fl.server.start_server(
        server_address=f"{ipnip}:{ipnport}",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy_dpfixed,
        client_manager=NormalShareClientManager(total_clients=total_clients, num_to_sample=num_to_sample),
    )
