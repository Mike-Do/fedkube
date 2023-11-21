import flwr as fl

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Sample 100% of available clients for the next round
        fraction_evaluate=1.0, # Use 100% of available clients to evaluate
        evaluate_metrics_aggregation_fn=weighted_average
    )

    # Start server
    ipnip = "172.20.211.1"
    ipnport = "8230"
    fl.server.start_server(
        server_address=f"{ipnip}:{ipnport}",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )