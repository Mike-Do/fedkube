"""
In this file, we implement the normal share mechanism.
Broadly speaking, we take in a list of the differences 
between privacy budgets and their demands, and then
plot the differences based on the normal distribution.

We then sample the normal distribution to get the
best X number of clients to run our federated learning
model.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import OrderedDict

class Client:
    def __init__(self, id, demands, budgets):
        self.id = id
        self.demands = demands
        self.budgets = budgets

def read_csv(csv_file):
    """
    This function takes in a csv file and returns a list of client 
    class instances that have the demands and budgets of the clients.
    """

    # read the csv file
    df = pd.read_csv(csv_file)

    # create a list of client class instances
    clients = []
    for i in range(len(df)):
        client = Client(i, df["Demand"][i], df["Budget"][i])
        clients.append(client)

    return clients


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

    '''Print sorted clients'''
    # print the sorted positive differences to verify
    # print("Sorted positive differences:")
    # for diff in client_diffs:
    #     print(diff)
    # # plot the differences and make sure they follow the normal distribution
    # # x-axis: client id based on ascending order of differences
    # # y-axis: differences
    # # Create a new list for x-axis labels (client IDs)
    # client_ids = [client[0] for client in client_diffs]
    
    # # plot the differences and make sure they follow the normal distribution
    # # x-axis: sequential numbers representing sorted order
    # # y-axis: differences
    # plt.bar(range(len(client_diffs)), [diff[1] for diff in client_diffs])
    
    # # Set the x-axis labels to client IDs
    # plt.xticks(range(len(client_diffs)), client_ids)
    
    # plt.xlabel('Client ID')
    # plt.ylabel('Positive Difference')
    # plt.title('Sorted Positive Differences between Budgets and Demands per Client')
    # plt.show()

    # sample from the normal distribution
    median_index = len(client_diffs) // 2
    median_value = client_diffs[median_index][1]

    # get last value as sorted in ascending order
    largest_diff = client_diffs[-1][1]

    # mean = median, standard_dev = largest_diff - median / 3 (based on empiricial rule)
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


# create main function to run the program
def main():
    # run the read_csv function on file ./clients_budget_demands.csv
    clients = read_csv("./clients_budget_demand.csv")

    '''Run Normal Share Once'''
    # # run the normal_share function on the clients list
    # sampled_clients = normal_share(clients, 10, 2)

    # # print length of sampled clients
    # print("Length:", len(sampled_clients))

    # # print the sampled clients
    # for client in sampled_clients:
    #     print(client.id)

    '''Run Normal Share 100 Times'''
    # Initialize a hashmap to store frequencies
    client_frequency = {client.id: 0 for client in clients}

    # run normal_share 100 times
    for _ in range(100):
        sampled_clients = normal_share(clients, len(clients), 2)
        for client in sampled_clients:
            client_frequency[client.id] += 1

    # print the frequency map
    print("Client Sampling Frequency:")
    for client_id, frequency in client_frequency.items():
        print(f"Client ID {client_id}: {frequency}")

main()