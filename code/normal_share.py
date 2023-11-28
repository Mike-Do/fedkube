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
import math
import random
import argparse
import os
import tempfile
import sys
import pandas as pd
from collections import OrderedDict
import warnings

# inheritance of Flower's client manager and rewrite sampling function

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
    # calculate the differences between the demands and budgets and associate them with their client id
    # plot the differences and make sure they follow the normal distribution
    # find median
    # handle duplicate differences
    # sort client' id's in ascending order
    # sample client id's based on how their differences are normally distributed
    # calculate percetange based on X / total

    # Calculate positive differences and associate them with client IDs
    client_diffs = [(client.id, client.budgets - client.demands) for client in clients if client.budgets - client.demands > 0]

    if not client_diffs:
        print("No clients with positive differences found.")
        return []

    # sort based on ascending order of client id
    # get the median
    # calculate the percents based on num_to_sample / total clients
    # sample from the left and right of the median based on the percents

    # Sort based on closeness to the mean
    mean_diff = np.mean([diff[1] for diff in client_diffs])
    std_diff = np.std([diff[1] for diff in client_diffs])
    client_diffs.sort(key=lambda x: abs(x[1] - mean_diff))

    # print the sorted positive differences
    print("Sorted positive differences:")
    for diff in client_diffs:
        print(diff)

    # Plot the sorted positive differences
    plt.bar([client[0] for client in client_diffs], [diff[1] for diff in client_diffs])
    plt.xlabel('Client ID')
    plt.ylabel('Positive Difference')
    plt.title('Sorted Positive Differences between Budgets and Demands per Client')
    plt.show()

    # Sample from the normal distribution
    sampled_values = np.random.normal(mean_diff, std_diff, num_to_sample)
    sampled_clients = []
    for val in sampled_values:
        closest_client = min(client_diffs, key=lambda x: abs(x[1] - val))
        sampled_clients.append(closest_client[0])
        client_diffs.remove(closest_client)  # Remove to avoid duplicates

    return [client for client in clients if client.id in sampled_clients]


# create main function to run the program
def main():
    # run the read_csv function on file ./clients_budget_demands.csv
    clients = read_csv("./clients_budget_demand.csv")

    # run the normal_share function on the clients list
    sampled_clients = normal_share(clients, 10, 2)

    # print length of sampled clients
    print("Length:", len(sampled_clients))

    # print the sampled clients
    for client in sampled_clients:
        print(client.id)

main()