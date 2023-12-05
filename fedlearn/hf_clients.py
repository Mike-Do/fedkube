from collections import OrderedDict
import warnings
import json

import flwr as fl
import torch
import numpy as np

import random
from torch.utils.data import DataLoader


from datasets import load_dataset
from evaluate import load as load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

from datasets import DatasetDict

from tqdm import tqdm
import pandas as pd
from datasets import Dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--total_clients", type=int)
parser.add_argument("--client_idx", type=int)
args = parser.parse_args()
total_clients = args.total_clients
client_idx = args.client_idx

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda")
CHECKPOINT = "google/bert_uncased_L-4_H-256_A-4"  # transformer model checkpoint

INPUT_COL = ["summary", "review"]
NUM_LABELS = 11

def load_data():
    """Load Amazon dataset"""
    df = pd.read_json("/users/zyong2/data/zyong2/class/data/external/reviews.jsonl", lines=True)
    ### select blocks
    blocks = {
            0: "0-0",
            1: "1-0",
            2: "2-0",
            3: "3-0",
            4: "4-0",
            5: "5-0",
            6: "6-0",
            7: "7-0",
            8: "8-0",
            9: "9-0",
            10: "10-0",
    }
    block = blocks[client_idx]
    # select df with group_key of 0-0
    train_df = df[df.group_key == block]

    test_blocks = set(['11-98','43-39','8-41','5-74','14-41','31-5','19-64','29-36','1-52','31-36','45-72','2-64','6-83','11-39','17-14','37-72','12-84','34-31','24-8','30-69','1-62','22-66','12-10','47-61','7-77','18-95','36-58','47-70','42-19','12-57','45-86','43-64','48-90','26-10','30-2','20-65','42-79','40-9','43-97','4-19','41-40','31-54','6-48','22-73','14-16','28-74','14-77','46-77','11-78','36-17','44-1','42-45','1-4','23-58','25-77','48-49','13-39','7-66','29-26','34-78','28-19','17-39','32-41','36-52','48-94','12-67','27-52','33-96','5-3','27-41','11-82','44-41','16-65','41-96','13-7','2-82','18-10','1-43','41-31','38-29','34-87','11-16','40-34','11-12','2-72','16-24','18-98','20-35','26-68','40-84','36-21','14-90','46-8','13-10','10-91','35-35','34-21','47-21','34-37','19-44','17-53','20-80','7-88','46-10','16-91','42-76','41-83','22-32','46-86','25-40','1-20','47-60','40-94','21-68','21-54','31-20','47-85','43-2','48-73','31-88','27-33','48-22','21-72','48-52','5-49','43-50','10-21','30-68','8-4','20-55','14-8','8-66','36-56','17-44','13-27','41-54','26-76','49-7','28-58','38-89','29-23','1-47','4-89','46-36','3-35','24-10','46-27','42-39','43-68','4-48','12-13','41-59','46-71','46-54','10-98','15-40','6-74','3-72','0-47','47-49','14-30','5-84','40-95','14-69','24-93','36-1','9-95','6-2','49-5','19-17','29-51','47-52','46-42','13-65','30-56','33-15','43-36','33-83','46-72','23-81','19-29','0-53','18-91','1-42','36-98','11-10','39-65','0-2','5-45','32-37','45-98','6-94','1-36','19-46','0-36','33-78','5-14','25-11','48-66','17-79'])
    test_df = df[df.group_key.isin(test_blocks)]

    ###

    train_df = train_df.drop(columns=["group_key", "data_idx", "date", "user"])
    test_df = test_df.drop(columns=["group_key", "data_idx", "date", "user"])

    # df["label"] = df.apply(lambda x: int2str(x.label), axis=1)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("train_dataset has size", len(train_dataset))
    print("test_dataset has size", len(test_dataset))

    test_valid = test_dataset.train_test_split(test_size=0.75)
    
    raw_datasets = DatasetDict({
        'train': train_dataset,
        'test': test_valid['test'],
        'valid': test_valid['train']})

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(example):
        return tokenizer(example["summary"] + example["review"], truncation=True, max_length=512)

    # random 100 samples

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=False)

    tokenized_datasets = tokenized_datasets.remove_columns("summary")
    tokenized_datasets = tokenized_datasets.remove_columns("review")
    
    tokenized_datasets = tokenized_datasets.rename_column("category", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns("rating")
    tokenized_datasets = tokenized_datasets.remove_columns('__index_level_0__')


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=1,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader

def load_data_imdb():
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # random 100 samples
    population = random.sample(range(len(raw_datasets["train"])), 100)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(population)
    tokenized_datasets["test"] = tokenized_datasets["test"].select(population)

    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()

    print("in train()")
    for epoch in range(epochs):
        running_loss = 0
        for i, batch in tqdm(enumerate(trainloader)):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            if i % 100 == 0:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


def main():
    print("Model Loaded")
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=NUM_LABELS
    ).to(DEVICE) 

    print("Data Loaded")
    trainloader, testloader = load_data()
    print("Done")

    print("Privacy Config loaded")
    client_privacy_configs = None
    privacy_configs_csv = pd.read_csv("/users/zyong2/data/zyong2/class/scripts/clients_budget_demand.csv")
    # rename columns: Budget -> privacy_budget, Demand -> demand
    privacy_configs_csv = privacy_configs_csv.rename(columns={"Budget": "privacy_budget", "Demand": "demand"})
    # get row of f"Client_{client_idx+1}"
    _client_privacy_configs = privacy_configs_csv[privacy_configs_csv["Client"] == f"Client_{client_idx+1}"]
    client_privacy_configs = {}
    client_privacy_configs["privacy_budget"] = _client_privacy_configs["privacy_budget"].values[0]
    client_privacy_configs["demand"] = _client_privacy_configs["demand"].values[0]

    # Flower client
    class AmazonClient(fl.client.NumPyClient):
        def __init__(self, client_privacy_configs):
            fl.client.NumPyClient.__init__(self)
            assert client_privacy_configs is not None
            self.privacy_budget = int(client_privacy_configs["privacy_budget"])
            self.demand = int(client_privacy_configs["demand"])

        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
        
        def get_properties(self, config):
            return {"privacy_budget": self.privacy_budget, "demand": self.demand}

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("(fit) Training Started...")
            train(net, trainloader, epochs=3)
            print("(fit) Training Finished.")

            print("(fit) Deduct privacy budget...")
            self.privacy_budget -= self.demand

            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            print("🔥 (evaluate) Accuracy: ", accuracy)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    ipnip = "0.0.0.0"
    ipnport = "8000"
    fl.client.start_numpy_client(server_address=f"{ipnip}:{ipnport}", client=AmazonClient(client_privacy_configs=client_privacy_configs))


if __name__ == "__main__":
    main()
