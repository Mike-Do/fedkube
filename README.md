# Fedkube

Extending PrivateKube with Federated Learning.

## Getting Started

### Obtaining the Data

1. `cd code`
2. `mkdir data`
3. `touch reviews.h5`
4. `python3 dataset.py getmini`
5. `python3 convert_h5dataset_to_text.py`

### Run the Federated Learning with Privacy Budget

1. `cd example`
2. `python3 server.py`
3. Open new terminal and `python3 client.py`
4. Open new terminal and `python3 client.py`
