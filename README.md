# Fedkube

Extending PrivateKube with Federated Learning.

## Getting Started with Fedkube

### Obtaining the Data

1. `cd code`
2. `mkdir data`
3. `touch reviews.h5`
4. `python3 dataset.py getmini`
5. `python3 convert_h5dataset_to_text.py`

The instructions convert `reviews.h5` to `reviews.jsonl` data, which will be used by the Federated Learning code below.

### Run the Federated Learning with Privacy Budget

We tested the code on Brown's CCV compute cluster (`gpu-he`) with 6 V100 GPUs on Dec 8, 2023. Here's the following instructions:

1. Calls interactive 6 GPUs with 50GB CPU memory for 4 hours. We need the interactive mode for ease of launching server and clients.

`interact -q gpu-he -g 6 -m 50g -n 4 -t 4:00:00` 

2. 