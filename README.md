# Fedkube

Extending PrivateKube with Federated Learning.

## Getting Started with Fedkube

### Obtaining the Amazon Reviews Product Classication Data

1. `cd code`
2. `mkdir data`
3. `touch reviews.h5`
4. `python3 dataset.py getmini`
5. `python3 convert_h5dataset_to_text.py`

The instructions convert `reviews.h5` to `reviews.jsonl` data, which will be used by the Federated Learning code below.

### Run the Federated Learning with Privacy Budget

We tested the code on Brown's CCV compute cluster (`gpu-he`) with 6 V100 GPUs on Dec 8, 2023 using Python 3.9.0. Here's the following instructions:

1. Calls interactive 6 GPUs with 50GB CPU memory (for 4 hours). We need the interactive mode for ease of launching server and clients.

    `interact -q gpu-he -g 6 -m 50g -n 4 -t 4:00:00` 

2. Set up virtual environment. 

    ```
    python3 -m venv env
    pip3 install -r requirements.txt
    source env/bin/activate
    ```

3. Use `screen` to launch global server. Here the server would select 2 clients at each round of training and evaluation out of 10 total clients.

    ```bash
    screen -S server

    # inside server
    python3 fedlearn/dpfedavgfixed_hf_server.py \
        --num_to_sample 2 \
        --total_clients 10
    ```

3. 