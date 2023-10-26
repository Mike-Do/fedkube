from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import cifar
import amazon_polarity
import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedLSTMClient(fl.client.NumPyClient):
    """Flower client implementing Amazon Review classification using
    PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def build_split_dataset():
    block_dir = tempfile.mkdtemp()
    test_block_dir = tempfile.mkdtemp()

    if FLAGS.dataset_dir[0:5] == "gs://":
        os.system(
            "gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS"
        )
        fs = gcsfs.GCSFileSystem(
            project=os.get_env("GCP_PROJECT"), token="google_default"
        )  # Get the local Gcloud token
        logging.info("Listing bucket files.")
        all_blocks = list(
            map(
                lambda blob: os.path.basename(blob["name"]),
                fs.listdir(FLAGS.dataset_dir),
            )
        )
        logging.info(f"Got {len(all_blocks)} blocks.")
        logging.warning(f"The evaluation set is not fixed.")
    elif FLAGS.dataset_dir == "":
        logging.info("Listing the block names.")
        all_blocks = list(load_yaml(FLAGS.block_counts).keys())
    else:
        all_blocks = os.listdir(FLAGS.dataset_dir)

    logging.info(f"Selecting {FLAGS.n_blocks_test} test blocks (fixed randomness).")
    test_blocks = np.random.choice(all_blocks, FLAGS.n_blocks_test, replace=False)

    for tb in test_blocks:
        all_blocks.remove(tb)

    # Use every user to the maximum.
    def sort_by_user(block_name):
        if block_name.endswith(".h5"):
            block_name = block_name[: -len(".h5")]
        name = block_name.split("-")
        user_slice = int(name[1])
        return user_slice

    logging.info(
        f"Selecting as few users as possible.\n Pseudorandom and deterministic (hashed user ids)."
    )
    selected_blocks = sorted(all_blocks, key=sort_by_user)[0 : FLAGS.n_blocks]

    if FLAGS.dataset_dir[0:5] == "gs://":
        pool = Pool()

        bucket_path = FLAGS.dataset_dir

        def download_datasource(block_name):
            block_path = os.path.join(bucket_path, block_name)
            dest = os.path.join(block_dir, block_name)
            os.system(f"gsutil cp {block_path} {dest}")
            return

        logging.warning("Downloading the blocks in parallel.")
        b = pool.map(download_datasource, selected_blocks)
        pool.close()
        pool.join()
        block_names = None
        test_block_names = None
    elif FLAGS.dataset_dir == "":
        block_dir = None
        test_block_dir = None
        block_names = selected_blocks
        test_block_names = test_blocks

    else:
        for b in selected_blocks:
            os.symlink(os.path.join(FLAGS.dataset_dir, b), os.path.join(block_dir, b))
        for b in test_blocks:
            os.symlink(
                os.path.join(FLAGS.dataset_dir, b), os.path.join(test_block_dir, b)
            )
        block_names = None
        test_block_names = None

    # Store for the logs
    FLAGS.dataset_dir = block_dir
    if not FLAGS.dataset_monofile:
        if FLAGS.model == "bert":
            from_h5 = DEFAULT_DATA_PATH.joinpath("reviews.h5")
        else:
            from_h5 = DEFAULT_DATA_PATH.joinpath("reviews_custom_vocab.h5")
    else:
        from_h5 = FLAGS.dataset_monofile

    if FLAGS.dp and FLAGS.user_level:
        train_data = UserTimeLevelDataset(
            blocks_dir=block_dir,
            timeframe=FLAGS.timeframe_days * 86400,
            from_h5=from_h5,
            block_names=block_names,
        )
    else:
        train_data = EventLevelDataset(
            blocks_dir=block_dir,
            from_h5=from_h5,
            block_names=block_names,
        )

    test_data = EventLevelDataset(
        blocks_dir=test_block_dir,
        from_h5=from_h5,
        block_names=test_block_names,
    )
    test_data, valid_data = test_data.split([0.75, 0.25])
    logging.info(f"Test size: {len(test_data)}\n Valid size: {len(valid_data)}")

    # Values from the preprocessing
    # (max text len doesn't matter here)
    text_field = torchtext.data.Field(
        batch_first=True,
        use_vocab=True,
        init_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        include_lengths=True,
    )
    build_public_vocab(
        text_field,
        max_size=FLAGS.vocab_size - 4,
        vectors=f"glove.6B.{FLAGS.embedding_dim}d",
        unk_init=torch.Tensor.normal_,
        vectors_cache=FLAGS.emb_path,
    )

    return train_data, test_data, valid_data, text_field


def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = models.LSTMClassifier(
        batch_size=FLAGS.batch_size,
        output_size=output_dim,
        hidden_size=FLAGS.hidden_dim,
        vocab_size=INPUT_DIM,
        embedding_length=FLAGS.embedding_dim,
        weights=word_embeddings,
        dropout=FLAGS.dropout,
        dp=FLAGS.dp,
    )
    model.to(DEVICE)

    train_data, test_data, valid_data, text_field = build_split_dataset()
    model = build_model(text_field).to(FLAGS.device)

    # trainloader, testloader, num_examples = cifar.load_data()

    # Start client
    client = FedLSTMClient(model, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()
