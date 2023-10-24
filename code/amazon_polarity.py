import torch.nn as nn
from torch.autograd import Variable
import torch
from opacus.layers import DPLSTM
from absl import logging


class LSTMClassifier(nn.Module):
    # https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/load_data.py
    # + Opacus example
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_length,
        weights,
        dropout,
        dp,
    ):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.embedding = nn.Embedding(
            vocab_size, embedding_length
        )  # Initializing the look-up table.
        self.embedding.weight = nn.Parameter(
            weights, requires_grad=False
        )  # Assigning the look-up table to the pre-trained GloVe word embedding.
        if dp:
            logging.info("Building DPLSTM.")
            self.lstm = DPLSTM(embedding_length, hidden_size, batch_first=True)
        else:
            logging.info("Building non-DP LSTM.")
            self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input, hidden):
        input_emb = self.embedding(input)
        lstm_out, _ = self.lstm(input_emb, hidden)
        if not self.dropout is None:
            lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output[:, -1]

    def init_hidden(self, batch_size):
        return (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size),
        )


def main(argv):
    start_time = time.time()

    # Convert flags for the epsilon = -1 shortcut
    if FLAGS.dp and FLAGS.epsilon < 0 and FLAGS.noise < 0:
        FLAGS.dp = False

    # No multiprocessing for large datasets (save RAM)
    if FLAGS.n_blocks > 50_000:
        logging.info(f"Large dataset, we use a single thread for the loader.")
        FLAGS.n_workers = 0

    # Build the dataset, either event level or user level
    train_data, test_data, valid_data, text_field = build_split_dataset()
    logging.info(
        f"Number of samples for training: {len(train_data)}, validation: {len(valid_data)} and testing: {len(test_data)}"
    )

    # Adapt the batch size and the virtual step size, unless it has been specified manually
    if FLAGS.dp and FLAGS.adaptive_batch_size and FLAGS.virtual_batch_multiplier <= 0:
        FLAGS.batch_size, FLAGS.virtual_batch_multiplier = compute_optimal_batch_size(
            FLAGS.batch_size, len(train_data)
        )
        logging.info(
            f"Using real batch {FLAGS.batch_size} with multiplier {FLAGS.virtual_batch_multiplier}"
        )
    if not FLAGS.dp:
        FLAGS.batch_size = FLAGS.non_dp_batch_size

    # Prepare the model and optimizer
    model = build_model(text_field).to(FLAGS.device)

    logging.info(f"Number of trainable parameters: {count_parameters(model)}")

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, eps=1e-8)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

    # train_it = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=2048,
    #     shuffle=False,
    #     num_workers=FLAGS.n_workers,
    #     drop_last=False,
    # )
    # counts = {}
    # for i in range(11):
    #     counts[i] = 0
    # for b in train_it:
    #     for cat in b[:, 3]:
    #         counts[int(cat)] += 1
    # s = sum(counts.values())
    # for cat, count in counts.items():
    #     counts[cat] = count / s
    # logging.info(counts)

    if FLAGS.task == "sentiment":
        criterion = nn.BCEWithLogitsLoss().to(FLAGS.device)
        accuracy_fn = binary_accuracy

    # automotive: 0.03036145803296712
    # books: 0.41258122723567553
    # cds: 0.012897189083383703
    # clothing: 0.2025265712144095
    # games: 0.031613111956201506
    # groceries: 0.01949595483554337
    # home: 0.119920985593197
    # movies: 0.0484712255807162
    # pets: 0.03665525816121956
    # sports: 0.04961580907019007
    # tools: 0.035861209236496445

    elif FLAGS.task == "product":
        # criterion = nn.CrossEntropyLoss(
        #     weight=torch.Tensor(
        #         [0.05, 0.035, 0.03, 0.035, 0.05, 0.02, 0.12, 0.01, 0.03, 0.20, 0.41]
        #     )
        # )
        criterion = nn.CrossEntropyLoss()
        accuracy_fn = multiclass_accuracy

    # Plug Opacus if DP training is activated
    if FLAGS.dp:
        if FLAGS.noise >= 0:
            logging.info(f"User-provided noise: {FLAGS.noise}.")
        else:
            logging.info("Computing noise for the given parameters.")
            FLAGS.noise = compute_noise_from_target_epsilon(
                target_epsilon=FLAGS.epsilon,
                target_delta=FLAGS.delta,
                epochs=FLAGS.n_epochs,
                batch_size=FLAGS.batch_size * FLAGS.virtual_batch_multiplier
                if FLAGS.virtual_batch_multiplier > 0
                else FLAGS.batch_size,
                dataset_size=len(train_data),
                alphas=ALPHAS,
            )
            logging.info(f"Noise computed from RDP budget: {FLAGS.noise}.")

        # NOTE: when user-level DP is activated, the training dataset __len__ method returns
        # the number of users, and the DataLoader calls the batch-of-user method that overrides
        # the regular __getitem__ method

        # WARNING: fishy non-DP adaptive clipping
        privacy_engine = opacus.PrivacyEngine(
            module=model,
            batch_size=FLAGS.batch_size * FLAGS.virtual_batch_multiplier
            if FLAGS.virtual_batch_multiplier > 0
            else FLAGS.batch_size,
            sample_size=len(train_data),
            alphas=ALPHAS,
            noise_multiplier=FLAGS.noise,
            max_grad_norm=FLAGS.max_grad_norm,
            experimental=bool(FLAGS.dynamic_clipping),
            clipping_method=FLAGS.dynamic_clipping,
            clip_per_layer=bool(FLAGS.per_layer_clipping),
        )
        privacy_engine.attach(optimizer)

    # Do the actual training
    t = time.time()
    (
        training_loss_epochs,
        training_accuracy_epochs,
        validation_loss_epochs,
        validation_accuracy_epochs,
    ) = train_validate(
        train_data, valid_data, model, optimizer, criterion, accuracy_fn, scheduler
    )
    training_time = time.time() - t

    if FLAGS.dp:
        epsilon_consumed, best_alpha = optimizer.privacy_engine.get_privacy_spent(
            FLAGS.delta
        )
        epsilon_consumed = float(epsilon_consumed)
        best_alpha = float(best_alpha)
        logging.info(f"Best alpha: {best_alpha}")
        rdp_epsilons_consumed = (
            optimizer.privacy_engine.get_renyi_divergence()
            * optimizer.privacy_engine.steps
        ).tolist()

        logging.info(f"RDP budget consumed: {rdp_epsilons_consumed} for orders.")

        # Identical to planned budget when we don't have early stopping
        # rdp_epsilon_planned = compute_rdp_sgm(
        #     epochs=FLAGS.n_epochs,
        #     batch_size=FLAGS.batch_size * FLAGS.virtual_batch_multiplier
        #     if FLAGS.virtual_batch_multiplier > 0
        #     else FLAGS.batch_size,
        #     dataset_size=len(train_data),
        #     noise=FLAGS.noise,
        #     alphas=ALPHAS,
        # )
        # logging.info(f"Planned RDP budget: {rdp_epsilon_planned}")
    else:
        epsilon_consumed = None
        rdp_epsilons_consumed = None
        best_alpha = None

    # Evaluate the model (non-DP evaluation here)
    testing_size = len(test_data)
    test_iterator = torch.utils.data.DataLoader(
        test_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.n_workers,
        drop_last=False,
    )
    final_loss, final_accuracy = evaluate(model, test_iterator, criterion, accuracy_fn)

    # Collect the metrics and the logs
    logs = {
        "training_time": training_time,
        "total_time": time.time() - start_time,
        "test_size": testing_size,
        "n_trainable_parameters": count_parameters(model),
    }

    # Update the logs with the training data
    if isinstance(train_data, UserTimeLevelDataset):
        logs["train_size"] = train_data.get_n_events()
        logs["n_train_users"] = len(train_data)
    else:
        logs["train_size"] = len(train_data)

    logs.update(
        flags_to_dict(dataset_args, model_args, training_args)
    )  # Dump the configuration flags
    metrics = {
        "accuracy": final_accuracy,
        "training_loss_epochs": training_loss_epochs,
        "training_accuracy_epochs": training_accuracy_epochs,
        "validation_loss_epochs": validation_loss_epochs,
        "validation_accuracy_epochs": validation_accuracy_epochs,
        "loss": final_loss,
        "epsilon": epsilon_consumed,
        "target_epsilon": FLAGS.epsilon,
        "alphas": ALPHAS,
        "rdp_epsilons": rdp_epsilons_consumed,
        "best_alpha": best_alpha,
        # "dataset_files": os.listdir(FLAGS.dataset_dir),
    }

    # Save or logging.info the outputs
    # Useless to separate for our experiments
    if FLAGS.metrics_path != "":
        save_yaml(FLAGS.metrics_path, metrics)
        logging.info(f"Saved metrics: {FLAGS.metrics_path}")
    else:
        logging.info("Metrics not saved but concatenated to the logs.")
        logs.update(metrics)

    if FLAGS.log_path != "":
        save_yaml(FLAGS.log_path, logs)
        logging.info(f"Saved logs: {FLAGS.log_path}")

    if FLAGS.model_path != "":
        save_model(FLAGS.model_path, model)
        logging.info(f"Saved model: {FLAGS.model_path}")

    logging.info(logs)
    logging.info(metrics)


if __name__ == "__main__":
    app.run(main)
