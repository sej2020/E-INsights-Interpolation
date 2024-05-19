"""
For training models. Edit the config object to change the parameters of the training.

Typical usage example:
```bash
>>> python -m src.actions.train
```
And to view the training progress, run the following command in the terminal:
```bash
>>> tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""

import torch
from src.models.LSTMs import BidirectionalLSTM
from src.config.trainer_configs import TrainerConfig, BidiMLPTrainerConfig
import argparse
parser = argparse.ArgumentParser("training bilstm")

parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--train_test_split", type=float, default=0.75)
parser.add_argument("--start_idx", type=int, default=None)
parser.add_argument("--stop_idx", type=int, default=None)
parser.add_argument("--logging_steps_ratio", type=float, default=0.1)
parser.add_argument("--save_steps_ratio", type=float, default=0.1)

parser.add_argument("--lstm_input_size", type=int, required=True, help="Number of features in the dataset. Need to know regardless of training LSTMs from scratch or not.")
parser.add_argument("--lstm_f_n_layers", type=int, required=True, help="Need to know regardless of training LSTMs from scratch or not.")
parser.add_argument("--lstm_b_n_layers", type=int, required=True, help="Need to know regardless of training LSTMs from scratch or not.")
parser.add_argument("--lstm_window_size", type=int, required=True, help="Need to know regardless of training LSTMs from scratch or not.")
parser.add_argument("--mlp_n_epochs", type=int, default=60)
parser.add_argument("--mlp_lr", type=float, default=0.001)
parser.add_argument("--mlp_ablation_max", type=int, default=50)
parser.add_argument("--mlp_lr_scheduler", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--lstm_f_cpt_file", type=str, default=None)
parser.add_argument("--lstm_b_cpt_file", type=str, default=None)

parser.add_argument("--training_lstms", action=argparse.BooleanOptionalAction, default=True, 
    help="Whether to train LSTM models from scratch or not. If False, the model will load the checkpoint files provided in lstm_f_cpt_file and lstm_b_cpt_file.")
parser.add_argument("--lstm_n_epochs", type=int, default=60, help="Only used if training_lstms is True.")
parser.add_argument("--lstm_lr", type=float, default=0.001, help="Only used if training_lstms is True.")
parser.add_argument("--lstm_batch_size", type=int, default=16, help="Only used if training_lstms is True.")
parser.add_argument("--lstm_lr_scheduler", action=argparse.BooleanOptionalAction, default=True, help="Only used if training_lstms is True.")
parser.add_argument("--run_name", type=str, default=None)

args = parser.parse_args()

if args.run_name is None:
    NAME = f"_lay-{args.lstm_f_n_layers}-{args.lstm_b_n_layers}_lr-{args.lstm_lr}_ws-{args.lstm_window_size}"
else:
    NAME = args.run_name

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/debug"
else:
    LOGGING_DIR = f"logs/{NAME}"

bidi = BidirectionalLSTM(input_size = args.lstm_input_size, lstm_f_layers = args.lstm_f_n_layers, lstm_b_layers = args.lstm_b_n_layers, window_size = args.lstm_window_size)

mlp_trainer_config = BidiMLPTrainerConfig(
        n_epochs = args.mlp_n_epochs,
        lr = args.mlp_lr,
        optimizer= torch.optim.Adam,
        lr_scheduler = args.mlp_lr_scheduler,
        ablation_max = args.mlp_ablation_max,
        lstm_training = args.training_lstms,
        lstm_f_cpt_file = args.lstm_f_cpt_file,
        lstm_b_cpt_file = args.lstm_b_cpt_file
    )

trainer_config = TrainerConfig(
        dataset_path = args.dataset,
        train_set_size = args.train_test_split,
        n_epochs = args.lstm_n_epochs,
        batch_size = args.lstm_batch_size,
        lr = args.lstm_lr,
        start_idx = args.start_idx,
        stop_idx = args.stop_idx,
        optimizer = torch.optim.Adam,
        logging_dir = LOGGING_DIR,
        logging_steps_ratio = args.logging_steps_ratio,
        save_steps_ratio = args.save_steps_ratio,
        lr_scheduler = args.lstm_lr_scheduler,
        resume_from_checkpoint = False,
        checkpoint_path = None,
        run_name = NAME,
        mlp_trainer_config=mlp_trainer_config
    )

bidi.train(trainer_config)
print(f"Training complete for {NAME}", flush=True)
print(f"Logs saved at {LOGGING_DIR}: don't forget to clean up the logging directory when you're done", flush=True)