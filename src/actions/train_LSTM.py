"""
For training model the LSTM model.

Typical usage example:
```bash
>>> python -m src.actions.train_LSTM --dataset data/processed/processed_data.csv --n_dataset_features 5 --n_layers 2 --lr 0.001 
...     --window_size 10 --n_epochs 60 --batch_size 16
```
And to view the training progress, run the following command in the terminal:
```bash
>>> tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""

import torch
from src.models.LSTMs import LSTM
from src.config.trainer_configs import TrainerConfig
import argparse
parser = argparse.ArgumentParser("training lstm")

# Training Meta-parameters
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

# Training parameters
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--train_test_size", type=float, default=0.75)
parser.add_argument("--n_epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--start_idx", type=int, default=None)
parser.add_argument("--stop_idx", type=int, default=None)
parser.add_argument("--logging_frequency", type=float, default=0.1)
parser.add_argument("--saving_frequency", type=float, default=0.1)
parser.add_argument("--lr_scheduler", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, default=False)

# Model parameters
parser.add_argument("--n_dataset_features", type=int, required=True)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--window_size", type=int, default=10)

args = parser.parse_args()

if args.name is None:
    NAME = f"LSTM_lay-{args.layers}_lr-{args.lr}_ws-{args.window_size}"
else:
    NAME = args.name

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/debug"
else:
    LOGGING_DIR = f"logs/{NAME}"


lstm = LSTM(input_size = args.n_dataset_features, n_layers = args.n_layers, window_size = args.window_size)

lstm_trainer_config = TrainerConfig(
        dataset_path = args.dataset,
        train_set_size = args.train_test_size,
        n_epochs = args.n_epochs,
        batch_size = args.batch_size,
        lr = args.lr,
        start_idx = args.start_idx,
        stop_idx = args.stop_idx,
        optimizer = torch.optim.Adam,
        logging_dir = LOGGING_DIR,
        logging_frequency = args.logging_frequency,
        saving_frequency = args.saving_frequency,
        lr_scheduler = args.lr_scheduler,
        resume_from_checkpoint = args.resume_from_checkpoint,
        checkpoint_path = args.checkpoint_path,
        run_name = NAME,
        reverse = args.reverse
    )

lstm.train(lstm_trainer_config)
print(f"Training complete for {NAME}", flush=True)
print(f"Logs saved at {LOGGING_DIR}: don't forget to clean up the logging directory when you're done", flush=True)