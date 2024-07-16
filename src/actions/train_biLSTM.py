"""
For training models. Edit the config object to change the parameters of the training.

Typical usage example:
```bash
>>> python -m src.actions.train_biLSTM --dataset data/processed/processed_data.csv --n_dataset_features 5 --n_layers 2 --lr 0.001 
...     --window_size 10 --n_epochs 60 --batch_size 16 --ablation_max 50 --grad_accumulation_steps 1 
...     --lstm_f_cpt_file path/to/lstm_f_checkpoint.pt --lstm_b_cpt_file path/to/lstm_b_checkpoint.pt
```
And to view the training progress, run the following command in the terminal:
```bash
>>> tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""

import torch
from src.models.LSTMs import BidirectionalLSTM
from src.config.trainer_configs import BidiMLPTrainerConfig
import argparse
parser = argparse.ArgumentParser("training bilstm")

# Training Meta-parameters
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

# Training the MLP parameters
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

# Bidirectional-specific parameters
parser.add_argument("--ablation_max", type=int, default=50)
parser.add_argument("--grad_accumulation_steps", type=int, default=1)
parser.add_argument("--lstm_f_cpt_file", type=str, default=None)
parser.add_argument("--lstm_b_cpt_file", type=str, default=None)

# Model parameters
parser.add_argument("--n_dataset_features", type=int, required=True)
parser.add_argument("--lstm_f_n_layers", type=int, default=2)
parser.add_argument("--lstm_b_n_layers", type=int, default=2)
parser.add_argument("--window_size", type=int, default=10)

args = parser.parse_args()

if args.run_name is None:
    NAME = f"biLSTM_lay-{args.layers}_lr-{args.lr}_ws-{args.window_size}"
else:
    NAME = args.run_name

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/debug"
else:
    LOGGING_DIR = f"logs/{NAME}"

bidi = BidirectionalLSTM(input_size = args.n_dataset_features, lstm_f_layers = args.lstm_f_n_layers, lstm_b_layers = args.lstm_b_n_layers, window_size = args.window_size)

trainer_config = BidiMLPTrainerConfig(
        dataset_path = args.dataset,
        train_set_size = args.train_test_size,
        n_epochs = args.lstm_n_epochs,
        batch_size = args.lstm_batch_size,
        lr = args.lstm_lr,
        start_idx = args.start_idx,
        stop_idx = args.stop_idx,
        optimizer = torch.optim.Adam,
        logging_dir = LOGGING_DIR,
        logging_frequency = args.logging_frequency,
        saving_frequency = args.saving_frequency,
        lr_scheduler = args.lstm_lr_scheduler,
        resume_from_checkpoint = False,
        checkpoint_path = None,
        run_name = NAME,
        ablation_max = args.ablation_max,
        grad_accumulation_steps = args.grad_accumulation_steps,
        lstm_f_cpt_file = args.lstm_f_cpt_file,
        lstm_b_cpt_file = args.lstm_b_cpt_file
    )

bidi.train(trainer_config)
print(f"Training complete for {NAME}", flush=True)
print(f"Logs saved at {LOGGING_DIR}: don't forget to clean up the logging directory when you're done", flush=True)