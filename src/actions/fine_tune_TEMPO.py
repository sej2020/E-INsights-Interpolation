"""
For fine tuning the TEMPO model on E-INsights data.

Typical usage example:
```bash
>>> python -m src.actions.fine_tune_TEMPO
```
And to view the training progress, run the following command in the terminal:
```bash
>>> tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""

import torch
from src.models.TempoGPT import TempoGPT
from src.config.trainer_configs import TrainerConfig
import argparse
import datetime
parser = argparse.ArgumentParser("training TEMPO")


# Training Meta-parameters
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

# Training parameters
parser.add_argument("--dataset_path_lst", nargs="*", type=str, default=None)
parser.add_argument("--train_test_size", type=float, default=0.75)
parser.add_argument("--n_epochs", type=int, default=60)
parser.add_argument("--batch_stride", type=int, default=16)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--logging_frequency", type=float, default=0.1)
parser.add_argument("--saving_frequency", type=float, default=0.1)
parser.add_argument("--lr_scheduler", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")

args = parser.parse_args()

if args.run_name is None:
    NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
else:
    NAME = args.run_name

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/debug"
else:
    LOGGING_DIR = f"logs"

default_dataset_path_lst = [
    "data/high_var_oct16/train/high_var_train.csv", 
    "data/min_av/amatrol-Mar24/train_CNC/training_CNC_VF5v2.csv",
    "data/min_av/amatrol-Mar24/train_HVAC/training_HVAC_RTUv2.csv",
    "data/min_av/OptoMMP-Oct23/train_M00/training_M00_PhA.csv",
    "data/min_av/OptoMMP-Oct23/train_M02/training_M02_PhC.csv"]

if args.dataset_path_lst is None:
    DATASETS = default_dataset_path_lst
else:
    DATASETS = args.dataset_path_lst

model = TempoGPT()

tempo_trainer_config = TrainerConfig(
    dataset_path_lst = DATASETS,
    train_set_size = args.train_test_size,
    n_epochs = args.n_epochs,
    batch_size = args.batch_size,
    lr = args.lr,
    optimizer = torch.optim.Adam,
    logging_dir = LOGGING_DIR,
    logging_frequency = args.logging_frequency,
    saving_frequency = args.saving_frequency,
    lr_scheduler = args.lr_scheduler,
    resume_from_checkpoint = args.resume_from_checkpoint,
    checkpoint_path = args.checkpoint_path,
    run_name = NAME,
    batch_stride=args.batch_stride
    )

print("Starting training", flush=True)
model.fine_tune(tempo_trainer_config)
print(f"Training complete for {NAME}", flush=True)
print(f"Logs saved at {LOGGING_DIR}: don't forget to clean up the logging directory when you're done", flush=True)