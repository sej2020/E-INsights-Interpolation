"""
Searching over hyperparameters for finetuning the TEMPO model on energy insights data.

Typical usage example:
```bash
>>> python -m src.actions.search_hp --device cuda --n_runs 8
```
"""

import wandb
wandb.login()
from src.models.TempoGPT import TempoGPT
import argparse

parser = argparse.ArgumentParser("hyperparameter search TEMPO")

# Training Meta-parameters
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--n_runs", type=int, default=1)

# Training parameters
parser.add_argument("--dataset_path_lst", nargs="*", type=str, default=None)
parser.add_argument("--train_test_size", type=float, default=0.75)
parser.add_argument("--logging_frequency", type=float, default=0.2)
parser.add_argument("--saving_frequency", type=float, default=0.01)
parser.add_argument("--disable_tqdm", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")

args = parser.parse_args()

if args.run_name is None:
    NAME = 'hp_run_'
else:
    NAME = args.run_name

if args.debug:
    NAME = 'debug'
    LOGGING_DIR = f"logs/hp_search"
else:
    LOGGING_DIR = f"logs/hp_search"

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

model = TempoGPT(device=args.device)

hp_search_trainer_config = {
    'dataset_path_lst': {'value': DATASETS},
    'train_set_size': {'value': args.train_test_size},
    'n_epochs': {
        'values': [50, 150, 300]
        },
    'batch_size': {
        'values': [32, 64, 128, 256]
        },
    'lr': {
        'values': [0.0001, 0.0005, 0.001, 0.005, 0.01]
        },
    'optimizer': {
        'values': ['adam', 'adamw']
        },
    'logging_dir': {'value': LOGGING_DIR},
    'logging_frequency': {'value': args.logging_frequency},
    'saving_frequency': {'value': args.saving_frequency},
    'lr_scheduler': {
        'values': [True, False]
        },
    'disable_tqdm': {'value': args.disable_tqdm},
    'resume_from_checkpoint': {'value': args.resume_from_checkpoint},
    'checkpoint_path': {'value': args.checkpoint_path},
    'run_name': {'value': NAME},
    'batch_stride': {
        'values': [7, 8, 16, 64]
        },
}

sweep_config = {
    'method': 'random' # 'grid'
}

sweep_config['parameters'] = hp_search_trainer_config

sweep_id = wandb.sweep(sweep_config, project="search-hp-TempoGPT")

wandb.agent(sweep_id, function=lambda: model.fine_tune(hp_search=True), count=args.n_runs)

print(f"Hyperparameter Search complete for {NAME}", flush=True)