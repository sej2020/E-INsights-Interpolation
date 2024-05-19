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


# NOTES FOR LATER HYPERPARAMETER TUNING::
# check why I'm using these scalers
# can initialize with random hidden state or 0s or 1s
# can add dropout
# can adjust minibatch size
# can adjust learning rate
# can adjust number of epochs
# can adjust number of layers
# can adjust optimizer
# can add lr scheduler
# can clip gradients

import torch
from src.models.LSTMs import LSTM, BidirectionalLSTM
from src.config.trainer_configs import TrainerConfig, MLPTrainerConfig
import multiprocessing
import time


def train_lstm(dataset: str, hp_layers: int, hp_lr: float, hp_window_size: int, hp_epochs: int, hp_batch_size: int, run_name: str = ''):

    lstm = LSTM(input_size = 9, n_layers = hp_layers, window_size = hp_window_size)

    lstm_trainer_config = TrainerConfig(
        dataset_path = dataset,
        train_set_size = 0.75,
        n_epochs = hp_epochs,
        batch_size = hp_batch_size,
        lr = hp_lr,
        start_idx=None,
        stop_idx=None,
        optimizer = torch.optim.Adam,
        logging_dir = "logs",
        logging_steps_ratio = 0.1,
        save_steps_ratio = 0.1,
        lr_scheduler = True,
        resume_from_checkpoint = False,
        checkpoint_path = None,
        run_name = dataset.split("/")[-1].split(".")[0] + f"_lay-{hp_layers}_lr-{hp_lr}_ws-{hp_window_size}",
    )

    lstm.train(lstm_trainer_config)


def train_bilstm(
    dataset_path: str,
    lstm_input_size: int,
    lstm1_layers: int, 
    lstm2_layers: int,
    lstm_window_size: int, 
    mlp_epochs: int,
    mlp_lr: float,
    mlp_ablation_max: int, 
    run_name: str = '',
    lstm1_cpt_file: str = None,
    lstm2_cpt_file: str = None,
    lstm_epochs: int = 60,
    lstm_lr: float = 0.001,
    lstm_batch_size: int = 16,
    ):
    """
    Important Stuff: Provide lstm1_cpt_file and lstm2_cpt_file if you want to use pretrained models. If you want to train from scratch,
        leave them with value of None and set lstm_epochs, lstm_lr, lstm_batch_size to the values you want to use for training.
    """

    bidi = BidirectionalLSTM(input_size = lstm_input_size, lstm1_layers = lstm1_layers, lstm2_layers = lstm2_layers, window_size = lstm_window_size)

    mlp_trainer_config = MLPTrainerConfig(
        n_epochs=mlp_epochs,
        lr=mlp_lr,
        optimizer= torch.optim.Adam,
        lr_scheduler=True,
        ablation_max=mlp_ablation_max,
        lstm1_cpt_file=lstm1_cpt_file,
        lstm2_cpt_file=lstm2_cpt_file,
    )

    trainer_config = TrainerConfig(
        dataset_path = dataset_path,
        train_set_size = 0.75,
        n_epochs = lstm_epochs,
        batch_size = lstm_batch_size,
        lr = lstm_lr,
        start_idx=None,
        stop_idx=None,
        optimizer = torch.optim.Adam,
        logging_dir = "logs",
        logging_steps_ratio = 0.25,
        save_steps_ratio = 0.25,
        lr_scheduler = True,
        resume_from_checkpoint = False,
        checkpoint_path = None,
        run_name = f"{run_name}_lr{mlp_lr}",
        mlp_trainer_config=mlp_trainer_config
    )

    bidi.train(trainer_config)


if __name__ == "__main__":


    ## Unidirectional LSTM Hyperparam Tuning ##

    # train_lstm(2, 1e-2, 20, 100, 2, run_name_prefix='test_')
    # start = time.time()
    # with multiprocessing.Pool() as pool:
    #     pool.starmap_async(train_lstm, 
    #                  [ (ds, lay, lr, ws, epo, bs) 
    #                   for ds in ["data/min_av/OptoMMP-Oct23/training_M00_PhA.csv",
    #                              "data/min_av/OptoMMP-Oct23/training_M02_PhC.csv"]
    #                   for lay in [4, 6, 9]
    #                   for lr in [.0005] 
    #                   for ws in [10, 30] 
    #                   for epo in [60]
    #                   for bs in [16] ]
    #                   )
    # with multiprocessing.Pool() as pool:
    #     pool.starmap(train_lstm, 
    #                  [ (ds, lay, lr, ws, epo, bs) 
    #                   for ds in ["data/min_av/amatrol-Mar24/training_CNC_VF5v2.csv",
    #                              "data/min_av/amatrol-Mar24/training_HVAC_RTUv2.csv"]
    #                   for lay in [6, 9, 12]
    #                   for lr in [.0005, .00005] 
    #                   for ws in [10, 20, 40] 
    #                   for epo in [80]
    #                   for bs in [16] ]
    #                   )
    # print(f"Time taken: {time.time() - start}", flush=True)
        

    ## Bidirectional LSTM Testing ##
    with multiprocessing.Pool() as pool:
        pool.starmap(train_bilstm,
        [
            (
                "data/min_av/amatrol-Mar24/final_CNC/training_CNC_VF5v2.csv", 
                18, 6, 6, 10, 30, .005, 20, "biLSTM_CNC_VF5v2", 
                "logs/training_CNC_VF5v2/lay-6_lr-0.0005_ws-10/checkpoints/checkpt_e59.pt", 
                "logs/training_CNC_VF5v2/lay-6_lr-0.001_ws-10_rev/checkpoints/checkpt_e9.pt"
                ),
            (
                "data/min_av/amatrol-Mar24/final_HVAC/training_HVAC_RTUv2.csv", 
                18, 6, 6, 20, 30, .005, 20, "biLSTM_HVAC_RTUv2", 
                "logs/training_HVAC_RTUv2/lay-6_lr-0.0005_ws-20/checkpoints/checkpt_e59.pt", 
                "logs/training_HVAC_RTUv2/lay-6_lr-0.001_ws-20_rev/checkpoints/checkpt_e59.pt"
                ),
            (   
                "data/min_av/OptoMMP-Oct23/final_M00/training_M00_PhA.csv", 
                9, 3, 4, 10, 30, .005, 20, "biLSTM_M00_PhA", 
                "logs/training_M00_PhA/lay-3_lr-0.001_ws-10_/checkpoints/checkpt_e59.pt",
                "logs/training_M00_PhA/lay-4_lr-0.001_ws-10_rev/checkpoints\checkpt_e29.pt"
                ),
            (   
                "data/min_av/OptoMMP-Oct23/final_M02/training_M02_PhC.csv", 
                9, 4, 3, 10, 30, .005, 20, "biLSTM_M02_PhC",
                "logs/training_M02_PhC/lay-4_lr-0.001_ws-10_/checkpoints/checkpt_e59.pt",
                "logs/training_M02_PhC/lay-3_lr-0.001_ws-10_rev/checkpoints/checkpt_e59.pt"
                ),
            ])

    