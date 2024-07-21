"""
TEMPO model for time-series forecasting.
"""
import numpy as np
import pathlib
import pandas as pd
import pickle
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from statsmodels.tsa.seasonal import STL
from src.config.trainer_configs import TrainerConfig
import time
import src.models.backend.TEMPO as TEMPO

# class ToyModel:
#     def __init__(self) -> None:
#         self.lyr = torch.nn.Linear(336,96)
#     def __call__(self, x, itr, trend, season, noise, test=False):
#         return self.lyr(x.squeeze(2)).unsqueeze(2), None

class TempoGPT:
    """
    Time-series forecasting model using TEMPO - https://github.com/DC-research/TEMPO

    Attributes:
        x: evenly spaced values, potentially with missing values.
        y: values corresponding to x.
        config: configuration for TEMPO checkpoint model
        model: TEMPO model
        train_trend_stamp: trend stamp for training data
        train_seasonal_stamp: seasonal stamp for training data
        train_residual_stamp: residual stamp for training data
        val_trend_stamp: trend stamp for validation data
        val_seasonal_stamp: seasonal stamp for validation data
        val_residual_stamp: residual stamp for validation
        device: device to run the model on.
    """
    def __init__(self, device: str = "cpu"):
        """
        Initializes an instance of the TempoGPT class.

        Args:
            device: device to run the model on. Default is "cpu".
        """
        self.config = TempoConfig()
        # self.model = ToyModel()
        self.model = TEMPO.TEMPO(
           self.config,
           device=device 
        )
        self.model.load_state_dict(torch.load(self.config.best_model_path, map_location=torch.device(device)), strict=False)
        self.x = None
        self.y = None
        self.train_trend_stamp = None
        self.train_seasonal_stamp = None
        self.train_residual_stamp = None
        self.val_trend_stamp = None
        self.val_seasonal_stamp = None
        self.val_residual_stamp = None
        self.device = device


    def _stl_resolve(self, mode: str, data_train: np.ndarray = None, data_val: np.ndarray = None, dataset_path: str = None, units: str = "s"):
        """
        STL Global Decomposition. Sets the trend, seasonal, and residual stamps for the model.

        Args:
            mode: must be 'val' or 'train'
            data_train: training data sequences to decompose. must be shape [train_len, num_features]
            data_val: validation data sequences to decompose. must be shape [val_len, num_features]
            dataset_path: path to the dataset. Used to save/access STL decomposition results.
            units: unit of time for the sequence. Default is 's' for seconds.
        """
        if units == "s":
            period = 60
        elif units == "min":
            period = 60*24
        elif units == "h":
            period = 24
        else:
            raise NotImplementedError("Only seconds (s), minutes (min) and hours (h) are supported right now. Fix if getting this error.")

        val_trend_path = pathlib.Path(dataset_path).parent / "stl" / "val_trend.pkl"
        val_seasonal_path = pathlib.Path(dataset_path).parent / "stl" / "val_seasonal.pkl"
        val_residual_path = pathlib.Path(dataset_path).parent / "stl" / "val_residual.pkl"

        if val_trend_path.exists() and val_seasonal_path.exists() and val_residual_path.exists():
            with open(val_trend_path, 'rb') as f:
                val_trend_stamp = pickle.load(f)
                val_trend_stamp = val_trend_stamp.to(self.device)
            with open(val_seasonal_path, 'rb') as f:
                val_seasonal_stamp = pickle.load(f)
                val_seasonal_stamp = val_seasonal_stamp.to(self.device)
            with open(val_residual_path, 'rb') as f:
                val_residual_stamp = pickle.load(f)
                val_residual_stamp = val_residual_stamp.to(self.device)
        elif data_val is None:
            raise Exception("Existing validation STL decomposition files not found. Must provide validation data to perform decomposition.")
        else:
            val_stl_root = pathlib.Path(dataset_path).parent / "stl"
            val_stl_root.mkdir(parents=True, exist_ok=True)

            val_trend_stamp = torch.empty((data_val.shape[0], data_val.shape[1]), dtype=torch.float32, device=self.device)
            val_seasonal_stamp = torch.empty((data_val.shape[0], data_val.shape[1]), dtype=torch.float32, device=self.device)
            val_residual_stamp = torch.empty((data_val.shape[0], data_val.shape[1]), dtype=torch.float32, device=self.device)
            for feat_idx in range(data_val.shape[1]):
                res_val = STL(data_val[:, feat_idx], period=period).fit()
                val_trend_stamp[:, feat_idx] = torch.tensor(res_val.trend, dtype=torch.float32, device=self.device)
                val_seasonal_stamp[:, feat_idx] = torch.tensor(res_val.seasonal, dtype=torch.float32, device=self.device)
                val_residual_stamp[:, feat_idx] = torch.tensor(res_val.resid, dtype=torch.float32, device=self.device)

            with open(val_trend_path, 'wb') as f:
                pickle.dump(val_trend_stamp, f)
            with open(val_seasonal_path, 'wb') as f:
                pickle.dump(val_seasonal_stamp, f)
            with open(val_residual_path, 'wb') as f:
                pickle.dump(val_residual_stamp, f) 

        self.val_trend_stamp = val_trend_stamp # [val_len, num_features]
        self.val_seasonal_stamp = val_seasonal_stamp # .
        self.val_residual_stamp = val_residual_stamp # .


        if mode == "train":
            train_trend_path = pathlib.Path(dataset_path).parent / "stl" / "train_trend.pkl"
            train_seasonal_path = pathlib.Path(dataset_path).parent / "stl" / "train_seasonal.pkl"
            train_residual_path = pathlib.Path(dataset_path).parent / "stl" / "train_residual.pkl"

            if train_trend_path.exists() and train_seasonal_path.exists() and train_residual_path.exists():
                with open(train_trend_path, 'rb') as f:
                    train_trend_stamp = pickle.load(f)
                    train_trend_stamp = train_trend_stamp.to(self.device)
                with open(train_seasonal_path, 'rb') as f:
                    train_seasonal_stamp = pickle.load(f)
                    train_seasonal_stamp = train_seasonal_stamp.to(self.device)
                with open(train_residual_path, 'rb') as f:
                    train_residual_stamp = pickle.load(f)
                    train_residual_stamp = train_residual_stamp.to(self.device)
            elif data_train is None:
                raise Exception("Existing training STL decomposition files not found. Must provide training data to perform decomposition.")
            else:
                train_stl_root = pathlib.Path(dataset_path).parent / "stl"
                train_stl_root.mkdir(parents=True, exist_ok=True)

                train_trend_stamp = torch.empty((data_train.shape[0], data_train.shape[1]), dtype=torch.float32, device=self.device)
                train_seasonal_stamp = torch.empty((data_train.shape[0], data_train.shape[1]), dtype=torch.float32, device=self.device)
                train_residual_stamp = torch.empty((data_train.shape[0], data_train.shape[1]), dtype=torch.float32, device=self.device)
                for feat_idx in range(data_train.shape[1]):
                    res_train = STL(data_train[:, feat_idx], period=period).fit()
                    train_trend_stamp[:, feat_idx] = torch.tensor(res_train.trend, dtype=torch.float32, device=self.device)
                    train_seasonal_stamp[:, feat_idx] = torch.tensor(res_train.seasonal, dtype=torch.float32, device=self.device)
                    train_residual_stamp[:, feat_idx] = torch.tensor(res_train.resid, dtype=torch.float32, device=self.device)

                with open(train_trend_path, 'wb') as f:
                    pickle.dump(train_trend_stamp, f)
                with open(train_seasonal_path, 'wb') as f:
                    pickle.dump(train_seasonal_stamp, f)
                with open(train_residual_path, 'wb') as f:
                    pickle.dump(train_residual_stamp, f)

            self.train_trend_stamp = train_trend_stamp # [train_len, num_features]
            self.train_seasonal_stamp = train_seasonal_stamp # .
            self.train_residual_stamp = train_residual_stamp # .

            
        # self.train_trend_stamp = torch.zeros((data_train.shape[0], data_train.shape[1]), dtype=torch.float32)
        # self.train_seasonal_stamp = torch.zeros((data_train.shape[0], data_train.shape[1]), dtype=torch.float32)
        # self.train_residual_stamp = torch.zeros((data_train.shape[0], data_train.shape[1]), dtype=torch.float32)
        # self.val_trend_stamp = torch.zeros((data_val.shape[0], data_val.shape[1]), dtype=torch.float32)
        # self.val_seasonal_stamp = torch.zeros((data_val.shape[0], data_val.shape[1]), dtype=torch.float32)
        # self.val_residual_stamp = torch.zeros((data_val.shape[0], data_val.shape[1]), dtype=torch.float32)


    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits model to data. Modifies object attributes and returns nothing.

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        self.x = x # [pre+post_ab_len, num_features]
        self.y = y # [pre+post_ab_len, 1]

        
    def predict(self, x: np.ndarray, ablation_start: int, units: str = 's', mode="test") -> np.ndarray:
        """
        Predicts y values for x values.

        Args:
            x: x values [ablation_len, 1]
            ablation_start: index of where the first missing value would be placed in the array fitted to x. For example, if the array 
                fitted to x is [3,4,6,7], the ablation start index should be 2, because the missing value would be in position 2 if the array was
                uninterrupted.
            units: unit of time for the x values. Default is 's' for seconds.
            mode: whether the function is being used in a testing or training case. Default is 'test'.

        Returns:
            predicted y values.

        Raises:
            Exception: if model is not fitted.
        """
        if self.y is None:
            raise Exception("Model not fitted.")
        if self.val_trend_stamp is None and mode == "test":
            raise Exception("Must call STL decomposition with validation data before predicting in test mode.")
        if self.train_trend_stamp is None and mode == "train":
            raise Exception("Must call STL decomposition with training data before predicting in train mode.")
        
        x_tensor = torch.tensor(x, dtype=torch.float32) # [ablation_len, 1]

        pre_ablation_context = self.y[:ablation_start] # [pre_ab_len, 1] - same for s,t,r
        if mode == "train":
            trend = self.train_trend_stamp[:ablation_start]
            seasonal = self.train_seasonal_stamp[:ablation_start]
            residual = self.train_residual_stamp[:ablation_start]
        elif mode == "test":
            trend = self.val_trend_stamp[:ablation_start]
            seasonal = self.val_seasonal_stamp[:ablation_start]
            residual = self.val_residual_stamp[:ablation_start]
        else:
            raise Exception("Invalid mode. Must be 'train' or 'test'.")
        
        if len(pre_ablation_context) >= self.config.seq_len:
            pre_ablation_context = pre_ablation_context[-self.config.seq_len:] # [seq_len, 1] - same for s,t,r
            trend = trend[-self.config.seq_len:]
            seasonal = seasonal[-self.config.seq_len:]
            residual = residual[-self.config.seq_len:]

        else:
            # paddin'
            pre_ablation_context = torch.nn.functional.pad(pre_ablation_context, (0, 0, self.config.seq_len - len(pre_ablation_context), 0), mode='replicate') # [seq_len, 1] - same for s,t,r
            trend = torch.nn.functional.pad(trend, (0, 0, self.config.seq_len - len(trend), 0), mode='replicate')
            seasonal = torch.nn.functional.pad(seasonal, (0, 0, self.config.seq_len - len(seasonal), 0), mode='replicate')
            residual = torch.nn.functional.pad(residual, (0, 0, self.config.seq_len - len(residual), 0), mode='replicate')

        pre_ablation_context = pre_ablation_context.unsqueeze(0) # [1, seq_len, 1] - same for s,t,r
        trend = trend.unsqueeze(0)
        seasonal = seasonal.unsqueeze(0)
        residual = residual.unsqueeze(0)
        
        outputs, _ = self.model(
            x=x_tensor.repeat(2,1,1), 
            itr=0, 
            trend=trend.repeat(2,1,1), 
            season=seasonal.repeat(2,1,1), 
            noise=residual.repeat(2,1,1), 
            test=False
            )
        outputs = outputs[0, :x.shape[0], :]
        return outputs.detach().numpy() if mode=="test" else outputs
        
    
    def fine_tune(self, cfg):
        """
        Fine-tunes the model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir logs
        ```
        Clean up the logs directory after training is complete.

        Args:
            cfg: TrainerConfig object.
        """
        self.trainer_cfg = cfg

        # writing out a text file to the logging directory with the string of the trainer config
        hp_path = pathlib.Path(f"{self.trainer_cfg.logging_dir}/{self.trainer_cfg.run_name}")
        hp_path.mkdir(parents=True, exist_ok=True)
        with open(f"{hp_path}/trainer_config.txt", "w") as file:
            file.write(str(self.trainer_cfg))

        criterion = torch.nn.MSELoss()
        optimizer = self.trainer_cfg.optimizer(self.model.parameters(), lr=self.trainer_cfg.lr)

        pbar = tqdm.tqdm(range(self.trainer_cfg.n_epochs), disable=self.trainer_cfg.disable_tqdm)
        writer_path = pathlib.Path(f"{self.trainer_cfg.logging_dir}/{self.trainer_cfg.run_name}/tensorboard")
        writer_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.trainer_cfg.logging_frequency)
        checkpointing_steps = int(1 / self.trainer_cfg.saving_frequency)

        if self.trainer_cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer_cfg.n_epochs)

        if self.trainer_cfg.resume_from_checkpoint:
            self.checkpoint_dict = torch.load(self.trainer_cfg.checkpoint_path)
            model_dict = self.checkpoint_dict['model_state_dict']
            self.model.load_state_dict(model_dict)

            optimizer.load_state_dict(self.checkpoint_dict["optim_state_dict"])
            pbar = tqdm.tqdm(
                iterable=range(self.checkpoint_dict["epoch"] + 1, self.trainer_cfg.n_epochs),
                total=self.trainer_cfg.n_epochs,
                initial=self.checkpoint_dict["epoch"] + 1,
                disable=self.trainer_cfg.disable_tqdm
            )

        sequencerizer = lambda x: torch.unfold_copy(
            x, 
            0, 
            self.config.seq_len + self.config.pred_len, 
            self.trainer_cfg.batch_stride
            )
        
        # Go through the data epoch_n times
        for epoch_n in pbar:
            print(f"Epoch {epoch_n}")
            epoch_loss = []
            for dataset_path in self.trainer_cfg.dataset_path_lst:
                print(f". Dataset: {dataset_path}")
                df = pd.read_csv(dataset_path, index_col=0)
                train_idx = int(self.trainer_cfg.train_set_size*len(df))
                data_train = torch.tensor(df.values[:train_idx], dtype=torch.float32, device=self.device) # [train_len, num_features]
                data_val = torch.tensor(df.values[train_idx:], dtype=torch.float32, device=self.device) # [val_len, num_features]
                self._stl_resolve(mode="train", data_train = data_train.cpu().detach().numpy(), data_val = data_val.cpu().detach().numpy(), dataset_path=dataset_path)      
                
                # doing each feature of the dataset at a time
                for feat_idx in range(data_train.shape[1]):
                    print(f".. Feature {feat_idx}")
                    full_seq_train = data_train[epoch_n:, feat_idx]
                    seq_result = map(sequencerizer,
                        (
                            full_seq_train, 
                            self.train_trend_stamp[epoch_n:, feat_idx],
                            self.train_seasonal_stamp[epoch_n:, feat_idx],
                            self.train_residual_stamp[epoch_n:, feat_idx]
                            )
                        )
                    sequences, trend_seqs_raw, seasonal_seqs_raw, residual_seqs_raw = list(seq_result) # all are [num_sequences, seq_len+pred_len]
                    perm = torch.randperm(sequences.shape[0], device=self.device)
                    seqs_x = sequences[perm, :self.config.seq_len] # [num_sequences, seq_len]
                    seqs_y = sequences[perm, self.config.seq_len:] # [num_sequences, pred_len]
                    trend_seqs = trend_seqs_raw[perm, :self.config.seq_len] # [num_sequences, seq_len]
                    seasonal_seqs = seasonal_seqs_raw[perm, :self.config.seq_len]
                    residual_seqs = residual_seqs_raw[perm, :self.config.seq_len]

                    for batch_n in range(0, sequences.shape[0], self.trainer_cfg.batch_size):
                        print(f"... batch {batch_n/self.trainer_cfg.batch_size} of {sequences.shape[0]//self.trainer_cfg.batch_size}", flush=True, end="\r")
                        batch_x = seqs_x[batch_n: batch_n + self.trainer_cfg.batch_size] # [batch_size, seq_len]
                        batch_y = seqs_y[batch_n: batch_n + self.trainer_cfg.batch_size] # [batch_size, pred_len]
                        batch_trend = trend_seqs[batch_n: batch_n + self.trainer_cfg.batch_size] # [batch_size, seq_len]
                        batch_seasonal = seasonal_seqs[batch_n: batch_n + self.trainer_cfg.batch_size]
                        batch_residual = residual_seqs[batch_n: batch_n + self.trainer_cfg.batch_size]
                    
                        outputs, _ = self.model(
                            x = batch_x.unsqueeze(2), # [batch_size, seq_len, 1]
                            itr = 0, 
                            trend = batch_trend.unsqueeze(2), # [batch_size, seq_len, 1]
                            season = batch_seasonal.unsqueeze(2), # [batch_size, seq_len, 1]
                            noise = batch_residual.unsqueeze(2), # [batch_size, seq_len, 1]
                            test = False
                            )  # [batch_size, pred_len, 1]
                        loss = criterion(outputs, batch_y.unsqueeze(2)) # [batch_size, pred_len, 1]
                        loss.backward()
                        optimizer.step()
                        epoch_loss.append(loss.item())
                        optimizer.zero_grad()

            print("#"*50, flush=True)
            print("Now saving checkpoint...", flush=True)
            if (epoch_n+1) % checkpointing_steps == 0:
                self.save_checkpoint({
                    "epoch": epoch_n,
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                })
            print("#"*50, flush=True)

            if self.trainer_cfg.lr_scheduler:
                scheduler.step()
            
            pbar.set_description(f"Epoch Loss: {sum(epoch_loss)/len(epoch_loss)}")
            
            print("#"*50, flush=True)
            print("Now beginning evaluation...", flush=True)
            if (epoch_n+1) % logging_steps == 0:
                val_loss = self.evaluate(criterion, sequencerizer)
                writer.add_scalars(
                    "Loss", 
                    {"Training" : sum(epoch_loss)/len(epoch_loss), "Validation": sum(val_loss)/len(val_loss)}, 
                    epoch_n
                    )
            print("#"*50, flush=True)

        writer.flush()
        writer.close()
    

    def evaluate(self, criterion: callable, sequencerizer: callable) -> list:
        """
        Evaluates the model on the validation set.

        Args:
            criterion: loss function
            sequencerizer: function to create dataset of sequences [val_len, 1] -> [num_sequences, seq_len+pred_len]

        Returns:
            val_loss: loss on the validation set
        """
        val_loss = []
        for dataset_path in self.trainer_cfg.dataset_path_lst:
            print(f". Dataset: {dataset_path}")
            df = pd.read_csv(dataset_path, index_col=0)
            train_idx = int(self.trainer_cfg.train_set_size*len(df))
            data_val = torch.tensor(df.values[train_idx:], dtype=torch.float32, device=self.device) # [val_len, num_features]
            self._stl_resolve(mode="val", data_val = data_val.cpu().detach().numpy(), dataset_path=dataset_path)      
            
            # doing each feature of the dataset at a time
            for feat_idx in range(data_val.shape[1]):
                print(f".. Feature {feat_idx}")
                full_seq_val = data_val[:, feat_idx]
                seq_result = map(sequencerizer,
                    (
                        full_seq_val, 
                        self.val_trend_stamp[:, feat_idx],
                        self.val_seasonal_stamp[:, feat_idx],
                        self.val_residual_stamp[:, feat_idx]
                        )
                    )
                sequences, trend_seqs_raw, seasonal_seqs_raw, residual_seqs_raw = list(seq_result) # all are [num_sequences, seq_len+pred_len]

                seqs_x = sequences[:, :self.config.seq_len] # [num_sequences, seq_len]
                seqs_y = sequences[:, self.config.seq_len:] # [num_sequences, pred_len]
                trend_seqs = trend_seqs_raw[:, :self.config.seq_len] # [num_sequences, seq_len]
                seasonal_seqs = seasonal_seqs_raw[:, :self.config.seq_len]
                residual_seqs = residual_seqs_raw[:, :self.config.seq_len]

                for batch_n in range(0, sequences.shape[0], self.trainer_cfg.batch_size):
                    print(f"... batch {batch_n//self.trainer_cfg.batch_size} of {sequences.shape[0]//self.trainer_cfg.batch_size}", flush=True)
                    batch_x = seqs_x[batch_n: batch_n + self.trainer_cfg.batch_size] # [batch_size, seq_len]
                    batch_y = seqs_y[batch_n: batch_n + self.trainer_cfg.batch_size] # [batch_size, pred_len]
                    batch_trend = trend_seqs[batch_n: batch_n + self.trainer_cfg.batch_size] # [batch_size, seq_len]
                    batch_seasonal = seasonal_seqs[batch_n: batch_n + self.trainer_cfg.batch_size]
                    batch_residual = residual_seqs[batch_n: batch_n + self.trainer_cfg.batch_size]
                
                    with torch.no_grad():
                        outputs, _ = self.model(
                            x = batch_x.unsqueeze(2), # [batch_size, seq_len, 1]
                            itr = 0, 
                            trend = batch_trend.unsqueeze(2), # [batch_size, seq_len, 1]
                            season = batch_seasonal.unsqueeze(2), # [batch_size, seq_len, 1]
                            noise = batch_residual.unsqueeze(2), # [batch_size, seq_len, 1]
                            test = False
                            )  # [batch_size, pred_len, 1]

                        loss = criterion(outputs, batch_y.unsqueeze(2)) # [batch_size, pred_len, 1]
                    
                    val_loss.append(loss.item())

        return val_loss


    def save_checkpoint(self, checkpoint_dict: dict):   
        """
        Saves a model checkpoint to a file.

        Args:
            checkpoint_dict: dictionary of training information from checkpoint. Must contain an 'epoch' key.
        """
        checkpoint_path = pathlib.Path(f"{self.trainer_cfg.logging_dir}/{self.trainer_cfg.run_name}/checkpoints")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_dict,
            checkpoint_path / f"checkpt_e{checkpoint_dict['epoch']}.pt",
        )



class TempoConfig:
    """
    Configures the TEMPO model.

    Attributes:
        is_gpt: whether the model is a GPT model. Default is 1.
        patch_size: length of the time series patch for the gpt prompt. Default is 16.
        pretrain: whether the model is pre-trained. Default is 1.
        stride: by how many indices that each sequence will overlap with the previous sequence in the model prompt. Default is 8.
        seq_len: length of the sequence per prompt. Default is 336.
        gpt_layers: number of GPT layers. Default is 3.
        prompt: whether to use a prompt. Default is 1.
        pool: whether to use pooling in the model. Default is True.
        d_model: size of the model dimension. Default is 768.
        use_token: use prompt token's representation as the forecasting's information. Default is 0.
        pred_len: model's prediction horizon. Default is 96.
        freeze: whether to freeze the some of the model parameters during training. Default is 1.
        num_nodes: number of nodes for reverse-instance normalization. Default is 1.
        best_model_path: path to the best model. Default is "TEMPO_checkpoints/ettm2_TEMPO_3_prompt_learn_336_96_100_sl336_ll168_pl96_dm768_nh4_el3_gl3_df768_ebtimeF_itr0/checkpoint.pth".
    """
    def __init__(
        self,
        is_gpt: int = 1,
        patch_size: int = 16,
        pretrain: int = 1,
        stride: int = 8,
        seq_len: int = 336,
        gpt_layers: int = 3,
        prompt: int = 1,
        pool: int = True,
        d_model: int = 768,
        use_token: int = 0,
        pred_len: int = 96,
        freeze: int = 1,
        num_nodes: int = 1,
        best_model_path: str = "TEMPO_checkpoints/ettm2_TEMPO_3_prompt_learn_336_96_100_sl336_ll168_pl96_dm768_nh4_el3_gl3_df768_ebtimeF_itr0/checkpoint.pth",
        ):
        self.is_gpt = is_gpt
        self.patch_size = patch_size
        self.pretrain = pretrain
        self.stride = stride
        self.seq_len = seq_len
        self.gpt_layers = gpt_layers
        self.prompt = prompt
        self.pool = pool
        self.d_model = d_model
        self.use_token = use_token
        self.pred_len = pred_len
        self.freeze = freeze
        self.num_nodes = num_nodes
        self.best_model_path = best_model_path



if __name__ == '__main__':
    model = TempoGPT()
    model.fit(
        np.random.randn(100,3),
        np.random.randn(100,1)
    )
    model.predict(
        np.random.randn(4,3),
        84,
    )
