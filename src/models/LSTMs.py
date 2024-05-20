"""
LSTM and BidirectionalLSTM classes for time series interpolation.

Typical usage example:
```python
>>> lstm = LSTM(input_size = 2, n_layers = 4, window_size = 10)
>>> lstm_trainer_config = TrainerConfig(
...     dataset_path = "data/training_data.csv",
...     train_set_size = 0.8,
...     n_epochs = 1000,
...     batch_size = 16,
...     lr = 0.001,
...     start_idx = None,
...     stop_idx = None,
...     optimizer = torch.optim.Adam,
...     logging_dir = "logs",
...     logging_steps_ratio = 0.1,
...     save_steps_ratio = 0.01,
...     lr_scheduler = False,
...     resume_from_checkpoint = False,
...     checkpoint_path = None
... )
>>> lstm.train(lstm_trainer_config)

And to view the training progress, run the following command in the terminal:
```bash
tensorboard --logdir logs
```
Clean up the logs directory after training is complete.
"""
import torch
import torch.nn as nn
import tqdm
import pathlib
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import dump
import warnings

from src.config.trainer_configs import TrainerConfig, BidiMLPTrainerConfig

class LSTM(torch.nn.Module):
    """
    A unidirectional LSTM model.

    Attributes:
        input_size: number of features in the input
        n_layers: number of layers in the LSTM
        window_size: size of the sliding window for forward passes
        lstm: LSTM model
        --------------------
        (Established in Methods)
        x: independent variables for prediction
        y: dependent variable for prediction
        ss_x: StandardScaler for independent variables
        ss_y: StandardScaler for dependent variable
        cfg: TrainerConfig object for training
        checkpoint_dict: dictionary of training information from checkpoint
    """

    def __init__(self, input_size: int, n_layers: int, window_size: int):
        """
        Initializes an instance of the LSTM class.

        Args:
            input_size: number of features in the input.
            n_layers: number of layers in the LSTM.
            window_size: size of the sliding window for forward passes
        """
        super().__init__()
        self.x = None
        self.y = None

        self.input_size = input_size
        self.n_layers = n_layers
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=1, num_layers=n_layers)


    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Establishes x and y attributes for prediction

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) != torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32)

        self.x = x
        self.y = y


    def predict(self, x: torch.Tensor, ablation_start: int) -> torch.Tensor:
        """
        Predicts the dependent variable based on the independent variables. The instance's x and y attributes should be set to the
        dataset with an ablation before calling this method. The value of x passed to this method should be that same ablation.

        Args:
            x: independent variable ablation for prediction
            ablation_start: starting index of the ablation in the dataset fit to self.x and self.y
        
        Returns:
            y_pred: predicted dependent variable for an ablation
        """
        if type(x) != torch.Tensor:
            x = torch.tensor(x.copy(), dtype=torch.float32)

        with torch.no_grad():
            if self.window_size > ablation_start:
                x_interval = self.x[:ablation_start]
            else:
                x_interval = self.x[ablation_start-self.window_size:ablation_start]
            
            y_pred = torch.empty(len(x), 1)

            for i in range(len(x)):
                _, hidden, _ = self.forward(x_interval)
                y_pred[i] = hidden[-1].item()
                x_interval = torch.cat((x_interval[1:], x[i].reshape(1, -1)), dim=0)

            return y_pred


    def forward(self, x: torch.Tensor, batch_size = 1) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x: independent variables for prediction
            batch_size: size of the training batch
        
        Returns:
            output: output of the LSTM model
        """
        if batch_size > 1:
            h_0 = torch.zeros((self.n_layers, batch_size, 1), requires_grad=True) # hidden state
            c_0 = torch.zeros((self.n_layers, batch_size, 1), requires_grad=True) # internal state
        else:
            h_0 = torch.zeros((self.n_layers, 1), requires_grad=True)
            c_0 = torch.zeros((self.n_layers, 1), requires_grad=True)
        # output - [seq_len, batch, hidden_size]
        # hn - [num_layers, batch, hidden_size]
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        return output, hn, cn


    def train(self, cfg: TrainerConfig):
        """
        Trains the model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir logs
        ```
        Clean up the logs directory after training is complete.

        Args:
            cfg: TrainerConfig object.
        """
        self.cfg = cfg

        # writing out a text file to the logging directory with the string of the trainer config
        hp_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}")
        hp_path.mkdir(parents=True, exist_ok=True)
        with open(f"{hp_path}/trainer_config.txt", "w") as file:
            file.write(str(self.cfg))

        df = pd.read_csv(self.cfg.dataset_path, index_col=0)
        x = df.values
        y = df.values[:,-1]

        ### Trimming dataset for training 
        ###    - make sure this is disjoint with the evaluation dataset
        if self.cfg.start_idx and self.cfg.stop_idx:
            x,y = x[self.cfg.start_idx: self.cfg.stop_idx], y[self.cfg.start_idx: self.cfg.stop_idx]
        elif self.cfg.start_idx:
            x,y = x[self.cfg.start_idx:], y[self.cfg.start_idx:]
        elif self.cfg.stop_idx:
            x,y = x[:self.cfg.stop_idx], y[:self.cfg.stop_idx]

        if self.cfg.reverse:
            x,y = x[::-1], y[::-1]

        train_idx = int(self.cfg.train_set_size*len(x))

        self.ss_x = StandardScaler()
        self.ss_y = StandardScaler()
        self.ss_x.fit(x[:train_idx])
        self.ss_y.fit(y.reshape(-1,1)[:train_idx])

        # saving the fitted scalers
        scaler_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/scalers")
        scaler_path.mkdir(parents=True, exist_ok=True)
        dump(self.ss_x, scaler_path / "x_scaler.joblib")
        dump(self.ss_y, scaler_path / "y_scaler.joblib")

        x_ss = self.ss_x.transform(x)
        y_ss = self.ss_y.transform(y.reshape(-1,1))

        x_train = torch.tensor(x_ss[:train_idx], dtype=torch.float32, requires_grad=True)
        y_train = torch.tensor(y_ss[:train_idx], dtype=torch.float32, requires_grad=True)
        x_test = torch.tensor(x_ss[train_idx:], dtype=torch.float32)
        y_test = torch.tensor(y_ss[train_idx:], dtype=torch.float32)

        criterion = torch.nn.MSELoss()
        optimizer = self.cfg.optimizer(self.lstm.parameters(), lr=self.cfg.lr)

        pbar = tqdm.tqdm(range(self.cfg.n_epochs), disable=False)
        writer_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/tensorboard")
        writer_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.cfg.logging_steps_ratio)
        checkpointing_steps = int(1 / self.cfg.save_steps_ratio)

        if self.cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.n_epochs)

        if self.cfg.resume_from_checkpoint:
            self.checkpoint_dict = torch.load(self.cfg.checkpoint_path)
            model_dict = self.checkpoint_dict['model_state_dict']
            prefix = 'lstm.'
            modified_model_dict = {prefix+k: v for k, v in model_dict.items()}
            self.lstm.load_state_dict(modified_model_dict)

            optimizer.load_state_dict(self.checkpoint_dict["optim_state_dict"])
            pbar = tqdm.tqdm(
                iterable=range(self.checkpoint_dict["epoch"] + 1, self.cfg.n_epochs),
                total=self.cfg.n_epochs,
                initial=self.checkpoint_dict["epoch"] + 1,
            )

        # Train the model
        for epoch_n in pbar:
            # print(f"run: {str(self.cfg)}, epoch: {epoch_n}")
            epoch_loss = []
            for i in range(len(x_train) - self.window_size - self.cfg.batch_size + 1):
                x_train_window = x_train[i: i + self.window_size].unsqueeze(1)
                y_train_window = y_train[i + self.window_size].unsqueeze(1)
                for b in range(1, self.cfg.batch_size):
                    x_train_window = torch.cat((x_train_window, x_train[i+b: i+b + self.window_size].unsqueeze(1)), dim=1)
                    y_train_window = torch.cat((y_train_window, y_train[i+b + self.window_size].unsqueeze(1)), dim=1)

                # x_train_window = [seq_len, batch, input_size]
                # y_train_window = [1, batch] needs to be [batch, 1] for loss
                _, hidden, _ = self.forward(x_train_window, self.cfg.batch_size)

                # loss is on scaled values because would have to detach the tensors to inverse transform - could not then backpropagate
                loss = criterion(hidden[-1], y_train_window.T)
                epoch_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if (epoch_n+1) % logging_steps == 0:
                val_loss = self.evaluate(x_test, y_test, criterion, self.cfg.batch_size)
                writer.add_scalars(
                    "Loss", 
                    {"Training" : sum(epoch_loss)/len(epoch_loss), "Validation": sum(val_loss)/len(val_loss)}, 
                    epoch_n
                    )

            if (epoch_n+1) % checkpointing_steps == 0:
                self.save_checkpoint({
                    "epoch": epoch_n,
                    "model_state_dict": self.lstm.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                })

            if self.cfg.lr_scheduler:
                scheduler.step()
            
            pbar.set_description(f"Loss: {sum(epoch_loss)/len(epoch_loss)}")

        writer.flush()
        writer.close()
    

    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor, criterion: callable, batch_size = 1) -> list:
        """
        Evaluates the model on the test set.

        Args:
            x_test: independent variables for testing
            y_test: dependent variable for testing
            criterion: loss function
            batch_size: size of the training batch

        Returns:
            test_loss: loss on the test set
        """
        test_loss = []

        for i in range(0, len(x_test) - self.window_size - batch_size + 1, 1):

            # sliding window
            x_test_window = x_test[i: i + self.window_size].unsqueeze(1)
            y_test_window = y_test[i + self.window_size].unsqueeze(1)
            for b in range(1, batch_size):
                x_test_window = torch.cat((x_test_window, x_test[i+b: i+b + self.window_size].unsqueeze(1)), dim=1)
                y_test_window = torch.cat((y_test_window, y_test[i+b + self.window_size].unsqueeze(1)), dim=1)

            # x_test_window = [seq_len, batch, input_size]
            # y_test_window = [1, batch] needs to be [batch, 1] for loss

            with torch.no_grad():
                _, hidden, _ = self.forward(x_test_window, batch_size)

                loss = criterion(hidden[-1], y_test_window.T)
                
            test_loss.append(loss.item())

        return test_loss


    def save_checkpoint(self, checkpoint_dict: dict):
        """
        Saves a model checkpoint to a file.

        Args:
            checkpoint_dict: dictionary of training information from checkpoint. Must contain an 'epoch' key.
        """
        checkpoint_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/checkpoints")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_dict,
            checkpoint_path / f"checkpt_e{checkpoint_dict['epoch']}.pt",
        )



class BidirectionalLSTM(torch.nn.Module):
    """
    A bidirectional LSTM model will create two LSTMs and one MLP to combine the outputs.

    Attributes:
        input_size: number of features in the input.
        lstm_f_layers: number of layers in lstm_f.
        lstm_b_layers: number of layers in lstm_b.
        window_size: size of the sliding window for forward passes
        lstm_f: LSTM model for forecasting the data
        lstm_b: LSTM model for forecasting the data in reverse
        mlp: MLP model to combine the outputs of the LSTMs
        --------------------
        (Established in Methods)
        x: independent variables for prediction
        y: dependent variable for prediction
        ss_x: StandardScaler for independent variables
        ss_y: StandardScaler for dependent variable
        cfg: TrainerConfig object for training
        checkpoint_dict: dictionary of training information from checkpoint
    """
    def __init__(self, input_size: int, lstm_f_layers: int, lstm_b_layers: int, window_size: int):
        """
        Initializes an instance of the LSTM class.

        Args:
            input_size: number of features in the input
            lstm_f_layers: number of layers in lstm_f
            lstm_b_layers: number of layers in lstm_b
            window_size: size of the sliding window for forward passes of the LSTMs
        """
        super().__init__()
        self.x = None
        self.y = None

        self.input_size = input_size
        self.lstm_f_layers = lstm_f_layers
        self.lstm_b_layers = lstm_b_layers
        self.window_size = window_size
        self.lstm_f = LSTM(input_size=input_size, n_layers=lstm_f_layers, window_size=window_size)
        self.lstm_b = LSTM(input_size=input_size, n_layers=lstm_b_layers, window_size=window_size)
        self.mlp = nn.Sequential(
            nn.Linear(2,3), # input features: [lstm_f out, lstm_b out]
            nn.ReLU(),
            nn.Linear(3,1)
        )


    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Establishes x and y attributes for prediction

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) != torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32)

        self.x = x
        self.y = y


    def predict(self, x: torch.Tensor, ablation_start: int) -> torch.Tensor:
        """
        Predicts the dependent variable based on the independent variables. The instance's x and y attributes should be set to the
        dataset with an ablation before calling this method. The value of x passed to this method should be that same ablation.

        Args:
            x: independent variable ablation for prediction
            ablation_start: starting index of the ablation in the dataset fit to self.x and self.y
        
        Returns:
            y_pred: predicted dependent variable for an ablation
        """
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            if self.window_size > ablation_start:
                x_interval_f = self.x[:ablation_start]
            else:
                x_interval_f = self.x[ablation_start-self.window_size:ablation_start]

            if self.window_size > len(self.x) - ablation_start:
                x_interval_b = torch.flip(self.x[ablation_start:], dims=[0])
            else:
                x_interval_b = torch.flip(self.x[ablation_start:ablation_start+self.window_size], dims=[0])

            x_b = torch.flip(x, dims=[0])

            y_pred_f = torch.empty(len(x), 1)
            y_pred_b = torch.empty(len(x), 1)

            for i in range(len(x)):
                _, hidden_f, _ = self.lstm_f.forward(x_interval_f)
                _, hidden_b, _ = self.lstm_b.forward(x_interval_b)

                y_pred_f[i] = hidden_f[-1].item()
                y_pred_b[i] = hidden_b[-1].item()

                x_interval_f = torch.cat((x_interval_f[1:], x[i].reshape(1, -1)))
                x_interval_b = torch.cat((x_interval_b[1:], x_b[i].reshape(1, -1)))

            mlp_in = torch.stack((
                y_pred_f,
                torch.flip(y_pred_b, dims=[0])
            ), dim=1).squeeze()

            y_pred = self.mlp(mlp_in)
            return y_pred


    def train(self, cfg: BidiMLPTrainerConfig):
        """
        Trains the model. To view the training progress, run the following command in the terminal:
        ```bash
        tensorboard --logdir logs
        ```
        Clean up the logs directory after training is complete.

        Args:
            cfg: TrainerConfig object.
        """
        self.cfg = cfg

        # writing out a text file to the logging directory with the string of the trainer config
        hp_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}")
        hp_path.mkdir(parents=True, exist_ok=True)
        with open(f"{hp_path}/trainer_config.txt", "w") as file:
            file.write(str(self.cfg))

        self.ablation_max = self.cfg.ablation_max

        lstm_f_cpt_file = self.cfg.lstm_f_cpt_file
        lstm_b_cpt_file = self.cfg.lstm_b_cpt_file

        lstm_f_model_dict = torch.load(lstm_f_cpt_file)["model_state_dict"]
        lstm_b_model_dict = torch.load(lstm_b_cpt_file)["model_state_dict"]

        prefix = 'lstm.'
        modified_model_dict1 = {prefix+k: v for k, v in lstm_f_model_dict.items()}
        modified_model_dict2 = {prefix+k: v for k, v in lstm_b_model_dict.items()}
        self.lstm_f.load_state_dict(modified_model_dict1)
        self.lstm_b.load_state_dict(modified_model_dict2)
        warnings.warn("LSTM models loaded from checkpoint. Make sure the LSTM models were trained on the same dataset that is being used to train this BiLSTM.")
        
        # freezing the parameters
        for param in self.lstm_f.parameters():
            param.requires_grad = False
        for param in self.lstm_b.parameters():
            param.requires_grad = False

        # training the mlp
        df = pd.read_csv(self.cfg.dataset_path, index_col=0)
        x = df.values
        y = df.values[:,-1]

        ### Trimming dataset for training
        ###    - make sure this is disjoint with the test set
        if self.cfg.start_idx and self.cfg.stop_idx:
            x,y = x[self.cfg.start_idx: self.cfg.stop_idx], y[self.cfg.start_idx: self.cfg.stop_idx]
        elif self.cfg.start_idx:
            x,y = x[self.cfg.start_idx:], y[self.cfg.start_idx:]
        elif self.cfg.stop_idx:
            x,y = x[:self.cfg.stop_idx], y[:self.cfg.stop_idx]

        train_idx = int(self.cfg.train_set_size*len(x))

        self.ss_x = StandardScaler()
        self.ss_y = StandardScaler()
        self.ss_x.fit(x[:train_idx])
        self.ss_y.fit(y.reshape(-1,1)[:train_idx])

        scaler_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/scalers")
        scaler_path.mkdir(parents=True, exist_ok=True)

        dump(self.ss_x, scaler_path / "x_scaler.joblib")
        dump(self.ss_y, scaler_path / "y_scaler.joblib")

        x_ss = self.ss_x.transform(x)
        y_ss = self.ss_y.transform(y.reshape(-1,1))

        x_train = torch.tensor(x_ss[:train_idx], dtype=torch.float32, requires_grad=True)
        y_train = torch.tensor(y_ss[:train_idx], dtype=torch.float32, requires_grad=True)
        x_test = torch.tensor(x_ss[train_idx:], dtype=torch.float32)
        y_test = torch.tensor(y_ss[train_idx:], dtype=torch.float32)

        criterion = torch.nn.MSELoss()
        optimizer = self.cfg.optimizer(self.mlp.parameters(), lr=self.cfg.lr)

        pbar = tqdm.tqdm(range(self.cfg.n_epochs), disable=False)
        writer_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/tensorboard")
        writer_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=writer_path)
        logging_steps = int(1 / self.cfg.logging_steps_ratio)
        checkpointing_steps = int(1 / self.cfg.save_steps_ratio)

        if self.cfg.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.n_epochs)

        if self.cfg.resume_from_checkpoint:
            self.checkpoint_dict = torch.load(self.cfg.checkpoint_path)
            model_dict = self.checkpoint_dict['mlp_state_dict']
            prefix = 'mlp.'
            modified_model_dict = {prefix+k: v for k, v in model_dict.items()}
            self.mlp.load_state_dict(modified_model_dict)
            
            optimizer.load_state_dict(self.checkpoint_dict["optim_state_dict"])
            pbar = tqdm.tqdm(
                iterable=range(self.checkpoint_dict["epoch"] + 1, self.cfg.n_epochs),
                total=self.cfg.n_epochs,
                initial=self.checkpoint_dict["epoch"] + 1,
            )

        # Train the model
        for epoch_n in pbar:
            epoch_loss = []

            for i in range(0, len(x_train) - (self.window_size*2 + self.ablation_max), 1):

                x_rel = x_train[i: i + self.window_size*2 + self.ablation_max]
                y_rel = y_train[i: i + self.window_size*2 + self.ablation_max]

                x_ablated = torch.cat((x_rel[:self.window_size], x_rel[self.window_size + self.ablation_max:]))
                y_ablated = torch.cat((y_rel[:self.window_size], y_rel[self.window_size + self.ablation_max:]))

                x_ablation = x_rel[self.window_size:self.window_size + self.ablation_max]
                y_ablation = y_rel[self.window_size:self.window_size + self.ablation_max]

                self.lstm_f.fit(x_ablated, y_ablated)
                y_ablation_pred_f = self.lstm_f.predict(x_ablation, self.window_size).reshape(-1,1)
            
                self.lstm_b.fit(torch.flip(x_ablated, dims=[0]), torch.flip(y_ablated, dims=[0]))
                y_ablation_pred_b = self.lstm_b.predict(torch.flip(x_ablation, dims=[0]), self.window_size).reshape(-1,1)

                mlp_in = torch.stack((
                    y_ablation_pred_f, 
                    torch.flip(y_ablation_pred_b, dims=[0]), 
                    ), dim=1).squeeze()

                fin_preds = self.mlp(mlp_in)
                
                if i % self.cfg.grad_accumulation_steps == 0:
                    loss = criterion(fin_preds, y_ablation)
                else:
                    loss = loss + criterion(fin_preds, y_ablation)

                if i % self.cfg.grad_accumulation_steps == self.cfg.grad_accumulation_steps - 1:
                    loss = loss / self.cfg.grad_accumulation_steps
                    epoch_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            
            if (epoch_n+1) % logging_steps == 0:
                val_loss = self.evaluate(x_test, y_test, criterion)
                writer.add_scalars(
                    "Loss", 
                    {"Training" : sum(epoch_loss)/len(epoch_loss), "Validation": sum(val_loss)/len(val_loss)}, 
                    epoch_n
                    )
                
            if (epoch_n+1) % checkpointing_steps == 0:
                self.save_checkpoint({
                    "epoch": epoch_n,
                    "mlp_state_dict": self.mlp.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                })

            if self.cfg.lr_scheduler:
                scheduler.step()

            pbar.set_description(f"Run: {self.cfg.run_name} - Loss: {round(sum(epoch_loss)/len(epoch_loss),3)}")

        writer.flush()
        writer.close()


    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor, criterion: callable) -> list:
        """
        Evaluates the model on the test set.

        Args:
            x_test: independent variables for testing
            y_test: dependent variable for testing
            criterion: loss function

        Returns:
            test_loss: loss on the test set
        """
        test_loss = []

        for i in range(0, len(x_test) - (self.window_size*2 + self.ablation_max), 1):
            x_rel = x_test[i: i + self.window_size*2 + self.ablation_max]
            y_rel = y_test[i: i + self.window_size*2 + self.ablation_max]

            x_ablated = torch.cat((x_rel[:self.window_size], x_rel[self.window_size + self.ablation_max:]))
            y_ablated = torch.cat((y_rel[:self.window_size], y_rel[self.window_size + self.ablation_max:]))

            x_ablation = x_rel[self.window_size:self.window_size + self.ablation_max]
            y_ablation = y_rel[self.window_size:self.window_size + self.ablation_max]

            with torch.no_grad():
                self.lstm_f.fit(x_ablated, y_ablated)
                y_ablation_pred_f = self.lstm_f.predict(x_ablation, self.window_size).reshape(-1,1)
            
                self.lstm_b.fit(torch.flip(x_ablated, dims=[0]), torch.flip(y_ablated, dims=[0]))
                y_ablation_pred_b = self.lstm_b.predict(torch.flip(x_ablation, dims=[0]), self.window_size).reshape(-1,1)

                mlp_in = torch.stack((
                    y_ablation_pred_f, 
                    torch.flip(y_ablation_pred_b, dims=[0]), 
                    ), dim=1).squeeze()

                fin_preds = self.mlp(mlp_in)
                loss = criterion(fin_preds, y_ablation)
            
            test_loss.append(loss.item())

        return test_loss


    def save_checkpoint(self, checkpoint_dict: dict):
        """
        Saves a model checkpoint to a file.

        Args:
            checkpoint_dict: dictionary of training information from checkpoint. Must contain an 'epoch' key.
        """
        checkpoint_path = pathlib.Path(f"{self.cfg.logging_dir}/{self.cfg.run_name}/checkpoints")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_dict,
            checkpoint_path / f"checkpt_e{checkpoint_dict['epoch']}.pt",
        )