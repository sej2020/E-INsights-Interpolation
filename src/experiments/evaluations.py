"""
Implements several evaluation experiments to compare models.

Classes:
    DirectEvaluation: Evaluates model directly by comparing interpolated values to ground truth values for a given set of interval lenghts.
        The process is repeated for a given number of repetitions. The results are stored in a yaml file.

Typical usage example:
```python
>>> from src.models.baseline import LinearInterpolation
>>> from src.experiments.evaluations import DirectEvaluation
>>> model = LinearInterpolation()
>>> direct_eval = DirectEvaluation(model)
>>> dataset_directory = "data"
>>> direct_eval.evaluate(dataset_directory, ablation_len=None, ablation_start=None, repetitions=1000, max_missing_interval=360, plot=True)
```
"""

import numpy as np
import torch
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
import yaml
import joblib
from src.utils import searching_all_files
from src.models.baseline import LinearInterpolation
from src.models.statsforecast import StatsModels
from src.models.LSTMs import LSTM, BidirectionalLSTM
from src.models.TimeGPT import TimeGPT
from src.models.TimesFM import TimesFM
from src.models.TempoGPT import TempoGPT


### Global Styling ###
######################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
CHONK_SIZE = 24
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:black")
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black")  
sns.set_style("darkgrid", {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})
plt.ticklabel_format(style = 'plain')
######################


class DirectEvaluation:
    """Evaluates model directly by comparing interpolated values to ground truth values.

    Attributes:
        model: model to evaluate.
        x_scaler: scaler used to normalize the x values of the training data.
        y_scaler: scaler used to normalize the y values of the training data.
    """
    def __init__(self, model: object, version_path: pathlib.Path | list = None):
        """Initializes an instance of the DirectEvaluation class.

        Args:
            model: model to evaluate.
            version_path: which training run to use to instantiate the model. The path must be to the ".pt" checkpoint object.
                The parent of the parent folder to this checkpoint must contain a 'scalers' folder with the MinMaxScalers used to 
                normalize the training data. If using Bidirectional LSTM, provide a list of paths in the order [MLP, lstm_f, lstm_b]. 
                A value None for this parameter is only appropriate if model is LinearInterpolation or StatsForecast.
        """
        self.model = model
        
        if not version_path and type(model) not in  [LinearInterpolation, StatsModels, TimeGPT, TimesFM, TempoGPT]:
            raise Exception("version_path cannot be None unless model is LinearInterpolation, StatsModels or TimeGPT.")
        
        if version_path and type(model) == LSTM:
            state_dict = torch.load(version_path)
            model_dict = state_dict['model_state_dict']
            prefix = 'lstm.'
            modified_model_dict = {prefix+k: v for k, v in model_dict.items()}
            self.model.load_state_dict(modified_model_dict)
            self.x_scaler = joblib.load(version_path.parents[1] / "scalers" / "x_scaler.joblib")
            self.y_scaler = joblib.load(version_path.parents[1] / "scalers" / "y_scaler.joblib")

        if version_path and type(model) == BidirectionalLSTM:
            mlp_state_dict = torch.load(version_path[0])
            mlp_model_dict = mlp_state_dict['mlp_state_dict']
            self.model.mlp.load_state_dict(mlp_model_dict)

            lstm_f_model_dict = torch.load(version_path[1])["model_state_dict"]
            lstm_b_model_dict = torch.load(version_path[2])["model_state_dict"]

            prefix = 'lstm.'
            modified_model_dict1 = {prefix+k: v for k, v in lstm_f_model_dict.items()}
            modified_model_dict2 = {prefix+k: v for k, v in lstm_b_model_dict.items()}
            self.model.lstm_f.load_state_dict(modified_model_dict1)
            self.model.lstm_b.load_state_dict(modified_model_dict2)

            self.x_scaler = joblib.load(version_path[0].parents[1] / "scalers" / "x_scaler.joblib")
            self.y_scaler = joblib.load(version_path[0].parents[1] / "scalers" / "y_scaler.joblib")


    def _prepare_data(self, x: np.ndarray, y: np.ndarray, ablation_len: int, ablation_start: int) -> dict:
        """
        Prepares data for evaluation.

        Args:
            x: evenly spaced values, no missing values.
            y: values corresponding to x.
            ablation_len: length of time for which data is removed. If None, then a random length is chosen less than half the length 
                of the dataset.
            ablation_start: index at which data is removed. If None, then a random index is chosen. if ablation_len + ablation_start > 
                len(dataset) - 1, then the length of the ablation is reduced to fit the dataset. If ablation start is less than 1, it is 
                set to 1.
            
        Returns:
            dictionary of original x and y values, x and y values with missing interval removed, and the missing intervals.

        Raises:
            Exception: if ablation_len is greater than length of dataset.
        """
        if ablation_start is None:
            ablation_start = np.random.randint(1, len(x) - ablation_len - 1)

        if ablation_len > len(x) - 1:
            raise Exception("Ablation length cannot be greater than length of dataset.")
        if ablation_start < 1:
            ablation_start = 1

        if ablation_len + ablation_start >= len(x):
            warnings.warn("Ablation length and start are too large for dataset. Reducing ablation length.")
            ablation_len = len(x) - ablation_start - 1

        x_ablated = np.concatenate((x[:ablation_start], x[ablation_start + ablation_len:]))
        y_ablated = np.concatenate((y[:ablation_start], y[ablation_start + ablation_len:]))

        x_ablation = x[ablation_start:ablation_start + ablation_len]
        y_ablation = y[ablation_start:ablation_start + ablation_len]

        return {
            "x": x,
            "y": y,
            "x_ablated": x_ablated,
            "y_ablated": y_ablated,
            "x_ablation": x_ablation,
            "y_ablation": y_ablation,
            "new_ablation_len": ablation_len,
            "new_ablation_start": ablation_start
        }


    def evaluate(
        self, 
        dataset_directory: str, 
        ablation_lens: list, 
        ablation_start: int, 
        repetitions: int, 
        plot: bool = False, 
        reverse: bool = False,
        results_name: str = "direct_eval_results",
        units: str = "s"
    ):
        """
        Evaluates model over datasets in a directory. Datasets must have no missing values. The model will predict values of ablated 
        intervals of data from these datasets. The resulting RMSEs are stored in a yaml file.

        Args:
            dataset_directory: path to folder containing csvs of all datasets to evaluate. For the datasets, the prediction values should be in 
                the last column. Make sure datasets are disjoint with training data for LSTM, biLSTM models.
            ablation_lens: list of length of time for which data is removed. If None, then a single random length is chosen.
            ablation_start: index at which data is removed. If None, then a random index is chosen. if ablation_len + ablation_start > 
                len(dataset) - 1, then the length of the ablation is reduced to fit the dataset.
            repetitions: number of times to repeat the experiment for each dataset.
            plot: whether to produce plots of the ablations and predictions. Will block execution until plot is closed.
            reverse: whether to reverse the order of the dataset for bidirectional LSTM models.
            results_name: name of file to store results in. Do not add an extension.
            units: unit of time seperating each observation in the sequence. Options: s, min, h, D, W, MS, YS.
        """
        assert units in ["s", "min", "h", "D", "W", "MS", "YS"], "Invalid unit of time. Options: s, min, h, D, W, MS, YS"

        file_paths = searching_all_files(dataset_directory)

        for _, file_path in enumerate(file_paths):
            final_dict = {}

            RMSE_list = []
            outcome_ablation_lens = []
            outcome_ablation_starts = []

            dataset = pd.read_csv(file_path, index_col=0)
            
            if type(self.model) in [LinearInterpolation, StatsModels]:
                x = dataset.index.values
                y = dataset.values[:,-1]
                y = y.reshape(-1,1)
                y_av = np.mean(y)
            elif type(self.model) in [TimeGPT, TimesFM, TempoGPT]:
                x = dataset.values[:,:-1]
                y = dataset.values[:,-1]
                y = y.reshape(-1,1)
                y_av = np.mean(y)
            else:
                x = dataset.values
                y = dataset.values[:,-1]
                y = y.reshape(-1,1)
                y_av = np.mean(y)
                x = self.x_scaler.transform(x)
                y = self.y_scaler.transform(y)

            if isinstance(self.model, TempoGPT):
                self.model._stl_resolve(mode = "val", data_val = torch.tensor(y), dataset_path=file_path, units=units)

            if reverse:
                x,y = x[::-1], y[::-1]

            if ablation_lens is None:
                ablation_lens = [np.random.randint(math.floor(len(x)/2))]

            criterion = lambda x,y: np.sqrt(np.mean(((x - y)/y_av)) ** 2)

            for ab_length in ablation_lens:
                if isinstance(self.model, TimesFM):
                    self.model = TimesFM(ablation_len=ab_length)
                print(ab_length, flush=True)
                for _ in range(repetitions):
                    data = self._prepare_data(x, y, ab_length, ablation_start)
                    self.model.fit(data["x_ablated"], data["y_ablated"])
                    y_ablation_pred = self.model.predict(data["x_ablation"], data["new_ablation_start"], units=units).reshape(-1,1)
                    if type(self.model) in [LinearInterpolation, StatsModels, TimeGPT, TimesFM, TempoGPT]:
                        RMSE_list.append(criterion(y_ablation_pred, data["y_ablation"]).item())
                    else:
                        y_ablation_pred = y_ablation_pred.detach().numpy()
                        RMSE_list.append(
                            criterion(
                                self.y_scaler.inverse_transform(y_ablation_pred),
                                self.y_scaler.inverse_transform(data["y_ablation"])
                                ).item()
                            )
                        
                    if plot:
                        self._plot(data, data["new_ablation_start"], y_ablation_pred)
                    
                    outcome_ablation_lens.append(data["new_ablation_len"])
                    outcome_ablation_starts.append(data["new_ablation_start"])

                final_dict[str(file_path)] = {
                    "RMSE": RMSE_list, 
                    "Ablation Length": outcome_ablation_lens, 
                    "Ablation Start": outcome_ablation_starts
                }
        yaml.dump(final_dict, open(f"output/{results_name}.yaml", "w"))
    

    def _plot(self, data: dict, ablation_start: int, y_ablation_pred: np.ndarray):
        """
        Plots the original data and the predicted data.

        Args:
            data: dictionary of original x and y values, x and y values with missing interval removed, and the missing intervals.
            ablation_start: index at which data is removed.
            y_ablation_pred: predicted y values for missing interval.

        Notes:
            Needs to be improved so that labeling of the plot is automated.
        """
        if type(self.model) in [LinearInterpolation, StatsModels, TimeGPT, TimesFM]:
            x_fin = data["x"]
            y_fin = data["y"]
            y_ablation_pred_fin = y_ablation_pred
            y_ablated_fin = data["y_ablated"]
        else:
            x_fin = self.x_scaler.inverse_transform(data["x"])
            y_fin = self.y_scaler.inverse_transform(data["y"])
            y_ablation_pred_fin = self.y_scaler.inverse_transform(y_ablation_pred)
            y_ablated_fin = self.y_scaler.inverse_transform(data["y_ablated"])

        plt.plot(
            np.arange(x_fin.shape[0])[ablation_start-50: ablation_start + data["new_ablation_len"] + 50], 
            y_fin[ablation_start-50: ablation_start + data["new_ablation_len"] + 50], 
            label="Original", 
            color="gray"
        )
        
        predicted_interval_x = np.arange(ablation_start-1, ablation_start + data["new_ablation_len"] + 1)
        predicted_interval_y = np.concatenate((
            np.array([y_ablated_fin[ablation_start-1]]), 
            y_ablation_pred_fin, 
            np.array([y_ablated_fin[ablation_start]])
        ))
        
        plt.plot(predicted_interval_x, predicted_interval_y, label="LSTM", linestyle="dashed", color="blue")
        
        # plotting a straight line from the beginning of the ablated interval to the end
        plt.plot([ablation_start-1, ablation_start + data["new_ablation_len"]], [y_ablated_fin[ablation_start-1], y_ablated_fin[ablation_start]], label="Baseline", linestyle="dotted", color="red")

        plt.title("Example Prediction")
        plt.legend()
        plt.show()

