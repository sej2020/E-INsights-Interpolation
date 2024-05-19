"""Implements StatsModels class for time-series forecasting using the statsforecast package.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple
from statsforecast.models import (
    AutoARIMA as AA,
    HoltWinters as HW,
    SeasonalNaive as SN,
    HistoricAverage as HA,
    DynamicOptimizedTheta as DOT
)

import time

class StatsModels:
    """
    Time-series forecasting models from the statsforecast package.

    Attributes:
        x: evenly spaced values, potentially with missing values.
        y: values corresponding to x.
        statsmodel: model from the statsforecast package.
    """
    def __init__(self, model_type: str = "AA", season_length: int = 86400):
        """
        Initializes an instance of the StatsModels class.

        Args:
            model_type: name of the model to use. Options are "AA" for AutoARIMA, "HW" for HoltWinters, "SN" for SeasonalNaive, 
                "HA" for HistoricAverage, "DOT" for DynamicOptimizedTheta.
            season_length: length of the season. Only used for SeasonalNaive and DynamicOptimizedTheta models.
        
        Raises:
            ValueError: if model_type is not one of the options.
        """
        self.x = None
        self.y = None
        if model_type == "AA":
            self.statsmodel = AA()
        elif model_type == "HW":
            self.statsmodel = HW(season_length=season_length)
        elif model_type == "SN":
            self.statsmodel = SN(season_length=season_length)
        elif model_type == "HA":
            self.statsmodel = HA()
        elif model_type == "DOT":
            self.statsmodel = DOT(season_length=season_length)
        else:
            raise ValueError("Invalid model type.")

        self.model_type = model_type
        self.season_length = season_length


    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fits model to data. Modifies object attributes and returns nothing.

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        self.y = y.squeeze()

        
    def predict(self, x: np.ndarray, ablation_start: int) -> np.ndarray:
        """Predicts y values for x values.

        Args:
            x: x values.
            ablation_start: index of where the first missing value would be placed in the array fitted to x. For example, if the array 
                fitted to x is [3,4,6,7], the ablation start index should be 2, because the missing value would be in position 2 if the array was
                uninterrupted.

        Returns:
            predicted y values.

        Raises:
            Exception: if model is not fitted.
        """
        if self.y is None:
            raise Exception("Model not fitted.")
        
        if ablation_start > 1200:
            self.statsmodel.fit(y = self.y[ablation_start-1200:ablation_start])
        else:
            self.statsmodel.fit(y = self.y[:ablation_start])
        
        predictions = self.statsmodel.predict(h=len(x))
        return predictions['mean']
        
    
    

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 6, 7])
    y = np.array([20, 19, 18, 17, 11, 10])
    model = StatsModels()
    model.fit(x, y)
    print(model.predict(np.array([5]), ablation_start=4))
    complete_x, complete_y = model.interpolate()
    print(complete_x)
    print(complete_y)
    plt.plot(complete_x, complete_y)
    plt.show()