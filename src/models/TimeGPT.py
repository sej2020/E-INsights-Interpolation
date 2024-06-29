"""
TimeGPT model for time-series forecasting.

Typical usage example:
```python
```
"""
import numpy as np
import pandas as pd
from nixtla import NixtlaClient
from dotenv import load_dotenv
import os
load_dotenv('secret.env')

# api_key = os.getenv("NIXTLA_API_KEY")

# nixtla_client = NixtlaClient(api_key=api_key)
# nixtla_client.validate_api_key()

class TimeGPT:
    """
    Time-series forecasting model using TimeGPT from the Nixtla package.

    Attributes:
        x: evenly spaced values, potentially with missing values.
        y: values corresponding to x.
    """
    def __init__(self):
        """
        Initializes an instance of the TimeGPT class.
        """
        self.x = None
        self.y = None


    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits model to data. Modifies object attributes and returns nothing.

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        self.x = x
        self.y = y

        
    def predict(self, x: np.ndarray, ablation_start: int, units: str = 's') -> np.ndarray:
        """
        Predicts y values for x values.

        Args:
            x: x values.
            ablation_start: index of where the first missing value would be placed in the array fitted to x. For example, if the array 
                fitted to x is [3,4,6,7], the ablation start index should be 2, because the missing value would be in position 2 if the array was
                uninterrupted.
            units: unit of time for the x values. Default is 's' for seconds.

        Returns:
            predicted y values.

        Raises:
            Exception: if model is not fitted.
        """
        if self.y is None:
            raise Exception("Model not fitted.")
        
        ds = pd.date_range(start='1/1/2010', periods=len(self.y[:ablation_start]) + len(x), freq=units)
        df = pd.DataFrame({'ds': ds[:ablation_start], 'y': self.y.flatten()[:ablation_start]})
        for col in range(self.x.shape[1]):
            df[f'x{col}'] = self.x[:ablation_start, col]

        # exo_vars = pd.DataFrame({'ds': ds[ablation_start:]})
        # for col in range(x.shape[1]):
        #     exo_vars[f'x{col}'] = x[:, col]

        fcst = nixtla_client.forecast(df=df, h=len(x), freq=units)
        # fcst = nixtla_client.forecast(df=df, X_df=exo_vars, h=len(x), freq=units)

        return fcst['TimeGPT'].values
