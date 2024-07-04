"""
TimesFM model for time-series forecasting.
"""
import numpy as np
import timesfm

class TimesFM:
    """
    Time-series forecasting model using TimesFM - https://github.com/google-research/timesfm

    Attributes:
        x: evenly spaced values, potentially with missing values.
        y: values corresponding to x.
        ablation_len: length of the ablation
        tfm: TimesFM model
    """
    def __init__(self, ablation_len: int = 128):
        """
        Initializes an instance of the TimesFM class.

        Args:
            ablation_len: length of the ablation
        """
        self.ablation_len = ablation_len
        self.tfm = timesfm.TimesFm(
            context_len=128,
            horizon_len=ablation_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        )
        self.tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
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
        
        assert self.ablation_len == x.shape[0], f"Ablation length does not match the length of the input: {self.ablation_len} != {x.shape[0]}"
        
        pre_ablation_context = self.x[:ablation_start]
        forecast_input = [pre_ablation_context[:,col] for col in range(self.x.shape[1])]
        forecast_input.append(self.y[:ablation_start, 0])

        point_forecast, _ = self.tfm.forecast(
            forecast_input
        )
        return point_forecast[-1, :]
    
if __name__ == '__main__':
    tfm = TimesFM()
    x = np.ones((150300, 8))
    y = np.ones((150300, 1))
    tfm.fit(x,y)
    print(tfm.predict(np.zeros((20, 8)), 20).shape, flush=True)
