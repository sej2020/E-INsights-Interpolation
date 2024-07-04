"""
TEMPO model for time-series forecasting.
"""
import numpy as np
import torch
import src.models.backend.TEMPO as TEMPO

class TempoGPT:
    """
    Time-series forecasting model using TEMPO - https://github.com/DC-research/TEMPO

    Attributes:
        x: evenly spaced values, potentially with missing values.
        y: values corresponding to x.
        config: configuration for TEMPO checkpoint model
        model: TEMPO model
    """
    def __init__(self):
        """
        Initializes an instance of the TempoGPT class.

        Args:
            ablation_len: length of the ablation
        """
        self.config = TempoConfig()
        self.model = TEMPO.TEMPO(
           self.config,
           device="cpu" 
        )
        best_model_path = "TEMPO_checkpoints/ettm2_TEMPO_3_prompt_learn_336_96_100_sl336_ll168_pl96_dm768_nh4_el3_gl3_df768_ebtimeF_itr0/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        self.x = None
        self.y = None


    def set_str(self, trend_stamp: np.ndarray, seasonal_stamp: np.ndarray, residual_stamp: np.ndarray):
        """
        Sets the trend, seasonal, and residual stamps for the model.

        Args:
            trend_stamp
            seasonal_stamp
            residual_stamp
        """
        self.trend_stamp = trend_stamp
        self.seasonal_stamp = seasonal_stamp
        self.residual_stamp = residual_stamp


    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits model to data. Modifies object attributes and returns nothing.

        Args:
            x: independent variables with an ablation
            y: dependent variable with an ablation
        """
        self.x = x # [pre+post_ab_len, num_features]
        self.y = y # [pre+post_ab_len, 1]

        
    def predict(self, x: np.ndarray, ablation_start: int, units: str = 's') -> np.ndarray:
        """
        Predicts y values for x values.

        Args:
            x: x values [ablation_len, num_features]
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
        
        pre_ablation_context = self.y[:ablation_start] # [pre_ab_len, 1] - same for s,t,r
        trend = self.trend_stamp[:ablation_start]
        seasonal = self.seasonal_stamp[:ablation_start]
        residual = self.residual_stamp[:ablation_start]

        if len(pre_ablation_context) >= self.config.seq_len:
            pre_ablation_context = pre_ablation_context[-self.config.seq_len:] # [model.seq_len, 1] - same for s,t,r
            trend = trend[-self.config.seq_len:]
            seasonal = seasonal[-self.config.seq_len:]
            residual = residual[-self.config.seq_len:]

        else:
            # padding with sequence average
            pre_ablation_context = np.pad(pre_ablation_context,((self.config.seq_len - len(pre_ablation_context), 0), (0,0)), mode='mean') # [model.seq_len, 1] - same for s,t,r
            trend = np.pad(trend,((self.config.seq_len - len(trend), 0), (0,0)), mode='mean')
            seasonal = np.pad(seasonal,((self.config.seq_len - len(seasonal), 0), (0,0)), mode='mean')
            residual = np.pad(residual,((self.config.seq_len - len(residual), 0), (0,0)), mode='mean')

        pre_ablation_context = np.expand_dims(pre_ablation_context, axis=0) # [1, model.seq_len, 1] - same for s,t,r
        trend = np.expand_dims(trend, axis=0)
        seasonal = np.expand_dims(seasonal, axis=0)
        residual = np.expand_dims(residual, axis=0)
        
        outputs, _ = self.model(
            x=pre_ablation_context, 
            itr=0, 
            trend=trend, 
            season=seasonal, 
            noise=residual, 
            test=True
            )
        outputs = outputs[:, -self.config.pred_len:, :]
        outputs = outputs[0, :x.shape[0], :]
        return outputs
        

class TempoConfig:
    def __init__(self):
        self.is_gpt = 1
        self.patch_size = 16
        self.pretrain = 1
        self.stride = 8
        self.seq_len = 336
        self.gpt_layers = 3
        self.prompt = 1
        self.pool = True
        self.d_model = 768
        self.use_token = 0
        self.pred_len = 96
        self.freeze = 1
        self.num_nodes = 1


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
