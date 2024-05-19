"""
A standalone script to evaluate a run an evaluation over a model. Modify the direct_eval_param_dict to change the parameters 
of the evaluation. You can change the model and evaluation method by changing the model and eval variables at the bottom.
Right now, the evaluation method is DirectEvaluation, but more will be added in the future.

Typical usage example:
```bash
>>> python -m src.actions.evaluate
```
"""
import pathlib
from src.models.LSTMs import LSTM, BidirectionalLSTM
from src.models.baseline import LinearInterpolation
from src.experiments.evaluations import DirectEvaluation
from src.models.statsforecast import StatsModels


#================================================

# Direct Evaluation

direct_eval_param_dict = {
    # string
    "directory" : "data/high_var_oct16/final",
    # list or None (in seconds)
    "ablation_lens" : [30, 90, 300],
    # int or None
    "ablation_start" : None,
    # int
    "repetitions" : 1000,
    # bool
    "plot" : False,
    # string, no file extension
    "results_name" : f"testing_baseline"
}

### LSTM Model ###
# model = LSTM(input_size=9, n_layers=4, window_size=20)
# version_path = pathlib.Path("logs/hyp_tune_1_logs/hyp_tune_1_l4_lr0.001_ws20_ep250/checkpoints/checkpt_e249.pt")
# eval = DirectEvaluation(model, version_path=version_path)
# eval.evaluate(**direct_eval_param_dict)

### Linear Interpolation Model ###
model = LinearInterpolation()
eval = DirectEvaluation(model)
eval.evaluate(**direct_eval_param_dict)

### StatsModels Model ###
# model = StatsModels(model_type="AA")
# eval = DirectEvaluation(model)
# eval.evaluate(**direct_eval_param_dict)

### Bidirectional LSTM Model ###
# model = BidirectionalLSTM(input_size=9, n_layers=4, window_size=20)
# version_path = [pathlib.Path("logs/hyp_tune_bidi_1_logs/bidi_ht1_lr0.001_e50/checkpoints/checkpt_e49.pt"),
#                 pathlib.Path("logs/hyp_tune_bidi_1_logs/bidi_ht1_lr0.001_e50/bidi_ht1_lr0.001_e50_lstm1/checkpoints/checkpt_e49.pt"),
#                 pathlib.Path("logs/hyp_tune_bidi_1_logs/bidi_ht1_lr0.001_e50/bidi_ht1_lr0.001_e50_lstm2/checkpoints/checkpt_e49.pt")
#                 ]
# eval = DirectEvaluation(model, version_path=version_path)
# eval.evaluate(**direct_eval_param_dict)
