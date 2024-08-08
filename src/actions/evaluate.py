"""
A standalone script to evaluate a model on a dataset.

Typical usage example:
```bash
>>> python -m src.actions.evaluate --dataset_directory data/processed/processed_data.csv --model lstm --lstm_n_layers 4 --lstm_input_size 9 --lstm_window_size 20 --version_path output/lstm_checkpoint.pth --ablation_lens 30 60 90 --results_name lstm_results
```
"""
import pathlib
from src.models.LSTMs import LSTM, BidirectionalLSTM
from src.models.baseline import LinearInterpolation
from src.experiments.evaluations import DirectEvaluation
from src.models.statsforecast import StatsModels
from src.models.TimeGPT import TimeGPT
from src.models.TimesFM import TimesFM
from src.models.TempoGPT import TempoGPT
from src.models.TempoGPT import TempoConfig
import argparse

parser = argparse.ArgumentParser("evaluation")
parser.add_argument("--dataset_directory", type=str, required=True)

parser.add_argument("--model", type=str, choices=["lstm", "bidi_lstm", "linear", "stats", "timegpt", "timesfm", "tempo"], required=True)
parser.add_argument("--stats_model_type", type=str, choices=["AA", "HW", "SN", "HA", "DOT"], default="AA", help="Model type for the statsmodels model.")
parser.add_argument("--lstm_n_layers", type=int, default=4, help="Number of layers in the lstm or bilstm model.")
parser.add_argument("--lstm_input_size", type=int, default=9, help="Number of features in the dataset if you choose an lstm or bilstm evaluation.")
parser.add_argument("--lstm_window_size", type=int, default=20, help="Number of timesteps in the past to consider for the lstm or bilstm model.")
parser.add_argument("--version_path", nargs="+", type=str, help="Path to the checkpoint file for the lstm or bilstm model. For the bilstm model, the paths should be in the following order: mlp, forward lstm, backward lstm.")
parser.add_argument("--reverse", action=argparse.BooleanOptionalAction, default=False)

parser.add_argument("--ablation_lens", nargs="+", type=int, required=True)
parser.add_argument("--ablation_start", type=int, default=None)
parser.add_argument("--repetitions", type=int, default=100)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--units", type=str, default="s")
parser.add_argument("--results_name", type=str, required=True)

parser.add_argument("--best_tempo_model_path", type=str, default=None)

args = parser.parse_args()

eval_param_dict = {
    "dataset_directory": args.dataset_directory,
    "ablation_lens": args.ablation_lens,
    "ablation_start": args.ablation_start,
    "repetitions": args.repetitions,
    "plot": args.plot,
    "reverse": args.reverse,
    "results_name": args.results_name,
    "units": args.units
}

if args.model == "lstm":
    model = LSTM(input_size=args.lstm_input_size, n_layers=args.lstm_n_layers, window_size=args.lstm_window_size)
    eval = DirectEvaluation(model, version_path=pathlib.Path(args.version_path[0]))
elif args.model == "bidi_lstm":
    model = BidirectionalLSTM(input_size=args.lstm_input_size, n_layers=args.lstm_n_layers, window_size=args.lstm_window_size)
    eval = DirectEvaluation(model, version_path=[pathlib.Path(args.version_path[0]), pathlib.Path(args.version_path[1]), pathlib.Path(args.version_path[2])])
elif args.model == "linear":
    model = LinearInterpolation()
    eval = DirectEvaluation(model)
elif args.model == "stats":
    model = StatsModels(model_type=args.stats_model_type)
    eval = DirectEvaluation(model)
elif args.model == "timegpt":
    model = TimeGPT()
    eval = DirectEvaluation(model)
elif args.model == "timesfm":
    model = TimesFM()
    eval = DirectEvaluation(model)
elif args.model == "tempo":
    if args.best_tempo_model_path is not None:
        cfg = TempoConfig(best_model_path=args.best_tempo_model_path)
        model = TempoGPT(config=cfg)
    else:
        model = TempoGPT()
    eval = DirectEvaluation(model)

eval.evaluate(**eval_param_dict)

print(f"Evaluation complete. Your results are in output/{args.results_name}.yaml", flush=True)
