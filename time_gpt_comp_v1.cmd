python -m src.actions.evaluate --dataset_directory data/high_var_oct16/test --results_name high_var_timesfm --repetitions 2 --ablation_lens 15 30 90 --units s --model timesfm

@REM python -m src.actions.evaluate --dataset_directory data/high_var_oct16/test --results_name high_var_lstm --ablation_lens 15 30 90 --units s --model lstm --lstm_n_layers 4 --lstm_input_size 9 --lstm_window_size 20 --version_path logs/time_gpt_comp_v1/high_var/hyp_tune_1_l4_lr0.001_ws20_ep250/checkpoints/checkpt_e249.pt
@REM python -m src.actions.evaluate --dataset_directory data/high_var_oct16/test --results_name high_var_linear --ablation_lens 15 30 90 --units s --model linear
@REM python -m src.actions.evaluate --dataset_directory data/high_var_oct16/test --results_name high_var_exogpt --ablation_lens 15 30 90 --units s --model timegpt

@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_CNC --results_name CNC_lstm --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 6 --lstm_input_size 18 --lstm_window_size 10 --version_path logs/time_gpt_comp_v1/CNC_VF5v2/lay-6_lr-0.0005_ws-10/checkpoints/checkpt_e59.pt
@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_CNC --results_name CNC_lstm_rev --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 6 --lstm_input_size 18 --lstm_window_size 10 --version_path logs/time_gpt_comp_v1/CNC_VF5v2/lay-6_lr-0.001_ws-10_rev/checkpoints/checkpt_e9.pt --reverse
@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_CNC --results_name CNC_linear --ablation_lens 15 30 90 --units min --model linear
@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_CNC --results_name CNC_exogpt --ablation_lens 15 30 90 --units min --model timegpt

@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_HVAC --results_name HVAC_lstm --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 6 --lstm_input_size 18 --lstm_window_size 20 --version_path logs/time_gpt_comp_v1/HVAC_RTUv2/lay-6_lr-0.0005_ws-20/checkpoints/checkpt_e59.pt
@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_HVAC --results_name HVAC_lstm_rev --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 6 --lstm_input_size 18 --lstm_window_size 20 --version_path logs/time_gpt_comp_v1/HVAC_RTUv2/lay-6_lr-0.001_ws-20_rev/checkpoints/checkpt_e59.pt --reverse
@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_HVAC --results_name HVAC_linear --ablation_lens 15 30 90 --units min --model linear
@REM python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_HVAC --results_name HVAC_exogpt --ablation_lens 15 30 90 --units min --model timegpt

@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M00 --results_name M00_lstm --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 3 --lstm_input_size 9 --lstm_window_size 10 --version_path logs/time_gpt_comp_v1/M00_PhA/lay-3_lr-0.001_ws-10_/checkpoints/checkpt_e59.pt
@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M00 --results_name M00_lstm_rev --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 4 --lstm_input_size 9 --lstm_window_size 10 --version_path logs/time_gpt_comp_v1/M00_PhA/lay-4_lr-0.001_ws-10_rev/checkpoints/checkpt_e29.pt --reverse
@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M00 --results_name M00_linear --ablation_lens 15 30 90 --units min --model linear
@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M00 --results_name M00_exogpt --ablation_lens 15 30 90 --units min --model timegpt

@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M02 --results_name M02_lstm --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 4 --lstm_input_size 9 --lstm_window_size 10 --version_path logs/time_gpt_comp_v1/M02_PhC/lay-4_lr-0.001_ws-10_/checkpoints/checkpt_e59.pt
@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M02 --results_name M02_lstm_rev --ablation_lens 15 30 90 --units min --model lstm --lstm_n_layers 3 --lstm_input_size 9 --lstm_window_size 10 --version_path logs/time_gpt_comp_v1/M02_PhC/lay-3_lr-0.001_ws-10_rev/checkpoints/checkpt_e59.pt --reverse
@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M02 --results_name M02_linear --ablation_lens 15 30 90 --units min --model linear
@REM python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M02 --results_name M02_exogpt --ablation_lens 15 30 90 --units min --model timegpt
