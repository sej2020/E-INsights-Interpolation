#!/bin/bash

python -m src.actions.evaluate --dataset_directory data/high_var_oct16/test --results_name high_var_timesfm --ablation_lens 15 30 90 --units s --model timesfm
python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_CNC --results_name CNC_timesfm --ablation_lens 15 30 90 --units min --model timesfm
python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M00 --results_name M00_timesfm --ablation_lens 15 30 90 --units min --model timesfm
python -m src.actions.evaluate --dataset_directory data/min_av/amatrol-Mar24/test_HVAC --results_name HVAC_timesfm --ablation_lens 15 30 90 --units min --model timesfm
python -m src.actions.evaluate --dataset_directory data/min_av/OptoMMP-Oct23/test_M02 --results_name M02_timesfm --ablation_lens 15 30 90 --units min --model timesfm
