# Energy INsights Transformer Interpolation Model
Energy INsights has partnered with Indiana University for research in a.) carbon-efficient ML infrastructure and b.) ML models for energy-usage time series analytics. This repository is the workspace for creation of an interpolation model based on the transformer architecture for energy-usage time series data from Indiana manufacturing centers.

How to configure this repo:
1. clone the repository
2. install miniconda - https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
3. run `conda env create -f environment.yml`
4. run `conda activate e-insights-interp`
5. run `conda install -c conda-forge tensorboard` to install tensorboard

To be able to use TimeGPT:
1. run `conda install pip` and then `where pip` or `which -a pip` to locate pip executable path in conda environment
2. run `(your pip executable path ending with .exe) install nixtla`
3. go to https://nixtlaverse.nixtla.io/nixtla/docs/getting-started/setting_up_your_api_key.html and follow procedure 2b.
