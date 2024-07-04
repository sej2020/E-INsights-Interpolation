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

To be able to use TimesFM:
1. clone the [TimesFM repo](https://github.com/google-research/timesfm/tree/master) to the same parent folder as this repo
2. complete the TimesFM environment setup
3. install pytorch, seaborn, and dotenv into the TimesFM environment
4. run scripts in the E-INsights-Interpolation from the TimesFM environment

To be able to use TEMPO:
1. activate the TimesFM environment (tfm_env) and run `pip install peft`
2. download TEMPO model checkpoints from [Google Drive](https://drive.google.com/file/d/11Ho_seP9NGh-lQCyBkvQhAQFy_3XVwKp/view) and make sure the top level of this folder is named 'TEMPO_checkpoints/'