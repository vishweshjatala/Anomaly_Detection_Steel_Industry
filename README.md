# Continual Anomaly Detection for Evolving Time Series in Steel Industry

This repository contains code for anomaly detection in steel-industry time-series data. It includes PCA- and neural-network-based models, data preprocessing, model training, and threshold tuning.

> **Data and configuration notice:** The industrial data and region configuration files are not included in this repository. The preprocessing code expects these files in a local directory structure and currently contains absolute paths. Update the paths in `preprocess.py` to match your local data and configuration directories before running the commands below. Do not commit private or restricted data.

## Repository contents

- `preprocess.py` — loads the time-series data and region configuration, and prepares training and evaluation data.
- `trainer.py` — trains a model and saves its artifacts and anomaly scores.
- `tuner.py` — evaluates trained models and tunes anomaly-detection thresholds.
- `continual.py` — runs continual-learning experiments.
- `scaler.py`, `lstm_basic.py`, `unet.py`, `usad.py` — scaling and model implementations.
- `requirements.txt` — Python package dependencies.

## Requirements

- Python 3.9 or later
- pip and a virtual-environment tool
- A CUDA-capable NVIDIA GPU and compatible CUDA-enabled PyTorch installation for the `lstm`, `usad`, and `unet` models. The current training implementation calls `.cuda()` for these models. PCA can run on a CPU.
- Access to the required Parquet data, preprocessing pickle files, cobble-interval spreadsheet, and region YAML configuration (see [Data and configuration](#data-and-configuration)).

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
```

For GPU training, install the PyTorch build that matches your CUDA/driver setup by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data and configuration

`preprocess.py` currently reads from fixed absolute paths. Update the paths in `get_data()` and `load_region()` to point to your local files. For each profile (for example, `12mm`), the code expects a layout equivalent to:

```text
<DATA_ROOT>/<profile>/
├── *.parquet
└── preprocessed/
    ├── <profile>_autoencoder_train_data.pkl
    ├── <profile>_hypertuning_data.pkl
    └── <profile>_cobble_intervals.xlsx

<REGIONS_DIR>/<region>.yaml
```

The region name must have a matching YAML file. The code's command-line options currently accept these profiles: `12mm`, `16mm`; and these regions: `Stand_1-6`, `Stand_7-12`, `Stand_13-18`, `FFB_L1`, `FFB_L2`.

## Run the code

Run commands from the repository root after installing dependencies and configuring the data paths.

### Train a model

For example, train an LSTM model for profile `12mm` and region `FFB_L1`:

```bash
python trainer.py \\
  --profile 12mm \\
  --region FFB_L1 \\
  --model lstm \\
  --epochs 100 \\
  --lr 0.001 \\
  --seq_len 30 \\
  --bs 128 \\
  --patience 15
```

The supported models in `trainer.py` are `lstm`, `pca`, `usad`, and `unet`. To use the region-aware custom scaler, add `--custom_scaler`. See `python trainer.py --help` for the available options.

### Tune the anomaly threshold

After training, tune the threshold for the same profile, region, and model:

```bash
python tuner.py \\
  --profile 12mm \\
  --region FFB_L1 \\
  --model lstm
```

The tuner writes its results to the corresponding model output directory. It supports the optional `--custom_scaler` flag; use it if the model was trained with that flag.

### Run continual learning

For a continual-learning run, use `continual.py` with the same profile, region, and model options:

```bash
python continual.py \\
  --profile 12mm \\
  --region FFB_L1 \\
  --model lstm \\
  --epochs 50 \\
  --lr 0.001 \\
  --seq_len 30 \\
  --bs 16
```

This workflow requires the same data and region configuration described above and may take substantially longer than a single training run.

## Outputs

Training artifacts are saved under `trained_models/`, organized by profile, region, scaler type, and model. Tuning results are written to the relevant model directory.

## Reproducibility

Record the profile, region, model, command-line options, dependency versions, and hardware used for each experiment. Results depend on the input data and configuration files, which are not distributed in this repository.
