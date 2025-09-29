<div align="center">

# Distance-preseving Variational AutoEncoder applied to spatial transcriptomics data

</div>

## Introduction

This repository provides scripts to reproduce a distance-preserving variatonal autoencoder based on the following paper:

<details>

<summary>
Zhou, Wenbin and Du, Jin-Hong. Distance-preserving spatial representations in genomic data. 2024
</summary>

```bibtex
@article{zhou2024distance,
  title={Distance-preserving spatial representations in genomic data},
  author={Zhou, Wenbin and Du, Jin-Hong},
  journal={arXiv preprint arXiv:2408.00911},
  year={2024}
}
```
</details>

This repository includes QC and preprocessing steps, parameter sweeping, model training and evaluation, heatmaps reconstruction and geometric tests.

The repository is designed to work on 10x Visium spatial transcriptome datasets and includes as a default some datasets available inside the [scanpy] library:
- V1_Mouse_Brain_Sagittal_Posterior
- V1_Mouse_Brain_Sagittal_Posterior_Section_2
- V1_Mouse_Brain_Sagittal_Anterior
- V1_Mouse_Brain_Sagittal_Anterior_Section_2
- V1_Adult_Mouse_Brain
- V1_Mouse_Kidney

<div align="center">

## Installation
</div>

From terminal go into an empty folder and clone this repository:

```shell
git clone https://github.com/zandoff/VAE_scRNA
```

Once the repository is cloned the user will have access to all required files.

<div align="center">

Required packages

</div>

To install all required packages, run the following command:

```shell
pip install -r requirements.txt
```

This will install all necessary dependencies including PyTorch (with CUDA support), scanpy, and other scientific computing libraries needed to run the code. If you prefer to use a virtual environment (recommended), you can create one first:

```shell
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n vae_scrna python=3.10
conda activate vae_scrna

# Then install requirements
pip install -r requirements.txt
```

The directory structure should look like this:

```
├── data/                 <- Folder containing all datasets
├── tests/                <- Folder containing all test files
│   ├── __init__.py
│   ├── test_geometry_utils.py
│   ├── test_mask_k.py
│   └── test_training.py
├── dp_VAE/
│   ├── __init__.py
│   ├── __main__.py
│   ├── dp_VAE.py             <- Module for VAE implementation
│   ├── train_eval.py         <- Module for training loop and evaluation
│   ├── utils.py              <- Module with utilities
├── .gitignore
├── .gitattributes
├── requirements.txt  <- List of required libraries
├── setup.py
├── pyproject.toml
├── LICENSE
└── README.md
```

Note: The `data/` folder is not included when you clone the repository. It will be created automatically when you first run the script, and the required datasets will be downloaded at that time.

<div align="center">

## Running
</div>

After installing the requirements, you can run the script using the following commands from the terminal:

### Basic usage

To run the full pipeline with default parameters (all datasets, default hyperparameter ranges):

```bash
# Navigate to the repository directory if not already there
cd VAE_scRNA

# Run the script with default parameters
python -m dp_VAE run
```

### Advanced usage

You can customize various parameters:

```bash
# Run with specific datasets
python -m dp_VAE run --datasets sagittal_posterior sagittal_anterior

# Run with custom hyperparameters
python -m dp_VAE run --alpha2 5 10 20 --mask_k None 5 10 --lam_factors 0.1 0.5 1

# Specify output directory
python -m dp_VAE run --output_dir ./my_results

# Run on specific device
python -m dp_VAE run --device cuda  # or --device cpu
```

### Available parameters

- `--datasets`: Specific datasets to use (options: sagittal_posterior, sagittal_posterior_2, sagittal_anterior, sagittal_anterior_2, whole_brain, kidney)
- `--alpha2`: Values for alpha2 parameter (DP loss weight)
- `--mask_k`: Values for mask_k parameter (use 'None' for no masking)
- `--lam_factors`: Scaling factors for lambda
- `--max_epochs`: Maximum number of training epochs (default: 2000)
- `--patience`: Patience for early stopping (default: 200)
- `--output_dir`: Directory for output files (default: ./results)
- `--device`: Device to use (cuda or cpu)

### Using pre-trained models

After training, you can use the pre-trained models for analysis without retraining:

```bash
# Analyze pre-trained models on their original datasets
python -m dp_VAE analyze --model_dir ./results
```

This will:
- Load the pre-trained models from the specified directory
- Run the analysis steps (triplet geometry and heatmaps) without retraining
- Save the analysis results in a subfolder named "analysis" within your model directory

### Cross-dataset analysis

You can also apply a model trained on one dataset to analyze different datasets:

```bash
# Apply a model trained on one dataset to different datasets
python -m dp_VAE analyze --model_dir ./results --source_dataset sagittal_posterior --target_datasets sagittal_anterior whole_brain
```

This feature allows you to:
- Test how well a model generalizes to other datasets
- Apply a well-trained model to new or smaller datasets
- Compare performance across different spatial transcriptomics samples