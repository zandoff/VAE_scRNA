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

Once the repository is cloned the user will have access to all the files needed.

### Required packages

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
