import dp_VAE as dp
import functions as FN
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy.plotting._utils")
import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import leidenalg
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from scipy.spatial import procrustes
import igraph as ig
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.spatial import procrustes
from scipy.stats import spearmanr

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Visium datasets from scanpy
    # Sagittal Posterior
    adata_post = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Posterior")
    
    adata_post2 = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Posterior_Section_2")
    
    # Sagittal Anterior 
    adata_ant = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior")
    
    adata_ant2 = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior_Section_2")
    
    # Whole Brain
    adata_whole = sc.datasets.visium_sge(sample_id="V1_Adult_Mouse_Brain")
    
    # Kidney
    adata_kidney = sc.datasets.visium_sge(sample_id="V1_Mouse_Kidney")
    
    datasets = {
        "sagittal_posterior": adata_post,
        "sagittal_posterior_2": adata_post2,
        "sagittal_anterior": adata_ant,
        "sagittal_anterior_2": adata_ant2,
        "whole_brain": adata_whole,
        "kidney": adata_kidney
    }
    
    # Quality control
    datasets, XS_pairs, splits = FN.QC(datasets, device)

    # Correlation statistics
    FN.correlation_stats(XS_pairs)


if __name__ == "__main__":
    main()