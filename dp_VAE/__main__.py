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
    
    XS_pairs = {}
    splits = {}
    
    # Preprocess the data
    # Before filtering
    for key, adata in datasets.items():
        n_cells_start = adata.n_obs
        n_genes_start = adata.n_vars
        adata.var_names_make_unique()
    
    
        # Filter cells with min counts
        sc.pp.filter_cells(adata, min_counts=4000)
        n_filtered_min = n_cells_start - adata.n_obs
        print(f"filtered out {n_filtered_min} cells that have less than 4000 counts")
    
        # Filter cells with max counts
        n_cells_min = adata.n_obs
        sc.pp.filter_cells(adata, max_counts=38000)
        n_filtered_max = n_cells_min - adata.n_obs
        print(f"filtered out {n_filtered_max} cells that have more than 38000 counts")
    
        # Mitochondrial filter
        # Mark mitochondrial genes - for mouse data mt genes usually start with "mt-"
        adata.var['mt'] = adata.var_names.str.startswith('mt-')  
    
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
        # Now you can filter cells based on mitochondrial content
        adata = adata[adata.obs['pct_counts_mt'] < 20].copy()
        n_cells_max = adata.n_obs
        n_filtered_mt = n_cells_max - adata.n_obs
        print(f"filtered out {n_filtered_mt} cells with pct_counts_mt ≥ 20%")
        print(f"#cells after MT filter: {adata.n_obs}")
    
        # Filter genes
        n_genes_before = adata.n_vars
        sc.pp.filter_genes(adata, min_cells=10)
        n_genes_filtered = n_genes_before - adata.n_vars
        print(f"filtered out {n_genes_filtered} genes that are detected in less than 10 cells")
    
        # Normalize and log-transform (paper method)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
        # Select top 250 highly variable genes (paper method)
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=250)
        adata = adata[:, adata.var['highly_variable']].copy()
        print(f"Subsetted to top 250 highly variable genes. Shape: {adata.shape}")
    
        # Extract gene expression matrix X and spatial coordinates S
        X = torch.tensor(adata.X.toarray(), dtype=torch.float32, device=device)
        # Fix spatial coordinates (ensure float array, not object)
        S_raw = np.array(adata.obsm["spatial"], dtype=float)
        S_raw = torch.tensor(S_raw, dtype=torch.float32, device=device)
    
        S_mu = S_raw.mean(dim=0, keepdim=True)
        S_std = S_raw.std(dim=0, keepdim=True)
        S = (S_raw - S_mu) / S_std  # Standardize
    
    
        print(f"Data shape after preprocessing: {X.shape}")
        print(f"Coordinates shape: {S.shape}")
    
        XS_pairs[key] = (X, S)
        # Train/test split (point 1)
        n = X.shape[0]
        idx = torch.randperm(n, device=device)
        n_test = max(1, int(0.2 * n))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        splits[key] = (X[train_idx], S[train_idx], X[test_idx], S[test_idx])
    
        print(f"[{key}] Preprocessed: X {X.shape}, S {S.shape}")
