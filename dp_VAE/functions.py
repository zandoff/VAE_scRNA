import math
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import procrustes
from scipy.stats import spearmanr


def QC (datasets, device):

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
    
    return datasets, XS_pairs, splits

def pairwise_corr_expr_spatial(X, S, use_torch=True):
    """
    X: torch.Tensor (n, g) or numpy array
    S: torch.Tensor (n, 2) or numpy array
    returns: r, pvalue, ci_lower, ci_upper, n_pairs
    """
    # convert to numpy
    if isinstance(X, torch.Tensor):
        Xn = X.cpu().numpy()
    else:
        Xn = np.asarray(X)
    if isinstance(S, torch.Tensor):
        Sn = S.cpu().numpy()
    else:
        Sn = np.asarray(S)

    n = Xn.shape[0]
    # 1) pairwise distances (Euclidean)
    # gene-expression distances between spots (rows of X)
    # use sklearn pairwise_distances or numpy broadcasting
    # simple efficient numpy:
    def pdist_sq(A):
        # returns full pairwise Euclidean distances (n,n)
        # using (a-b)^2 = a^2 + b^2 - 2ab
        norms = (A**2).sum(axis=1)
        D2 = norms[:, None] + norms[None, :] - 2.0 * (A @ A.T)
        D2 = np.maximum(D2, 0.0)
        return np.sqrt(D2)

    D_expr = pdist_sq(Xn)   # shape (n,n)
    D_spat = pdist_sq(Sn)   # shape (n,n)

    # 2) flatten upper triangle (i<j)
    iu = np.triu_indices(n, k=1)
    vec_expr = D_expr[iu]
    vec_spat = D_spat[iu]
    n_pairs = vec_expr.shape[0]

    # 3) Pearson correlation
    r, pval = pearsonr(vec_expr, vec_spat)

    # 4) 95% CI via Fisher z
    # beware r can be exactly 1/-1; clip slightly
    r_clip = np.clip(r, -0.9999999, 0.9999999)
    z = np.arctanh(r_clip)
    se = 1.0 / math.sqrt(n_pairs - 3)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    ci_lo = np.tanh(z_lo)
    ci_hi = np.tanh(z_hi)

    return dict(r=r, pval=pval, ci=(ci_lo, ci_hi), n_pairs=n_pairs,
                vec_expr=vec_expr, vec_spat=vec_spat)

def correlation_stats(data_dict):
    corr_results = {}
    
    for key, (X, S) in data_dict.items():
        res = pairwise_corr_expr_spatial(X, S)
        rho, p_rho = spearmanr(res['vec_spat'], res['vec_expr'])

        corr_results[key] = {
            "pearson_r": res['r'],
            "pearson_p": res['pval'],
            "pearson_CI": res['ci'],
            "spearman_rho": rho,
            "spearman_p": p_rho,
            "n_pairs": res['n_pairs']
        }

        print(f"\n=== {key.upper()} ===")
        print("Pearson r = %.4f (p=%.2e, 95%% CI=[%.4f, %.4f])"
              % (res['r'], res['pval'], res['ci'][0], res['ci'][1]))
        print("Spearman rho = %.4f (p=%.2e)" % (rho, p_rho))
        print("n_pairs =", res['n_pairs'])
    return corr_results


def sample_points(n_cells : int):
    """Sample 3 distinct random indices from n_cells."""
    return np.random.choice(n_cells, size=3, replace=False)

def get_coordinates(indices, coordinates, latent_space):
    """Get the coordinates in normal and latent space"""
    normal_coords = coordinates[indices, :2]
    latent_coords = latent_space[indices, :2]
    return normal_coords, latent_coords

def plot_triplets(spatial_pts, latent_pts, spatial_pts2=None, latent_pts2=None, S=None, Z=None, savepath="triplets.png"):
    colors = ['red', 'blue']
    labels = ['Triplet 1', 'Triplet 2']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Spatial space background
    if S is not None:
        axes[0].scatter(S[:,0], S[:,1], s=5, alpha=0.2, color='gray')
    # First triplet points
    for i in range(spatial_pts.shape[0]):
        axes[0].scatter(spatial_pts[i,0], spatial_pts[i,1], c=colors[0], s=100, label=labels[0] if i==0 else None)
    # Second triplet (optional)
    if spatial_pts2 is not None:
        for i in range(spatial_pts2.shape[0]):
            axes[0].scatter(spatial_pts2[i,0], spatial_pts2[i,1], c=colors[1], s=100, label=labels[1] if i==0 else None)
        # connect second triangle
        axes[0].plot([spatial_pts2[0,0], spatial_pts2[1,0], spatial_pts2[2,0], spatial_pts2[0,0]],
                     [spatial_pts2[0,1], spatial_pts2[1,1], spatial_pts2[2,1], spatial_pts2[0,1]],
                     color='blue', linestyle='--')
    # connect first triangle
    axes[0].plot([spatial_pts[0,0], spatial_pts[1,0], spatial_pts[2,0], spatial_pts[0,0]],
                 [spatial_pts[0,1], spatial_pts[1,1], spatial_pts[2,1], spatial_pts[0,1]],
                 color='red', linestyle='--')
    axes[0].set_title("Spatial coords")
    axes[0].legend()

    # Latent space background
    if Z is not None:
        axes[1].scatter(Z[:,0], Z[:,1], s=5, alpha=0.2, color='gray')
    # First latent triplet
    for i in range(latent_pts.shape[0]):
        axes[1].scatter(latent_pts[i,0], latent_pts[i,1], c=colors[0], s=100, label=labels[0] if i==0 else None)
    # Second latent triplet (optional)
    if latent_pts2 is not None:
        for i in range(latent_pts2.shape[0]):
            axes[1].scatter(latent_pts2[i,0], latent_pts2[i,1], c=colors[1], s=100, label=labels[1] if i==0 else None)
        axes[1].plot([latent_pts2[0,0], latent_pts2[1,0], latent_pts2[2,0], latent_pts2[0,0]],
                     [latent_pts2[0,1], latent_pts2[1,1], latent_pts2[2,1], latent_pts2[0,1]],
                     color='blue', linestyle='--')
    # connect first latent triangle
    axes[1].plot([latent_pts[0,0], latent_pts[1,0], latent_pts[2,0], latent_pts[0,0]],
                 [latent_pts[0,1], latent_pts[1,1], latent_pts[2,1], latent_pts[0,1]],
                 color='red', linestyle='--')
    axes[1].set_title("Latent coords")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def test_triplet(model, X, S, seed=42, savepath="triplets.png", mu=None, z=None):
    """Sample a triplet of points and return their spatial + latent coordinates.

    Allows passing precomputed mu or z to avoid full re-encoding cost for large X.
    If both mu and z are None, encodes X once (deterministic using mu as embedding).
    """
    
    np.random.seed(seed)

    device = next(model.parameters()).device
    # Ensure tensors and on CPU for numpy conversion later
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    if isinstance(S, np.ndarray):
        S = torch.tensor(S, dtype=torch.float32, device=device)

    n = X.shape[0]
    
    if n < 3:
        raise ValueError("Need at least 3 points to sample a triangle.")
    
    idx = sample_points(n)
    idx2 = sample_points(n)
    

    # Use provided embedding if available
    if z is None:
        if mu is None:
            with torch.no_grad():
                mu, _ = model.encode(X)
        z = mu  # deterministic embedding

    spatial_pts = S[idx].detach().cpu().numpy()
    latent_pts = z[idx].detach().cpu().numpy()

    spatial_pts2 = S[idx2].detach().cpu().numpy()
    latent_pts2 = z[idx2].detach().cpu().numpy()

    plot_triplets(spatial_pts, latent_pts, spatial_pts2, latent_pts2, S=S.detach().cpu(), Z=z.detach().cpu(), savepath=f"{savepath}")

    return idx, spatial_pts, latent_pts

def triangle_angles(pts):
    """
    Compute the 3 internal angles (in radians) of a triangle
    defined by points pts: array shape (3, 2).
    """
    a = np.linalg.norm(pts[1] - pts[2])
    b = np.linalg.norm(pts[0] - pts[2])
    c = np.linalg.norm(pts[0] - pts[1])
    # cosine rule
    alpha = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
    beta  = np.arccos((a**2 + c**2 - b**2) / (2*a*c))
    gamma = np.pi - alpha - beta
    return np.array([alpha, beta, gamma])


def compare_triangles(spatial_pts, latent_pts):
    """Compare triangle geometry between spatial and latent points.

    Parameters
    ----------
    spatial_pts, latent_pts : array-like (3,2)
        Triangle vertices."""
    
    angles_spatial = triangle_angles(spatial_pts)
    angles_latent  = triangle_angles(latent_pts)

    # angle error in degrees
    angle_err = np.degrees(np.abs(angles_spatial - angles_latent))
    mean_angle_err = angle_err.mean()

    # also compare side length ratios (scale-invariant)
    dS = [np.linalg.norm(spatial_pts[i] - spatial_pts[j]) for i,j in [(0,1),(1,2),(2,0)]]
    dZ = [np.linalg.norm(latent_pts[i] - latent_pts[j]) for i,j in [(0,1),(1,2),(2,0)]]
    ratios = np.array(dZ) / np.array(dS)
    ratios /= ratios.mean() if ratios.mean() != 0 else 1.0

    comparison = {
        "angles_spatial_deg": np.degrees(angles_spatial),
        "angles_latent_deg": np.degrees(angles_latent),
        "angle_error_deg": angle_err,
        "mean_angle_error_deg": mean_angle_err,
        "normalized_side_ratios": ratios
    }
    return comparison

# Procrustes distance
def procrustes_distance(z, s):
    _, _, disparity = procrustes(s.cpu().numpy(), z.cpu().numpy())
    return disparity

def spatial_reconstruction_error(model, X, S, mask=None):
    """
    Compute per-spot reconstruction error of spatial distances.
    Returns vector of errors, one per spot.
    """
    with torch.no_grad():
        mu, _ = model.encode(X)
        z = mu  # deterministic embedding
        Dz = torch.cdist(z, z, p=2)   # latent distances (detached)
        Ds = torch.cdist(S, S, p=2)   # spatial distances
        lam_val = float(model.lam.detach())
        diff = torch.abs(Dz - lam_val * Ds)
        if mask is not None:
            diff = diff * mask
        per_spot_err = diff.mean(dim=1).cpu().numpy()
    return per_spot_err

def plot_spatial_error(S, errors, title, savepath):
    plt.figure(figsize=(6,6))
    plt.scatter(S[:,0], S[:,1], c=errors, cmap="viridis", s=8)
    plt.colorbar(label="Reconstruction error")
    plt.title(title)
    plt.gca().invert_yaxis()  # match Visium convention
    plt.axis("equal")
    plt.savefig(savepath, dpi=150)
    plt.close()

##################
