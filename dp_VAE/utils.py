import dp_VAE as dp
import math
import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import procrustes
from scipy.stats import spearmanr
import os

def QC (datasets: dict, device: torch.device):
    """
    Perform QC and preprocessing on a dict of AnnData datasets.

    Parameters
    ----------
    datasets : dict[str, anndata.AnnData]
        Mapping from dataset key to AnnData object. Each AnnData is expected
        to have raw counts in `.X` and spatial coordinates in `obsm["spatial"]`.
    device : torch.device
        Target device for returned tensors (CPU or CUDA).

    Returns
    -------
    datasets : dict[str, anndata.AnnData]
        The same mapping, with in-place filtering applied (cells/genes subset).
    XS_pairs : dict[str, tuple[torch.Tensor, torch.Tensor]]
        For each dataset key, a tuple `(X, S)` where:
        - `X`: float32 tensor of shape (n_cells, n_genes) on `device`, normalized
          and log1p-transformed, subset to top 250 highly variable genes.
        - `S`: float32 tensor of shape (n_cells, 2) on `device`, standardized per
          coordinate to mean 0 and std 1.
    splits : dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        For each dataset key, a tuple `(X_train, S_train, X_test, S_test)`
        corresponding to an 80/20 random split of rows. Tensors are on `device`.

    Notes
    -----
    The following preprocessing is applied per dataset:
    - Make gene names unique via `var_names_make_unique()`.
    - Filter cells: min total counts ≥ 4,000 and max total counts ≤ 38,000.
    - Compute QC metrics with mitochondrial genes flagged as names starting with
      "mt-" (mouse convention), then filter cells with `pct_counts_mt` < 20%.
    - Filter genes detected in at least 10 cells.
    - Normalize total counts per cell to 1e4 and apply `log1p` transform.
    - Select top 250 highly variable genes (Seurat flavor) and subset `adata`.
    - Convert `.X` to a dense tensor via `.toarray()` (beware memory for large data).
    - Standardize spatial coordinates `S` per axis: `(S - mean) / std`.
    - Create an 80/20 random train/test split on rows.

    Examples
    --------
    >>> datasets, XS_pairs, splits = QC(datasets, torch.device('cpu'))
    >>> X, S = XS_pairs['sagittal_posterior']
    >>> X.shape, S.shape
    ((n_cells, 250), (n_cells, 2))
    """

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
    Compute Pearson correlation (and 95% CI) between pairwise distances in
    expression space (X) and spatial coordinates (S).

    Parameters
    ----------
    X : torch.Tensor | numpy.ndarray, shape (n_cells, n_genes)
        Gene expression matrix. If a Tensor, it will be moved to CPU for
        distance computation; dtype should be float32/float64.
    S : torch.Tensor | numpy.ndarray, shape (n_cells, 2)
        Spatial coordinates (e.g., Visium spot positions). If a Tensor,
        it will be moved to CPU; dtype should be float32/float64.
    use_torch : bool, optional
        Unused placeholder (kept for backward compatibility).

    Returns
    -------
    out : dict
        Dictionary with keys:
        - 'r' (float): Pearson correlation coefficient.
        - 'pval' (float): Two-sided p-value for r.
        - 'ci' (tuple[float, float]): 95% CI via Fisher's z-transform.
        - 'n_pairs' (int): Number of unique pairs (n_cells choose 2).
        - 'vec_expr' (numpy.ndarray): Upper-triangular pairwise distances from X.
        - 'vec_spat' (numpy.ndarray): Upper-triangular pairwise distances from S.

    Notes
    -----
    - Distances are Euclidean; only the upper triangle (i < j) is used.
    - r is clipped to (-0.9999999, 0.9999999) before Fisher transform.

    See Also
    --------
    correlation_stats : Batch version that prints stats per dataset.
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
    """
    Compute and print Pearson/Spearman stats for multiple datasets.

    Parameters
    ----------
    data_dict : dict[str, tuple[torch.Tensor | numpy.ndarray, torch.Tensor | numpy.ndarray]]
        Mapping of dataset key to a tuple (X, S).

    Returns
    -------
    corr_results : dict[str, dict]
        For each key, a dictionary with 'pearson_r', 'pearson_p', 'pearson_CI',
        'spearman_rho', 'spearman_p', and 'n_pairs'. Also prints a formatted
        summary to stdout as a side effect.
    """
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


def sample_points(n_cells: int):
    """
    Sample three distinct random indices from 0..n_cells-1.

    Parameters
    ----------
    n_cells : int
        Total number of points.

    Returns
    -------
    idx : numpy.ndarray, shape (3,)
        Random indices without replacement.
    """
    return np.random.choice(n_cells, size=3, replace=False)

def get_coordinates(indices, coordinates, latent_space):
    """
    Extract coordinates for given indices from spatial and latent arrays.

    Parameters
    ----------
    indices : array-like, shape (k,)
        Integer indices to select.
    coordinates : numpy.ndarray, shape (n, 2)
        Spatial coordinates array.
    latent_space : numpy.ndarray, shape (n, 2)
        Latent coordinates array.

    Returns
    -------
    normal_coords : numpy.ndarray, shape (k, 2)
        Selected rows from `coordinates`.
    latent_coords : numpy.ndarray, shape (k, 2)
        Selected rows from `latent_space`.
    """
    normal_coords = coordinates[indices, :2]
    latent_coords = latent_space[indices, :2]
    return normal_coords, latent_coords

def plot_triplets(spatial_pts, latent_pts, spatial_pts2=None, latent_pts2=None, S=None, Z=None, savepath="triplets.png"):
    """
    Plot two triplets over spatial and latent backgrounds and save to disk.

    Parameters
    ----------
    spatial_pts : numpy.ndarray, shape (3, 2)
        First spatial triplet.
    latent_pts : numpy.ndarray, shape (3, 2)
        First latent triplet.
    spatial_pts2 : numpy.ndarray, shape (3, 2), optional
        Second spatial triplet.
    latent_pts2 : numpy.ndarray, shape (3, 2), optional
        Second latent triplet.
    S : numpy.ndarray, shape (n, 2), optional
        Background spatial coordinates to scatter in gray.
    Z : numpy.ndarray, shape (n, 2), optional
        Background latent coordinates to scatter in gray.
    savepath : str, optional
        Path to save the PNG.
    """
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
    """
    Sample two random triplets and return the first; save a comparison plot.

    Allows passing precomputed mu or z to avoid full re-encoding cost for large X.
    If both mu and z are None, encodes X once (deterministic using mu as embedding).

    Parameters
    ----------
    model : torch.nn.Module
        Trained model exposing `encode(X) -> (mu, logvar)`.
    X : torch.Tensor | numpy.ndarray, shape (n_cells, n_genes)
        Expression matrix.
    S : torch.Tensor | numpy.ndarray, shape (n_cells, 2)
        Spatial coordinates.
    seed : int, optional
        RNG seed for reproducible triplet sampling.
    savepath : str, optional
        File path to save the visualization.
    mu : torch.Tensor, shape (n_cells, 2), optional
        Precomputed mean embeddings.
    z : torch.Tensor, shape (n_cells, 2), optional
        Precomputed latent coordinates; if provided, used directly.

    Returns
    -------
    idx : numpy.ndarray, shape (3,)
        Indices of the first triplet.
    spatial_pts : numpy.ndarray, shape (3, 2)
        Spatial coordinates for the first triplet.
    latent_pts : numpy.ndarray, shape (3, 2)
        Latent coordinates for the first triplet.

    Raises
    ------
    ValueError
        If fewer than 3 points are available in `X`.
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
    Compute the three internal angles (in radians) of a triangle.

    Parameters
    ----------
    pts : numpy.ndarray, shape (3, 2)
        Triangle vertices.

    Returns
    -------
    angles : numpy.ndarray, shape (3,)
        Angles (alpha, beta, gamma) in radians.
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
    """
    Compare triangle geometry between spatial and latent points.

    Parameters
    ----------
    spatial_pts : numpy.ndarray, shape (3, 2)
        Spatial triangle vertices.
    latent_pts : numpy.ndarray, shape (3, 2)
        Latent triangle vertices.

    Returns
    -------
    comparison : dict
        Contains:
        - 'angles_spatial_deg': angles of spatial triangle (degrees)
        - 'angles_latent_deg' : angles of latent triangle (degrees)
        - 'angle_error_deg'   : absolute per-angle errors (degrees)
        - 'mean_angle_error_deg' : average angle error (degrees)
        - 'normalized_side_ratios': side ratios in latent normalized by mean
    """
    
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
    """
    Compute Procrustes disparity between latent and spatial coordinates.

    Parameters
    ----------
    z : torch.Tensor, shape (n, 2)
        Latent coordinates.
    s : torch.Tensor, shape (n, 2)
        Spatial coordinates.

    Returns
    -------
    disparity : float
        The Procrustes disparity (lower is better), as returned by SciPy.
    """
    _, _, disparity = procrustes(s.cpu().numpy(), z.cpu().numpy())
    return disparity

def spatial_reconstruction_error(model, X, S, mask=None):
    """
    Compute per-spot mean absolute discrepancy in the distance-preserving term.

    Parameters
    ----------
    model : torch.nn.Module
        Model exposing `.encode(X)` and scalar attribute `.lam`.
    X : torch.Tensor, shape (n, g)
        Expression matrix on the same device as `model`.
    S : torch.Tensor, shape (n, 2)
        Spatial coordinates on the same device.
    mask : torch.Tensor | None, shape (n, n), optional
        Optional binary mask to restrict pairwise contributions.

    Returns
    -------
    errors : numpy.ndarray, shape (n,)
        Mean absolute discrepancy per spot: mean_j | ||z_i - z_j|| - lam * ||s_i - s_j|| |.
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
    """
    Scatter spatial points colored by a per-spot error scalar and save to disk.

    Parameters
    ----------
    S : numpy.ndarray, shape (n, 2)
        Spatial coordinates.
    errors : numpy.ndarray, shape (n,)
        Per-spot scalar errors.
    title : str
        Plot title.
    savepath : str
        File path to save the figure (PNG).
    """
    plt.figure(figsize=(6,6))
    plt.scatter(S[:,0], S[:,1], c=errors, cmap="viridis", s=8)
    plt.colorbar(label="Reconstruction error")
    plt.title(title)
    plt.gca().invert_yaxis()  # match Visium convention
    plt.axis("equal")
    plt.savefig(savepath, dpi=150)
    plt.close()

def compute_stress(model, Xset, Sset):
    """
    Compute the stress metric between latent and spatial distances.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model exposing `.encode(X)` and scalar attribute `.lam`.
    Xset : torch.Tensor, shape (n, g)
        Expression matrix on the same device as `model`.
    Sset : torch.Tensor, shape (n, 2)
        Spatial coordinates on the same device.
        
    Returns
    -------
    stress : float
        The stress metric (lower is better).
    """
    with torch.no_grad():
        mu_enc, _ = model.encode(Xset)
        Dz = torch.cdist(mu_enc, mu_enc, p=2)
        Ds = torch.cdist(Sset, Sset, p=2)
        lam_val = float(model.lam)
        if lam_val > 0:
            Dtarget = Dz / lam_val
        else:
            Dtarget = Dz
        triu = torch.triu(torch.ones(Ds.size(0), Ds.size(0), device=Ds.device, dtype=torch.bool), diagonal=1)
        num = ((Dtarget - Ds)**2)[triu].sum()
        denom = (Ds**2)[triu].sum() + 1e-12
        stress = torch.sqrt(num / denom).item()
    return stress

def analyze_triplet_geometry(XS_pairs, splits, best_models_stress, best_models_procrustes, best_models_mixed, output_dir="."):
    """
    Analyze triplet geometry for given models.
    
    Parameters
    ----------
    XS_pairs : dict
        Dictionary of (X, S) pairs
    splits : dict
        Dictionary of train/test splits
    best_models_stress : dict
        Best models based on stress
    best_models_procrustes : dict
        Best models based on Procrustes
    best_models_mixed : dict
        Best models based on combined metrics
    output_dir : str
        Directory to save output files
        
    Returns
    -------
    dict
        Dictionary of triplet diagnostics results
    """

    os.makedirs(output_dir, exist_ok=True)
    triplet_results = {}
    
    for key, (X, S) in XS_pairs.items():
        Xtr, Str, Xte, Ste = splits[key]
        triplet_results[key] = {}
        
        print(f"\n=== Triplet diagnostics for {key} ===")
        
        # Stress-selected model
        model_s = best_models_stress[key]
        idx_s, spatial_s, latent_s = test_triplet(model_s, Xte, Ste, seed=42, 
                                                    savepath=os.path.join(output_dir, f"triplet_stresssel_{key}.png"))
        metrics_s = compare_triangles(spatial_s, latent_s)
        print(f"[Stress-selected] Indices: {idx_s} | Mean angle err={metrics_s['mean_angle_error_deg']:.2f} deg")
        print("  Spatial angles (deg):", metrics_s['angles_spatial_deg'])
        print("  Latent angles  (deg):", metrics_s['angles_latent_deg'])
        print("  Angle errors  (deg):", metrics_s['angle_error_deg'])
        print("  Normalized side ratios:", metrics_s['normalized_side_ratios'])
        triplet_results[key]['stress'] = metrics_s
        
        # Procrustes-selected model
        model_p = best_models_procrustes[key]
        idx_p, spatial_p, latent_p = test_triplet(model_p, Xte, Ste, seed=43, 
                                                    savepath=os.path.join(output_dir, f"triplet_procrustesel_{key}.png"))
        metrics_p = compare_triangles(spatial_p, latent_p)
        print(f"[Procrustes-selected] Indices: {idx_p} | Mean angle err={metrics_p['mean_angle_error_deg']:.2f} deg")
        print("  Spatial angles (deg):", metrics_p['angles_spatial_deg'])
        print("  Latent angles  (deg):", metrics_p['angles_latent_deg'])
        print("  Angle errors  (deg):", metrics_p['angle_error_deg'])
        print("  Normalized side ratios:", metrics_p['normalized_side_ratios'])
        triplet_results[key]['procrustes'] = metrics_p
        
        # Mixed-selected model
        model_m = best_models_mixed[key]
        idx_m, spatial_m, latent_m = test_triplet(model_m, Xte, Ste, seed=44, 
                                                    savepath=os.path.join(output_dir, f"triplet_mixedsel_{key}.png"))
        metrics_m = compare_triangles(spatial_m, latent_m)
        print(f"[Mixed-selected] Indices: {idx_m} | Mean angle err={metrics_m['mean_angle_error_deg']:.2f} deg")
        print("  Spatial angles (deg):", metrics_m['angles_spatial_deg'])
        print("  Latent angles  (deg):", metrics_m['angles_latent_deg'])
        print("  Angle errors  (deg):", metrics_m['angle_error_deg'])
        print("  Normalized side ratios:", metrics_m['normalized_side_ratios'])
        triplet_results[key]['mixed'] = metrics_m
    
    return triplet_results


def preprocess_data(dataset_keys=None, device=None):
    """
    Preprocess the data for selected datasets.
    
    Parameters
    ----------
    dataset_keys : list or None
        List of dataset keys to process, or None for all datasets
    device : torch.device
        Device to use for tensor operations
        
    Returns
    -------
    dict, dict
        XS_pairs: Dictionary mapping dataset keys to (X, S) tensor pairs
        splits: Dictionary mapping dataset keys to (X_train, S_train, X_test, S_test)
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    

    # Filter to selected datasets if specified
    if dataset_keys is not None:
        datasets = {k: v for k, v in datasets.items() if k in dataset_keys}

    # Quality control
    datasets, XS_pairs, splits = QC(datasets, device)

    # Correlation statistics
    correlation_stats(XS_pairs)
    
    return XS_pairs, splits

def generate_heatmaps(XS_pairs, splits, best_models_stress, output_dir="."):
    """
    Generate heatmaps for spatial reconstruction errors.
    
    Parameters
    ----------
    XS_pairs : dict
        Dictionary of (X, S) pairs
    splits : dict
        Dictionary of train/test splits
    best_models_stress : dict
        Best models based on stress
    output_dir : str
        Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for key, (X, S) in XS_pairs.items():
        Xtr, Str, Xte, Ste = splits[key]
        errors = spatial_reconstruction_error(best_models_stress[key], Xte, Ste, mask=None)
        plot_spatial_error(
            Ste.cpu().numpy(),
            errors,
            title=f"Test reconstruction error (stress-selected): {key}",
            savepath=os.path.join(output_dir, f"heatmap_error_stresssel_{key}.png")
        )
