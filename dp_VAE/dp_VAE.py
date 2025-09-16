import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CUstomizable hyperparameters for training
pre_lambda_epochs = 50  # Number of epochs to train with λ=0 before enabling distance-preserving loss
LAM_MIN = 1e-3  # Minimum value for λ
LAM_MAX = 1e3   # Maximum value for λ
LAM_REG_WEIGHT = 0  # Weight for λ regularization term

class dpVAE(nn.Module):
    """
    Distance-Preserving Variational Autoencoder (dp-VAE).

    This VAE augments the standard ELBO with a distance-preserving (DP)
    regularizer to align pairwise distances in the latent space with those in
    the spatial domain. It supports optional k-NN masking (single or multi-
    scale union) for the DP term, and a learnable scale parameter λ using a
    stable log-parameterization.

    Parameters
    ----------
    input_dim : int
        Number of input genes (features).
    z_dim : int, default=2
        Latent dimensionality.
    hidden_dim : int, default=124
        Hidden layer size for encoder/decoder MLPs.
    alpha1 : float, default=1.0
        Weight on the ELBO term.
    beta : float, default=2.0
        Temperature on the KL term (β-VAE style).
    alpha2 : float, default=0.1
        Weight on the distance-preserving loss.
    lam_init : float | None, default=None
        Initial value for λ (distance scale). If None, uses 1.0.
    mask_k : int | Iterable[int | None] | None, default=None
        Mask specification for the DP term:
        - None → no mask (all pairs)
        - int → single k-NN (undirected)
        - iterable → union of specified k-NN graphs; if any element is None,
          masking is disabled (full pairwise).
    learn_lam : bool, default=True
        If True, λ is learned via log-parameterization; otherwise fixed.

    Notes
    -----
    - λ is represented internally as `lam_raw` with λ = clamp(exp(lam_raw),
      [LAM_MIN, LAM_MAX]).
    - Pairwise distances use Euclidean metric.
    """
    def __init__(self, input_dim, z_dim=2, hidden_dim=124,
                 alpha1=1.0, beta=2.0, alpha2=0.1, lam_init=None, mask_k=None, learn_lam=True):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.alpha1 = alpha1
        self.beta = beta
        self.alpha2 = alpha2

        # mask_k specification:
        #   * None → full pairwise
        #   * int  → single k-NN neighborhood
        #   * Iterable[int] → union of multi-scale k-NN neighborhoods
        self.mask_k = mask_k
        self.learn_lam = learn_lam

        # Encoder layers
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_logvar = nn.Linear(hidden_dim, z_dim)

        # Decoder layers
        self.dec_fc1 = nn.Linear(z_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

        # λ reparameterization (log-space). Store lam_raw; λ = clamp(exp(lam_raw)).
        if lam_init is None:
            lam_init = 1.0
        if learn_lam:
            self.lam_raw = nn.Parameter(torch.log(torch.tensor(float(lam_init), dtype=torch.float32)))
        else:
            self.register_buffer("lam_buffer", torch.tensor(float(lam_init), dtype=torch.float32))

        # Cache dict: keys ('single', n, (k,)) or ('multi', n, (k1,k2,...))
        self._mask_cache = {}
    
    def encode(self, x):
        """
        Encode inputs into mean and log-variance of the latent Gaussian.

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)
            Input expression tensor on the same device as the model.

        Returns
        -------
        mu : torch.Tensor, shape (n, z_dim)
            Mean of q(z|x).
        logvar : torch.Tensor, shape (n, z_dim)
            Log-variance of q(z|x).
        """
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z):
        """
        Decode latent variables back to the input space.

        Parameters
        ----------
        z : torch.Tensor, shape (n, z_dim)
            Latent samples or means.

        Returns
        -------
        x_recon : torch.Tensor, shape (n, input_dim)
            Reconstruction logits/values (Gaussian likelihood assumed).
        """
        h = F.relu(self.dec_fc1(z))
        return self.dec_out(h)  # Gaussian likelihood → linear output

    def reparameterize(self, mu, logvar):
        """
        Sample z via the reparameterization trick.

        Parameters
        ----------
        mu : torch.Tensor, shape (n, z_dim)
            Mean of q(z|x).
        logvar : torch.Tensor, shape (n, z_dim)
            Log-variance of q(z|x).

        Returns
        -------
        z : torch.Tensor, shape (n, z_dim)
            Latent samples.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- Loss components ----------
    def kl_loss(self, mu, logvar):
        """
        Compute the KL divergence term E[ D_KL(q(z|x) || p(z)) ].

        Parameters
        ----------
        mu : torch.Tensor, shape (n, z_dim)
        logvar : torch.Tensor, shape (n, z_dim)

        Returns
        -------
        kl : torch.Tensor
            Mean KL divergence across the batch.
        """
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_per_sample.mean()

    def recon_loss(self, x, x_recon):
        """
        Compute per-sample reconstruction loss (MSE sum over features).

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)
        x_recon : torch.Tensor, shape (n, input_dim)

        Returns
        -------
        rec_vec : torch.Tensor, shape (n,)
            Squared-error per sample summed over feature dimension.
        """
        return ((x_recon - x)**2).sum(dim=1)

    def distance_preserving_loss(self, z, s, mask=None):
        """
        Distance-preserving loss as mean absolute deviation over selected pairs.

        Parameters
        ----------
        z : torch.Tensor, shape (n, z_dim)
            Latent coordinates.
        s : torch.Tensor, shape (n, 2)
            Spatial coordinates.
        mask : torch.Tensor | None, shape (n, n), optional
            Optional binary mask. If None, uses `mask_k` to construct a k-NN
            union mask; otherwise all upper-triangular pairs are used.

        Returns
        -------
        ldp : torch.Tensor
            Mean absolute discrepancy | ||z_i - z_j|| - λ ||s_i - s_j|| | over
            selected (i, j), i < j.
        """
        Dz = torch.cdist(z, z, p=2)
        Ds = torch.cdist(s, s, p=2)
        n = s.size(0)

        # Build / fetch mask of edges to include
        if mask is not None:
            pair_mask = mask.bool()
        elif self.mask_k is not None:
            pair_mask = self._get_cached_mask(s)
        else:
            pair_mask = None

        triu = torch.triu(torch.ones(n, n, device=z.device, dtype=torch.bool), diagonal=1)
        if pair_mask is not None:
            G = pair_mask & triu
        else:
            G = triu

        lam = self.current_lambda()
        diff_abs = (Dz - lam * Ds).abs()
        selected = diff_abs[G]
        if selected.numel() == 0:
            return torch.tensor(0.0, device=z.device)
        return selected.mean()

    # ---------- Forward ----------
    def forward(self, x):
        """
        Standard VAE forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)

        Returns
        -------
        x_recon : torch.Tensor, shape (n, input_dim)
        mu : torch.Tensor, shape (n, z_dim)
        logvar : torch.Tensor, shape (n, z_dim)
        z : torch.Tensor, shape (n, z_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    # ---------- Full dp-VAE loss ----------
    def loss_function(self, x, s, mask=None):
        """
        Compute the weighted ELBO + distance-preserving loss.

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)
        s : torch.Tensor, shape (n, 2)
        mask : torch.Tensor | None, shape (n, n), optional

        Returns
        -------
        total : torch.Tensor
            Weighted total objective.
        rec_vec : torch.Tensor, shape (n,)
            Per-sample reconstruction losses (sum over features).
        kl : torch.Tensor
            KL divergence term (scalar).
        ldp : torch.Tensor
            Distance-preserving loss (scalar).
        lam : torch.Tensor
            Current clamped λ.
        """
        x_recon, mu, logvar, z = self.forward(x)
        rec_vec = self.recon_loss(x, x_recon)          # per-sample
        rec_mean = rec_vec.mean()                      # scalar for objective
        kl = self.kl_loss(mu, logvar)                  # scalar
        ldp = self.distance_preserving_loss(z, s, mask)  # scalar
        lam = self.current_lambda()                    # Current λ
        total = self.alpha1 * (rec_mean + self.beta * kl) + self.alpha2 * ldp
        return total, rec_vec, kl, ldp, lam

    # --------- λ helpers & mask utilities ---------
    def current_lambda(self):
        """
        Return current clamped λ value.

        Returns
        -------
        lam : torch.Tensor
            Scalar tensor equal to clamp(exp(lam_raw), [LAM_MIN, LAM_MAX])
            if learnable; otherwise the fixed buffer, clamped similarly.
        """
        if self.learn_lam:
            lam = torch.exp(self.lam_raw)
        else:
            lam = self.lam_buffer
        return torch.clamp(lam, LAM_MIN, LAM_MAX)

    # Backward compatibility: allow model.lam to be accessed (was a Parameter before)
    @property
    def lam(self):
        return self.current_lambda()

    def _get_cached_mask(self, s: torch.Tensor):
        """
        Build or fetch a symmetric boolean mask (n, n) per `mask_k` spec.

        Rules (paper-consistent Eq. (5)):
        - single int k: standard k-NN graph (undirected) ⇒ binary mask
        - iterable of ints: union of each individual k-NN graph ⇒ binary mask
        - iterable containing None: interpret as full pairwise (disable masking)

        Parameters
        ----------
        s : torch.Tensor, shape (n, 2)
            Spatial coordinates.

        Returns
        -------
        mask : torch.Tensor | None, shape (n, n)
            Symmetric boolean mask on the same device as `s`, or None to
            indicate full pairwise usage.

        Notes
        -----
        Masks are cached by (mode, n, ks_sorted) for efficiency.
        """
        n = s.size(0)
        if self.mask_k is None:
            return None

        # Normalize specification
        if isinstance(self.mask_k, int):
            ks = [self.mask_k]
            multi = False
        elif isinstance(self.mask_k, Iterable):
            # Allow None inside iterable; if present, treat as no masking
            if any(k is None for k in self.mask_k):
                return None
            ks = [int(k) for k in self.mask_k]
            ks = [k for k in ks if k > 0]
            if len(ks) == 0:
                return None
            multi = len(ks) > 1
        else:
            raise ValueError("mask_k must be int, iterable of ints, or None")

        ks_sorted = tuple(sorted(min(k, n - 1) for k in ks))
        cache_key = ('multi' if multi else 'single', n, ks_sorted)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        with torch.no_grad():
            # Precompute squared distances once
            sq = (s * s).sum(dim=1, keepdim=True)
            Ds2 = sq + sq.T - 2 * (s @ s.T)
            Ds2.clamp_(min=0)
            Ds2.fill_diagonal_(float('inf'))
            union_mask = torch.zeros(n, n, dtype=torch.bool, device=s.device)
            row = torch.arange(n, device=s.device)
            for k in ks_sorted:
                if k <= 0:
                    continue
                k_eff = min(k, n - 1)
                _, knn_idx = torch.topk(Ds2, k_eff, dim=1, largest=False)
                r_exp = row.view(-1, 1).expand_as(knn_idx)
                tmp_mask = torch.zeros(n, n, dtype=torch.bool, device=s.device)
                tmp_mask[r_exp, knn_idx] = True
                union_mask |= tmp_mask
            union_mask = union_mask | union_mask.T  # symmetrize
        self._mask_cache[cache_key] = union_mask
        return union_mask
        

def train_and_eval(alpha2, beta, lam, X, S, mask=None, X_val=None, S_val=None,
                   max_epochs=2000, patience=200, stress_tol=1e-5, obj_tol=1e-5,
                   select_metric="stress", objective_resets_patience=False,
                   include_lam_reg_in_obj=False, return_all=False, mask_k=None):
    """
    Train a dpVAE with dual monitoring (stress + objective) and early stopping.

    Parameters
    ----------
    alpha2 : float
        Weight for the distance-preserving term.
    beta : float
        KL temperature (β) in the ELBO.
    lam : float
        Initial λ value.
    X : torch.Tensor, shape (n, g)
        Training expression matrix on current device.
    S : torch.Tensor, shape (n, 2)
        Training spatial coordinates on current device.
    mask : torch.Tensor | None, shape (n, n), optional
        Optional pairwise mask for the DP term.
    X_val : torch.Tensor | None, shape (n_val, g), optional
        Optional validation expressions. If provided, stress is computed on
        validation; otherwise on train.
    S_val : torch.Tensor | None, shape (n_val, 2), optional
        Optional validation spatial coordinates.
    max_epochs : int, default=2000
        Maximum number of epochs.
    patience : int, default=200
        Epochs without primary metric improvement before stopping (post warm-up).
    stress_tol : float, default=1e-5
        Minimum improvement to accept a new best stress.
    obj_tol : float, default=1e-5
        Minimum improvement to accept a new best objective.
    select_metric : {"stress", "objective"}, default="stress"
        Which metric determines the final restored state.
    objective_resets_patience : bool, default=False
        If True, objective improvements also reset patience.
    include_lam_reg_in_obj : bool, default=False
        If True, include the λ-regularization in the objective tracked for selection.
    return_all : bool, default=False
        If True, return a dict of both best states and metadata; otherwise
        return a tuple (best_value, best_state).
    mask_k : int | Iterable[int | None] | None, optional
        k-NN masking configuration forwarded to the model.

    Returns
    -------
    best_value, best_state : tuple[float, dict]
        If `return_all=False`, the best metric value and the corresponding
        state_dict (as a plain dict) according to `select_metric`.
    info : dict
        If `return_all=True`, a dict with keys 'best_stress', 'best_dp_obj',
        'best_stress_state', 'best_obj_state', 'final_epoch',
        'selected_metric', and 'selected_value'.

    Notes
    -----
    Early stopping:
    - Warm-up (epochs ≤ `pre_lambda_epochs`): DP term disabled; no tracking.
    - Post warm-up: track best stress and best objective independently;
      patience resets on stress improvement, and optionally on objective
      improvement depending on `objective_resets_patience`.
    """
    model = dpVAE(
        input_dim=X.shape[1], z_dim=2,
        alpha1=1.0, beta=beta, alpha2=alpha2,
        lam_init=lam, learn_lam=True, mask_k=mask_k
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Precompute distance matrices for train / val
    Ds_full = torch.cdist(S, S, p=2)
    Ds_val_full = torch.cdist(S_val, S_val, p=2) if (S_val is not None) else None

    best_stress = float('inf')
    best_dp_obj = float('inf')
    best_stress_state = None
    best_obj_state = None
    patience_counter = 0  # counts epochs since last primary (stress) improvement (or objective if configured)
    lam_init_value = None

    lambda_initialized = False
    for epoch in range(1, max_epochs + 1):
        
        optimizer.zero_grad()

        if pre_lambda_epochs > 0 and epoch <= pre_lambda_epochs:
            x_recon, mu, logvar, _ = model.forward(X)
            rec_vec = model.recon_loss(X, x_recon)
            rec = rec_vec.mean()
            kl = model.kl_loss(mu, logvar)
            ldp = torch.tensor(0.0, device=device)
            lam_val = model.current_lambda()  # not used during warm-up
            lam_reg = torch.tensor(0.0, device=device)
            loss = model.alpha1 * (rec + model.beta * kl)  # scalar
        else:
            if pre_lambda_epochs > 0 and not lambda_initialized:
                with torch.no_grad():
                    mu_init, _ = model.encode(X)
                    Dz0 = torch.cdist(mu_init, mu_init, p=2)
                    Ds0 = torch.cdist(S, S, p=2)
                    ratio = Dz0[Ds0 > 0] / Ds0[Ds0 > 0]
                    lam_est = torch.median(ratio).item()
                    if (lam_est < LAM_MIN) or (not np.isfinite(lam_est)):
                        lam_est = LAM_MIN
                    lam_est = min(lam_est, LAM_MAX)
                    # set lam_raw so that exp(lam_raw)=lam_est
                    if model.learn_lam:
                        model.lam_raw.data = torch.log(torch.tensor(lam_est, device=model.lam_raw.device))
                    else:
                        model.lam_buffer.data.fill_(lam_est)
                    lam_init_value = lam_est
                lambda_initialized = True
                print(f"[λ initialized (learnable) at epoch {epoch}] λ = {model.current_lambda().item():.6f}")
            loss, rec_vec, kl, ldp, lam_val = model.loss_function(X, S, mask=mask)
            rec = rec_vec.mean() if rec_vec.dim() > 0 else rec_vec
            if lam_init_value is not None:
                lam_cur = model.current_lambda()
                lam_reg = (lam_cur - lam_init_value) ** 2
                loss = loss + LAM_REG_WEIGHT * lam_reg
            else:
                lam_reg = torch.tensor(0.0, device=device)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Train stress
            mu_eval, _ = model.encode(X)
            Dz_train = torch.cdist(mu_eval, mu_eval, p=2)
            lam_cur_val = float(model.current_lambda())
            Dtarget_train = Dz_train / lam_cur_val if lam_cur_val > 0 else Dz_train
            n_pts_tr = Dz_train.size(0)
            triu_mask_tr = torch.triu(torch.ones(n_pts_tr, n_pts_tr, device=Dz_train.device, dtype=torch.bool), diagonal=1)
            diff_sq_tr = (Dtarget_train - Ds_full).pow(2)[triu_mask_tr]
            denom_tr = (Ds_full.pow(2))[triu_mask_tr].sum() + 1e-12
            stress_train = torch.sqrt(diff_sq_tr.sum() / denom_tr)

            # Validation stress (if provided)
            if X_val is not None and S_val is not None:
                mu_val, _ = model.encode(X_val)
                Dz_val = torch.cdist(mu_val, mu_val, p=2)
                Dtarget_val = Dz_val / lam_cur_val if lam_cur_val > 0 else Dz_val
                n_pts_val = Dz_val.size(0)
                triu_mask_val = torch.triu(torch.ones(n_pts_val, n_pts_val, device=Dz_val.device, dtype=torch.bool), diagonal=1)
                diff_sq_val = (Dtarget_val - Ds_val_full).pow(2)[triu_mask_val]
                denom_val = (Ds_val_full.pow(2))[triu_mask_val].sum() + 1e-12
                stress_val = torch.sqrt(diff_sq_val.sum() / denom_val)
                stress = stress_val
            else:
                stress = stress_train

            # Positive dp objective (detach scalar)
            # rec may be per-sample; reduce mean for objective scale
            rec_mean = rec.mean() if hasattr(rec, 'dim') and rec.dim() > 0 else rec
            base_obj = model.alpha1 * (rec_mean + model.beta * kl) + model.alpha2 * ldp
            dp_obj = base_obj + (LAM_REG_WEIGHT * lam_reg if include_lam_reg_in_obj else 0.0)

        # Skip tracking improvements during warm-up (λ not yet active)
        in_warmup = (pre_lambda_epochs > 0 and epoch <= pre_lambda_epochs)
        if not in_warmup:
            improved_stress = stress.item() < (best_stress - stress_tol)
            improved_obj = dp_obj.item() < (best_dp_obj - obj_tol)

            if improved_stress:
                best_stress = stress.item()
                best_stress_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            elif improved_obj:
                best_dp_obj = dp_obj.item()
                best_obj_state = {k: v.clone() for k, v in model.state_dict().items()}
                if objective_resets_patience:
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                patience_counter += 1
        # If in warmup, do not advance patience (gives full warm-up always)

        if epoch % 100 == 0 or epoch in (1, pre_lambda_epochs, pre_lambda_epochs + 1):
            with torch.no_grad():
                mu_enc, logvar_enc = model.encode(X)
                mu_mean_abs = mu_enc.abs().mean().item()
                logvar_mean = logvar_enc.mean().item()
            core_elbo = rec_mean + model.beta * kl
            dp_term = ldp
            weighted_elbo = model.alpha1 * core_elbo
            weighted_dp = model.alpha2 * dp_term
            total_obj = weighted_elbo + weighted_dp + (LAM_REG_WEIGHT * lam_reg)
            stress_disp = stress.item() if isinstance(stress, torch.Tensor) else float(stress)
            print(
                f"[{epoch}/{max_epochs}] Total={total_obj.item():.4f} | ELBO_core={core_elbo.item():.4f} (α1*ELBO={weighted_elbo.item():.4f}) "
                f"Recon_mean={rec_mean.item():.4f} KL={kl.item():.4f} β={model.beta:.3f} | DP={dp_term.item():.4f} "
                f"(α2*DP={weighted_dp.item():.4f}) λ={model.current_lambda().item():.6f} Stress={stress_disp:.4f} (BestStress={best_stress:.4f}) "
                f"BestObj={best_dp_obj:.4f}{'(incl λ_reg)' if include_lam_reg_in_obj else ''} λ_reg={lam_reg.item():.4e} | μ̄|={mu_mean_abs:.3f} logvar̄={logvar_mean:.3f} Pat={patience_counter}"
            )

        # Early stopping condition only after warm-up
        if (pre_lambda_epochs == 0 or epoch > pre_lambda_epochs) and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Stress={best_stress:.4f} | BestObj={best_dp_obj:.4f}")
            break

    # Decide which state to restore
    if select_metric == "objective":
        target_state = best_obj_state if best_obj_state is not None else best_stress_state
        best_value = best_dp_obj
    else:
        target_state = best_stress_state if best_stress_state is not None else best_obj_state
        best_value = best_stress

    if target_state is not None:
        model.load_state_dict(target_state)

    print(f"α₂={alpha2}, λ_init={lam}, Final Best {select_metric.capitalize()}={best_value:.4f}")

    if return_all:
        return {
            "best_stress": best_stress,
            "best_dp_obj": best_dp_obj,
            "best_stress_state": best_stress_state,
            "best_obj_state": best_obj_state,
            "final_epoch": epoch,
            "selected_metric": select_metric,
            "selected_value": best_value,
        }
    else:
        return best_value, target_state
