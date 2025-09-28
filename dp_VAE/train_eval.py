import torch
from dp_VAE import dp_VAE as dp
from dp_VAE import utils as FN

def parameter_sweep(XS_pairs, splits, alpha2_values, mask_k_values, lam_factors=None, 
                    max_epochs=2000, patience=200, device=None):
    """
    Perform parameter sweep for training dp-VAE models.
    
    Parameters
    ----------
    XS_pairs : dict
        Dictionary of (X, S) pairs
    splits : dict
        Dictionary of train/test splits
    alpha2_values : list
        List of alpha2 values to sweep
    mask_k_values : list
        List of mask_k values to sweep
    lam_factors : list or None
        List of lambda scaling factors, or None for defaults
    max_epochs : int
        Maximum number of training epochs
    patience : int
        Patience for early stopping
    device : torch.device
        Device to use for tensor operations
        
    Returns
    -------
    dict, dict, dict, dict
        results_all: All stress results
        best_params_all: Best parameters based on stress
        procrustes_all: All Procrustes distance results
        best_params_procrustes_all: Best parameters based on Procrustes
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if lam_factors is None:
        lam_factors = [0.1, 0.3, 0.5, 1, 2, 3, 10]
    
    results_all = {}
    best_states_all = {}
    best_params_all = {}
    procrustes_all = {}
    best_params_procrustes_all = {}
    best_states_procrustes_all = {}
    best_params_mixed_all = {}

    for key, (X, S) in XS_pairs.items():
        print(f"\n=== Sweeping (train stress) for dataset: {key} ===")
        results = {}               # stress
        best_states = {}
        procrustes_results = {}    # procrustes
        procrustes_states = {}
        Xtr, Str, Xte, Ste = splits[key]
        lam0 = 1.0
        lam_values = [lam0 * f for f in lam_factors]
        print(f"Init λ(train) for {key}: {lam0:.4f}, sweeping {lam_values}")
        
        for a2 in alpha2_values:
            for lm in lam_values:
                for mk in mask_k_values:
                    print(f"Training with α₂={a2}, λ={lm}, mask_k={mk}")
                    
                    # Train the model with current parameters
                    stress_val, state = dp.train_and_eval(
                        a2, 2.0, lm, 
                        X=Xtr, S=Str, 
                        mask=None, mask_k=mk,
                        max_epochs=max_epochs,
                        patience=patience
                    )
                    results[(a2, lm, mk)] = stress_val
                    best_states[(a2, lm, mk)] = state
                    
                    # Compute Procrustes on held-out set
                    model_tmp = dp.dpVAE(
                        input_dim=Xtr.shape[1], z_dim=2,
                        alpha1=1.0, beta=2.0, alpha2=a2, 
                        lam_init=lm, learn_lam=True,
                        mask_k=mk
                    ).to(device)
                    
                    if state is not None:
                        model_tmp.load_state_dict(state)
                    model_tmp.eval()
                    
                    with torch.no_grad():
                        mu_te, _ = model_tmp.encode(Xte)
                        Z_np = mu_te.cpu().numpy()
                        S_np = Ste.cpu().numpy()
                        _, _, disp = FN.procrustes(S_np, Z_np)
                    
                    procrustes_results[(a2, lm, mk)] = disp
                    procrustes_states[(a2, lm, mk)] = state
                    
                    # Log result
                    print(f"  Result: stress={stress_val:.4f}, procrustes={disp:.4f}")
        
        # Find best parameters
        best_params = min(results, key=results.get)
        best_params_procrustes = min(procrustes_results, key=procrustes_results.get)
        best_params_mixed = min(
            {k: (results[k] + procrustes_results[k]) / 2 for k in results}, 
            key=lambda k: (results[k] + procrustes_results[k]) / 2
        )
        
        print(f"\nBest params (train stress) for {key}: {best_params} = {results[best_params]:.4f}")
        print(f"Best params (Procrustes) for {key}: {best_params_procrustes} = {procrustes_results[best_params_procrustes]:.4f}")
        print(f"Best params (Mixed) for {key}: {best_params_mixed} = {(results[best_params_mixed] + procrustes_results[best_params_mixed]) / 2:.4f}")
        
        # Store results
        results_all[key] = results
        best_states_all[key] = best_states
        best_params_all[key] = best_params
        procrustes_all[key] = procrustes_results
        best_states_procrustes_all[key] = procrustes_states
        best_params_procrustes_all[key] = best_params_procrustes
        best_params_mixed_all[key] = best_params_mixed

    return (results_all, best_states_all, best_params_all, 
            procrustes_all, best_states_procrustes_all, 
            best_params_procrustes_all, best_params_mixed_all)

def build_best_models(XS_pairs, splits, best_params_all, best_states_all,
                     best_params_procrustes_all, best_states_procrustes_all, 
                     best_params_mixed_all, device=None):
    """
    Build and evaluate the best models based on sweep results.
    
    Parameters
    ----------
    XS_pairs : dict
        Dictionary of (X, S) pairs
    splits : dict
        Dictionary of train/test splits
    best_params_all : dict
        Best parameters based on stress
    best_states_all : dict
        Best model states based on stress
    best_params_procrustes_all : dict
        Best parameters based on Procrustes distance
    best_states_procrustes_all : dict
        Best model states based on Procrustes distance
    best_params_mixed_all : dict
        Best parameters based on combined metrics
    device : torch.device
        Device to use for tensor operations
        
    Returns
    -------
    dict, dict, dict, dict
        best_models_stress: Best models based on stress
        best_models_procrustes: Best models based on Procrustes
        best_models_mixed: Best models based on combined metrics
        selection_summary: Summary of test performance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_models_stress = {}
    best_models_procrustes = {}
    best_models_mixed = {}
    selection_summary = {}
    
    for key, (X, S) in XS_pairs.items():
        Xtr, Str, Xte, Ste = splits[key]
        
        # Stress-selected
        a2s, lms, mks = best_params_all[key]
        model_s = dp.dpVAE(input_dim=Xtr.shape[1], z_dim=2, alpha1=1.0, beta=2.0, alpha2=a2s, lam_init=lms, learn_lam=True, mask_k=mks).to(device)
        model_s.load_state_dict(best_states_all[key][(a2s, lms, mks)])
        model_s.eval()
        best_models_stress[key] = model_s
        
        # Procrustes-selected
        a2p, lmp, mkp = best_params_procrustes_all[key]
        model_p = dp.dpVAE(input_dim=Xtr.shape[1], z_dim=2, alpha1=1.0, beta=2.0, alpha2=a2p, lam_init=lmp, learn_lam=True, mask_k=mkp).to(device)
        model_p.load_state_dict(best_states_procrustes_all[key][(a2p, lmp, mkp)])
        model_p.eval()
        best_models_procrustes[key] = model_p
        
        # Mixed-selected
        a2m, lmm, mkm = best_params_mixed_all[key]
        model_m = dp.dpVAE(input_dim=Xtr.shape[1], z_dim=2, alpha1=1.0, beta=2.0, alpha2=a2m, lam_init=lmm, learn_lam=True, mask_k=mkm).to(device)
        model_m.load_state_dict(best_states_all[key][(a2m, lmm, mkm)])
        model_m.eval()
        best_models_mixed[key] = model_m
        
        # Evaluate both metrics on test set for all selections
        with torch.no_grad():
            mu_ts_s, _ = model_s.encode(Xte)
            _, _, proc_s = FN.procrustes(Ste.cpu().numpy(), mu_ts_s.cpu().numpy())
            stress_s = FN.compute_stress(model_s, Xte, Ste)
            
            mu_ts_p, _ = model_p.encode(Xte)
            _, _, proc_p = FN.procrustes(Ste.cpu().numpy(), mu_ts_p.cpu().numpy())
            stress_p = FN.compute_stress(model_p, Xte, Ste)
            
            mu_ts_m, _ = model_m.encode(Xte)
            _, _, proc_m = FN.procrustes(Ste.cpu().numpy(), mu_ts_m.cpu().numpy())
            stress_m = FN.compute_stress(model_m, Xte, Ste)
        
        selection_summary[key] = {
            'stress_selected_params': (a2s, lms, mks),
            'procrustes_selected_params': (a2p, lmp, mkp),
            'stress_selected_test_stress': stress_s,
            'stress_selected_test_procrustes': proc_s,
            'procrustes_selected_test_stress': stress_p,
            'procrustes_selected_test_procrustes': proc_p,
            'mixed_selected_params': (a2m, lmm, mkm),
            'mixed_selected_test_stress': stress_m,
            'mixed_selected_test_procrustes': proc_m
        }

    print("\n=== SELECTION COMPARISON SUMMARY ===")
    for key, summary in selection_summary.items():
        print(f"Dataset: {key}")
        print(f"  Stress-selected params: {summary['stress_selected_params']} -> Test Stress={summary['stress_selected_test_stress']:.4f}, Test Procrustes={summary['stress_selected_test_procrustes']:.4f}")
        print(f"  Procrustes-selected params: {summary['procrustes_selected_params']} -> Test Stress={summary['procrustes_selected_test_stress']:.4f}, Test Procrustes={summary['procrustes_selected_test_procrustes']:.4f}")
        print(f"  Mixed-selected params: {summary['mixed_selected_params']} -> Test Stress={summary['mixed_selected_test_stress']:.4f}, Test Procrustes={summary['mixed_selected_test_procrustes']:.4f}")
        print("  --")

    return best_models_stress, best_models_procrustes, best_models_mixed, selection_summary
