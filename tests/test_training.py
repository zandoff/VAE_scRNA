import torch
from dp_VAE import dp_VAE as dp

def test_train_and_eval_smoke_cpu():
    """
    Smoke test for the train_and_eval function on CPU.
    
    Tests that the training and evaluation pipeline:
    - Runs without errors for a small synthetic dataset
    - Returns expected output types (float for best_val, None or dict for state)
    - Handles validation split correctly
    - Processes mask_k parameters correctly, including None values
    
    Returns
    -------
    None
    """
    torch.manual_seed(0)
    X = torch.randn(30, 12)
    S = torch.randn(30, 2)
    # Make S have some structure
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    val_split = 10
    X_tr, S_tr = X[:-val_split], S[:-val_split]
    X_val, S_val = X[-val_split:], S[-val_split:]

    best_val, state = dp.train_and_eval(
        alpha2=5.0,
        beta=2.0,
        lam=1.0,
        X=X_tr,
        S=S_tr,
        X_val=X_val,
        S_val=S_val,
        max_epochs=50,
        patience=5,
        select_metric="stress",
        mask_k=[None, 4]  # ensure iterable w/ None handled
    )
    assert isinstance(best_val, float)
    assert state is None or isinstance(state, dict)

    # Rebuild and encode a batch
    model = dp.dpVAE(input_dim=X.shape[1], z_dim=2, alpha1=1.0, beta=2.0, alpha2=5.0, lam_init=1.0, learn_lam=True, mask_k=4)
    if state is not None:
        model.load_state_dict(state)
    with torch.no_grad():
        mu, _ = model.encode(X)
    assert mu.shape == (30, 2)
