import torch
from dp_VAE import dp_VAE as dp


def test_mask_k_none_returns_none():
    """
    Test mask generation when mask_k is None.
    
    Verifies that when mask_k is None, the _get_cached_mask function
    returns None, indicating that no k-NN mask should be applied.
    
    Returns
    -------
    None
    """
    X = torch.randn(5, 10)
    S = torch.randn(5, 2)
    model = dp.dpVAE(input_dim=10, z_dim=2, mask_k=None)
    mask = model._get_cached_mask(S)
    assert mask is None


def test_mask_k_int_returns_binary_mask():
    """
    Test mask generation with an integer k value.
    
    Verifies that when mask_k is an integer:
    - A non-None mask is returned
    - The mask is binary (boolean type)
    - The mask is symmetric (undirected graph)
    - The diagonal is zero (no self-loops)
    
    Returns
    -------
    None
    """
    S = torch.randn(6, 2)
    model = dp.dpVAE(input_dim=10, z_dim=2, mask_k=3)
    mask = model._get_cached_mask(S)
    assert mask is not None
    assert mask.dtype == torch.bool
    # symmetric and zero diagonal
    assert torch.all(mask == mask.T)
    assert torch.count_nonzero(torch.diag(mask)) == 0
