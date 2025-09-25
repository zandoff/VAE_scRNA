import numpy as np
import torch
from dp_VAE import utils as FN


def test_pairwise_corr_keys_and_shapes():
    """
    Test the output structure of pairwise correlation function.
    
    Verifies that pairwise_corr_expr_spatial returns a dictionary with
    all expected keys and that the output vectors have the correct shapes
    corresponding to the number of pairwise comparisons.
    
    Returns
    -------
    None
    """
    X = torch.randn(10, 5)
    S = torch.randn(10, 2)
    res = FN.pairwise_corr_expr_spatial(X, S)
    for k in ["r", "pval", "ci", "n_pairs", "vec_expr", "vec_spat"]:
        assert k in res
    n = X.shape[0]
    iu = n*(n-1)//2
    assert res["vec_expr"].shape[0] == iu
    assert res["vec_spat"].shape[0] == iu


def test_triangle_angles_and_compare_triangles_identity():
    """
    Test triangle geometry utilities with identity case.

    Verifies that:
    - triangle_angles correctly calculates angles for a right isosceles triangle
      (should be [π/2, π/4, π/4] in some order)
    - compare_triangles produces near-zero angle error when comparing identical triangles
    - normalized side ratios have the expected shape

    Returns
    -------
    None
    """
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ang = FN.triangle_angles(tri)
    # Check that angles are close to [π/2, π/4, π/4] in any order
    expected = np.array([np.pi/2, np.pi/4, np.pi/4])
    ang_sorted = np.sort(ang)
    expected_sorted = np.sort(expected)
    np.testing.assert_allclose(ang_sorted, expected_sorted, atol=1e-6)

    comp = FN.compare_triangles(tri, tri.copy())
    assert abs(comp["mean_angle_error_deg"]) < 1e-6
    assert comp["normalized_side_ratios"].shape == (3,)


def test_procrustes_distance_basic():
    """
    Test basic properties of Procrustes distance calculation.
    
    Creates two similar point sets with small random differences and verifies
    that the Procrustes distance is non-negative.
    
    Returns
    -------
    None
    """
    z = torch.randn(10, 2)
    s = z + 0.01*torch.randn(10, 2)
    d = FN.procrustes_distance(z, s)
    assert d >= 0


def test_spatial_reconstruction_error_shape():
    """
    Test the spatial reconstruction error function output.
    
    Verifies that:
    - The error values have the expected shape (one per sample)
    - All error values are non-negative
    - Function handles a basic model implementation correctly
    
    Returns
    -------
    None
    """
    class Dummy:
        def __init__(self):
            self._device = torch.device("cpu")
        def encode(self, X):
            return X[:, :2], None
        @property
        def lam(self):
            return torch.tensor(1.0)
    m = Dummy()
    X = torch.randn(9, 4)
    S = torch.randn(9, 2)
    errs = FN.spatial_reconstruction_error(m, X, S, mask=None)
    assert errs.shape == (9,)
    assert np.all(errs >= 0)
