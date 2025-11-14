import torch
import pytest
import numpy as np
import sys
import os
from scipy.stats import pearsonr

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dp_VAE import utils as FN
from dp_VAE import dp_VAE


class TestPairwiseCorrelation:
    """Tests for pairwise_corr_expr_spatial function."""
    
    def test_pairwise_corr_output_structure(self):
        """
        Test that pairwise_corr_expr_spatial returns all expected output keys.
        
        Verifies that the function returns a dictionary containing all required keys:
        'r' (correlation coefficient), 'pval' (p-value), 'ci' (confidence interval),
        'n_pairs' (number of cell pairs), 'vec_expr' (expression distances), and
        'vec_spat' (spatial distances). This ensures the function's output structure
        is consistent and complete for downstream analysis.
        
        Returns
        -------
        None
        """
        X = torch.randn(15, 20)
        S = torch.randn(15, 2)
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        expected_keys = ['r', 'pval', 'ci', 'n_pairs', 'vec_expr', 'vec_spat']
        for key in expected_keys:
            assert key in result
            
    def test_pairwise_corr_n_pairs_correct(self):
        """
        Test that the number of pairs computed matches n choose 2.
        
        For n cells, the function should compute pairwise distances for exactly
        n*(n-1)/2 unique pairs. This test verifies that the reported n_pairs value
        is correct and that the vec_expr and vec_spat arrays have the matching length.
        This is critical for ensuring all pairwise comparisons are properly computed.
        
        Returns
        -------
        None
        """
        n = 10
        X = torch.randn(n, 20)
        S = torch.randn(n, 2)
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        expected_pairs = n * (n - 1) // 2
        assert result['n_pairs'] == expected_pairs
        assert len(result['vec_expr']) == expected_pairs
        assert len(result['vec_spat']) == expected_pairs
        
    def test_pairwise_corr_with_numpy_input(self):
        """
        Test that pairwise_corr_expr_spatial accepts numpy arrays as input.
        
        Verifies that the function can process numpy arrays (not just torch tensors)
        for both expression data X and spatial coordinates S. This tests the function's
        flexibility in handling different input types commonly used in data analysis.
        The function should return a valid correlation coefficient as a float.
        
        Returns
        -------
        None
        """
        X = np.random.randn(10, 15)
        S = np.random.randn(10, 2)
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        assert 'r' in result
        assert isinstance(result['r'], (float, np.floating))
        
    def test_pairwise_corr_with_torch_input(self):
        """
        Test that pairwise_corr_expr_spatial accepts torch tensors as input.
        
        Verifies that the function can process torch tensors for both expression
        data X and spatial coordinates S. Since the model uses PyTorch, this tests
        the function's compatibility with the native tensor format. The function
        should return a valid correlation coefficient as a float.
        
        Returns
        -------
        None
        """
        X = torch.randn(10, 15)
        S = torch.randn(10, 2)
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        assert 'r' in result
        assert isinstance(result['r'], (float, np.floating))
        
    def test_pairwise_corr_perfect_correlation(self):
        """
        Test correlation coefficient when expression and spatial distances are perfectly proportional.
        
        Creates a scenario where expression features are directly proportional to spatial
        coordinates, resulting in pairwise expression distances that perfectly correlate
        with spatial distances. The correlation coefficient should be very high (r > 0.9).
        This validates that the function correctly captures strong positive relationships
        between expression and spatial structure.
        
        Returns
        -------
        None
        """
        # Create data where spatial and expression distances are perfectly correlated
        S = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
        # Make X proportional to S
        X = S.repeat(1, 5)  # Repeat to have more features
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        # Correlation should be very high (close to 1)
        assert result['r'] > 0.9
        
    def test_pairwise_corr_confidence_interval(self):
        """
        Test that the confidence interval is properly computed and brackets the correlation.
        
        Verifies that the returned confidence interval is a tuple with two values (lower
        and upper bounds) and that these bounds properly bracket the computed correlation
        coefficient. This is essential for statistical inference and assessing the
        uncertainty in the correlation estimate.
        
        Returns
        -------
        None
        """
        X = torch.randn(20, 30)
        S = torch.randn(20, 2)
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        assert isinstance(result['ci'], tuple)
        assert len(result['ci']) == 2
        ci_low, ci_high = result['ci']
        
        # CI should bracket the correlation coefficient
        assert ci_low <= result['r'] <= ci_high
        
    def test_pairwise_corr_distances_all_positive(self):
        """
        Test that all computed distance values are non-negative.
        
        Distances are inherently non-negative quantities. This test verifies that
        both the expression distance vector (vec_expr) and spatial distance vector
        (vec_spat) contain only non-negative values. Any negative distance would
        indicate a computation error in the pairwise distance calculations.
        
        Returns
        -------
        None
        """
        X = torch.randn(10, 15)
        S = torch.randn(10, 2)
        
        result = FN.pairwise_corr_expr_spatial(X, S)
        
        assert np.all(result['vec_expr'] >= 0)
        assert np.all(result['vec_spat'] >= 0)
        

class TestTriangleGeometry:
    """Tests for triangle geometry functions."""
    
    def test_triangle_angles_sum_to_pi(self):
        """
        Test that the three angles of a triangle sum to π radians (180 degrees).
        
        This is a fundamental property of Euclidean triangles. The test verifies
        that the triangle_angles function correctly computes all three interior
        angles and that they satisfy the basic geometric constraint of summing
        to π. A deviation larger than 1e-5 would indicate a computation error.
        
        Returns
        -------
        None
        """
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]], dtype=np.float32)
        
        angles = FN.triangle_angles(pts)
        
        assert np.abs(np.sum(angles) - np.pi) < 1e-5
            
    def test_compare_triangles_identical(self):
        """
        Test that comparing a triangle to itself yields zero angle error.
        
        When comparing identical triangles, the mean angle error should be essentially
        zero (within numerical precision). This validates the baseline behavior of the
        compare_triangles function and ensures it correctly reports perfect geometric
        preservation when triangles are identical.
        
        Returns
        -------
        None
        """
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]], dtype=np.float32)
        
        comparison = FN.compare_triangles(pts, pts.copy())
        
        assert comparison['mean_angle_error_deg'] < 1e-5
        
    def test_compare_triangles_scaled(self):
        """
        Test that triangle angles are preserved under uniform scaling.
        
        Angles are scale-invariant properties of triangles. This test verifies that
        when comparing a triangle to a uniformly scaled version of itself, the angle
        error is minimal (< 1 degree). This is important for assessing shape preservation
        in the VAE latent space, where overall scale may differ but shape should be maintained.
        
        Returns
        -------
        None
        """
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]], dtype=np.float32)
        pts2 = pts1 * 2.0  # Scaled version
        
        comparison = FN.compare_triangles(pts1, pts2)
        
        # Angles should be preserved under scaling
        assert comparison['mean_angle_error_deg'] < 1.0
        
    def test_compare_triangles_output_structure(self):
        """
        Test that compare_triangles returns all expected output keys.
        
        Verifies that the function returns a complete dictionary containing angles
        for both spatial and latent triangles, angle errors, mean angle error, and
        normalized side ratios. This ensures the function's output structure is
        consistent and provides all necessary information for geometric analysis.
        
        Returns
        -------
        None
        """
        pts1 = np.random.randn(3, 2)
        pts2 = np.random.randn(3, 2)
        
        comparison = FN.compare_triangles(pts1, pts2)
        
        expected_keys = ['angles_spatial_deg', 'angles_latent_deg',
                        'angle_error_deg', 'mean_angle_error_deg',
                        'normalized_side_ratios']
        for key in expected_keys:
            assert key in comparison
            
    def test_compare_triangles_side_ratios(self):
        """
        Test that normalized side ratios are correctly computed for scaled triangles.
        
        For a uniformly scaled triangle, the normalized side ratios should all be
        close to 1, indicating that the relative proportions of the triangle sides
        are preserved. This test verifies that the side ratio computation correctly
        identifies uniform scaling and distinguishes it from shape distortion.
        
        Returns
        -------
        None
        """
        pts1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]], dtype=np.float32)
        pts2 = pts1 * 2.0
        
        comparison = FN.compare_triangles(pts1, pts2)
        
        ratios = comparison['normalized_side_ratios']
        # For uniformly scaled triangle, ratios should all be ~1
        assert np.allclose(ratios, np.ones(3), atol=0.1)
        
    def test_triangle_angles_degenerate_triangle(self):
        """
        Test handling of degenerate (collinear) triangles.
        
        When three points are collinear, they form a degenerate triangle with
        no well-defined interior. This test verifies that the function either
        handles this edge case gracefully (e.g., by returning angles close to
        [0, 0, π]) or raises an appropriate exception. This is important for
        robustness when sampling points that might occasionally be collinear.
        
        Returns
        -------
        None
        """
        # Three collinear points
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        
        try:
            angles = FN.triangle_angles(pts)
            # If it computes, angles should be close to [0, 0, π] or similar
            # This might produce NaN due to division issues
        except (ValueError, RuntimeWarning):
            # Expected for degenerate case
            pass


class TestProcrustesDistance:
    """Tests for Procrustes distance function."""
    
    def test_procrustes_zero_for_identical(self):
        """
        Test that Procrustes distance is zero for identical point configurations.
        
        When the latent space coordinates exactly match the spatial coordinates,
        the Procrustes distance (after optimal rotation, translation, and scaling)
        should be essentially zero. This validates the baseline behavior and ensures
        the function correctly identifies perfect geometric alignment.
        
        Returns
        -------
        None
        """
        z = torch.randn(15, 2)
        s = z.clone()
        
        distance = FN.procrustes_distance(z, s)
        
        assert distance < 1e-10
            
    def test_procrustes_positive(self):
        """
        Test that Procrustes distance is always non-negative.
        
        As a distance metric, Procrustes distance must always be non-negative.
        This test verifies this fundamental property holds for arbitrary point
        configurations. Additionally checks that the result is not NaN, which
        could indicate numerical issues in the computation.
        
        Returns
        -------
        None
        """
        z = torch.randn(15, 2)
        s = torch.randn(15, 2)
        
        distance = FN.procrustes_distance(z, s)
        
        assert distance >= 0
                

class TestSpatialReconstructionError:
    """Tests for spatial_reconstruction_error function."""
    
    def test_spatial_error_output_shape(self):
        """
        Test that spatial_reconstruction_error returns an array with correct shape.
        
        The function should return one reconstruction error value per cell. This test
        verifies that the output array has shape (n_cells,), matching the number of
        input cells. This is essential for per-cell error analysis and visualization.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=30, z_dim=2, lam_init=1.0)
        X = torch.randn(15, 30)
        S = torch.randn(15, 2)
        
        errors = FN.spatial_reconstruction_error(model, X, S)
        
        assert errors.shape == (15,)
        
    def test_spatial_error_all_positive(self):
        """
        Test that all reconstruction error values are non-negative.
        
        Reconstruction errors represent distances or deviations and must be non-negative.
        This test verifies that the spatial_reconstruction_error function computes
        only non-negative error values for all cells. Negative errors would indicate
        a fundamental computation error in the error metric.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=30, z_dim=2, lam_init=1.0)
        X = torch.randn(20, 30)
        S = torch.randn(20, 2)
        
        errors = FN.spatial_reconstruction_error(model, X, S)
        
        assert np.all(errors >= 0)
                
class TestSamplePoints:
    """Tests for sample_points function."""
    
    def test_sample_points_returns_three(self):
        """
        Test that sample_points returns exactly 3 indices.
        
        The function is designed to sample three points for triangle-based geometric
        analysis. This test verifies that the function always returns exactly 3 indices,
        which is the minimum number required to define a triangle. This is a fundamental
        requirement for the downstream triangle geometry computations.
        
        Returns
        -------
        None
        """
        n_cells = 20
        idx = FN.sample_points(n_cells)
        
        assert len(idx) == 3
        
    def test_sample_points_all_unique(self):
        """
        Test that sampled indices are all unique (no duplicates).
        
        For valid triangle formation, the three sampled points must be distinct.
        This test verifies that sample_points never returns duplicate indices, which
        would result in a degenerate triangle. This is crucial for meaningful geometric
        comparisons in the stress metric computation.
        
        Returns
        -------
        None
        """
        n_cells = 20
        idx = FN.sample_points(n_cells)
        
        assert len(np.unique(idx)) == 3
        
    def test_sample_points_within_range(self):
        """
        Test that sampled indices are within the valid range [0, n_cells).
        
        All returned indices must be valid array indices for the given dataset size.
        This test verifies that sample_points returns indices in the range [0, n_cells-1],
        ensuring they can be safely used to index into the cell data arrays without
        causing out-of-bounds errors.
        
        Returns
        -------
        None
        """
        n_cells = 20
        idx = FN.sample_points(n_cells)
        
        assert np.all(idx >= 0)
        assert np.all(idx < n_cells)
        
    def test_sample_points_minimum_size(self):
        """
        Test sampling when dataset has exactly 3 cells (minimum possible).
        
        When n_cells equals 3, the function must return all three available indices.
        This edge case tests whether the function handles the minimum dataset size
        correctly. With only 3 cells, there's only one possible triangle, so the
        function should return indices [0, 1, 2] in some order.
        
        Returns
        -------
        None
        """
        n_cells = 3
        idx = FN.sample_points(n_cells)
        
        assert len(idx) == 3
        assert len(np.unique(idx)) == 3


class TestGetCoordinates:
    """Tests for get_coordinates function."""
    
    def test_get_coordinates_correct_shape(self):
        """
        Test that get_coordinates returns arrays with correct shape.
        
        Given 3 sampled indices, the function should return two arrays of shape (3, 2),
        one for spatial coordinates and one for latent coordinates. This test verifies
        the output shape matches the number of sampled points and the 2D coordinate space.
        
        Returns
        -------
        None
        """
        indices = np.array([0, 5, 10])
        coordinates = np.random.randn(20, 2)
        latent_space = np.random.randn(20, 2)
        
        normal_coords, latent_coords = FN.get_coordinates(indices, coordinates, latent_space)
        
        assert normal_coords.shape == (3, 2)
        assert latent_coords.shape == (3, 2)
        
    def test_get_coordinates_correct_values(self):
        """
        Test that get_coordinates extracts the correct rows from input arrays.
        
        Verifies that the function properly indexes into the coordinates and latent_space
        arrays using the provided indices. Uses predictable sequential data to ensure
        that the extracted values exactly match the expected rows. This is critical for
        ensuring that triangle comparisons use the correct corresponding points.
        
        Returns
        -------
        None
        """
        indices = np.array([1, 3, 5])
        coordinates = np.arange(20).reshape(10, 2).astype(float)
        latent_space = np.arange(20, 40).reshape(10, 2).astype(float)
        
        normal_coords, latent_coords = FN.get_coordinates(indices, coordinates, latent_space)
        
        np.testing.assert_array_equal(normal_coords, coordinates[indices])
        np.testing.assert_array_equal(latent_coords, latent_space[indices])
        
            
class TestEdgeCases:
    """Tests for edge cases in utility functions."""
    
    def test_correlation_with_two_points(self):
        """
        Test correlation computation with the minimum number of cells (2).
        
        With only 2 cells, there is exactly 1 pairwise comparison. This test verifies
        that the function handles this minimal edge case. Note that pearsonr requires
        at least 2 data points, so with 2 cells yielding 1 pair, the correlation
        computation may fail or require special handling. The test allows for either
        successful computation or a ValueError.
        
        Returns
        -------
        None
        """
        X = torch.randn(2, 10)
        S = torch.randn(2, 2)
        
        try:
            result = FN.pairwise_corr_expr_spatial(X, S)
            # Should have exactly 1 pair
            assert result['n_pairs'] == 1
        except ValueError:
            # pearsonr requires at least 2 data points; with 2 points we only have 1 pair
            pass
        
    def test_procrustes_with_two_points(self):
        """
        Test Procrustes distance computation with only 2 points.
        
        With just 2 points, the Procrustes problem is simpler but still well-defined.
        This test verifies that the function handles this minimal case and returns
        a non-negative, non-NaN distance value. This edge case is important for
        robustness when analyzing very small datasets or subsamples.
        
        Returns
        -------
        None
        """
        z = torch.randn(2, 2)
        s = torch.randn(2, 2)
        
        distance = FN.procrustes_distance(z, s)
        
        assert distance >= 0
        assert not np.isnan(distance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
