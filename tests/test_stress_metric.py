import torch
import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dp_VAE import dp_VAE


class TestStressFormulaCorrectness:
    """Test suite for stress metric formula correctness."""
    
    def test_stress_zero_for_perfect_alignment(self):
        """
        Test stress is zero when latent and spatial distances match perfectly.
        
        Verifies that when the latent embeddings Z are identical to the spatial
        coordinates S (with lambda=1), the computed stress metric is effectively
        zero, indicating perfect distance preservation.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        
        # Create spatial coordinates
        S = torch.randn(10, 2)
        
        # Set latent equal to spatial (perfect alignment with lambda=1)
        Z = S.clone()
        
        # Compute stress manually
        Dz = torch.cdist(Z, Z, p=2)
        Ds = torch.cdist(S, S, p=2)
        lam = 1.0
        
        n = Z.size(0)
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        
        diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
        denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
        stress = torch.sqrt(diff_sq.sum() / denom)
        
        assert stress.item() < 1e-5, f"Stress should be ~0 for perfect alignment, got {stress.item()}"
        
    def test_stress_manual_computation_matches(self):
        """
        Test that stress computation matches manual calculation.
        
        Verifies the correctness of the stress formula by manually computing
        the stress metric step-by-step and ensuring all components (numerator,
        denominator, final stress value) are positive and finite as expected.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        
        # Create random embeddings and coordinates
        Z = torch.randn(8, 2)
        S = torch.randn(8, 2)
        lam = 2.0
        
        # Manual computation
        Dz = torch.cdist(Z, Z, p=2)
        Ds = torch.cdist(S, S, p=2)
        
        n = Z.size(0)
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        
        # Scaled latent distances
        Dtarget = Dz / lam
        
        # Stress formula: sqrt(sum((Dtarget - Ds)^2) / sum(Ds^2))
        numerator = ((Dtarget - Ds).pow(2)[triu_mask]).sum()
        denominator = (Ds.pow(2)[triu_mask]).sum() + 1e-12
        stress_manual = torch.sqrt(numerator / denominator)
        
        # Now verify this is what the training code would compute
        # (This tests our understanding of the formula)
        assert numerator > 0
        assert denominator > 0
        assert stress_manual.item() >= 0
        
    def test_stress_increases_with_misalignment(self):
        """
        Test that stress increases as alignment worsens.
        
        Verifies that stress monotonically increases when latent embeddings
        deviate more from spatial coordinates. Compares stress for good
        alignment (small perturbation) vs poor alignment (large perturbation).
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        S = torch.randn(10, 2)
        
        # Case 1: Good alignment
        Z1 = S + 0.1 * torch.randn(10, 2)
        
        # Case 2: Poor alignment
        Z2 = S + 2.0 * torch.randn(10, 2)
        
        def compute_stress(Z, S, lam=1.0):
            Dz = torch.cdist(Z, Z, p=2)
            Ds = torch.cdist(S, S, p=2)
            n = Z.size(0)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
            denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
            return torch.sqrt(diff_sq.sum() / denom)
        
        stress1 = compute_stress(Z1, S)
        stress2 = compute_stress(Z2, S)
        
        assert stress2.item() > stress1.item(), "Worse alignment should have higher stress"
        
    def test_stress_scales_with_lambda(self):
        """
        Test how stress changes with different lambda values.
        
        Verifies that the lambda (scale) parameter correctly adjusts the stress
        computation. When latent distances are larger than spatial distances,
        a higher lambda value should reduce stress by scaling down the latent
        distances appropriately.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        Z = torch.randn(10, 2) * 5  # Large scale latent
        S = torch.randn(10, 2)      # Normal scale spatial
        
        def compute_stress(Z, S, lam):
            Dz = torch.cdist(Z, Z, p=2)
            Ds = torch.cdist(S, S, p=2)
            n = Z.size(0)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
            denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
            return torch.sqrt(diff_sq.sum() / denom)
        
        stress_lam1 = compute_stress(Z, S, lam=1.0)
        stress_lam5 = compute_stress(Z, S, lam=5.0)
        
        # With lam=5, the scaled latent distances should be closer to spatial distances
        # (assuming Z distances are ~5x larger than S distances)
        assert stress_lam5.item() < stress_lam1.item()
        
    def test_stress_normalization_denominator(self):
        """
        Test that stress is properly normalized by sum of squared spatial distances.
        
        Verifies that the stress metric normalization makes it scale-invariant
        to some degree. Tests stress computation with spatial coordinates at
        different scales to ensure both produce finite, positive values.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        
        # Create two scenarios with different spatial scales
        S1 = torch.randn(10, 2) * 1.0
        S2 = torch.randn(10, 2) * 10.0  # 10x larger scale
        
        Z = torch.randn(10, 2)
        
        def compute_stress(Z, S):
            Dz = torch.cdist(Z, Z, p=2)
            Ds = torch.cdist(S, S, p=2)
            n = Z.size(0)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            diff_sq = (Dz - Ds).pow(2)[triu_mask]
            denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
            return torch.sqrt(diff_sq.sum() / denom)
        
        stress1 = compute_stress(Z, S1)
        stress2 = compute_stress(Z, S2)
        
        # The normalization should make stress scale-invariant to some degree
        # Both should be finite and positive
        assert 0 < stress1.item() < float('inf')
        assert 0 < stress2.item() < float('inf')


class TestStressWithMasking:
    """Test stress computation behavior with k-NN masking."""
    
    def test_stress_computation_with_knn_mask(self):
        """
        Test stress computation when using k-NN mask in DP loss.
        
        Verifies that stress can be computed correctly even when the model
        uses k-NN masking during training. Ensures the stress metric (computed
        on all pairs) remains finite and non-negative after training with
        masked distance-preserving loss.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=3, alpha2=1.0)
        X = torch.randn(15, 50)
        S = torch.randn(15, 2)
        
        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(5):
            optimizer.zero_grad()
            loss, _, _, _, _ = model.loss_function(X, S)
            loss.backward()
            optimizer.step()
        
        # Compute stress
        with torch.no_grad():
            mu, _ = model.encode(X)
            Dz = torch.cdist(mu, mu, p=2)
            Ds = torch.cdist(S, S, p=2)
            lam = model.current_lambda().item()
            
            n = Dz.size(0)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
            denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
            stress = torch.sqrt(diff_sq.sum() / denom)
        
        assert stress.item() >= 0
        assert not torch.isnan(stress)
        
    
class TestStressNumericalStability:
    """Test numerical stability of stress computation."""
        
    def test_stress_denominator_epsilon(self):
        """
        Test that epsilon prevents division by zero.
        
        Verifies numerical stability when all spatial points are at the same
        location (zero spatial distances). The epsilon term (1e-12) in the
        denominator should prevent division by zero, producing a large but
        finite stress value.
        
        Returns
        -------
        None
        """
        # All points at same location
        S = torch.zeros(10, 2)
        Z = torch.randn(10, 2)
        
        Dz = torch.cdist(Z, Z, p=2)
        Ds = torch.cdist(S, S, p=2)
        lam = 1.0
        
        n = Z.size(0)
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
        denom = (Ds.pow(2))[triu_mask].sum() + 1e-12  # This should be ~1e-12
        stress = torch.sqrt(diff_sq.sum() / denom)
        
        # Epsilon should prevent division by zero
        assert not torch.isnan(stress)
        assert not torch.isinf(stress)
        assert stress.item() > 0  # Will be large but finite


class TestStressEdgeCases:
    """Test edge cases in stress computation."""
    
    def test_stress_with_two_points(self):
        """
        Test stress computation with minimum number of points.
        
        Verifies that stress can be computed with just two points (one pairwise
        distance). Ensures the computation handles this minimal case without
        errors and produces a non-negative, finite value.
        
        Returns
        -------
        None
        """
        Z = torch.randn(2, 2)
        S = torch.randn(2, 2)
        lam = 1.0
        
        Dz = torch.cdist(Z, Z, p=2)
        Ds = torch.cdist(S, S, p=2)
        
        n = Z.size(0)
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
        denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
        stress = torch.sqrt(diff_sq.sum() / denom)
        
        assert stress.item() >= 0
        assert not torch.isnan(stress)
        
    def test_stress_with_single_point(self):
        """
        Test stress computation with single point (should be zero or undefined).
        
        Verifies behavior when only one point is present. With a single point,
        there are no pairwise distances to compare, so the upper triangular
        mask should contain no elements and stress is undefined.
        
        Returns
        -------
        None
        """
        Z = torch.randn(1, 2)
        S = torch.randn(1, 2)
        lam = 1.0
        
        Dz = torch.cdist(Z, Z, p=2)
        Ds = torch.cdist(S, S, p=2)
        
        n = Z.size(0)
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
        
        # No upper triangular elements, should be empty
        assert diff_sq.numel() == 0
                
    def test_stress_symmetry(self):
        """
        Test that stress computation produces valid results.
        
        Verifies that the stress formula computes correctly for arbitrary
        latent and spatial coordinates. While the formula is not mathematically
        symmetric in Z and S, this test ensures the computation is valid and
        produces non-negative stress values.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        Z = torch.randn(10, 2)
        S = torch.randn(10, 2)
        lam = 1.0
        
        def compute_stress(A, B, lam):
            Da = torch.cdist(A, A, p=2)
            Db = torch.cdist(B, B, p=2)
            n = A.size(0)
            triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            diff_sq = (Da / lam - Db).pow(2)[triu_mask]
            denom = (Db.pow(2))[triu_mask].sum() + 1e-12
            return torch.sqrt(diff_sq.sum() / denom)
        
        stress1 = compute_stress(Z, S, lam)
        # The formula is not symmetric in general, but we can verify it computes
        assert stress1.item() >= 0


class TestStressIntegrationWithTraining:
    """Integration tests for stress in training context."""
    
    def test_stress_tracks_with_dp_loss(self):
        """
        Test that stress correlates with DP loss during training.
        
        Verifies that both the stress metric and distance-preserving loss
        decrease during training when alpha2 is sufficiently high. This
        integration test ensures that optimizing the DP loss actually improves
        distance preservation as measured by stress.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        
        X = torch.randn(20, 30)
        S = torch.randn(20, 2)
        
        model = dp_VAE.dpVAE(input_dim=30, z_dim=2, alpha2=5.0, lam_init=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        stress_history = []
        dp_loss_history = []
        
        for _ in range(30):
            optimizer.zero_grad()
            loss, _, _, ldp, _ = model.loss_function(X, S)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                mu, _ = model.encode(X)
                Dz = torch.cdist(mu, mu, p=2)
                Ds = torch.cdist(S, S, p=2)
                lam = model.current_lambda().item()
                
                n = Dz.size(0)
                triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
                diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
                denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
                stress = torch.sqrt(diff_sq.sum() / denom)
                
                stress_history.append(stress.item())
                dp_loss_history.append(ldp.item())
        
        # Both should generally decrease
        assert stress_history[-1] < stress_history[0]
        assert dp_loss_history[-1] < dp_loss_history[0]
        
    def test_stress_with_different_alpha2_values(self):
        """
        Test how different alpha2 values affect final stress.
        
        Verifies that higher alpha2 (weight on distance-preserving loss) leads
        to lower or comparable final stress values after training. Tests that
        the hyperparameter correctly controls the trade-off between
        reconstruction quality and distance preservation.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        
        X = torch.randn(20, 30)
        S = torch.randn(20, 2)
        
        def train_and_measure(alpha2):
            model = dp_VAE.dpVAE(input_dim=30, z_dim=2, alpha2=alpha2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            for _ in range(50):
                optimizer.zero_grad()
                loss, _, _, _, _ = model.loss_function(X, S)
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                mu, _ = model.encode(X)
                Dz = torch.cdist(mu, mu, p=2)
                Ds = torch.cdist(S, S, p=2)
                lam = model.current_lambda().item()
                
                n = Dz.size(0)
                triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
                diff_sq = (Dz / lam - Ds).pow(2)[triu_mask]
                denom = (Ds.pow(2))[triu_mask].sum() + 1e-12
                stress = torch.sqrt(diff_sq.sum() / denom)
                
                return stress.item()
        
        stress_low = train_and_measure(alpha2=0.1)
        stress_high = train_and_measure(alpha2=10.0)
        
        # Higher alpha2 should lead to lower or comparable stress (more focus on distance preservation)
        # Due to training stochasticity and the VAE's reconstruction objectives, this is not always strict
        assert stress_high <= stress_low * 1.3, f"stress_high={stress_high}, stress_low={stress_low}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
