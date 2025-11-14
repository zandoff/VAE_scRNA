import torch
import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dp_VAE import dp_VAE


class TestDpVAEInitialization:
    """Test suite for model initialization."""
    
    def test_basic_initialization(self):
        """
        Test model creation with minimal required parameters.
        
        Verifies that the dpVAE model can be instantiated with just input_dim and z_dim,
        and that all default hyperparameters (hidden_dim, alpha1, beta) are correctly
        set to their expected values. This ensures the model has sensible defaults and
        can be used without extensive configuration.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=100, z_dim=2)
        assert model.z_dim == 2
        assert model.hidden_dim == 124  # default
        assert model.alpha1 == 1.0  # default
        assert model.beta == 2.0  # default
        
    def test_custom_hidden_dim(self):
        """
        Test model initialization with custom hidden layer dimension.
        
        Verifies that the hidden_dim parameter can be customized during model
        initialization and that the model correctly stores this value. The hidden
        dimension controls the capacity of the encoder and decoder networks,
        affecting the model's representational power.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=3, hidden_dim=64)
        assert model.hidden_dim == 64
        
    def test_learnable_lambda_initialization(self):
        """
        Test initialization of learnable lambda parameter.
        
        When learn_lam=True, lambda should be a trainable parameter initialized to
        lam_init. This test verifies that the model correctly sets up a learnable
        lambda parameter and that its initial value matches the specified initialization.
        Learnable lambda allows the model to automatically tune the weight of the
        distance-preserving loss during training.
        
        Returns
        -------
        None
        """
        lam_init = 2.5
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, lam_init=lam_init, learn_lam=True)
        assert model.learn_lam is True
        # Check that lambda is close to init value
        lam_current = model.current_lambda().item()
        assert abs(lam_current - lam_init) < 0.01
        
    def test_fixed_lambda_initialization(self):
        """
        Test initialization of fixed (non-learnable) lambda parameter.
        
        When learn_lam=False, lambda should be a constant buffer (not a trainable
        parameter) set to lam_init. This test verifies that the model correctly sets
        up a fixed lambda value that remains constant during training. Fixed lambda
        is useful when the optimal weight is known a priori.
        
        Returns
        -------
        None
        """
        lam_val = 1.5
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, lam_init=lam_val, learn_lam=False)
        assert model.learn_lam is False
        assert abs(model.current_lambda().item() - lam_val) < 1e-6
        
    def test_mask_k_single_value(self):
        """
        Test initialization with a single mask_k value.
        
        Verifies that mask_k can be set to a single integer value, which specifies
        the number of nearest neighbors to consider in the k-NN masking for the
        distance-preserving loss. This allows focusing the loss on local spatial
        neighborhoods rather than all pairwise distances.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=5)
        assert model.mask_k == 5
        
    def test_mask_k_list(self):
        """
        Test initialization with mask_k as a list of values.
        
        Verifies that mask_k can be set to a list of integers, enabling multi-scale
        k-NN masking. The model randomly selects one k value from the list during
        each training iteration, allowing the loss to capture both local and broader
        spatial relationships. This provides more flexible distance preservation.
        
        Returns
        -------
        None
        """
        mask_list = [3, 5, 10]
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=mask_list)
        assert model.mask_k == mask_list
        
    def test_hyperparameter_values(self):
        """
        Test initialization with custom hyperparameter values.
        
        Verifies that the loss function weighting hyperparameters (alpha1, beta, alpha2)
        can be customized during initialization. alpha1 weights the reconstruction and KL
        losses, beta weights the KL loss relative to reconstruction, and alpha2 weights
        the distance-preserving loss. Proper initialization of these values is critical
        for balancing the different loss components.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(
            input_dim=50, z_dim=2,
            alpha1=2.0, beta=3.0, alpha2=0.5
        )
        assert model.alpha1 == 2.0
        assert model.beta == 3.0
        assert model.alpha2 == 0.5


class TestDpVAEForwardPass:
    """Test suite for forward pass and basic operations."""
    
    def test_forward_output_shapes(self):
        """
        Test that forward pass produces outputs with correct shapes.
        
        The forward method should return four tensors: reconstructed input (x_recon),
        mean of latent distribution (mu), log-variance of latent distribution (logvar),
        and sampled latent representation (z). This test verifies that all outputs
        have the expected shapes matching the batch size, input dimension, and latent
        dimension. Correct shapes are essential for downstream loss computation.
        
        Returns
        -------
        None
        """
        batch_size = 32
        input_dim = 100
        z_dim = 2
        
        model = dp_VAE.dpVAE(input_dim=input_dim, z_dim=z_dim)
        x = torch.randn(batch_size, input_dim)
        
        x_recon, mu, logvar, z = model.forward(x)
        
        assert x_recon.shape == (batch_size, input_dim)
        assert mu.shape == (batch_size, z_dim)
        assert logvar.shape == (batch_size, z_dim)
        assert z.shape == (batch_size, z_dim)
        
    def test_encode_output_shapes(self):
        """
        Test that encoder produces outputs with correct shapes.
        
        The encode method maps input data to the latent space parameters (mu and logvar)
        that define the approximate posterior distribution. This test verifies that both
        outputs have shape (batch_size, z_dim), ensuring the encoder correctly projects
        input data to the latent space dimensionality.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=3)
        x = torch.randn(10, 50)
        
        mu, logvar = model.encode(x)
        
        assert mu.shape == (10, 3)
        assert logvar.shape == (10, 3)
        
    def test_decode_output_shape(self):
        """
        Test that decoder produces output with correct shape.
        
        The decode method maps latent representations back to the input space to
        reconstruct the original data. This test verifies that the decoder output
        has shape (batch_size, input_dim), ensuring the decoder correctly projects
        from latent space back to the original data dimensionality.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2)
        z = torch.randn(10, 2)
        
        x_recon = model.decode(z)
        
        assert x_recon.shape == (10, 50)
            
class TestDpVAELossComponents:
    """Test suite for individual loss components."""
    
    def test_kl_loss_positive(self):
        """
        Test that KL divergence is always non-negative.
        
        By the properties of KL divergence, D_KL(q||p) ≥ 0 with equality only when
        q = p. In the VAE, we compute KL divergence between the approximate posterior
        q(z|x) and the prior p(z) = N(0,I). This test verifies this fundamental
        mathematical property holds in the implementation, ensuring correctness of
        the KL loss computation.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2)
        mu = torch.randn(10, 2)
        logvar = torch.randn(10, 2)
        
        kl = model.kl_loss(mu, logvar)
        
        assert kl.item() >= 0, "KL divergence must be non-negative"
                
    def test_recon_loss_shape(self):
        """
        Test that reconstruction loss returns per-sample values.
        
        The reconstruction loss should be computed separately for each sample in the
        batch, returning a vector of shape (batch_size,). This per-sample granularity
        is important for analyzing which samples are harder to reconstruct and for
        potential weighted loss formulations. This test verifies the correct output shape.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2)
        x = torch.randn(10, 50)
        x_recon = torch.randn(10, 50)
        
        rec_vec = model.recon_loss(x, x_recon)
        
        assert rec_vec.shape == (10,), "Should return per-sample losses"
            
    def test_recon_loss_positive(self):
        """
        Test that all reconstruction loss values are non-negative.
        
        Reconstruction loss typically uses MSE or similar metrics that are inherently
        non-negative. Each per-sample reconstruction loss should be ≥ 0, with zero
        indicating perfect reconstruction. This test verifies that the loss computation
        maintains this property for all samples in the batch, ensuring no numerical
        errors produce negative loss values.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2)
        x = torch.randn(10, 50)
        x_recon = torch.randn(10, 50)
        
        rec_vec = model.recon_loss(x, x_recon)
        
        assert torch.all(rec_vec >= 0)
        
    def test_distance_preserving_loss_no_mask(self):
        """
        Test distance-preserving loss without k-NN masking.
        
        When mask_k=None, the distance-preserving loss should consider all pairwise
        distances between cells. This test verifies that the loss can be computed
        without masking, is non-negative (as it's based on squared distance differences),
        and returns a scalar value for backpropagation. This mode is useful for small
        datasets where all pairwise relationships matter.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=None)
        z = torch.randn(10, 2)
        s = torch.randn(10, 2)
        
        ldp = model.distance_preserving_loss(z, s, mask=None)
        
        assert ldp.item() >= 0
        assert ldp.shape == torch.Size([])  # scalar
        
    def test_distance_preserving_loss_with_mask_k(self):
        """
        Test distance-preserving loss with k-nearest neighbor masking.
        
        When mask_k is specified, the distance-preserving loss should only consider
        distances between each cell and its k nearest neighbors in spatial coordinates.
        This test verifies that the loss can be computed with masking and returns a
        non-negative scalar. k-NN masking focuses the loss on local neighborhoods,
        which is important for large datasets and spatially heterogeneous tissues.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=3)
        z = torch.randn(10, 2)
        s = torch.randn(10, 2)
        
        ldp = model.distance_preserving_loss(z, s, mask=None)
        
        assert ldp.item() >= 0
        
class TestDpVAEFullLoss:
    """Test suite for the complete loss function."""
    
    def test_loss_function_output_structure(self):
        """
        Test that loss_function returns the correct tuple structure and shapes.
        
        The loss_function should return a 5-element tuple: (total_loss, rec_vec, kl, ldp, lam).
        This test verifies that all components are present and have the expected shapes:
        total_loss is a scalar for backpropagation, rec_vec has per-sample reconstruction
        losses (batch_size,), and kl, ldp, lam are scalars. This structure is essential
        for monitoring individual loss components during training.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2)
        x = torch.randn(10, 50)
        s = torch.randn(10, 2)
        
        result = model.loss_function(x, s)
        
        assert len(result) == 5
        total, rec_vec, kl, ldp, lam = result
        assert total.shape == torch.Size([])  # scalar
        assert rec_vec.shape == (10,)  # per-sample
        assert kl.shape == torch.Size([])  # scalar
        assert ldp.shape == torch.Size([])  # scalar
        assert lam.shape == torch.Size([])  # scalar
        
    def test_loss_function_components_contribute(self):
        """
        Test that all loss components are correctly combined into the total loss.
        
        The total loss should be computed as: alpha1 * (mean(rec_vec) + beta * kl) + alpha2 * ldp.
        This test verifies that the weighted combination of reconstruction, KL divergence,
        and distance-preserving losses is correctly implemented with the specified weighting
        hyperparameters. Correct loss composition is critical for balancing the different
        training objectives.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, alpha1=1.0, beta=1.0, alpha2=1.0)
        x = torch.randn(10, 50)
        s = torch.randn(10, 2)
        
        total, rec_vec, kl, ldp, lam = model.loss_function(x, s)
        
        # Total should be roughly: alpha1 * (rec_mean + beta * kl) + alpha2 * ldp
        expected = model.alpha1 * (rec_vec.mean() + model.beta * kl) + model.alpha2 * ldp
        assert torch.allclose(total, expected, rtol=1e-4)
            
    def test_loss_backpropagation(self):
        """
        Test that the loss function supports gradient backpropagation.
        
        For training, the total loss must be differentiable with respect to all model
        parameters. This test verifies that calling backward() on the total loss
        successfully computes gradients for at least some model parameters. This is
        essential for gradient-based optimization and ensures the loss computation
        maintains the computational graph.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2)
        x = torch.randn(10, 50)
        s = torch.randn(10, 2)
        
        total, _, _, _, _ = model.loss_function(x, s)
        
        # Check gradient computation works
        total.backward()
        
        # Check that gradients were computed for some parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should be computed"

class TestMaskGeneration:
    """Test suite for mask generation and caching."""
    
    def test_no_mask_when_mask_k_none(self):
        """
        Test that no mask is generated when mask_k is None.
        
        When mask_k is None, the model should not apply k-NN masking to the distance-
        preserving loss, considering all pairwise distances. This test verifies that
        the mask generation function correctly returns None in this case, which signals
        downstream code to use the full distance matrix without masking.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=None)
        s = torch.randn(10, 2)
        
        mask = model._get_cached_mask(s)
        
        assert mask is None
        
    def test_mask_is_symmetric(self):
        """
        Test that the generated k-NN mask is symmetric.
        
        Since spatial proximity is a symmetric relation (if A is a neighbor of B, then B
        is a neighbor of A), the k-NN mask should be symmetric. This test verifies that
        mask[i,j] == mask[j,i] for all i,j, ensuring that the distance-preserving loss
        treats pairwise relationships consistently in both directions. This is important
        for stable and unbiased distance preservation.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=3)
        s = torch.randn(10, 2)
        
        mask = model._get_cached_mask(s)
        
        assert torch.all(mask == mask.T), "Mask should be symmetric"
        
    def test_mask_diagonal_is_false(self):
        """
        Test that mask diagonal contains False values (no self-loops).
        
        The distance between a cell and itself is always zero and provides no useful
        information for the distance-preserving loss. This test verifies that the
        diagonal of the mask is False, ensuring that self-distances are excluded from
        the loss computation. This prevents trivial zero-distance pairs from dominating
        the loss and ensures focus on meaningful pairwise relationships.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=3)
        s = torch.randn(10, 2)
        
        mask = model._get_cached_mask(s)
        
        assert torch.all(~torch.diag(mask)), "Diagonal should be False"
                
    def test_mask_with_none_in_list_disables_masking(self):
        """
        Test that including None in mask_k list can disable masking.
        
        When mask_k is a list containing None, the model randomly selects a k value
        from the list on each iteration. If None is selected, masking should be disabled
        for that iteration. This test verifies that the mask generation correctly handles
        None values, allowing for a mix of masked and unmasked training iterations to
        capture both local and global distance relationships.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=[3, None])
        s = torch.randn(10, 2)
        
        mask = model._get_cached_mask(s)
        
        assert mask is None
        
    def test_mask_caching(self):
        """
        Test that k-NN masks are cached and reused for efficiency.
        
        Computing k-NN masks can be expensive, especially for large datasets. The model
        should cache masks based on spatial coordinates to avoid recomputation. This test
        verifies that calling _get_cached_mask twice with the same spatial coordinates
        returns the exact same mask object (not just equal values), confirming that
        caching is working. This optimization is important for training efficiency.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=3)
        s = torch.randn(10, 2)
        
        # First call should compute mask
        mask1 = model._get_cached_mask(s)
        
        # Second call should return cached mask (same object)
        mask2 = model._get_cached_mask(s)
        
        assert mask1 is mask2, "Mask should be cached"
        
class TestEdgeCases:
    """Test suite for edge cases and error handling."""
        
    def test_model_on_cuda_if_available(self):
        """
        Test that model can be moved to CUDA device if available.
        
        For large-scale training, GPU acceleration is essential. This test verifies
        that the model and all its operations can be executed on CUDA devices. It
        checks that moving the model and tensors to GPU works correctly and that
        the loss computation produces outputs on the correct device. This is skipped
        if CUDA is not available on the system.
        
        Returns
        -------
        None
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2).cuda()
        x = torch.randn(10, 50).cuda()
        s = torch.randn(10, 2).cuda()
        
        total, _, _, _, _ = model.loss_function(x, s)
        
        assert total.device.type == 'cuda'
        
    def test_mask_k_larger_than_n(self):
        """
        Test mask generation when k exceeds the number of available neighbors.
        
        When mask_k is larger than n-1 (where n is the batch size), there aren't enough
        cells to form k neighbors for each cell. The implementation should handle this
        gracefully, typically by clamping k to n-1 or including all available neighbors.
        This test verifies robustness to misconfigured mask_k values, which can occur
        with small batches or datasets.
        
        Returns
        -------
        None
        """
        model = dp_VAE.dpVAE(input_dim=50, z_dim=2, mask_k=20)
        s = torch.randn(10, 2)  # Only 10 samples, so max k is 9
        
        mask = model._get_cached_mask(s)
        
        # Should still work, clamping k to n-1
        assert mask is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
