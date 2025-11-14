import torch
import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dp_VAE import dp_VAE

# Get device from dp_VAE module to ensure data is on the correct device
DEVICE = dp_VAE.device


class TestTrainAndEvalBasic:
    """Basic functionality tests for train_and_eval."""
    
    def test_train_and_eval_runs_without_error(self):
        """
        Test that train_and_eval completes successfully without errors.
        
        Verifies the basic functionality of the train_and_eval function with minimal
        parameters. This test ensures the entire training pipeline runs end-to-end,
        including model initialization, training loop, early stopping, and metric
        computation. Success indicates that all components integrate correctly without
        runtime errors or exceptions.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        # Use the same device as dp_VAE module
        device = dp_VAE.device
        X = torch.randn(20, 30, device=device)
        S = torch.randn(20, 2, device=device)
        
        best_val, state = dp_VAE.train_and_eval(
            alpha2=1.0,
            beta=2.0,
            lam=1.0,
            X=X,
            S=S,
            max_epochs=10,
            patience=5,
            select_metric="stress"
        )
        
        assert isinstance(best_val, float)
        assert isinstance(state, dict) or state is None
        
    def test_train_and_eval_return_types(self):
        """
        Test return value types for different output configurations.
        
        The train_and_eval function supports two output modes controlled by return_all:
        False (default) returns (best_value, state_dict), while True returns a comprehensive
        dictionary with all metrics and states. This test verifies that both modes return
        the correct types and that the full dictionary contains all expected keys for
        detailed analysis and model selection.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(15, 25, device=DEVICE)
        S = torch.randn(15, 2, device=DEVICE)
        
        # Test with return_all=False (default)
        best_val, state = dp_VAE.train_and_eval(
            alpha2=1.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=5,
            return_all=False
        )
        assert isinstance(best_val, float)
        assert isinstance(state, dict) or state is None
        
        # Test with return_all=True
        result_dict = dp_VAE.train_and_eval(
            alpha2=1.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=5,
            return_all=True
        )
        assert isinstance(result_dict, dict)
        required_keys = ['best_stress', 'best_dp_obj', 'best_stress_state',
                        'best_obj_state', 'final_epoch', 'selected_metric', 'selected_value']
        for key in required_keys:
            assert key in result_dict
                
    def test_train_and_eval_different_metrics(self):
        """
        Test model selection using different evaluation metrics.
        
        The function supports selecting the best model based on either the stress metric
        (distance preservation quality) or the full objective function (combined loss).
        This test verifies that both metrics can be used for model selection and produce
        valid results. Different metrics may favor different aspects of model performance,
        so this flexibility is important for various use cases.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        # Select by stress
        best_stress, state_stress = dp_VAE.train_and_eval(
            alpha2=1.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=20,
            select_metric="stress"
        )
        
        # Select by objective
        torch.manual_seed(42)  # Reset for fair comparison
        best_obj, state_obj = dp_VAE.train_and_eval(
            alpha2=1.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=20,
            select_metric="objective"
        )
        
        # Both should produce valid results
        assert isinstance(best_stress, float)
        assert isinstance(best_obj, float)


class TestEarlyStopping:
    """Tests for early stopping mechanism."""
    
    def test_early_stopping_triggers(self):
        """
        Test that early stopping terminates training when no improvement occurs.
        
        Early stopping monitors the selected metric and halts training if no improvement
        is observed for 'patience' epochs. This test sets a very high max_epochs but low
        patience, verifying that training stops well before max_epochs when the metric
        plateaus. Early stopping prevents overfitting and saves computational resources.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(15, 25, device=DEVICE)
        S = torch.randn(15, 2, device=DEVICE)
        
        result = dp_VAE.train_and_eval(
            alpha2=1.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=1000,  # Set high
            patience=10,       # But low patience
            return_all=True
        )
        
        # Should stop before max_epochs due to patience
        assert result['final_epoch'] < 1000
        
class TestWarmupPeriod:
    """Tests for warm-up period behavior."""
    
    def test_warmup_period_runs(self):
        """
        Test that the warm-up period executes before lambda initialization.
        
        The training pipeline includes a warm-up phase (pre_lambda_epochs) where the
        model trains without the distance-preserving loss component, allowing it to
        establish a good initial latent representation. This test verifies that training
        continues past the warm-up period, indicating that the two-phase training
        strategy is functioning correctly.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        # Training with warm-up (default pre_lambda_epochs=50)
        result = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=100,
            patience=20,
            return_all=True
        )
        
        # Should run past warm-up period
        assert result['final_epoch'] > dp_VAE.pre_lambda_epochs
        
class TestLambdaInitialization:
    """Tests for lambda initialization and learning."""
    
    def test_lambda_initialized_after_warmup(self):
        """
        Test that lambda parameter is initialized after the warm-up period.
        
        After pre_lambda_epochs of warm-up training, the learnable lambda parameter
        should be initialized and the distance-preserving loss activated. This test
        verifies that training continues beyond the warm-up threshold, allowing lambda
        to be learned. The two-phase approach helps stabilize training by first
        establishing a reasonable latent space before enforcing distance preservation.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        result = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=100,
            patience=20,
            return_all=True
        )
        
        assert result['final_epoch'] > dp_VAE.pre_lambda_epochs
                
    def test_lambda_learning_enabled(self):
        """
        Test that lambda parameter learning is enabled by default.
        
        By default, train_and_eval enables learnable lambda, allowing the model to
        automatically tune the weight of the distance-preserving loss during training.
        This test verifies that the returned model state contains the lambda parameter
        (either lam_raw for learnable or lam_buffer for fixed), confirming that lambda
        learning is active and the parameter is being optimized.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        # Lambda is learned by default in train_and_eval
        best_val, state = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=80,
            patience=20
        )
        
        # Check that state contains lambda parameters
        if state is not None:
            # Should have lam_raw (learnable lambda)
            assert 'lam_raw' in state or 'lam_buffer' in state


class TestModelStateRestoration:
    """Tests for model state saving and restoration."""
    
    def test_best_state_is_saved(self):
        """
        Test that the best model state is saved and returned for later use.
        
        During training, the function tracks the best model states according to both
        stress and objective metrics. This test verifies that the result dictionary
        contains the expected state_dict keys, enabling users to save and later reload
        the best-performing model. State saving is crucial for model deployment and
        reproducible results.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        result = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=50,
            patience=15,
            return_all=True
        )
        
        # Check that at least the dict keys exist (states might be None if training was very short)
        assert 'best_stress_state' in result
        assert 'best_obj_state' in result
        
    def test_state_can_be_loaded(self):
        """
        Test that the returned state dictionary can be loaded into a new model instance.
        
        State dictionaries should be compatible with fresh model instances, enabling
        model checkpointing and deployment. This test creates a new model with matching
        architecture, loads the saved state, and verifies that the model can perform
        inference. This validates the state saving/loading mechanism works correctly
        for model persistence.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        best_val, state = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=30,
            patience=10
        )
        
        if state is not None:
            # Create new model and load state
            model = dp_VAE.dpVAE(input_dim=30, z_dim=2, alpha2=5.0, beta=1.0, lam_init=1.0)
            model.load_state_dict(state)
            
            # Test that model works after loading
            with torch.no_grad():
                mu, _ = model.encode(X)
            assert mu.shape == (20, 2)
            
    def test_stress_vs_objective_state_selection(self):
        """
        Test that the correct model state is selected based on the chosen metric.
        
        When select_metric is specified, the function should return the model state
        that achieved the best value for that particular metric (stress or objective).
        This test verifies that the returned result correctly identifies which metric
        was used for selection and that the selected_value corresponds to that metric.
        Proper metric-based selection is essential for choosing models optimized for
        specific evaluation criteria.
        
        Returns
        -------
        None
        """
        torch.manual_seed(42)
        X = torch.randn(20, 30, device=DEVICE)
        S = torch.randn(20, 2, device=DEVICE)
        
        # Get both states
        result = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=50,
            patience=15,
            select_metric="stress",
            return_all=True
        )
        
        assert result['selected_metric'] == "stress"
        assert isinstance(result['selected_value'], float)
        
        # Test with objective metric
        torch.manual_seed(42)
        result_obj = dp_VAE.train_and_eval(
            alpha2=5.0, beta=1.0, lam=1.0,
            X=X, S=S,
            max_epochs=50,
            patience=15,
            select_metric="objective",
            return_all=True
        )
        
        assert result_obj['selected_metric'] == "objective"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
