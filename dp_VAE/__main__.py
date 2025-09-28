from dp_VAE import dp_VAE as dp
from dp_VAE import utils as FN
from dp_VAE import train_eval as TR
import warnings
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scanpy as sc
import torch
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy.plotting._utils")

matplotlib.use("Agg")

def main():
    """
    Main entry point for the dp-VAE testing script.
    Parses command line arguments and calls the appropriate functions.
    """
    
    parser = argparse.ArgumentParser(description="dp-VAE Testing Script")
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for running the full test pipeline
    run_parser = subparsers.add_parser('run', help='Run the full testing pipeline')
    
    # Dataset selection
    run_parser.add_argument("--datasets", type=str, nargs='+', 
                           help="Specific datasets to use. Options: sagittal_posterior, sagittal_posterior_2, sagittal_anterior, sagittal_anterior_2, whole_brain, kidney. Default: all datasets")
    
    # Hyperparameter sweep settings
    run_parser.add_argument("--alpha2", type=float, nargs='+', default=[5, 10, 20, 40, 80],
                           help="Values for alpha2 parameter (DP loss weight). Default: [5, 10, 20, 40, 80]")
    
    run_parser.add_argument("--mask_k", type=str, nargs='+', default=["None", "5", "10", "16", "32"],
                           help="Values for mask_k parameter. Use 'None' for no masking. Default: ['None', '5', '10', '16', '32']")
    
    run_parser.add_argument("--lam_factors", type=float, nargs='+', default=[0.1, 0.3, 0.5, 1, 2, 3, 10],
                           help="Scaling factors for lambda. Default: [0.1, 0.3, 0.5, 1, 2, 3, 10]")
    
    # Training settings
    run_parser.add_argument("--max_epochs", type=int, default=2000,
                           help="Maximum number of training epochs. Default: 2000")
    
    run_parser.add_argument("--patience", type=int, default=200,
                           help="Patience for early stopping. Default: 200")
    
    # Output settings
    run_parser.add_argument("--output_dir", type=str, default="./results",
                           help="Directory for output files. Default: ./results")
    
    # Device
    run_parser.add_argument("--device", type=str, default=None,
                           help="Device to use (cuda or cpu). Default: auto-detect")

    # Subparser for analysis only
    analyze_parser = subparsers.add_parser('analyze', help='Analyze saved models')
    analyze_parser.add_argument("--model_dir", type=str, required=True,
                               help="Directory containing saved models")
    
    args = parser.parse_args()
    
    # Set device
    if args.command == 'run' and args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    if args.command == 'run':
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Parse mask_k values
        mask_k_values = []
        for mk in args.mask_k:
            if mk.lower() == 'none':
                mask_k_values.append(None)
            elif mk.isdigit():
                mask_k_values.append(int(mk))
            else:
                try:
                    # Try to parse as tuple/list
                    mask_k_values.append(eval(mk))
                except:
                    print(f"Warning: Could not parse mask_k value '{mk}'. Skipping.")
        
        print("\n=== PREPROCESSING DATA ===")
        # If datasets is None, pass None to use all available datasets
        dataset_keys = args.datasets if args.datasets else None
        XS_pairs, splits = FN.preprocess_data(dataset_keys=dataset_keys, device=device)
        
        print("\n=== PARAMETER SWEEP ===")
        (results_all, best_states_all, best_params_all,
         procrustes_all, best_states_procrustes_all,
         best_params_procrustes_all, best_params_mixed_all) = TR.parameter_sweep(
            XS_pairs, splits, 
            alpha2_values=args.alpha2,
            mask_k_values=mask_k_values,
            lam_factors=args.lam_factors,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device
        )
        
        print("\n=== BUILDING BEST MODELS ===")
        best_models_stress, best_models_procrustes, best_models_mixed, selection_summary = TR.build_best_models(
            XS_pairs, splits,
            best_params_all, best_states_all,
            best_params_procrustes_all, best_states_procrustes_all,
            best_params_mixed_all, device=device
        )
        
        print("\n=== ANALYZING TRIPLET GEOMETRY ===")
        triplet_results = FN.analyze_triplet_geometry(
            XS_pairs, splits,
            best_models_stress, best_models_procrustes, best_models_mixed,
            output_dir=args.output_dir
        )
        
        print("\n=== GENERATING HEATMAPS ===")
        FN.generate_heatmaps(
            XS_pairs, splits, best_models_stress,
            output_dir=args.output_dir
        )
        
        print("\n=== SAVING RESULTS ===")
        # Save results summary
        for key in selection_summary:
            # Save best models
            model_path = os.path.join(args.output_dir, f"best_model_stress_{key}.pt")
            torch.save(best_models_stress[key].state_dict(), model_path)
            
            model_path = os.path.join(args.output_dir, f"best_model_procrustes_{key}.pt")
            torch.save(best_models_procrustes[key].state_dict(), model_path)
            
            model_path = os.path.join(args.output_dir, f"best_model_mixed_{key}.pt")
            torch.save(best_models_mixed[key].state_dict(), model_path)
        
        # Save parameters and summary
        with open(os.path.join(args.output_dir, "selection_summary.txt"), 'w') as f:
            for key, summary in selection_summary.items():
                f.write(f"Dataset: {key}\n")
                f.write(f"  Stress-selected params: {summary['stress_selected_params']} -> Test Stress={summary['stress_selected_test_stress']:.4f}, Test Procrustes={summary['stress_selected_test_procrustes']:.4f}\n")
                f.write(f"  Procrustes-selected params: {summary['procrustes_selected_params']} -> Test Stress={summary['procrustes_selected_test_stress']:.4f}, Test Procrustes={summary['procrustes_selected_test_procrustes']:.4f}\n")
                f.write(f"  Mixed-selected params: {summary['mixed_selected_params']} -> Test Stress={summary['mixed_selected_test_stress']:.4f}, Test Procrustes={summary['mixed_selected_test_procrustes']:.4f}\n")
                f.write("  --\n")
        
        print(f"\nAll results saved to {args.output_dir}")
    
    elif args.command == 'analyze':
        model_dir = args.model_dir
        if not os.path.exists(model_dir):
            print(f"Error: Model directory '{model_dir}' not found.")
            return
            
        print(f"\n=== LOADING MODELS FROM {model_dir} ===")
        
        # Get the dataset keys from the available model files
        model_files = [f for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pt")]
        dataset_keys = set()
        for mf in model_files:
            # Extract dataset key from filename: best_model_TYPE_KEY.pt
            parts = mf.split('_')
            if len(parts) >= 4:
                dataset_keys.add('_'.join(parts[3:]).replace('.pt', ''))
        
        if not dataset_keys:
            print("Error: No model files found with expected naming pattern.")
            return
            
        print(f"Found models for datasets: {', '.join(dataset_keys)}")
        
        # If datasets is None, pass None to use all available datasets
        print("\n=== PREPROCESSING DATA ===")
        XS_pairs, splits = FN.preprocess_data(dataset_keys=list(dataset_keys) if dataset_keys else None, device=device)
        
        # Load the models
        best_models_stress = {}
        best_models_procrustes = {}
        best_models_mixed = {}
        
        for key in dataset_keys:
            # Get dataset dimensions to create model instances
            X, _ = XS_pairs[key]
            input_dim = X.shape[1]
            
            # Create model instances
            model_stress = dp.dpVAE(input_dim=input_dim)
            model_procrustes = dp.dpVAE(input_dim=input_dim)
            model_mixed = dp.dpVAE(input_dim=input_dim)
            
            # Load state dictionaries
            stress_path = os.path.join(model_dir, f"best_model_stress_{key}.pt")
            procrustes_path = os.path.join(model_dir, f"best_model_procrustes_{key}.pt")
            mixed_path = os.path.join(model_dir, f"best_model_mixed_{key}.pt")
            
            if os.path.exists(stress_path):
                model_stress.load_state_dict(torch.load(stress_path, map_location=device))
                model_stress = model_stress.to(device)
                best_models_stress[key] = model_stress
                print(f"Loaded stress model for {key} (on {device})")
                
            if os.path.exists(procrustes_path):
                model_procrustes.load_state_dict(torch.load(procrustes_path, map_location=device))
                model_procrustes = model_procrustes.to(device)
                best_models_procrustes[key] = model_procrustes
                print(f"Loaded procrustes model for {key} (on {device})")
                
            if os.path.exists(mixed_path):
                model_mixed.load_state_dict(torch.load(mixed_path, map_location=device))
                model_mixed = model_mixed.to(device)
                best_models_mixed[key] = model_mixed
                print(f"Loaded mixed model for {key} (on {device})")
        
        # Set up output directory
        output_dir = os.path.join(model_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== ANALYZING TRIPLET GEOMETRY ===")
        triplet_results = FN.analyze_triplet_geometry(
            XS_pairs, splits,
            best_models_stress, best_models_procrustes, best_models_mixed,
            output_dir=output_dir
        )
        
        print("\n=== GENERATING HEATMAPS ===")
        FN.generate_heatmaps(
            XS_pairs, splits, best_models_stress,
            output_dir=output_dir
        )
        
        print(f"\nAnalysis results saved to {output_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()