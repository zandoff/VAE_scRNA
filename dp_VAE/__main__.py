import dp_VAE as dp
import utils as FN
import training as TR
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
        print("Analysis mode not yet implemented")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()