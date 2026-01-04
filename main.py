"""
MHT (Multiple Hypothesis Testing) Framework

This is the main framework for running LTT (Local Threshold Testing) algorithm
on cascaded LLM systems with statistical guarantees.

Usage:
    python main.py --data data/sampled_teleQnA_1k.txt --alpha 0.3 --delta 0.05

Arguments:
    --data: Path to dataset file
    --alpha: Target misalignment upper bound (default: 0.3)
    --delta: Risk tolerance level (default: 0.05)
    --M: Number of epsilon values for grid (default: 5)
    --Q: Number of lambda values for grid (default: 100)
    --output: Output directory for results (default: ./results/)

Example:
    python main.py \
        --data data/sampled_teleQnA_1k.txt \
        --alpha 0.3 \
        --delta 0.05 \
        --M 5 \
        --Q 100 \
        --output ./ltt_results/
"""

import argparse
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from ltt_fix_sequence_testing import ThreeTierLTTSolution


def load_data(data_path):
    """
    Load calibration and validation data
    
    Args:
        data_path: Path to data file
        
    Returns:
        calibration_data: Dict with calibration data
        validation_data: Dict with validation data
    """
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split data into calibration and validation (e.g., 70-30 split)
    total_questions = len(data)
    split_idx = int(total_questions * 0.7)
    
    question_ids = list(data.keys())
    np.random.shuffle(question_ids)
    
    calibration_ids = question_ids[:split_idx]
    validation_ids = question_ids[split_idx:]
    
    print(f"Total questions: {total_questions}")
    print(f"Calibration: {len(calibration_ids)}, Validation: {len(validation_ids)}")
    
    # For demonstration, create dummy uncertainty and confidence data
    # In practice, these should come from your model predictions
    calibration_data = {
        'U_edge': np.random.rand(len(calibration_ids)),
        'C_edge': np.random.rand(len(calibration_ids)),
        'U_cloud': np.random.rand(len(calibration_ids)),
        'C_cloud': np.random.rand(len(calibration_ids)),
        'y': ['edge' if np.random.rand() > 0.5 else 'cloud' for _ in calibration_ids]
    }
    
    validation_data = {
        'U_edge': np.random.rand(len(validation_ids)),
        'C_edge': np.random.rand(len(validation_ids)),
        'U_cloud': np.random.rand(len(validation_ids)),
        'C_cloud': np.random.rand(len(validation_ids))
    }
    
    return calibration_data, validation_data


def build_parameter_grid(M=5, Q=100):
    """
    Build parameter grid for epsilon and lambda
    
    Args:
        M: Number of epsilon values
        Q: Number of lambda values
        
    Returns:
        epsilon_values: Array of epsilon values
        lambda_values: Array of lambda values
    """
    # Epsilon values (uncertainty thresholds)
    epsilon_values = np.linspace(0.1, 0.5, M)
    
    # Lambda values (confidence thresholds) - monotonically decreasing
    lambda_values = np.linspace(0.9, 0.1, Q)
    
    print(f"Parameter grid: {M} epsilon values, {Q} lambda values")
    print(f"Epsilon range: [{epsilon_values[0]:.3f}, {epsilon_values[-1]:.3f}]")
    print(f"Lambda range: [{lambda_values[-1]:.3f}, {lambda_values[0]:.3f}]")
    
    return epsilon_values, lambda_values


def run_ltt_framework(args):
    """
    Run complete LTT framework
    
    Args:
        args: Command line arguments
    """
    print("="*80)
    print("LTT (Local Threshold Testing) Framework")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data: {args.data}")
    print(f"  Alpha (misalignment): {args.alpha}")
    print(f"  Delta (risk level): {args.delta}")
    print(f"  M (epsilon grid): {args.M}")
    print(f"  Q (lambda grid): {args.Q}")
    print("="*80)
    
    # Load data
    calibration_data, validation_data = load_data(args.data)
    
    # Build parameter grid
    epsilon_values, lambda_values = build_parameter_grid(M=args.M, Q=args.Q)
    
    # Initialize LTT solution
    ltt_solution = ThreeTierLTTSolution(risk_level=args.delta)
    
    # Build parallel sequences
    ltt_solution.build_parallel_sequences(epsilon_values, lambda_values)
    
    # Visualize sequences (optional)
    if args.visualize:
        print("\nVisualizing parallel sequences...")
        ltt_solution.visualize_parallel_sequences()
    
    # Run LTT algorithm
    print("\n" + "="*80)
    print("Running LTT Algorithm...")
    print("="*80)
    
    feasible_params = ltt_solution.run_ltt_on_parameters(
        calibration_data, 
        alpha_A=args.alpha
    )
    
    # Find optimal parameter
    if len(feasible_params) > 0:
        print("\n" + "="*80)
        print("Finding Optimal Solution...")
        print("="*80)
        
        # Cost parameters (you can adjust these)
        L_edge = 1.5
        L_cloud = 7.0
        L_human = 10.0
        
        optimal_param, min_cost = ltt_solution.find_optimal_param(
            feasible_params,
            validation_data,
            L_edge,
            L_cloud,
            L_human
        )
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        
        results = {
            'configuration': {
                'alpha': args.alpha,
                'delta': args.delta,
                'M': args.M,
                'Q': args.Q
            },
            'feasible_parameters': [
                {'epsilon': float(p[0]), 'lambda': float(p[1])} 
                for p in feasible_params
            ],
            'optimal_parameter': {
                'epsilon': float(optimal_param[0]),
                'lambda': float(optimal_param[1]),
                'cost': float(min_cost)
            },
            'cost_parameters': {
                'L_edge': L_edge,
                'L_cloud': L_cloud,
                'L_human': L_human
            }
        }
        
        output_file = os.path.join(args.output, 'ltt_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Summary
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Total parameter combinations tested: {len(ltt_solution.parameter_grid)}")
        print(f"Feasible parameters with guarantees: {len(feasible_params)}")
        print(f"Optimal parameter: ε={optimal_param[0]:.4f}, λ={optimal_param[1]:.4f}")
        print(f"Minimum cost: {min_cost:.4f}")
        print("="*80)
        
    else:
        print("\n" + "="*80)
        print("WARNING: No feasible parameters found!")
        print("Consider:")
        print("  - Relaxing alpha (increase misalignment tolerance)")
        print("  - Increasing delta (increase risk tolerance)")
        print("  - Expanding parameter grid (increase M and Q)")
        print("="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='LTT Framework for Cascaded LLM Systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --data data/sampled_teleQnA_1k.txt
  
  # Custom parameters
  python main.py --data data/sampled_teleQnA_1k.txt --alpha 0.2 --delta 0.1
  
  # Fine-grained grid
  python main.py --data data/sampled_teleQnA_1k.txt --M 10 --Q 200
  
  # With visualization
  python main.py --data data/sampled_teleQnA_1k.txt --visualize
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/sampled_teleQnA_1k.txt',
        help='Path to dataset file (default: data/sampled_teleQnA_1k.txt)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Target misalignment upper bound (default: 0.3)'
    )
    
    parser.add_argument(
        '--delta',
        type=float,
        default=0.05,
        help='Risk tolerance level (default: 0.05)'
    )
    
    parser.add_argument(
        '--M',
        type=int,
        default=5,
        help='Number of epsilon values for grid (default: 5)'
    )
    
    parser.add_argument(
        '--Q',
        type=int,
        default=100,
        help='Number of lambda values for grid (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results/',
        help='Output directory for results (default: ./results/)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize parallel sequences'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run LTT framework
    run_ltt_framework(args)


if __name__ == "__main__":
    main()