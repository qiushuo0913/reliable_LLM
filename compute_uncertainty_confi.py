"""
Uncertainty and Confidence Computation Module

This module provides two methods for computing uncertainty and confidence
in cascaded LLM systems:

1. Bayesian Method: Uses Bayesian inference with dropout-based uncertainty
2. Permutation Method: Uses option permutation-based uncertainty

Usage:
    # Bayesian method
    results = compute_uncertainty_confidence(
        bayesian_file='bayesian_results.json',
        ensemble_file='ensemble_results.json', 
        method='bayesian'
    )
    
    # Permutation method
    results = compute_uncertainty_confidence(
        models_config={'qwen2-1.5B': 'Qwen/Qwen2-1.5B-Instruct'},
        K=5,
        method='permutation'
    )

Functions:
    - compute_uncertainty_confidence(): Main interface for both methods
    - compute_bayesian_uncertainty(): Bayesian inference method
    - compute_permutation_uncertainty(): Option permutation method
    - compute_ece(): Expected Calibration Error
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from glob import glob
import random


def load_json_data(file_path):
    """
    Load JSON data from file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        ece: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in current bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and average confidence in bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Weighted ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


# ==================== BAYESIAN METHOD ====================

def find_common_questions_bayesian(qwen2_data, qwen3_data):
    """
    Find common questions with tested answers between two models (Bayesian method)
    
    Args:
        qwen2_data: Data from first model (e.g., Qwen2-1.5B)
        qwen3_data: Data from second model (e.g., Qwen2-7B or Qwen3-4B)
        
    Returns:
        common_questions: Dictionary of common questions with both models' data
    """
    qwen2_questions = {}
    qwen3_questions = {}
    
    # Extract questions with tested_answer from Qwen2
    for q_id, q_data in sorted(qwen2_data.items()):
        if "tested answer" in q_data:
            qwen2_questions[q_id] = q_data
    
    # Extract questions with tested_answer from Qwen3
    for q_id, q_data in sorted(qwen3_data.items()):
        if "tested answer" in q_data:
            qwen3_questions[q_id] = q_data
    
    # Find common question IDs
    common_question_ids = sorted(list(set(qwen2_questions.keys()) & set(qwen3_questions.keys())))
    
    # Create result dictionary with both models' data
    common_questions = {}
    for q_id in common_question_ids:
        common_questions[q_id] = {
            "question": qwen2_questions[q_id]["question"],
            "qwen2-1.5B": {
                "tested answer": qwen2_questions[q_id]["tested answer"],
                "correct": qwen2_questions[q_id].get("correct", False),
                "confidence": qwen2_questions[q_id].get("confidence", 0.0),
            },
            "qwen2-7B": {
                "tested answer": qwen3_questions[q_id]["tested answer"],
                "correct": qwen3_questions[q_id].get("correct", False),
                "confidence": qwen3_questions[q_id].get("confidence", 0.0),
            }
        }
    
    return common_questions


def apply_bayesian_calibration(common_questions, bayesian_data, ensemble_data):
    """
    Apply Bayesian calibration to compute uncertainty and confidence
    
    Args:
        common_questions: Common questions between models
        bayesian_data: Bayesian inference results
        ensemble_data: Ensemble inference results
        
    Returns:
        results: Dictionary with calibrated confidence and uncertainty
    """
    results = {}
    
    # Store all predictions for computing metrics
    model_metrics_data = {
        "qwen2-1.5B": {"all_correct": [], "all_conf": []},
        "qwen2-7B": {"all_correct": [], "all_conf": []}
    }
    
    for q_id, q_data in common_questions.items():
        results[q_id] = {"question": q_data["question"]}
        
        # Process each model
        for model_key in ["qwen2-1.5B", "qwen2-7B"]:
            if model_key in q_data:
                correct = q_data[model_key]["correct"]
                original_confidence = q_data[model_key]["confidence"]
                
                # Get Bayesian and ensemble confidences
                bayesian_conf = bayesian_data.get(q_id, {}).get(model_key, {}).get("confidence", original_confidence)
                ensemble_conf = ensemble_data.get(q_id, {}).get(model_key, {}).get("confidence", original_confidence)
                
                # Compute calibrated confidence (average of Bayesian and ensemble)
                calibrated_confidence = (bayesian_conf + ensemble_conf) / 2.0
                
                # Compute uncertainty (standard deviation)
                uncertainty = np.std([bayesian_conf, ensemble_conf])
                
                # Collect data for metrics
                model_metrics_data[model_key]["all_correct"].append(correct)
                model_metrics_data[model_key]["all_conf"].append(calibrated_confidence)
                
                results[q_id][model_key] = {
                    "correct": correct,
                    "original_confidence": float(original_confidence),
                    "calibrated_confidence": float(calibrated_confidence),
                    "uncertainty": float(uncertainty)
                }
    
    # Compute evaluation metrics
    evaluation_metrics = {"qwen2-1.5B": {}, "qwen2-7B": {}}
    
    for model_key in ["qwen2-1.5B", "qwen2-7B"]:
        if len(model_metrics_data[model_key]["all_correct"]) > 0:
            all_correct = np.array(model_metrics_data[model_key]["all_correct"])
            all_conf = np.array(model_metrics_data[model_key]["all_conf"])
            
            accuracy = np.mean(all_correct)
            ece = compute_ece(all_correct, all_conf)
            
            evaluation_metrics[model_key] = {
                "accuracy": float(accuracy),
                "ece": float(ece)
            }
    
    results["evaluation_metrics"] = evaluation_metrics
    
    return results


# ==================== PERMUTATION METHOD ====================

def find_common_questions_permutation(models_config, K=5, seed=42):
    """
    Find common questions across all permutation files
    
    Args:
        models_config: Dictionary mapping model keys to model names
        K: Number of permutations
        seed: Random seed for sampling
        
    Returns:
        common_questions: Common questions data
        all_permutation_data: All permutation files data
    """
    all_permutation_data = {"qwen2-1.5B": [], "qwen2-7B": []}
    all_question_sets = []
    
    # Load all permutation files
    for model_key, model_name in models_config.items():
        results_dir = f"{model_name.replace('/', '_')}_permutation_self_eval_results"
        
        if not os.path.exists(results_dir):
            print(f"Results directory {results_dir} not found!")
            continue
        
        # Find all permutation files for this model
        pattern = os.path.join(results_dir, f"{model_name.replace('/', '_')}_self_eval_permutation_*.json")
        result_files = sorted(glob(pattern))
        
        # Randomly select K files
        random.seed(seed) 
        if len(result_files) > K:
            selected_files = random.sample(result_files, K)
        else:
            selected_files = result_files
        
        for i, file_path in enumerate(selected_files):
            data = load_json_data(file_path)
            all_permutation_data[model_key].append(data)
            
            # Collect question IDs with tested answers
            questions_with_answers = set()
            for q_id, q_data in data.items():
                if "tested answer" in q_data:
                    questions_with_answers.add(q_id)
            
            all_question_sets.append(questions_with_answers)
            print(f"Loaded {model_key} permutation {i}: {len(questions_with_answers)} questions with tested answers")
    
    # Find common questions across all files
    if all_question_sets:
        common_question_ids = sorted(list(set.intersection(*all_question_sets)))
        print(f"Found {len(common_question_ids)} common questions across all {len(all_question_sets)} permutation files")
    else:
        common_question_ids = []
        print("No permutation files loaded!")
    
    # Build common questions data structure
    common_questions = {}
    if common_question_ids and all_permutation_data["qwen2-1.5B"]:
        # Use first qwen2 permutation file as base structure
        base_data = all_permutation_data["qwen2-1.5B"][0]
        
        for q_id in common_question_ids:
            if q_id in base_data:
                common_questions[q_id] = {
                    "question": base_data[q_id].get("question", "")
                }
    
    return common_questions, all_permutation_data


def apply_permutation_calibration(common_questions, all_permutation_data):
    """
    Compute calibrated confidence and uncertainty using option permutation
    
    Args:
        common_questions: Common questions
        all_permutation_data: All permutation data
    
    Returns:
        results: Results with calibrated confidence and uncertainty
    """
    results = {}
    
    # Store all predictions for computing metrics
    model_metrics_data = {
        "qwen2-1.5B": {"all_correct": [], "all_conf": []},
        "qwen2-7B": {"all_correct": [], "all_conf": []}
    }
    
    for q_id in common_questions.keys():
        # Initialize result structure
        results[q_id] = {
            "question": common_questions[q_id]["question"]
        }
        
        # Compute calibration metrics for each model
        for model_key in ["qwen2-1.5B", "qwen2-7B"]:
            confidences = []
            tested_answer = ""
            correct = False
            
            # Collect confidence across all permutations
            for perm_data in all_permutation_data[model_key]:
                if q_id in perm_data:
                    q_data = perm_data[q_id]
                    
                    # Collect confidence
                    if "confidence" in q_data:
                        confidences.append(q_data["confidence"])
                    
                    # Get tested_answer and correct (use first valid value)
                    if not tested_answer and "tested answer" in q_data:
                        tested_answer = q_data["tested answer"]
                    if "correct" in q_data:
                        correct = q_data["correct"]
            
            # Compute calibration metrics
            if len(confidences) > 0:
                original_confidence = confidences[0]  # Use first permutation as original
                calibrated_confidence = np.mean(confidences)  # Average confidence
                uncertainty = np.std(confidences)  # Standard deviation as uncertainty
                
                # Collect data for metrics
                model_metrics_data[model_key]["all_correct"].append(correct)
                model_metrics_data[model_key]["all_conf"].append(calibrated_confidence)
                
                results[q_id][model_key] = {
                    "correct": correct,
                    "original_confidence": float(original_confidence),
                    "calibrated_confidence": float(calibrated_confidence),
                    "uncertainty": float(uncertainty)
                }
            else:
                print(f"Warning: No confidence data found for question {q_id} in model {model_key}")
    
    # Compute evaluation metrics for each model
    evaluation_metrics = {"qwen2-1.5B": {}, "qwen2-7B": {}}
    
    for model_key in ["qwen2-1.5B", "qwen2-7B"]:
        if len(model_metrics_data[model_key]["all_correct"]) > 0:
            # Convert to numpy arrays
            all_correct = np.array(model_metrics_data[model_key]["all_correct"])
            all_conf = np.array(model_metrics_data[model_key]["all_conf"])
            
            # Compute metrics
            accuracy = np.mean(all_correct)
            ece = compute_ece(all_correct, all_conf)
            
            evaluation_metrics[model_key] = {
                "accuracy": float(accuracy),
                "ece": float(ece)
            }
        else:
            print(f"Warning: No data available for model {model_key}")
    
    # Add evaluation metrics to results
    results["evaluation_metrics"] = evaluation_metrics
    
    return results


# ==================== UNIFIED INTERFACE ====================

def compute_uncertainty_confidence(method='bayesian', **kwargs):
    """
    Unified interface for computing uncertainty and confidence
    
    Args:
        method: 'bayesian' or 'permutation'
        **kwargs: Method-specific arguments
        
            For Bayesian method:
                - bayesian_file: Path to Bayesian results file
                - ensemble_file: Path to ensemble results file
                - output_file: Path to save results (optional)
                
            For Permutation method:
                - models_config: Dict mapping model keys to names
                - K: Number of permutations (default: 5)
                - seed: Random seed (default: 42)
                - output_file: Path to save results (optional)
    
    Returns:
        results: Dictionary with calibrated confidence and uncertainty
    """
    print("="*60)
    print(f"Computing Uncertainty and Confidence using {method.upper()} method")
    print("="*60)
    
    if method == 'bayesian':
        # Bayesian method
        bayesian_file = kwargs.get('bayesian_file')
        ensemble_file = kwargs.get('ensemble_file')
        output_file = kwargs.get('output_file', './data/Bayesian_confi_uncertainty.json')
        
        if not bayesian_file or not ensemble_file:
            raise ValueError("Bayesian method requires 'bayesian_file' and 'ensemble_file'")
        
        # Load data
        bayesian_data = load_json_data(bayesian_file)
        ensemble_data = load_json_data(ensemble_file)
        
        # Find common questions
        qwen2_data = bayesian_data  # Adjust based on your data structure
        qwen3_data = ensemble_data
        common_questions = find_common_questions_bayesian(qwen2_data, qwen3_data)
        
        print(f"Found {len(common_questions)} common questions")
        
        # Apply Bayesian calibration
        results = apply_bayesian_calibration(common_questions, bayesian_data, ensemble_data)
        
    elif method == 'permutation':
        # Permutation method
        models_config = kwargs.get('models_config', {
            "qwen2-1.5B": "Qwen/Qwen2-1.5B-Instruct",
            "qwen2-7B": "Qwen/Qwen2-7B-Instruct"
        })
        K = kwargs.get('K', 5)
        seed = kwargs.get('seed', 42)
        output_file = kwargs.get('output_file', './data/Ensemble_confi_uncertainty_perm.json')
        
        # Find common questions across permutations
        common_questions, all_permutation_data = find_common_questions_permutation(
            models_config, K, seed
        )
        
        if len(common_questions) == 0:
            print("No common questions found!")
            return {}
        
        print(f"Found {len(common_questions)} common questions")
        
        # Apply permutation calibration
        results = apply_permutation_calibration(common_questions, all_permutation_data)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bayesian' or 'permutation'")
    
    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")
    
    # Print summary
    print("="*60)
    print("Calibration Results Summary")
    print("="*60)
    print(f"Total questions processed: {len(results) - 1}")  # Subtract evaluation_metrics
    
    if "evaluation_metrics" in results:
        print("\nEvaluation Metrics:")
        for model_key, metrics in results["evaluation_metrics"].items():
            print(f"  {model_key}:")
            print(f"    Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"    ECE: {metrics.get('ece', 0):.4f}")
    
    return results


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example 1: Bayesian method
    # results_bayesian = compute_uncertainty_confidence(
    #     method='bayesian',
    #     bayesian_file='./data/bayesian_results.json',
    #     ensemble_file='./data/ensemble_results.json',
    #     output_file='./data/Bayesian_confi_uncertainty.json'
    # )
    
    # Example 2: Permutation method
    results_permutation = compute_uncertainty_confidence(
        method='permutation',
        models_config={
            "qwen2-1.5B": "Qwen/Qwen2-1.5B-Instruct",
            "qwen2-7B": "Qwen/Qwen2-7B-Instruct"
        },
        K=5,
        seed=42,
        output_file='./data/Ensemble_confi_uncertainty_perm.json'
    )
