"""
Run Experiment for Conventional Edge-Cloud Deployment

This script runs experiments for the conventional edge-cloud deployment with:
- Edge model: Qwen2-1.5B-Instruct
- Cloud model: Qwen2-7B-Instruct (can be switched to Qwen2-1.5B)

Supports two confidence evaluation modes:
- use_self_confidence=False: Logit-based confidence (default)
- use_self_confidence=True: Model self-evaluation confidence

To switch between self and non-self confidence:
    Set USE_SELF_CONFIDENCE = True/False below

To change save directory based on confidence mode:
    Results saved to './results_self/' or './results_logit/'
    
To switch models:
    Change model_name to "Qwen/Qwen2-1.5B-Instruct" or "Qwen/Qwen2-7B-Instruct"
"""

from evaluation_tool import *
import os 
import json
import numpy as np
import pandas as pd
import random
from copy import deepcopy


# ============= CONFIGURATION =============
# Set to True to use model self-evaluation for confidence
# Set to False to use logit-based confidence
USE_SELF_CONFIDENCE = False  # Change this to switch modes

# Set save directory based on confidence mode
SAVE_DIR = './results_self/' if USE_SELF_CONFIDENCE else './results_logit/'
# =========================================


def compute_ece(confidences, accuracies, n_bins=10):
    """
    Compute Expected Calibration Error (ECE)
    
    Args:
        confidences: List of model confidence scores
        accuracies: List of corresponding accuracies (0 or 1)
        n_bins: Number of bins
    
    Returns:
        ece: Expected calibration error value
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Create confidence bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0

    # Compute accuracy and average confidence for each bin
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_size = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if i == n_bins - 1:  # Last bin includes upper boundary
            bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])

        if np.sum(bin_mask) > 0:
            bin_acc[i] = np.mean(accuracies[bin_mask])
            bin_conf[i] = np.mean(confidences[bin_mask])
            bin_size[i] = np.sum(bin_mask)
    
    # Compute Expected Calibration Error (ECE)
    ece = np.sum(bin_size / len(confidences) * np.abs(bin_acc - bin_conf))
    
    return ece


def permute_options(question_data, permutation_id):
    """
    Permute question options
    
    Args:
        question_data: Original question data
        permutation_id: Permutation ID (0 means original order)
    
    Returns:
        permuted_question: Permuted question data
    """
    if permutation_id == 0:
        return deepcopy(question_data)
    
    # Set seed for reproducibility
    random.seed(42 + permutation_id)
    
    # Deep copy to avoid modifying original data
    permuted_question = deepcopy(question_data)
    
    # Extract all options
    options = []
    option_keys = []
    for key in sorted(question_data.keys()):
        if key.startswith('option '):
            options.append(question_data[key])
            option_keys.append(key)
    
    if len(options) == 0:
        return permuted_question
    
    # Find original correct answer option
    original_answer = question_data.get('answer', '')
    original_correct_option = None
    for i, key in enumerate(option_keys):
        if f"{key}: {options[i]}" in original_answer:
            original_correct_option = options[i]
            break
    
    # Shuffle options
    shuffled_options = options.copy()
    random.shuffle(shuffled_options)
    
    # Update permuted options
    for i, key in enumerate(option_keys):
        permuted_question[key] = shuffled_options[i]
    
    # Update correct answer
    if original_correct_option:
        for i, key in enumerate(option_keys):
            if shuffled_options[i] == original_correct_option:
                permuted_question['answer'] = f"{key}: {original_correct_option}"
                break
    
    return permuted_question


def evaluate_with_permutation(model_name, permutation_id, subset_questions, max_attempts=10, n_questions=4, save_dir="permutation_results"):
    """
    Evaluate subset questions with specified permutation ID
    
    Args:
        model_name: Model name
        permutation_id: Permutation ID
        subset_questions: Question subset to evaluate
        max_attempts: Maximum number of attempts
        n_questions: Number of questions per batch
        save_dir: Directory to save results
        
    Returns:
        accuracy: Overall accuracy
        ece: Expected calibration error
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # File path
    save_path = os.path.join(save_dir, f"{model_name.replace('/', '_')}_permutation_{permutation_id}.json")
    
    # Initialize results
    results = {}
    categories = []
    correct = []
    confidences = []
    
    no_answer_categories = {}
    accuracy_all = {}
    count_all = {}
    
    print(f"Start using Permutation {permutation_id} evaluating {model_name}")

    # Apply permutation to all questions
    permuted_questions = {}
    for q_name, q_data in subset_questions.items():
        permuted_questions[q_name] = permute_options(q_data, permutation_id)

    # Convert dict to list for batch processing
    questions_list = list(permuted_questions.items())
    total_questions = len(questions_list)
    
    # Process in batches
    for batch_start in range(0, total_questions, n_questions):
        batch_end = min(batch_start + n_questions, total_questions)
        batch_items = questions_list[batch_start:batch_end]
        batch_questions = dict(batch_items)
        
        print(f"Processing batch {batch_start//n_questions + 1}/{(total_questions + n_questions - 1)//n_questions}")
        
        attempts = 0
        while attempts < max_attempts and batch_questions:
            accepted_questions, parsed_predicted_answers, confidence_info = check_questions_with_val_output_with_confidence(
                batch_questions, model_name, attempts, temperature=1.0, use_self_confidence=USE_SELF_CONFIDENCE
            )
            
            attempts += 1
            
            # Process results
            questions_to_process = list(batch_questions.keys())
            for q in questions_to_process:
                category = batch_questions[q]['category']
                try:
                    if "answer" in parsed_predicted_answers[q]:
                        # Save result
                        results[q] = deepcopy(batch_questions[q])
                        results[q]["tested answer"] = parsed_predicted_answers[q]["answer"]
                        
                        # Get confidence score
                        if confidence_info and "option_confidences" in confidence_info:
                            if q in confidence_info["option_confidences"]:
                                conf_score = confidence_info["option_confidences"][q]
                            else:
                                conf_score = confidence_info.get("mean_confidence", 0.5)
                        else:
                            conf_score = 0.5
                        
                        results[q]["confidence"] = conf_score
                        confidences.append(conf_score)
                        
                        # Check correctness using Rouge
                        predicted_answer = parsed_predicted_answers[q]["answer"].lower().strip()
                        ground_truth_answer = batch_questions[q]["answer"].lower().strip()
                        
                        try:
                            rouge = Rouge()
                            rouge_scores = rouge.get_scores(predicted_answer, ground_truth_answer)[0]
                            is_correct = rouge_scores['rouge-l']['f'] > 0.5
                        except:
                            is_correct = (predicted_answer == ground_truth_answer)
                        
                        results[q]["correct"] = is_correct
                        correct.append(1 if is_correct else 0)
                        categories.append(category)
                        
                        # Update category statistics
                        if category not in accuracy_all:
                            accuracy_all[category] = 0
                            count_all[category] = 0
                        
                        if is_correct:
                            accuracy_all[category] += 1
                        count_all[category] += 1
                        
                        # Remove from batch
                        batch_questions.pop(q)
                    else:
                        # No answer provided
                        if category not in no_answer_categories:
                            no_answer_categories[category] = 0
                        no_answer_categories[category] += 1
                        
                except Exception as e:
                    print(f"Error processing {q}: {str(e)}")
            
            if batch_questions:
                print(f"Permutation: {permutation_id}, need more batches")

            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    

    # Compute overall accuracy and calibration error
    if len(correct) > 0 and len(confidences) > 0:
        accuracy = sum(correct) / len(correct)
        ece = compute_ece(confidences, correct)
        print(f"Permutation: {permutation_id}, Accuracy: {accuracy:.4f}, ECE: {ece:.4f}")
    else:
        accuracy = 0.0
        ece = 1.0
        print(f"Permutation: {permutation_id}, invalid results")
    
    return accuracy, ece


# Fixed random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Model setting
# To switch models:
# - For edge: model_name = "Qwen/Qwen2-1.5B-Instruct"
# - For cloud: model_name = "Qwen/Qwen2-7B-Instruct"
model_name = "Qwen/Qwen2-1.5B-Instruct"

# Number of permutations
K = 5  # Can adjust based on needs

# Read sampled question subset
with open("sampled_teleQnA_1k.txt", encoding="utf-8") as f:
    loaded_json = f.read()
subset_questions = json.loads(loaded_json)

print(f"Sampled {len(subset_questions)} questions for {model_name} option permutation experiment")
print(f"Using {'self-evaluation' if USE_SELF_CONFIDENCE else 'logit-based'} confidence")

# Create results directory based on confidence mode
os.makedirs(SAVE_DIR, exist_ok=True)

# Evaluate model with different permutations
results_summary = []
for perm_id in range(K):
    # For running additional experiments (5-9)
    perm_id = perm_id + 5
    accuracy, ece = evaluate_with_permutation(
        model_name, 
        permutation_id=perm_id, 
        subset_questions=subset_questions,
        save_dir=SAVE_DIR
    )
    
    results_summary.append({
        'permutation_id': perm_id,
        'accuracy': accuracy,
        'ece': ece
    })

# Save summary results
with open(os.path.join(SAVE_DIR, "permutation_results_summary.json"), 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"Completed {K} permutations. Results saved in {SAVE_DIR}")
