"""
Evaluation Tools for Conventional Edge-Cloud Deployment

Supports two confidence modes:
- use_self_confidence=False: Logit-based confidence (default)
- use_self_confidence=True: Model self-evaluation confidence
"""

from copy import deepcopy
import json
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import hugging_face_model as Hugface
from rouge import Rouge
import re


system_prompt = """
Please provide the answers to the following telecommunications related multiple choice questions. The questions will be in a JSON format, the answers must also be in a JSON format as follows:
{
"question 1": {
"question": question,
"answer": "option {answer id}: {answer string}"
},
...
}
"""


def parse_model_output(predicted_answers_str, questions_dict):
    """Parse model output to extract answers"""
    import re
    import json

    # Try to parse as JSON
    try:
        parsed_answers = json.loads(predicted_answers_str)
        return parsed_answers
    except:
        # If direct parsing fails, use regex
        pattern = r'"(question \d+)"\s*:\s*\{[^}]*"question"\s*:\s*"[^"]*",\s*"answer"\s*:\s*"[^"]*"\s*\}'
        matches = re.findall(pattern, predicted_answers_str, re.DOTALL)
        
        parsed_answers = {}
        for match in matches:
            question_pattern = f'"({match})"\s*:\s*{{[^}}]*"question":\s*"[^"]*",\s*"answer":\s*"[^"]*"\s*}}'
            specific_match = re.search(question_pattern, predicted_answers_str, re.DOTALL)
            
            if specific_match:
                try:
                    specific_json = '{' + specific_match.group(0) + '}'
                    parsed_question = json.loads(specific_json)
                    parsed_answers.update(parsed_question)
                except:
                    continue
    
    # Ensure all original questions have answers
    final_answers = {}
    for q in questions_dict:
        if q in parsed_answers:
            final_answers[q] = parsed_answers[q]
        else:
            final_answers[q] = {
                "question": questions_dict[q]["question"]
            }
    
    return final_answers


def check_questions_with_val_output_with_confidence(questions_dict, model, attempts, temperature, use_self_confidence=False):
    """
    Question evaluation function with confidence information
    
    Args:
        questions_dict: Dictionary of questions
        model: Model name (e.g., "Qwen/Qwen2-1.5B-Instruct" or "Qwen/Qwen2-7B-Instruct")
        attempts: Number of attempts
        temperature: Temperature parameter (0.1-2.0)
        use_self_confidence: Whether to use model self-evaluation for confidence
                             False: Use logit-based confidence (default)
                             True: Use model self-evaluation
        
    Returns:
        Tuple (accepted_questions, parsed_predicted_answers, confidences)
    """
    rouge = Rouge()
    
    questions_only = {}
    answers_only = {}
    for q in questions_dict:
        answers_only[q] = {
            "question": questions_dict[q]["question"],
            "answer": questions_dict[q]["answer"]
        }
        
        questions_only[q] = deepcopy(questions_dict[q])
        if "answer" in questions_only[q]:
            questions_only[q].pop("answer")
        
        for key in ['explanation', 'category']:
            if key in questions_only[q]:
                questions_only[q].pop(key)
    
    user_prompt = "Here are the questions: \n "
    user_prompt += json.dumps(questions_only)

    # Call model with confidence mode selection
    response = Hugface.EnhancedChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        output_confidence=True,
        use_self_confidence=use_self_confidence
    )
    
    content = response.choices[0].message.content
    confidences = response.confidence
    
    parsed_predicted_answers = parse_model_output(content, questions_dict)
    
    # Evaluate accuracy using Rouge-L
    correct_answers = {}
    for q in questions_dict:
        if "answer" in parsed_predicted_answers[q]:
            predicted_answer = parsed_predicted_answers[q]["answer"].lower().strip()
            ground_truth_answer = questions_dict[q]["answer"].lower().strip()
            
            try:
                rouge_scores = rouge.get_scores(predicted_answer, ground_truth_answer)[0]
                rouge_l_f1 = rouge_scores['rouge-l']['f']
                
                # Consider correct if Rouge-L F1 > 0.5
                if rouge_l_f1 > 0.5:
                    correct_answers[q] = True
                else:
                    correct_answers[q] = False
            except:
                # If Rouge fails, use exact match
                correct_answers[q] = (predicted_answer == ground_truth_answer)
        else:
            correct_answers[q] = False
    
    accepted_questions = questions_dict
    
    return accepted_questions, parsed_predicted_answers, confidences


def evaluate_model_on_dataset(questions_dict, model_name, temperature=1.0, use_self_confidence=False):
    """
    Evaluate model on full dataset
    
    Args:
        questions_dict: Dictionary of questions
        model_name: Model name
        temperature: Temperature for sampling
        use_self_confidence: Whether to use self-evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    accepted_questions, predicted_answers, confidences = check_questions_with_val_output_with_confidence(
        questions_dict,
        model_name,
        attempts=1,
        temperature=temperature,
        use_self_confidence=use_self_confidence
    )
    
    # Compute accuracy
    correct_count = 0
    total_count = len(questions_dict)
    
    rouge = Rouge()
    for q in questions_dict:
        if "answer" in predicted_answers[q]:
            predicted_answer = predicted_answers[q]["answer"].lower().strip()
            ground_truth_answer = questions_dict[q]["answer"].lower().strip()
            
            try:
                rouge_scores = rouge.get_scores(predicted_answer, ground_truth_answer)[0]
                if rouge_scores['rouge-l']['f'] > 0.5:
                    correct_count += 1
            except:
                if predicted_answer == ground_truth_answer:
                    correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'predicted_answers': predicted_answers,
        'confidences': confidences
    }
