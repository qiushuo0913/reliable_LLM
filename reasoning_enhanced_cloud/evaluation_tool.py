"""
Evaluation Tools for Reasoning-Enhanced Cloud Deployment

This module evaluates Qwen3-4B reasoning model with thinking budget
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

    try:
        parsed_answers = json.loads(predicted_answers_str)
        return parsed_answers
    except:
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
    
    final_answers = {}
    for q in questions_dict:
        if q in parsed_answers:
            final_answers[q] = parsed_answers[q]
        else:
            final_answers[q] = {
                "question": questions_dict[q]["question"]
            }
    
    return final_answers


def check_questions_with_budget_scaling(questions_dict, model, attempts, temperature, 
                                        total_budget=200, answer_thinking_ratio=0.5):
    """
    Question evaluation with thinking budget allocation
    
    Args:
        questions_dict: Dictionary of questions
        model: Model name (e.g., "Qwen/Qwen3-4B")
        attempts: Number of attempts
        temperature: Temperature parameter
        total_budget: Total thinking token budget (default: 200)
        answer_thinking_ratio: Ratio for answer thinking (default: 0.5)
        
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

    # Call model with budget allocation
    response = Hugface.BudgetAllocationChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        total_budget=total_budget,
        answer_thinking_ratio=answer_thinking_ratio,
        output_confidence=True
    )
    
    content = response.choices[0].message.content
    confidences = response.confidence
    
    parsed_predicted_answers = parse_model_output(content, questions_dict)
    
    # Evaluate accuracy
    correct_answers = {}
    for q in questions_dict:
        if "answer" in parsed_predicted_answers[q]:
            predicted_answer = parsed_predicted_answers[q]["answer"].lower().strip()
            ground_truth_answer = questions_dict[q]["answer"].lower().strip()
            
            try:
                rouge_scores = rouge.get_scores(predicted_answer, ground_truth_answer)[0]
                rouge_l_f1 = rouge_scores['rouge-l']['f']
                
                if rouge_l_f1 > 0.5:
                    correct_answers[q] = True
                else:
                    correct_answers[q] = False
            except:
                correct_answers[q] = (predicted_answer == ground_truth_answer)
        else:
            correct_answers[q] = False
    
    accepted_questions = questions_dict
    
    return accepted_questions, parsed_predicted_answers, confidences


def evaluate_model_on_dataset(questions_dict, model_name, temperature=1.0, 
                              total_budget=200, answer_thinking_ratio=0.5):
    """
    Evaluate reasoning model on full dataset
    
    Args:
        questions_dict: Dictionary of questions
        model_name: Model name
        temperature: Temperature for sampling
        total_budget: Thinking budget
        answer_thinking_ratio: Budget allocation ratio
        
    Returns:
        Dictionary with evaluation results
    """
    accepted_questions, predicted_answers, confidences = check_questions_with_budget_scaling(
        questions_dict,
        model_name,
        attempts=1,
        temperature=temperature,
        total_budget=total_budget,
        answer_thinking_ratio=answer_thinking_ratio
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
