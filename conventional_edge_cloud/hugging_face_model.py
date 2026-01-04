"""
LLM Model Interface for Conventional Edge-Cloud Deployment

This module provides inference interface for:
- Edge model: Qwen2-1.5B-Instruct
- Cloud model: Qwen2-7B-Instruct

Supports two confidence evaluation methods:
- use_self_confidence=False: Token-level confidence from logits (default)
- use_self_confidence=True: Model self-evaluation confidence

To switch between Qwen2-1.5B and Qwen2-7B:
    Simply change the model_name parameter:
    - Edge: model_name = "Qwen/Qwen2-1.5B-Instruct"
    - Cloud: model_name = "Qwen/Qwen2-7B-Instruct"
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import numpy as np
import re
from copy import deepcopy


class EnhancedChatCompletion:
    """
    Enhanced chat completion with confidence estimation
    
    Methods:
    - create(): Generate response with confidence scores
    """
    
    @staticmethod
    def create(model, messages, temperature, output_confidence=True, use_self_confidence=False):
        """
        Generate response with confidence estimation
        
        Args:
            model: Model name or path
                   - "Qwen/Qwen2-1.5B-Instruct" for edge
                   - "Qwen/Qwen2-7B-Instruct" for cloud
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Temperature for sampling (0.1-2.0)
            output_confidence: Whether to output confidence scores
            use_self_confidence: Whether to use model self-evaluation for confidence
                                 False: Use token-level logits (default)
                                 True: Ask model to evaluate its own confidence
            
        Returns:
            EnhancedResponse object with:
                - choices[0].message.content: Generated text
                - confidence: Dict with confidence scores
        """
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        
        # Load model with memory configuration
        # Adjust max_memory based on your GPU (e.g., "38GB" for A100)
        model_instance = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "38GB"},
            low_cpu_mem_usage=True
        )
        
        if use_self_confidence:
            return EnhancedChatCompletion._generate_with_self_confidence(
                model_instance, tokenizer, messages, temperature, output_confidence
            )
        else:
            return EnhancedChatCompletion._generate_with_logit_confidence(
                model_instance, tokenizer, messages, temperature, output_confidence
            )
    
    @staticmethod
    def _generate_with_logit_confidence(model_instance, tokenizer, messages, 
                                        temperature, output_confidence):
        """
        Generate response using token-level logit confidence
        
        Confidence is computed from model output logits
        """
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Encode input
        model_inputs = tokenizer([text], return_tensors="pt", padding=False).to(model_instance.device)
        
        # Generate with temperature scaling
        max_new_tokens = 512
        
        with torch.no_grad():
            outputs = model_instance.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,  # Nucleus sampling
                return_dict_in_generate=True,
                output_scores=True  # Return token scores for confidence
            )
        
        # Get generated IDs (excluding prompt)
        generated_ids = outputs.sequences[:, model_inputs.input_ids.shape[-1]:]
        
        # Decode response
        response_text = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Compute token-level confidence
        confidences = None
        if output_confidence:
            logits = outputs.scores  # Logits for each generation step
            probabilities = [torch.softmax(step_logits / temperature, dim=-1) for step_logits in logits]
            
            token_confidences = []
            tokens = []
            
            # Extract confidence for each generated token
            for step, (prob, token_id) in enumerate(zip(probabilities, generated_ids[0])):
                token = tokenizer.decode([token_id])
                token_confidence = prob[0, token_id].item()
                token_confidences.append(token_confidence)
                tokens.append(token)
            
            # Compute overall confidence metrics
            mean_confidence = float(np.mean(token_confidences))
            min_confidence = float(np.min(token_confidences))
            
            # Combine tokens and confidences
            token_conf_dict = dict(zip(tokens, [float(c) for c in token_confidences]))
            
            # Decode full text for option extraction
            full_text = tokenizer.decode(generated_ids[0])
            
            # Extract confidence for each option
            option_confidences = {}
            
            try:
                # Try to parse as JSON
                response_json = json.loads(response_text)
                
                # Process each question
                for q_key, q_data in response_json.items():
                    if "answer" in q_data:
                        answer = q_data["answer"]
                        
                        # Extract option number
                        option_match = re.search(r'option\s+(\d+)', answer, re.IGNORECASE)
                        if option_match:
                            option_num = option_match.group(1)
                            
                            # Find position in generated text
                            option_pos = full_text.find(f"option {option_num}")
                            if option_pos == -1:
                                option_pos = full_text.find(f"option{option_num}")
                            
                            if option_pos != -1:
                                # Find position of the digit
                                digit_pos = option_pos + full_text[option_pos:].find(option_num)
                                
                                # Find corresponding token
                                char_count = 0
                                option_token_idx = None
                                for i, token in enumerate(tokens):
                                    prev_count = char_count
                                    char_count += len(token)
                                    if prev_count <= digit_pos < char_count:
                                        option_token_idx = i
                                        break
                                
                                if option_token_idx is not None:
                                    option_confidences[q_key] = float(token_confidences[option_token_idx])
                                else:
                                    option_confidences[q_key] = mean_confidence
                            else:
                                option_confidences[q_key] = mean_confidence
                        else:
                            option_confidences[q_key] = mean_confidence
            except:
                # If not valid JSON, extract options directly from text
                option_matches = re.finditer(r'option\s+(\d+)', full_text, re.IGNORECASE)
                
                for match in option_matches:
                    option_num = match.group(1)
                    digit_pos = match.start(1)
                    
                    # Find corresponding token
                    char_count = 0
                    option_token_idx = None
                    for i, token in enumerate(tokens):
                        prev_count = char_count
                        char_count += len(token)
                        if prev_count <= digit_pos < char_count:
                            option_token_idx = i
                            break
                    
                    if option_token_idx is not None:
                        option_confidences[f"unknown_question_{len(option_confidences)}"] = float(token_confidences[option_token_idx])
            
            # Build final confidence information
            confidences = {
                "mean_confidence": mean_confidence,
                "min_confidence": min_confidence,
                "option_confidences": option_confidences
            }
        
        # Create response object
        class EnhancedResponse:
            class Message:
                def __init__(self, content, confidence=None):
                    self.content = content
                    self.confidence = confidence
                    
            class Choice:
                def __init__(self, message):
                    self.message = message
            
            def __init__(self, text, confidence=None):
                self.choices = [self.Choice(self.Message(text, confidence))]
                self.confidence = confidence
        
        return EnhancedResponse(response_text, confidences)
    
    @staticmethod
    def _generate_with_self_confidence(model_instance, tokenizer, messages, 
                                       temperature, output_confidence):
        """
        Generate response using model self-evaluation for confidence
        
        Two-stage process:
        1. Generate answer
        2. Ask model to evaluate its confidence in the answer
        """
        # Stage 1: Generate original answer
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt", padding=False).to(model_instance.device)
        max_new_tokens = 512
        
        with torch.no_grad():
            outputs = model_instance.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=False  # Don't need scores for self-evaluation
            )
        
        # Get generated IDs (excluding prompt)
        generated_ids = outputs.sequences[:, model_inputs.input_ids.shape[-1]:]
        
        # Decode response
        response_text = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Stage 2: Self-evaluation for confidence
        confidences = None
        if output_confidence:
            confidences = EnhancedChatCompletion._evaluate_confidence(
                model_instance, tokenizer, messages, response_text, temperature
            )
        
        # Create response object
        class EnhancedResponse:
            class Message:
                def __init__(self, content, confidence=None):
                    self.content = content
                    self.confidence = confidence
                    
            class Choice:
                def __init__(self, message):
                    self.message = message
            
            def __init__(self, text, confidence=None):
                self.choices = [self.Choice(self.Message(text, confidence))]
                self.confidence = confidence
        
        return EnhancedResponse(response_text, confidences)
    
    @staticmethod
    def _evaluate_confidence(model_instance, tokenizer, original_messages, response_text, temperature):
        """
        Ask model to evaluate its confidence in the generated response
        
        Returns confidence scores for each option in the response
        """
        try:
            # Parse questions and answers
            parsed_qa_pairs = EnhancedChatCompletion._parse_questions_and_answers(
                original_messages, response_text
            )
            
            option_confidences = {}
            all_confidences = []
            
            # Evaluate confidence for each Q&A pair
            for qa_pair in parsed_qa_pairs:
                question = qa_pair['question']
                answer = qa_pair['answer']
                question_id = qa_pair['question_id']
                
                # Build self-evaluation prompt
                eval_prompt = f"""Please act as an impartial telecommunications expert and evaluate the quality of the answer provided by an AI assistant to the user question displayed below.
Your evaluation should assess the probability that the given answer to a telecommunications question is correct. 
Return ONLY a number BETWEEN 0 AND 1, where:
- 0 means definitely incorrect
- 1 means definitely correct

Question: {question}
Answer: {answer}

Return your response in the following JSON format ONLY:
{{"probability": 0.X}}"""
                
                eval_messages = [
                    {"role": "user", "content": eval_prompt}
                ]
                
                # Generate evaluation
                eval_text = tokenizer.apply_chat_template(
                    eval_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                eval_inputs = tokenizer([eval_text], return_tensors="pt", padding=False).to(model_instance.device)
                
                with torch.no_grad():
                    eval_outputs = model_instance.generate(
                        **eval_inputs,
                        max_new_tokens=50,
                        temperature=1.0,
                        do_sample=True,
                        top_p=0.9,
                        return_dict_in_generate=True
                    )
                
                eval_generated_ids = eval_outputs.sequences[:, eval_inputs.input_ids.shape[-1]:]
                eval_response = tokenizer.batch_decode(
                    eval_generated_ids, 
                    skip_special_tokens=True
                )[0]
                
                # Extract confidence score from evaluation response
                confidence_score = EnhancedChatCompletion._extract_confidence_score(eval_response)
                
                # Save confidence
                if question_id:
                    option_confidences[question_id] = confidence_score
                all_confidences.append(confidence_score)
            
            # Compute overall statistics
            mean_confidence = float(np.mean(all_confidences)) if all_confidences else 0.5
            min_confidence = float(np.min(all_confidences)) if all_confidences else 0.5
            
            return {
                "mean_confidence": mean_confidence,
                "min_confidence": min_confidence,
                "option_confidences": option_confidences
            }
            
        except Exception as e:
            print(f"Error in self-evaluation: {str(e)}")
            # Return default confidence
            return {
                "mean_confidence": 0.5,
                "min_confidence": 0.5,
                "option_confidences": {}
            }
    
    @staticmethod
    def _parse_questions_and_answers(original_messages, response_text):
        """
        Parse question-answer pairs from original messages and response
        
        Args:
            original_messages: List of original messages
            response_text: Model response text
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        
        try:
            # Extract questions from original messages
            user_content = ""
            for msg in original_messages:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            
            # Extract JSON-formatted questions from user_content
            questions_dict = {}
            if "Here are the questions:" in user_content:
                json_start = user_content.find("{")
                if json_start != -1:
                    json_str = user_content[json_start:]
                    questions_dict = json.loads(json_str)
            
            # Parse response
            try:
                response_dict = json.loads(response_text)
            except:
                # If not valid JSON, return empty list
                return qa_pairs
            
            # Match questions and answers
            for q_id in questions_dict:
                if q_id in response_dict and "answer" in response_dict[q_id]:
                    qa_pairs.append({
                        'question_id': q_id,
                        'question': questions_dict[q_id].get('question', ''),
                        'answer': response_dict[q_id]['answer']
                    })
            
        except Exception as e:
            print(f"Error parsing Q&A pairs: {str(e)}")
        
        return qa_pairs
    
    @staticmethod
    def _extract_confidence_score(eval_response):
        """
        Extract confidence score from evaluation response
        
        Args:
            eval_response: Model's evaluation response
            
        Returns:
            Confidence score (float between 0-1)
        """
        try:
            # Try multiple patterns to extract number
            patterns = [
                r'"probability":\s*([0-9.]+)',  # JSON format
                r'(\d+\.?\d*)',  # Decimal or integer
                r'0\.(\d+)',     # Decimal starting with 0.
                r'1\.0+',        # 1.0 format
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, eval_response)
                if matches:
                    # Take first match
                    score_str = matches[0]
                    score = float(score_str)
                    
                    # Ensure in 0-1 range
                    if score > 1.0:
                        score = score / 10.0  # Might be in base 10
                        if score > 1.0:
                            score = 1.0
                    elif score < 0.0:
                        score = 0.0
                    
                    return score
            
            # If no number found, try keyword matching
            eval_lower = eval_response.lower()
            if any(word in eval_lower for word in ['definitely correct', 'certainly correct', 'absolutely correct']):
                return 1.0
            elif any(word in eval_lower for word in ['definitely incorrect', 'certainly incorrect', 'absolutely incorrect']):
                return 0.0
            elif any(word in eval_lower for word in ['likely correct', 'probably correct']):
                return 0.8
            elif any(word in eval_lower for word in ['likely incorrect', 'probably incorrect']):
                return 0.2
            elif any(word in eval_lower for word in ['uncertain', 'unsure', 'maybe']):
                return 0.5
            
            # Default return medium confidence
            return 0.5
            
        except Exception as e:
            print(f"Error extracting confidence score from '{eval_response}': {str(e)}")
            return 0.5
