"""
LLM Model Interface for Reasoning-Enhanced Cloud Deployment

This module provides inference interface for:
- Edge model: Qwen2-1.5B-Instruct (same as conventional)
- Cloud model: Qwen3-4B with thinking budget scaling

The reasoning model (Qwen3-4B) supports:
- Thinking budget allocation between answer and confidence evaluation
- Extended reasoning capabilities for complex questions

To adjust thinking budget:
    total_budget: Total thinking tokens (default: 200)
    answer_thinking_ratio: Ratio for answer vs confidence (default: 0.5)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import numpy as np
import re
from copy import deepcopy


class BudgetAllocationChatCompletion:
    """
    Chat completion with thinking budget allocation for reasoning models
    
    Budget allocation:
    - answer_thinking_ratio=1.0: All budget for answer thinking
    - answer_thinking_ratio=0.5: Half for answer, half for confidence
    - answer_thinking_ratio=0.0: All budget for confidence thinking
    """
    
    @staticmethod
    def create(model, messages, temperature, total_budget=200, answer_thinking_ratio=0.5, 
               thinking_tokens_per_wait=50, output_confidence=True):
        """
        Generate response with fixed thinking budget allocation
        
        Args:
            model: Model name (e.g., "Qwen/Qwen3-4B" for reasoning cloud)
            messages: List of message dicts
            temperature: Temperature for sampling
            total_budget: Total thinking token budget (default: 200)
            answer_thinking_ratio: Ratio of budget for answer thinking (0.0-1.0)
                - 1.0: All budget for answer, confidence without thinking
                - 0.5: Half for answer, half for confidence
                - 0.0: All budget for confidence, answer without thinking
            thinking_tokens_per_wait: Tokens per thinking step
            output_confidence: Whether to output confidence
            
        Returns:
            Enhanced response with thinking processes
        """
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Calculate budget allocation
        answer_budget = int(total_budget * answer_thinking_ratio)
        confidence_budget = total_budget - answer_budget
        
        print(f"Budget allocation: Total={total_budget}, Answer={answer_budget}, Confidence={confidence_budget}")
        
        # Use budget allocation generation method
        return BudgetAllocationChatCompletion._generate_with_budget_allocation(
            model_obj, tokenizer, messages, temperature, output_confidence,
            answer_budget, confidence_budget, thinking_tokens_per_wait
        )
    
    @staticmethod
    def _generate_with_budget_allocation(model_obj, tokenizer, messages, temperature, 
                                       output_confidence, answer_budget, confidence_budget,
                                       thinking_tokens_per_wait):
        """Generate response using budget allocation"""
        
        # Stage 1: Generate answer
        if answer_budget > 0:
            # With thinking budget, use thinking mode
            print(f"Stage 1: Generating answer with thinking (budget: {answer_budget} tokens)")
            answer_response = BudgetAllocationChatCompletion._generate_answer_with_thinking(
                model_obj, tokenizer, messages, temperature, answer_budget
            )
        else:
            # No thinking budget, direct answer
            print(f"Stage 1: Generating answer without thinking")
            answer_response = BudgetAllocationChatCompletion._generate_answer_without_thinking(
                model_obj, tokenizer, messages, temperature
            )
        
        # Extract answer content and thinking text
        answer_content, answer_thinking_text = BudgetAllocationChatCompletion._extract_answer_and_thinking(answer_response)
        
        # Stage 2: Evaluate confidence
        if output_confidence:
            if confidence_budget > 0:
                # With thinking budget, use thinking mode for confidence
                print(f"Stage 2: Evaluating confidence with thinking (budget: {confidence_budget} tokens)")
                confidence_response = BudgetAllocationChatCompletion._evaluate_confidence_with_thinking(
                    model_obj, tokenizer, messages, answer_content, temperature, confidence_budget
                )
                # Extract confidence info and thinking text
                confidence_info, confidence_thinking_text = BudgetAllocationChatCompletion._extract_confidence_and_thinking(confidence_response)
            else:
                # No thinking budget, direct confidence evaluation
                print(f"Stage 2: Evaluating confidence without thinking")
                confidence_response = BudgetAllocationChatCompletion._evaluate_confidence_without_thinking(
                    model_obj, tokenizer, messages, answer_content, temperature
                )
                confidence_info = BudgetAllocationChatCompletion._extract_confidence_from_response(confidence_response)
                confidence_thinking_text = None
        else:
            confidence_info = {}
            confidence_thinking_text = None
        
        # Create response with all thinking texts
        return BudgetAllocationChatCompletion._create_enhanced_response(
            answer_content, confidence_info, answer_thinking_text, confidence_thinking_text
        )
    
    @staticmethod
    def _generate_answer_with_thinking(model_obj, tokenizer, messages, temperature, thinking_budget):
        """Generate answer with thinking budget"""
        # Add system message for thinking
        thinking_messages = [
            {"role": "system", "content": "Please think step by step before answering. Take your time to reason through the problem."}
        ] + messages
        
        text = tokenizer.apply_chat_template(
            thinking_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt", padding=False).to(model_obj.device)
        
        # Generate with thinking budget
        max_tokens = min(512, thinking_budget + 100)  # Answer + thinking
        
        with torch.no_grad():
            outputs = model_obj.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True
            )
        
        generated_ids = outputs.sequences[:, model_inputs.input_ids.shape[-1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    @staticmethod
    def _generate_answer_without_thinking(model_obj, tokenizer, messages, temperature):
        """Generate answer without thinking"""
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt", padding=False).to(model_obj.device)
        
        with torch.no_grad():
            outputs = model_obj.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True
            )
        
        generated_ids = outputs.sequences[:, model_inputs.input_ids.shape[-1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    @staticmethod
    def _evaluate_confidence_with_thinking(model_obj, tokenizer, original_messages, answer_content, temperature, thinking_budget):
        """Evaluate confidence with thinking budget"""
        # Parse answer to extract Q&A pairs
        qa_pairs = BudgetAllocationChatCompletion._parse_qa_from_messages_and_answer(
            original_messages, answer_content
        )
        
        option_confidences = {}
        all_thinking_texts = []
        
        for qa_pair in qa_pairs:
            question = qa_pair['question']
            answer = qa_pair['answer']
            question_id = qa_pair['question_id']
            
            # Build thinking confidence evaluation prompt
            eval_prompt = f"""Think step by step to assess the correctness of this answer.

Question: {question}
Answer: {answer}

After your reasoning, provide your confidence as a number between 0 and 1.
Format: {{"confidence score": [number]}}"""
            
            eval_messages = [
                {"role": "user", "content": eval_prompt}
            ]
            
            eval_text = tokenizer.apply_chat_template(
                eval_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            eval_inputs = tokenizer([eval_text], return_tensors="pt", padding=False).to(model_obj.device)
            
            max_tokens = min(512, thinking_budget + 100)
            
            with torch.no_grad():
                eval_outputs = model_obj.generate(
                    **eval_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    return_dict_in_generate=True
                )
            
            eval_generated_ids = eval_outputs.sequences[:, eval_inputs.input_ids.shape[-1]:]
            eval_response = tokenizer.batch_decode(eval_generated_ids, skip_special_tokens=True)[0]
            
            all_thinking_texts.append(eval_response)
            
            # Extract confidence score
            confidence_score = BudgetAllocationChatCompletion._extract_confidence_score(eval_response)
            
            if question_id:
                option_confidences[question_id] = confidence_score
        
        # Compute statistics
        all_confidences = list(option_confidences.values())
        mean_confidence = float(np.mean(all_confidences)) if all_confidences else 0.5
        min_confidence = float(np.min(all_confidences)) if all_confidences else 0.5
        
        confidence_info = {
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "option_confidences": option_confidences
        }
        
        return confidence_info, "\n\n".join(all_thinking_texts)
    
    @staticmethod
    def _evaluate_confidence_without_thinking(model_obj, tokenizer, original_messages, answer_content, temperature):
        """Evaluate confidence without thinking"""
        qa_pairs = BudgetAllocationChatCompletion._parse_qa_from_messages_and_answer(
            original_messages, answer_content
        )
        
        option_confidences = {}
        
        for qa_pair in qa_pairs:
            question = qa_pair['question']
            answer = qa_pair['answer']
            question_id = qa_pair['question_id']
            
            eval_prompt = f"""Assess the correctness of this answer. Return ONLY a confidence score between 0 and 1.

Question: {question}
Answer: {answer}

Format: {{"confidence score": [number]}}"""
            
            eval_messages = [{"role": "user", "content": eval_prompt}]
            
            eval_text = tokenizer.apply_chat_template(
                eval_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            eval_inputs = tokenizer([eval_text], return_tensors="pt", padding=False).to(model_obj.device)
            
            with torch.no_grad():
                eval_outputs = model_obj.generate(
                    **eval_inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=False,
                    return_dict_in_generate=True
                )
            
            eval_generated_ids = eval_outputs.sequences[:, eval_inputs.input_ids.shape[-1]:]
            eval_response = tokenizer.batch_decode(eval_generated_ids, skip_special_tokens=True)[0]
            
            confidence_score = BudgetAllocationChatCompletion._extract_confidence_score(eval_response)
            
            if question_id:
                option_confidences[question_id] = confidence_score
        
        all_confidences = list(option_confidences.values())
        mean_confidence = float(np.mean(all_confidences)) if all_confidences else 0.5
        min_confidence = float(np.min(all_confidences)) if all_confidences else 0.5
        
        return {
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "option_confidences": option_confidences
        }
    
    @staticmethod
    def _extract_answer_and_thinking(response):
        """Extract answer content and thinking text"""
        # Simple extraction - in practice, may need more sophisticated parsing
        return response, None
    
    @staticmethod
    def _extract_confidence_and_thinking(response_tuple):
        """Extract confidence info and thinking text from tuple"""
        if isinstance(response_tuple, tuple):
            return response_tuple
        else:
            return response_tuple, None
    
    @staticmethod
    def _extract_confidence_from_response(response):
        """Extract confidence from response dict"""
        return response
    
    @staticmethod
    def _extract_confidence_score(response_text):
        """Extract confidence score from response text"""
        try:
            # Try JSON format
            match = re.search(r'"confidence score":\s*([0-9.]+)', response_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            
            # Try simple decimal
            numbers = re.findall(r'0\.\d+|1\.0', response_text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            
            return 0.5
        except:
            return 0.5
    
    @staticmethod
    def _parse_qa_from_messages_and_answer(messages, answer_content):
        """Parse Q&A pairs from messages and answer"""
        qa_pairs = []
        
        try:
            # Extract questions from user messages
            user_content = ""
            for msg in messages:
                if msg["role"] == "user":
                    user_content = msg["content"]
                    break
            
            # Extract JSON questions
            questions_dict = {}
            if "Here are the questions:" in user_content:
                json_start = user_content.find("{")
                if json_start != -1:
                    json_str = user_content[json_start:]
                    questions_dict = json.loads(json_str)
            
            # Parse answer
            try:
                answer_dict = json.loads(answer_content)
            except:
                return qa_pairs
            
            # Match questions and answers
            for q_id in questions_dict:
                if q_id in answer_dict and "answer" in answer_dict[q_id]:
                    qa_pairs.append({
                        'question_id': q_id,
                        'question': questions_dict[q_id].get('question', ''),
                        'answer': answer_dict[q_id]['answer']
                    })
        except Exception as e:
            print(f"Error parsing Q&A: {str(e)}")
        
        return qa_pairs
    
    @staticmethod
    def _create_enhanced_response(answer_content, confidence_info, answer_thinking, confidence_thinking):
        """Create enhanced response object"""
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
                self.answer_thinking = answer_thinking
                self.confidence_thinking = confidence_thinking
        
        return EnhancedResponse(answer_content, confidence_info)
