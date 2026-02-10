#!/usr/bin/env python3
"""
Script to check whether prompt-response pairs form continuous sentences.

This script:
1. Takes the first N entries from a dataset
2. Creates batches using the same method as test_data_loading.py
3. Checks continuity using one of three methods:
   - "prompt": Asks an instruction-tuned LLM if the pair is continuous (Yes/No)
   - "perplexity": Measures perplexity of response conditioned on prompt (for base models)
   - "reward_model": Uses an off-the-shelf reward model to score the prompt-response pair
4. Saves inference results (prompt, response, answer) to a JSON file

Usage:
    # Using instruction-tuned model (prompt method)
    python check_continuity.py \
        --train_files /path/to/train.parquet \
        --model_path /path/to/base_model \
        --inference_model_path /path/to/instruct_model \
        --method prompt \
        --output_json continuity_results.json
    
    # Using base model (perplexity method)
    python check_continuity.py \
        --train_files /path/to/train.parquet \
        --model_path /path/to/base_model \
        --method perplexity \
        --perplexity_threshold 50.0 \
        --output_json continuity_results.json
    
    # Using reward model
    python check_continuity.py \
        --train_files /path/to/train.parquet \
        --model_path /path/to/base_model \
        --reward_model_path /path/to/reward_model \
        --method reward_model \
        --reward_threshold 0.0 \
        --output_json continuity_results.json
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm

# Add verl to path
verl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, verl_root)

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import get_generation_config
from verl.utils.fs import copy_to_local


def get_tokenizer(model_path: str, trust_remote_code: bool = False):
    """Load tokenizer from model path."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def combine_prompt_target_batch(gen_batch: DataProto, tokenizer, eos_token_id, pad_token_id) -> DataProto:
    """
    Combine prompts with target responses (for freeze_generation mode).
    Creates all-to-all combinations of prompts and responses.
    """
    if pad_token_id is None:
        pad_token_id = eos_token_id if not isinstance(eos_token_id, list) else eos_token_id[0]
    
    prompt_ids = gen_batch.batch["input_ids"]
    prompt_mask = gen_batch.batch["attention_mask"]
    prompt_pos = gen_batch.batch["position_ids"]
    resp_ids = gen_batch.batch["chosen_input_ids"]
    resp_mask = gen_batch.batch["chosen_attention_mask"]
    resp_pos = gen_batch.batch["chosen_position_ids"]

    if resp_ids.dim() == 1:
        resp_ids = resp_ids.unsqueeze(0)
        resp_mask = resp_mask.unsqueeze(0)
        resp_pos = resp_pos.unsqueeze(0)

    batch_size = prompt_ids.size(0)
    n_responses = batch_size

    # Create all-to-all combinations
    prompt_ids = prompt_ids.repeat_interleave(n_responses, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(n_responses, dim=0)
    prompt_pos = prompt_pos.repeat_interleave(n_responses, dim=0)

    resp_ids = resp_ids.repeat(batch_size, 1)
    resp_mask = resp_mask.repeat(batch_size, 1)
    resp_pos = resp_pos.repeat(batch_size, 1)
    batch_size = batch_size * batch_size
    
    # Convert response from LEFT-padded to RIGHT-padded
    response_length = resp_ids.size(1)
    resp_ids_right_pad = torch.full_like(resp_ids, pad_token_id)
    resp_mask_right_pad = torch.zeros_like(resp_mask)
    
    for i in range(batch_size):
        attended_mask = resp_mask[i] == 1
        content_tokens = resp_ids[i][attended_mask]
        content_length = content_tokens.size(0)
        resp_ids_right_pad[i, :content_length] = content_tokens
        resp_mask_right_pad[i, :content_length] = 1
    
    resp_ids = resp_ids_right_pad
    resp_mask = resp_mask_right_pad
    
    seq = torch.cat([prompt_ids, resp_ids], dim=-1)

    delta_position_id = torch.arange(1, response_length + 1, device=resp_pos.device)
    delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

    response_position_ids = prompt_pos[:, -1:] + delta_position_id
    position_ids = torch.cat([prompt_pos, response_position_ids], dim=-1)
    
    attention_mask = torch.cat((prompt_mask, resp_mask), dim=-1)

    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": resp_ids,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    return DataProto(batch=batch)


class ContinuityChecker:
    """Checks if prompt-response pairs form continuous sentences using an LLM."""
    
    def __init__(self, model_path: str, tokenizer_path: str = None, 
                 device: str = "cuda", trust_remote_code: bool = False,
                 torch_dtype: str = "auto"):
        """
        Initialize the continuity checker with a language model.
        
        Args:
            model_path: Path to the inference model
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to run inference on
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model weights
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device
        tokenizer_path = tokenizer_path or model_path
        
        print(f"Loading inference model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Parse torch dtype
        if torch_dtype == "auto":
            dtype = "auto"
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device if device != "cuda" else "auto",
        )
        self.model.eval()
        print(f"Model loaded. Device: {self.model.device}")
    
    def create_continuity_prompt(self, text_a: str, text_b: str) -> str:
        """
        Create a prompt to ask the model if two texts form a continuous sentence.
        
        Args:
            text_a: First text segment (prompt)
            text_b: Second text segment (response)
            
        Returns:
            Formatted prompt for the model
        """
        prompt = f"""I will show you two text segments. Please determine if the second segment is a natural and coherent continuation of the first segment.

Text A:
{text_a}

Text B:
{text_b}

Is Text B a natural continuation of Text A? Answer with just "Yes" or "No"."""
        
        return prompt
    
    def check_continuity(self, text_a: str, text_b: str, max_new_tokens: int = 10) -> Dict[str, Any]:
        """
        Check if text_b is a natural continuation of text_a.
        
        Args:
            text_a: First text segment
            text_b: Second text segment
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'answer' (Yes/No), 'raw_response', and 'is_continuous' (bool)
        """
        prompt = self.create_continuity_prompt(text_a, text_b)
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        raw_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Parse answer
        answer_lower = raw_response.lower()
        if answer_lower.startswith("yes"):
            answer = "Yes"
            is_continuous = True
        elif answer_lower.startswith("no"):
            answer = "No"
            is_continuous = False
        else:
            # Try to find yes/no in the response
            if "yes" in answer_lower and "no" not in answer_lower:
                answer = "Yes"
                is_continuous = True
            elif "no" in answer_lower and "yes" not in answer_lower:
                answer = "No"
                is_continuous = False
            else:
                answer = "Unknown"
                is_continuous = None
        
        return {
            "answer": answer,
            "raw_response": raw_response,
            "is_continuous": is_continuous
        }
    
    def check_batch(self, texts_a: List[str], texts_b: List[str], 
                    show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Check continuity for a batch of text pairs.
        
        Args:
            texts_a: List of first text segments
            texts_b: List of second text segments
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dictionaries
        """
        results = []
        iterator = zip(texts_a, texts_b)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Checking continuity")
        
        for text_a, text_b in iterator:
            result = self.check_continuity(text_a, text_b)
            results.append(result)
        
        return results
    
    def compute_perplexity(self, text_a: str, text_b: str, 
                           max_length: int = 4096) -> Dict[str, Any]:
        """
        Compute perplexity of text_b conditioned on text_a.
        
        Lower perplexity indicates the model finds text_b to be a more
        natural continuation of text_a.
        
        Args:
            text_a: First text segment (prompt/context)
            text_b: Second text segment (response/continuation)
            max_length: Maximum sequence length
            
        Returns:
            Dict with perplexity metrics:
            - 'perplexity': Perplexity of text_b given text_a
            - 'loss': Cross-entropy loss
            - 'num_tokens': Number of tokens in text_b
            - 'total_perplexity': Perplexity of full sequence (for reference)
        """
        # Combine texts
        full_text = text_a + text_b
        
        # Tokenize the full sequence
        full_encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=False,
        ).to(self.model.device)
        
        # Tokenize just text_a to find the boundary
        prompt_encoding = self.tokenizer(
            text_a,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        full_input_ids = full_encoding['input_ids']
        seq_length = full_input_ids.shape[1]
        
        # If response is empty or truncated away, return high perplexity
        if seq_length <= prompt_length:
            return {
                'perplexity': float('inf'),
                'loss': float('inf'),
                'num_tokens': 0,
                'total_perplexity': float('inf'),
                'response_tokens': 0,
            }
        
        # Create labels: -100 for prompt tokens (not included in loss), actual ids for response
        labels = full_input_ids.clone()
        labels[:, :prompt_length] = -100  # Mask prompt tokens
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_encoding.get('attention_mask'),
                labels=labels,
            )
            
            # Get loss (already computed by the model for response tokens only)
            response_loss = outputs.loss.item()
            
            # Compute perplexity
            response_perplexity = torch.exp(outputs.loss).item()
            
            # Also compute total perplexity for reference
            total_labels = full_input_ids.clone()
            total_outputs = self.model(
                input_ids=full_input_ids,
                attention_mask=full_encoding.get('attention_mask'),
                labels=total_labels,
            )
            total_perplexity = torch.exp(total_outputs.loss).item()
        
        num_response_tokens = seq_length - prompt_length
        
        return {
            'perplexity': response_perplexity,
            'loss': response_loss,
            'num_tokens': num_response_tokens,
            'total_perplexity': total_perplexity,
            'response_tokens': num_response_tokens,
        }
    
    def check_continuity_perplexity(self, text_a: str, text_b: str, 
                                     threshold: float = 50.0) -> Dict[str, Any]:
        """
        Check if text_b is a natural continuation of text_a using perplexity.
        
        Args:
            text_a: First text segment
            text_b: Second text segment
            threshold: Perplexity threshold below which text is considered continuous
            
        Returns:
            Dict with 'answer' (Yes/No), 'perplexity', 'is_continuous' (bool)
        """
        metrics = self.compute_perplexity(text_a, text_b)
        perplexity = metrics['perplexity']
        
        # Lower perplexity = more natural continuation
        if perplexity < threshold:
            answer = "Yes"
            is_continuous = True
        else:
            answer = "No"
            is_continuous = False
        
        return {
            "answer": answer,
            "perplexity": perplexity,
            "loss": metrics['loss'],
            "num_tokens": metrics['num_tokens'],
            "total_perplexity": metrics['total_perplexity'],
            "is_continuous": is_continuous,
            "threshold": threshold,
        }
    
    def compute_perplexity_batch(self, texts_a: List[str], texts_b: List[str],
                                  threshold: float = 50.0,
                                  show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Compute perplexity-based continuity for a batch of text pairs.
        
        Args:
            texts_a: List of first text segments
            texts_b: List of second text segments
            threshold: Perplexity threshold for continuity
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dictionaries
        """
        results = []
        iterator = zip(texts_a, texts_b)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Computing perplexity")
        
        for text_a, text_b in iterator:
            result = self.check_continuity_perplexity(text_a, text_b, threshold)
            results.append(result)
        
        return results


class RewardModelScorer:
    """Uses an off-the-shelf reward model to score prompt-response pairs."""
    
    def __init__(self, model_path: str, tokenizer_path: str = None,
                 device: str = "cuda", trust_remote_code: bool = False,
                 torch_dtype: str = "auto", model_type: str = "auto"):
        """
        Initialize the reward model scorer.
        
        Args:
            model_path: Path to the reward model
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to run inference on
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model weights
            model_type: Type of reward model: 'auto', 'sequence_classification', 
                        'causal_lm_value_head', or 'custom'
        """
        from transformers import AutoTokenizer
        
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        tokenizer_path = tokenizer_path or model_path
        
        print(f"Loading reward model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Parse torch dtype
        if torch_dtype == "auto":
            dtype = "auto"
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "auto"
        
        # Try to load as different model types
        self.model = None
        self.value_head = None
        
        if model_type == "auto" or model_type == "sequence_classification":
            try:
                from transformers import AutoModelForSequenceClassification
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=dtype,
                    device_map=device if device != "cuda" else "auto",
                )
                self.model_type = "sequence_classification"
                print(f"Loaded as sequence classification model")
            except Exception as e:
                if model_type == "sequence_classification":
                    raise e
                print(f"Could not load as sequence classification: {e}")
        
        if self.model is None and (model_type == "auto" or model_type == "causal_lm_value_head"):
            try:
                # Try loading as a causal LM with value head (common for RLHF reward models)
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=dtype,
                    device_map=device if device != "cuda" else "auto",
                )
                # Check if model has a value head or score attribute
                if hasattr(self.model, 'v_head') or hasattr(self.model, 'value_head'):
                    self.model_type = "causal_lm_value_head"
                    print(f"Loaded as causal LM with value head")
                elif hasattr(self.model, 'score'):
                    self.model_type = "causal_lm_score"
                    print(f"Loaded as causal LM with score head")
                else:
                    # Fall back to using last hidden state
                    self.model_type = "causal_lm_hidden"
                    print(f"Loaded as causal LM (will use last token logits for scoring)")
            except Exception as e:
                if model_type == "causal_lm_value_head":
                    raise e
                print(f"Could not load as causal LM: {e}")
        
        if self.model is None:
            raise ValueError(f"Could not load reward model from {model_path}")
        
        self.model.eval()
        print(f"Reward model loaded. Type: {self.model_type}, Device: {next(self.model.parameters()).device}")
    
    def compute_reward(self, text_a: str, text_b: str, max_length: int = 4096) -> Dict[str, Any]:
        """
        Compute reward score for a prompt-response pair.
        
        Args:
            text_a: Prompt text
            text_b: Response text
            max_length: Maximum sequence length
            
        Returns:
            Dict with 'reward_score' and other metrics
        """
        # Combine prompt and response
        # Some reward models expect specific formatting
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            # Use chat template if available
            messages = [
                {"role": "user", "content": text_a},
                {"role": "assistant", "content": text_b}
            ]
            try:
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception:
                # Fallback to simple concatenation
                full_text = text_a + text_b
        else:
            full_text = text_a + text_b
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            if self.model_type == "sequence_classification":
                outputs = self.model(**inputs)
                # Get the score (could be logits for binary classification or regression output)
                if outputs.logits.shape[-1] == 1:
                    # Regression model
                    reward_score = outputs.logits[0, 0].item()
                elif outputs.logits.shape[-1] == 2:
                    # Binary classification - use probability of positive class
                    probs = torch.softmax(outputs.logits, dim=-1)
                    reward_score = probs[0, 1].item()
                else:
                    # Multi-class - use the logit/prob of highest class
                    reward_score = outputs.logits[0].max().item()
                
                return {
                    'reward_score': reward_score,
                    'raw_logits': outputs.logits[0].tolist(),
                    'num_tokens': inputs['input_ids'].shape[1],
                }
            
            elif self.model_type == "causal_lm_value_head":
                # Model with explicit value head
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(self.model, 'v_head'):
                    value = self.model.v_head(outputs.hidden_states[-1][:, -1, :])
                elif hasattr(self.model, 'value_head'):
                    value = self.model.value_head(outputs.hidden_states[-1][:, -1, :])
                else:
                    value = outputs.hidden_states[-1][:, -1, :].mean(dim=-1)
                
                reward_score = value.squeeze().item()
                
                return {
                    'reward_score': reward_score,
                    'num_tokens': inputs['input_ids'].shape[1],
                }
            
            elif self.model_type == "causal_lm_score":
                # Model with score head
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(self.model, 'score'):
                    value = self.model.score(outputs.hidden_states[-1][:, -1, :])
                else:
                    value = outputs.hidden_states[-1][:, -1, :].mean(dim=-1)
                
                reward_score = value.squeeze().item()
                
                return {
                    'reward_score': reward_score,
                    'num_tokens': inputs['input_ids'].shape[1],
                }
            
            else:  # causal_lm_hidden - fallback
                # Use the mean of last hidden state or some heuristic
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get last hidden state at last token position
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                # Use L2 norm as a rough "quality" score (higher norm = more confident)
                reward_score = torch.norm(last_hidden, dim=-1).item()
                
                return {
                    'reward_score': reward_score,
                    'num_tokens': inputs['input_ids'].shape[1],
                    'note': 'Using hidden state norm as fallback score',
                }
    
    def check_continuity_reward(self, text_a: str, text_b: str,
                                 threshold: float = 0.0) -> Dict[str, Any]:
        """
        Check if text_b is a natural continuation of text_a using reward score.
        
        Args:
            text_a: First text segment (prompt)
            text_b: Second text segment (response)
            threshold: Reward threshold above which text is considered continuous
            
        Returns:
            Dict with 'answer' (Yes/No), 'reward_score', 'is_continuous' (bool)
        """
        metrics = self.compute_reward(text_a, text_b)
        reward_score = metrics['reward_score']
        
        # Higher reward = better continuation
        if reward_score >= threshold:
            answer = "Yes"
            is_continuous = True
        else:
            answer = "No"
            is_continuous = False
        
        return {
            "answer": answer,
            "reward_score": reward_score,
            "is_continuous": is_continuous,
            "threshold": threshold,
            **{k: v for k, v in metrics.items() if k != 'reward_score'},
        }
    
    def compute_reward_batch(self, texts_a: List[str], texts_b: List[str],
                             threshold: float = 0.0,
                             show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Compute reward-based continuity for a batch of text pairs.
        
        Args:
            texts_a: List of first text segments
            texts_b: List of second text segments
            threshold: Reward threshold for continuity
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dictionaries
        """
        results = []
        iterator = zip(texts_a, texts_b)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Computing reward scores")
        
        for text_a, text_b in iterator:
            result = self.check_continuity_reward(text_a, text_b, threshold)
            results.append(result)
        
        return results


def extract_text_from_batch(batch: DataProto, tokenizer, idx: int) -> Dict[str, str]:
    """
    Extract prompt and response text from a batch item.
    
    Args:
        batch: DataProto with prompts and responses
        tokenizer: Tokenizer for decoding
        idx: Index of the item in the batch
        
    Returns:
        Dict with 'prompt_text' and 'response_text'
    """
    prompts = batch.batch.get("prompts")
    responses = batch.batch.get("responses")
    attention_mask = batch.batch.get("attention_mask")
    
    if prompts is None or responses is None:
        raise ValueError("prompts or responses not found in batch")
    
    prompt_length = prompts.shape[1]
    
    # Get prompt tokens (only attended ones)
    prompt_ids = prompts[idx]
    if attention_mask is not None:
        prompt_mask = attention_mask[idx][:prompt_length]
        attended_prompt_ids = prompt_ids[prompt_mask.bool()]
    else:
        attended_prompt_ids = prompt_ids[prompt_ids != tokenizer.pad_token_id]
    
    prompt_text = tokenizer.decode(attended_prompt_ids, skip_special_tokens=True)
    
    # Get response tokens (only attended ones)
    response_ids = responses[idx]
    if attention_mask is not None:
        response_mask = attention_mask[idx][prompt_length:]
        attended_response_ids = response_ids[response_mask.bool()]
    else:
        attended_response_ids = response_ids[response_ids != tokenizer.pad_token_id]
    
    response_text = tokenizer.decode(attended_response_ids, skip_special_tokens=True)
    
    return {
        "prompt_text": prompt_text,
        "response_text": response_text
    }


def main():
    parser = argparse.ArgumentParser(description="Check if prompt-response pairs form continuous sentences")
    
    # Data configuration
    parser.add_argument("--train_files", type=str, required=True,
                        help="Path to training data file(s)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to base model (for tokenizer and data processing). "
                             "For reward_model method, defaults to reward_model_path.")
    parser.add_argument("--inference_model_path", type=str, default=None,
                        help="Path to inference model (defaults to model_path)")
    
    # Dataset parameters
    parser.add_argument("--num_entries", type=int, default=100,
                        help="Number of entries to process (default: 100)")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--max_prompt_length", type=int, default=550,
                        help="Maximum prompt length (default: 550)")
    parser.add_argument("--max_response_length", type=int, default=1000,
                        help="Maximum response length (default: 1000)")
    parser.add_argument("--prompt_truncation", type=str, default="left",
                        choices=["left", "right", "error"],
                        help="Prompt truncation strategy (default: left)")
    parser.add_argument("--response_truncation", type=str, default="right",
                        choices=["left", "right", "error"],
                        help="Response truncation strategy (default: right)")
    
    # Processing options
    parser.add_argument("--freeze_generation", action="store_true", default=False,
                        help="Use freeze generation mode")
    parser.add_argument("--reward_model_enable_train", action="store_true", default=False,
                        help="Enable reward model training data loading")
    
    # Inference options
    parser.add_argument("--method", type=str, default="prompt",
                        choices=["prompt", "perplexity", "reward_model"],
                        help="Method to check continuity: 'prompt' (ask LLM Yes/No, requires instruct model), "
                             "'perplexity' (measure perplexity, works with base models), or "
                             "'reward_model' (use off-the-shelf reward model scores). Default: prompt")
    parser.add_argument("--perplexity_threshold", type=float, default=50.0,
                        help="Perplexity threshold for 'perplexity' method. Lower perplexity = more continuous. "
                             "Pairs with perplexity < threshold are considered continuous. Default: 50.0")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to reward model (for 'reward_model' method). If not specified, uses inference_model_path.")
    parser.add_argument("--reward_threshold", type=float, default=0.0,
                        help="Reward threshold for 'reward_model' method. Higher reward = more continuous. "
                             "Pairs with reward >= threshold are considered continuous. Default: 0.0")
    parser.add_argument("--reward_model_type", type=str, default="auto",
                        choices=["auto", "sequence_classification", "causal_lm_value_head", "causal_lm_score"],
                        help="Type of reward model architecture. Default: auto (tries to detect automatically)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Torch dtype for model (default: auto)")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code")
    
    # Output options
    parser.add_argument("--output_json", type=str, default="continuity_results.json",
                        help="Path to output JSON file (default: continuity_results.json)")
    parser.add_argument("--max_prompt_chars", type=int, default=2000,
                        help="Max chars to save for prompt in JSON (default: 2000)")
    parser.add_argument("--max_response_chars", type=int, default=2000,
                        help="Max chars to save for response in JSON (default: 2000)")
    parser.add_argument("--cache_dir", type=str, default="~/.cache/verl/rlhf",
                        help="Cache directory for datasets")
    
    args = parser.parse_args()
    
    # Validate and set model paths
    if args.method == "reward_model":
        # For reward_model method, reward_model_path is primary
        if args.reward_model_path is None and args.model_path is None:
            parser.error("--reward_model_path (or --model_path) is required for reward_model method")
        if args.reward_model_path is None:
            args.reward_model_path = args.model_path
        if args.model_path is None:
            args.model_path = args.reward_model_path
        # Use reward model's tokenizer for dataset building (mimics prime_ray_trainer)
        tokenizer_model_path = args.reward_model_path
    else:
        # For prompt/perplexity methods, model_path is required
        if args.model_path is None:
            parser.error("--model_path is required for prompt/perplexity methods")
        tokenizer_model_path = args.model_path
    
    # Set inference model path
    if args.inference_model_path is None:
        args.inference_model_path = args.model_path
    
    # Set reward model path (if not already set)
    if args.reward_model_path is None:
        args.reward_model_path = args.inference_model_path
    
    print(f"\n{'='*80}")
    print("CONTINUITY CHECK SCRIPT")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Load tokenizer for data processing
    print(f"\n--- Loading tokenizer from {tokenizer_model_path} ---")
    local_model_path = copy_to_local(tokenizer_model_path)
    tokenizer = get_tokenizer(local_model_path, trust_remote_code=args.trust_remote_code)
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    
    # Get generation config
    generation_config = get_generation_config(local_model_path, trust_remote_code=args.trust_remote_code)
    eos_token_id = generation_config.eos_token_id if generation_config is not None else tokenizer.eos_token_id
    pad_token_id = generation_config.pad_token_id if generation_config is not None else tokenizer.pad_token_id
    
    # Create dataset config
    class DataConfig:
        def __init__(self, args):
            self.prompt_key = "prompt"
            self.reward_model_key = "reward_model"
            self.reward_model_enable_train = args.reward_model_enable_train
            self.image_key = "images"
            self.video_key = "videos"
            self.max_prompt_length = args.max_prompt_length
            self.max_response_length = args.max_response_length
            self.return_raw_chat = False
            self.truncation = args.prompt_truncation
            self.prompt_truncation = args.prompt_truncation
            self.response_truncation = args.response_truncation
            self.filter_overlong_prompts = True
            self.filter_overlong_prompts_workers = 1
            self.chat_template_func = None
            self.need_tools_kwargs = False
            self.filter_prompts = True
            self.cache_dir = args.cache_dir
            
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    data_config = DataConfig(args)
    
    # Parse train files
    train_files = [f.strip() for f in args.train_files.split(",")]
    
    # Create dataset
    print(f"\n--- Creating dataset ---")
    print(f"  Train files: {train_files}")
    dataset = RLHFDataset(
        data_files=train_files,
        tokenizer=tokenizer,
        config=data_config,
        processor=None
    )
    print(f"  Dataset size: {len(dataset)}")
    
    # Limit to num_entries
    num_entries = min(args.num_entries, len(dataset))
    print(f"  Processing first {num_entries} entries")
    
    # Create dataloader
    from torch.utils.data import DataLoader, SequentialSampler, Subset
    
    # Use Subset to get only first num_entries
    subset = Subset(dataset, range(num_entries))
    
    effective_batch_size = args.train_batch_size
    print(f"\n--- Creating dataloader ---")
    print(f"  Batch size: {effective_batch_size}")
    
    dataloader = DataLoader(
        dataset=subset,
        batch_size=effective_batch_size,
        num_workers=0,
        drop_last=False,  # Don't drop last batch to process all entries
        collate_fn=collate_fn,
        sampler=SequentialSampler(subset),
    )
    print(f"  Number of batches: {len(dataloader)}")
    
    # Collect all prompt-response pairs
    all_pairs = []
    
    print(f"\n--- Processing batches ---")
    for batch_idx, batch_dict in enumerate(tqdm(dataloader, desc="Preparing batches")):
        # Create DataProto from batch dict
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        # Pop keys for generation
        if args.freeze_generation:
            pop_keys = ["input_ids", "attention_mask", "position_ids",
                       "chosen_input_ids", "chosen_attention_mask", "chosen_position_ids"]
        else:
            pop_keys = ["input_ids", "attention_mask", "position_ids"]
        
        available_keys = [k for k in pop_keys if k in batch.batch.keys()]
        gen_batch = batch.pop(batch_keys=available_keys)
        
        # Generate or combine sequences
        if args.freeze_generation:
            if "chosen_input_ids" not in gen_batch.batch.keys():
                print(f"  WARNING: freeze_generation=True but chosen_input_ids not found!")
                continue
            
            gen_batch_output = combine_prompt_target_batch(
                gen_batch, tokenizer, eos_token_id, pad_token_id
            )
        else:
            # For non-freeze_generation, create mock output using prompts as both prompt and response
            # This is just for testing - in real use, you'd have actual responses
            prompt_length = gen_batch.batch["input_ids"].shape[1]
            mock_responses = gen_batch.batch["input_ids"][:, -min(50, prompt_length//2):]
            
            gen_batch_output = DataProto.from_dict(tensors={
                "prompts": gen_batch.batch["input_ids"],
                "responses": mock_responses,
                "input_ids": torch.cat([gen_batch.batch["input_ids"], mock_responses], dim=-1),
                "attention_mask": torch.cat([gen_batch.batch["attention_mask"], 
                                            torch.ones_like(mock_responses)], dim=-1),
            })
        
        # Add UID and repeat batch
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
        )
        
        if args.freeze_generation:
            repeat_factor = args.train_batch_size
        else:
            repeat_factor = 1
        
        batch = batch.repeat(repeat_times=repeat_factor, interleave=True)
        
        # Handle size mismatch
        if len(batch) != len(gen_batch_output):
            min_size = min(len(batch), len(gen_batch_output))
            batch = batch[:min_size]
            gen_batch_output = gen_batch_output[:min_size]
        
        batch = batch.union(gen_batch_output)
        
        # Extract text pairs
        for idx in range(len(batch)):
            try:
                texts = extract_text_from_batch(batch, tokenizer, idx)
                
                # Get metadata
                original_idx = batch_idx * effective_batch_size
                if args.freeze_generation:
                    prompt_idx = idx // args.train_batch_size
                    response_idx = idx % args.train_batch_size
                    is_diagonal = prompt_idx == response_idx
                else:
                    prompt_idx = idx
                    response_idx = idx
                    is_diagonal = True
                
                pair_info = {
                    "batch_idx": batch_idx,
                    "item_idx": idx,
                    "prompt_idx": prompt_idx,
                    "response_idx": response_idx,
                    "is_original_pair": is_diagonal,
                    "prompt_text": texts["prompt_text"][:args.max_prompt_chars],
                    "response_text": texts["response_text"][:args.max_response_chars],
                }
                
                # Add ground truth if available
                if "reward_model" in batch.non_tensor_batch:
                    gt = batch.non_tensor_batch["reward_model"][idx]
                    if isinstance(gt, dict):
                        gt = gt.get("ground_truth", "N/A")
                    pair_info["ground_truth"] = str(gt)[:500]
                
                all_pairs.append(pair_info)
                
            except Exception as e:
                print(f"  Error extracting pair at batch {batch_idx}, idx {idx}: {e}")
    
    print(f"\n  Total pairs collected: {len(all_pairs)}")
    
    # Initialize checker based on method
    print(f"\n--- Initializing {'reward model' if args.method == 'reward_model' else 'inference model'} ---")
    
    if args.method == "reward_model":
        scorer = RewardModelScorer(
            model_path=args.reward_model_path,
            tokenizer_path=None,  # Use reward model's own tokenizer
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            model_type=args.reward_model_type,
        )
    else:
        checker = ContinuityChecker(
            model_path=args.inference_model_path,
            tokenizer_path=args.model_path,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
        )
    
    # Check continuity for each pair
    print(f"\n--- Checking continuity for {len(all_pairs)} pairs using '{args.method}' method ---")
    if args.method == "perplexity":
        print(f"  Perplexity threshold: {args.perplexity_threshold}")
    elif args.method == "reward_model":
        print(f"  Reward threshold: {args.reward_threshold}")
    results = []
    
    for pair in tqdm(all_pairs, desc=f"Checking continuity ({args.method})"):
        try:
            if args.method == "prompt":
                result = checker.check_continuity(pair["prompt_text"], pair["response_text"])
                
                # Combine pair info with result
                entry = {
                    **pair,
                    "answer": result["answer"],
                    "raw_response": result["raw_response"],
                    "is_continuous": result["is_continuous"],
                }
            elif args.method == "perplexity":
                result = checker.check_continuity_perplexity(
                    pair["prompt_text"], 
                    pair["response_text"],
                    threshold=args.perplexity_threshold
                )
                
                # Combine pair info with result
                entry = {
                    **pair,
                    "answer": result["answer"],
                    "perplexity": result["perplexity"],
                    "loss": result["loss"],
                    "num_tokens": result["num_tokens"],
                    "total_perplexity": result["total_perplexity"],
                    "threshold": result["threshold"],
                    "is_continuous": result["is_continuous"],
                }
            else:  # reward_model method
                result = scorer.check_continuity_reward(
                    pair["prompt_text"],
                    pair["response_text"],
                    threshold=args.reward_threshold
                )
                
                # Combine pair info with result
                entry = {
                    **pair,
                    "answer": result["answer"],
                    "reward_score": result["reward_score"],
                    "threshold": result["threshold"],
                    "is_continuous": result["is_continuous"],
                }
                # Add any additional metrics from the result
                for k, v in result.items():
                    if k not in entry:
                        entry[k] = v
            
            results.append(entry)
            
        except Exception as e:
            print(f"  Error checking pair: {e}")
            entry = {
                **pair,
                "answer": "Error",
                "error_message": str(e),
                "is_continuous": None,
            }
            results.append(entry)
    
    # Compute statistics
    print(f"\n--- Computing statistics ---")
    total = len(results)
    yes_count = sum(1 for r in results if r["answer"] == "Yes")
    no_count = sum(1 for r in results if r["answer"] == "No")
    unknown_count = sum(1 for r in results if r["answer"] not in ["Yes", "No"])
    
    # Stats for original pairs (diagonal) vs non-original
    original_pairs = [r for r in results if r.get("is_original_pair", True)]
    non_original_pairs = [r for r in results if not r.get("is_original_pair", True)]
    
    original_yes = sum(1 for r in original_pairs if r["answer"] == "Yes")
    non_original_yes = sum(1 for r in non_original_pairs if r["answer"] == "Yes") if non_original_pairs else 0
    
    stats = {
        "method": args.method,
        "total_pairs": total,
        "yes_count": yes_count,
        "no_count": no_count,
        "unknown_count": unknown_count,
        "yes_percentage": 100 * yes_count / total if total > 0 else 0,
        "no_percentage": 100 * no_count / total if total > 0 else 0,
        "original_pairs_count": len(original_pairs),
        "original_pairs_yes_count": original_yes,
        "original_pairs_yes_percentage": 100 * original_yes / len(original_pairs) if original_pairs else 0,
        "non_original_pairs_count": len(non_original_pairs),
        "non_original_pairs_yes_count": non_original_yes,
        "non_original_pairs_yes_percentage": 100 * non_original_yes / len(non_original_pairs) if non_original_pairs else 0,
    }
    
    # Add method-specific statistics
    import statistics as stat_module
    
    if args.method == "perplexity":
        stats["perplexity_threshold"] = args.perplexity_threshold
        
        # Compute perplexity statistics
        valid_perplexities = [r["perplexity"] for r in results 
                             if "perplexity" in r and r["perplexity"] != float('inf')]
        if valid_perplexities:
            stats["perplexity_mean"] = stat_module.mean(valid_perplexities)
            stats["perplexity_median"] = stat_module.median(valid_perplexities)
            stats["perplexity_stdev"] = stat_module.stdev(valid_perplexities) if len(valid_perplexities) > 1 else 0
            stats["perplexity_min"] = min(valid_perplexities)
            stats["perplexity_max"] = max(valid_perplexities)
        
        # Perplexity stats for original vs non-original pairs
        original_perplexities = [r["perplexity"] for r in original_pairs 
                                 if "perplexity" in r and r["perplexity"] != float('inf')]
        non_original_perplexities = [r["perplexity"] for r in non_original_pairs 
                                     if "perplexity" in r and r["perplexity"] != float('inf')]
        
        if original_perplexities:
            stats["original_pairs_perplexity_mean"] = stat_module.mean(original_perplexities)
            stats["original_pairs_perplexity_median"] = stat_module.median(original_perplexities)
        if non_original_perplexities:
            stats["non_original_pairs_perplexity_mean"] = stat_module.mean(non_original_perplexities)
            stats["non_original_pairs_perplexity_median"] = stat_module.median(non_original_perplexities)
    
    elif args.method == "reward_model":
        stats["reward_threshold"] = args.reward_threshold
        stats["reward_model_path"] = args.reward_model_path
        stats["reward_model_type"] = args.reward_model_type
        
        # Compute reward score statistics
        valid_rewards = [r["reward_score"] for r in results 
                        if "reward_score" in r and r["reward_score"] is not None]
        if valid_rewards:
            stats["reward_mean"] = stat_module.mean(valid_rewards)
            stats["reward_median"] = stat_module.median(valid_rewards)
            stats["reward_stdev"] = stat_module.stdev(valid_rewards) if len(valid_rewards) > 1 else 0
            stats["reward_min"] = min(valid_rewards)
            stats["reward_max"] = max(valid_rewards)
        
        # Reward stats for original vs non-original pairs
        original_rewards = [r["reward_score"] for r in original_pairs 
                           if "reward_score" in r and r["reward_score"] is not None]
        non_original_rewards = [r["reward_score"] for r in non_original_pairs 
                               if "reward_score" in r and r["reward_score"] is not None]
        
        if original_rewards:
            stats["original_pairs_reward_mean"] = stat_module.mean(original_rewards)
            stats["original_pairs_reward_median"] = stat_module.median(original_rewards)
        if non_original_rewards:
            stats["non_original_pairs_reward_mean"] = stat_module.mean(non_original_rewards)
            stats["non_original_pairs_reward_median"] = stat_module.median(non_original_rewards)
    
    print(f"\nResults Summary:")
    print(f"  Method: {args.method}")
    print(f"  Total pairs: {total}")
    print(f"  Yes (continuous): {yes_count} ({stats['yes_percentage']:.1f}%)")
    print(f"  No (not continuous): {no_count} ({stats['no_percentage']:.1f}%)")
    print(f"  Unknown/Error: {unknown_count}")
    
    # Print perplexity-specific stats
    if args.method == "perplexity" and "perplexity_mean" in stats:
        print(f"\n  Perplexity Statistics:")
        print(f"    Threshold: {args.perplexity_threshold}")
        print(f"    Mean: {stats['perplexity_mean']:.2f}")
        print(f"    Median: {stats['perplexity_median']:.2f}")
        print(f"    Stdev: {stats['perplexity_stdev']:.2f}")
        print(f"    Min: {stats['perplexity_min']:.2f}")
        print(f"    Max: {stats['perplexity_max']:.2f}")
    
    # Print reward model-specific stats
    if args.method == "reward_model" and "reward_mean" in stats:
        print(f"\n  Reward Model Statistics:")
        print(f"    Model: {args.reward_model_path}")
        print(f"    Threshold: {args.reward_threshold}")
        print(f"    Mean: {stats['reward_mean']:.4f}")
        print(f"    Median: {stats['reward_median']:.4f}")
        print(f"    Stdev: {stats['reward_stdev']:.4f}")
        print(f"    Min: {stats['reward_min']:.4f}")
        print(f"    Max: {stats['reward_max']:.4f}")
    
    if args.freeze_generation:
        print(f"\n  Original pairs (diagonal):")
        print(f"    Count: {len(original_pairs)}")
        print(f"    Yes: {original_yes} ({stats['original_pairs_yes_percentage']:.1f}%)")
        if args.method == "perplexity" and "original_pairs_perplexity_mean" in stats:
            print(f"    Perplexity mean: {stats['original_pairs_perplexity_mean']:.2f}")
            print(f"    Perplexity median: {stats['original_pairs_perplexity_median']:.2f}")
        if args.method == "reward_model" and "original_pairs_reward_mean" in stats:
            print(f"    Reward mean: {stats['original_pairs_reward_mean']:.4f}")
            print(f"    Reward median: {stats['original_pairs_reward_median']:.4f}")
        
        print(f"\n  Non-original pairs (off-diagonal):")
        print(f"    Count: {len(non_original_pairs)}")
        print(f"    Yes: {non_original_yes} ({stats['non_original_pairs_yes_percentage']:.1f}%)")
        if args.method == "perplexity" and "non_original_pairs_perplexity_mean" in stats:
            print(f"    Perplexity mean: {stats['non_original_pairs_perplexity_mean']:.2f}")
            print(f"    Perplexity median: {stats['non_original_pairs_perplexity_median']:.2f}")
        if args.method == "reward_model" and "non_original_pairs_reward_mean" in stats:
            print(f"    Reward mean: {stats['non_original_pairs_reward_mean']:.4f}")
            print(f"    Reward median: {stats['non_original_pairs_reward_median']:.4f}")
    
    # Save results to JSON
    output_data = {
        "config": vars(args),
        "statistics": stats,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n--- Results saved to {args.output_json} ---")
    print(f"\n{'='*80}")
    print("CONTINUITY CHECK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

