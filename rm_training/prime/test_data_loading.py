#!/usr/bin/env python3
"""
Standalone script to test dataset creation, data loading, batch preparation,
and verification scoring for the PRIME trainer. This script validates the 
data pipeline up to:
    batch = batch.union(gen_batch_output)
    scores = reward_fn.verify(batch)

Usage:
    # Basic data loading test
    python test_data_loading.py --train_files /path/to/train.parquet --model_path /path/to/model
    
    # With output file
    python test_data_loading.py --train_files /path/to/train.parquet --model_path /path/to/model --output_file output.txt
    
    # With verification (freeze_generation mode - all-to-all prompt-response combinations)
    python test_data_loading.py --train_files /path/to/train.parquet --model_path /path/to/model \
        --freeze_generation --reward_model_enable_train --run_verification
    
    # Full example with all options
    python test_data_loading.py --train_files /path/to/train.parquet --model_path /path/to/model \
        --freeze_generation --reward_model_enable_train --run_verification \
        --train_batch_size 4 --max_prompt_length 550 --max_response_length 1000 \
        --output_file verification_output.txt

This script removes Ray/FSDP dependencies and allows direct visualization of:
- Dataset loading
- Batch preparation  
- Attention mask positions
- Position IDs
- Token sequences
- Verification scores for each prompt-response combination
"""

import argparse
import os
import sys
import uuid
from datetime import datetime
import builtins

import numpy as np
import torch
from tensordict import TensorDict

# Add verl to path
verl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, verl_root)

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import get_generation_config
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import _default_compute_score


class Logger:
    """Logger that writes to both stdout and optionally a file."""
    
    def __init__(self, output_file=None):
        self.output_file = output_file
        self.file_handle = None
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True) if os.path.dirname(output_file) else None
            self.file_handle = open(output_file, 'w', encoding='utf-8')
            self.log(f"# Output file created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log(f"# Output file path: {os.path.abspath(output_file)}\n")
    
    def log(self, *args, **kwargs):
        """Print to stdout and write to file if configured."""
        # Convert args to string
        message = ' '.join(str(arg) for arg in args)
        
        # Print to stdout using built-in print
        builtins.print(message, **kwargs)
        
        # Write to file
        if self.file_handle:
            self.file_handle.write(message + '\n')
            self.file_handle.flush()  # Ensure it's written immediately
    
    def close(self):
        """Close the file handle."""
        if self.file_handle:
            self.log(f"\n# Output file closed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.file_handle.close()
            self.file_handle = None


# Global logger instance
logger = Logger()


def log(*args, **kwargs):
    """Convenience function to use the global logger."""
    logger.log(*args, **kwargs)


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
    This creates all-to-all combinations of prompts and responses.
    
    The responses are converted from left-padded to right-padded to avoid
    a gap of padding tokens between prompt and response in the combined sequence.
    
    Result: [prompt_padding | prompt_content | response_content | response_padding]
    """
    # Fallback: use eos_token_id if pad_token_id is None
    if pad_token_id is None:
        pad_token_id = eos_token_id if not isinstance(eos_token_id, list) else eos_token_id[0]
    
    prompt_ids = gen_batch.batch["input_ids"]
    prompt_mask = gen_batch.batch["attention_mask"]
    prompt_pos = gen_batch.batch["position_ids"]
    resp_ids = gen_batch.batch["chosen_input_ids"]
    resp_mask = gen_batch.batch["chosen_attention_mask"]
    resp_pos = gen_batch.batch["chosen_position_ids"]

    # Handle both single response and batched responses case
    if resp_ids.dim() == 1:
        resp_ids = resp_ids.unsqueeze(0)
        resp_mask = resp_mask.unsqueeze(0)
        resp_pos = resp_pos.unsqueeze(0)

    batch_size = prompt_ids.size(0)
    n_responses = batch_size

    # Create all-to-all combinations
    # Repeat prompts n_responses times
    prompt_ids = prompt_ids.repeat_interleave(n_responses, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(n_responses, dim=0)
    prompt_pos = prompt_pos.repeat_interleave(n_responses, dim=0)

    # Repeat each response batch_size times
    resp_ids = resp_ids.repeat(batch_size, 1)
    resp_mask = resp_mask.repeat(batch_size, 1)
    resp_pos = resp_pos.repeat(batch_size, 1)
    batch_size = batch_size * batch_size
    
    # Convert response from LEFT-padded to RIGHT-padded
    # This removes the gap between prompt content and response content
    # Before: [PAD, PAD, content, content, content]
    # After:  [content, content, content, PAD, PAD]
    response_length = resp_ids.size(1)
    resp_ids_right_pad = torch.full_like(resp_ids, pad_token_id)
    resp_mask_right_pad = torch.zeros_like(resp_mask)
    
    for i in range(batch_size):
        # Find attended tokens (mask == 1)
        attended_mask = resp_mask[i] == 1
        content_tokens = resp_ids[i][attended_mask]
        content_length = content_tokens.size(0)
        
        # Place content at the beginning (right-pad style)
        resp_ids_right_pad[i, :content_length] = content_tokens
        resp_mask_right_pad[i, :content_length] = 1
    
    # Use the right-padded versions
    resp_ids = resp_ids_right_pad
    resp_mask = resp_mask_right_pad
    
    seq = torch.cat([prompt_ids, resp_ids], dim=-1)

    # Calculate position IDs for responses
    # Response positions continue from where prompt ended
    delta_position_id = torch.arange(1, response_length + 1, device=resp_pos.device)
    delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

    # prompt: left pad + response: right pad (converted)
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,1,1,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
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


def visualize_prompt_response_alignment(batch: DataProto, tokenizer, idx: int = 0, title: str = ""):
    """
    Thoroughly visualize how prompt and response are linked in the sequence.
    Shows token-by-token alignment with attention masks.
    """
    log(f"\n{'#'*100}")
    log(f"# PROMPT-RESPONSE ALIGNMENT VISUALIZATION {title} (item {idx})")
    log(f"{'#'*100}")
    
    # Get tensors
    input_ids = batch.batch["input_ids"][idx]
    attention_mask = batch.batch["attention_mask"][idx]
    position_ids = batch.batch["position_ids"][idx]
    
    has_separate_prompt_response = "prompts" in batch.batch.keys() and "responses" in batch.batch.keys()
    
    if has_separate_prompt_response:
        prompts = batch.batch["prompts"][idx]
        responses = batch.batch["responses"][idx]
        prompt_len = prompts.shape[0]
        response_len = responses.shape[0]
    else:
        prompt_len = 0
        response_len = 0
    
    total_len = input_ids.shape[0]
    
    # ============ SECTION 1: OVERVIEW ============
    log(f"\n{'='*80}")
    log("SECTION 1: SEQUENCE OVERVIEW")
    log(f"{'='*80}")
    log(f"  Total sequence length: {total_len}")
    if has_separate_prompt_response:
        log(f"  Prompt tensor length:  {prompt_len}")
        log(f"  Response tensor length: {response_len}")
        log(f"  Expected total: {prompt_len + response_len} (prompt + response)")
    log(f"  Attended tokens (mask=1): {attention_mask.sum().item()}")
    log(f"  Masked tokens (mask=0):   {(attention_mask == 0).sum().item()}")
    
    # ============ SECTION 2: ATTENTION MASK STRUCTURE ============
    log(f"\n{'='*80}")
    log("SECTION 2: ATTENTION MASK STRUCTURE")
    log(f"{'='*80}")
    
    # Find transitions in attention mask
    mask_changes = []
    prev_val = attention_mask[0].item()
    mask_changes.append((0, prev_val))
    for i in range(1, len(attention_mask)):
        curr_val = attention_mask[i].item()
        if curr_val != prev_val:
            mask_changes.append((i, curr_val))
            prev_val = curr_val
    
    log("  Attention mask transitions (position, value):")
    for pos, val in mask_changes:
        region_type = "ATTENDED" if val == 1 else "MASKED/PADDED"
        log(f"    Position {pos:4d}: mask={val} ({region_type})")
    
    # Identify regions
    log("\n  Attention mask regions:")
    for i in range(len(mask_changes)):
        start_pos = mask_changes[i][0]
        end_pos = mask_changes[i+1][0] if i+1 < len(mask_changes) else total_len
        val = mask_changes[i][1]
        length = end_pos - start_pos
        region_type = "ATTENDED" if val == 1 else "PADDING"
        log(f"    [{start_pos:4d} - {end_pos:4d}] length={length:4d} : {region_type}")
    
    if has_separate_prompt_response:
        log(f"\n  Expected structure for prompt+response:")
        log(f"    [0 - {prompt_len}] = PROMPT region (length {prompt_len})")
        log(f"    [{prompt_len} - {total_len}] = RESPONSE region (length {response_len})")
    
    # ============ SECTION 3: RAW TOKEN IDS ============
    log(f"\n{'='*80}")
    log("SECTION 3: RAW TOKEN IDS (before decoding)")
    log(f"{'='*80}")
    
    if has_separate_prompt_response:
        log("\n  PROMPT token IDs:")
        log(f"    First 20: {prompts[:20].tolist()}")
        log(f"    Last 20:  {prompts[-20:].tolist()}")
        
        log("\n  RESPONSE token IDs:")
        log(f"    First 20: {responses[:20].tolist()}")
        log(f"    Last 20:  {responses[-20:].tolist()}")
    
    log("\n  FULL SEQUENCE token IDs:")
    log(f"    First 20: {input_ids[:20].tolist()}")
    log(f"    Last 20:  {input_ids[-20:].tolist()}")
    
    # ============ SECTION 4: DECODED TEXT ============
    log(f"\n{'='*80}")
    log("SECTION 4: DECODED TEXT")
    log(f"{'='*80}")
    
    if has_separate_prompt_response:
        log("\n  [PROMPT - DECODED]:")
        log("-" * 60)
        prompt_text = tokenizer.decode(prompts, skip_special_tokens=False)
        log(prompt_text)
        log("-" * 60)
        
        log("\n  [RESPONSE - DECODED]:")
        log("-" * 60)
        response_text = tokenizer.decode(responses, skip_special_tokens=False)
        log(response_text)
        log("-" * 60)
    
    log("\n  [FULL SEQUENCE - DECODED]:")
    log("-" * 60)
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    log(full_text)
    log("-" * 60)
    
    log("\n  [ATTENDED PORTION ONLY - DECODED]:")
    log("-" * 60)
    attended_ids = input_ids[attention_mask.bool()]
    attended_text = tokenizer.decode(attended_ids, skip_special_tokens=False)
    log(attended_text)
    log("-" * 60)
    
    # ============ SECTION 5: TOKEN-BY-TOKEN ALIGNMENT ============
    log(f"\n{'='*80}")
    log("SECTION 5: TOKEN-BY-TOKEN ALIGNMENT (showing boundaries)")
    log(f"{'='*80}")
    
    # Show around the prompt/response boundary
    if has_separate_prompt_response and prompt_len > 0:
        boundary = prompt_len
        start_show = max(0, boundary - 15)
        end_show = min(total_len, boundary + 15)
        
        log(f"\n  Tokens around PROMPT/RESPONSE boundary (position {boundary}):")
        log(f"  {'Pos':>5} | {'Mask':>4} | {'PosID':>5} | {'TokID':>8} | {'Region':>10} | Token Text")
        log(f"  {'-'*5}-+-{'-'*4}-+-{'-'*5}-+-{'-'*8}-+-{'-'*10}-+-{'-'*30}")
        
        for i in range(start_show, end_show):
            tok_id = input_ids[i].item()
            mask_val = attention_mask[i].item()
            pos_id = position_ids[i].item() if len(position_ids.shape) == 1 else position_ids[0, i].item()
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
            tok_text = repr(tok_text)[1:-1][:25]  # Escape special chars, limit length
            
            if i < boundary:
                region = "PROMPT"
            else:
                region = "RESPONSE"
            
            marker = " <<< BOUNDARY" if i == boundary else ""
            log(f"  {i:5d} | {mask_val:4d} | {pos_id:5d} | {tok_id:8d} | {region:>10} | {tok_text}{marker}")
    
    # ============ SECTION 6: ATTENTION MASK VISUAL ============
    log(f"\n{'='*80}")
    log("SECTION 6: ATTENTION MASK VISUAL REPRESENTATION")
    log(f"{'='*80}")
    
    # Create visual representation
    # Use blocks of 50 tokens per line
    block_size = 50
    log(f"\n  Legend: █=attended(1), ░=masked(0)")
    log(f"  Each character = 1 token, {block_size} tokens per line")
    
    if has_separate_prompt_response:
        log(f"  Prompt ends at position {prompt_len}, Response starts at position {prompt_len}")
    
    log()
    for block_start in range(0, total_len, block_size):
        block_end = min(block_start + block_size, total_len)
        visual = ""
        for i in range(block_start, block_end):
            if attention_mask[i].item() == 1:
                visual += "█"
            else:
                visual += "░"
        
        # Add boundary marker
        if has_separate_prompt_response and block_start <= prompt_len < block_end:
            boundary_pos = prompt_len - block_start
            log(f"  [{block_start:4d}-{block_end:4d}] {visual}")
            log(f"            {' ' * boundary_pos}↑ PROMPT|RESPONSE boundary")
        else:
            log(f"  [{block_start:4d}-{block_end:4d}] {visual}")
    
    # ============ SECTION 7: POSITION IDS ============
    log(f"\n{'='*80}")
    log("SECTION 7: POSITION IDS")
    log(f"{'='*80}")
    
    # Handle multi-dimensional position_ids (e.g., for rotary embeddings)
    if len(position_ids.shape) == 1:
        log(f"  Position IDs shape: {position_ids.shape} (1D)")
        log(f"  First 30: {position_ids[:30].tolist()}")
        log(f"  Last 30:  {position_ids[-30:].tolist()}")
        log(f"  Min: {position_ids.min().item()}, Max: {position_ids.max().item()}")
        
        # Check for continuity
        if has_separate_prompt_response:
            prompt_last_pos = position_ids[prompt_len - 1].item()
            response_first_pos = position_ids[prompt_len].item()
            log(f"\n  At boundary:")
            log(f"    Last prompt position_id:     {prompt_last_pos}")
            log(f"    First response position_id:  {response_first_pos}")
            log(f"    Difference: {response_first_pos - prompt_last_pos} (should be 1 for continuity)")
    else:
        log(f"  Position IDs shape: {position_ids.shape} (multi-dimensional)")
        log(f"  First dim, first 20: {position_ids[0, :20].tolist()}")
    
    # ============ SECTION 8: VERIFICATION CHECKS ============
    log(f"\n{'='*80}")
    log("SECTION 8: VERIFICATION CHECKS")
    log(f"{'='*80}")
    
    checks_passed = 0
    checks_total = 0
    
    if has_separate_prompt_response:
        # Check 1: Sequence length matches
        checks_total += 1
        expected_len = prompt_len + response_len
        if total_len == expected_len:
            log(f"  ✓ Sequence length check: PASS ({total_len} = {prompt_len} + {response_len})")
            checks_passed += 1
        else:
            log(f"  ✗ Sequence length check: FAIL ({total_len} != {prompt_len} + {response_len})")
        
        # Check 2: Prompt tokens match start of sequence
        checks_total += 1
        prompt_match = torch.equal(input_ids[:prompt_len], prompts)
        if prompt_match:
            log(f"  ✓ Prompt alignment check: PASS (input_ids[:prompt_len] == prompts)")
            checks_passed += 1
        else:
            log(f"  ✗ Prompt alignment check: FAIL (input_ids[:prompt_len] != prompts)")
        
        # Check 3: Response tokens match end of sequence
        checks_total += 1
        response_match = torch.equal(input_ids[prompt_len:], responses)
        if response_match:
            log(f"  ✓ Response alignment check: PASS (input_ids[prompt_len:] == responses)")
            checks_passed += 1
        else:
            log(f"  ✗ Response alignment check: FAIL (input_ids[prompt_len:] != responses)")
        
        # Check 4: Position IDs are continuous at boundary
        checks_total += 1
        if len(position_ids.shape) == 1 and prompt_len > 0:
            pos_diff = position_ids[prompt_len].item() - position_ids[prompt_len - 1].item()
            if pos_diff == 1:
                log(f"  ✓ Position ID continuity: PASS (diff at boundary = {pos_diff})")
                checks_passed += 1
            else:
                log(f"  ✗ Position ID continuity: FAIL (diff at boundary = {pos_diff}, expected 1)")
        else:
            log(f"  - Position ID continuity: SKIPPED (multi-dim or no prompt)")
            checks_passed += 1  # Don't count as failure
    
    # Check: No NaN values
    checks_total += 1
    has_nan = torch.isnan(input_ids.float()).any() or torch.isnan(attention_mask.float()).any()
    if not has_nan:
        log(f"  ✓ No NaN values: PASS")
        checks_passed += 1
    else:
        log(f"  ✗ No NaN values: FAIL")
    
    log(f"\n  Summary: {checks_passed}/{checks_total} checks passed")
    
    log(f"\n{'#'*100}\n")


def visualize_batch_item(batch: DataProto, tokenizer, idx: int = 0, title: str = ""):
    """Wrapper that calls the detailed visualization."""
    visualize_prompt_response_alignment(batch, tokenizer, idx, title)


def visualize_dataset_entry(dataset, tokenizer, idx: int = 0, max_prompt_length: int = 550, max_response_length: int = 1000, prompt_truncation: str = "left", response_truncation: str = "right"):
    """
    Visualize a single entry from the RLHFDataset before batching.
    Shows input_ids, chosen_input_ids, and their attention masks.
    Also highlights truncation effects by showing original vs truncated.
    
    Args:
        prompt_truncation: 'left' (keep end), 'right' (keep start), 'error'
        response_truncation: 'left' (keep end), 'right' (keep start), 'error'
    """
    log(f"\n{'#'*100}")
    log(f"# DATASET ENTRY VISUALIZATION (index {idx})")
    log(f"{'#'*100}")
    
    # Get the item from dataset (after truncation/padding)
    item = dataset[idx]
    
    # Get raw data from dataframe (before truncation)
    raw_row = dataset.dataframe[idx]
    
    # ============ BASIC INFO ============
    log(f"\n{'='*80}")
    log("BASIC INFO")
    log(f"{'='*80}")
    log(f"  Dataset index: {idx}")
    log(f"  Item keys: {list(item.keys())}")
    log(f"  Config: max_prompt_length={max_prompt_length}, max_response_length={max_response_length}")
    log(f"  Truncation: prompt='{prompt_truncation}', response='{response_truncation}'")
    
    # Check for index field
    if "index" in item:
        log(f"  Item 'index' field: {item['index']}")
    
    # ============ ORIGINAL (PRE-TRUNCATION) DATA ============
    log(f"\n{'='*80}")
    log("ORIGINAL DATA (BEFORE TRUNCATION)")
    log(f"{'='*80}")
    
    # Get original prompt
    prompt_key = dataset.prompt_key
    if prompt_key in raw_row:
        original_messages = raw_row[prompt_key]
        log(f"\n  [ORIGINAL PROMPT MESSAGES]:")
        log(f"  Number of messages: {len(original_messages)}")
        for i, msg in enumerate(original_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            content_preview = content[:200] + "..." if len(content) > 200 else content
            log(f"    Message {i} ({role}): {content_preview}")
        
        # Tokenize original prompt to get original length
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            original_prompt_text = tokenizer.apply_chat_template(original_messages, add_generation_prompt=True, tokenize=False)
            original_prompt_ids = tokenizer.encode(original_prompt_text, add_special_tokens=False)
        else:
            original_prompt_text = " ".join([msg.get("content", "") for msg in original_messages if isinstance(msg, dict)])
            original_prompt_ids = tokenizer.encode(original_prompt_text, add_special_tokens=False)
        
        original_prompt_len = len(original_prompt_ids)
        log(f"\n  Original prompt length: {original_prompt_len} tokens")
        log(f"  Max allowed: {max_prompt_length} tokens")
        
        if original_prompt_len > max_prompt_length:
            tokens_removed = original_prompt_len - max_prompt_length
            log(f"  ⚠️  PROMPT TRUNCATION OCCURRED: {tokens_removed} tokens removed!")
            log(f"  Truncation mode: '{prompt_truncation}'")
            
            if prompt_truncation == "left":
                log(f"  → Kept LAST {max_prompt_length} tokens (removed first {tokens_removed} tokens)")
                log(f"\n  [REMOVED PORTION - first {min(tokens_removed, 100)} tokens decoded]:")
                log("-" * 60)
                removed_ids = original_prompt_ids[:min(tokens_removed, 100)]
                log(tokenizer.decode(removed_ids, skip_special_tokens=False))
                log("-" * 60)
                if tokens_removed > 100:
                    log(f"  ... ({tokens_removed - 100} more tokens removed)")
            elif prompt_truncation == "right":
                log(f"  → Kept FIRST {max_prompt_length} tokens (removed last {tokens_removed} tokens)")
                log(f"\n  [REMOVED PORTION - last {min(tokens_removed, 100)} tokens decoded]:")
                log("-" * 60)
                removed_ids = original_prompt_ids[-min(tokens_removed, 100):]
                log(tokenizer.decode(removed_ids, skip_special_tokens=False))
                log("-" * 60)
                if tokens_removed > 100:
                    log(f"  ... ({tokens_removed - 100} more tokens removed)")
            
            # Show truncation boundary
            log(f"\n  [PROMPT TRUNCATION BOUNDARY VISUALIZATION]:")
            log(f"  Original: |{'█' * min(original_prompt_len // 10, 50)}| ({original_prompt_len} tokens)")
            if prompt_truncation == "left":
                kept_visual = min(max_prompt_length // 10, 50)
                removed_visual = min(tokens_removed // 10, 50)
                log(f"  Removed:  |{'░' * removed_visual}| ← removed from START")
                log(f"  Kept:     |{' ' * removed_visual}{'█' * kept_visual}| ({max_prompt_length} tokens)")
            elif prompt_truncation == "right":
                kept_visual = min(max_prompt_length // 10, 50)
                removed_visual = min(tokens_removed // 10, 50)
                log(f"  Kept:     |{'█' * kept_visual}| ({max_prompt_length} tokens)")
                log(f"  Removed:  |{' ' * kept_visual}{'░' * removed_visual}| ← removed from END")
        else:
            log(f"  ✓ No prompt truncation needed (within limit)")
    
    # Get original response (ground truth)
    reward_model_key = dataset.reward_model_key
    if hasattr(dataset, 'reward_model_enable_train') and dataset.reward_model_enable_train:
        if reward_model_key in raw_row:
            reward_data = raw_row[reward_model_key]
            if isinstance(reward_data, dict) and "ground_truth" in reward_data:
                original_response = reward_data["ground_truth"]
                if isinstance(original_response, list):
                    original_response = original_response[0] if original_response else ""
                
                original_response_ids = tokenizer.encode(original_response, add_special_tokens=False)
                original_response_len = len(original_response_ids)
                
                log(f"\n  Original response length: {original_response_len} tokens")
                log(f"  Max allowed: {max_response_length} tokens")
                
                log(f"\n  [ORIGINAL RESPONSE TEXT (first 500 chars)]:")
                log("-" * 60)
                log(original_response[:500] + ("..." if len(original_response) > 500 else ""))
                log("-" * 60)
                
                if original_response_len > max_response_length:
                    tokens_removed = original_response_len - max_response_length
                    log(f"\n  ⚠️  RESPONSE TRUNCATION OCCURRED: {tokens_removed} tokens removed!")
                    log(f"  Truncation mode: '{response_truncation}'")
                    
                    if response_truncation == "left":
                        log(f"  → Kept LAST {max_response_length} tokens (removed first {tokens_removed} tokens)")
                        log(f"\n  [REMOVED RESPONSE PORTION - first {min(tokens_removed, 200)} tokens decoded]:")
                        log("-" * 60)
                        removed_ids = original_response_ids[:min(tokens_removed, 200)]
                        log(tokenizer.decode(removed_ids, skip_special_tokens=False))
                        log("-" * 60)
                        if tokens_removed > 200:
                            log(f"  ... ({tokens_removed - 200} more tokens removed)")
                        
                        log(f"\n  [KEPT RESPONSE PORTION - first 200 tokens of kept part decoded]:")
                        log("-" * 60)
                        kept_ids = original_response_ids[-max_response_length:][:200]
                        log(tokenizer.decode(kept_ids, skip_special_tokens=False))
                        log("-" * 60)
                        
                    elif response_truncation == "right":
                        log(f"  → Kept FIRST {max_response_length} tokens (removed last {tokens_removed} tokens)")
                        log(f"\n  [KEPT RESPONSE PORTION - last 200 tokens of kept part decoded]:")
                        log("-" * 60)
                        kept_ids = original_response_ids[:max_response_length][-200:]
                        log(tokenizer.decode(kept_ids, skip_special_tokens=False))
                        log("-" * 60)
                        
                        log(f"\n  [REMOVED RESPONSE PORTION - last {min(tokens_removed, 200)} tokens decoded]:")
                        log("-" * 60)
                        removed_ids = original_response_ids[-min(tokens_removed, 200):]
                        log(tokenizer.decode(removed_ids, skip_special_tokens=False))
                        log("-" * 60)
                        if tokens_removed > 200:
                            log(f"  ... ({tokens_removed - 200} more tokens removed)")
                    
                    # Show truncation boundary visualization
                    log(f"\n  [RESPONSE TRUNCATION BOUNDARY VISUALIZATION]:")
                    log(f"  Original: |{'█' * min(original_response_len // 20, 50)}| ({original_response_len} tokens)")
                    if response_truncation == "left":
                        kept_visual = min(max_response_length // 20, 50)
                        removed_visual = min(tokens_removed // 20, 50)
                        log(f"  Removed:  |{'░' * removed_visual}| ← removed from START")
                        log(f"  Kept:     |{' ' * removed_visual}{'█' * kept_visual}| ({max_response_length} tokens)")
                    elif response_truncation == "right":
                        kept_visual = min(max_response_length // 20, 50)
                        removed_visual = min(tokens_removed // 20, 50)
                        log(f"  Kept:     |{'█' * kept_visual}| ({max_response_length} tokens)")
                        log(f"  Removed:  |{' ' * kept_visual}{'░' * removed_visual}| ← removed from END")
                else:
                    log(f"  ✓ No response truncation needed (within limit)")
    
    # ============ INPUT_IDS (PROMPT) ============
    log(f"\n{'='*80}")
    log("INPUT_IDS (PROMPT)")
    log(f"{'='*80}")
    
    if "input_ids" in item:
        input_ids = item["input_ids"]
        prompt_len = input_ids.shape[0]
        log(f"  Shape: {input_ids.shape}")
        log(f"  Dtype: {input_ids.dtype}")
        log(f"  Length: {prompt_len} tokens (max allowed: {max_prompt_length})")
        
        # Check if at max length (likely truncated)
        if prompt_len == max_prompt_length:
            log(f"  ⚠️  TRUNCATION LIKELY: Prompt is exactly at max_prompt_length!")
            log(f"      Truncation mode: '{prompt_truncation}' ({'keep end of prompt' if prompt_truncation == 'left' else 'keep start of prompt' if prompt_truncation == 'right' else 'would raise error'})")
        
        log(f"  First 20 token IDs: {input_ids[:20].tolist()}")
        log(f"  Last 20 token IDs: {input_ids[-20:].tolist()}")
        
        # Decode
        log(f"\n  [DECODED INPUT_IDS]:")
        log("-" * 60)
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        log(decoded)
        log("-" * 60)
    else:
        log("  (input_ids not found)")
    
    # ============ INPUT ATTENTION MASK ============
    log(f"\n{'='*80}")
    log("ATTENTION_MASK (for input_ids)")
    log(f"{'='*80}")
    
    if "attention_mask" in item:
        attn_mask = item["attention_mask"]
        log(f"  Shape: {attn_mask.shape}")
        log(f"  Dtype: {attn_mask.dtype}")
        log(f"  Sum (attended tokens): {attn_mask.sum().item()}")
        log(f"  First 50 values: {attn_mask[:50].tolist()}")
        log(f"  Last 50 values: {attn_mask[-50:].tolist()}")
        
        # Find first attended token
        attended_indices = (attn_mask == 1).nonzero(as_tuple=True)[0]
        if len(attended_indices) > 0:
            first_attended = attended_indices[0].item()
            last_attended = attended_indices[-1].item()
            log(f"  First attended position: {first_attended}")
            log(f"  Last attended position: {last_attended}")
            log(f"  Padding tokens (mask=0): {(attn_mask == 0).sum().item()}")
        
        # Visual representation
        log(f"\n  [VISUAL REPRESENTATION] (█=attended, ░=padding)")
        block_size = 50
        for block_start in range(0, len(attn_mask), block_size):
            block_end = min(block_start + block_size, len(attn_mask))
            visual = ""
            for i in range(block_start, block_end):
                visual += "█" if attn_mask[i].item() == 1 else "░"
            log(f"  [{block_start:4d}-{block_end:4d}] {visual}")
    else:
        log("  (attention_mask not found)")
    
    # ============ CHOSEN_INPUT_IDS (RESPONSE) ============
    log(f"\n{'='*80}")
    log("CHOSEN_INPUT_IDS (RESPONSE/GROUND TRUTH)")
    log(f"{'='*80}")
    
    if "chosen_input_ids" in item:
        chosen_ids = item["chosen_input_ids"]
        response_len = chosen_ids.shape[0]
        log(f"  Shape: {chosen_ids.shape}")
        log(f"  Dtype: {chosen_ids.dtype}")
        log(f"  Length: {response_len} tokens (max allowed: {max_response_length})")
        
        # Check if at max length (likely truncated)
        if response_len == max_response_length:
            log(f"  ⚠️  TRUNCATION LIKELY: Response is exactly at max_response_length!")
            log(f"      Truncation mode: '{response_truncation}' ({'keep end of response' if response_truncation == 'left' else 'keep start of response' if response_truncation == 'right' else 'would raise error'})")
        
        log(f"  First 20 token IDs: {chosen_ids[:20].tolist()}")
        log(f"  Last 20 token IDs: {chosen_ids[-20:].tolist()}")
        
        # Decode
        log(f"\n  [DECODED CHOSEN_INPUT_IDS]:")
        log("-" * 60)
        decoded = tokenizer.decode(chosen_ids, skip_special_tokens=False)
        log(decoded)
        log("-" * 60)
    else:
        log("  (chosen_input_ids not found - reward_model_enable_train may be False)")
    
    # ============ CHOSEN ATTENTION MASK ============
    log(f"\n{'='*80}")
    log("CHOSEN_ATTENTION_MASK (for chosen_input_ids)")
    log(f"{'='*80}")
    
    if "chosen_attention_mask" in item:
        chosen_attn_mask = item["chosen_attention_mask"]
        log(f"  Shape: {chosen_attn_mask.shape}")
        log(f"  Dtype: {chosen_attn_mask.dtype}")
        log(f"  Sum (attended tokens): {chosen_attn_mask.sum().item()}")
        log(f"  First 50 values: {chosen_attn_mask[:50].tolist()}")
        log(f"  Last 50 values: {chosen_attn_mask[-50:].tolist()}")
        
        # Find first attended token
        attended_indices = (chosen_attn_mask == 1).nonzero(as_tuple=True)[0]
        if len(attended_indices) > 0:
            first_attended = attended_indices[0].item()
            last_attended = attended_indices[-1].item()
            log(f"  First attended position: {first_attended}")
            log(f"  Last attended position: {last_attended}")
            log(f"  Padding tokens (mask=0): {(chosen_attn_mask == 0).sum().item()}")
        
        # Visual representation
        log(f"\n  [VISUAL REPRESENTATION] (█=attended, ░=padding)")
        block_size = 50
        for block_start in range(0, len(chosen_attn_mask), block_size):
            block_end = min(block_start + block_size, len(chosen_attn_mask))
            visual = ""
            for i in range(block_start, block_end):
                visual += "█" if chosen_attn_mask[i].item() == 1 else "░"
            log(f"  [{block_start:4d}-{block_end:4d}] {visual}")
    else:
        log("  (chosen_attention_mask not found - reward_model_enable_train may be False)")
    
    # ============ POSITION IDS ============
    log(f"\n{'='*80}")
    log("POSITION IDS")
    log(f"{'='*80}")
    
    if "position_ids" in item:
        pos_ids = item["position_ids"]
        log(f"  Shape: {pos_ids.shape}")
        if len(pos_ids.shape) == 1:
            log(f"  First 30: {pos_ids[:30].tolist()}")
            log(f"  Last 30: {pos_ids[-30:].tolist()}")
        else:
            log(f"  (Multi-dimensional position_ids)")
    
    if "chosen_position_ids" in item:
        chosen_pos_ids = item["chosen_position_ids"]
        log(f"\n  chosen_position_ids shape: {chosen_pos_ids.shape}")
        if len(chosen_pos_ids.shape) == 1:
            log(f"  First 30: {chosen_pos_ids[:30].tolist()}")
            log(f"  Last 30: {chosen_pos_ids[-30:].tolist()}")
    
    # ============ OTHER FIELDS ============
    log(f"\n{'='*80}")
    log("OTHER FIELDS")
    log(f"{'='*80}")
    
    skip_keys = {"input_ids", "attention_mask", "position_ids", 
                 "chosen_input_ids", "chosen_attention_mask", "chosen_position_ids",
                 "index", "multi_modal_data", "multi_modal_inputs"}
    
    for key, val in item.items():
        if key in skip_keys:
            continue
        if isinstance(val, torch.Tensor):
            log(f"  {key}: Tensor shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            log(f"  {key}: {type(val).__name__} len={len(val)}")
            if len(val) > 0 and len(val) <= 5:
                log(f"    Content: {val}")
        elif isinstance(val, dict):
            log(f"  {key}: dict with keys={list(val.keys())}")
        else:
            val_str = str(val)[:200]
            log(f"  {key}: {type(val).__name__} = {val_str}")
    
    log(f"\n{'#'*100}\n")


def visualize_non_tensor_batch(batch: DataProto, max_items: int = 3):
    """Visualize non-tensor batch data."""
    log(f"\n{'='*80}")
    log("NON-TENSOR BATCH DATA")
    log(f"{'='*80}")
    
    if not batch.non_tensor_batch:
        log("  (empty)")
        return
    
    for key, val in batch.non_tensor_batch.items():
        log(f"\n  {key}:")
        log(f"    Type: {type(val)}, Shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
        if hasattr(val, '__len__') and len(val) > 0:
            for i in range(min(max_items, len(val))):
                item_str = str(val[i])[:200]
                log(f"    [{i}]: {item_str}...")


def visualize_verification_scores(batch: DataProto, scores: list, tokenizer, train_batch_size: int = 4, 
                                   freeze_generation: bool = False, max_prompt_chars: int = 200, 
                                   max_response_chars: int = 500):
    """
    Visualize verification scores for all prompt-response combinations in the batch.
    
    For freeze_generation mode, the batch contains all-to-all combinations of prompts and responses
    (batch_size^2 combinations from batch_size prompts x batch_size responses).
    
    Args:
        batch: DataProto with verified batch
        scores: List of verification scores
        tokenizer: Tokenizer for decoding
        train_batch_size: Original batch size before all-to-all combinations
        freeze_generation: Whether freeze_generation mode was used
        max_prompt_chars: Maximum characters to show for prompt
        max_response_chars: Maximum characters to show for response
    """
    log(f"\n{'#'*100}")
    log(f"# VERIFICATION SCORES VISUALIZATION")
    log(f"{'#'*100}")
    
    batch_size = len(batch)
    log(f"\n  Total samples in batch: {batch_size}")
    log(f"  Number of scores: {len(scores)}")
    
    if freeze_generation:
        n_prompts = train_batch_size
        n_responses = train_batch_size
        log(f"  Mode: freeze_generation (all-to-all combinations)")
        log(f"  Original prompts: {n_prompts}, Original responses: {n_responses}")
        log(f"  Expected combinations: {n_prompts * n_responses}")
    else:
        log(f"  Mode: standard generation")
    
    # ============ SCORE SUMMARY ============
    log(f"\n{'='*80}")
    log("SCORE SUMMARY")
    log(f"{'='*80}")
    
    if scores:
        import statistics
        log(f"  Mean score: {statistics.mean(scores):.4f}")
        log(f"  Std score:  {statistics.stdev(scores) if len(scores) > 1 else 0:.4f}")
        log(f"  Min score:  {min(scores):.4f}")
        log(f"  Max score:  {max(scores):.4f}")
        
        # Count correct/incorrect
        correct = sum(1 for s in scores if s > 0.5)
        incorrect = len(scores) - correct
        log(f"  Correct (score > 0.5): {correct}/{len(scores)} ({100*correct/len(scores):.1f}%)")
        log(f"  Incorrect (score <= 0.5): {incorrect}/{len(scores)} ({100*incorrect/len(scores):.1f}%)")
    
    # ============ SCORE MATRIX (for freeze_generation) ============
    if freeze_generation:
        log(f"\n{'='*80}")
        log("SCORE MATRIX (Prompt x Response)")
        log(f"{'='*80}")
        log("\n  Rows = Prompts (P0, P1, ...), Columns = Responses (R0, R1, ...)")
        log("  Each cell shows the score for combining that prompt with that response.")
        log("  Diagonal (Pi, Ri) = original prompt-response pairs.")
        
        # Build matrix
        n = train_batch_size
        header = f"  {'':>5}"
        for j in range(n):
            header += f" R{j:>5}"
        log(header)
        log(f"  {'-'*5}-" + "-"*(7*n))
        
        for i in range(n):
            row = f"  P{i:>3} |"
            for j in range(n):
                idx = i * n + j
                if idx < len(scores):
                    score = scores[idx]
                    marker = "*" if i == j else " "  # Mark diagonal
                    row += f" {score:5.2f}{marker}"
                else:
                    row += f"   N/A "
            log(row)
        
        log(f"\n  * = diagonal (original prompt-response pair)")
    
    # ============ DETAILED SCORES ============
    log(f"\n{'='*80}")
    log("DETAILED SCORES FOR EACH COMBINATION")
    log(f"{'='*80}")
    
    # Get tensors
    prompts = batch.batch.get("prompts")
    responses = batch.batch.get("responses")
    attention_mask = batch.batch.get("attention_mask")
    
    if prompts is None or responses is None:
        log("  ERROR: prompts or responses not found in batch")
        return
    
    prompt_length = prompts.shape[1]
    
    for idx in range(min(len(scores), batch_size)):
        score = scores[idx]
        
        if freeze_generation:
            prompt_idx = idx // train_batch_size
            response_idx = idx % train_batch_size
            is_diagonal = prompt_idx == response_idx
            diagonal_marker = " [DIAGONAL - original pair]" if is_diagonal else ""
        else:
            prompt_idx = idx
            response_idx = idx
            diagonal_marker = ""
        
        log(f"\n  --- Sample {idx}: Prompt {prompt_idx} + Response {response_idx}{diagonal_marker} ---")
        log(f"  Score: {score:.4f} ({'CORRECT' if score > 0.5 else 'INCORRECT'})")
        
        # Get ground truth
        if "reward_model" in batch.non_tensor_batch:
            gt = batch.non_tensor_batch["reward_model"][idx]
            if isinstance(gt, dict):
                gt = gt.get("ground_truth", "N/A")
            log(f"  Ground Truth: {str(gt)[:200]}...")
        
        # Get data source
        if "data_source" in batch.non_tensor_batch:
            data_source = batch.non_tensor_batch["data_source"][idx]
            log(f"  Data Source: {data_source}")
        
        # Decode prompt
        prompt_ids = prompts[idx]
        # Get attended prompt tokens only
        prompt_mask = attention_mask[idx][:prompt_length] if attention_mask is not None else torch.ones(prompt_length)
        attended_prompt_ids = prompt_ids[prompt_mask.bool()] if prompt_mask is not None else prompt_ids
        prompt_text = tokenizer.decode(attended_prompt_ids, skip_special_tokens=True)
        
        # Decode response
        response_ids = responses[idx]
        response_mask = attention_mask[idx][prompt_length:] if attention_mask is not None else torch.ones(len(response_ids))
        attended_response_ids = response_ids[response_mask.bool()] if response_mask is not None else response_ids
        response_text = tokenizer.decode(attended_response_ids, skip_special_tokens=True)
        
        log(f"\n  [PROMPT - truncated to {max_prompt_chars} chars]:")
        log("-" * 60)
        prompt_display = prompt_text[:max_prompt_chars] + ("..." if len(prompt_text) > max_prompt_chars else "")
        log(prompt_display)
        log("-" * 60)
        
        log(f"\n  [RESPONSE - truncated to {max_response_chars} chars]:")
        log("-" * 60)
        response_display = response_text[:max_response_chars] + ("..." if len(response_text) > max_response_chars else "")
        log(response_display)
        log("-" * 60)
    
    # ============ SCORE DISTRIBUTION ============
    log(f"\n{'='*80}")
    log("SCORE DISTRIBUTION")
    log(f"{'='*80}")
    
    # Create histogram buckets
    buckets = [0] * 10
    for score in scores:
        bucket_idx = min(int(score * 10), 9)
        buckets[bucket_idx] += 1
    
    max_count = max(buckets) if buckets else 1
    log("\n  Score Range   | Count | Histogram")
    log("  " + "-" * 50)
    for i in range(10):
        low = i / 10
        high = (i + 1) / 10
        count = buckets[i]
        bar_len = int(30 * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        log(f"  [{low:.1f} - {high:.1f}] | {count:5d} | {bar}")
    
    log(f"\n{'#'*100}\n")


def main():
    parser = argparse.ArgumentParser(description="Test data loading and batch preparation for PRIME trainer")
    
    # Data configuration
    parser.add_argument("--train_files", type=str, required=True, 
                        help="Path to training data file(s), comma-separated for multiple files")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (for tokenizer and generation config)")
    
    # Dataset parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size (default: 4)")
    parser.add_argument("--oversample_factor", type=float, default=1.0,
                        help="Oversample factor (default: 1.0)")
    parser.add_argument("--max_prompt_length", type=int, default=550,
                        help="Maximum prompt length (default: 550)")
    parser.add_argument("--max_response_length", type=int, default=1000,
                        help="Maximum response length (default: 1000)")
    parser.add_argument("--prompt_truncation", type=str, default="left", choices=["left", "right", "error"],
                        help="Prompt truncation strategy: 'left' (keep end), 'right' (keep start), 'error' (raise exception). Default: left")
    parser.add_argument("--response_truncation", type=str, default="right", choices=["left", "right", "error"],
                        help="Response truncation strategy: 'left' (keep end), 'right' (keep start), 'error' (raise exception). Default: right")
    parser.add_argument("--filter_overlong_prompts", action="store_true", default=True,
                        help="Filter out prompts longer than max_prompt_length (default: True)")
    parser.add_argument("--no_filter_overlong_prompts", dest="filter_overlong_prompts", action="store_false",
                        help="Disable filtering of overlong prompts (use with truncation='left' or 'right')")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Shuffle the dataset")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for shuffling (default: 1)")
    
    # Processing options
    parser.add_argument("--freeze_generation", action="store_true", default=False,
                        help="Use freeze generation mode (combine prompt with target)")
    parser.add_argument("--rollout_n", type=int, default=1,
                        help="Number of rollout samples per prompt (default: 1)")
    parser.add_argument("--reward_model_enable_train", action="store_true", default=False,
                        help="Enable reward model training data loading")
    
    # Output options
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to process (default: 1)")
    parser.add_argument("--visualize_items", type=int, default=2,
                        help="Number of items to visualize per batch (default: 2)")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="Trust remote code when loading tokenizer")
    parser.add_argument("--cache_dir", type=str, default="~/.cache/verl/rlhf",
                        help="Cache directory for datasets")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output file for saving results (default: None, print to stdout only)")
    parser.add_argument("--visualize_dataset_entries", type=int, default=3,
                        help="Number of individual dataset entries to visualize before batching (default: 3)")
    parser.add_argument("--run_verification", action="store_true", default=False,
                        help="Run verification step to compute scores for each prompt-response combination")
    parser.add_argument("--reward_manager", type=str, default="prime", choices=["prime", "naive", "batch"],
                        help="Reward manager type to use for verification (default: prime)")
    
    args = parser.parse_args()
    
    # Initialize global logger with output file
    global logger
    logger = Logger(args.output_file)
    if args.output_file:
        log(f"Saving output to: {args.output_file}")
    
    log(f"\n{'#'*80}")
    log("# DATA LOADING TEST SCRIPT")
    log(f"{'#'*80}")
    log(f"\nConfiguration:")
    for arg, value in vars(args).items():
        log(f"  {arg}: {value}")
    
    # Load tokenizer
    log(f"\n--- Loading tokenizer from {args.model_path} ---")
    local_model_path = copy_to_local(args.model_path)
    tokenizer = get_tokenizer(local_model_path, trust_remote_code=args.trust_remote_code)
    log(f"Tokenizer loaded: {type(tokenizer).__name__}")
    log(f"  Vocab size: {tokenizer.vocab_size}")
    log(f"  Pad token ID: {tokenizer.pad_token_id}")
    log(f"  EOS token ID: {tokenizer.eos_token_id}")
    
    # Get generation config for eos_token_id
    generation_config = get_generation_config(local_model_path, trust_remote_code=args.trust_remote_code)
    eos_token_id = generation_config.eos_token_id if generation_config is not None else tokenizer.eos_token_id
    pad_token_id = generation_config.pad_token_id if generation_config is not None else tokenizer.pad_token_id
    log(f"  Generation config EOS: {eos_token_id}")
    log(f"  Generation config PAD: {pad_token_id}")
    
    # Create dataset config (mimicking OmegaConf DictConfig)
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
            # Note: RLHFDataset uses single truncation for both prompt and response
            # We use prompt_truncation for the dataset, but store both for visualization
            self.truncation = args.prompt_truncation  # Used by RLHFDataset
            self.prompt_truncation = args.prompt_truncation
            self.response_truncation = args.response_truncation
            self.filter_overlong_prompts = args.filter_overlong_prompts
            # Use 1 worker to avoid nested multiprocessing issues
            # (daemon processes cannot spawn children)
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
    log(f"\n--- Creating dataset ---")
    log(f"  Train files: {train_files}")
    log(f"  Truncation strategy: prompt='{args.prompt_truncation}', response='{args.response_truncation}'")
    log(f"  Filter overlong prompts: {args.filter_overlong_prompts}")
    log(f"  Max prompt length: {args.max_prompt_length}")
    log(f"  Max response length: {args.max_response_length}")
    if not args.filter_overlong_prompts and args.prompt_truncation == "error":
        log("  WARNING: filter_overlong_prompts=False with truncation='error' may cause errors for long prompts!")
    dataset = RLHFDataset(
        data_files=train_files,
        tokenizer=tokenizer,
        config=data_config,
        processor=None
    )
    log(f"  Dataset size after filtering: {len(dataset)}")
    
    # Visualize individual dataset entries BEFORE batching
    log(f"\n--- Visualizing individual dataset entries (before batching) ---")
    num_entries_to_show = min(args.visualize_dataset_entries, len(dataset))
    log(f"  Showing {num_entries_to_show} entries from dataset")
    for entry_idx in range(num_entries_to_show):
        visualize_dataset_entry(
            dataset, tokenizer, idx=entry_idx,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            prompt_truncation=args.prompt_truncation,
            response_truncation=args.response_truncation
        )
    
    # Create dataloader
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    
    effective_batch_size = int(args.train_batch_size * args.oversample_factor)
    log(f"\n--- Creating dataloader ---")
    log(f"  Effective batch size: {effective_batch_size}")
    
    if args.shuffle:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        sampler = RandomSampler(data_source=dataset, generator=generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=effective_batch_size,
        num_workers=0,  # Use 0 for debugging
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    log(f"  Number of batches: {len(dataloader)}")
    
    # Process batches
    for batch_idx, batch_dict in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break
        
        log(f"\n{'*'*80}")
        log(f"* PROCESSING BATCH {batch_idx}")
        log(f"{'*'*80}")
        
        # Step 1: Create DataProto from batch dict
        log(f"\n--- Step 1: DataProto.from_single_dict ---")
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        log(f"  Batch size: {len(batch)}")
        log(f"  Batch keys: {list(batch.batch.keys())}")
        log(f"  Non-tensor keys: {list(batch.non_tensor_batch.keys())}")
        
        # Visualize initial batch
        for i in range(min(args.visualize_items, len(batch))):
            visualize_batch_item(batch, tokenizer, i, title="[Initial Batch]")
        visualize_non_tensor_batch(batch, max_items=args.visualize_items)
        
        # Step 2: Pop keys for generation
        log(f"\n--- Step 2: Pop keys for generation ---")
        if args.freeze_generation:
            pop_keys = ["input_ids", "attention_mask", "position_ids", 
                       "chosen_input_ids", "chosen_attention_mask", "chosen_position_ids"]
        else:
            pop_keys = ["input_ids", "attention_mask", "position_ids"]
        
        available_keys = [k for k in pop_keys if k in batch.batch.keys()]
        log(f"  Popping keys: {available_keys}")
        gen_batch = batch.pop(batch_keys=available_keys)
        log(f"  gen_batch keys: {list(gen_batch.batch.keys())}")
        log(f"  Remaining batch keys: {list(batch.batch.keys())}")
        
        # Visualize gen_batch
        for i in range(min(args.visualize_items, len(gen_batch))):
            visualize_batch_item(gen_batch, tokenizer, i, title="[gen_batch]")
        
        # Step 3: Generate or combine sequences
        log(f"\n--- Step 3: Generate sequences ---")
        if args.freeze_generation:
            # Check if required keys exist
            if "chosen_input_ids" not in gen_batch.batch.keys():
                log("  WARNING: freeze_generation=True but chosen_input_ids not found!")
                log("  Make sure reward_model_enable_train=True and data has reward_model field")
                gen_batch_output = gen_batch  # Fallback
            else:
                log("  Using freeze_generation mode - combining prompts with targets")
                gen_batch_output = combine_prompt_target_batch(
                    gen_batch, tokenizer, eos_token_id, pad_token_id
                )
                log(f"  gen_batch_output batch size: {len(gen_batch_output)}")
                log(f"  gen_batch_output keys: {list(gen_batch_output.batch.keys())}")
        else:
            # In actual training, this calls actor_rollout_wg.generate_sequences()
            # For testing, we just simulate with the prompt itself
            log("  Skipping actual generation (would call actor_rollout_wg.generate_sequences)")
            log("  Creating mock gen_batch_output for visualization")
            
            # Create a simple mock output (just copying prompts as "responses")
            mock_responses = gen_batch.batch["input_ids"][:, -50:]  # Take last 50 tokens as mock response
            mock_seq = torch.cat([gen_batch.batch["input_ids"], mock_responses], dim=-1)
            mock_attn = torch.cat([gen_batch.batch["attention_mask"], 
                                  torch.ones_like(mock_responses)], dim=-1)
            
            # Calculate position IDs
            response_length = mock_responses.size(1)
            delta_pos = torch.arange(1, response_length + 1, device=mock_responses.device)
            delta_pos = delta_pos.unsqueeze(0).repeat(len(gen_batch), 1)
            mock_pos = torch.cat([gen_batch.batch["position_ids"], 
                                 gen_batch.batch["position_ids"][:, -1:] + delta_pos], dim=-1)
            
            gen_batch_output = DataProto.from_dict(tensors={
                "prompts": gen_batch.batch["input_ids"],
                "responses": mock_responses,
                "input_ids": mock_seq,
                "attention_mask": mock_attn,
                "position_ids": mock_pos,
            })
        
        # Visualize gen_batch_output
        for i in range(min(args.visualize_items, len(gen_batch_output))):
            visualize_batch_item(gen_batch_output, tokenizer, i, title="[gen_batch_output]")
        
        # Step 4: Add UID and repeat batch
        log(f"\n--- Step 4: Add UID and repeat batch ---")
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
        )
        log(f"  Added {len(batch)} UIDs")
        
        if args.freeze_generation:
            repeat_factor = int(args.train_batch_size * args.oversample_factor)
        else:
            repeat_factor = args.rollout_n
        
        log(f"  Repeat factor: {repeat_factor}")
        batch = batch.repeat(repeat_times=repeat_factor, interleave=True)
        log(f"  Batch size after repeat: {len(batch)}")
        
        # Step 5: Union batch with gen_batch_output
        log(f"\n--- Step 5: Union batch with gen_batch_output ---")
        log(f"  batch size before union: {len(batch)}")
        log(f"  batch keys before union: {list(batch.batch.keys())}")
        log(f"  gen_batch_output size: {len(gen_batch_output)}")
        log(f"  gen_batch_output keys: {list(gen_batch_output.batch.keys())}")
        
        # Handle size mismatch for freeze_generation mode
        if len(batch) != len(gen_batch_output):
            log(f"  WARNING: Size mismatch! batch={len(batch)}, gen_batch_output={len(gen_batch_output)}")
            # Truncate or expand to match
            min_size = min(len(batch), len(gen_batch_output))
            batch = batch[:min_size]
            gen_batch_output = gen_batch_output[:min_size]
            log(f"  Truncated to size: {min_size}")
        
        batch = batch.union(gen_batch_output)
        log(f"  Batch size after union: {len(batch)}")
        log(f"  Batch keys after union: {list(batch.batch.keys())}")
        
        # Final visualization
        log(f"\n--- FINAL BATCH VISUALIZATION ---")
        for i in range(min(args.visualize_items, len(batch))):
            visualize_batch_item(batch, tokenizer, i, title="[FINAL]")
        
        visualize_non_tensor_batch(batch, max_items=args.visualize_items)
        
        # Step 6: Run verification if enabled
        if args.run_verification:
            log(f"\n--- Step 6: Verification ---")
            log(f"  Running verification to compute scores...")
            
            # Check required fields
            required_keys = ["prompts", "responses", "attention_mask"]
            missing_keys = [k for k in required_keys if k not in batch.batch.keys()]
            if missing_keys:
                log(f"  ERROR: Missing required keys for verification: {missing_keys}")
                log(f"  Available keys: {list(batch.batch.keys())}")
            else:
                # Check non-tensor batch fields
                if "reward_model" not in batch.non_tensor_batch:
                    log(f"  ERROR: 'reward_model' not found in non_tensor_batch")
                    log(f"  Available non-tensor keys: {list(batch.non_tensor_batch.keys())}")
                    log(f"  Make sure reward_model_enable_train=True")
                elif "data_source" not in batch.non_tensor_batch:
                    log(f"  ERROR: 'data_source' not found in non_tensor_batch")
                    log(f"  Available non-tensor keys: {list(batch.non_tensor_batch.keys())}")
                else:
                    # Create reward manager
                    log(f"  Creating {args.reward_manager} reward manager...")
                    
                    if args.reward_manager == "prime":
                        from verl.workers.reward_manager import PrimeRewardManager
                        reward_fn = PrimeRewardManager(
                            tokenizer=tokenizer,
                            num_examine=0,
                            compute_score=_default_compute_score,
                            reward_fn_key="data_source"
                        )
                    elif args.reward_manager == "naive":
                        from verl.workers.reward_manager import NaiveRewardManager
                        reward_fn = NaiveRewardManager(
                            tokenizer=tokenizer,
                            num_examine=0,
                            compute_score=_default_compute_score,
                            reward_fn_key="data_source"
                        )
                    elif args.reward_manager == "batch":
                        from verl.workers.reward_manager import BatchRewardManager
                        reward_fn = BatchRewardManager(
                            tokenizer=tokenizer,
                            num_examine=0,
                            compute_score=_default_compute_score,
                            reward_fn_key="data_source"
                        )
                    
                    log(f"  Reward manager created: {type(reward_fn).__name__}")
                    
                    # Run verification
                    try:
                        log(f"  Calling verify()...")
                        scores = reward_fn.verify(batch)
                        log(f"  Verification complete. Got {len(scores)} scores.")
                        
                        # Visualize scores
                        visualize_verification_scores(
                            batch=batch,
                            scores=scores,
                            tokenizer=tokenizer,
                            train_batch_size=args.train_batch_size,
                            freeze_generation=args.freeze_generation,
                            max_prompt_chars=300,
                            max_response_chars=800
                        )
                        
                        # Store scores in batch for reference
                        if "acc" in batch.batch.keys():
                            log(f"\n  Scores stored in batch.batch['acc']: {batch.batch['acc'].tolist()}")
                        
                    except Exception as e:
                        log(f"  ERROR during verification: {e}")
                        import traceback
                        log(f"  Traceback:\n{traceback.format_exc()}")
        
        # Summary statistics
        log(f"\n{'='*80}")
        log("SUMMARY STATISTICS")
        log(f"{'='*80}")
        log(f"  Final batch size: {len(batch)}")
        log(f"  Tensor keys: {list(batch.batch.keys())}")
        log(f"  Non-tensor keys: {list(batch.non_tensor_batch.keys())}")
        
        if "attention_mask" in batch.batch.keys():
            attn_mask = batch.batch["attention_mask"]
            attended_per_sample = attn_mask.sum(dim=-1).float()
            log(f"  Attended tokens per sample:")
            log(f"    Mean: {attended_per_sample.mean().item():.1f}")
            log(f"    Min: {attended_per_sample.min().item():.0f}")
            log(f"    Max: {attended_per_sample.max().item():.0f}")
        
        if "responses" in batch.batch.keys():
            responses = batch.batch["responses"]
            log(f"  Response shape: {responses.shape}")
        
        if "prompts" in batch.batch.keys():
            prompts = batch.batch["prompts"]
            log(f"  Prompts shape: {prompts.shape}")
    
    log(f"\n{'#'*80}")
    log("# TEST COMPLETED SUCCESSFULLY")
    log(f"{'#'*80}\n")
    
    # Close the logger and file handle
    logger.close()
    
    if args.output_file:
        builtins.print(f"\n>>> Output saved to: {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()

