# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os
import torch
import torch.distributed
from tensordict import TensorDict
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.utils.debug import GPUMemoryLogger
from .prime_core_algos import compute_bt_loss, compute_bt_loss_weighted,compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm

__all__ = ["DataParallelPRIMERewardModel"]

#logger = logging.getLogger(__file__)
#logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DataParallelPRIMERewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        
        # Get the proper device for this worker
        # import os
        # self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # self.device = torch.device(f"cuda:{self.local_rank}")
        
        # # Set the device for this process
        # torch.cuda.set_device(self.local_rank)
        
        # print(f"Worker using device: {self.device} (local_rank: {self.local_rank})")
        
        # Debug: Check parameter dtypes right after initialization
        # print("=== REWARD MODEL PARAMETER DTYPES AFTER INIT ===")
        # for i, (name, param) in enumerate(self.reward_module.named_parameters()):
        #     print(f"Param {i}: {name} -> {param.dtype}, device={param.device}")
        #     if i >= 10:  # Just first few to avoid spam
        #         print("... (showing first 11 params)")
        #         break

    def _forward_micro_batch(self, micro_batch, prompt_length):
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        num_actions = micro_batch["input_ids"].shape[-1] - prompt_length
        max_positions = micro_batch["attention_mask"][:, prompt_length:].sum(-1)

        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            rm_output_logits = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False).logits.squeeze(0)  # copied. I don't really know why there is a squeeze
            rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)
            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]

        else:
            rm_output_logits = self.reward_module(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                use_cache=False,
            ).logits
            rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
            rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)

        if self.ref_module is not None:
            # do not have to pad again
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output_logits = self.ref_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False).logits.squeeze(0)
                    ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)
                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]
                else:
                    ref_output_logits = self.ref_module(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                        use_cache=False,
                    ).logits
                    torch.cuda.empty_cache()
                    ref_log_prob = torch.nn.functional.log_softmax(ref_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
                    ref_log_labels = ref_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)
        else:
            ref_log_labels = micro_batch["old_log_probs"]

        ref_log_labels.to(rm_log_labels.dtype)
        q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]  # this is actually diff of q

        # trim unnecessary logprobs here
        for i in range(micro_batch["input_ids"].shape[0]):
            q[i, max_positions[i] :] = 0

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            # generalized estimation of r should go before the reward filling. r means process reward for policy model, or the advantage of reward model.
            lam = self.config.get("lambda", 0.0)
            beta = self.config.model.get("beta_train", 0.05)
            if lam == 0.0:
                r = q * beta
            else:
                # reward coefficient takes no effect here
                acc = micro_batch["acc"]
                q_ = q * beta
                r = torch.zeros_like(q)
                lastgaelam = 0
                # change the last token and mask out all paddings to make this process easier if we rely on outcome reward to calculate V
                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, : max_positions[i] - 1].sum()
                    q_[i, max_positions[i] :] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            token_level_score = torch.zeros_like(q)

            if self.config.prime_granularity == "token":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, : max_positions[i] - 1] = r[i, : max_positions[i] - 1]
            elif self.config.prime_granularity == "whole":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, max_positions[i] - 1] = r[i, : max_positions[i]]
            else:
                raise NotImplementedError

        return token_level_score, q

    # def _forward_micro_batch_v2(self, micro_batch, require_grad=False):
    #     from contextlib import ExitStack
        

    #     # Use ExitStack to cleanly handle multiple context managers
    #     with ExitStack() as stack:
    #         # Always use autocast for mixed precision
    #         stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.bfloat16))
            
    #         # Only use no_grad when we don't need gradients (evaluation mode)
    #         if not require_grad:
    #             stack.enter_context(torch.no_grad())
            
    #         return self._forward_micro_batch_v2_inner(micro_batch)
    
    def _forward_micro_batch_v2(self, micro_batch):
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input

        from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        
            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.reward_module(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False)  # prevent model thinks we are generating
            reward_rmpad = output.logits
            reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

            # gather output if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                reward_rmpad = gather_outpus_and_unpad(reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            # pad it back
            rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            #rm_score = pad_input(reward_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
        else:
            output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
            rm_score = output.logits  # (batch_size, seq_len, 1)
            rm_score = rm_score.squeeze(-1)

        # extract the result of the last valid token
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
        return rm_score

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        # Debug: Check parameter dtypes before optimizer step
        # print("=== PARAMETER DTYPES BEFORE OPTIMIZER STEP ===")
        # for i, (name, param) in enumerate(self.reward_module.named_parameters()):
        #     if param.grad is not None:
        #         print(f"Param {i}: {name} -> param.dtype={param.dtype}, grad.dtype={param.grad.dtype}")
        #     else:
        #         print(f"Param {i}: {name} -> param.dtype={param.dtype}, grad=None")
        #     if i >= 10:  # Just first few to avoid spam
        #         print("... (showing first 11 params)")
        #         break


        
        # Check optimizer state devices
        # for group_idx, param_group in enumerate(self.reward_optimizer.param_groups):
        #     for param_idx, param in enumerate(param_group['params']):
        #         if param.grad is not None:
        #             print(f"DEBUG OPTIMIZER: Group {group_idx}, Param {param_idx}: param.device={param.device}, grad.device={param.grad.device}")
        #             # Check if this parameter has optimizer state
        #             if param in self.reward_optimizer.state:
        #                 state = self.reward_optimizer.state[param]
        #                 for state_name, state_tensor in state.items():
        #                     if torch.is_tensor(state_tensor):
        #                         print(f"DEBUG OPTIMIZER: Group {group_idx}, Param {param_idx}, State {state_name}: {state_tensor.device}")
        #             if param_idx >= 4:  # Only print first few to avoid spam
        #                 print(f"... (showing first 5 params per group)")
        #                 break

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        
        # Check for non-finite gradients like actor/critic models do
        if not torch.isfinite(grad_norm):
            print(f"WARN: reward model grad_norm is not finite: {grad_norm}")
            self.reward_optimizer.zero_grad()
        else:
            # # Debug: Check parameter and gradient dtypes
            # print("DEBUG DTYPE CHECK:")
            # for group_idx, group in enumerate(self.reward_optimizer.param_groups):
            #     for param_idx, param in enumerate(group["params"]):
            #         if param.grad is not None:
            #             print(f"Group {group_idx}, Param {param_idx}: param.dtype={param.dtype}, grad.dtype={param.grad.dtype}")
            #         else:
            #             print(f"Group {group_idx}, Param {param_idx}: param.dtype={param.dtype}, grad=None")
            #         if param_idx >= 5:  # Only print first few to avoid spam
            #             print(f"... (showing first 6 params per group)")
            #             break
            self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        if self.config.prime_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "acc"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )
    
    #@GPUMemoryLogger(role="dp rm", logger=logger)
    def forward_rm(self, data: DataProto):
        """
        Forward pass only - no gradient calculation.
        
        Args:
            data: DataProto object containing the input data
            
        Returns:
            dict containing:
            - reward_tensor: torch.Tensor of shape (batch_size, max_response_length) 
            with rm scores only at the last token position
            - reward_extra_info: defaultdict(list) with rm_score and acc lists
        """
        # make sure we are in eval mode for forward pass
        self.reward_module.eval()
        
        from collections import defaultdict
        reward_extra_info = defaultdict(list)
        
        # Select the keys we need
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts"]
        batch = data.select(batch_keys=select_keys).batch
        
        # Get batch dimensions
        batch_size = batch["input_ids"].shape[0]
        max_response_length = batch["responses"].shape[1]
        
        # Initialize reward tensor with zeros
        reward_tensor = torch.zeros((batch_size, max_response_length), dtype=torch.float32, device=batch["input_ids"].device)
        
        # Get attention mask to find valid response lengths
        attention_mask = batch["attention_mask"]
        prompt_length = batch["prompts"].shape[1]
        valid_response_lengths = attention_mask[:, prompt_length:].sum(dim=-1)
        
        # Process in mini-batches to handle memory efficiently
        dataloader = batch.split(self.config.mini_batch_size)
        
        batch_idx = 0  # Track position in full batch
        
        for mini_batch in dataloader:
            mini_batch_size = len(mini_batch["acc"])
            
            # Get RM scores for this mini-batch
            with torch.no_grad():  # No gradient computation
                rm_scores = self._forward_micro_batch_v2(mini_batch)
            
            # Compute a Bradley–Terry style minibatch loss: 1 pos vs all negs
            accs = mini_batch["acc"]
            pos_idx = torch.argmax(accs)
            neg_mask = torch.arange(mini_batch_size, device=accs.device) != pos_idx
            neg_scores = rm_scores[neg_mask]
            if neg_scores.numel() > 0:
                logits = rm_scores[pos_idx] - neg_scores
                bt_loss_mb = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.ones_like(logits), reduction="mean"
                )
            else:
                bt_loss_mb = torch.tensor(0.0, device=rm_scores.device)
            
            # Minibatch-level averages for reporting
            avg_pos_rm_score_mb = rm_scores[pos_idx].item()
            avg_neg_rm_score_mb = neg_scores.mean().item() if neg_scores.numel() > 0 else 0.0
            
            # Extract the last token score for each sequence in this mini-batch
            for i in range(mini_batch_size):
                # Get the valid response length for this sample
                valid_len = valid_response_lengths[batch_idx + i].item()
                
                # Get the RM score for this sequence (already last-token score)
                if valid_len > 0:
                    last_token_score = rm_scores[i]
                    # Store in reward tensor at the last token position
                    reward_tensor[batch_idx + i, valid_len - 1] = last_token_score
                else:
                    last_token_score = torch.tensor(0.0, device=rm_scores.device)
                
                # Add to reward_extra_info
                reward_extra_info["rm_score"].append(last_token_score.item())
                reward_extra_info["acc"].append(mini_batch["acc"][i].item())
                reward_extra_info["bt_loss"].append(bt_loss_mb.item())
                reward_extra_info["avg_pos_rm_score"].append(avg_pos_rm_score_mb)
                reward_extra_info["avg_neg_rm_score"].append(avg_neg_rm_score_mb)
            
            batch_idx += mini_batch_size
        
        # Create result dict
        result = {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info
        }
        
        return result

    #@GPUMemoryLogger(role="dp rm", logger=logger)
    def update_rm(self, data: DataProto):
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts"]

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        # ---- SPECIAL CASE: Bradley–Terry two-pass ----
        if self.config.model.loss_type == "bt":
            print(f"DEBUG: batch.shape: {batch.shape}")

            rm_scores_lst = []
            pos_scores_for_logging = []
            
            # — PASS 1: collect all rm_scores & accs (no grad) — 
            dataloader = batch.split(self.config.mini_batch_size)
            for batch_idx, mini_batch in enumerate(dataloader):
                accs = mini_batch["acc"]
                print(f"DEBUG: accs.shape: {accs.shape}")
                #print(f"DEBUG: accs: {accs}")
                print(f"DEBUG: accs.min(): {accs.min()}, accs.max(): {accs.max()}")
                top1_acc, top1_idx = torch.topk(accs, k=1, largest=True)        # [1]
                print(f"DEBUG: top1_acc: {top1_acc}, top1_idx: {top1_idx}")
                top1_idx = top1_idx.item()
                # Use TensorDict indexing directly to get a single example
                pos_example = mini_batch[top1_idx:top1_idx+1].to(torch.cuda.current_device())  # Use consistent device
                top1_acc = top1_acc.to(torch.cuda.current_device())  # Use consistent device
                print(f"DEBUG BT PASS1: pos_example input_ids shape: {pos_example['input_ids'].shape}")
                print(f"DEBUG BT: pos_example device: {pos_example['input_ids'].device}")

                ## New 06/27
                mini_batch_size = len(mini_batch["acc"])
                micro_size = self.config.micro_batch_size_per_gpu
                all_idxs = torch.arange(mini_batch_size, device=torch.cuda.current_device())
                neg_mask = all_idxs != top1_idx
                neg_idxs = all_idxs[neg_mask]               # shape [batch_size-1]
                # 3) chunk negatives into groups of (micro_size-1)
                step = micro_size - 1
                micro_batches = []
                for start in range(0, neg_idxs.size(0), step):
                    this_neg = neg_idxs[start : start + step]
                    # prepend the single positive idx
                    mb_idxs = torch.cat([all_idxs[top1_idx : top1_idx+1], this_neg], dim=0)
                    # 4) index your dict‐batch
                    mb = {k: v[mb_idxs].to(torch.cuda.current_device()) for k, v in mini_batch.items()}
                    mb_td = TensorDict(mb, batch_size=[mb_idxs.size(0)])
                    micro_batches.append(mb_td)
                 ## New 06/27
                #micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                total_accum = len(micro_batches) if len(micro_batches) > 0 else 1
                self.gradient_accumulation = total_accum
                self.reward_optimizer.zero_grad()
                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    print(f"DEBUG BT PASS2: iter {micro_batch_idx + 1}/{total_accum} start")
                    if os.getenv("VERL_DEBUG_MEM", "0") == "1":
                        try:
                            alloc = torch.cuda.memory_allocated()
                            reserved = torch.cuda.memory_reserved()
                            print(f"DEBUG BT PASS2: mem alloc={alloc/1e6:.1f}MB reserved={reserved/1e6:.1f}MB")
                        except Exception as e:
                            print(f"DEBUG BT PASS2: mem stats error: {e}")
                    micro_batch = micro_batch.to(torch.cuda.current_device())  # Use consistent device

                    # rm_pos_score = self._forward_micro_batch_v2(pos_example)
                    # print(f"DEBUG BT PASS2: rm_pos_score: {rm_pos_score}")
                    # print(f"DEBUG BT: rm_pos_score device: {rm_pos_score.device}")
                    rm_scores = self._forward_micro_batch_v2(micro_batch)
                    if os.getenv("VERL_DEBUG_SYNC", "0") == "1":
                        try:
                            torch.cuda.synchronize()
                            print("DEBUG BT PASS2: sync after forward")
                        except Exception as e:
                            print(f"DEBUG BT PASS2: sync after forward error: {e}")
                    rm_pos_score = rm_scores[0]
                    print(f"DEBUG BT PASS2: rm_pos_score: {rm_pos_score}")
                    rm_negs = rm_scores[1:]       # [B-1]
                    #print(f"DEBUG BT: rm_score device: {rm_score.device}")
                    
                    # Only append rm_pos_score at iteration 0, append rm_negs for other iterations
                    if micro_batch_idx == 0:
                        if rm_pos_score.dim() == 0:  # If it's a scalar, unsqueeze to make it 1D
                            rm_pos_score = rm_pos_score.unsqueeze(0)
                        rm_scores_lst.append(rm_pos_score.detach().cpu())  # Move to CPU immediately
                        pos_scores_for_logging.append(rm_pos_score.detach().float().cpu())
                    
                    # Only append rm_negs if it has elements
                    if rm_negs.numel() > 0:
                        rm_scores_lst.append(rm_negs.detach().cpu())  # Move to CPU immediately
                    # TODO: repeated positive scores
                    
                    # Determine positives and negatives in this micro-batch using same logic
                    # micro_accs = micro_batch["acc"]
                    # micro_neg_mask = micro_accs < top1_acc
                    # rm_negs = rm_score[micro_neg_mask]  # Convert to sequence-level and filter negatives
                    margin = self.config.model.get("margin", 0.0)
                    label_smoothing = self.config.model.get("label_smoothing", 0.0)
                    centering_coeff = self.config.model.get("centering_coeff", 0.0)
                    logits = None
                    if rm_negs.numel() == 0:
                        loss = (rm_pos_score * 0.0).sum()                # dummy zero loss
                    else:
                        logits = rm_pos_score - rm_negs - margin
                        print(f"DEBUG BT PASS2: logits.shape: {logits.shape}")
                        print(f"DEBUG BT: logits device: {logits.device}")
                        labels = torch.ones_like(logits) * (1.0 - label_smoothing)
                        bt_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        reduction="mean"
                    )
                        #print(f"DEBUG BT PASS2: loss: {loss}")
                        #print(f"DEBUG BT: loss device: {loss.device}")
                        if centering_coeff > 0.0:
                            centering_loss = centering_coeff * (rm_pos_score**2 + rm_negs**2).mean()
                        else:
                            centering_loss = torch.tensor(0.0, device=rm_pos_score.device)
                        loss = bt_loss + centering_loss
                    loss_metrics = {"reward_model/bt_loss": bt_loss.detach().item(), "reward_model/centering_loss": centering_loss.detach().item()}
                    # Scale loss same as normal case
                    loss = loss / self.gradient_accumulation
                    
                    loss.backward()
                    if os.getenv("VERL_DEBUG_SYNC", "0") == "1":
                        try:
                            torch.cuda.synchronize()
                            print("DEBUG BT PASS2: sync after backward")
                        except Exception as e:
                            print(f"DEBUG BT PASS2: sync after backward error: {e}")
                    append_to_dict(metrics, loss_metrics)
                    
                    # Clear intermediate tensors to save memory
                    del micro_batch, rm_scores, rm_negs, loss
                    if logits is not None:
                        del logits
                    if os.getenv("VERL_EMPTY_CACHE_INNER", "0") == "1":
                        torch.cuda.empty_cache()

                # Optimizer step after each mini-batch, same as normal case
                
                grad_norm = self._optimizer_step()
                metrics["reward_model/grad_norm"] = grad_norm.detach().item()
                
                # Clean up after each mini-batch
                del pos_example, rm_pos_score
                torch.cuda.empty_cache()

            # rebuild outputs & metrics exactly as before
            # Apply prime_norm to the actual rm_scores (token-level scores)
            self.reward_optimizer.zero_grad()
            rm_scores = torch.cat([score.to(torch.cuda.current_device()) for score in rm_scores_lst], dim=0)  # Use consistent device
            #rm_scores = self.prime_norm(rm_scores)
            print(f"DEBUG BT FINAL: rm_scores.shape: {rm_scores.shape}")
            print(f"DEBUG BT FINAL: rm_scores device: {rm_scores.device}")
            #print(f"DEBUG BT FINAL: q.shape: {q.shape}")
            if len(pos_scores_for_logging) > 0:
                metrics["reward_model/positive_reward"] = torch.stack(pos_scores_for_logging).mean().item()
            metrics.update({
                "reward_model/reward": rm_scores.mean().item(),
                #"reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            })
            return rm_scores, metrics
        
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()
                attention_mask = data["attention_mask"]
                acc = data["acc"]

                prompt_ids = data["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(data, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "dpo":
                    # the implementation of dpo is actually detached, which means we have to know the average value of w/l reward before the update.
                    dpo_loss = compute_detach_dpo_loss_rm(q, acc, Q_bc=data["Q_bc"], acc_bc=data["acc_bc"], response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "bon_acc":
                    # change the original distribution of each sample to BoN distribution, then update reward model
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )
                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                data = {"reward_model/dpo_loss": dpo_loss.detach().item()}

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = dpo_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
