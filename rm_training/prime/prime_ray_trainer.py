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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
import uuid
import time
import datetime
from copy import deepcopy
from pprint import pprint
from tensordict import TensorDict

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import _compute_response_info, compute_throughout_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType, _timer, reduce_metrics
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import get_generation_config
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from . import prime_core_algos
from typing import Optional
import json



def combine_prompt_target_batch(gen_batch: DataProto, config) -> DataProto:
    """
    Combine prompts with target responses (for freeze_generation mode).
    This creates all-to-all combinations of prompts and responses.
    
    The responses are converted from left-padded to right-padded to avoid
    a gap of padding tokens between prompt and response in the combined sequence.
    
    Result: [prompt_padding | prompt_content | response_content | response_padding]
    """
    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    generation_config = get_generation_config(local_path, trust_remote_code=False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=False)
    
    # Get pad_token_id with fallback to eos_token_id
    eos_token_id = generation_config.eos_token_id if generation_config is not None else tokenizer.eos_token_id
    pad_token_id = generation_config.pad_token_id if generation_config is not None else tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id if not isinstance(eos_token_id, list) else eos_token_id[0]
    
    prompt_ids      = gen_batch.batch["input_ids"]         # (B, Lp)
    prompt_mask     = gen_batch.batch["attention_mask"]    # (B, Lp)
    prompt_pos      = gen_batch.batch["position_ids"]      # (B, Lp)
    resp_ids        = gen_batch.batch["chosen_input_ids"]      # (Lr,) or (B, Lr)
    resp_mask       = gen_batch.batch["chosen_attention_mask"] # (Lr,) or (B, Lr)
    resp_pos        = gen_batch.batch["chosen_position_ids"]   # (Lr,) or (B, Lr)
    
    # Handle both single response and batched responses case
    if resp_ids.dim() == 1:
        resp_ids = resp_ids.unsqueeze(0)
        resp_mask = resp_mask.unsqueeze(0) 
        resp_pos = resp_pos.unsqueeze(0)

    batch_size = prompt_ids.size(0)
    print(f"batch_size: {batch_size}")
    n_responses = batch_size

    # Create all-to-all combinations
    # Repeat prompts n_responses times
    prompt_ids = prompt_ids.repeat_interleave(n_responses, dim=0)  # (B*R, Lp)
    prompt_mask = prompt_mask.repeat_interleave(n_responses, dim=0)  # (B*R, Lp) 
    prompt_pos = prompt_pos.repeat_interleave(n_responses, dim=0)  # (B*R, Lp)

    # Repeat each response batch_size times
    resp_ids = resp_ids.repeat(batch_size, 1)  # (B*R, Lr)
    resp_mask = resp_mask.repeat(batch_size, 1)  # (B*R, Lr)
    resp_pos = resp_pos.repeat(batch_size, 1)  # (B*R, Lr)
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

    # Print debug information
    print("\nDEBUG INFO:")
    print("Example position_ids:")
    print(position_ids[0])  # Print first example
    print("\nExample attention_mask:")
    print(attention_mask[0])  # Print first example
    print("\nExample input_ids decoded:")
    print("Prompt:", tokenizer.decode(prompt_ids[0]))
    print("Response:", tokenizer.decode(resp_ids[0]))
    print("Full sequence:", tokenizer.decode(seq[0]))
    print("\nExample input_ids (raw):")
    print(seq[0])
    print("\nShapes:")
    print(f"position_ids shape: {position_ids.shape}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"input_ids shape: {seq.shape}")

    # all the tp ranks should contain the same data here. data in all ranks are valid
    batch = TensorDict(
        {
            "prompts": prompt_ids,
            "responses": resp_ids,
            "input_ids": seq,  # here input_ids become the whole sentences
            # 'old_log_probs': log_probs, # we will recompute old log prob with actor
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    return DataProto(batch=batch)


def compute_advantage(data: DataProto, adv_estimator, config):
    if adv_estimator == "rloo":
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = prime_core_algos.compute_rloo_advantage_return(data, response_mask, config.actor_rollout_ref.rollout.n, config)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def compute_data_metrics(batch, use_critic=True):
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    # JF changes -> data batch 
    prompt_mask = batch.batch["prompt_attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["prompt_attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    # JF changes -> data batch 
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


class RayPRIMETrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            reward_fn,
            val_reward_fn,
        )

        self.use_critic = False

    def _validate_config(self):
        # Skip strict config validation during validate-only runs to allow minimal configs
        if getattr(self.config.trainer, "enable_actor", True) is False:
            print("[validate_config] Skipping config validation in validate-only mode.")
            return
        super()._validate_config()
        # TODO: Additional config checks can be added here

    def _create_dataloader(self, *args, **kwargs):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(data_files=self.config.data.train_files, tokenizer=self.tokenizer, config=self.config.data)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=int(self.config.data.train_batch_size * self.config.data.oversample_factor),
            num_workers=self.config.data.get("dataloader_num_workers", 8),  # Match base trainer default
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(data_files=self.config.data.val_files, tokenizer=self.tokenizer, config=self.config.data)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.get("val_batch_size", len(self.val_dataset)),  # Use configurable val_batch_size with fallback to full dataset
            num_workers=self.config.data.get("dataloader_num_workers", 8),  # Match base trainer default
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
            # Inject total_training_steps into reward model config as well
            if hasattr(self.config, 'reward_model') and self.config.reward_model.enable:
                self.config.reward_model.model.optim.total_training_steps = total_training_steps

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        print(f"local_global_step_folder: {local_global_step_folder}")
        
        # Save actor only if save_actor is True (default to True for backward compatibility)
        if self.config.trainer.enable_actor and self.config.trainer.get("save_actor", True):
            actor_local_path = os.path.join(local_global_step_folder, "actor")
            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                actor_remote_path,
                self.global_steps,
            )

        if self.use_rm:
            reward_local_path = os.path.join(local_global_step_folder, "reward")
            reward_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "reward")
            self.rm_wg.save_checkpoint(
                reward_local_path,
                reward_remote_path,
                self.global_steps,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        reward_path = os.path.join(global_step_folder, "reward")
        # load actor
        if self.config.trainer.enable_actor:
            self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load rm
        if self.use_rm:
            # Driver-side quick diagnostics before worker checkpoint load
            try:
                import torch, torch.distributed as dist
                print(
                    "[DRV-DIAG] before_rm_ckpt_load | "
                    f"torch={torch.__version__} cuda_runtime={torch.version.cuda} "
                    f"cuda_available={torch.cuda.is_available()} num_gpus={torch.cuda.device_count()} "
                    f"nccl_available={getattr(dist, 'is_nccl_available', lambda: False)()} "
                    f"has_allgather_coalesced={hasattr(torch.ops._c10d_functional, 'all_gather_into_tensor_coalesced')}"
                )
            except Exception as e:
                print("[DRV-DIAG] before_rm_ckpt_load: ERROR |", repr(e))
            self.rm_wg.load_checkpoint(reward_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            try:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
            except (AssertionError, KeyError) as e:
                print(f"Warning: Failed to load dataloader state from {dataloader_local_path}: {e}")
                print("Continuing training with fresh dataloader state...")
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to
        construct the PPO dataflow. The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                step_start_time = time.time()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[HEARTBEAT] {timestamp} - Starting step {self.global_steps} (epoch {epoch})")
                
                # Optional: Check worker health every 10 steps (can be configured)
                if self.global_steps % 10 == 0:
                    self._check_worker_health()
                
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # Print example of batch_dict structure at this step
                
                #print(batch.non_tensor_batch)

                # pop those keys for generation
                # JF changes -> data batch 
                if not self.config.trainer.get("freeze_generation", False):
                    gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                else:
                    gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids","chosen_input_ids","chosen_attention_mask","chosen_position_ids"])

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting generation phase")
                        if not self.config.trainer.get("freeze_generation", False):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            # When freeze_generation is True, we use the existing responses
                            # Add your transformation for gen_batch here
                            gen_batch_output = combine_prompt_target_batch(gen_batch, self.config)
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed generation phase")

                    if self.config.algorithm.adv_estimator == "remax":
                        with _timer("gen_max", timing_raw):
                            if not self.config.trainer.get("freeze_generation", False):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output
                            else:
                                # When freeze_generation is True, we skip baseline generation
                                batch.batch["reward_baselines"] = torch.zeros(batch.batch["responses"].shape[0], device=batch.batch["responses"].device)

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    # JF changes -> data batch 
                    # TODO: check if this is correct
                    if not self.config.trainer.get("freeze_generation", False):
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    else:
                        repeat_factor = int(self.config.data.train_batch_size * self.config.data.oversample_factor)
                        batch = batch.repeat(repeat_times=repeat_factor, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # verify
                    with _timer("verify", timing_raw):
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting verification phase")
                        print("Performing verify")
                        print(f"batch.batch['responses'].shape: {batch.batch['responses'].shape}")
                        scores = self.reward_fn.verify(batch)
                        metrics["acc/mean"] = statistics.mean(scores)
                        metrics["acc/std"] = statistics.stdev(scores)
                        metrics["acc/min"] = min(scores)
                        metrics["acc/max"] = max(scores)
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed verification phase")

                    # filter the batch. 1/oversample_factor samples will be kept.
                    # If there is a filter, prompts passing it will be prioritized.

                    batch = self.filter_and_downsample(scores, batch)
                    if self.config.trainer.get("freeze_generation", False):
                        n_samples = int(self.config.data.train_batch_size * self.config.data.oversample_factor)
                    else:
                        n_samples = int(self.config.actor_rollout_ref.rollout.n)
                    batch.meta_info["n"] = n_samples
                    #n_samples = self.config.actor_rollout_ref.rollout.n

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting old_log_prob computation")
                        if self.config.trainer.enable_actor:
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = compute_response_mask(batch)
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed old_log_prob computation")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting reference policy computation")
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed reference policy computation")

                    with _timer("adv", timing_raw):
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting advantage computation")
                        if self.use_rm:
                            update_style = self.config.reward_model.model.get("update", "none")
                            if update_style == "none":  # only run forward
                                print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting RM forward pass")
                                reward_output = self.rm_wg.compute_rm_score(batch)
                                print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed RM forward pass")
                            elif update_style == "after":  # update and directly return the reward
                                reward_output = self.rm_wg.update_rm(batch)
                            elif update_style == "before":  # update reward model, and then run forward
                                reward_output = self.rm_wg.update_rm(batch)
                                if "metrics" in reward_output.meta_info.keys():
                                    reward_output_metrics = reduce_metrics(reward_output.meta_info["metrics"])
                                    metrics.update(reward_output_metrics)

                                reward_output = self.rm_wg.compute_rm_score(batch)
                            elif update_style == "reverse":  # run forward to calculate statistics, then update reward model
                                reward_output = self.rm_wg.compute_rm_score(batch)
                                # broadcast q and acc tensor to each result
                                bc_td = DataProto.from_dict(
                                    tensors={
                                        "Q_bc": reward_output.batch["q"].sum(dim=-1).view(-1, n_samples).unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, n_samples),
                                        "acc_bc": batch.batch["acc"].view(-1, n_samples).unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, n_samples),
                                    }
                                )
                                batch = batch.union(bc_td)
                                reward_output = self.rm_wg.update_rm(batch)
                            else:
                                raise NotImplementedError
                            batch = batch.union(reward_output)
                            if "metrics" in reward_output.meta_info.keys():
                                reward_output_metrics = reduce_metrics(reward_output.meta_info["metrics"])
                                metrics.update(reward_output_metrics)

                        # compute advantages, executed on the driver process
                        if self.config.trainer.enable_actor:    
                            batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator, config=self.config)
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed advantage computation")

                    # update actor
                    with _timer("update_actor", timing_raw):
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Starting actor update")
                        if not self.config.trainer.get("freeze_actor", False):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        else:
                            actor_output = None
                        print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed actor update")
                    if actor_output is not None:
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                if self.config.trainer.enable_actor:   
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                step_duration = time.time() - step_start_time
                self.global_steps += 1
                print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')} - Completed step {self.global_steps-1} in {step_duration:.2f}s")
                print(f"DEBUG: global_steps: {self.global_steps}")

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return
    
    def validate_only(self, resume_from_path: Optional[str] = None, log: bool = True):
        """
        Load a checkpoint and run validation only, without any training.
        If resume_from_path is provided, it overrides the resume settings in config.
        Returns the metrics dict from validation.
        """
        from omegaconf import OmegaConf, open_dict
        from pprint import pprint
        from verl.utils.tracking import Tracking
        
        # if self.val_reward_fn is None:
        #     raise RuntimeError("val_reward_fn is not configured; cannot run validation.")
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        if resume_from_path is not None:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.trainer.resume_mode = "resume_path"
                self.config.trainer.resume_from_path = resume_from_path
        
        # Ensure dataloaders exist
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            self._create_dataloader()
        
        # Load checkpoint and validate
        self._load_checkpoint()
        val_metrics = self._validate_for_rm()
        pprint(f"Validation metrics: {val_metrics}")
        if log:
            logger.log(data=val_metrics, step=self.global_steps)
        # Save validation metrics to val.json in the evaluated checkpoint folder
        if resume_from_path is not None:
            ckpt_dir = resume_from_path
            if not os.path.isabs(ckpt_dir):
                ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)
        else:
            ckpt_dir = self.config.trainer.default_local_dir
            if not os.path.isabs(ckpt_dir):
                ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)
            ckpt_dir = os.path.join(ckpt_dir, f"global_step_{self.global_steps}")

        os.makedirs(ckpt_dir, exist_ok=True)

        json_path = os.path.join(ckpt_dir, "val.json")
        with open(json_path, "w") as f:
            json.dump(val_metrics, f, indent=2, sort_keys=True)
        return val_metrics

    def filter_and_downsample(self, scores, batch: DataProto):
        """
        downsample the batch according to oversample_factor
        samples passing the filters will be prioritized
        """
        if self.config.trainer.get("freeze_generation", False):
            n_samples = int(self.config.actor_rollout_ref.rollout.n)
        else:
            n_samples = int(self.config.data.train_batch_size * self.config.data.oversample_factor)
        
        # # Debug prints to understand the filtering
        # print(f"DEBUG FILTER: len(batch)={len(batch)}")
        # print(f"DEBUG FILTER: n_samples={n_samples}")
        # print(f"DEBUG FILTER: oversample_factor={self.config.data.oversample_factor}")
        # print(f"DEBUG FILTER: freeze_generation={self.config.trainer.get('freeze_generation', False)}")
        # print(f"DEBUG FILTER: rollout.n={self.config.actor_rollout_ref.rollout.n}")
        
        reward_matrix = torch.tensor(scores).reshape(-1, n_samples)

        filter_mask = torch.ones((reward_matrix.shape[0]), dtype=torch.bool)

        if self.config.data.filter_accuracy:
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            filter_mask[(acc_tensor > self.config.data.accuracy_upper_bound) | (acc_tensor < self.config.data.accuracy_lower_bound)] = False

        if self.config.data.filter_truncate:
            # JF changes -> data batch 
            length_matrix = batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1] :].sum(dim=-1).reshape(-1, n_samples)
            length_tensor = torch.max(length_matrix, dim=-1)[0]
            filter_mask[length_tensor >= self.config.data.max_response_length - 1] = False

        reorder_index = torch.argsort(filter_mask, descending=True)
        reorder_index = (reorder_index.unsqueeze(-1) * n_samples + torch.arange(0, n_samples).unsqueeze(0)).view(-1)
        
        final_size = int(len(batch) // self.config.data.oversample_factor)
        print(f"DEBUG FILTER: final_size = {len(batch)} // {self.config.data.oversample_factor} = {final_size}")
        
        batch.reorder(reorder_index[: final_size])  # this operation is inplace

        return batch

    def _check_worker_health(self):
        """Simple worker health check - logs worker status"""
        try:
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"[WORKER_HEALTH] {timestamp} - Checking worker health...")
            
            # Check actor rollout workers if they exist
            if hasattr(self, 'actor_rollout_wg') and self.actor_rollout_wg is not None:
                for i, worker in enumerate(self.actor_rollout_wg.workers):
                    try:
                        alive = self._is_ray_actor_alive(worker)
                        print(f"[WORKER_HEALTH] Actor worker {i}: {'ALIVE' if alive else 'DEAD/UNKNOWN'}")
                    except Exception as e:
                        print(f"[WORKER_HEALTH] Actor worker {i}: ERROR checking status - {e}")
            
            # Check reward model workers if they exist
            if hasattr(self, 'rm_wg') and self.rm_wg is not None:
                for i, worker in enumerate(self.rm_wg.workers):
                    try:
                        alive = self._is_ray_actor_alive(worker)
                        print(f"[WORKER_HEALTH] RM worker {i}: {'ALIVE' if alive else 'DEAD/UNKNOWN'}")
                    except Exception as e:
                        print(f"[WORKER_HEALTH] RM worker {i}: ERROR checking status - {e}")
                        
        except Exception as e:
            print(f"[WORKER_HEALTH] Error during health check: {e}")
    
    def _is_ray_actor_alive(self, worker):
        """
        Check if a Ray actor is alive using a simple ping method instead of experimental API.
        This avoids dependency on Ray dashboard/API server.
        """
        import ray
        try:
            # Try to get a simple property from the actor with a short timeout
            # If the actor is dead, this will raise an exception
            future = worker.__ray_ready__.remote()
            ray.get(future, timeout=5.0)  # 5 second timeout
            return True
        except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError, Exception):
            return False
