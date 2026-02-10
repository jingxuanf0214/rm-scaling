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
import logging
import os
import warnings
import time

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from .prime_core_algos import compute_dpo_abs_accuracy, compute_dpo_accuracy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def set_random_seed(seed):
    """Set random seed for reproducible model initialization."""
    import random
    import numpy as np
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


class PRIMERewardModelWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
            assert self.config.mini_batch_size % self.config.micro_batch_size_per_gpu == 0

    def _build_reward_ref_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_local_path_from_hdfs(config.model.path)

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.tokenizer.add_special_tokens({"pad_token": "<extra_0>"})

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Reward model overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM

        trust_remote_code = False
        reward_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        reward_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not reward_model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            reward_model_config.hidden_dropout = "0"
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if config.model.get("use_remove_padding", False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            # some parameters may not in torch_dtype
            reward_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # Resize token embeddings OUTSIDE the init context to avoid meta tensor issues
        reward_module.resize_token_embeddings(len(self.tokenizer))
        
        if self.rank == 0:
            print_model_size(reward_module)

        self.reward_model_config = reward_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage("Before reward model FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            reward_model_config.hidden_dropout = "0"
            ref_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=copy_local_path_from_hdfs(config.model.ref_path),
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype
            ref_module.to(torch_dtype)
        
        # Resize token embeddings OUTSIDE the init context to avoid meta tensor issues  
        ref_module.resize_token_embeddings(len(self.tokenizer))

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After reward FSDP", logger=None)

        ref_module = FSDP(
            ref_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        reward_optimizer = optim.AdamW(
            reward_module.parameters(),
            lr=config.model.optim.lr,
            betas=config.model.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.model.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.model.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.model.optim.get("lr_warmup_steps", -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.model.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup

        reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=reward_optimizer, num_warmup_steps=num_warmup_steps)

        return reward_module, ref_module, reward_optimizer, reward_lr_scheduler

    def _build_reward_ref_model_optimizer_v2(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        # Set random seed for reproducible model initialization
        seed = config.model.get("seed", 42)  # Default seed is 42 if not specified
        set_random_seed(seed)
        if self.rank == 0:
            print(f"Setting random seed to {seed} for reproducible model initialization")

        local_path = copy_local_path_from_hdfs(config.model.path)

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.tokenizer.add_special_tokens({"pad_token": "<extra_0>"})

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Reward model overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification

        trust_remote_code = config.model.get("trust_remote_code", False)
        start_from_rm = bool(config.model.get("start_from_rm", False))
        reward_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        reward_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not reward_model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            #reward_model_config.hidden_dropout = "0"
            reward_module, loading_info = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
                output_loading_info=True,
            )
            if start_from_rm:
                missing_keys = list(loading_info.get("missing_keys", []) or [])
                unexpected_keys = list(loading_info.get("unexpected_keys", []) or [])
                mismatched_keys = list(loading_info.get("mismatched_keys", []) or [])
                if missing_keys or unexpected_keys or mismatched_keys:
                    raise RuntimeError(
                        "start_from_rm=True requires a strict HF RM checkpoint load, but got incompatible weights.\n"
                        f"- reward_model.model.path: {local_path}\n"
                        f"- missing_keys (first 20): {missing_keys[:20]}\n"
                        f"- unexpected_keys (first 20): {unexpected_keys[:20]}\n"
                        f"- mismatched_keys (first 20): {mismatched_keys[:20]}\n"
                        "If you intended to initialize the RM from a base LM checkpoint (randomly init the head), set "
                        "reward_model.model.start_from_rm=False."
                    )
            elif self.rank == 0:
                # In LM-init mode, missing head weights are expected; keep behavior but make it explicit in logs.
                missing_keys = list(loading_info.get("missing_keys", []) or [])
                unexpected_keys = list(loading_info.get("unexpected_keys", []) or [])
                mismatched_keys = list(loading_info.get("mismatched_keys", []) or [])
                if missing_keys or unexpected_keys or mismatched_keys:
                    print(
                        "[RM-INIT] start_from_rm=False (LM-init mode): non-strict load.\n"
                        f"- reward_model.model.path: {local_path}\n"
                        f"- missing_keys (first 20): {missing_keys[:20]}\n"
                        f"- unexpected_keys (first 20): {unexpected_keys[:20]}\n"
                        f"- mismatched_keys (first 20): {mismatched_keys[:20]}"
                    )

            if config.model.get("use_remove_padding", False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            # some parameters may not in torch_dtype
            reward_module.to(torch_dtype)
        
        # Resize token embeddings OUTSIDE the init context to avoid meta tensor issues
        # This must be done after the model is materialized on actual devices
        reward_module.resize_token_embeddings(len(self.tokenizer),mean_resizing=False)
        
        # IMPORTANT: Update config.architectures to reflect that this is now a TokenClassification model
        # This ensures the checkpoint manager saves it correctly as ForTokenClassification
        if hasattr(reward_module.config, 'architectures') and reward_module.config.architectures:
            original_arch = reward_module.config.architectures[0]
            # Convert from CausalLM to TokenClassification architecture
            if "ForCausalLM" in original_arch:
                new_arch = original_arch.replace("ForCausalLM", "ForTokenClassification")
                reward_module.config.architectures = [new_arch]
                if self.rank == 0:
                    print(f"Updated model architecture from {original_arch} to {new_arch}")
        
        print(f"Reward model config: {reward_module.config}")

        if config.model.get("enable_gradient_checkpointing", False):
            reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(reward_module)

        self.reward_model_config = reward_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage("Before reward model FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After reward FSDP", logger=None)

        reward_optimizer = optim.AdamW(
            reward_module.parameters(),
            lr=config.model.optim.lr,
            betas=config.model.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.model.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.model.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.model.optim.get("lr_warmup_steps", -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.model.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        warmup_style = config.model.optim.get("warmup_style", "constant")
        if warmup_style == "constant":
            reward_lr_scheduler = get_constant_schedule_with_warmup(optimizer=reward_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            min_lr_ratio = config.model.optim.get("min_lr_ratio", 0.0)
            reward_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=reward_optimizer, 
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio
            )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported for reward model")

        return reward_module, None, reward_optimizer, reward_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from .prime_dp_rm import DataParallelPRIMERewardModel

        #self.reward_module, self.ref_module, self.reward_optimizer, self.reward_lr_scheduler = self._build_reward_ref_model_optimizer(config=self.config)
        self.reward_module, self.ref_module, self.reward_optimizer, self.reward_lr_scheduler = self._build_reward_ref_model_optimizer_v2(config=self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            #offload_fsdp_model_to_cpu(self.ref_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        self.rm = DataParallelPRIMERewardModel(
            config=self.config,
            reward_module=self.reward_module,
            ref_module=self.ref_module,
            reward_optimizer=self.reward_optimizer,
        )

        self.flops_counter = FlopsCounter(self.reward_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.reward_module,
            optimizer=self.reward_optimizer,
            lr_scheduler=self.reward_lr_scheduler,
            tokenizer=self.tokenizer,
        )

        #Save initial checkpoint after model initialization
        if self.config.model.get("save_initial_checkpoint", False):
            initial_ckpt_path = self.config.model.get("initial_checkpoint_path", "./initial_checkpoint")
            
            # Create the checkpoint directory path with step 0
            import os
            step_0_path = os.path.join(initial_ckpt_path, "global_step_0")
            
            # Only save if this is a fresh start (checkpoint folder missing or empty)
            if (not os.path.exists(step_0_path)) or (os.path.isdir(step_0_path) and len(os.listdir(step_0_path)) == 0):
                if self.rank == 0:
                    print(f"Fresh start detected - saving initial checkpoint to: {initial_ckpt_path}")
                
                self.save_checkpoint(
                    local_path=step_0_path,
                    hdfs_path=None,  # Set to appropriate HDFS path if needed
                    global_step=0,
                    max_ckpt_to_keep=None
                )
                
                if self.rank == 0:
                    print(f"Initial checkpoint saved successfully at step 0")
            else:
                if self.rank == 0:
                    print(f"Resuming from existing training - non-empty checkpoint folder found at {step_0_path}, skipping save")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data = data.to("cuda")

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
            load_fsdp_model_to_gpu(self.ref_module)
        micro_batch_size = self.config.micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            rm_scores, q, metrics = self.rm.compute_rm_score(data=data)

            prompt_length = data.batch["prompts"].shape[-1]
            response_mask = data.batch["attention_mask"][:, prompt_length:]
            acc = data.batch["acc"]

            #dpo_acc = compute_dpo_accuracy(rm_scores, acc, response_mask=response_mask, n_samples=data.meta_info["n"])
            #dpo_acc_abs = compute_dpo_abs_accuracy(rm_scores, acc, response_mask, n_samples=data.meta_info["n"])

            #metrics["reward_model/dpo_acc"] = dpo_acc.detach().item()
            #metrics["reward_model/dpo_acc_abs"] = dpo_acc_abs.detach().item()

            output = DataProto.from_dict(tensors={"rm_scores": rm_scores, "q": q}, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            offload_fsdp_model_to_cpu(self.ref_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_rm(self, data: DataProto):
        data = data.to("cuda")
        if self._is_offload_param:
            #load_fsdp_model_to_gpu(self.ref_module)
            load_fsdp_model_to_gpu(self.reward_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.reward_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            rm_scores, metrics = self.rm.update_rm(data=data)

            self.reward_lr_scheduler.step()
            lr = self.reward_lr_scheduler.get_last_lr()[0]
            metrics["rm/lr"] = lr

            # prompt_length = data.batch["prompts"].shape[-1]
            # response_mask = data.batch["attention_mask"][:, prompt_length:]
            # acc = data.batch["acc"]

            #dpo_acc_before = compute_dpo_accuracy(rm_scores, acc, response_mask=response_mask, n_samples=data.meta_info["n"])
            #dpo_acc_abs = compute_dpo_abs_accuracy(rm_scores, acc, response_mask, n_samples=data.meta_info["n"])

            # metrics["reward_model/dpo_acc_before"] = dpo_acc_before.detach().item()
            # metrics["reward_model/dpo_acc_abs_before"] = dpo_acc_abs.detach().item()

            output = DataProto.from_dict(tensors={"rm_scores": rm_scores}, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            #offload_fsdp_model_to_cpu(self.ref_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def forward_rm(self, data: DataProto):
        data = data.to("cuda")
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # Call the underlying RM forward (returns a dict: {"reward_tensor", "reward_extra_info"})
            result = self.rm.forward_rm(data=data)
            
            # Postprocess only the tensor output to restore original partitioning/order if needed
            reward_tensor_dp = DataProto.from_dict(tensors={"reward_tensor": result["reward_tensor"]})
            reward_tensor_dp = self.ulysses_sharding_manager.postprocess_data(data=reward_tensor_dp)
            reward_tensor = reward_tensor_dp.batch["reward_tensor"]
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
        
        # Move tensor outputs to CPU for stable serialization
        reward_tensor = reward_tensor.to("cpu")
        
        # Package extras as non_tensors so dispatch/collect can handle DataProto
        non_tensors = {}
        extra = result.get("reward_extra_info", {}) if isinstance(result, dict) else {}
        for key, vals in extra.items():
            non_tensors[key] = vals
        
        output_dp = DataProto.from_dict(tensors={"reward_tensor": reward_tensor}, non_tensors=non_tensors)
        return output_dp

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch
        
        start_time = time.time()
        logger.info(f"[DEBUG] Starting save_checkpoint at step {global_step}")

        if self._is_offload_param:
            logger.info(f"[DEBUG] Loading FSDP model to GPU...")
            load_start = time.time()
            load_fsdp_model_to_gpu(self.reward_module)
            load_time = time.time() - load_start
            logger.info(f"[DEBUG] Loading FSDP model to GPU took {load_time:.2f}s")

        logger.info(f"[DEBUG] Starting checkpoint_manager.save_checkpoint...")
        save_start = time.time()
        self.checkpoint_manager.save_checkpoint(local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        save_time = time.time() - save_start
        logger.info(f"[DEBUG] checkpoint_manager.save_checkpoint took {save_time:.2f}s")

        logger.info(f"[DEBUG] Starting distributed barrier...")
        barrier_start = time.time()
        torch.distributed.barrier()
        barrier_time = time.time() - barrier_start
        logger.info(f"[DEBUG] distributed barrier took {barrier_time:.2f}s")
        
        if self._is_offload_param:
            logger.info(f"[DEBUG] Offloading FSDP model to CPU...")
            offload_start = time.time()
            offload_fsdp_model_to_cpu(self.reward_module)
            offload_time = time.time() - offload_start
            logger.info(f"[DEBUG] Offloading FSDP model to CPU took {offload_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"[DEBUG] Total save_checkpoint took {total_time:.2f}s")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, del_local_after_load=True):
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
        ## 09/26
        # Diagnostics: print distributed/NCCL/DTensor environment and probe DTensor collectives
        try:
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            backend = (dist.get_backend() if dist.is_initialized() else "<not-initialized>")
        except Exception:
            world_size, rank, backend = 1, 0, "<error>"

        try:
            import torch.cuda.nccl as nccl
            nccl_version = getattr(nccl, "version", lambda: "<no-func>")()
        except Exception as e:
            nccl_version = f"<error: {repr(e)}>"

        has_coalesced = hasattr(torch.ops._c10d_functional, "all_gather_into_tensor_coalesced")
        cuda_ok = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_ok else 0
        current_device = torch.cuda.current_device() if cuda_ok else None
        device_name = torch.cuda.get_device_name(current_device) if (cuda_ok and device_count > 0) else "<cpu>"

        print(
            f"[RM-DIAG] before_ckpt_load | rank={rank}/{world_size} backend={backend} "
            f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cudnn={getattr(torch.backends.cudnn, 'version', lambda: None)()} "
            f"cuda_available={cuda_ok} num_gpus={device_count} cur_device={current_device} dev_name={device_name} "
            f"nccl_available={getattr(torch.distributed, 'is_nccl_available', lambda: False)()} nccl_version={nccl_version} "
            f"has_allgather_coalesced={has_coalesced}"
        )

        # DTensor redistribute smoke test to proactively trigger the same path
        try:
            if cuda_ok and world_size >= 2:
                try:
                    from torch.distributed.device_mesh import init_device_mesh
                    from torch.distributed._tensor import distribute_tensor, Shard, Replicate
                    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
                    x = torch.arange(8, device="cuda") + (rank * 100)
                    dt = distribute_tensor(x, mesh, placements=[Shard(0)])
                    dt_rep = dt.redistribute(mesh, placements=[Replicate()])
                    if rank == 0:
                        print("[RM-DIAG] dtensor_redistribute: OK | local_shape=", dt_rep.to_local().shape)
                except Exception as e:
                    if rank == 0:
                        print("[RM-DIAG] dtensor_redistribute: ERROR |", repr(e))
            else:
                print(f"[RM-DIAG] dtensor_redistribute: SKIP | cuda={cuda_ok} world_size={world_size}")
        except Exception as e:
            print("[RM-DIAG] dtensor_diag_block: ERROR |", repr(e))
        ## 09/26
        self.checkpoint_manager.load_checkpoint(local_path=local_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
