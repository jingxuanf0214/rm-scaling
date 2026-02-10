# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        # JF changes -> data batch 
        self.reward_model_key = config.get("reward_model_key", "reward_model")
        self.reward_model_enable_train = config.get("reward_model_enable_train", False)
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 550)
        self.max_response_length = config.get("max_response_length", 1000)
        self.return_raw_chat = config.get("return_raw_chat", False)
        # Support separate truncation for prompt and response, with fallback to single 'truncation' config
        default_truncation = config.get("truncation", "error")
        self.prompt_truncation = config.get("prompt_truncation", default_truncation)
        self.response_truncation = config.get("response_truncation", default_truncation)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            
            def get_prompt_length(doc):
                if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                else:
                    return len(doc[prompt_key])
            
            self.dataframe = self.dataframe.filter(
                lambda doc: get_prompt_length(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )
            if self.reward_model_enable_train:
                response_key = self.reward_model_key
                def filter_response_length(doc):
                    ground_truth = doc[response_key]["ground_truth"]
                    # Handle case where ground_truth is a list (take first element) or string
                    if isinstance(ground_truth, list):
                        ground_truth = ground_truth[0] if ground_truth else ""
                    # Use tokenizer directly on the string, not apply_chat_template
                    return len(tokenizer.encode(ground_truth, add_special_tokens=False)) <= self.max_response_length
                
                self.dataframe = self.dataframe.filter(
                    filter_response_length,
                    num_proc=self.num_workers,
                    desc=f"Filtering responses longer than {self.max_response_length} tokens",
                )
            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        # JF changes -> data batch 
        if self.reward_model_enable_train:
            chosen_answer = example.get(self.reward_model_key, {}).get("ground_truth", None)
        else:
            chosen_answer = None

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list
        # JF changes -> data batch 
        return messages, chosen_answer

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        # JF changes -> data batch 
        messages, chosen_answer = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            # Check if processor has chat_template, otherwise use direct encoding
            #if hasattr(self.processor, 'chat_template') and self.processor.chat_template is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            #else:
                # For processors without chat template, join messages directly
                #raw_prompt = " ".join([msg.get("content", "") for msg in messages if isinstance(msg, dict) and "content" in msg])
            
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            # Check if tokenizer has chat_template, otherwise use direct encoding
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                # For tokenizers without chat template, join messages directly
                raw_prompt = " ".join([msg.get("content", "") for msg in messages if isinstance(msg, dict) and "content" in msg])
            
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.prompt_truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Process chosen answer if it exists
        # JF changes -> data batch 
        if chosen_answer is not None:
            # Handle case where chosen_answer is a list (take first element) or string
            if isinstance(chosen_answer, list):
                chosen_answer = chosen_answer[0] if chosen_answer else ""
            
            chosen_model_inputs = self.tokenizer(chosen_answer, return_tensors="pt", add_special_tokens=False)
            chosen_input_ids = chosen_model_inputs["input_ids"]
            chosen_attention_mask = chosen_model_inputs["attention_mask"]
            
            chosen_input_ids, chosen_attention_mask = verl_F.postprocess_data(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                max_length=self.max_response_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.response_truncation,
            )
            
            chosen_position_ids = compute_position_id_with_mask(chosen_attention_mask)
            
            row_dict["chosen_input_ids"] = chosen_input_ids[0]
            row_dict["chosen_attention_mask"] = chosen_attention_mask[0]
            row_dict["chosen_position_ids"] = chosen_position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.prompt_truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.prompt_truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.prompt_truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        # If dataframe was removed during serialization, restore it
        if not self.serialize_dataset and not hasattr(self, 'dataframe'):
            self.resume_dataset_state()
