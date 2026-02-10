# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def load_and_check_parquet(local_dir):
    """
    Load the train and test parquet files from the given local directory,
    then double-check that the train dataset is sorted by difficulty level.
    """
    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')

    print("Loading train dataset from:", train_path)
    train_dataset = datasets.load_dataset("parquet", data_files={"train": train_path})["train"]

    print("Loading test dataset from:", test_path)
    test_dataset = datasets.load_dataset("parquet", data_files={"test": test_path})["test"]

    # Print dataset sizes
    print(f"Train dataset contains {len(train_dataset)} examples.")
    print(f"Test dataset contains {len(test_dataset)} examples.")

    # Double-check if train dataset is sorted by level.
    # Assumes that each train example's 'extra_info' dict contains a 'level' key.
    levels = [example["extra_info"].get("level", None) for example in train_dataset]
    if None in levels:
        print("Warning: Some entries in the train dataset do not contain level info.")
    else:
        if levels == sorted(levels):
            print("Train dataset is correctly sorted by level (from 1 to 5).")
        else:
            print("Train dataset is NOT sorted by level. Please check the ordering.")

    return train_dataset, test_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='../data/math_rm')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'DigitalLearningGmbH/MATH-lighteval'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            #if split == 'train':
            level = example.pop('level')
            question_raw = example.pop('problem')

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('solution')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "model",
                    "ground_truth": solution
                },
                "extra_info": {
                    'use_nonembedding': False,
                    'similarity_method': 'meteor',
                    'split': split,
                    'index': idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    #'level': level
                }
            }
            # Optionally include level info for train split
            #if split == 'train':
            #data["extra_info"]["level"] = level
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

    # Optional: load and double-check the saved parquet files.
    load_and_check_parquet(local_dir)