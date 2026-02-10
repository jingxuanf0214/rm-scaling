"""
Preprocess the FineMath dataset from local source to parquet format
"""

import os
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='../data/finemath-v4/for_rm/nonembedding/rouge/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_source_path', default='/n/netscratch/konkle_lab/Everyone/Jingxuan/rl-sampling/verl_test/data/finemath_highquality_cpt_4plus_256prefix_filter/finemath-4plus')
    parser.add_argument('--tokenizer_name', default='meta-llama/Meta-Llama-3-8B', help='Tokenizer to use for detokenization if data is tokenized')
    parser.add_argument('--output_filename', default='train_512chunk_filter_highquality_4plus.parquet', help='Output filename for the parquet file')
    parser.add_argument('--data_source', default='finemath_local_4plus', help='Name of the data source to use in the output dataset')

    args = parser.parse_args()

    data_source = args.data_source
    print(f"Loading the {data_source} dataset from local path: {args.data_source_path}...", flush=True)
    
    # Load dataset from local path
    dataset = datasets.load_from_disk(args.data_source_path)
    
    # Check dataset structure
    print(f"Dataset keys: {list(dataset.keys()) if hasattr(dataset, 'keys') else 'No splits found'}", flush=True)
    print(f"Dataset type: {type(dataset)}", flush=True)
    
    # Handle different dataset structures
    if hasattr(dataset, 'keys') and 'train' in dataset:
        # Dataset has splits
        train_data = dataset['train']
        print("Dataset has train split")
    else:
        # Dataset is flat (no splits)
        train_data = dataset
        print("Dataset is flat (no splits)")
    
    # Check if we need to detokenize the data
    sample = train_data[0]
    print(f"Sample data fields: {list(sample.keys())}", flush=True)
    print(f"Sample prefix type: {type(sample['prefix'])}, target type: {type(sample['target'])}", flush=True)
    
    needs_detokenization = isinstance(sample['prefix'], list) and len(sample['prefix']) > 0 and isinstance(sample['prefix'][0], int)
    
    if needs_detokenization:
        print("Data appears to be tokenized. Loading tokenizer for detokenization...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
        def detokenize_fn(example):
            """Detokenize prefix and target from token IDs to text"""
            prefix_text = tokenizer.decode(example['prefix'], skip_special_tokens=True)
            target_text = tokenizer.decode(example['target'], skip_special_tokens=True)
            return {
                'prefix': prefix_text,
                'target': target_text
            }
        
        print("Detokenizing dataset...", flush=True)
        train_data = train_data.map(detokenize_fn, desc="Detokenizing")
    else:
        print("Data appears to already be in text format.", flush=True)
    
    # Use all data for training
    train_dataset = train_data

    #instruction_following = "Below is a paragraph of mathematical text. Please continue writing the text with step by step reasoning that follow from the context:"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prefix = example.pop('prefix')
            target = example.pop('target')
            question = prefix
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": target
                },
                "extra_info": {
                    'use_nonembedding': True,
                    'similarity_method': 'rouge',
                    'split': split,
                    'index': idx,
                    'answer': target,
                    'question': prefix,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    # Print first example as a check
    print("\n" + "="*50)
    print("FIRST EXAMPLE CHECK:")
    print("="*50)
    first_example = train_dataset[0]
    print(f"Keys in assembled dataset: {list(first_example.keys())}")
    print(f"\ndata_source: {first_example['data_source']}")
    print(f"\nability: {first_example['ability']}")
    print(f"\nprompt: {first_example['prompt']}")
    print(f"\nreward_model: {first_example['reward_model']}")
    print(f"\nextra_info keys: {list(first_example['extra_info'].keys())}")
    print(f"\nextra_info['split']: {first_example['extra_info']['split']}")
    print(f"extra_info['index']: {first_example['extra_info']['index']}")
    print(f"extra_info['use_nonembedding']: {first_example['extra_info']['use_nonembedding']}")
    print(f"extra_info['similarity_method']: {first_example['extra_info']['similarity_method']}")
    print(f"\nFirst 200 chars of question: {first_example['extra_info']['question'][:200]}...")
    print(f"\nFirst 200 chars of answer: {first_example['extra_info']['answer'][:200]}...")
    print("="*50 + "\n")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, args.output_filename))

    print(f"Saved train dataset with {len(train_dataset)} examples to {os.path.join(local_dir, args.output_filename)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
