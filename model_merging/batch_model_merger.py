#!/usr/bin/env python3
"""
Batch Model Merger Script

This script automatically merges checkpoints from multiple training steps.
It loops through checkpoint directories and calls the model merger for each step.

The script looks for checkpoints in:
- Regular global_step_* directories (e.g., global_step_60, global_step_120)
- Special initial checkpoint at initial_checkpoint/global_step_0

Usage:
    python batch_model_merger.py \
        --checkpoint_root_path "/path/to/checkpoints/model_name" \
        --hf_root_name "your_hf_username/model_name" \
        --model_type "reward" \
        [--dry_run] [--steps 60,120,180]

Example:
    python batch_model_merger.py \
        --checkpoint_root_path "/n/netscratch/konkle_lab/Everyone/Jingxuan/rl-sampling/verl_test/checkpoints/rm_test/Qwen2.5-0.5B-Instruct-finemath-rouge-orm-new-debug7" \
        --hf_root_name "fjxdaisy/qwen2.5-0.5b-instruct-finemath-orm-v7" \
        --model_type "reward"
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_global_step_dirs(checkpoint_root_path: str) -> List[tuple[str, int, bool]]:
    """
    Find all global_step directories in the checkpoint root path.
    
    This function looks for:
    1. Regular global_step_* directories directly under the root
    2. Special case: initial_checkpoint/global_step_0
    
    Args:
        checkpoint_root_path: Path to the checkpoint root directory
        
    Returns:
        List of tuples (directory_path, step_number, is_initial_checkpoint) sorted by step number
        where directory_path is relative to checkpoint_root_path and is_initial_checkpoint 
        indicates if this is the special initial checkpoint case
    """
    checkpoint_root = Path(checkpoint_root_path)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root path does not exist: {checkpoint_root_path}")
    
    global_step_dirs = []
    
    # Check for regular global_step_* directories
    for item in checkpoint_root.iterdir():
        if item.is_dir():
            match = re.match(r"global_step_(\d+)", item.name)
            if match:
                step_number = int(match.group(1))
                global_step_dirs.append((item.name, step_number, False))
    
    # Check for special case: initial_checkpoint/global_step_0
    initial_checkpoint_dir = checkpoint_root / "initial_checkpoint"
    if initial_checkpoint_dir.exists() and initial_checkpoint_dir.is_dir():
        global_step_0_dir = initial_checkpoint_dir / "global_step_0"
        if global_step_0_dir.exists() and global_step_0_dir.is_dir():
            # Use relative path from checkpoint root
            relative_path = "initial_checkpoint/global_step_0"
            global_step_dirs.append((relative_path, 0, True))
            print(f"Found initial checkpoint at: {relative_path}")
    
    if not global_step_dirs:
        raise ValueError(f"No global_step directories found in: {checkpoint_root_path}")
    
    # Sort by step number
    global_step_dirs.sort(key=lambda x: x[1])
    
    return global_step_dirs


def check_checkpoint_exists(checkpoint_path: str, model_type: str, is_initial_checkpoint: bool = False) -> bool:
    """
    Check if the checkpoint directory exists and contains the model type subdirectory.
    
    Args:
        checkpoint_path: Path to the global_step directory
        model_type: Type of model (e.g., "reward", "actor", "critic")
        is_initial_checkpoint: True if this is the initial checkpoint (no model_type subfolder)
        
    Returns:
        True if the checkpoint exists and is valid
    """
    if is_initial_checkpoint:
        # For initial checkpoint, check if the directory exists directly (no model_type subfolder)
        return Path(checkpoint_path).exists() and Path(checkpoint_path).is_dir()
    else:
        # For regular checkpoints, check for model_type subfolder
        model_path = Path(checkpoint_path) / model_type
        return model_path.exists() and model_path.is_dir()


def run_model_merger(
    backend: str,
    local_dir: str,
    target_dir: str,
    hf_upload_path: str,
    dry_run: bool = False
) -> bool:
    """
    Run the model merger for a single checkpoint.
    
    Args:
        backend: Backend type ("fsdp" or "megatron")
        local_dir: Path to the checkpoint directory
        target_dir: Path to save the merged model
        hf_upload_path: Hugging Face repository path
        dry_run: If True, only print the command without executing
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        sys.executable, "model_merger.py", "merge",
        "--backend", backend,
        "--local_dir", local_dir,
        "--target_dir", target_dir,
    ]
    
    if hf_upload_path:
        cmd.extend(["--hf_upload_path", hf_upload_path])
    
    print(f"Running command: {' '.join(cmd)}")
    
    if dry_run:
        print("DRY RUN: Command would be executed but --dry_run is enabled")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"SUCCESS: Merged checkpoint saved to {target_dir}")
        if hf_upload_path:
            print(f"SUCCESS: Model uploaded to {hf_upload_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Model merger failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch merge checkpoints from multiple training steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint_root_path",
        type=str,
        required=True,
        help="Path to the checkpoint root directory containing global_step_* folders"
    )
    
    parser.add_argument(
        "--hf_root_name", 
        type=str,
        help="Base Hugging Face repository name (e.g., 'username/model-name'). If not provided, model won't be uploaded to HF."
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="reward",
        help="Type of model to merge (e.g., 'reward', 'actor', 'critic')"
    )
    
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of specific steps to merge. Can be numeric (e.g., '60,120,180') or full identifiers (e.g., 'global_step_60,initial_checkpoint/global_step_0'). If not provided, all steps will be merged."
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip merging if target directory already exists"
    )
    
    args = parser.parse_args()
    
    # Find all global_step directories
    try:
        global_step_dirs = find_global_step_dirs(args.checkpoint_root_path)
        print(f"Found {len(global_step_dirs)} global_step directories")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Filter steps if specified
    if args.steps:
        step_parts = [s.strip() for s in args.steps.split(",")]
        
        # Check if steps are numeric or full identifiers
        specified_steps = set()
        specified_dirs = set()
        
        for s in step_parts:
            # Try to parse as numeric
            try:
                specified_steps.add(int(s))
            except ValueError:
                # Not numeric - treat as full identifier (e.g., "global_step_60" or "initial_checkpoint/global_step_0")
                specified_dirs.add(s)
                # Also extract step number for matching
                match = re.search(r'global_step_(\d+)', s)
                if match:
                    specified_steps.add(int(match.group(1)))
        
        # Filter: match by step number or by directory name
        global_step_dirs = [
            (dir_name, step, is_initial) 
            for dir_name, step, is_initial in global_step_dirs 
            if step in specified_steps or dir_name in specified_dirs
        ]
        print(f"Filtering to {len(global_step_dirs)} specified steps")
    
    if not global_step_dirs:
        print("ERROR: No valid steps found to process")
        sys.exit(1)
    
    # Process each checkpoint
    success_count = 0
    total_count = len(global_step_dirs)
    
    for dir_name, step_number, is_initial_checkpoint in global_step_dirs:
        print(f"\n{'='*60}")
        print(f"Processing step {step_number} ({dir_name})")
        print(f"{'='*60}")
        
        # Construct paths
        # For regular checkpoints: checkpoint_root/global_step_X/model_type
        # For initial checkpoint: checkpoint_root/initial_checkpoint/global_step_0 (no model_type subfolder)
        if is_initial_checkpoint:
            local_dir = os.path.join(args.checkpoint_root_path, dir_name)
            target_dir = os.path.join(args.checkpoint_root_path, dir_name, "merged")
        else:
            local_dir = os.path.join(args.checkpoint_root_path, dir_name, args.model_type)
            target_dir = os.path.join(args.checkpoint_root_path, dir_name, args.model_type, "merged")
        
        hf_upload_path = f"{args.hf_root_name}-step{step_number}" if args.hf_root_name else None
        
        # Check if checkpoint exists
        if not check_checkpoint_exists(os.path.join(args.checkpoint_root_path, dir_name), args.model_type, is_initial_checkpoint):
            print(f"WARNING: Checkpoint not found at {local_dir}, skipping...")
            continue
        
        # Check if target already exists with .safetensors files
        if args.skip_existing and os.path.exists(target_dir):
            # Only skip if there are actual .safetensors files (indicates successful merge)
            safetensor_files = list(Path(target_dir).glob("*.safetensors"))
            if safetensor_files:
                print(f"INFO: Target directory already has {len(safetensor_files)} .safetensors file(s) at {target_dir}, skipping...")
                success_count += 1
                continue
            else:
                print(f"INFO: Target directory exists but has no .safetensors files, will re-merge: {target_dir}")
        
        # Run model merger
        success = run_model_merger(
            backend="fsdp",
            local_dir=local_dir,
            target_dir=target_dir,
            hf_upload_path=hf_upload_path,
            dry_run=args.dry_run
        )
        
        if success:
            success_count += 1
        else:
            print(f"FAILED: Step {step_number}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{total_count} checkpoints processed successfully")
    print(f"{'='*60}")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main() 