# =============================================================================
# CONFIGURATION: Define your combinations here
# =============================================================================

# Define arrays for each parameter
TASKS=(
    "openai/gsm8k"
    "openai/gsm8k"
    "openai/gsm8k"
    "openai/gsm8k"
    "openai/gsm8k"
    "openai/gsm8k"
    "HuggingFaceH4/MATH-500"
    "HuggingFaceH4/MATH-500"
    "HuggingFaceH4/MATH-500"
    "HuggingFaceH4/MATH-500"
    "HuggingFaceH4/MATH-500"
    "HuggingFaceH4/MATH-500"
    # Added evolm-4B-160BT actors for both tasks, temp 1.0 only
    "openai/gsm8k"
    "HuggingFaceH4/MATH-500"
    # Added Llama-3.2-3B BASE, both tasks, temp 1.0
    "openai/gsm8k"
    "HuggingFaceH4/MATH-500"
    # Added gpqa_diamond
    "Idavidrein/gpqa"
    "Idavidrein/gpqa"
    "Idavidrein/gpqa"
    # Added Toxigen
    "skg/toxigen-data"
    "skg/toxigen-data"
    "skg/toxigen-data"
    # added ifeval
    "google/IFEval"
    "google/IFEval"
    "google/IFEval"
)

ACTORS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    # Added evolm-4B-160BT
    "zhenting/evolm-4B-160BT"
    "zhenting/evolm-4B-160BT"
    # Added Llama-3.2-3B BASE, both tasks, temp 1.0
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B"
    # Added gpqa_diamond
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    # Added Toxigen
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    # Added ifeval
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
)

TEMPERATURES=(
    "1.0"
    "0.4"
    "1.0"
    "0.4"
    "1.0"
    "0.4"
    "1.0"
    "0.4"
    "1.0"
    "0.4"
    "1.0"
    "0.4"
    # Add temp 1.0 for added actor runs only
    "1.0"
    "1.0"
    # Temp 1.0 for base-3B runs
    "1.0"
    "1.0"
    # Added gpqa_diamond
    "1.0"
    "1.0"
    "1.0"
    # Added Toxigen
    "1.0"
    "1.0"
    "1.0"
    # Added ifeval
    "1.0"
    "1.0"
    "1.0"
)

OUT_DIRS=(
    "bon_out_gsm8k_llama1b_temp1.0"
    "bon_out_gsm8k_llama1b_temp0.4"
    "bon_out_gsm8k_llama8b_temp1.0"
    "bon_out_gsm8k_llama8b_temp0.4"
    "bon_out_gsm8k_llama3b_temp1.0"
    "bon_out_gsm8k_llama3b_temp0.4"
    "bon_out_math500_llama1b_temp1.0"
    "bon_out_math500_llama1b_temp0.4"
    "bon_out_math500_llama8b_temp1.0"
    "bon_out_math500_llama8b_temp0.4"
    "bon_out_math500_llama3b_temp1.0"
    "bon_out_math500_llama3b_temp0.4"
    # Out dirs for evolm-4B-160BT
    "bon_out_gsm8k_evolm4b_temp1.0"
    "bon_out_math500_evolm4b_temp1.0"
    # Out dirs for base-3B runs
    "bon_out_gsm8k_llama3b_base_temp1.0"
    "bon_out_math500_llama3b_base_temp1.0"
    # Out dirs for gpqa_diamond
    "bon_out_gpqa_diamond_llama1b_temp1.0"
    "bon_out_gpqa_diamond_llama8b_temp1.0"
    "bon_out_gpqa_diamond_llama3b_temp1.0"
    # Out dirs for Toxigen
    "bon_out_toxigen_llama1b_temp1.0"
    "bon_out_toxigen_llama8b_temp1.0"
    "bon_out_toxigen_llama3b_temp1.0"
    # Out dirs for ifeval
    "bon_out_ifeval_llama1b_temp1.0"
    "bon_out_ifeval_llama8b_temp1.0"
    "bon_out_ifeval_llama3b_temp1.0"
)

# Optional: dataset splits (default to "test" if not specified)
SPLITS=(
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    "test"
    # Splits for extra evolm-4B-160BT runs
    "test"
    "test"
    # Splits for base-3B runs
    "test"
    "test"
    # Splits for gpqa_diamond runs
    "train"
    "train"
    "train"
    # Splits for Toxigen runs
    "test"
    "test"
    "test"
    # Splits for ifeval runs
    "train"
    "train"
    "train"
)

# =============================================================================
# Get configuration for this array task
# =============================================================================

# Validate SLURM_ARRAY_TASK_ID
NUM_CONFIGS=${#TASKS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_CONFIGS" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) >= NUM_CONFIGS ($NUM_CONFIGS)"
    exit 1
fi

# Get parameters for this task
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
ACTOR=${ACTORS[$SLURM_ARRAY_TASK_ID]}
TEMPERATURE=${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=${OUT_DIRS[$SLURM_ARRAY_TASK_ID]}
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]:-test}

echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Task/Dataset:  $TASK"
echo "Actor:         $ACTOR"
echo "Temperature:   $TEMPERATURE"
echo "Output Dir:    $OUT_DIR"
echo "Split:         $SPLIT"
echo "========================================"

# =============================================================================
# Define reward models to test
# =============================================================================

#
# Format:
# - `RM_REPOS` is a comma-separated list of HF model repos
# - Most runs use a single RM, but you can evaluate multiple by adding commas
#
# Example (single RM):
RM_REPOS="Skywork/Skywork-Reward-V2-Llama-3.2-3B"
#
# Example (multiple RMs):
# RM_REPOS="Skywork/Skywork-Reward-V2-Llama-3.2-3B,Skywork/Skywork-Reward-V2-Qwen3-4B"
#


# =============================================================================
# Run the Python script
# =============================================================================

python bon_ifeval.py \
    --ifeval \
    --actor_repo "${ACTOR}" \
    --rm_repo "${RM_REPOS}" \
    --rm_logits_index 0 \
    --dataset_path "${TASK}" \
    --split "${SPLIT}" \
    --n_list 1,2,4,8,16,32 \
    --out_dir "${OUT_DIR}" \
    --seed 2025 \
    --temperature "${TEMPERATURE}" \
    --top_p 1.0 \