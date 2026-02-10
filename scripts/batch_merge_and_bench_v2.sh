
# ============================================================================
# CONFIGURATION 
# ============================================================================

# Array job index (0-based)
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Dictionary of checkpoint paths, batch sizes, and optional steps to evaluate
# Format: "checkpoint_path:batch_size" or "checkpoint_path:batch_size:step1,step2,..."
# Steps can be: global_step_15, initial_checkpoint/global_step_0, etc.
# If no steps specified, all steps will be evaluated
# You can generate this from check_unevaled_checkpoints.py --output-configs
declare -a CONFIGURATIONS=(
    # input checkpoints:batch_size:steps_to_eval
    )

# Validate array index
if [ $ARRAY_INDEX -ge ${#CONFIGURATIONS[@]} ]; then
    echo "Error: Array index $ARRAY_INDEX is out of range. Max index is $((${#CONFIGURATIONS[@]} - 1))"
    exit 1
fi

# Parse checkpoint path, batch size, and optional steps for current array job
CONFIG_LINE="${CONFIGURATIONS[$ARRAY_INDEX]}"
# Split by colon - may have 2 or 3 fields
IFS=':' read -r CHECKPOINT_ROOT_PATH BATCH_SIZE STEPS_TO_EVAL <<< "$CONFIG_LINE"

# STEPS_TO_EVAL is optional - if provided, only these steps will be evaluated
# Format: "global_step_5,global_step_10,initial_checkpoint/global_step_0"

# Fixed parameters (same for all configurations)
HF_ROOT_NAME=""
MODEL_TYPE="reward"
ADDITIONAL_MERGER_ARGS=""
ADDITIONAL_BENCH_ARGS="--chat_template=Ziya --batch_size=$BATCH_SIZE --disable_beaker_save"
#ADDITIONAL_BENCH_ARGS="--chat_template=Ziya --batch_size=$BATCH_SIZE --disable_beaker_save --do_not_save"
BENCH_SCRIPT="run_v2.py"  # Options: "run_rm.py" or "run_v2.py"

# ============================================================================

echo "=========================================="
echo "Batch Model Merger and RewardBench Runner"
echo "=========================================="
echo "Array Job Index: $ARRAY_INDEX"
echo "Checkpoint root path: $CHECKPOINT_ROOT_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Steps to eval: ${STEPS_TO_EVAL:-ALL}"
echo "HF root name: $HF_ROOT_NAME"
echo "Model type: $MODEL_TYPE"
echo "Bench script: $BENCH_SCRIPT"
echo "Additional merger args: $ADDITIONAL_MERGER_ARGS"
echo "Additional bench args: $ADDITIONAL_BENCH_ARGS"
echo ""

# Check if checkpoint root path exists
if [ ! -d "$CHECKPOINT_ROOT_PATH" ]; then
    echo "Error: Checkpoint root path does not exist: $CHECKPOINT_ROOT_PATH"
    exit 1
fi

# Step 1: Run batch model merger
echo "=========================================="
echo "Step 1: Running batch model merger..."
echo "=========================================="

MERGER_CMD="python batch_model_merger.py --checkpoint_root_path \"$CHECKPOINT_ROOT_PATH\" --model_type \"$MODEL_TYPE\" --skip_existing"

if [ -n "$HF_ROOT_NAME" ]; then
    MERGER_CMD="$MERGER_CMD --hf_root_name \"$HF_ROOT_NAME\""
fi

if [ -n "$ADDITIONAL_MERGER_ARGS" ]; then
    MERGER_CMD="$MERGER_CMD $ADDITIONAL_MERGER_ARGS"
fi

# If specific steps are provided, add them to the merger command
if [ -n "$STEPS_TO_EVAL" ]; then
    MERGER_CMD="$MERGER_CMD --steps \"$STEPS_TO_EVAL\""
fi

echo "Running: $MERGER_CMD"
eval $MERGER_CMD

if [ $? -ne 0 ]; then
    echo "Error: Batch model merger failed"
    exit 1
fi

echo ""
echo "Batch model merger completed successfully!"
echo ""

# Step 2: Find all merged models and run rewardbench
echo "=========================================="
echo "Step 2: Running RewardBench on merged models..."
echo "=========================================="

# Find all merged model directories
# Look for both regular checkpoints and the special initial checkpoint case
ALL_MERGED_DIRS=$(find "$CHECKPOINT_ROOT_PATH" -type d -name "merged" \( \
    -path "*/global_step_*/$MODEL_TYPE/merged" -o \
    -path "*/initial_checkpoint/global_step_0/merged" \
\) | sort)

if [ -z "$ALL_MERGED_DIRS" ]; then
    echo "Warning: No merged model directories found in $CHECKPOINT_ROOT_PATH"
    echo "Expected patterns: */global_step_*/$MODEL_TYPE/merged or */initial_checkpoint/global_step_0/merged"
    exit 1
fi

# Filter merged directories if specific steps are provided
if [ -n "$STEPS_TO_EVAL" ]; then
    echo "Filtering to specified steps: $STEPS_TO_EVAL"
    MERGED_DIRS=""
    # Convert comma-separated steps to array
    IFS=',' read -ra STEP_ARRAY <<< "$STEPS_TO_EVAL"
    for MERGED_DIR in $ALL_MERGED_DIRS; do
        for STEP in "${STEP_ARRAY[@]}"; do
            # Check if this merged dir matches the step
            # Handle both regular steps (global_step_X) and initial checkpoint
            if [[ "$MERGED_DIR" == *"/$STEP/"* ]] || [[ "$MERGED_DIR" == *"/$STEP/$MODEL_TYPE/merged" ]]; then
                MERGED_DIRS="$MERGED_DIRS $MERGED_DIR"
                break
            fi
        done
    done
    MERGED_DIRS=$(echo "$MERGED_DIRS" | xargs -n1 | sort -u)
else
    MERGED_DIRS="$ALL_MERGED_DIRS"
fi

if [ -z "$MERGED_DIRS" ]; then
    echo "Warning: No merged model directories match the specified steps"
    echo "Steps requested: $STEPS_TO_EVAL"
    exit 1
fi

echo "Found merged model directories to evaluate:"
echo "$MERGED_DIRS"
echo ""

# Counter for tracking results
TOTAL_MODELS=0
SUCCESSFUL_EVALS=0
FAILED_EVALS=0

# Run rewardbench on each merged model
for MERGED_DIR in $MERGED_DIRS; do
    TOTAL_MODELS=$((TOTAL_MODELS + 1))
    
    # Extract step number from path for identification
    STEP_INFO=$(echo "$MERGED_DIR" | grep -o "global_step_[0-9]*" || echo "unknown_step")
    
    echo "----------------------------------------"
    echo "Evaluating model $TOTAL_MODELS: $STEP_INFO"
    echo "Model path: $MERGED_DIR"
    echo "----------------------------------------"
    
    # Construct rewardbench command
    BENCH_CMD="python reward_bench/$BENCH_SCRIPT --model=\"$MERGED_DIR\" $ADDITIONAL_BENCH_ARGS"
    
    echo "Running: $BENCH_CMD"
    
    # Run rewardbench and capture exit code
    if eval $BENCH_CMD; then
        echo "SUCCESS: RewardBench completed for $STEP_INFO"
        SUCCESSFUL_EVALS=$((SUCCESSFUL_EVALS + 1))
    else
        echo "FAILED: RewardBench failed for $STEP_INFO"
        FAILED_EVALS=$((FAILED_EVALS + 1))
    fi
    
    echo ""
done

# Final summary
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Array Job Index: $ARRAY_INDEX"
echo "Checkpoint: $CHECKPOINT_ROOT_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Bench Script: $BENCH_SCRIPT"
echo "Total models processed: $TOTAL_MODELS"
echo "Successful evaluations: $SUCCESSFUL_EVALS"
echo "Failed evaluations: $FAILED_EVALS"
echo ""

if [ $FAILED_EVALS -gt 0 ]; then
    echo "Warning: Some evaluations failed. Check the logs above for details."
    exit 1
else
    echo "All evaluations completed successfully!"
fi 