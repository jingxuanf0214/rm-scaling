
# ============================================================================
# CONFIGURATION 
# ============================================================================

CHECKPOINTS_BASE="rm_training/checkpoints"
RESULTS_BASE=results/eval-set"
MODEL_TYPE="reward"

PROJECT_NAME="${PROJECT_NAME:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS_TO_EVAL="${STEPS_TO_EVAL:-}"
BENCH_SCRIPT="${BENCH_SCRIPT:-run_rm.py}"   # v1 rewardbench
CHAT_TEMPLATE="${CHAT_TEMPLATE:-Ziya}"
FORCE_RERUN="${FORCE_RERUN:-0}"  # set to 1 (or pass --force) to re-run even if results exist

usage() {
    echo "Usage: $0 --project_name <name> [--batch_size N] [--steps step1,step2,...] [--bench_script run_rm.py] [--chat_template Ziya] [--force]"
    echo ""
    echo "Example:"
    echo "  sbatch $0 --project_name rlsampling-rm_training_test_centering --batch_size 8"
    echo "  sbatch $0 --project_name rlsampling-rm_training_test_centering --steps \"initial_checkpoint/global_step_0,global_step_15\""
    echo ""
    echo "Notes:"
    echo "  - By default, this script SKIPS models that already have local RewardBench v1 results at:"
    echo "      ${RESULTS_BASE}<absolute_merged_path>.json"
    echo "    Example:"
    echo "      ${RESULTS_BASE}/n/netscratch/.../global_step_195/reward/merged.json"
    echo "  - Use --force to re-run evaluation even if the result file exists."
}

# Parse CLI args (SLURM will pass these through after the script name)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --steps|--steps_to_eval)
            STEPS_TO_EVAL="$2"
            shift 2
            ;;
        --bench_script)
            BENCH_SCRIPT="$2"
            shift 2
            ;;
        --chat_template)
            CHAT_TEMPLATE="$2"
            shift 2
            ;;
        --force)
            FORCE_RERUN=1
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$PROJECT_NAME" ]; then
    echo "Error: --project_name is required (or set env var PROJECT_NAME)."
    usage
    exit 1
fi

CHECKPOINT_ROOT_PATH="$CHECKPOINTS_BASE/$PROJECT_NAME"

# Fixed bench args
ADDITIONAL_BENCH_ARGS="--chat_template=$CHAT_TEMPLATE --batch_size=$BATCH_SIZE --disable_beaker_save --do_not_save"

# ============================================================================

echo "=========================================="
echo "Batch Model Merger and RewardBench Runner"
echo "=========================================="
echo "Checkpoint root path: $CHECKPOINT_ROOT_PATH"
echo "Project name: $PROJECT_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Steps to eval: ${STEPS_TO_EVAL:-ALL}"
echo "Model type: $MODEL_TYPE"
echo "Bench script: $BENCH_SCRIPT"
echo "Chat template: $CHAT_TEMPLATE"
echo "Additional bench args: $ADDITIONAL_BENCH_ARGS"
echo "Results base: $RESULTS_BASE"
echo "Skip if results exist: $([ "$FORCE_RERUN" -eq 1 ] && echo "NO (force re-run)" || echo "YES")"
echo ""

# Check if checkpoint root path exists
if [ ! -d "$CHECKPOINT_ROOT_PATH" ]; then
    echo "Error: Checkpoint root path does not exist: $CHECKPOINT_ROOT_PATH"
    exit 1
fi

# Step 1: Find all merged models and run rewardbench
echo "=========================================="
echo "Step 1: Running RewardBench on merged models (skipping merge)..."
echo "=========================================="

# Find all merged model directories
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
    
    # Check if local results already exist for this merged checkpoint and skip if so.
    # RewardBench v1 saves results to: ./results/eval-set/<model_path>.json
    # When --model is an absolute path, this becomes:
    #   <RESULTS_BASE>/<absolute model path>.json
    RESULT_JSON="${RESULTS_BASE}${MERGED_DIR}.json"
    if [ "$FORCE_RERUN" -ne 1 ] && [ -s "$RESULT_JSON" ]; then
        echo "----------------------------------------"
        echo "Skipping model $TOTAL_MODELS: $STEP_INFO (already evaluated)"
        echo "Model path: $MERGED_DIR"
        echo "Found results: $RESULT_JSON"
        echo "----------------------------------------"
        echo ""
        continue
    fi
    
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
echo "Project name: $PROJECT_NAME"
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