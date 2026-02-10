# Flexible array job configuration
# Define your commands here - easily add/remove/modify commands
COMMANDS=(
    "python finemath_cpt.py --block_size 1536 --prefix_len 1024 --save_dir ../data/finemath_highquality_cpt_4plus_1024prefix_filter_overlap0 --filter_url --overlap_size 0"
    # Add more commands here as needed:
)

# Validate array task ID
if [ $SLURM_ARRAY_TASK_ID -ge ${#COMMANDS[@]} ] || [ $SLURM_ARRAY_TASK_ID -lt 0 ]; then
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"
    echo "Valid range: 0 to $(( ${#COMMANDS[@]} - 1 ))"
    exit 1
fi

# Execute the command for this array task
echo "Running array task $SLURM_ARRAY_TASK_ID:"
echo "Command: ${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
echo "----------------------------------------"

eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

echo "----------------------------------------"
echo "Array job $SLURM_ARRAY_TASK_ID completed successfully"