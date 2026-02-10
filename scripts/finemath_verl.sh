# Define arrays of data source paths and corresponding output filenames
# Add your data source paths here
DATA_SOURCE_PATHS=(
    "data/finemath_highquality_cpt_4plus_512prefix_filter_sentence_aware/finemath-4plus"
 )

# Define corresponding output filenames
OUTPUT_FILENAMES=(
   "train_1536chunk_512prefix_filter_sentence_aware_4plus.parquet"
)

# Get the current array task ID
ARRAY_ID=$SLURM_ARRAY_TASK_ID

# Get the data source path and output filename for this array task
DATA_SOURCE_PATH=${DATA_SOURCE_PATHS[$ARRAY_ID]}
OUTPUT_FILENAME=${OUTPUT_FILENAMES[$ARRAY_ID]}

echo "Processing array task $ARRAY_ID"
echo "Data source path: $DATA_SOURCE_PATH"
echo "Output filename: $OUTPUT_FILENAME"

# Run the Python script with the specific data source and output filename
python finemath_local.py \
    --local_dir "../data/finemath_4plus/" \
    --data_source_path "$DATA_SOURCE_PATH" \
    --output_filename "$OUTPUT_FILENAME" \
    --data_source "finemath_local_4plus"