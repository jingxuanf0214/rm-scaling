
seed=2025

# Define dataset paths
finemath_train_path_1=../data/finemath-v4/for_rm/nonembedding/rouge/train_1536chunk_filter_highquality_4plus_part1.parquet
finemath_train_path_2=../data/finemath-v4/for_rm/nonembedding/rouge/train_1536chunk_filter_highquality_4plus_part2.parquet
infiwebmath_train_path_1=../data/finemath_4plus/train_1536chunk_filter_overlap0_4plus_part1.parquet
infiwebmath_train_path_2=../data/finemath_4plus/train_1536chunk_filter_overlap0_4plus_part2.parquet
finemath_test_path=../data/finemath-v4/match_target/nonembedding/rouge/test.parquet

# Define arrays of learning rates, warmup ratios, styles, batch sizes, centering coeffs, label smoothing, save frequencies, and train files to sweep
learning_rates=("1e-6" "1e-6") #"1e-5" "3e-5" "1e-4")
warmup_ratios=(0.05 0.05) #0.05 0.05 0.05 0.05 0.05)
styles=("constant" "constant")
batch_sizes=(32 32)
centering_coeffs=(0.01 0.01) # centering coefficient for reward model
label_smoothings=(0 0) # label smoothing for reward model
save_freqs=(15 15) # save frequency for each run
# Add different train file combinations here
train_file_configs=("['$finemath_train_path_1','$finemath_train_path_2']" "['$infiwebmath_train_path_1','$infiwebmath_train_path_2']") # Example: add "['$finemath_train_path']" "['$gsm8k_train_path']" for more options
# Corresponding readable names for experiments (must match order of train_file_configs)
dataset_names=("finemath_train_part1_part2_sentence_aware" "infiwebmath_part1_part2")

# Get the array task ID (1-indexed)
array_id=$SLURM_ARRAY_TASK_ID

# Convert to 0-indexed for array access
idx=$((array_id - 1))

# Select learning rate, warmup ratio, style, batch size, centering coeff, label smoothing, save frequency, train files, and dataset name based on array ID
learning_rate="${learning_rates[$idx]}"
warmup_steps_ratio="${warmup_ratios[$idx]}"
style="${styles[$idx]}"
batch_size="${batch_sizes[$idx]}"
centering_coeff="${centering_coeffs[$idx]}"
label_smoothing="${label_smoothings[$idx]}"
save_freq="${save_freqs[$idx]}"
train_files_config="${train_file_configs[$idx]}"
dataset_name="${dataset_names[$idx]}"

export RAY_TMPDIR=~/ray_temp
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_SHOW_CPP_STACKTRACES=1
export VERL_DEBUG_SYNC=1
export VERL_DEBUG_MEM=1
export VERL_EMPTY_CACHE_INNER=1
export TORCH_DISABLE_ADDR2LINE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=300
export TORCH_DISTRIBUTED_DEBUG=DETAIL

train_files="$train_files_config"
test_files="['$finemath_test_path']"

model_path=meta-llama/Llama-3.2-3B
declare -i num_gpus=1
project_name='rlsampling-rm_training_data_ablation'
grad_clip=1.0
experiment_name="$(basename "$model_path")-${dataset_name}-rm-lr${learning_rate}-${style}-warmup_${warmup_steps_ratio}-bs${batch_size}-gc${grad_clip}-cc${centering_coeff}-ls${label_smoothing}"



PYTHONUNBUFFERED=1 python3 -m recipe.prime.main_prime \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=$batch_size \
    data.max_prompt_length=560 \
    data.max_response_length=1100 \
    data.filter_overlong_prompts=True \
    data.filter_accuracy=False \
    data.shuffle=False \
    data.accuracy_lower_bound=0.0 \
    data.accuracy_upper_bound=1.0 \
    data.oversample_factor=1 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$learning_rate \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(($batch_size*$num_gpus)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=False \
    reward_model.model.path=$model_path \
    reward_model.micro_batch_size_per_gpu=2 \
    reward_model.model.seed=$seed \
    reward_model.model.update=after \
    reward_model.model.beta_train=0.05 \
    reward_model.model.centering_coeff=$centering_coeff \
    reward_model.model.label_smoothing=$label_smoothing \
    reward_model.model.optim.lr=$learning_rate \
    reward_model.model.optim.lr_warmup_steps_ratio=$warmup_steps_ratio \
    reward_model.model.optim.warmup_style=$style \
    reward_model.model.optim.grad_clip=$grad_clip \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=$(($batch_size*$num_gpus)) \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=0 \
    trainer.total_epochs=1 