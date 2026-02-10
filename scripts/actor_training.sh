
# Configuration variables - modify these as needed
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=~/ray_temp

KL_LOSS_COEF="0.08"
NUM_GPUS=2
lr="1e-6"
ROLLOUT_N=16
MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"

# Dataset configuration (used both for data paths and experiment naming)
TRAIN_FILE="../data/gsm8k/train.parquet"
VAL_FILE="../data/gsm8k/test.parquet"

# Derive a short dataset identifier from the parent folder name (e.g., gsm8k, math)
DATASET_NAME="$(basename "$(dirname "$TRAIN_FILE")")"

# Array of reward model paths - add more as needed
REWARD_MODEL_PATHS=(
     "Skywork/Skywork-Reward-V2-Llama-3.2-3B"
     # Add more reward model paths here
)

# Select the current reward model based on array task ID
REWARD_MODEL_PATH="${REWARD_MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}"

# Construct experiment name automatically from model paths and hyperparameters
ADV_ESTIMATOR="grpo"
RM_IDENTIFIER="${REWARD_MODEL_PATH##*/}"
ACTOR_IDENTIFIER="${MODEL_PATH##*/}"
EXPERIMENT_NAME="${ACTOR_IDENTIFIER}_${DATASET_NAME}_${RM_IDENTIFIER}_kl${KL_LOSS_COEF}_lr${lr}_n${ROLLOUT_N}_${ADV_ESTIMATOR}"


# Validate array configuration
if [ $SLURM_ARRAY_TASK_ID -ge ${#REWARD_MODEL_PATHS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of range for arrays of size ${#REWARD_MODEL_PATHS[@]}"
    exit 1
fi

echo "Running array task $SLURM_ARRAY_TASK_ID of ${#REWARD_MODEL_PATHS[@]}"
echo "Selected REWARD_MODEL_PATH: $REWARD_MODEL_PATH"
echo "Selected EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "Selected DATASET_NAME: $DATASET_NAME"


# Load the model first; this will download and cache it if not already done.
python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"

# Once the model is loaded, start the PPO training.
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=$ADV_ESTIMATOR \
 data.train_files=$TRAIN_FILE \
 data.val_files=$VAL_FILE \
 data.train_batch_size=512 \
 data.max_prompt_length=1024 \
 data.max_response_length=512 \
 data.filter_overlong_prompts=True \
 data.truncation='error' \
 data.return_raw_chat=True \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=$lr \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=512 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
 actor_rollout_ref.actor.kl_loss_type=mse \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.n=$ROLLOUT_N \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.use_kl_in_reward=False \
 reward_model.enable=True \
 reward_model.enable_train=False \
 reward_model.model.path=$REWARD_MODEL_PATH \
 reward_model.model.use_remove_padding=True \
 reward_model.model.fsdp_config.param_offload=True \
 reward_model.micro_batch_size_per_gpu=4 \
 reward_model.use_dynamic_bsz=True \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name="rlsampling-verl_math_rm_rl" \
 trainer.experiment_name="$EXPERIMENT_NAME" \
 trainer.val_before_train=True \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$NUM_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=8 2>&1 | tee verl_demo.log