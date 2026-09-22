export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_CHECK_DISABLE=1


# export PDSH_RCMD_TYPE=ssh
export PYTHONWARNINGS="ignore::Warning"
export SHELL_FILE_REAL_PATH=$(realpath $0)
export DIR=$(dirname ${SHELL_FILE_REAL_PATH})
export CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")


export DIR=$(dirname ${SHELL_FILE_REAL_PATH})
export CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")

MODEL_NAME=Wan-AI/Wan2.2-T2V-A14B
CONFIG_NAME="config/wan2.2/wan_cogomni_14b_civitai.yaml"
DATASET_NAME=""
LLM_NAME=Qwen/Qwen3-VL-8B-Thinking
LLM_LORA_NAME=weights/CogVLM

# Custom params
RESUME_FROM_CKPT=""
LORA_PATH=weights/low_noise/Wan2_2-CogOmni_low_lora_connector_14B.safetensors
CONNECTOR_PATH=weights/low_noise/Wan2_2-CogOmni_low_lora_connector_14B.safetensors

LORA_RANK=256
LORA_ALPHA=$LORA_RANK
CKPT_STEPS=50
BOUNDARY_TYPE="low"
GRAD_ACC_STEP=4
DATASET_META_NAME="Your data txt path"
EXP_NAME="${BOUNDARY_TYPE}_cogomni_icl_vlm_connector_joint_layout_visual_cot_prompting_fast_repeat_frame_720P"
OUTPUT_BASE_DIR=$DIR/output_dir/${EXP_NAME}/${CURRENT_TIME}
SEED=1024
VIDEO_SAMPLE_N_FRAMES=81
SP_SIZE=8
# SP_SIZE=1
# VIDEO_SAMPLE_SIZE=640
VIDEO_SAMPLE_SIZE=960
INPUT_PERTUB=0
# VIDEO_SAMPLE_SIZE=1440

# TODO: increase max_ref_num_per_subject
BATCH_SZIE=1
MAX_SUBJECT_NUM=3
MAX_REF_NUM_PER_SUBJECT=1

FILTER_NODE_IP_LIST=$(echo "$NODE_IP_LIST" | tr ',' '\n' | grep -v "^${REMOVE_IP}:" | tr '\n' ',' | sed 's/,$//')
echo $NODE_IP_LIST
export FILTER_NODE_IP_LIST=$NODE_IP_LIST
export node_ip=$(echo ${FILTER_NODE_IP_LIST} | sed 's/:8//g')

# Generate hostfile
mkdir -p ${OUTPUT_BASE_DIR}/config
export hostfile=${OUTPUT_BASE_DIR}/config/hostfile
echo "$FILTER_NODE_IP_LIST" | sed 's/:[0-9]*//g' | tr ',' '\n' | sed 's/$/ slots=8/' > "$hostfile"


# Generate accelerate config yaml
IFS=',' read -ra ip_array <<< "$node_ip"
main_ip="${ip_array[0]}"
num_machines="${#ip_array[@]}"
num_processes="$(( num_machines * 8 ))"
sed -e "s/\${MAIN_PROCESS_IP}/$main_ip/g" \
    -e "s/\${NUM_MACHINES}/$num_machines/g" \
    -e "s/\${NUM_PROCESSES}/$num_processes/g" \
    -e "s|\${HOSTFILE}|$hostfile|g" \
    config/accelerate_ds_TEMPLATE.yaml > "${OUTPUT_BASE_DIR}/config/accelerate_ds_runtime.yaml"

# Copy run shell
mkdir -p ${OUTPUT_BASE_DIR}/
cp $SHELL_FILE_REAL_PATH $OUTPUT_BASE_DIR/

accelerate launch --config_file ${OUTPUT_BASE_DIR}/config/accelerate_ds_runtime.yaml \
  scripts/wan2.2_cogomni_control/train_icl_vlm_connector_joint_viusal_cot_prompting_model_fast.py \
  --config_path=$CONFIG_NAME \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_llm_model_name_or_path=$LLM_NAME \
  --pretrained_llm_lora_model_name_or_path=$LLM_LORA_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1440 \
  --video_sample_size=${VIDEO_SAMPLE_SIZE} \
  --token_sample_size=${VIDEO_SAMPLE_SIZE} \
  --video_sample_stride=1 \
  --video_sample_n_frames=${VIDEO_SAMPLE_N_FRAMES} \
  --train_batch_size=${BATCH_SZIE} \
  --video_repeat=0 \
  --gradient_accumulation_steps=${GRAD_ACC_STEP} \
  --dataloader_num_workers=8 \
  --num_train_epochs=200 \
  --checkpointing_steps=${CKPT_STEPS} \
  --learning_rate=5e-05 \
  --seed=${SEED} \
  --output_dir=${OUTPUT_BASE_DIR} \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --use_8bit_adam \
  --use_deepspeed \
  --low_vram \
  --gradient_checkpointing \
  --boundary_type=${BOUNDARY_TYPE} \
  --sp_size=${SP_SIZE} \
  --max_subject_num=${MAX_SUBJECT_NUM} \
  --max_ref_num_per_subject=${MAX_REF_NUM_PER_SUBJECT} \
  --lora_weight=1.0 \
  --caption_mode '[["dense_caption", 0.5], ["caption", 0.5]]' \
  --lora_rank ${LORA_RANK} \
  --lora_alpha ${LORA_ALPHA} \
  --random_ref_frame_mode \
  --lora_path=${LORA_PATH} \
  --connector_path=${CONNECTOR_PATH} \
  --input_perturbation ${INPUT_PERTUB} \
  --sparse_control_mode \
  >> ${OUTPUT_BASE_DIR}/log.txt 2>&1
