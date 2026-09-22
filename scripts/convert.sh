# Custom params
export ZERO_CKPT_DIR=/path/to/zero/checkpoint-xxxx
export OUTPUT_DIFFUSERS_DIR=/path/to/zero/checkpoint-xxxx-lora
# export NOISE_TYPE=high
export NOISE_TYPE=low

sudo mkdir -p $OUTPUT_DIFFUSERS_DIR
sudo chmod 777 -R $OUTPUT_DIFFUSERS_DIR

# Convert dit zero to bf16, Important: use --exclude_frozen_parameters!
python scripts/zero_to_bf16.py $ZERO_CKPT_DIR $OUTPUT_DIFFUSERS_DIR --weights_name "Wan2_2-CogOmni_${NOISE_TYPE}_lora_connector_14B.safetensors" --exclude_frozen_parameters

