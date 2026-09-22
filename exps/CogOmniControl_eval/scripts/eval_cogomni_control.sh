export PYTHONPATH=$(pwd)


export PATH=/usr/local/bin:$PATH


# Custom params
export INFER_CFG_PATH=./exps/CogOmniControl_eval/configs/flow_len81_cogomni_connector_720p_infer.yaml
export EVAL_ANNO_PATH=(
    benchmark/annotations.jsonl
)
export BENCHMARK_PATH_PREFIX=benchmark
export OUPUT_BASE_DIR=exps/CogOmniControl_eval/output_cogomni_VLM_RL_DiT_RL_dir
export CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")

export INFER_CFG_NAME=$(basename ${INFER_CFG_PATH} .yaml)
export OUTPUT_DIR=${OUPUT_BASE_DIR}/${CURRENT_TIME}_${INFER_CFG_NAME}


python exps/CogOmniControl_eval/scripts/main_cogomni_control_connector_eval.py \
                --eval_anno_path ${EVAL_ANNO_PATH[@]}\
                --infer_cfg_path $INFER_CFG_PATH \
                --output_dir  $OUTPUT_DIR \
                --benchmark_path_prefix $BENCHMARK_PATH_PREFIX

echo "OUTPUT_DIR: ${OUTPUT_DIR}"
