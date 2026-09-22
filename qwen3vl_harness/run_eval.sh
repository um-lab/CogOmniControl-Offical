#!/bin/bash

set -e
cd "$(dirname "$0")"

MODEL_PATH="Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
TP=2                       # 张量并行 GPU 数
MAX_MODEL_LEN=32768
MAX_NEW_TOKENS=4096
VIDEO_MAX_FRAMES=32
VIDEO_FPS=2.0
GPU_UTIL=0.9

IN_DIR="input"
OUT_DIR="output"
mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES=0,1 python qwen3vl_harness.py evaluate \
        --input_jsonl  "input/fused_results_sample1.json" \
        --output_jsonl "output/fused_results_sample1_harness_all_metrics.json" \
        --model_path "${MODEL_PATH}" \
        --tensor_parallel_size ${TP} \
        --max_model_len ${MAX_MODEL_LEN} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --video_max_frames ${VIDEO_MAX_FRAMES} \
        --video_fps ${VIDEO_FPS} \
        --gpu_memory_utilization ${GPU_UTIL}

CUDA_VISIBLE_DEVICES=0,1 python qwen3vl_harness.py evaluate \
        --input_jsonl  "input/fused_results_sample2.json" \
        --output_jsonl "output/fused_results_sample2_harness_all_metrics.json" \
        --model_path "${MODEL_PATH}" \
        --tensor_parallel_size ${TP} \
        --max_model_len ${MAX_MODEL_LEN} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --video_max_frames ${VIDEO_MAX_FRAMES} \
        --video_fps ${VIDEO_FPS} \
        --gpu_memory_utilization ${GPU_UTIL}

CUDA_VISIBLE_DEVICES=0,1 python qwen3vl_harness.py evaluate \
        --input_jsonl  "input/fused_results_sample3.json" \
        --output_jsonl "output/fused_results_sample3_harness_all_metrics.json" \
        --model_path "${MODEL_PATH}" \
        --tensor_parallel_size ${TP} \
        --max_model_len ${MAX_MODEL_LEN} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --video_max_frames ${VIDEO_MAX_FRAMES} \
        --video_fps ${VIDEO_FPS} \
        --gpu_memory_utilization ${GPU_UTIL}

CUDA_VISIBLE_DEVICES=0,1 python qwen3vl_harness.py evaluate \
        --input_jsonl  "input/fused_results_sample4.json" \
        --output_jsonl "output/fused_results_sample4_harness_all_metrics.json" \
        --model_path "${MODEL_PATH}" \
        --tensor_parallel_size ${TP} \
        --max_model_len ${MAX_MODEL_LEN} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --video_max_frames ${VIDEO_MAX_FRAMES} \
        --video_fps ${VIDEO_FPS} \
        --gpu_memory_utilization ${GPU_UTIL}


echo "全部评估完成，开始 select-best ..."
python qwen3vl_harness.py select-best \
    --inputs "${OUT_DIR}/fused_results_sample1_harness_all_metrics.json" \
             "${OUT_DIR}/fused_results_sample2_harness_all_metrics.json" \
             "${OUT_DIR}/fused_results_sample3_harness_all_metrics.json" \
             "${OUT_DIR}/fused_results_sample4_harness_all_metrics.json" \
    --output "${OUT_DIR}/best_by_score.json" \
    --rank-by total
